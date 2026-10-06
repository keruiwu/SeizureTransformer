"""SeizureTransformer (Wu, Zhao & Yener, 2025; arXiv:2504.00336).

U-shaped sequence-to-sequence network, winner of the 2025 SzCORE seizure detection
challenge: a 1-D convolutional encoder (5 levels, max-pool /2), a residual CNN
stack and a Transformer encoder at the bottleneck (T/32 tokens), and a decoder with
nearest upsampling and additive skip connections that emits one score per input
sample.

Port of the original ``time_step_level/model.py`` with these deliberate changes
(none alters the function computed for the original configuration):

* returns **logits** ``(B, num_outputs, T)`` instead of ``sigmoid`` probabilities, so
  the framework can use numerically stable ``BCEWithLogits`` (also required under
  mixed precision) and multi-class heads;
* the unused ``self.transformer_encoder_layer`` attribute is removed
  (``nn.TransformerEncoder`` deep-copies the layer it is given, so the original
  carried ~3.2M dead parameters that the optimizer still had to track);
* odd lengths are handled with ``MaxPool1d(ceil_mode=True)`` (identical to the
  original right-padding with -1e10) and the decoder crops to each skip's length,
  so the network accepts any input length >= 32, not only ``in_samples``;
* the positional-encoding table is sized from the input length (the original fixed
  6000 tokens, i.e. at most 192 000 samples) and is a non-persistent buffer;
* the Transformer runs ``batch_first`` (same parameters, faster kernels).

Parameter names are unchanged, so weights trained with the original code load via
:func:`load_original_state_dict`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import MODELS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, D)
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence of {x.size(1)} tokens exceeds positional table ({self.pe.size(1)})")
        return self.dropout(x + self.pe[:, : x.size(1)].to(x.dtype))


class Encoder(nn.Module):
    def __init__(self, input_channels: int, filters: Sequence[int], kernel_sizes: Sequence[int]):
        super().__init__()
        convs = []
        for cin, cout, k in zip([input_channels] + list(filters[:-1]), filters, kernel_sizes):
            convs.append(nn.Conv1d(cin, cout, k, padding=k // 2))
        self.convs = nn.ModuleList(convs)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for conv in self.convs:
            x = F.elu(conv(x))
            skips.append(x)
            x = self.pool(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, input_channels: int, filters: Sequence[int], kernel_sizes: Sequence[int]):
        super().__init__()
        convs = []
        for cin, cout, k in zip([input_channels] + list(filters[:-1]), filters, kernel_sizes):
            convs.append(nn.Conv1d(cin, cout, k, padding=k // 2))
        self.convs = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            skip = skips[-(i + 1)]
            x = F.interpolate(x, scale_factor=2, mode="nearest")[..., : skip.shape[-1]]
            x = F.elu(conv(x)) + skip
        return x


class SpatialDropout1d(nn.Module):
    def __init__(self, drop_rate: float):
        super().__init__()
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x.unsqueeze(-1)).squeeze(-1)


class ResCNNBlock(nn.Module):
    def __init__(self, filters: int, ker: int, drop_rate: float):
        super().__init__()
        # kernel 2 is padded on the right only (TensorFlow 'same'), kernel 3 symmetric
        self.manual_padding = ker == 2
        padding = 0 if self.manual_padding else ker // 2
        self.dropout = SpatialDropout1d(drop_rate)
        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)
        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.relu(self.norm1(x)))
        if self.manual_padding:
            y = F.pad(y, (0, 1))
        y = self.conv1(y)
        y = self.dropout(F.relu(self.norm2(y)))
        if self.manual_padding:
            y = F.pad(y, (0, 1))
        y = self.conv2(y)
        return x + y


class ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes: Sequence[int], filters: int, drop_rate: float):
        super().__init__()
        self.members = nn.ModuleList([ResCNNBlock(filters, k, drop_rate) for k in kernel_sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.members:
            x = m(x)
        return x


@MODELS.register("seizure_transformer")
class SeizureTransformer(nn.Module):
    output_stride = 1

    def __init__(
        self,
        in_channels: int = 18,
        in_samples: int = 15360,
        num_outputs: int = 1,
        dim_feedforward: int = 2048,
        num_layers: int = 8,
        num_heads: int = 4,
        drop_rate: float = 0.1,
        transformer_dropout: float = 0.1,
        filters: Sequence[int] = (32, 64, 128, 256, 512),
        kernel_sizes: Sequence[int] = (11, 9, 7, 7, 5, 5, 3),
        res_cnn_kernels: Sequence[int] = (3, 3, 3, 3, 2, 3, 2),
        norm_first: bool = False,
    ):
        super().__init__()
        filters, kernel_sizes = list(filters), list(kernel_sizes)
        if len(kernel_sizes) < len(filters):
            raise ValueError("need at least one kernel size per encoder level")
        self.in_channels, self.in_samples, self.num_outputs = in_channels, in_samples, num_outputs
        self.min_samples = 2 ** len(filters)
        d_model = filters[-1]
        self.encoder = Encoder(in_channels, filters, kernel_sizes)
        self.res_cnn_stack = ResCNNStack(res_cnn_kernels, d_model, drop_rate)
        max_tokens = max(6000, math.ceil(in_samples / 2 ** len(filters)) * 4)
        self.position_encoding = PositionalEncoding(d_model, dropout=transformer_dropout, max_len=max_tokens)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
                                           dropout=transformer_dropout, batch_first=True, norm_first=norm_first)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                                         enable_nested_tensor=False)
        # same zip-truncation as the original: decoder kernels are the reversed list's head
        self.decoder_d = Decoder(d_model, filters[::-1], kernel_sizes[::-1])
        self.conv_d = nn.Conv1d(filters[0], num_outputs, kernel_size=11, padding=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, T), got {tuple(x.shape)}")
        if x.shape[-1] < self.min_samples:
            raise ValueError(f"input length {x.shape[-1]} < minimum {self.min_samples}")
        x, skips = self.encoder(x)
        res_x = self.res_cnn_stack(x)
        h = self.position_encoding(res_x.transpose(1, 2))
        h = self.transformer_encoder(h).transpose(1, 2)
        x = h + res_x
        x = self.decoder_d(x, skips)
        return self.conv_d(x)


def load_original_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert a state dict saved by the original repository (``torch.save(model.state_dict())``)
    for :class:`SeizureTransformer`: drops the dead ``transformer_encoder_layer.*`` weights,
    the positional table and any ``module.`` prefix from DataParallel.

    Note the original model applied ``sigmoid`` inside ``forward``; the converted model
    returns logits, so apply ``sigmoid`` (the framework does) to get identical outputs.
    """
    out = {}
    for k, v in state.items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k.startswith("transformer_encoder_layer.") or k == "position_encoding.pe":
            continue
        out[k] = v
    return out
