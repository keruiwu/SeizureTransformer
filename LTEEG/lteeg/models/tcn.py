"""Lightweight dilated temporal convolutional network.

A strided convolutional stem reduces the sampling rate by ``stem_stride``, a stack of
residual dilated convolutions (dilation 1, 2, 4, ...) builds a long receptive field,
and a 1x1 head emits logits at the reduced rate (``output_stride = stem_stride``);
the framework interpolates them to one value per sample. Small and fast: useful as
a second baseline, for smoke tests, and as a template for new models.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..registry import MODELS


class _ResidualDilated(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@MODELS.register("tcn")
class DilatedTCN(nn.Module):
    def __init__(self, in_channels: int, in_samples: int, num_outputs: int = 1, hidden: int = 64,
                 levels: int = 8, kernel_size: int = 3, stem_stride: int = 8, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.output_stride = stem_stride
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=2 * stem_stride + 1, stride=stem_stride, padding=stem_stride),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[_ResidualDilated(hidden, kernel_size, 2 ** i, dropout) for i in range(levels)])
        self.head = nn.Conv1d(hidden, num_outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))
