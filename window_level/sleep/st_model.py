import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Encoder(nn.Module):
    """
    Encoder stack
    """
    def __init__(self, input_channels, filters, kernel_sizes, in_samples):
        super().__init__()

        convs = []
        pools = []
        elus = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )

            # To be consistent with the behaviour in tensorflow,
            # padding needs to be added for odd numbers of input_samples
            padding = in_samples % 2

            # Padding for MaxPool1d needs to be handled manually to conform with tf padding
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            elus.append(nn.ELU(inplace=True))
            in_samples = (in_samples + padding) // 2

        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)
        self.elus = nn.ModuleList(elus)
    def forward(self, x):
        skips = []
        for conv, pool, padding, elu in zip(self.convs, self.pools, self.paddings, self.elus):
            x = elu(conv(x))
            skips.append(x)
            if padding != 0:
                # Only pad right, use -1e10 as negative infinity
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        kernel_sizes,
        out_samples,
        original_compatible=False,
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.original_compatible = original_compatible

        # We need to trim off the final sample sometimes to get to the right number of output samples
        self.crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding) // 2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        convs = []
        elus = []
        for in_channels, out_channels, kernel_size in zip(
            [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )
            elus.append(nn.ELU(inplace=True))

        self.convs = nn.ModuleList(convs)
        self.elus = nn.ModuleList(elus)
    def forward(self, x, skip_connections):
        for i, (conv, elu) in enumerate(zip(self.convs, self.elus)):
            x = self.upsample(x)
            if self.original_compatible:
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crops:
                    x = x[:, :, :-1]
            x = elu(conv(x))
            if skip_connections is not None and i < len(skip_connections):
                # Use reverse order: first decoder block gets skip from the last encoder block.
                skip = skip_connections[-(i+1)]
                x = x + skip
        return x


class ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes, filters, drop_rate):
        super().__init__()

        members = []
        for ker in kernel_sizes:
            members.append(ResCNNBlock(filters, ker, drop_rate))
        self.members = nn.ModuleList(members)
    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x


class ResCNNBlock(nn.Module):
    def __init__(self, filters, ker, drop_rate):
        super().__init__()

        self.manual_padding = False
        if ker == 3:
            padding = 1
        else:
            # ker == 2
            # Manual padding emulate the padding in tensorflow
            self.manual_padding = True
            padding = 0

        self.dropout = SpatialDropout1d(drop_rate)

        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)

        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)
    def forward(self, x):
        y = self.norm1(x)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv1(y)

        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv2(y)

        return x + y


class SpatialDropout1d(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)
    def forward(self, x):
        x = x.unsqueeze(dim=-1)  # Add fake dimension
        x = self.dropout(x)
        x = x.squeeze(dim=-1)  # Remove fake dimension
        return x


class SeizureTransformer_window(nn.Module):

    def __init__(
        self,
        in_channels=19,
        in_samples=15360,
        num_classes=1,
        dim_feedforward=2048,
        num_layers=8,
        num_heads=4,
        drop_rate=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.in_samples = in_samples
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        # Parameters from EQTransformer repository
        self.filters = [
            16,
            32,
            64,
            128
        ]  # Number of filters for the convolutions
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]  # Kernel sizes for the convolutions
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]


        # Encoder stack
        self.encoder = Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples,
        )

        # Res CNN Stack
        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=self.drop_rate,
        )

        self.position_encoding = PositionalEncoding(d_model=128)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers,
        )

        # Detection decoder and final Conv
        self.decoder_d = Decoder(
            input_channels=128,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=in_samples,
            original_compatible=False,
        )

        self.pooling = AttentionPooling(hidden_dim=16)
        self.classifier = nn.Linear(16, num_classes)
        if num_classes == 1:
            self.softmax = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x, logits=True):
        assert x.ndim == 3
        assert x.shape[1:] == (self.in_channels, self.in_samples)

        x, skips = self.encoder(x)
        # print('x', x.shape)
        res_x = self.res_cnn_stack(x)
        x = res_x.permute(2, 0, 1)
        x = self.position_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)
        x = x + res_x

        detection = self.decoder_d(x, skips)
        # print('deteciton', detection.shape)
        
        detection = self.pooling(detection)
        detection = self.classifier(detection)
        detection = self.softmax(detection)
        if self.num_classes == 1:
            return detection.squeeze(dim=-1)
        else:
            return detection


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)  # Learn score for each time step

    def forward(self, x):
        # x: (B, H, T)
        x_perm = x.permute(0, 2, 1)               # (B, T, H)
        score = self.attn(x_perm).squeeze(-1)     # (B, T)
        weights = torch.softmax(score, dim=1)     # (B, T)
        pooled = torch.bmm(x, weights.unsqueeze(-1)).squeeze(-1)  # (B, H)
        return pooled