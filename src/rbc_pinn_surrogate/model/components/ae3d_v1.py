from collections import OrderedDict
from typing import Tuple, List

import torch
import torch.nn as nn
from torch import Tensor


class Autoencoder3D(nn.Module):
    def __init__(
        self,
        input_size: int,
        channels: List[int],
        pooling: List[bool],
        latent_dimension: int,
        kernel_size: int,
        drop_rate: float,
        batch_norm: bool,
    ):
        super().__init__()
        self.input_channel = input_size[0]
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        self.activation = nn.GELU
        self.padding = 2

        # Build models
        self.encoder = self.build_encoder(channels, pooling)
        self.decoder = self.build_decoder(channels, pooling)

        # Compute encoder output shape
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            enc_out = self.encoder(dummy)
            spatial_dim = enc_out.shape[1:]
            flatten_dim = enc_out.numel()

        # Latent layers
        self.encoder_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, latent_dimension),
            self.activation(),
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dimension, flatten_dim),
            self.activation(),
            nn.Unflatten(1, spatial_dim),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.decode(self.encode(x))

    def encode(self, x: Tensor) -> Tensor:
        # Pass through encoder convs and flatten
        f = self.encoder(x)
        return self.encoder_linear(f)

    def decode(self, z: Tensor) -> Tensor:
        f = self.decoder_linear(z)
        return self.decoder(f)

    def build_encoder(
        self,
        channels: List[int],
        pooling: List[int],
    ) -> nn.Module:
        layer = OrderedDict()
        inp = self.input_channel
        for i, (ch, pool) in enumerate(zip(channels, pooling)):
            # Downsampling learned by strided conv
            if pool:
                s = (2, 2, 2)
            else:
                s = (1, 1, 1)

            # Build layers
            layer[f"conv3d{i}"] = nn.Conv3d(
                inp,
                ch,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=s,
            )
            layer[f"activation{i}"] = self.activation()
            if self.drop_rate > 0:
                layer[f"dropout{i}"] = nn.Dropout3d(p=self.drop_rate)
            if self.batch_norm:
                layer[f"batch_norm{i}"] = nn.BatchNorm3d(ch)
            inp = ch

        return nn.Sequential(layer)

    def build_decoder(
        self,
        channels: List[int],
        pooling: List[int],
    ) -> nn.Module:
        layer = OrderedDict()
        inp = channels[-1]
        channels = reversed([self.input_channel] + channels[:-1])
        pooling = reversed(pooling)
        for i, (ch, pool) in enumerate(zip(channels, pooling)):
            # Upsampling learned by strided conv transpose
            if pool:
                s = (2, 2, 2)
                op = 1
            else:
                s = (1, 1, 1)
                op = 0

            # Build layers
            layer[f"convtrans3d{i}"] = nn.ConvTranspose3d(
                inp,
                ch,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=s,
                output_padding=op,
            )
            layer[f"activation{i}"] = self.activation()
            if self.drop_rate > 0:
                layer[f"dropout{i}"] = nn.Dropout3d(p=self.drop_rate)
            if self.batch_norm:
                layer[f"batch_norm{i}"] = nn.BatchNorm3d(ch)
            inp = ch

        return nn.Sequential(layer)
