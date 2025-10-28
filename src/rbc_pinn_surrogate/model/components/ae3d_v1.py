from collections import OrderedDict
from typing import Tuple, List

import torch.nn as nn
from torch import Tensor


class Autoencoder3D(nn.Module):
    def __init__(
        self,
        input_size: int,
        channels: List[int],
        pooling: List[bool],
        latent_channels: int,
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

        # Latent layers
        self.encoder_latent = nn.Sequential(
            nn.Conv3d(
                channels[-1],
                latent_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            self.activation(),
        )
        self.decoder_latent = nn.Sequential(
            nn.Conv3d(
                latent_channels,
                channels[-1],
                kernel_size=self.kernel_size,
                padding=self.padding,
            ),
            self.activation(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.decode(self.encode(x))

    def encode(self, x: Tensor) -> Tensor:
        # Pass through encoder convs and flatten
        f = self.encoder(x)
        return self.encoder_latent(f)

    def decode(self, z: Tensor) -> Tensor:
        f = self.decoder_latent(z)
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
            layer[f"conv3d_{i}"] = nn.Conv3d(
                inp,
                ch,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=s,
            )
            layer[f"activation_{i}"] = self.activation()
            if self.drop_rate > 0:
                layer[f"dropout_{i}"] = nn.Dropout3d(p=self.drop_rate)
            if self.batch_norm:
                layer[f"batch_norm_{i}"] = nn.BatchNorm3d(ch)
            inp = ch

        return nn.Sequential(layer)

    def build_decoder(
        self,
        channels: List[int],
        pooling: List[int],
    ) -> nn.Module:
        layer = OrderedDict()
        inp = channels[-1]
        length = len(channels)
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
            layer[f"convtrans3d_{i}"] = nn.ConvTranspose3d(
                inp,
                ch,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=s,
                output_padding=op,
            )

            # no addtional layers after last conv
            if i == length - 1:
                break

            layer[f"activation_{i}"] = self.activation()
            if self.drop_rate > 0:
                layer[f"dropout_{i}"] = nn.Dropout3d(p=self.drop_rate)
            if self.batch_norm:
                layer[f"batch_norm_{i}"] = nn.BatchNorm3d(ch)
            inp = ch

        return nn.Sequential(layer)
