from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Autoencoder3D(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        activation: nn.Module,
        input_shape: Tuple[int, int, int],  # (D, H, W)
    ):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.input_channel = input_channel
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape  # (D, H, W)

        # Build models
        self.encoder = _Encoder(
            input_channel=input_channel,
            base_filters=base_filters,
            kernel_size=kernel_size,
            activation=activation,
        )
        # Infer encoder output feature dim and spatial shape dynamically
        with torch.no_grad():
            dummy = torch.zeros(
                1, input_channel, input_shape[0], input_shape[1], input_shape[2]
            )
            enc_feat, enc_shape = self.encoder.feature_map(dummy)
        self.encoder_out_channels = enc_feat.shape[1]
        self.encoder_out_spatial = enc_shape  # (D_e, H_e, W_e)
        self.encoder_out_dim = int(
            self.encoder_out_channels * enc_shape[0] * enc_shape[1] * enc_shape[2]
        )

        self.decoder = _Decoder(
            output_channel=input_channel,
            base_filters=base_filters,
            kernel_size=kernel_size,
            activation=activation,
            start_channels=self.encoder_out_channels,
            start_shape=self.encoder_out_spatial,
        )
        print(f"Autoencoder3D encoder output dim: {self.encoder_out_dim}")
        # Linear layers
        self.encoder_linear = nn.Linear(self.encoder_out_dim, latent_dimension)
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dimension, self.encoder_out_dim),
            activation(),
        )

    def load_weights(self, ckpt: dict, freeze: bool) -> None:
        # Check hyperparameters
        params = ckpt["hyper_parameters"]
        assert self.input_channel == params["input_channel"], (
            f"'input_channel' does not match. ({self.input_channel} != {params['input_channel']})"
        )
        assert self.base_filters == params["base_filters"], (
            f"'base_filters' does not match. ({self.base_filters} != {params['base_filters']})"
        )
        assert self.kernel_size == params["kernel_size"], (
            f"'kernel_size' does not match. ({self.kernel_size} != {params['kernel_size']})"
        )
        # Note: input_shape might differ; we infer dynamically at runtime.

        # Load weights
        state = ckpt["state_dict"]
        encoder_weights = {
            k.replace("autoencoder.encoder.", ""): v
            for k, v in state.items()
            if k.startswith("autoencoder.encoder.")
        }
        decoder_weights = {
            k.replace("autoencoder.decoder.", ""): v
            for k, v in state.items()
            if k.startswith("autoencoder.decoder.")
        }
        self.encoder.load_state_dict(encoder_weights, strict=False)
        self.decoder.load_state_dict(decoder_weights, strict=False)

        # Freeze
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def encode(self, x: Tensor) -> Tensor:
        # Pass through encoder convs and flatten
        feats, _ = self.encoder.feature_map(x)
        feats = feats.reshape(feats.shape[0], -1)
        return self.encoder_linear(feats)

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder_linear(z)
        # reshape to starting feature map for decoder
        B = h.shape[0]
        h = h.view(B, self.encoder_out_channels, *self.encoder_out_spatial)
        return self.decoder(h)


class _Encoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        hid = base_filters
        k = kernel_size
        s = (2, 2, 2)
        p = 2

        # Five downsampling stages (like original 2D code) but in 3D
        self.net = nn.Sequential(
            nn.Conv3d(input_channel, hid, kernel_size=k, padding=p, stride=s),
            activation(),
            nn.Conv3d(hid, 2 * hid, kernel_size=k, padding=p, stride=s),
            activation(),
            nn.Conv3d(2 * hid, 2 * hid, kernel_size=k, padding=p, stride=s),
            activation(),
            nn.Conv3d(2 * hid, 4 * hid, kernel_size=k, padding=p, stride=s),
            activation(),
        )

    def feature_map(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
        """Return last feature map and its spatial shape (D, H, W)."""
        y = self.net(x)
        D, H, W = y.shape[-3], y.shape[-2], y.shape[-1]
        return y, (D, H, W)


class _Decoder(nn.Module):
    def __init__(
        self,
        output_channel: int,
        base_filters: int,
        kernel_size: int,
        activation: nn.Module,
        start_channels: int,
        start_shape: Tuple[int, int, int],  # (D_e, H_e, W_e)
    ) -> None:
        super().__init__()
        hid = base_filters
        k = kernel_size
        s = (2, 2, 2)
        p = 2
        op = 1

        # The first ConvTranspose3d expects `start_channels`
        self.net = nn.Sequential(
            nn.ConvTranspose3d(
                start_channels, 4 * hid, k, padding=p, stride=s, output_padding=op
            ),
            activation(),
            nn.ConvTranspose3d(
                4 * hid, 2 * hid, k, padding=p, stride=s, output_padding=op
            ),
            activation(),
            nn.ConvTranspose3d(2 * hid, hid, k, padding=p, stride=s, output_padding=op),
            activation(),
            nn.ConvTranspose3d(
                hid, output_channel, k, padding=p, stride=s, output_padding=op
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
