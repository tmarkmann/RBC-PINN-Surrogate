from typing import Tuple

import torch.nn as nn
from torch import Tensor


class Autoencoder(nn.Module):
    def __init__(
        self,
        latent_dimension: int,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        activation: nn.Module,
    ):
        """Initialize an `Autoencoder`.

        Args:
            latent_dimension (int): The latent dimension of the autoencoder.
            input_channel (int): The number of channels of the input.
            base_filters (int): The number of filters used in the first layer of the encoder.
            activation (torch.nn.Module): The activation function usde in encoder and decoder.
        """
        super().__init__()
        self.latent_dimension = latent_dimension
        self.input_channel = input_channel
        self.base_filters = base_filters
        self.kernel_size = kernel_size

        # Build models
        self.encoder = _Encoder(input_channel, base_filters, kernel_size, activation)
        self.decoder = _Decoder(input_channel, base_filters, kernel_size, activation)

        # linear layers
        self.encoder_linear = nn.Linear(self.encoder.out_dim, latent_dimension)
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dimension, self.encoder.out_dim),
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
        self.encoder.load_state_dict(encoder_weights)
        self.decoder.load_state_dict(decoder_weights)

        # Freeze
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform a forward pass through the model consisting of encoding and decoding the input.

        Args:
            x (Tensor): A tensor x of size (b,c,h,w).

        Returns:
            - x_hat (Tensor): Reconstructed image of size (b,c,h,w).
            - z (Tensor): Latent variable z if size (b, l).
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def encode(self, x: Tensor) -> Tensor:
        """Encodes the input image to latent space.

        Args:
            x (Tensor): A tensor x of size (b,c,h,w).

        Returns:
            z (Tensor): Latent variable z if size (b, l).
        """
        return self.encoder_linear(self.encoder(x))

    def decode(self, z: Tensor) -> Tensor:
        """Decodes latent variable to image.

        Args:
            z (Tensor): Latent variable z if size (b, l).

        Returns:
            x_hat (Tensor): A tensor x of size (b,c,h,w)
        """
        return self.decoder(self.decoder_linear(z))


class _Encoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        # Parameters
        hid = base_filters
        k = kernel_size
        self.out_dim = 2 * 3 * 4 * hid

        self.net = nn.Sequential(
            nn.Conv2d(input_channel, hid, kernel_size=k, padding=2, stride=2),
            activation(),
            nn.Conv2d(hid, 2 * hid, kernel_size=k, padding=2, stride=2),
            activation(),
            nn.Conv2d(2 * hid, 2 * hid, kernel_size=k, padding=2, stride=2),
            activation(),
            nn.Conv2d(2 * hid, 4 * hid, kernel_size=k, padding=2, stride=2),
            activation(),
            nn.Conv2d(4 * hid, 4 * hid, kernel_size=k, padding=2, stride=2),
            activation(),
            nn.Flatten(),
        )

    def forward(self, x) -> Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        # Parameters
        hid = base_filters
        k = kernel_size

        # Model
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                4 * hid, 4 * hid, k, padding=2, stride=2, output_padding=1
            ),
            activation(),
            nn.ConvTranspose2d(
                4 * hid, 2 * hid, k, padding=2, stride=2, output_padding=1
            ),
            activation(),
            nn.ConvTranspose2d(
                2 * hid, 2 * hid, k, padding=2, stride=2, output_padding=1
            ),
            activation(),
            nn.ConvTranspose2d(2 * hid, hid, k, padding=2, stride=2, output_padding=1),
            activation(),
            nn.ConvTranspose2d(
                hid, input_channel, k, padding=2, stride=2, output_padding=1
            ),
        )

    def forward(self, x) -> Tensor:
        x = x.reshape(x.shape[0], -1, 2, 3)
        return self.net(x)
