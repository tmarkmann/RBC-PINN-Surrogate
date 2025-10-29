from collections import OrderedDict
from typing import List, Sequence, Type, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _triple


class Conv3DBlock(nn.Module):
    def __init__(
        self,
        index: int | str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = (1, 1, 1),
        drop_rate: float = 0,
        batch_norm: bool = False,
        activation: Type[nn.Module] = None,
    ):
        super().__init__()

        layers = OrderedDict()
        kH, kW, kD = _triple(kernel_size)
        pH, pW, pD = (kH // 2, kW // 2, kD // 2)

        # vertical zero padding for vertical direction
        layers[f"pad_{index}"] = nn.ConstantPad3d((0, 0, 0, 0, pH, pH), 0)

        # convolution with periodic horizontal padding
        layers[f"conv3d_{index}"] = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(kH, kW, kD),
            stride=stride,
            padding=(0, pW, pD),
            padding_mode="circular",
        )

        # activation function
        if activation is not None:
            layers[f"activation_{index}"] = activation()

        # drop layer
        if drop_rate > 0:
            layers[f"dropout_{index}"] = nn.Dropout3d(p=drop_rate)

        # batch norm layer
        if batch_norm:
            layers[f"batch_norm_{index}"] = nn.BatchNorm3d(out_channels)

        self.block = nn.Sequential(layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Autoencoder3D(nn.Module):
    def __init__(
        self,
        input_size: Sequence[int],
        channels: List[int],
        pooling: List[bool],
        latent_channels: int,
        kernel_size: Union[int, Sequence[int]],
        latent_kernel_size: Union[int, Sequence[int]],
        drop_rate: float,
        batch_norm: bool,
        activation: Type[nn.Module],
    ):
        super().__init__()
        self.input_channel = input_size[0]
        self.latent_channels = latent_channels
        self.kernel_size = kernel_size
        self.latent_kernel_size = latent_kernel_size
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        self.activation = activation

        # Build models
        self.encoder = self.build_encoder(channels, pooling)
        self.decoder = self.build_decoder(channels, pooling)

        # Latent layers
        self.encoder_latent = Conv3DBlock(
            index="latent",
            in_channels=channels[-1],
            out_channels=latent_channels,
            kernel_size=self.latent_kernel_size,
            batch_norm=batch_norm,
            drop_rate=drop_rate,
            activation=activation,
        )

        self.decoder_out = Conv3DBlock(
            index="out",
            in_channels=channels[0],
            out_channels=input_size[0],
            kernel_size=self.latent_kernel_size,
            batch_norm=batch_norm,
            drop_rate=drop_rate,
            activation=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def encode(self, x: Tensor) -> Tensor:
        # Pass through encoder convs and flatten
        f = self.encoder(x)
        return self.encoder_latent(f)

    def decode(self, z: Tensor) -> Tensor:
        f = self.decoder(z)
        return self.decoder_out(f)

    def build_encoder(
        self,
        channels: List[int],
        pooling: List[bool],
    ) -> nn.Module:
        layer: "OrderedDict[str, nn.Module]" = OrderedDict()
        inp = self.input_channel
        for i, (ch, pool) in enumerate(zip(channels, pooling)):
            # Downsampling learned by strided conv
            if pool:
                stride = (2, 2, 2)
            else:
                stride = (1, 1, 1)

            # build conv block
            layer[f"block_{i}"] = Conv3DBlock(
                index=i,
                in_channels=inp,
                out_channels=ch,
                kernel_size=self.kernel_size,
                stride=stride,
                drop_rate=self.drop_rate,
                batch_norm=self.batch_norm,
                activation=self.activation,
            )

            inp = ch

        return nn.Sequential(layer)

    def build_decoder(
        self,
        channels: List[int],
        pooling: List[bool],
    ) -> nn.Module:
        layer: "OrderedDict[str, nn.Module]" = OrderedDict()
        inp = self.latent_channels
        channels = reversed(channels)
        upsampling = reversed(pooling)
        for i, (ch, up) in enumerate(zip(channels, upsampling)):
<<<<<<< Updated upstream
            # conv block
            layer[f"block_{i}"] = Conv3DBlock(
=======
            # Upsampling
            if up:
                layer[f"upsample_{i}"] = nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear")

            # Build layers
            layer[f"convtrans3d_{i}"] = Conv3DBlock(
>>>>>>> Stashed changes
                index=i,
                in_channels=inp,
                out_channels=ch,
                kernel_size=self.kernel_size,
                stride=(1, 1, 1),
                drop_rate=self.drop_rate,
                batch_norm=self.batch_norm,
                activation=self.activation,
            )

            # Upsampling
            if up:
                layer[f"upsample_{i}"] = nn.Upsample(
                    scale_factor=(2, 2, 2), mode="trilinear", align_corners=False
                )

            inp = ch

        return nn.Sequential(layer)

    @classmethod
    def from_checkpoint(cls, ckpt: dict):
        # Initialize model from loaded params
        params = ckpt["hyper_parameters"]
        print("Loading Autoencoder3D with params:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        model = cls(
            input_size=params["input_size"],
            channels=params["channels"],
            pooling=params["pooling"],
            latent_channels=params["latent_channels"],
            kernel_size=params["kernel_size"],
            latent_kernel_size=params["latent_kernel_size"],
            drop_rate=params["drop_rate"],
            batch_norm=params["batch_norm"],
            activation=nn.GELU,
        )

        # Load weights
        state = ckpt["state_dict"]
        weights = {
            k.replace("autoencoder.", ""): v
            for k, v in state.items()
            if k.startswith("autoencoder.")
        }
        model.load_state_dict(weights)

        return model
