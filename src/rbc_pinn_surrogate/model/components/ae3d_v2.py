from torch import nn
from torch import Tensor

from collections import OrderedDict
from typing import Callable

from rbc_pinn_surrogate.model.layers import RB3DConv, RBPooling, RBUpsampling


class _Conv3DBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_dims: tuple,
        v_kernel_size: int,
        h_kernel_size: int,
        input_drop_rate: float,
        bias: bool = True,
        nonlinearity: Callable = nn.ELU,
        batch_norm: bool = True,
    ):
        """A convolution block (with vertical parameter sharing) with dropout, batch normalization
        and nonlinearity.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels
            in_dims (tuple): The spatial dimensions of the input data.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            input_drop_rate (float): The drop rate for dropout applied to the input of the conv block. Set to 0
                to turn off dropout.
            bias (bool, optional): Whether to apply a bias to the output of the convolution.
                Bias is turned off automatically when using batch normalization as a bias has no effect when
                using batch normalization. Defaults to True.
            nonlinearity (Callable, optional): The nonlinearity applied to the conv output. Set to `None` to
                have no nonlinearity. Defaults to nn.ELU.
            batch_norm (bool, optional): Whether to use batch normalization. Defaults to True.
        """

        conv = RB3DConv(
            in_channels=in_channels,
            out_channels=out_channels,
            in_dims=in_dims,
            v_kernel_size=v_kernel_size,
            h_kernel_size=h_kernel_size,
            bias=bias and not batch_norm,  # bias has no effect when using batch norm
            v_stride=1,
            h_stride=1,
            v_pad_mode="zeros",
            h_pad_mode="circular",
        )

        layers = []
        if input_drop_rate > 0:
            layers.append(nn.Dropout(p=input_drop_rate))
        layers.append(conv)
        if batch_norm:
            layers.append(nn.BatchNorm3d(conv.out_channels))
        if nonlinearity:
            layers.append(nonlinearity())

        super().__init__(*layers)

        self.in_dims, self.out_dims = in_dims, conv.out_dims
        self.in_channels, self.out_channels = in_channels, out_channels


class Autoencoder3Dv2(nn.Sequential):
    def __init__(
        self,
        rb_dims: tuple,
        encoder_channels: tuple,
        latent_channels: int,
        v_kernel_size: int,
        h_kernel_size: int,
        latent_v_kernel_size: int,
        latent_h_kernel_size: int,
        drop_rate: float,
        pool_layers: tuple[bool] = None,
        nonlinearity: Callable = nn.ELU,
        **kwargs,
    ):
        """A Rayleigh-Bénard autoencoder based on standard 3D convolutions.

        Args:
            rb_dims (tuple): The spatial dimensions of the simulation data.
            encoder_channels (tuple): The channels of the encoder. Each entry results in a corresponding layer.
                The decoder uses the channels in reversed order.
            latent_channels (int): The number of channels in the latent space.
            v_kernel_size (int): The vertical kernel size.
            h_kernel_size (int): The horizontal kernel size (in both directions).
            latent_v_kernel_size (int): The vertical kernel size applied on the latent space.
            latent_h_kernel_size (int): The horizontal kernel size (in both directions) applied on the latent space.
            drop_rate (float): The drop rate used for dropout. Set to 0 to turn off dropout.
            pool_layers (tuple[bool], optional): A boolean tuple specifying the encoder layer to pool afterwards.
                The same is used in reversed order for upsampling in the decoder. Defaults to pooling/upsampling
                after each layer.
            nonlinearity (Callable, optional): The nonlinearity applied to the conv output. Set to `None` to
                have no nonlinearity. Defaults to enn.ELU.
        """
        super().__init__()
        if pool_layers is None:
            pool_layers = [True] * len(encoder_channels)

        self.encoder_layers = OrderedDict()
        self.decoder_layers = OrderedDict()

        #####################
        ####   Encoder   ####
        #####################
        channels, dims = 4, rb_dims
        for i, (out_channels, pool) in enumerate(zip(encoder_channels, pool_layers), 1):
            conv = _Conv3DBlock(
                in_channels=channels,
                out_channels=out_channels,
                in_dims=dims,
                v_kernel_size=v_kernel_size,
                h_kernel_size=h_kernel_size,
                input_drop_rate=0 if i == 1 else drop_rate,
                nonlinearity=nonlinearity,
                batch_norm=True,
            )
            self.encoder_layers[f"EncoderConv{i}"] = conv
            channels = conv.out_channels

            if pool:
                pool = RBPooling(
                    in_channels=channels, in_dims=dims, v_kernel_size=2, h_kernel_size=2
                )
                self.encoder_layers[f"Pooling{i}"] = pool
                dims = pool.out_dims

        ######################
        #### Latent Space ####
        ######################
        conv = _Conv3DBlock(
            in_channels=channels,
            out_channels=latent_channels,
            in_dims=dims,
            v_kernel_size=latent_v_kernel_size,
            h_kernel_size=latent_h_kernel_size,
            input_drop_rate=drop_rate,
            nonlinearity=nonlinearity,
            batch_norm=True,
        )
        self.encoder_layers["LatentConv"] = conv
        channels = conv.out_channels

        self.latent_shape = [*dims, latent_channels]

        #####################
        ####   Decoder   ####
        #####################
        decoder_channels = reversed(encoder_channels)
        upsample_layers = reversed(pool_layers)
        for i, (out_channels, upsample) in enumerate(
            zip(decoder_channels, upsample_layers), 1
        ):
            conv = _Conv3DBlock(
                in_channels=channels,
                out_channels=out_channels,
                in_dims=dims,
                v_kernel_size=v_kernel_size,
                h_kernel_size=h_kernel_size,
                input_drop_rate=drop_rate,
                nonlinearity=nonlinearity,
                batch_norm=True,
            )
            self.decoder_layers[f"DecoderConv{i}"] = conv
            channels = conv.out_channels

            if upsample:
                upsample = RBUpsampling(
                    in_channels=channels, in_dims=dims, v_scale=2, h_scale=2
                )
                self.decoder_layers[f"Upsampling{i}"] = upsample
                dims = upsample.out_dims

        ######################
        ####    Output    ####
        ######################
        conv = _Conv3DBlock(
            in_channels=channels,
            out_channels=4,
            in_dims=dims,
            v_kernel_size=v_kernel_size,
            h_kernel_size=h_kernel_size,
            input_drop_rate=drop_rate,
            nonlinearity=None,
            batch_norm=False,
        )
        self.decoder_layers["OutputConv"] = conv

        first_layer = self.encoder_layers[next(iter(self.encoder_layers))]
        last_layer = self.decoder_layers[next(reversed(self.decoder_layers))]
        self.in_dims, self.out_dims = (
            tuple(first_layer.in_dims),
            tuple(last_layer.out_dims),
        )

        assert self.out_dims == self.in_dims == tuple(rb_dims)

        self.encoder = nn.Sequential(self.encoder_layers)
        self.decoder = nn.Sequential(self.decoder_layers)

    def forward(self, input: Tensor) -> Tensor:
        """Forwards the input through the network and returns the output.

        Args:
            input (Tensor): The network input of shape [batch, channel, height, width, depth]

        Returns:
            Tensor: The decoded output of shape [batch, channel, height, width, depth]
        """
        input = self._from_input_shape(input)

        latent = self.encoder(input)
        output = self.decoder(latent)

        return self._to_output_shape(output)

    def encode(self, input: Tensor) -> Tensor:
        """Forwards the input through the encoder part and returns the latent representation.

        Args:
            input (Tensor): The network input of shape [batch, channel, height, width, depth]

        Returns:
            Tensor: The latent representation of shape [batch, channel, height, width, depth]
        """
        input = self._from_input_shape(input)

        latent = self.encoder(input)

        return self._to_latent_shape(latent)

    def decode(self, latent: Tensor) -> Tensor:
        """Forwards the latent representation through the decoder part and returns the decoded output.

        Args:
            latent (Tensor): The latent representation of shape [batch, channel, height, width, depth]

        Returns:
            Tensor: The decoded output of shape [batch, channel, height, width, depth]
        """
        latent = self._from_latent_shape(latent)

        output = self.decoder(latent)

        return self._to_output_shape(output)

    def _from_input_shape(self, tensor: Tensor) -> Tensor:
        """Transforms an input tensor of shape [batch, channel, height, width, depth] into the
        shape required for this model.

        Args:
            tensor (Tensor): Tensor of shape [batch, channel, depth, height, width].

        Returns:
            Tensor: Transformed tensor of shape [batch, channel, width, depth, height]
        """
        return tensor.permute(0, 1, 4, 2, 3)

    def _to_output_shape(self, tensor: Tensor) -> Tensor:
        """Transforms the output of the model into the desired shape of the output:
        [batch, channel, height, width, depth]

        Args:
            tensor (Tensor): Tensor of shape [batch, channel, width, depth, height].

        Returns:
            Tensor: Transformed tensor of shape [batch, channel, depth, height, width]
        """
        return tensor.permute(0, 1, 3, 4, 2)

    def _to_latent_shape(self, tensor: Tensor) -> Tensor:
        """Transforms the output of the encoder model into the desired
        shape of the latent representation: [batch, channel, height, width, depth]

        Args:
            tensor (Tensor): Tensor of shape [batch, channel, width, depth, height].

        Returns:
            Tensor: Transformed tensor of shape [batch, channel, depth, height, width]
        """
        return tensor.permute(0, 1, 3, 4, 2)

    def _from_latent_shape(self, tensor: Tensor) -> Tensor:
        """Transforms a latent representation of shape [batch, channel, height, width, depth]
        into the shape required for the decoder model

        Args:
            tensor (Tensor): Tensor of shape [batch, channel, depth, height, width].

        Returns:
            Tensor: Transformed tensor of shape [batch, channel, width, depth, height]
        """
        return tensor.permute(0, 1, 4, 2, 3)
