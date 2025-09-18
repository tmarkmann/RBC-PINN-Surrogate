from typing import Any, Dict, Tuple, Callable

import lightning.pytorch as pl
import torch.nn as nn
import torch
from torch import Tensor

from rbc_pinn_surrogate.model.components import Autoencoder, KoopmanOperator
from rbc_pinn_surrogate.metrics import NormalizedSumSquaredError


class LRANModule(pl.LightningModule):
    def __init__(
        self,
        # Autoencoder params
        latent_dimension: int,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        ae_ckpt: str,
        # Loss params
        loss: str,
        lambda_id: float,
        lambda_fwd: float,
        lambda_hid: float,
        # Optimizer params
        lr_operator: float,
        lr_autoencoder: float,
        # Misc
        inv_transform: Callable = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["inv_transform"])

        # Model
        activation = nn.GELU
        self.autoencoder = Autoencoder(
            latent_dimension, input_channel, base_filters, kernel_size, activation
        )
        if ae_ckpt is not None:
            ckpt = torch.load(ae_ckpt, map_location=self.device)
            self.autoencoder.load_weights(ckpt, freeze=False)
        self.operator = KoopmanOperator(latent_dimension)

        # Loss
        if loss == "r-mse":
            self.loss = NormalizedSumSquaredError()
        elif loss == "mse":
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"Loss {loss} not supported")

        # Denormalize
        self.denormalize = inv_transform

        # Debugging
        self.example_input_array = torch.zeros(1, input_channel, 64, 96)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        g = self.autoencoder.encode(x)
        g_next = self.operator(g)
        x_hat = self.autoencoder.decode(g_next)
        return x_hat, g_next, g

    def model_step(self, x: Tensor, stage: str) -> Dict[str, Tensor]:
        seq_length = x.shape[2]
        g, g_hat, x_hat = [], [], []
        # Get ground truth for observables
        with torch.no_grad():
            for tau in range(0, seq_length):
                g.append(self.autoencoder.encode(x[:, :, tau]).detach())

        # Prediction
        g0 = g[0].detach()
        g_hat.append(g0)
        x_hat.append(self.autoencoder.decode(g0))
        # Predict sequence
        for tau in range(1, seq_length):
            g_hat.append(self.operator(g_hat[tau - 1]))
            x_hat.append(self.autoencoder.decode(g_hat[tau]))
        # To tensor
        g = torch.stack(g, dim=1)
        g_hat = torch.stack(g_hat, dim=1)
        x_hat = torch.stack(x_hat, dim=2)

        # Loss
        reconstruction = self.loss(x_hat[:, :, :1], x[:, :, :1])
        forward = self.loss(x_hat[:, :, 1:], x[:, :, 1:])
        hidden = self.loss(g_hat[:, 1:], g[:, 1:])
        loss = (
            self.hparams.lambda_id * reconstruction
            + self.hparams.lambda_fwd * forward
            + self.hparams.lambda_hid * hidden
        )

        # Log
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            {
                f"{stage}/loss/reconstruction": reconstruction,
                f"{stage}/loss/forward": forward,
                f"{stage}/loss/hidden": hidden,
            },
            on_step=False,
            on_epoch=True,
        )

        # Apply inverse transform
        if self.denormalize is not None:
            with torch.no_grad():
                x = self.denormalize(x.detach())
                x_hat = self.denormalize(x_hat.detach())

        return {
            "loss": loss,
            "y": x,
            "y_hat": x_hat,
        }

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        _, y = batch
        return self.model_step(y, stage="train")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        _, y = batch
        return self.model_step(y, stage="val")

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        _, y = batch
        return self.model_step(y, stage="test")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            params=[
                {
                    "params": self.autoencoder.parameters(),
                    "lr": self.hparams.lr_autoencoder,
                },
                {
                    "params": self.operator.parameters(),
                    "lr": self.hparams.lr_operator,
                },
            ],
        )
        return {"optimizer": optimizer}
