from typing import Any, Dict, Tuple, Callable

import lightning.pytorch as pl
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch
from torch import Tensor

from rbc_pinn_surrogate.model.components import (
    Autoencoder3D,
    KoopmanOperator,
)
from rbc_pinn_surrogate.metrics import NormalizedSumSquaredError


class LRAN3DModule(pl.LightningModule):
    def __init__(
        self,
        # Autoencoder params
        latent_dimension: int,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        input_shape: Tuple[int, int, int],
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
        denormalize: Callable = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["denormalize"])

        # Model
        activation = nn.GELU
        self.autoencoder = Autoencoder3D(
            latent_dimension=latent_dimension,
            input_channel=input_channel,
            base_filters=base_filters,
            kernel_size=kernel_size,
            activation=activation,
            input_shape=input_shape,
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
        self.denormalize = denormalize

        # Debugging
        D, H, W = input_shape
        self.example_input_array = torch.zeros(1, input_channel, D, H, W)

    def forward(self, x: Tensor) -> Tensor:
        g = self.autoencoder.encode(x)
        g_next = self.operator(g)
        x_hat = self.autoencoder.decode(g_next)
        return x_hat

    def predict(self, input: Tensor, length) -> Tensor:
        with torch.no_grad():
            pred = []
            # autoregressive model steps
            out = input.squeeze(dim=2)
            for _ in range(length):
                out = self.forward(out)
                if self.denormalize is not None:
                    out = self.denormalize(out.detach().cpu())
                pred.append(out)
            return torch.stack(pred, dim=2)

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

        # Metrics
        rmse = self.rmse(x_hat, x)
        nmse = self.nmse(x_hat, x)

        # Log
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            {
                f"{stage}/RMSE": rmse.mean(),
                f"{stage}/NMSE": nmse.mean(),
                f"{stage}/loss/reconstruction": reconstruction,
                f"{stage}/loss/forward": forward,
                f"{stage}/loss/hidden": hidden,
            },
            on_step=False,
            on_epoch=True,
        )

        return {
            "loss": loss,
            "rmse": rmse.detach(),
            "nmse": nmse.detach(),
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

    def rmse(self, pred: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(mse_loss(pred, target, reduction="none").mean(dim=[1, 3, 4]))

    def nmse(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = torch.finfo(pred.dtype).eps
        diff = pred - target
        # sum over C,H,W,D, keep batch dimension
        nom = (diff * diff).sum(dim=(1, 3, 4))
        denom = (target * target).sum(dim=(1, 3, 4))
        denom = torch.clamp(denom, min=eps)
        return nom / denom
