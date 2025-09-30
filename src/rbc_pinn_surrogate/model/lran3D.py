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
import rbc_pinn_surrogate.callbacks.metrics_3D as metrics


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
        self.loss = mse_loss

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

    def model_step(
        self, input: Tensor, target: Tensor, stage: str
    ) -> Dict[str, Tensor]:
        x = input.squeeze(dim=2)
        y = target
        seq_length = target.shape[2]
        g, g_hat, y_hat = [], [], []

        # Get ground truth for observables
        with torch.no_grad():
            g0 = self.autoencoder.encode(input.squeeze(dim=2)).detach()
            for t in range(0, seq_length):
                gt = self.autoencoder.encode(target[:, :, t]).detach()
                g.append(gt)

        # Reconstruction
        x_hat = self.autoencoder.decode(g0)

        # Predict sequence
        g0_hat = self.operator(g0)
        g_hat.append(g0_hat)
        for t in range(seq_length):
            # predict
            yt_hat = self.autoencoder.decode(g_hat[t])
            y_hat.append(yt_hat)
            # next g_hat
            g_hat.append(self.operator(g_hat[t]))

        # To tensor
        g = torch.stack(g, dim=1)
        g_hat = torch.stack(g_hat, dim=1)
        y_hat = torch.stack(y_hat, dim=2)

        # Loss
        reconstruction = self.loss(x_hat, x)
        forward = self.loss(y_hat, y)
        hidden = self.loss(g_hat, g)
        loss = (
            self.hparams.lambda_id * reconstruction
            + self.hparams.lambda_fwd * forward
            + self.hparams.lambda_hid * hidden
        )

        # compute Metrics for each t
        rmse = [metrics.rmse(y_hat[:, :, t], y[:, :, t]) for t in range(seq_length)]
        nrsse = [metrics.nrsse(y_hat[:, :, t], y[:, :, t]) for t in range(seq_length)]
        rmse = (torch.stack(rmse).detach().cpu(),)
        nrsse = (torch.stack(nrsse).detach().cpu(),)

        # Log
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(
            {
                f"{stage}/RMSE": rmse.mean(),
                f"{stage}/NRSSE": nrsse.mean(),
                f"{stage}/loss/reconstruction": reconstruction,
                f"{stage}/loss/forward": forward,
                f"{stage}/loss/hidden": hidden,
            },
            on_step=False,
            on_epoch=True,
        )

        return {
            "loss": loss,
            "rmse": rmse,
            "nrsse": nrsse,
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
