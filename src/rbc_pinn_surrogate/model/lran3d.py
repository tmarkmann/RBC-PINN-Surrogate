from typing import Any, Dict, Callable

import lightning.pytorch as pl
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch
from torch import Tensor

from rbc_pinn_surrogate.model.components import (
    Autoencoder3D,
    KoopmanOperator,
)
import rbc_pinn_surrogate.callbacks.metrics_3d as metrics


class LRAN3DModule(pl.LightningModule):
    def __init__(
        self,
        # LRAN params
        input_size: int,
        ae_ckpt: str,
        latent_dimension: int,
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

        # Debugging
        self.example_input_array = torch.zeros(1, *input_size)

        # Autoencoder
        ckpt = torch.load(ae_ckpt, map_location=self.device)
        self.autoencoder = Autoencoder3D.from_checkpoint(ckpt)

        # Get latent spatial dimensions
        with torch.no_grad():
            dummy = self.autoencoder.encode(self.example_input_array)
            latent_shape = dummy.shape[1:]

        self.operator = nn.Sequential(
            nn.Flatten(),
            KoopmanOperator(latent_dimension),
            nn.Unflatten(1, latent_shape),
        )

        # Loss
        self.loss = mse_loss

        # Denormalize
        self.denormalize = denormalize

    def forward(self, x: Tensor) -> Tensor:
        # Encode input
        g = self.autoencoder.encode(x)
        g_next = self.operator(g)
        x_hat = self.autoencoder.decode(g_next)
        return x_hat

    def predict(self, input: Tensor, length) -> Tensor:
        self.eval()
        with torch.no_grad():
            pred = []
            # autoregressive model steps
            out = input.squeeze(dim=2).to(self.device)
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
            g0 = self.autoencoder.encode(x).detach()
            g = [
                self.autoencoder.encode(y[:, :, t]).detach() for t in range(seq_length)
            ]

        # Reconstruction
        x_hat = self.autoencoder.decode(g0)

        # Predict sequence in latent space
        g_hat, y_hat = [], []
        g_prev = g0
        for _ in range(seq_length):
            g_next = self.operator(g_prev)
            y_hat.append(self.autoencoder.decode(g_next))
            g_hat.append(g_next)
            g_prev = g_next

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
        rmse = (
            torch.stack(
                [metrics.rmse(y_hat[:, :, t], y[:, :, t]) for t in range(seq_length)]
            )
            .detach()
            .cpu()
        )
        nrsse = (
            torch.stack(
                [metrics.nrsse(y_hat[:, :, t], y[:, :, t]) for t in range(seq_length)]
            )
            .detach()
            .cpu()
        )

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
        x, y = batch
        return self.model_step(x, y, stage="train")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        return self.model_step(x, y, stage="val")

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        return self.model_step(x, y, stage="test")

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
