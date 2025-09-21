from typing import Any, Dict, Tuple, Callable

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor

from rbc_pinn_surrogate.model.components import Autoencoder


class AutoencoderModule(LightningModule):
    def __init__(
        self,
        latent_dimension: int,
        input_channel: int,
        base_filters: int,
        kernel_size: int,
        lr: float,
        compile: bool,
        inv_transform: Callable = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.activation = nn.GELU
        self.autoencoder = Autoencoder(
            latent_dimension, input_channel, base_filters, kernel_size, self.activation
        )

        # Loss function
        self.criterion = torch.nn.functional.mse_loss

        # Debugging
        self.example_input_array = torch.zeros(1, input_channel, 64, 96)

    def forward(self, x: Tensor) -> Tensor:
        # forward
        x_hat, _ = self.autoencoder(x)
        return x_hat

    def model_step(self, batch: Tensor, stage: str) -> Tuple[Tensor, Tensor, Tensor]:
        # check input dimensions
        assert batch.shape[1] == 1, (
            "Expect sequence length of 1 for autoencoder training"
        )
        x = batch.squeeze(dim=1)

        # model forward
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Apply inverse transform
        if self.denormalize is not None:
            with torch.no_grad():
                x = self.denormalize(x.detach())
                x_hat = self.denormalize(x_hat.detach())

        return {
            "loss": loss,
            "x_hat": x_hat,
            "x": x,
        }

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, stage="test")

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            params=self.trainer.model.parameters(), lr=self.hparams.lr
        )
        return {"optimizer": optimizer}
