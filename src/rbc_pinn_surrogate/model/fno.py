from typing import Dict
import torch
from torch import Tensor
import lightning as L
import neuralop as no


class FNOModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        n_modes_width: int = 16,
        n_modes_height: int = 16,
        n_modes_depth: int = 16,
        hidden_channels: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        lifting_channels: int = 16,
        projection_channels: int = 16,
        n_layers: int = 2,
        loss: str = "H1",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pino_loss", "operator"])

        # Model parameters
        self.model = no.models.TFNO3d(
            n_modes_width=n_modes_width,
            n_modes_height=n_modes_height,
            n_modes_depth=n_modes_depth,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
        )

        # Loss Function
        if loss == "L2":
            self.loss = no.L2Loss()
        elif loss == "H1":
            self.loss = no.H1Loss(d=3)
        elif loss == "mse":
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")

    def forward(self, x):
        return self.model(x)

    def model_step(self, x: Tensor, y: Tensor, stage: str) -> Dict[str, Tensor]:
        # Forward pass and compute loss
        pred = self.forward(x)

        # data loss
        loss = self.loss(pred, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True)

        return {
            "loss": loss,
            "x": x,
            "y": y,
            "y_hat": pred,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="train")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="val")

    def test_step(self, batch, batch_idx):
        input, target = batch
        input_length = input.shape[2]
        target_length = target.shape[2]

        # autoregressive model steps
        length = 0
        x = input
        pred = torch.empty(
            x.shape[0],
            x.shape[1],
            target_length,
            *x.shape[3:],
            device=x.device,
            dtype=x.dtype,
        )

        while length < target_length:
            y = self.forward(x)
            step = min(input_length, target_length - length)
            pred[:, :, length : length + step] = y[:, :, :step]
            x = y
            length += step

        # Compute and log metrics
        loss = self.loss(pred, target)
        self.log("test/loss", loss, logger=True)

        return {
            "loss": loss,
            "y": target,
            "y_hat": pred,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
        )
        return {"optimizer": optimizer}

    def load_state_dict(self, state_dict, strict: bool = True):
        # Remove metadat from neuralop library TODO check if useful
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)
