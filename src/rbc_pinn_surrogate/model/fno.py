from typing import Dict, Callable
import torch
from torch import Tensor
import lightning as L
import neuralop as no
import math


class FNOModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_modes_space: int = 16,
        n_modes_time: int = 16,
        hidden_channels: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        lifting_channels: int = 16,
        projection_channels: int = 16,
        n_layers: int = 2,
        loss: str = "H1",
        denormalize: Callable = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["denormalize"])
        self.denormalize = denormalize

        # Model parameters
        self.model = no.models.TFNO3d(
            n_modes_width=n_modes_space,
            n_modes_height=n_modes_space,
            n_modes_depth=n_modes_time,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
        )

        # Loss Function
        if loss == "L2":
            self.loss = no.LpLoss(d=3, p=2)
        elif loss == "H1":
            self.loss = no.H1Loss(d=3)
        elif loss == "mse":
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")

    def forward(self, x):
        return self.model(x)

    def single_step(self, x: Tensor, y: Tensor, stage: str) -> Dict[str, Tensor]:
        # Forward pass and compute loss
        pred = self.forward(x)

        # data loss
        loss = self.loss(pred, y)
        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True)

        # denormalize for logging
        if self.denormalize is not None:
            pred = self.denormalize(pred)
            y = self.denormalize(y)

        return {
            "loss": loss,
            "y": y,
            "y_hat": pred,
        }

    def multi_step(self, input: Tensor, target: Tensor, stage: str):
        input_length = input.shape[2]
        target_length = target.shape[2]

        # Preallocate prediction tensor 
        pred = input.new_empty(
            input.shape[0],
            input.shape[1],
            target_length,
            *input.shape[3:],
        )

        # no autograd
        with torch.inference_mode():
            n_chunks = math.ceil(target_length / input_length)
            x = input
            for k in range(n_chunks):
                y = self.forward(x)
                start = k * input_length
                end = min(start + input_length, target_length)
                step = end - start
                pred[:, :, start:end] = y[:, :, :step]
                # Always keep the latest input window size for the next step
                x = y[:, :, -input_length:]

        # Compute and log metrics
        loss = self.loss(pred, target)
        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True)

        # denormalize for logging
        if self.denormalize is not None:
            with torch.no_grad():
                y_hat_vis = self.denormalize(pred)
                y_vis = self.denormalize(target)
        else:
            y_hat_vis = pred
            y_vis = target

        return {
            "loss": loss,
            "y": y_vis,
            "y_hat": y_hat_vis,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.single_step(x, y, stage="train")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self.multi_step(x, y, stage="val")
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        return self.multi_step(x, y, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.get("weight_decay", 0.0),
        )
        return {"optimizer": optimizer}

    def load_state_dict(self, state_dict, strict: bool = True):
        # Remove metadat from neuralop library TODO check if useful
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)
