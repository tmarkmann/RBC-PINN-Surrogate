from typing import Dict, Callable
import torch
from torch import Tensor
import lightning as L
import neuralop as no


class AutoRegressiveFNOModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_modes_space: int = 16,
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
        self.model = no.models.TFNO2d(
            n_modes_width=n_modes_space,
            n_modes_height=n_modes_space,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
        )

        # Loss Function
        if loss == "L2":
            self.loss = no.LpLoss(d=2, p=2)
        elif loss == "H1":
            self.loss = no.H1Loss(d=2)
        elif loss == "mse":
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss}")

    def forward(self, x):
        return self.model(x)

    def model_step(self, x: Tensor, y: Tensor, stage: str) -> Dict[str, Tensor]:
        loss_list: list[Tensor] = []
        pred_list: list[Tensor] = []

        # x0 is the last input frame
        out = x[:, :, -1]  # [B, C, T, H, W]

        # autoregressive prediction
        for t in range(y.shape[2]):
            out = self.forward(out)
            y_t = y[:, :, t]  # [B, C, H, W]

            # save loss and prediction
            loss_list.append(self.loss(out, y_t))
            pred_list.append(out)
        
        # stack predictions and loss
        loss = torch.stack(loss_list, dim=0).mean()
        y_hat = torch.stack(pred_list, dim=2)

        # denormalize for logging
        if self.denormalize is not None:
            y_hat = self.denormalize(y_hat)
            y = self.denormalize(y)

        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "x": x,
            "y": y,
            "y_hat": y_hat,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="train")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="test")

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
