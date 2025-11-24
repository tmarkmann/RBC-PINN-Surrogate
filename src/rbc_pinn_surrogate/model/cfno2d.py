from typing import Callable, Dict, Optional
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
import torch.nn as nn
import lightning as L
import neuralop as no


class cFNO2DModule(L.LightningModule):
    def __init__(
        self,
        control_mask: bool = False,
        mask_version: int = 1,
        lr: float = 1e-3,
        n_modes_width: int = 16,
        n_modes_height: int = 16,
        hidden_channels: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        lifting_channels: int = 16,
        projection_channels: int = 16,
        n_layers: int = 4,
        denormalize: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["denormalize"])
        self.denormalize = denormalize

        # Model parameters
        ch = 1 if control_mask else 0
        self.model = no.models.TFNO2d(
            n_modes_width=n_modes_width,
            n_modes_height=n_modes_height,
            hidden_channels=hidden_channels,
            in_channels=in_channels + ch,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
        )

        # Loss Function
        self.loss = mse_loss  # no.H1Loss(d=2)

    def control_mask_v1(self, x: Tensor, a: Tensor):
        # control mask
        B, C, H, W = x.shape
        a = a.unsqueeze(1).unsqueeze(2)

        # Interpolate to W
        a = nn.functional.interpolate(a, size=(1, W), mode="nearest")
        mask = a.expand(B, 1, H, W)

        return torch.cat([x, mask], dim=1)

    def control_mask_v2(self, x: Tensor, a: Tensor):
        # control mask
        B, C, H, W = x.shape
        a = a.unsqueeze(1).unsqueeze(2)

        # Interpolate to W
        a = nn.functional.interpolate(a, size=(1, W), mode="nearest")

        # Write to bottom boundary
        mask = torch.zeros((B, 1, H, W), device=x.device)
        mask[:, :, -1, :] = a.squeeze(2)

        return torch.cat([x, mask], dim=1)

    def control_mask(self, x: Tensor, a: Tensor):
        v = self.hparams.mask_version
        if v == 1:
            return self.control_mask_v1(x, a)
        elif v == 2:
            return self.control_mask_v2(x, a)
        else:
            raise ValueError(f"Unknown control mask version: {v}")

    def forward(self, x: Tensor, a: Tensor | None = None):
        if self.hparams.control_mask:
            x = self.control_mask(x, a)
        return self.model(x)

    def multi_step_2d(self, x: Tensor, actions: Tensor) -> Tensor:
        # x has shape [B, C, H, W]
        xt = x
        length = actions.shape[1]
        # preds has shape [length, B, C, H, W]
        preds = x.new_empty(length, *x.shape)

        # autoregressive prediction
        for t in range(length):
            at = actions[:, t]
            y_next = self.forward(xt, at)
            preds[t] = y_next
            xt = y_next

        # return [B, C, T, H, W]
        return preds.permute(1, 2, 0, 3, 4)

    def model_step(
        self, sequence: Tensor, actions: Tensor, stage: str
    ) -> Dict[str, Tensor]:
        x0 = sequence[:, :, 0]  # [B, C, H, W]
        target = sequence[:, :, 1:]  # [B, C, T-1, H, W]

        preds = self.multi_step_2d(x0, actions)

        # loss
        loss = self.loss(preds, target)
        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True)

        # unnormalize for vis and metrics
        preds = self.denormalize(preds)
        target = self.denormalize(target)

        return {
            "loss": loss,
            "ground_truth": target,
            "prediction": preds,
        }

    def predict(self, x: Tensor, actions: Tensor) -> Tensor:
        self.eval()
        with torch.no_grad():
            return self.multi_step_2d(x.to(self.device), actions.to(self.device))

    def training_step(self, batch, batch_idx):
        x, a = batch
        return self.model_step(x, a, stage="train")

    def validation_step(self, batch, batch_idx):
        x, a = batch
        return self.model_step(x, a, stage="val")

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, a = batch
            return self.model_step(x, a, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        # Remove metadata from neuralop library TODO check if useful
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)
