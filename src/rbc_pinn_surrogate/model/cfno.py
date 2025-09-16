from typing import Dict
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
import lightning as L
import neuralop as no


class cFNOModule(L.LightningModule):
    def __init__(
        self,
        control_mask: bool = False,
        lr: float = 1e-3,
        n_modes_width: int = 16,
        n_modes_height: int = 16,
        hidden_channels: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        lifting_channels: int = 16,
        projection_channels: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model parameters
        self.model = no.models.TFNO2d(
            n_modes_width=n_modes_width,
            n_modes_height=n_modes_height,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
        )

        # Loss Function
        self.loss = mse_loss  # no.H1Loss(d=2)

        # inverse normalization for visualization
        self.denorm_mean = torch.tensor([1.5, 0.0, 0.0]).view(1, out_channels, 1, 1, 1)
        self.denorm_std = torch.tensor([0.25, 0.35, 0.35]).view(
            1, out_channels, 1, 1, 1
        )

    def control_mask(self, x, a):
        # control mask
        B, C, H, W = x.shape
        mask = torch.zeros((B, 1, H, W), device=x.device)

        # upsample action to match input shape
        nh = a.shape[1]
        ax = a.view(B, 1, 1, nh)
        ax = torch.nn.functional.interpolate(ax, size=(1, W), mode="nearest")
        ax = ax.squeeze()

        # write mask on the bottom boundary
        mask[:, 0, H - 1, :] = ax

        return torch.cat([x, mask], dim=1)

    def forward(self, x: Tensor, a: Tensor | None = None):
        if self.hparams.control_mask and a is not None:
            x = self.control_mask(x, a)
        return self.model(x)

    def predict(
        self, input: Tensor, length: int | None = None, actions: Tensor | None = None
    ) -> Tensor:
        if actions is not None:
            T = actions.shape[1]
        else:
            assert length is not None and length > 0, (
                "Provide `length` when `actions` is None"
            )
            T = length

        preds = []
        out = input
        for t in range(T):
            a_t = None if actions is None else actions[:, t]
            out = self.forward(out, a_t)
            preds.append(out)
        return torch.stack(preds, dim=2)

    def model_step(
        self, sequence: Tensor, actions: Tensor, stage: str
    ) -> Dict[str, Tensor]:
        x0 = sequence[:, 0]  # [B, C, H, W]
        target = sequence[:, 1:]  # [B, T-1, C, H, W]

        loss_list: list[Tensor] = []
        rmse_list: list[Tensor] = []
        pred_list: list[Tensor] = []

        out = x0
        for t in range(target.shape[1]):
            a_t = actions[:, t]  # [B, A]
            out = self.forward(out, a_t)  # [B, C, H, W]
            y_t = target[:, t]  # [B, C, H, W]

            # primary training loss (LpLoss or H1Loss)
            loss_list.append(self.loss(out, y_t))

            # RMSE metric (no grad to avoid graph bloat)
            with torch.no_grad():
                rmse_list.append(torch.sqrt(mse_loss(out, y_t)))

            # store prediction for logging
            pred_list.append(out)

            # If you want truncated BPTT, uncomment:
            # out = out.detach()

        loss = torch.stack(loss_list, dim=0).mean()
        rmse = torch.stack(rmse_list, dim=0).mean()

        y = target.detach().transpose(1, 2)  # [B, T-1, C, H, W] -> [B, C, T-1, H, W]
        y_hat = torch.stack(pred_list, dim=2)  # [B, C, T-1, H, W]

        # unnormalize for vis
        y = y * self.denorm_std.to(y.device) + self.denorm_mean.to(y.device)
        y_hat = y_hat * self.denorm_std.to(y_hat.device) + self.denorm_mean.to(
            y_hat.device
        )

        # log metrics
        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True)
        self.log(f"{stage}/RMSE", rmse, prog_bar=True, logger=True)
        return {
            "loss": loss,
            "rmse": torch.stack(rmse_list),
            "y": y,
            "y_hat": y_hat,
        }

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
        return {"optimizer": optimizer}

    def load_state_dict(self, state_dict, strict: bool = True):
        # Remove metadata from neuralop library TODO check if useful
        state_dict.pop("_metadata", None)
        return super().load_state_dict(state_dict, strict)
