from typing import Dict
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
import lightning as L
import neuralop as no


class cFNOModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        n_modes_width: int = 16,
        n_modes_height: int = 16,
        hidden_channels: int = 16,
        in_channels: int = 4,
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
        self.loss = no.H1Loss(d=3)

    def forward(self, x, a):
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

        # rest of forward method continues...

        # concat x and a
        mask[:, 0, H - 1, :] = ax
        return self.model(torch.cat([x, mask], dim=1))

    def predict(self, input: Tensor, length) -> Tensor:
        pred = []
        # autoregressive model steps
        out = input.squeeze(dim=2)
        for _ in range(length):
            out = self.forward(out)
            pred.append(out.detach().cpu())
        return torch.stack(pred, dim=2)

    def model_step(
        self, input: Tensor, actions: Tensor, target: Tensor, stage: str
    ) -> Dict[str, Tensor]:
        loss_list = []
        rmse_list = []
        # autoregressive model steps
        out = input.squeeze(dim=2)
        for idx in range(target.shape[2]):
            action = actions[:, idx]
            out = self.forward(out, action)
            loss_list.append(self.loss(out, target[:, idx]))
            rmse_list.append(torch.sqrt(mse_loss(out, target[:, idx])))
            # out = out.detach()

        # log
        loss = torch.stack(loss_list).mean()
        rmse = torch.stack(rmse_list).mean()
        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True)
        self.log(f"{stage}/RMSE", rmse, prog_bar=True, logger=True)

        return {
            "loss": loss,
            "rmse": torch.stack(rmse_list).detach(),
        }

    def training_step(self, batch, batch_idx):
        (x, a), y = batch
        return self.model_step(x, a, y, stage="train")

    def validation_step(self, batch, batch_idx):
        (x, a), y = batch
        return self.model_step(x, a, y, stage="val")

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            (x, a), y = batch
            return self.model_step(x, a, y, stage="test")

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
