from typing import Dict
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
import lightning as L
import neuralop as no


class FNO3DModule(L.LightningModule):
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
        self.loss = no.H1Loss(d=3)

    def forward(self, x):
        return self.model(x)

    def predict(self, input: Tensor, length) -> Tensor:
        pred = []
        # autoregressive model steps
        out = input.squeeze(dim=2)
        for _ in range(length):
            out = self.forward(out)
            pred.append(out.detach())
        return torch.stack(pred, dim=2).detach().cpu()

    def model_step(
        self, input: Tensor, target: Tensor, stage: str, return_pred: bool = False
    ) -> Dict[str, Tensor]:
        loss_list = []
        rmse_list = []
        # autoregressive model steps
        out = input.squeeze(dim=2)
        for idx in range(target.shape[2]):
            out = self.forward(out)
            loss_list.append(self.loss(out, target[:, :, idx]))
            rmse_list.append(torch.sqrt(mse_loss(out, target[:, :, idx])))
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
        x, y = batch
        return self.model_step(x, y, stage="train")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self.model_step(x, y, stage="val")

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x, y = batch
            return self.model_step(x, y, stage="test")

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
