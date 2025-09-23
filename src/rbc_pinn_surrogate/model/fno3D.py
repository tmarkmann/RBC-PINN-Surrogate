from typing import Dict, Callable
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
import lightning as L
import neuralop as no


class FNO3DModule(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        preprocess: bool = False,
        n_modes_width: int = 16,
        n_modes_height: int = 16,
        n_modes_depth: int = 16,
        hidden_channels: int = 16,
        in_channels: int = 3,
        out_channels: int = 3,
        lifting_channels: int = 16,
        projection_channels: int = 16,
        n_layers: int = 2,
        denormalize: Callable = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pino_loss", "operator", "denormalize"])

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

        # Denormalize
        self.denormalize = denormalize

    def forward(self, x):
        if self.hparams.preprocess:
            # Subtract mean temperature field
            # x is expected to be (B, C, H, W, D); height dimension is index 2
            B, C, H, W, D = x.shape
            Tt, Tb = 1.0, 2.0
            vprof = torch.linspace(Tt, Tb, H, device=x.device, dtype=x.dtype)
            vprof = vprof.view(1, 1, H, 1, 1)
            vprof = vprof.expand(1, 1, H, W, D)
            # one-hot mask for the temperature channel
            ch_mask = torch.zeros(1, C, 1, 1, 1, device=x.device, dtype=x.dtype)
            ch_mask[:, 0] = 1
            profile = ch_mask * vprof

            return self.model(x - profile) + profile
        else:
            return self.model(x)

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
        self, input: Tensor, target: Tensor, stage: str, return_pred: bool = False
    ) -> Dict[str, Tensor]:
        loss = []
        rmse = []
        nmse = []
        # autoregressive model steps
        out = input.squeeze(dim=2)
        for idx in range(target.shape[2]):
            out = self.forward(out)
            loss.append(self.loss(out, target[:, :, idx]))
            # metrics in size (B, T)
            rmse.append(self.rmse(out, target[:, :, idx]))
            nmse.append(self.nmse(out, target[:, :, idx]))

            # for autograd
            # out = out.detach()

        # log
        loss = torch.stack(loss).mean()
        rmse = torch.stack(rmse, dim=1)
        nmse = torch.stack(nmse, dim=1)
        self.log(f"{stage}/loss", loss, prog_bar=True, logger=True)
        self.log(f"{stage}/RMSE", rmse.mean(), prog_bar=True, logger=True)
        self.log(f"{stage}/NMSE", nmse.mean(), prog_bar=True, logger=True)

        return {
            "loss": loss,
            "rmse": rmse.detach(),
            "nmse": nmse.detach(),
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

    def rmse(self, pred: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(
            mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3, 4])
        )

    def nmse(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = torch.finfo(pred.dtype).eps
        diff = pred - target
        # sum over C,H,W,D, keep batch dimension
        nom = (diff * diff).sum(dim=(1, 2, 3, 4))
        denom = (target * target).sum(dim=(1, 2, 3, 4))
        denom = torch.clamp(denom, min=eps)
        return nom / denom
