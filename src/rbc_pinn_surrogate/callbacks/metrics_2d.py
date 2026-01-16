from typing import Dict, Tuple

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger, Logger
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
import wandb

from rbc_pinn_surrogate.data.dataset import Field2D


class Metrics2DCallback(Callback):
    def __init__(
        self,
        key_groundtruth: str = "ground_truth",
        key_prediction: str = "prediction",
    ):
        self.key_gt = key_groundtruth
        self.key_pred = key_prediction

    # Set up w&b metrics
    def on_train_start(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger):
            for stage in ["train", "val", "test"]:
                wandb.define_metric(f"{stage}/loss", summary="min")
                wandb.define_metric(f"{stage}/RMSE", summary="min")
                wandb.define_metric(f"{stage}/NRSSE", summary="min")

    # Training callbacks
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.log_metrics(outputs, stage="train", logger=trainer.logger)

    # Validation callbacks
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.log_metrics(outputs, stage="val", logger=trainer.logger)

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.log_metrics(outputs, stage="test", logger=trainer.logger)

    # Helper function
    def log_metrics(self, output: Dict[str, Tensor], stage: str, logger: Logger):
        gt = output[self.key_gt].detach().cpu()
        pred = output[self.key_pred].detach().cpu()

        if isinstance(logger, WandbLogger):
            logger.log_metrics(
                {
                    f"{stage}/RMSE": rmse(pred, gt),
                    f"{stage}/NRSSE": nrsse(pred, gt),
                }
            )


def rmse(preds: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse_loss(preds, target))


def nrsse(pred: Tensor, target: Tensor) -> Tensor:
    # state: [C,H,W]
    eps = torch.finfo(pred.dtype).eps
    num = torch.linalg.vector_norm(pred - target, dim=[1, 2])
    denom = torch.linalg.vector_norm(target, dim=[1, 2]) + eps
    return (num / denom).mean()


def compute_q(state: Tensor, T_mean_ref: Tensor, profile=False):
    # state: [C,H,W]
    T = state[Field2D.T]
    w = state[Field2D.W]
    theta = T - T_mean_ref
    q = w * theta

    if profile:
        return q.mean(dim=1).numpy()
    else:
        return q.mean().numpy()


def compute_profile_qprime_rms(state_seq: Tensor, T_mean_ref: Tensor):
    # [C,T,H,W]
    T = state_seq[Field2D.T]
    w = state_seq[Field2D.W]

    theta = T - T_mean_ref
    q = w * theta

    q_mean_th = q.mean(dim=(0, 2), keepdim=True)
    q_prime = q - q_mean_th

    profile = torch.sqrt(torch.mean(q_prime**2, dim=(0, 2))).cpu().numpy()
    return profile  # shape (H,)


def compute_qprime_z(state_seq: Tensor, T_mean_ref: Tensor, z: int):
    # [C,T,H,W]
    T = state_seq[Field2D.T]
    w = state_seq[Field2D.W]

    theta = T - T_mean_ref
    q = w * theta

    # time–horizontal mean per height for q
    q_mean_th = q.mean(dim=(0, 2), keepdim=True)  # [1, 1, H, 1]
    q_prime = q - q_mean_th

    return q_prime[:, z, :].reshape(-1).detach().cpu().numpy()


def compute_histogram(qprime, xlim=(-1, 1)):
    bins = 100
    hist, _ = np.histogram(qprime, bins=bins, range=xlim, density=True)
    return hist


def compute_kinetic_energy(state: Tensor):
    # [C,H,W]
    u = state[Field2D.U]
    w = state[Field2D.W]

    ke = 0.5 * (u**2 + w**2)
    return ke.mean()


def compute_divergence(state: Tensor, domain_size: Tuple[float, float]):
    # state in [C,H,W]
    u = state[Field2D.U]
    w = state[Field2D.W]

    # grid spacings inferred from tensor shape and physical lengths (Lx, Ly, Lz)
    Lx, Lz = domain_size
    nz, nx = u.shape
    dx = Lx / nx
    dz = Lz / nz

    dudx = torch.gradient(u, dim=1, spacing=dx)[0]
    dwdz = torch.gradient(w, dim=0, spacing=dz)[0]
    div = dudx + dwdz

    return div, dudx, dwdz
