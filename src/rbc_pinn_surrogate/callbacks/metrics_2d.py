from typing import Dict

from lightning.pytorch.callbacks import Callback
from torch import Tensor
import torch
from torch.nn.functional import mse_loss


class Metrics2DCallback(Callback):
    def __init__(
        self,
        key_groundtruth: str = "ground_truth",
        key_prediction: str = "prediction",
    ):
        self.key_gt = key_groundtruth
        self.key_pred = key_prediction

    # Training callbacks
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self.log_metrics(outputs, stage="train")

    # Validation callbacks
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.log_metrics(outputs, stage="val")

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        self.log_metrics(outputs, stage="test")

    # Helper function
    def log_metrics(self, output: Dict[str, Tensor], stage: str):
        gt = output[self.key_gt].detach().cpu()
        pred = output[self.key_pred].detach().cpu()

        self.log(f"{stage}/RMSE", rmse(pred, gt))
        self.log(f"{stage}/NRSSE", nrsse(pred, gt))


def rmse(preds: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse_loss(preds, target))


def nrsse(pred: Tensor, target: Tensor) -> Tensor:
    eps = torch.finfo(pred.dtype).eps
    num = torch.linalg.vector_norm(pred - target, dim=[0, 2, 3])
    denom = torch.linalg.vector_norm(target, dim=[0, 2, 3]) + eps
    return (num / denom).mean()


def compute_q(state, T_mean_ref, profile=False):
    pass


def compute_profile_qprime_rms(state_seq, T_mean_ref):
    pass


def compute_qprime_z(state_seq, T_mean_ref, z):
    pass


def compute_histogram(qprime, xlim=(-1, 1)):
    pass
