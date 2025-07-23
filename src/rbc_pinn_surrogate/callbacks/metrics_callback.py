from typing import Dict

from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchmetrics import MeanSquaredError

from rbc_pinn_surrogate.metrics import NormalizedSumSquaredError, NormalizedSumError


class MetricsCallback(Callback):
    def __init__(self, name: str, key_groundtruth: str, key_prediction: str):
        self.name = name
        self.key_groundtruth = key_groundtruth
        self.key_prediction = key_prediction

        # metrics
        self.metrics = [
            NormalizedSumSquaredError(),
            NormalizedSumError(),
            MeanSquaredError(squared=False),
        ]
        self.names = [
            "NSSE",
            "NSE",
            "RMSE",
        ]

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
        for metric, name in zip(self.metrics, self.names):
            self.log(
                f"{stage}/{self.name}-{name}",
                metric(
                    output[self.key_prediction].detach().cpu(),
                    output[self.key_groundtruth].detach().cpu(),
                ),
            )
