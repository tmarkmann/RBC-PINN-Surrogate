import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import Tensor
from torch.nn.functional import mse_loss
import wandb


class SequenceMetricsCallback(Callback):
    def __init__(
        self,
        key_groundtruth: str,
        key_prediction: str,
    ):
        self.key_groundtruth = key_groundtruth
        self.key_prediction = key_prediction
        self.data = []

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        pred = outputs[self.key_prediction].cpu()
        gt = outputs[self.key_groundtruth].cpu()

        # Update each metric
        self.update(pred, gt, batch_idx)

    def on_test_end(self, trainer, pl_module) -> None:
        df = self.get_dataframe()
        im1 = self.plot_metrics(df, "rmse")
        im2 = self.plot_metrics(df, "nsse")

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_table("test/Table-Metrics", dataframe=df)
            trainer.logger.log_image("test/Plot-RMSE", [im1])
            trainer.logger.log_image("test/Plot-NSSE", [im2])

    def update(self, pred: list[Tensor], target: list[Tensor], batch_idx: int):
        for idx in range(target.shape[0]):
            for step in range(target.shape[2]):
                self.data.append(
                    {
                        "idx": batch_idx * target.shape[0] + idx,
                        "batch_idx": batch_idx,
                        "sample_idx": idx,
                        "step": step,
                        "rmse": self.rmse(pred[idx, :, step], target[idx, :, step]),
                        "nsse": self.nsse(pred[idx, :, step], target[idx, :, step]),
                        "mean_q_pred": self.mean_q(pred[idx, :, step]),
                        "mean_q_target": self.mean_q(target[idx, :, step]),
                    }
                )

    def rmse(self, preds: Tensor, target: Tensor):
        return (torch.sqrt(mse_loss(preds, target)).item(),)

    def nsse(self, pred: Tensor, target: Tensor):
        eps = torch.finfo(pred.dtype).eps
        mse_val = mse_loss(pred, target, reduction="sum")
        denom = torch.sum(target * target) + eps
        return mse_val / denom

    def mean_q(self, state: Tensor):
        T = state[0]
        uz = state[2]
        q = uz * (T - torch.mean(T))
        return torch.mean(q).item()

    def div_loss(self, state: Tensor):
        dx = 6 * torch.pi / 96
        dz = 2 / 64

        dudx = torch.gradient(state[1], dim=3, spacing=dx)[0]
        dwdz = torch.gradient(state[2], dim=2, spacing=dz)[0]
        div = dudx + dwdz

        return mse_loss(div, torch.zeros_like(div))

    def get_dataframe(self):
        return pd.DataFrame(self.data)

    def plot_metrics(self, df: pd.DataFrame, metric: str):
        fig = plt.figure()
        sns.set_theme()
        ax = sns.lineplot(data=df, x="step", y=metric)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("Time Step")
        ax.set_ylim(bottom=0, top=0.5)

        # save as image
        im = wandb.Image(fig, caption=metric)
        plt.close(fig)
        return im
