import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.nn.functional import mse_loss
import torch
import wandb


class Metrics3DCallback(Callback):
    def __init__(
        self,
    ):
        self.data = []

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        # get metrics in size (T)
        rmse = outputs["rmse"].numpy()
        nrsse = outputs["nrsse"].numpy()

        # save in df format
        for t in range(rmse.shape[0]):
            self.data.append(
                {
                    "batch_idx": batch_idx,
                    "step": t,
                    "rmse": rmse[t],
                    "nrsse": nrsse[t],
                }
            )

    def on_test_end(self, trainer, pl_module) -> None:
        df = pd.DataFrame(self.data)
        im1 = self.plot_metric(df, "rmse")
        im2 = self.plot_metric(df, "nrsse")

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_table("test/Table-Metrics", dataframe=df)
            trainer.logger.log_image("test/Plot-RMSE", [im1])
            trainer.logger.log_image("test/Plot-NRSSE", [im2])

    def plot_metric(self, df: pd.DataFrame, metric: str):
        fig = plt.figure()
        sns.set_theme()
        ax = sns.lineplot(data=df, x="step", y=metric)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("Time Step")
        ax.set_ylim(bottom=0, top=1)

        # save as image
        im = wandb.Image(fig, caption=metric)
        plt.close(fig)
        return im


def rmse(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse_loss(pred, target))


def nrsse(pred: Tensor, target: Tensor) -> Tensor:
    eps = torch.finfo(pred.dtype).eps
    num = torch.linalg.vector_norm(pred - target, dim=[0, 2, 3, 4])
    denom = torch.linalg.vector_norm(target, dim=[0, 2, 3, 4]) + eps
    # return mean across channels for scalar metric
    return (num / denom).mean()
