import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
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
        # get metrics in form (B, T)
        rmse = outputs["rmse"].cpu().numpy()
        nmse = outputs["nmse"].cpu().numpy()

        # save in df form
        samples = rmse.shape[0]
        steps = rmse.shape[1]
        for i in range(samples):
            for j in range(steps):
                self.data.append(
                    {
                        "idx": batch_idx * samples + i,
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "step": j,
                        "rmse": rmse[i, j],
                        "nmse": nmse[i, j],
                    }
                )

    def on_test_end(self, trainer, pl_module) -> None:
        df = pd.DataFrame(self.data)
        im1 = self.plot_metric(df, "rmse")
        im2 = self.plot_metric(df, "nmse")

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_table("test/Table-Metrics", dataframe=df)
            trainer.logger.log_image("test/Plot-RMSE", [im1])
            trainer.logger.log_image("test/Plot-NMSE", [im2])

    def plot_metric(self, df: pd.DataFrame, metric: str):
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
