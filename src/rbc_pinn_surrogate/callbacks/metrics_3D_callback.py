import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger


class Metrics3DCallback(Callback):
    def __init__(
        self,
    ):
        self.samples = []

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        rmse_seq = outputs["rmse"].cpu().numpy()
        self.samples.append(rmse_seq)
        # Remove rmse from outputs to free memory
        del outputs["rmse"]

    def on_test_end(self, trainer, pl_module) -> None:
        im = self.plot_metric()
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image("test/Plot-RMSE", [im])

    def plot_metric(self):
        # plot the rmse sequence using matplotlib
        df = pd.DataFrame(
            self.samples, columns=[i for i in range(self.samples[0].shape[0])]
        )
        df = df.melt(var_name="Time", value_name="RMSE")
        sns.set_theme()
        plt.figure(figsize=(7, 7))
        sns.lineplot(data=df, x="Time", y="RMSE")
        plt.title("RMSE Sequence Over Time")
        plt.xlabel("Time Step")
        plt.ylim(0, 0.5)
        plt.ylabel("RMSE")
        plt.tight_layout()
        im = plt.gcf()
        plt.close()
        return im
