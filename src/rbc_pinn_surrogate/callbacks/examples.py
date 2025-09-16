import logging
import pathlib
import tempfile

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger, WandbLogger
from matplotlib import animation
from matplotlib import pyplot as plt


class ExamplesCallback(Callback):
    def __init__(self, train_freq: int = 5):
        self.train_freq = train_freq

        logger = logging.getLogger("matplotlib.animation")
        logger.setLevel(logging.ERROR)

    # Training callbacks
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 and trainer.current_epoch % self.train_freq == 0:
            with torch.no_grad():
                self.log_output(outputs, 0, "train", trainer.logger)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0 and trainer.current_epoch % self.train_freq == 0:
            with torch.no_grad():
                self.log_output(outputs, 0, "val", trainer.logger)

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        for idx in range(len(batch)):
            self.log_output(outputs, idx, "test", trainer.logger)

    def log_output(self, outputs: dict, idx: int, stage: str, logger: Logger):
        # unpack sequence
        y = outputs["y"][idx].detach().cpu()
        y_hat = outputs["y_hat"][idx].detach().cpu()

        # generate videos
        videos = []
        captions = []
        for field in ["T", "U", "W"]:
            videos.append(self.sequence2video(y, "Ground Truth", field))
            captions.append(f"{field} - Ground Truth")
            videos.append(self.sequence2video(y_hat, "Prediction", field))
            captions.append(f"{field} - Prediction")
            videos.append(
                self.sequence2video(
                    np.abs(y - y_hat), "Difference", field, colormap="binary"
                )
            )
            captions.append(f"{field} - Difference")

        # log to wandb
        if isinstance(logger, WandbLogger):
            logger.log_video(
                f"{stage}/examples",
                videos,
                caption=captions,
                format=["mp4" for i in range(len(videos))],
            )

    def sequence2video(
        self,
        sequence,
        caption: str,
        field="T",
        colormap="rainbow",
        fps=2,
    ) -> str:
        # set up path
        path = pathlib.Path(f"{tempfile.gettempdir()}/rbcfno").resolve()
        path.mkdir(parents=True, exist_ok=True)
        # config fig
        fig, ax = plt.subplots()
        ax.set_axis_off()

        if field == "T":
            vmin, vmax = 1, 2
            channel = 0
        elif field == "U":
            vmin, vmax = None, None
            channel = 1
        elif field == "W":
            vmin, vmax = None, None
            channel = 2
        else:
            raise ValueError(f"Unknown field: {field}")

        # create video
        artists = []
        steps = sequence.shape[1]
        for i in range(steps):
            artists.append(
                [
                    ax.imshow(
                        sequence[channel][i],
                        cmap=colormap,
                        origin="lower",
                        vmin=vmin,
                        vmax=vmax,
                    )
                ],
            )
        ani = animation.ArtistAnimation(fig, artists, blit=True)

        # save as mp4
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        path = path / f"video_{field}_{caption}.mp4"
        ani.save(path, writer=writer)
        plt.close(fig)
        return str(path)
