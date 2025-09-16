from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

from rbc_pinn_surrogate.utils.vis3D import animation_3d


class Example3DCallback(Callback):
    def __init__(self, dir: str):
        self.dir = dir

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        logger = trainer.logger
        gt = outputs["y"].cpu().numpy()
        pred = outputs["y_hat"].cpu().numpy()

        for idx in range(gt.shape[0]):
            video = animation_3d(
                gt=gt[idx],
                pred=pred[idx],
                feature="T",
                anim_dir=self.dir,
                anim_name=f"{batch_idx}_{idx}_T.mp4",
            )
            if isinstance(logger, WandbLogger):
                logger.log_video(
                    "test/videos",
                    videos=[video],
                    caption=[f"{batch_idx}-{idx}-T"],
                    format=["mp4"],
                )
