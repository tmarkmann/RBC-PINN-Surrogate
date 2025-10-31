from lightning.pytorch.callbacks import Callback
from rbc_pinn_surrogate.utils.vis_3d import animation_3d


class Example3DCallback(Callback):
    def __init__(self, dir: str):
        self.dir = dir

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx % 10 == 0:
            vid = self.get_example_anim(outputs, batch_idx)
            trainer.logger.log_video(
                "val/videos",
                videos=[vid],
                caption=[f"{batch_idx}-T"],
                format=["mp4"],
            )

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        vid = self.get_example_anim(outputs, batch_idx)
        trainer.logger.log_video(
            "test/videos",
            videos=[vid],
            caption=[f"{batch_idx}-T"],
            format=["mp4"],
        )

    def get_example_anim(self, outputs, batch_idx: int):
        gt = outputs["ground_truth"].cpu().numpy()
        pred = outputs["prediction"].cpu().numpy()

        return animation_3d(
            gt=gt,
            pred=pred,
            feature="T",
            anim_dir=self.dir,
            anim_name=f"{batch_idx}_T.mp4",
        )
