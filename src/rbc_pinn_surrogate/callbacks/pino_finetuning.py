from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer

from rbc_pinn_surrogate.model import PINOModule


class FinetuneCallback(Callback):
    def __init__(self, finetune_epoch: int) -> None:
        super().__init__()
        self.finetune_epoch = finetune_epoch

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: PINOModule
    ) -> None:
        # start finetuning phase at the specified epoch
        if trainer.current_epoch == self.finetune_epoch:
            # set finetuning phase
            pl_module.set_finetuning_phase()
            print(f"Finetuning phase started at epoch {self.finetune_epoch}.")
