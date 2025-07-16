from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer

from rbc_pinn_surrogate.model import PINOModule


class FinetuneCallback(Callback):
    def __init__(self, finetune_epoch: int) -> None:
        super().__init__()
        self.finetune_epoch = finetune_epoch
        self._triggered: bool = False

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: PINOModule
    ) -> None:
        # start finetuning phase at the specified epoch
        if trainer.current_epoch == self.finetune_epoch:
            # get best checkpoint from trainer
            best_ckpt = trainer.checkpoint_callback.best_model_path
            if not best_ckpt:
                raise ValueError("No best checkpoint found. Ensure training has completed.")
            # load the best checkpoint and get model from module
            operator = PINOModule.load_from_checkpoint(
                best_ckpt,
                map_location=trainer.strategy.root_device,
            ).model
            # set weights of nn.Module to non-trainable
            for param in operator.parameters():
                param.requires_grad = False
            
            pl_module.set_finetuning_phase(operator=operator)
            print(f"Finetuning phase started at epoch {self.finetune_epoch}.")
