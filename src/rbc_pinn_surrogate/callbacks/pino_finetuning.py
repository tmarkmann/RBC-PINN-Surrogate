import copy
from lightning.pytorch.callbacks import Callback


class OperatorFinetuneCallback(Callback):
    """
    Freeze everything at first.  At `start_epoch` we
    * unfreeze the main model (or some part of it)
    * make an immutable copy of the weights to compare against
    """

    def __init__(self, start_epoch: int = 20):
        super().__init__()
        self.start_epoch = start_epoch
        self._frozen_ref = None

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.start_epoch:
            # take a deep copy of current weights
            self._frozen_ref = (
                copy.deepcopy(pl_module.model).eval().to(pl_module.device)
            )
            for p in self._frozen_ref.parameters():
                p.requires_grad = False

            pl_module.set_finetuning_phase()

    @property
    def reference_model(self):
        return self._frozen_ref
