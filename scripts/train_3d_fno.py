import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
)

from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import FNO3DModule
from rbc_pinn_surrogate.callbacks import Metrics3DCallback


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_fno")
def main(config: DictConfig):
    # set seed for reproducability
    L.seed_everything(config.seed)

    # data
    dm = RBCDatamodule3D(**config.data)

    # model
    denormalize = dm.datasets["train"].denormalize_batch
    model = FNO3DModule(
        denormalize=denormalize,
        **config.model,
    )

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RBC-3D-FNO",
        save_dir=config.paths.output_dir,
        log_model=False,
        config=dict(config),
    )

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=5,
        ),
        Metrics3DCallback(),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        default_root_dir=config.paths.output_dir,
        max_epochs=config.algo.epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=5,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
