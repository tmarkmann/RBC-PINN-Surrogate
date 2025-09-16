import hydra
from omegaconf import DictConfig

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
)

from rbc_pinn_surrogate.data import RBCDatamodule2DControl
from rbc_pinn_surrogate.model import cFNOModule
from rbc_pinn_surrogate.callbacks import (
    ExamplesCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="cfno")
def main(config: DictConfig):
    # data
    dm = RBCDatamodule2DControl(data_dir="data/datasets/2D-control", **config.data)

    # model
    model = cFNOModule(lr=config.algo.lr, **config.model)

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RayleighBenard-cFNO",
        save_dir=config.paths.output_dir,
        log_model=False,
    )

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(),
        # EarlyStopping(
        #    monitor="val/loss",
        #    mode="min",
        #    patience=7,
        # ),
        ExamplesCallback(train_freq=5),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        default_root_dir=config.paths.output_dir,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        max_epochs=config.algo.epochs,
        callbacks=callbacks,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
