import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from rbc_pinn_surrogate.data import RBCDatamodule2D
from rbc_pinn_surrogate.model import FNOModule
from rbc_pinn_surrogate.callbacks import (
    SequenceMetricsCallback,
    ExamplesCallback,
    MetricsCallback,
    ClearMemoryCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="fno")
def main(config: DictConfig):
    # data
    dm = RBCDatamodule2D(data_dir="data/datasets/2D", **config.data)

    # model
    model = FNOModule(lr=config.algo.lr, **config.model)

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RayleighBenard-FNO",
        save_dir=config.paths.output_dir,
        log_model=False,
    )

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=7,
        ),
        ExamplesCallback(
            train_freq=20,
            dm=dm,
        ),
        MetricsCallback(
            name="metrics",
            key_groundtruth="y",
            key_prediction="y_hat",
        ),
        SequenceMetricsCallback(
            name="sequence",
            key_groundtruth="y",
            key_prediction="y_hat",
        ),
        ClearMemoryCallback(),
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
