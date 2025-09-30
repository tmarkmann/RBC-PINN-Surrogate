import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from rbc_pinn_surrogate.data import RBCDatamodule2D
from rbc_pinn_surrogate.model import FNOModule, AutoRegressiveFNOModule
from rbc_pinn_surrogate.callbacks import (
    SequenceMetricsCallback,
    ExamplesCallback,
    MetricsCallback,
    ClearMemoryCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="2d_fno")
def main(config: DictConfig):
    # seed
    L.seed_everything(config.seed, workers=True)

    # data
    dm = RBCDatamodule2D(**config.data)
    dm.setup("fit")

    # model
    denormalize = dm.datasets["train"].denormalize_batch

    if config.model.type == "3d":
        model = FNOModule(denormalize=denormalize, **config.model)
    elif config.model.type == "2d":
        model = AutoRegressiveFNOModule(denormalize=denormalize, **config.model)
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RBC-2D-FNO",
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
            patience=15,
        ),
        MetricsCallback(
            key_groundtruth="y",
            key_prediction="y_hat",
        ),
        ExamplesCallback(
            train_freq=20,
        ),
        SequenceMetricsCallback(
            key_groundtruth="y",
            key_prediction="y_hat",
        ),
        ModelCheckpoint(
            dirpath=f"{config.paths.output_dir}/checkpoints/",
            save_top_k=1,
            save_weights_only=True,
            monitor="val/loss",
            mode="min",
        ),
        ClearMemoryCallback(),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        default_root_dir=config.paths.output_dir,
        check_val_every_n_epoch=1,
        max_epochs=config.algo.epochs,
        callbacks=callbacks,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
