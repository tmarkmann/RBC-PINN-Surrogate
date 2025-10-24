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
from rbc_pinn_surrogate.model import FNO2DModule
from rbc_pinn_surrogate.callbacks import (
    SequenceMetricsCallback,
    Examples2DCallback,
    Metrics2DCallback,
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
    model = FNO2DModule(denormalize=denormalize, **config.model)

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
        Metrics2DCallback(
            key_groundtruth="ground_truth",
            key_prediction="prediction",
        ),
        Examples2DCallback(
            train_freq=20,
        ),
        SequenceMetricsCallback(
            key_groundtruth="ground_truth",
            key_prediction="prediction",
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
        max_epochs=config.algo.epochs,
        callbacks=callbacks,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
