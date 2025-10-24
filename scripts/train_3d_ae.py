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
from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import Autoencoder3DModule


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_ae")
def main(config: DictConfig):
    # seed
    L.seed_everything(config.seed, workers=True)

    # data
    dm = RBCDatamodule3D(**config.data)
    dm.setup("fit")

    # model
    denormalize = dm.datasets["train"].denormalize_batch
    model = Autoencoder3DModule(
        **config.model,
        inv_transform=denormalize,
    )

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RBC-3D-AE",
        save_dir=config.paths.output_dir,
        log_model=False,
        config=dict(config),
    )

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=3),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=20,
            min_delta=1e-5,
        ),
        ModelCheckpoint(
            dirpath=f"{config.paths.output_dir}/checkpoints/",
            save_top_k=1,
            save_weights_only=True,
            monitor="val/loss",
            mode="min",
        ),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator=config.trainer.device,
        default_root_dir=config.paths.output_dir,
        max_epochs=config.trainer.epochs,
        detect_anomaly=config.trainer.detect_anomaly,
        callbacks=callbacks,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")

    # finish logging
    logger.experiment.finish()


if __name__ == "__main__":
    main()
