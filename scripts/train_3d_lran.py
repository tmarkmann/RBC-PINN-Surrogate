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
from rbc_pinn_surrogate.model import LRAN3DModule
from rbc_pinn_surrogate.callbacks import (
    Metrics3DCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_lran")
def main(config: DictConfig):
    # seed
    L.seed_everything(config.seed, workers=True)

    # data
    dm = RBCDatamodule3D(**config.data)
    dm.setup("fit")

    # model
    denormalize = dm.datasets["train"].denormalize_batch
    model = LRAN3DModule(
        input_shape=[32, 48, 48],
        denormalize=denormalize,
        **config.model,
    )

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RBC-3D-LRAN",
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
        ModelCheckpoint(
            dirpath=f"{config.paths.output_dir}/checkpoints/",
            save_top_k=1,
            monitor="val/RMSE",
            mode="min",
        ),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        default_root_dir=config.paths.output_dir,
        max_epochs=config.algo.epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=3,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
