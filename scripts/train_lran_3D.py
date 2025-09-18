import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import LRAN3DModule
from rbc_pinn_surrogate.callbacks import (
    Metrics3DCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="lran3D")
def main(config: DictConfig):
    # seed
    L.seed_everything(config.seed, workers=True)

    # data
    dm = RBCDatamodule3D(**config.data)
    dm.setup("fit")

    # model
    denormalize = dm.datasets["train"].denormalize_batch
    model = LRAN3DModule(
        input_shape=[32, 48, 48], inv_transform=denormalize, **config.model
    )

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RBC-3D-LRAN",
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
            patience=8,
        ),
        #Metrics3DCallback(),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="cpu",
        default_root_dir=config.paths.output_dir,
        check_val_every_n_epoch=2,
        log_every_n_steps=10,
        max_epochs=config.algo.epochs,
        callbacks=callbacks,
        enable_checkpointing=False,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    # trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
