import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import Autoencoder3DModule
from rbc_pinn_surrogate.callbacks import Example3DCallback


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_ae")
def main(config: DictConfig):
    # config convert
    config = OmegaConf.to_container(config, resolve=True)
    output_dir = config["paths"]["output_dir"]

    # seed
    L.seed_everything(config["seed"], workers=True)

    # data
    dm = RBCDatamodule3D(**config["data"])
    dm.setup("fit")

    # model
    denormalize = dm.datasets["train"].denormalize_batch
    model = Autoencoder3DModule(
        **config["model"],
        inv_transform=denormalize,
    )

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RBC-3D-AE",
        save_dir=output_dir,
        log_model=False,
        config=config,
        tags=config["tags"],
    )

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=3),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=15,
        ),
        ModelCheckpoint(
            dirpath=f"{output_dir}/checkpoints/",
            save_top_k=1,
            save_weights_only=True,
            monitor="val/loss",
            mode="min",
        ),
        Example3DCallback(dir=f"{output_dir}/examples/"),
    ]

    # trainer
    trainer = L.Trainer(
        **config["trainer"],
        logger=logger,
        default_root_dir=output_dir,
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
