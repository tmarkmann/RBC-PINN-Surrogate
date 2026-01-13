import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import UNet3DModule
from rbc_pinn_surrogate.callbacks import Metrics3DCallback


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_unet")
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
    model = UNet3DModule(**config["model"])

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RBC-3D-UNET",
        save_dir=output_dir,
        log_model=False,
        config=config,
        tags=config["tags"],
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
        Metrics3DCallback(),
        ModelCheckpoint(
            dirpath=f"{output_dir}/checkpoints/",
            save_top_k=1,
            save_weights_only=True,
            monitor="val/loss",
            mode="min",
        ),
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
