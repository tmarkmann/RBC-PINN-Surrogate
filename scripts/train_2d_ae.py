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
from rbc_pinn_surrogate.data import RBCDatamodule2D
from rbc_pinn_surrogate.model import Autoencoder2DModule
from rbc_pinn_surrogate.callbacks import (
    Examples2DCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="2d_ae")
def main(config: DictConfig):
    # config convert
    config = OmegaConf.to_container(config, resolve=True)
    output_dir = config["paths"]["output_dir"]

    # seed
    L.seed_everything(config["seed"], workers=True)

    # data
    dm = RBCDatamodule2D(**config["data"])
    dm.setup("fit")

    # model
    denormalize = dm.datasets["train"].denormalize_batch
    model = Autoencoder2DModule(**config["model"], inv_transform=denormalize)

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RBC-2D-AE",
        save_dir=output_dir,
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
        Examples2DCallback(
            train_freq=20,
        ),
        ModelCheckpoint(
            dirpath=f"{output_dir}/checkpoints/",
            save_top_k=1,
            save_weights_only=True,
            monitor="val/RMSE",
            mode="min",
        ),
    ]

    # trainer
    trainer = L.Trainer(
        **config["trainer"],
        logger=logger,
        default_root_dir=output_dir,
        log_every_n_steps=10,
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
