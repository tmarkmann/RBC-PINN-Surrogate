import hydra
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)

from rbc_pinn_surrogate.data import RBCDatamodule2DControl
from rbc_pinn_surrogate.model import cFNO2DModule
from rbc_pinn_surrogate.callbacks import (
    Examples2DCallback,
    Metrics2DCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="2d_cfno")
def main(config: DictConfig):
    # config convert
    config = OmegaConf.to_container(config, resolve=True)
    output_dir = config["paths"]["output_dir"]

    # seed
    L.seed_everything(config["seed"], workers=True)

    # data
    dm = RBCDatamodule2DControl(**config["data"])
    dm.setup("fit")

    # model
    denormalize = dm.denormalize
    model = cFNO2DModule(denormalize=denormalize, **config["model"])

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RBC-2D-cFNO",
        save_dir=output_dir,
        log_model=False,
        config=config,
    )

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=20,
            min_delta=1e-5,
        ),
        Examples2DCallback(train_freq=10),
        Metrics2DCallback(key_groundtruth="ground_truth", key_prediction="prediction"),
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
