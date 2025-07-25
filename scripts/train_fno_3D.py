import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
)

from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import FNO3DModule
from rbc_pinn_surrogate.callbacks import Metrics3DCallback


@hydra.main(version_base="1.3", config_path="../configs", config_name="fno3D")
def main(config: DictConfig):
    # data
    dm = RBCDatamodule3D(data_dir="data/datasets/3D", **config.data)

    # model
    # inv_transform = NormalizeInverse(mean=cfg.data.means, std=cfg.data.stds)
    model = FNO3DModule(**config.model)

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RayleighBenard-3D-FNO",
        save_dir=config.paths.output_dir,
        log_model=False,
    )

    # callbacks
    callbacks = [
        RichProgressBar(leave=True),
        RichModelSummary(),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=5,
        ),
        Metrics3DCallback(),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        precision="16-mixed",
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
