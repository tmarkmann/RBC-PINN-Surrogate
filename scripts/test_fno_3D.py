import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import FNO3DModule
from rbc_pinn_surrogate.callbacks import (
    MetricsCallback,
    SequenceMetricsCallback,
    Example3DCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="fno3D_test")
def main(config: DictConfig):
    # data
    dm = RBCDatamodule3D(data_dir="data/datasets/3D", **config.data)

    # model
    # inv_transform = NormalizeInverse(mean=cfg.data.means, std=cfg.data.stds)
    model = FNO3DModule.load_from_checkpoint(config.checkpoint)

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RayleighBenard-3D-FNO",
        save_dir=config.paths.output_dir,
        log_model=False,
    )

    # callbacks
    callbacks = [
        MetricsCallback(
            name="metrics",
            key_groundtruth="y",
            key_prediction="y_hat",
        ),
        SequenceMetricsCallback(
            name="sequence",
            key_groundtruth="y",
            key_prediction="y_hat",
            dt=0.5,
        ),
        Example3DCallback(dir=config.paths.output_dir + "/animation"),
    ]

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        default_root_dir=config.paths.output_dir,
        callbacks=callbacks,
    )

    # rollout on test set
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
