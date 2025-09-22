import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.tuner import Tuner

from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import FNO3DModule


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_fno")
def main(config: DictConfig):
    # data
    dm = RBCDatamodule3D(**config.data)

    # model
    model = FNO3DModule(**config.model)

    # trainer
    trainer = L.Trainer(
        accelerator="auto",
        default_root_dir=config.paths.output_dir,
        max_epochs=1,
    )

    tuner = Tuner(trainer)
    # Auto-scale batch size by growing it exponentially (default)
    tuner.scale_batch_size(model, datamodule=dm, mode="power")

    # training
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
