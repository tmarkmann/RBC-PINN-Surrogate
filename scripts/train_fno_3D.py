import hydra
from omegaconf import DictConfig
import lightning as L

from rbc_pinn_surrogate.data import RBCDatamodule
from rbc_pinn_surrogate.model import FNO3DModule



@hydra.main(version_base="1.3", config_path="../configs", config_name="fno3D")
def main(config: DictConfig):
    # data
    dm = RBCDatamodule(data_dir="data/datasets/3D", **config.data)
    
    # model
    # inv_transform = NormalizeInverse(mean=cfg.data.means, std=cfg.data.stds)
    model = FNO3DModule(**config.model)

    # trainer
    trainer = L.Trainer(
        accelerator="auto",
        default_root_dir=config.paths.output_dir,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        max_epochs=config.algo.epochs,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
