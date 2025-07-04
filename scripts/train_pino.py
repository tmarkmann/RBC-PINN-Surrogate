import hydra
import math
import numpy as np
from omegaconf import DictConfig
from rbc_pinn_surrogate.data import RBCDatamodule
from rbc_pinn_surrogate.loss.equation import RBCInteriorLoss


@hydra.main(version_base="1.3", config_path="../configs", config_name="pino")
def main(config: DictConfig):
    # data
    dm = RBCDatamodule(data_dir="data/2D", **config.data)

    # model
    # model = FNO3DModule(lr=config.algo.lr, **config.model)

    # logger
    # logger = WandbLogger(
    #    entity="sail-project",
    #    project="RayleighBenard-PINN",
    #    save_dir=config.paths.output_dir,
    #    log_model=False,
    # )

    # test loss on cfd data
    dm.setup("fit")
    train_time = config.data.train_length * config.data.stride_time
    kappa = 1 / np.sqrt(config.data.ra * 0.7)
    nu = np.sqrt(0.7 / config.data.ra)
    pino_loss = RBCInteriorLoss(
        domain_width=2 * math.pi,
        domain_height=2,
        time=train_time,
        kappa=kappa,
        nu=nu,
    )

    for batch in dm.train_dataloader():
        x, y = batch
        loss = pino_loss(y)
        print(f"Loss: {loss}")


if __name__ == "__main__":
    main()
