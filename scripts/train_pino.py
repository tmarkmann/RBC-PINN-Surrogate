import hydra
import math
import numpy as np
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from rbc_pinn_surrogate.model import PINOModule
from rbc_pinn_surrogate.data import RBCDatamodule
from rbc_pinn_surrogate.loss import RBCEquationLoss
from rbc_pinn_surrogate.callbacks import (
    SequenceMetricsCallback,
    SequenceExamplesCallback,
    MetricsCallback,
    ClearMemoryCallback,
    FinetuneCallback,
)


@hydra.main(version_base="1.3", config_path="../configs", config_name="pino")
def main(config: DictConfig):
    # data
    dm = RBCDatamodule(data_dir="data/datasets/2D", **config.data)

    # pino loss
    train_time = config.data.train_length * config.data.stride_time
    kappa = 1 / np.sqrt(config.data.ra * 0.7)
    nu = np.sqrt(0.7 / config.data.ra)
    pino_loss = RBCEquationLoss(
        domain_width=2 * math.pi,
        domain_height=2,
        time=train_time,
        kappa=kappa,
        nu=nu,
    )

    # model
    model = PINOModule(lr=config.algo.lr, pino_loss=pino_loss, **config.model)

    # logger
    logger = WandbLogger(
        entity="sail-project",
        project="RayleighBenard-PINN",
        save_dir=config.paths.output_dir,
        log_model=False,
    )

    # callbacks
    callbacks = [
        RichProgressBar(),
        RichModelSummary(),
        # EarlyStopping(
        #     monitor="val/loss",
        #     mode="min",
        #     patience=7,
        # ),
        # SequenceExamplesCallback(
        #     train_freq=20,
        # ),
        # MetricsCallback(
        #     name="metrics",
        #     key_groundtruth="y",
        #     key_prediction="y_hat",
        # ),
        # SequenceMetricsCallback(
        #     name="sequence",
        #     key_groundtruth="y",
        #     key_prediction="y_hat",
        #     dt=config.data.stride_time,
        # ),
        # ClearMemoryCallback(),
    ]
    if config.algo.do_finetuning:
        callbacks.append(
            FinetuneCallback(
                finetune_epoch=config.algo.epochs_finetuning,
            )
        )

    # trainer
    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        default_root_dir=config.paths.output_dir,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        max_epochs=config.algo.epochs,
        callbacks=callbacks,
    )

    # training
    trainer.fit(model, dm)

    # rollout on test set
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
