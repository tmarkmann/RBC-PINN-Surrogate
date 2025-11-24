import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import lightning as L
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb

from rbc_pinn_surrogate.data import RBCDatamodule2DControl
from rbc_pinn_surrogate.model import cFNO2DModule
import rbc_pinn_surrogate.callbacks.metrics_2d as metrics
from rbc_pinn_surrogate.utils.vis_2d import sequence2video


@hydra.main(version_base="1.3", config_path="../configs", config_name="2d_cFNO")
def main(config: DictConfig):
    # config convert
    config = OmegaConf.to_container(config, resolve=True)
    test_cfg = config["test"]
    output_dir = config["paths"]["output_dir"]

    # seed
    L.seed_everything(config["seed"], workers=True)

    # data config
    data_cfg = config["data"]
    data_cfg["horizon"] = test_cfg["horizon"]
    data_cfg["shift"] = test_cfg["shift"]
    data_cfg["types"] = test_cfg["types"]

    dm = RBCDatamodule2DControl(**data_cfg)
    dm.setup("test")
    denorm = dm.denormalize

    # model
    device = best_device()
    model = cFNO2DModule.load_from_checkpoint(test_cfg["checkpoint"])
    model.to(device)

    # wandb run
    wandb.init(
        project="RBC-2D-cFNO",
        config=config,
        dir=output_dir,
        tags=["test"],
    )

    # loop
    list_metrics = []
    for batch_idx, (x, a) in enumerate(tqdm(dm.test_dataloader(), desc="Testing")):
        # unpack batch data
        x0 = x[:, :, 0]
        target = x[:, :, 1:]
        actions = a

        # get model prediction
        pred = model.predict(x0, actions).cpu()
        pred = denorm(pred)
        target = denorm(target)

        # 1) Sequence Metrics NRSSE and RMSE
        for t in range(pred.shape[2]):
            # metrics per sample and time step
            loss = model.loss(pred[:, :, t], target[:, :, t])
            rmse = metrics.rmse(pred[:, :, t], target[:, :, t])
            nrsse = metrics.nrsse(pred[:, :, t], target[:, :, t])

            list_metrics.append(
                {
                    "batch_idx": batch_idx,
                    "step": t,
                    "rmse": rmse.item(),
                    "nrsse": nrsse.item(),
                    "loss": loss.item(),
                }
            )

        # 2) Visualize samples from first batch element
        videos = []
        for field in ["T"]:  # ["T", "U", "W"]
            path = sequence2video(target[0], pred[0], field)
            videos.append(wandb.Video(path, caption=field, format="mp4"))
        wandb.log({"test/examples": videos})

    # Process overall metrics
    df_metrics = pd.DataFrame(list_metrics)

    rmse = df_metrics["rmse"].mean()
    nrsse = df_metrics["nrsse"].mean()

    im_rmse = plot_metric(df_metrics, "rmse")
    im_nrsse = plot_metric(df_metrics, "nrsse")

    wandb.log(
        {
            "test/RMSE": rmse,
            "test/NRSSE": nrsse,
            "test/Table-Metrics": wandb.Table(dataframe=df_metrics),
            "test/Plot-RMSE": im_rmse,
            "test/Plot-NRSSE": im_nrsse,
        }
    )

    # Finish wandb run
    wandb.finish()


def plot_metric(df: pd.DataFrame, metric: str):
    fig = plt.figure()
    sns.set_theme()
    ax = sns.lineplot(data=df, x="step", y=metric)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xlabel("Time Step")
    ax.set_ylim(bottom=0, top=0.5)
    # save as image
    im = wandb.Image(fig, caption=metric)
    plt.close(fig)
    return im


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    logging.getLogger("matplotlib.animation").setLevel(logging.ERROR)
    main()
