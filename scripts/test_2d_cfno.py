import pathlib
import logging
import tempfile
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import lightning as L
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
import wandb

from rbc_pinn_surrogate.data import RBCDatamodule2DControl
from rbc_pinn_surrogate.model import cFNO2DModule
import rbc_pinn_surrogate.callbacks.metrics_2d as metrics



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
        log_videos(pred[0].numpy(), target[0].numpy())

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


def log_videos(preds, targets):
    # generate videos
    videos = []
    for field in ["T", "U", "W"]:
        # ground truth video
        vgt = sequence2video(targets, "Ground Truth", field)
        cgt = f"{field} - Ground Truth"
        videos.append(wandb.Video(vgt, caption=cgt, format="mp4"))
        # prediction video
        vp = sequence2video(preds, "Prediction", field)
        cp = f"{field} - Prediction"
        videos.append(wandb.Video(vp, caption=cp, format="mp4"))
    wandb.log({"test/examples": videos})


def sequence2video(
    sequence,
    caption: str,
    field="T",
    colormap="rainbow",
    fps=2,
) -> str:
    # set up path
    path = pathlib.Path(f"{tempfile.gettempdir()}/rbcfno").resolve()
    path.mkdir(parents=True, exist_ok=True)
    # config fig
    fig, ax = plt.subplots()
    ax.set_axis_off()

    if field == "T":
        vmin, vmax = 1, 2.75
        channel = 0
    elif field == "U":
        vmin, vmax = None, None
        channel = 1
    elif field == "W":
        vmin, vmax = None, None
        channel = 2
    else:
        raise ValueError(f"Unknown field: {field}")

    # create video
    artists = []
    steps = sequence.shape[1]
    for i in range(steps):
        artists.append(
            [
                ax.imshow(
                    sequence[channel][i],
                    cmap=colormap,
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                )
            ],
        )
    ani = animation.ArtistAnimation(fig, artists, blit=True)

    # save as mp4
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    path = path / f"video_{field}_{caption}.mp4"
    ani.save(path, writer=writer)
    plt.close(fig)
    return str(path)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    logging.getLogger("matplotlib.animation").setLevel(logging.ERROR)
    main()
