import hydra
from tqdm import tqdm
from omegaconf import DictConfig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import torch
from torch.nn.functional import mse_loss
from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import FNO3DModule, LRAN3DModule
from rbc_pinn_surrogate.utils.vis3D import animation_3d


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_test")
def main(config: DictConfig):
    # device
    device = best_device()

    # data
    dm = RBCDatamodule3D(**config.data)
    dm.setup("test")

    # model
    if config.model == "fno":
        model = FNO3DModule.load_from_checkpoint(config.checkpoint)
    elif config.model == "lran":
        model = LRAN3DModule.load_from_checkpoint(config.checkpoint)
    model.to(device)

    # wandb run
    wandb.init(
        project=f"RBC-3D-{str(config.model).capitalize()}",
        config=dict(config),
        dir=config.paths.output_dir,
        tags=["test"],
    )

    # loop
    metrics = []
    for batch, (x, y) in enumerate(tqdm(dm.test_dataloader(), desc="Testing")):
        with torch.no_grad():
            pred = model.predict(x.to(device), y.shape[2]).cpu()

        # loop through each sample in the batch
        batch_size = pred.shape[0]
        seq_len = pred.shape[2]
        for idx in range(batch_size):
            # metrics per sample and time step
            loss = model.loss(pred[idx], y[idx])
            rmse = compute_rmse(pred[idx], y[idx])
            nmse = compute_nmse(pred[idx], y[idx])

            for t in range(seq_len):
                metrics.append(
                    {
                        "idx": batch * batch_size + idx,
                        "batch_idx": batch,
                        "sample_idx": idx,
                        "step": t,
                        "rmse": rmse[t].item(),
                        "nmse": nmse[t].item(),
                    }
                )
            wandb.log(
                {
                    "test/loss": loss,
                    "test/RMSE": rmse.mean(),
                    "test/NMSE": nmse.mean(),
                }
            )

            # vis
            if idx == 0:  # only first sample in batch
                path = animation_3d(
                    gt=y[idx].cpu().numpy(),
                    pred=pred[idx].cpu().numpy(),
                    feature="T",
                    anim_dir=config.paths.output_dir + "/animations",
                    anim_name=f"test_{batch}_{idx}.mp4",
                )
                video = wandb.Video(path, format="mp4", caption=f"Test {batch}.{idx}")
                wandb.log({"test/video": video})

    # log metrics
    df = pd.DataFrame(metrics)
    im1 = plot_metric(df, "rmse")
    im2 = plot_metric(df, "nmse")
    wandb.log(
        {
            "test/Plot-RMSE": wandb.Image(im1),
            "test/Plot-NMSE": wandb.Image(im2),
            "test/Table-Metrics": wandb.Table(dataframe=df),
        }
    )


def compute_nmse(pred, target):
    eps = torch.finfo(pred.dtype).eps
    diff = pred - target
    # sum over C,H,W,D, keep batch dimension
    nom = (diff * diff).sum(dim=(0, 2, 3, 4))
    denom = (target * target).sum(dim=(0, 2, 3, 4))
    denom = torch.clamp(denom, min=eps)
    return nom / denom


def compute_rmse(pred, target):
    rmse = torch.sqrt(mse_loss(pred, target, reduction="none"))
    return rmse.mean(dim=(0, 2, 3, 4))


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
        return torch.device("cuda")  # first free NVIDIA GPU
    if torch.backends.mps.is_available():  # Apple-silicon
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
