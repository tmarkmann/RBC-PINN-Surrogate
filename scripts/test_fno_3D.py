import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import torch
from torch.nn.functional import mse_loss
from rbc_pinn_surrogate.data import RBCDatamodule3D
from rbc_pinn_surrogate.model import FNO3DModule
from rbc_pinn_surrogate.utils.vis3D import animation_3d


@hydra.main(version_base="1.3", config_path="../configs", config_name="fno3D_test")
def main(config: DictConfig):
    # device
    device = best_device()

    # data
    dm = RBCDatamodule3D(data_dir="data/datasets/3D", **config.data)
    dm.setup("test")

    # model
    model = FNO3DModule.load_from_checkpoint(config.checkpoint)
    model.to(device)

    # wandb run
    wandb.init(
        project="RayleighBenard-3D-FNO",
        config=dict(config),
        dir=config.paths.output_dir,
        tags=["test"],
    )

    # loop
    rmse_list = []
    for batch, (x, y) in enumerate(tqdm(dm.test_dataloader(), desc="Testing")):
        with torch.no_grad():
            pred = model.predict(x.to(device), y.shape[2])

        # loop through each sample in the batch
        for idx in range(pred.shape[0]):
            # metrics
            loss = model.loss(pred[idx], y[idx]).cpu()
            rmse = torch.sqrt(mse_loss(pred[idx], y[idx], reduction="none")).cpu()

            # rmse
            rmse_step = rmse.mean(dim=(0, 2, 3, 4))
            rmse_mean = rmse.mean()
            rmse_list.append(rmse_step.numpy())

            # vis
            path = animation_3d(
                gt=y[idx].cpu().numpy(),
                pred=pred[idx].cpu().numpy(),
                feature="T",
                anim_dir=config.paths.output_dir + "/animations",
                anim_name=f"test_{batch}_{idx}.mp4",
            )
            video = wandb.Video(path, format="mp4", caption=f"Test {batch}.{idx}")

            # log
            wandb.log(
                {
                    "test/loss": loss,
                    "test/RMSE": rmse_mean,
                    "test/video": video,
                }
            )

    # plot RMSE
    im = plot_rmse(rmse_list)
    wandb.log({"test/Plot-RMSE": wandb.Image(im)})


def plot_rmse(samples):
    # plot the rmse sequence using matplotlib
    df = pd.DataFrame(samples, columns=[i for i in range(samples[0].shape[0])])
    df = df.melt(var_name="Time", value_name="RMSE")
    sns.set_theme()
    plt.figure(figsize=(7, 7))
    sns.lineplot(data=df, x="Time", y="RMSE")
    plt.title("RMSE Sequence Over Time")
    plt.xlabel("Time Step")
    plt.ylim(0, 0.5)
    plt.ylabel("RMSE")
    plt.tight_layout()
    im = plt.gcf()
    plt.close()
    return im


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # first free NVIDIA GPU
    if torch.backends.mps.is_available():  # Apple-silicon
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
