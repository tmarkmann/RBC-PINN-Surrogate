import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.nn.functional import mse_loss
import torch
import wandb


class Metrics3DCallback(Callback):
    def __init__(
        self,
    ):
        self.data = []

    # Testing callbacks
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        # get metrics in size (T)
        rmse = outputs["rmse"].numpy()
        nrsse = outputs["nrsse"].numpy()

        # save in df format
        for t in range(rmse.shape[0]):
            self.data.append(
                {
                    "batch_idx": batch_idx,
                    "step": t,
                    "rmse": rmse[t],
                    "nrsse": nrsse[t],
                }
            )

    def on_test_end(self, trainer, pl_module) -> None:
        df = pd.DataFrame(self.data)
        im1 = self.plot_metric(df, "rmse")
        im2 = self.plot_metric(df, "nrsse")

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_table("test/Table-Metrics", dataframe=df)
            trainer.logger.log_image("test/Plot-RMSE", [im1])
            trainer.logger.log_image("test/Plot-NRSSE", [im2])

    def plot_metric(self, df: pd.DataFrame, metric: str):
        fig = plt.figure()
        sns.set_theme()
        ax = sns.lineplot(data=df, x="step", y=metric)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xlabel("Time Step")
        ax.set_ylim(bottom=0, top=1)

        # save as image
        im = wandb.Image(fig, caption=metric)
        plt.close(fig)
        return im


def rmse(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse_loss(pred, target))


def nrsse(pred: Tensor, target: Tensor) -> Tensor:
    eps = torch.finfo(pred.dtype).eps
    num = torch.linalg.vector_norm(pred - target, dim=[0, 2, 3, 4])
    denom = torch.linalg.vector_norm(target, dim=[0, 2, 3, 4]) + eps
    # return mean across channels for scalar metric
    return (num / denom).mean()


def compute_q(state, T_mean_ref, profile=False):
    T = state[0]
    uz = state[3]
    theta = T - T_mean_ref
    q = uz * theta

    if profile:
        return q.mean(dim=(1, 2)).numpy()
    else:
        return q.mean().numpy()


def compute_profile_qprime_rms(state_seq, T_mean_ref):
    # [T,H,W,D]
    T = state_seq[0]
    uz = state_seq[3]

    theta = T - T_mean_ref
    q = uz * theta

    q_mean_th = q.mean(dim=(0, 2, 3), keepdim=True)
    q_prime = q - q_mean_th

    profile = torch.sqrt(torch.mean(q_prime**2, dim=(0, 2, 3))).cpu().numpy()
    return profile  # shape (H,)


def compute_qprime_z(state_seq, T_mean_ref, z):
    # fields: [T, H, W, D]
    T = state_seq[0]
    uz = state_seq[3]

    theta = T - T_mean_ref
    q = uz * theta  # [T, H, W, D]

    # time–horizontal mean per height for q
    q_mean_th = q.mean(dim=(0, 2, 3), keepdim=True)  # [1, H, 1, 1]
    q_prime = q - q_mean_th  # [T, H, W, D]

    return q_prime[:, z, :, :].reshape(-1).detach().cpu().numpy()


def compute_histogram(qprime, xlim=(-1, 1)):
    bins = 100
    hist, _ = np.histogram(qprime, bins=bins, range=xlim, density=True)
    return hist


def compute_kinetic_energy(state):
    ux = state[1]
    uy = state[2]
    uz = state[3]

    ke = 0.5 * (ux**2 + uy**2 + uz**2)
    return ke.mean()


def compute_divergence(state, domain_size, mode="fd"):
    # state in [C, D, H, W]
    u = state[1]
    v = state[2]
    w = state[3]

    # grid spacings inferred from tensor shape and physical lengths (Lx, Ly, Lz)
    Lx, Ly, Lz = domain_size
    ny, nz, nx = u.shape
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    # centered finite difference in z-direction
    dwdz = torch.gradient(w, dim=1, spacing=dz)[0]

    # Mode: Finite Difference
    if mode == "fd":
        # finite difference in x, y, z
        dudx = torch.gradient(u, dim=2, spacing=dx)[0]
        dvdy = torch.gradient(v, dim=0, spacing=dy)[0]
    # Mode: Spectral + Finite Difference
    elif mode == "spec":
        # spectral derivatives along periodic x/y, finite difference along non-periodic z
        kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=dx).view(1, 1, nx)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(ny, d=dy).view(ny, 1, 1)

        u_hat = torch.fft.fftn(u, dim=(0, 2))
        v_hat = torch.fft.fftn(v, dim=(0, 2))

        dudx = torch.fft.ifftn(1j * kx * u_hat, dim=(0, 2)).real
        dvdy = torch.fft.ifftn(1j * ky * v_hat, dim=(0, 2)).real

    return dudx + dvdy + dwdz
