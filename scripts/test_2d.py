import pathlib
import tempfile
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import torch
from torch.nn.functional import mse_loss
import numpy as np
from rbc_pinn_surrogate.data import RBCDatamodule2D
from rbc_pinn_surrogate.model import FNOModule, LRANModule
from matplotlib import animation


@hydra.main(version_base="1.3", config_path="../configs", config_name="2d_test")
def main(config: DictConfig):
    # device
    device = "cpu"  # best_device()

    # data
    dm = RBCDatamodule2D(**config.data)
    dm.setup("test")

    # model
    if config.model == "fno":
        model = FNOModule.load_from_checkpoint(config.checkpoint)
    elif config.model == "lran":
        model = LRANModule.load_from_checkpoint(config.checkpoint)
    model.to(device)

    # wandb run
    wandb.init(
        project=f"RBC-2D-{str(config.model).capitalize()}",
        config=dict(config),
        dir=config.paths.output_dir,
        tags=["test"],
    )

    # loop
    metrics = []
    for batch, (x, y) in enumerate(tqdm(dm.test_dataloader(), desc="Testing")):
        with torch.no_grad():
            pred = model.predict(x.to(device), y.to(device)).cpu()

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
                        "rmse": float(rmse[t].item()),
                        "nmse": float(nmse[t].item()),
                    }
                )
            wandb.log(
                {
                    "test/loss": float(loss.detach().mean().item()),
                    "test/RMSE": float(rmse.mean().item()),
                    "test/NMSE": float(nmse.mean().item()),
                }
            )

            # vis
            if idx == 0:  # only first sample in batch
                # video vis
                log_videos(pred[idx].numpy(), y[idx].numpy())

                # --- Diagnostics for convective heat flux in 2D ---
                # Use a single spatio-temporal mean temperature over the target sequence as reference
                T_mean_ref = y[idx, 0].mean()

                # Nusselt-like time series: ⟨w (T - ⟨T⟩)⟩ over the full domain
                nu_pred = [
                    compute_mean_q(pred[idx, :, t], T_mean_ref) for t in range(seq_len)
                ]
                nu_target = [
                    compute_mean_q(y[idx, :, t], T_mean_ref) for t in range(seq_len)
                ]
                im_nu = plot_nusselt_2d(nu_pred, nu_target)
                wandb.log({f"test/Plot-Nusselt-{batch}": im_nu})

                # Mean profile over height (area-avg over x)
                q_profile_pred = [
                    compute_profile_q(pred[idx, :, t], T_mean_ref)
                    for t in range(seq_len)
                ]
                q_profile_target = [
                    compute_profile_q(y[idx, :, t], T_mean_ref) for t in range(seq_len)
                ]
                im_q_profile = plot_mean_profile_2d(q_profile_pred, q_profile_target)
                wandb.log({f"test/Plot-MeanProfile-{batch}": im_q_profile})

                # RMS over space of q' per time step
                rms_qp_pred_ts = compute_qprime_rms_timeseries_2d(pred[idx], T_mean_ref)
                rms_qp_target_ts = compute_qprime_rms_timeseries_2d(y[idx], T_mean_ref)
                im_qp_rms_ts = plot_rms_timeseries_2d(
                    rms_qp_pred_ts,
                    rms_qp_target_ts,
                    title="RMS of q' over Space vs Time (2D)",
                )
                wandb.log({f"test/Plot-QPrime-RMS-Timeseries-{batch}": im_qp_rms_ts})

                # PDF over q' at selected heights (z at 1/4, 1/2, 3/4 of HEIGHT)
                H = pred[idx].shape[2]  # HEIGHT in [C,T,H,W]
                z_sel = [H // 4, H // 2, (3 * H) // 4]
                qp_pred_by_z = compute_qprime_flat_at_heights_2d(
                    pred[idx], T_mean_ref, z_sel
                )
                qp_target_by_z = compute_qprime_flat_at_heights_2d(
                    y[idx], T_mean_ref, z_sel
                )
                im_qp_panels = plot_pdf_qprime_panels_2d(
                    qp_pred_by_z, qp_target_by_z, H
                )
                wandb.log({f"test/Plot-QPrime-PDF-Panels-{batch}": im_qp_panels})

    # log metrics
    df = pd.DataFrame(metrics)
    im1 = plot_metric(df, "rmse")
    im2 = plot_metric(df, "nmse")
    wandb.log(
        {
            "test/Plot-RMSE": im1,
            "test/Plot-NMSE": im2,
            "test/Table-Metrics": wandb.Table(dataframe=df),
        }
    )


def compute_nmse(pred, target):
    eps = torch.finfo(pred.dtype).eps
    diff = pred - target
    # pred,target: [C,T,H,W]  -> sum over C,H,W; keep time axis (dim=1)
    nom = (diff * diff).sum(dim=(0, 2, 3))
    denom = (target * target).sum(dim=(0, 2, 3))
    denom = torch.clamp(denom, min=eps)
    return nom / denom


def compute_rmse(pred, target):
    rmse = torch.sqrt(mse_loss(pred, target, reduction="none"))
    # mean over C,H,W; keep time axis
    return rmse.mean(dim=(0, 2, 3))


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


def compute_mean_q(state, T_mean_ref):
    """state: [C,H,W]; channels assumed [T, u, w] with vertical velocity w at index 2."""
    T = state[0]
    w = state[2]
    T_bar = torch.as_tensor(T_mean_ref, dtype=T.dtype, device=T.device)
    theta = T - T_bar
    q = w * theta
    return q.mean().item()


def compute_profile_q(state, T_mean_ref):
    """Area-avg over x -> profile along height (H). state: [C,H,W]."""
    T = state[0]
    w = state[2]
    T_bar = torch.as_tensor(T_mean_ref, dtype=T.dtype, device=T.device)
    theta = T - T_bar
    q = w * theta
    q_profile = torch.mean(q, dim=1)  # mean over x (W) -> shape (H,)
    return q_profile.cpu().numpy()


def plot_nusselt_2d(nu_pred, nu_target):
    fig = plt.figure()
    sns.set_theme()
    steps = list(range(len(nu_pred)))
    plt.plot(steps, nu_pred, label="Prediction")
    plt.plot(steps, nu_target, label="Target")
    plt.title("Nusselt-like Flux vs Time (2D)")
    plt.xlabel("Time Step")
    plt.ylabel("⟨w (T - ⟨T⟩_{space,time})⟩")
    plt.legend()
    im = wandb.Image(fig, caption="Nusselt-like Flux 2D")
    plt.close(fig)
    return im


def plot_mean_profile_2d(q_profiles_pred, q_profiles_target):
    # stack over time and average along time axis
    prof_pred = np.mean(np.stack(q_profiles_pred, axis=0), axis=0)
    prof_target = np.mean(np.stack(q_profiles_target, axis=0), axis=0)

    fig = plt.figure()
    sns.set_theme()
    height = np.arange(len(prof_pred))
    plt.plot(prof_pred, height, label="Prediction")
    plt.plot(prof_target, height, label="Target")
    plt.title("Mean q Profile over Height (2D)")
    plt.xlabel("q = w (T - ⟨T⟩_{space,time})")
    plt.ylabel("Height index")
    plt.legend()
    plt.gca().invert_yaxis()
    im = wandb.Image(fig, caption="Mean q Profile 2D")
    plt.close(fig)
    return im


def compute_qprime_rms_timeseries_2d(state_seq, T_mean_ref):
    """Per-time RMS over space of q' in 2D. state_seq: [C,T,H,W]"""
    T_seq = state_seq[0]
    w_seq = state_seq[2]
    T_bar = torch.as_tensor(T_mean_ref, dtype=T_seq.dtype, device=T_seq.device)
    theta_seq = T_seq - T_bar
    q_seq = w_seq * theta_seq  # [T,H,W]
    # horizontal-time mean at each height: ⟨w θ(z)⟩_{x,t}
    q_bar_h = q_seq.mean(dim=(0, 2))  # [H]
    q_bar_b = q_bar_h[None, :, None]
    q_prime = q_seq - q_bar_b  # [T,H,W]
    rms_t = torch.sqrt(torch.mean(q_prime**2, dim=(1, 2))).cpu().numpy().tolist()
    return rms_t


def plot_rms_timeseries_2d(rms_pred, rms_target, title="RMS of q' vs Time (2D)"):
    fig = plt.figure()
    sns.set_theme()
    steps = list(range(len(rms_pred)))
    plt.plot(steps, rms_pred, label="Prediction")
    plt.plot(steps, rms_target, label="Target")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("RMS over space")
    plt.legend()
    im = wandb.Image(fig, caption=title)
    plt.close(fig)
    return im


def compute_qprime_flat_at_heights_2d(state_seq, T_mean_ref, z_indices):
    """Return {z_idx: 1D array} of q'(t,x,z_idx) flattened over time and x. state_seq: [C,T,H,W]."""
    T_seq = state_seq[0]
    w_seq = state_seq[2]
    T_bar = torch.as_tensor(T_mean_ref, dtype=T_seq.dtype, device=T_seq.device)
    theta_seq = T_seq - T_bar
    q_seq = w_seq * theta_seq  # [T,H,W]
    q_bar_h = q_seq.mean(dim=(0, 2))  # [H]
    q_bar_b = q_bar_h[None, :, None]
    q_prime = q_seq - q_bar_b  # [T,H,W]
    out = {}
    H = q_prime.shape[1]
    for z in z_indices:
        zc = int(max(0, min(H - 1, z)))
        slab = q_prime[:, zc, :]  # [T,W]
        out[zc] = slab.reshape(-1).detach().cpu().numpy()
    return out


def plot_pdf_qprime_panels_2d(
    qp_pred_by_z, qp_target_by_z, H, title_prefix="PDF of q' at heights (2D)"
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    sns.set_theme()
    zs = sorted(qp_pred_by_z.keys())[:3]
    for ax, z in zip(axes, zs):
        pred_flat = qp_pred_by_z[z]
        targ_flat = qp_target_by_z[z]
        p_lo = min(np.percentile(pred_flat, 0.5), np.percentile(targ_flat, 0.5))
        p_hi = max(np.percentile(pred_flat, 99.5), np.percentile(targ_flat, 99.5))
        xlim = (-max(abs(p_lo), abs(p_hi)), max(abs(p_lo), abs(p_hi)))
        bins = 50
        hist_p, edges = np.histogram(pred_flat, bins=bins, range=xlim, density=True)
        hist_t, _ = np.histogram(targ_flat, bins=bins, range=xlim, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        ax.semilogy(centers, hist_p + 1e-16, label="Prediction")
        ax.semilogy(centers, hist_t + 1e-16, label="Target")
        ax.set_xlabel("q'")
        z_rel = z / max(1, H - 1)
        ax.set_title(f"z = {z_rel:.2f}")
        ax.set_xlim(xlim)
    axes[0].set_ylabel("PDF(q')")
    axes[-1].legend(loc="best")
    fig.suptitle(title_prefix)
    im = wandb.Image(fig, caption=title_prefix)
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
        vmin, vmax = 1, 2
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
        return torch.device("cuda")  # first free NVIDIA GPU
    if torch.backends.mps.is_available():  # Apple-silicon
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
