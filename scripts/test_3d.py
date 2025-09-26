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
from rbc_pinn_surrogate.model import FNO3DModule, LRAN3DModule
from rbc_pinn_surrogate.utils.vis3D import animation_3d


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_test")
def main(config: DictConfig):
    # device
    device = "cpu"  # best_device()

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
                # animation
                path = animation_3d(
                    gt=y[idx].cpu().numpy(),
                    pred=pred[idx].cpu().numpy(),
                    feature="T",
                    anim_dir=config.paths.output_dir + "/animations",
                    anim_name=f"test_{batch}_{idx}.mp4",
                )
                video = wandb.Video(path, format="mp4", caption=f"Test {batch}.{idx}")
                wandb.log({"test/video": video})

                # nusselt number plot (use single spatio-temporal mean ⟨T⟩ over target sequence)
                T_mean_ref = y[idx, 0].mean()  # scalar mean over (t, h, w, d)
                nu_pred = [
                    compute_mean_q(pred[idx, :, t], T_mean_ref) for t in range(seq_len)
                ]
                nu_target = [
                    compute_mean_q(y[idx, :, t], T_mean_ref) for t in range(seq_len)
                ]
                im_nu = plot_nusselt(nu_pred, nu_target)
                wandb.log({f"test/Plot-Nusselt-{batch}": wandb.Image(im_nu)})

                # mean profile of q (area-avg) over time
                q_profile_pred = [
                    compute_profile_q(pred[idx, :, t], T_mean_ref)
                    for t in range(seq_len)
                ]
                q_profile_target = [
                    compute_profile_q(y[idx, :, t], T_mean_ref) for t in range(seq_len)
                ]
                im_q_profile = plot_mean_profile(q_profile_pred, q_profile_target)
                wandb.log({f"test/Plot-ProfileQ-{batch}": im_q_profile})

                # RMS over space of q' per time step
                rms_qp_pred_ts = compute_qprime_rms_timeseries(pred[idx], T_mean_ref)
                rms_qp_target_ts = compute_qprime_rms_timeseries(y[idx], T_mean_ref)
                im_qp_rms_ts = plot_rms_timeseries(
                    rms_qp_pred_ts,
                    rms_qp_target_ts,
                    title="RMS of q' over Space vs Time",
                )
                wandb.log({f"test/Plot-ProfileQPrime-{batch}": im_qp_rms_ts})

                # PDF over q' field across space and time
                qp_pred_flat = compute_qprime_flat(pred[idx], T_mean_ref)
                qp_target_flat = compute_qprime_flat(y[idx], T_mean_ref)
                im_qp_pdf = plot_pdf_qprime(
                    qp_pred_flat,
                    qp_target_flat,
                    bins=201,
                    title="PDF of q' (space–time)",
                )
                wandb.log({f"test/Plot-QPrime-PDF-{batch}": im_qp_pdf})

                # PDF over q' at selected heights (z at 1/4, 1/2, 3/4 of HEIGHT)
                H = pred[idx].shape[2]  # HEIGHT dimension in [C,T,H,W,D]
                z_sel = [H // 4, H // 2, (3 * H) // 4]
                qp_pred_by_z = compute_qprime_flat_at_heights(
                    pred[idx], T_mean_ref, z_sel
                )
                qp_target_by_z = compute_qprime_flat_at_heights(
                    y[idx], T_mean_ref, z_sel
                )
                im_qp_panels = plot_pdf_qprime_panels(qp_pred_by_z, qp_target_by_z, H)
                wandb.log({f"test/Plot-QPrime-PDF-Panels-{batch}": im_qp_panels})

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


def compute_mean_q(state, T_mean_ref):
    T = state[0]
    uz = state[3]
    theta = T - T_mean_ref
    q = uz * theta
    return q.mean().item()


def compute_profile_q(state, T_mean_ref):
    T = state[0]
    uz = state[3]
    theta = T - T_mean_ref
    q = uz * theta
    # area-average over (x,y) to get a profile along height
    q_profile = torch.mean(q, dim=(1, 2))  # mean over horizontal (x, y)
    return q_profile.numpy()  # shape (D,)


def compute_qprime_rms_timeseries(state_seq, T_mean_ref):
    """Return per-time RMS of q' over the whole spatial domain.
    state_seq: [C, T, H, W, D]
    RMS_t = sqrt(mean_{x,y,z}(q'(t)^2))
    """
    T_seq = state_seq[0]
    uz_seq = state_seq[3]
    T_bar = torch.as_tensor(T_mean_ref, dtype=T_seq.dtype, device=T_seq.device)

    theta_seq = T_seq - T_bar  # [T,H,W,D]
    q_seq = uz_seq * theta_seq  # [T,H,W,D]
    # horizontal-time mean per height
    q_bar_zy = q_seq.mean(dim=(0, 2, 3))  # [H]
    q_bar_broadcast = q_bar_zy[None, :, None, None]
    q_prime = q_seq - q_bar_broadcast  # [T,H,W,D]

    # per-time RMS over whole domain
    rms_t = torch.sqrt(torch.mean(q_prime**2, dim=(1, 2, 3))).cpu().numpy().tolist()
    return rms_t


def compute_qprime_flat(state_seq, T_mean_ref):
    """Return q'(t,x,y,z) flattened over space and time as a 1D numpy array.
    state_seq: [C, T, H, W, D]
    q' = u_z (T - ⟨T⟩) − ⟨u_z (T - ⟨T⟩)⟩_{x,y,t}(z)
    """
    T_seq = state_seq[0]
    uz_seq = state_seq[3]
    T_bar = torch.as_tensor(T_mean_ref, dtype=T_seq.dtype, device=T_seq.device)

    theta_seq = T_seq - T_bar  # [T,H,W,D]
    q_seq = uz_seq * theta_seq  # [T,H,W,D]
    # horizontal-time mean per height z
    q_bar_zy = q_seq.mean(dim=(0, 2, 3))  # [H]
    q_bar_broadcast = q_bar_zy[None, :, None, None]

    q_prime = q_seq - q_bar_broadcast  # [T,H,W,D]
    return q_prime.reshape(-1).detach().cpu().numpy()


def compute_qprime_flat_at_heights(state_seq, T_mean_ref, z_indices):
    """Return dict {z_idx: 1D numpy array} with q'(t,x,y,z_idx) flattened over space and time.
    state_seq: [C, T, H, W, D]; here HEIGHT is the second spatial dim (index 1 in [T,H,W,D]).
    z_indices: iterable of integer indices along HEIGHT (H).
    """
    T_seq = state_seq[0]  # [T,H,W,D]
    uz_seq = state_seq[3]  # [T,H,W,D]
    T_bar = torch.as_tensor(T_mean_ref, dtype=T_seq.dtype, device=T_seq.device)

    theta_seq = T_seq - T_bar
    q_seq = uz_seq * theta_seq  # [T,H,W,D]

    # horizontal-time mean at each height: ⟨u_z θ(z)⟩_{x,y,t}
    q_bar_h = q_seq.mean(dim=(0, 2, 3))  # [H]
    q_bar_b = q_bar_h[None, :, None, None]  # [1,H,1,1]
    q_prime = q_seq - q_bar_b  # [T,H,W,D]

    out = {}
    H = q_prime.shape[1]
    for z in z_indices:
        z_clamped = int(max(0, min(H - 1, z)))
        slab = q_prime[:, z_clamped, :, :]  # [T,W,D]
        out[z_clamped] = slab.reshape(-1).detach().cpu().numpy()
    return out


def plot_rms_timeseries(rms_pred, rms_target, title="RMS of q' vs Time"):
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


def plot_nusselt(nu_pred, nu_target):
    fig = plt.figure()
    sns.set_theme()
    steps = list(range(len(nu_pred)))
    plt.plot(steps, nu_pred, label="Prediction")
    plt.plot(steps, nu_target, label="Target")
    plt.title("Nusselt Number vs Time")
    plt.xlabel("Time Step")
    plt.ylabel("⟨u_z (T - ⟨T⟩_{space,time})⟩")
    plt.legend()
    im = wandb.Image(fig, caption="Nusselt Number")
    plt.close(fig)
    return im


def plot_mean_profile(q_profiles_pred, q_profiles_target):
    """q_profiles_*: list of (D,) numpy arrays for each time step"""
    # stack over time and average along time axis
    prof_pred = np.mean(np.stack(q_profiles_pred, axis=0), axis=0)
    prof_target = np.mean(np.stack(q_profiles_target, axis=0), axis=0)

    fig = plt.figure()
    sns.set_theme()
    height = np.arange(len(prof_pred))
    plt.plot(prof_pred, height, label="Prediction")
    plt.plot(prof_target, height, label="Target")
    plt.title("Mean q Profile over Height")
    plt.xlabel("q = u_z (T - ⟨T⟩_{space,time})")
    plt.ylabel("Height index")
    plt.legend()
    plt.gca().invert_yaxis()
    im = wandb.Image(fig, caption="Mean q Profile")
    plt.close(fig)
    return im


def plot_pdf_qprime(
    qp_pred_flat, qp_target_flat, bins=201, title="PDF of q' (space–time)"
):
    fig = plt.figure()
    sns.set_theme()
    # choose symmetric range based on robust percentiles
    p_lo = min(np.percentile(qp_pred_flat, 0.5), np.percentile(qp_target_flat, 0.5))
    p_hi = max(np.percentile(qp_pred_flat, 99.5), np.percentile(qp_target_flat, 99.5))
    xlim = (-max(abs(p_lo), abs(p_hi)), max(abs(p_lo), abs(p_hi)))

    hist_p, edges = np.histogram(qp_pred_flat, bins=bins, range=xlim, density=True)
    hist_t, _ = np.histogram(qp_target_flat, bins=bins, range=xlim, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])

    plt.semilogy(centers, hist_p + 1e-16, label="Prediction")
    plt.semilogy(centers, hist_t + 1e-16, label="Target")
    plt.title(title)
    plt.xlabel("q'")
    plt.ylabel("PDF(q')")
    plt.legend()
    plt.xlim(xlim)
    im = wandb.Image(fig, caption=title)
    plt.close(fig)
    return im


def plot_pdf_qprime_panels(
    qp_pred_by_z, qp_target_by_z, H, title_prefix="PDF of q' at heights"
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    sns.set_theme()
    zs = sorted(qp_pred_by_z.keys())[:3]
    for ax, z in zip(axes, zs):
        pred_flat = qp_pred_by_z[z]
        targ_flat = qp_target_by_z[z]
        # robust symmetric limits
        p_lo = min(np.percentile(pred_flat, 0.5), np.percentile(targ_flat, 0.5))
        p_hi = max(np.percentile(pred_flat, 99.5), np.percentile(targ_flat, 99.5))
        xlim = (-max(abs(p_lo), abs(p_hi)), max(abs(p_lo), abs(p_hi)))
        bins = 201
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


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # first free NVIDIA GPU
    if torch.backends.mps.is_available():  # Apple-silicon
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
