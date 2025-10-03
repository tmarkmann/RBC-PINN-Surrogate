import hydra
from matplotlib import pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig
import pandas as pd
import wandb
import torch
import h5py
import seaborn as sns
from rbc_pinn_surrogate.utils.vis3D import animation_3d
import rbc_pinn_surrogate.callbacks.metrics_3D as metrics


@hydra.main(version_base="1.3", config_path="../configs", config_name="3d_test")
def main(config: DictConfig):
    # load data
    with h5py.File("data/thorben.h5", "r") as f:
        ds_pred = f["forecasts"]
        ds_target = f["ground_truth"]

        print("Dataset shape:", ds_pred.shape)  # (B, T, W, D, H, C)
        samples = ds_pred.shape[0]

        # wandb run
        wandb.init(
            project="RBC-3D-LSTM",
            dir=config.paths.output_dir,
            tags=["test"],
        )

        # loop
        list_metrics = []
        list_nusselt = []
        list_profile = []
        for idx in tqdm(range(samples), desc="Testing"):
            # prepare data
            pred = torch.as_tensor(ds_pred[idx], dtype=torch.float32)
            target = torch.as_tensor(ds_target[idx], dtype=torch.float32)
            # reorder to # [1,C,T,H,W,D]
            pred = pred.permute(4, 0, 3, 1, 2).unsqueeze(0)
            target = target.permute(4, 0, 3, 1, 2).unsqueeze(0)

            # 1) Sequence Metrics NRSSE and RMSE
            seq_len = pred.shape[2]
            for t in range(seq_len):
                # metrics per sample and time step
                rmse = metrics.rmse(pred[:, :, t], target[:, :, t])
                nrsse = metrics.nrsse(pred[:, :, t], target[:, :, t])

                list_metrics.append(
                    {
                        "batch_idx": idx,
                        "step": t,
                        "rmse": rmse.item(),
                        "nrsse": nrsse.item(),
                    }
                )

            # 2) Visualize samples from first batch element (only every 10th)
            if idx % 10 == 0:
                path = animation_3d(
                    gt=target[0].numpy(),
                    pred=pred[0].numpy(),
                    feature="T",
                    anim_dir=config.paths.output_dir + "/animations",
                    anim_name=f"test_{idx}.mp4",
                )
                video = wandb.Video(path, format="mp4", caption=f"Batch {idx}")
                wandb.log({"test/video": video})

            # 3) Nusselt Number: mean q
            T_mean_ref = target[0, 0].mean()
            for t in range(seq_len):
                nu_pred = metrics.compute_q(pred[0, :, t], T_mean_ref)
                nu_target = metrics.compute_q(target[0, :, t], T_mean_ref)
                list_nusselt.append(
                    {
                        "idx": idx,
                        "step": t,
                        "nu_pred": nu_pred.item(),
                        "nu_target": nu_target.item(),
                    }
                )

            # 4) Profile of mean q and q' (area-avg over time)
            for t in range(seq_len):
                # q profile
                q_profile_pred = metrics.compute_q(
                    pred[0, :, t], T_mean_ref, profile=True
                )
                q_profile_target = metrics.compute_q(
                    target[0, :, t], T_mean_ref, profile=True
                )

                # q' profile
                # rms_qp_pred_ts = compute_qprime_rms_timeseries(pred[0], T_mean_ref)
                # rms_qp_target_ts = compute_qprime_rms_timeseries(target[0], T_mean_ref)

                for z, (p, q) in enumerate(zip(q_profile_pred, q_profile_target)):
                    list_profile.append(
                        {
                            "idx": idx,
                            "step": t,
                            "height": z,
                            "q_pred": p,
                            "q_target": q,
                        }
                    )

            # 5) PDF over q' at selected heights
            # PDF over q' at selected heights (z at 1/4, 1/2, 3/4 of HEIGHT)
            # H = pred[0].shape[2]  # HEIGHT dimension in [C,T,H,W,D]
            # z_sel = [H // 4, H // 2, (3 * H) // 4]
            # qp_pred_by_z = compute_qprime_flat_at_heights(pred[0], T_mean_ref, z_sel)
            # qp_target_by_z = compute_qprime_flat_at_heights(y[0], T_mean_ref, z_sel)
            # im_qp_panels = plot_pdf_qprime_panels(qp_pred_by_z, qp_target_by_z, H)
            # wandb.log({f"test/Plot-QPrime-PDF-Panels-{batch}": im_qp_panels})

        # log metrics
        df_metrics = pd.DataFrame(list_metrics)
        df_nusselt = pd.DataFrame(list_nusselt)
        df_profile = pd.DataFrame(list_profile)

        # overall metrics
        rmse = df_metrics["rmse"].mean()
        nrsse = df_metrics["nrsse"].mean()

        # plots
        im_rmse = plot_metric(df_metrics, "rmse")
        im_nrsse = plot_metric(df_metrics, "nrsse")

        wandb.log(
            {
                "test/RMSE": rmse,
                "test/NRSSE": nrsse,
                "test/Table-Metrics": wandb.Table(dataframe=df_metrics),
                "test/Table-Nusselt": wandb.Table(dataframe=df_nusselt),
                "test/Table-Profile": wandb.Table(dataframe=df_profile),
                "test/Plot-RMSE": im_rmse,
                "test/Plot-NRSSE": im_nrsse,
            }
        )


def plot_metric(df: pd.DataFrame, metric: str):
    fig = plt.figure()
    sns.set_theme()
    ax = sns.lineplot(data=df, x="step", y=metric)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_xlabel("Time Step")
    ax.set_ylim(bottom=0, top=0.8)
    # save as image
    im = wandb.Image(fig, caption=metric)
    plt.close(fig)
    return im


if __name__ == "__main__":
    main()
