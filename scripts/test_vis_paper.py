import h5py
from tqdm import tqdm
import torch
from rbc_pinn_surrogate.utils.vis3D import plot_paper

def main():
    with h5py.File("data/thorben.h5", "r") as f:
        ds_pred = f["forecasts"]
        ds_target = f["ground_truth"]
        samples = ds_pred.shape[0]

        for idx in tqdm(range(samples), desc="Testing"):
            # prepare data
            pred = torch.as_tensor(ds_pred[idx], dtype=torch.float32)
            target = torch.as_tensor(ds_target[idx], dtype=torch.float32)
            # reorder to # [1,C,T,H,W,D]
            pred = pred.permute(4, 0, 3, 1, 2)
            target = target.permute(4, 0, 3, 1, 2)

            plot_paper(target.numpy(), pred.numpy(), "logs/vis_paper")
            
            break


if __name__ == "__main__":
    main()
