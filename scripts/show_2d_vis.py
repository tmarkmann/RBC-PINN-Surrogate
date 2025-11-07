import h5py
import numpy as np
from tqdm import tqdm
from rbc_pinn_surrogate.utils.vis_2d import TemperatureVisualizer


def vis(split: str = "train", dataset: str = "zero", ra: int = 10000, episode: int = 0):
    # Load dataset episode
    path = f"data/2D-control/{split}/ra{ra}/{dataset}.h5"
    with h5py.File(path, "r") as file:
        states = file[f"states{episode}"]

        # data params
        steps = file.attrs.get("steps")
        H, W = states.shape[-2], states.shape[-1]

        # visualizer
        vis = TemperatureVisualizer(size=[H, W], vmin=1.0, vmax=2.0, display=True)

        # test
        s1 = states[0]
        s2 = states[40]

        print(f"diff of step 0 and 40: {np.abs(s2 - s1).mean()}")

        for step in tqdm(range(steps - 1)):
            state = states[step]

            T = state[0]  # (H, W)
            vis.update(T)


if __name__ == "__main__":
    RA = 10000
    NR = 0
    DATASET = "random"
    SPLIT = "train"
    print(f"Visualizing RA={RA}, dataset={DATASET}, episode={NR} from {SPLIT} split")

    vis(split=SPLIT, dataset=DATASET, ra=RA, episode=NR)
