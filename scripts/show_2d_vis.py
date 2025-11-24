import h5py
from tqdm import tqdm
from rbc_pinn_surrogate.utils.vis_2d import PredictionVisualizer
import numpy as np


def vis(split: str = "train", dataset: str = "zero", ra: int = 10000, episode: int = 0):
    # Load dataset episode
    path = f"data/2D-control/{split}/ra{ra}/{dataset}.h5"
    with h5py.File(path, "r") as file:
        states = file[f"states{episode}"]

        # data params
        steps = file.attrs.get("steps")
        H, W = states.shape[-2], states.shape[-1]

        # visualizer
        vis = PredictionVisualizer(size=[H, W], field="T", display=True, fps=10)
        # vis = TemperatureVisualizer(size=[H, W], vmin=1.0, vmax=2.0, display=True)

        for step in tqdm(range(steps - 1)):
            state = states[step]

            T = state[0]  # (H, W)
            rng = np.random.default_rng()
            sigma = 0.1 * float(T.max() - T.min() + 1e-8)  # ~2% of dynamic range
            D = (T + rng.normal(0.0, sigma, size=T.shape)).astype(T.dtype, copy=False)
            vis.update((T, D, step))


if __name__ == "__main__":
    RA = 10000
    NR = 0
    DATASET = "pd"
    SPLIT = "train"
    print(f"Visualizing RA={RA}, dataset={DATASET}, episode={NR} from {SPLIT} split")

    vis(split=SPLIT, dataset=DATASET, ra=RA, episode=NR)
