import h5py
from matplotlib import pyplot as plt
from tqdm import tqdm
from rbc_pinn_surrogate.utils.vis_2d import RBCLiveVisualizer
import argparse


def vis(split: str = "train", dataset: str = "zero", ra: int = 10000, episode: int = 0):
    # Load dataset episode
    path = f"data/datasets/2D-control/{split}/ra{ra}/{dataset}.h5"
    with h5py.File(path, "r") as file:
        states = file[f"s-{episode}"]  # expected shape: (steps, C, H, W)
        actions = file[f"a-{episode}"]  # expected shape: (steps, 12)

        steps_attr = file.attrs.get("steps", None)
        steps = int(steps_attr) if steps_attr is not None else len(states)

        # Infer spatial size
        H, W = states.shape[-2], states.shape[-1]
        bins = actions.shape[-1]
        vis = RBCLiveVisualizer(
            size=[H, W], bins=bins, vmin=1, vmax=2, action_vmin=-1, action_vmax=1
        )

        for step in tqdm(
            range(steps - 1), total=int(steps // 4), desc=f"Episode {episode}"
        ):
            state = states[step * 4]  # (C, H, W)
            action = actions[step * 4]  # (bins,)

            T = state[0]  # (H, W)
            vis.update(T, action)
            plt.pause(0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RBC dataset episodes")
    parser.add_argument("--ra", type=int, default=10000, help="Rayleigh number")
    parser.add_argument(
        "--dataset", type=str, default="random", help="Dataset type/name"
    )
    parser.add_argument(
        "--episode", type=int, default=0, help="Episode index to visualize"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "val"],
        help="Dataset split",
    )
    args = parser.parse_args()

    vis(split=args.split, dataset=args.dataset, ra=args.ra, episode=args.episode)
