from pathlib import Path
import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RBCDataset2DControl(Dataset[Tensor]):
    def __init__(
        self,
        path: Path,
        nr_episodes: int = None,
        horizon: int = 6,
        shift: int = 1,
        normalize: bool = True,
        means: list[float] = [1.5, 0.0, 0.0, 0.0],
        stds: list[float] = [0.25, 0.35, 0.35, 0.35],
    ):
        # parameters
        self.path = path
        self.horizon = horizon
        self.shift = shift

        # retrieve dataset parameters
        self._file = None
        with h5py.File(path, "r") as file:
            self._set_data_properties(file)
            # number of episodes available in the file
            self.available_episodes = self.nr_episodes
            # how many episodes this dataset will expose
            if nr_episodes is None:
                self.used_episodes = self.available_episodes
            else:
                self._check_validity(nr_episodes)
                self.used_episodes = nr_episodes

        # normalization parameters (z-score standardization)
        self.normalize = normalize
        self.means = torch.as_tensor(means[: self.nr_channels]).view(
            self.nr_channels, 1, 1, 1
        )
        self.stds = torch.as_tensor(stds[: self.nr_channels]).view(
            self.nr_channels, 1, 1, 1
        )

        # permutation: [T, C, H, W] -> [C, T, H, W]
        self.permute = (1, 0, 2, 3)

        # basic parameter sanity checks
        assert self.shift >= 1, "shift must be >= 1"
        assert self.horizon >= 2, "horizon must be >= 2"
        assert self.horizon <= self.steps, "horizon must be <= steps per episode"

    def _require_file(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def _set_data_properties(self, file):
        # sets general data dimensions properties that are true for all episodes
        self.nr_channels = file["states0"].shape[1]
        self.height, self.width = file["states0"].shape[2:4]

        # set other properties
        parameters = dict(file.attrs.items())
        self.nr_episodes = int(parameters["episodes"])
        self.steps = int(parameters["steps"])
        self.shape = tuple(parameters["shape"])
        self.dt = float(parameters["dt"])
        self.episode_length = float(parameters["timesteps"])
        self.limit = float(parameters["limit"])
        self.base_seed = int(parameters["base_seed"])

        # Number of sliding windows of length `horizon` with stride `shift` within `steps`
        # start indices: 0, shift, ..., last_start where last_start = steps - horizon
        # hence count = floor((steps - horizon)/shift) + 1 when steps >= horizon, else 0
        self.nr_pairs = (
            0
            if self.steps < self.horizon
            else ((self.steps - self.horizon) // self.shift) + 1
        )

    def _check_validity(self, requested_episodes: int):
        assert requested_episodes >= 1, "requested nr_episodes must be >= 1"
        assert self.nr_episodes >= requested_episodes, (
            f"Requested {requested_episodes} episodes, but only {self.nr_episodes} available in file."
        )

    def __len__(self) -> int:
        return self.nr_pairs * self.used_episodes

    def __del__(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def normalize_batch(self, x: Tensor) -> Tensor:
        """Apply per-channel z-score normalization if enabled."""
        if not self.normalize:
            return x
        return (x - self.means.to(x.device)) / self.stds.to(x.device)

    def denormalize_batch(self, x: Tensor) -> Tensor:
        """Invert per-channel z-score normalization if enabled."""
        if not self.normalize:
            return x
        return x * self.stds.to(x.device) + self.means.to(x.device)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        f = self._require_file()
        # calculate episode and pair index
        episode_idx = idx // self.nr_pairs
        pair_idx = idx % self.nr_pairs
        # clamp to the used episodes range
        if episode_idx >= self.used_episodes:
            raise IndexError(
                f"Episode index {episode_idx} out of range (used {self.used_episodes})"
            )

        ds = f[f"states{episode_idx}"]
        da = f[f"actions{episode_idx}"]

        start_idx = pair_idx * self.shift
        end_idx = start_idx + self.horizon

        x_np = ds[start_idx:end_idx]  # [H, C, H, W]
        a_np = da[start_idx : end_idx - 1]  # [H-1, A]

        x = torch.from_numpy(x_np).permute(self.permute)
        a = torch.from_numpy(a_np).to(dtype=torch.float32)

        # normalize per channel
        x = self.normalize_batch(x)

        return x, a
