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

        # data normalization (hardcoded channel-wise mean/std)
        # shaped for broadcasting over sequences [T, C, H, W]
        self.mean = torch.tensor([1.5, 0.0, 0.0], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.25, 0.35, 0.35], dtype=torch.float32).view(
            1, 3, 1, 1
        )

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
        self.nr_channels = file["s-0"].shape[1]
        self.height, self.width = file["s-0"].shape[2:4]

        # set other properties
        parameters = dict(file.attrs.items())
        self.nr_episodes = int(parameters["episodes"])
        self.steps = int(parameters["steps"])
        self.shape = tuple(parameters["shape"])
        self.dt = float(parameters["dt"])
        self.episode_length = float(parameters["timesteps"])
        self.segments = int(parameters["segments"])
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

        ds = f[f"s-{episode_idx}"]
        da = f[f"a-{episode_idx}"]

        start_idx = pair_idx * self.shift
        end_idx = start_idx + self.horizon

        x_np = ds[start_idx:end_idx]  # [H, C, H, W]
        a_np = da[start_idx : end_idx - 1]  # [H-1, A]

        x = torch.from_numpy(x_np).to(dtype=torch.float32)
        a = torch.from_numpy(a_np).to(dtype=torch.float32)

        # Vectorized channel-wise normalization over the sequence
        # x: [T, C, H, W]; mean/std: [1, C, 1, 1]
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        return x, a
