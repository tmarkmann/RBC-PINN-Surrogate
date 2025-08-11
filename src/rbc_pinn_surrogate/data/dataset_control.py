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
        target_steps: int = 6,
        shift: int = 1,
    ):
        # parameters
        self.path = path
        self.target_steps = target_steps
        self.shift = shift

        # retrieve dataset parameters
        self._file = None
        with h5py.File(path, "r") as file:
            self._set_data_properties(file)
            if nr_episodes is not None:
                self.nr_episodes = nr_episodes
            else:
                nr_episodes = self.nr_episodes

            # check validity of parameters
            self._check_validity(nr_episodes)

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

        # number of sequence pairs per episode
        self.nr_pairs = (self.steps - self.target_steps) // self.shift

    def _check_validity(self, nr_episodes):
        assert self.nr_episodes >= nr_episodes, (
            f"Number of episodes {nr_episodes} exceeds available episodes {self.nr_episodes}"
        )

    def __len__(self) -> int:
        return self.nr_pairs * self.nr_episodes

    def __del__(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getitem__(self, idx: int) -> Tensor:
        f = self._require_file()
        # calculate episode and pair index
        episode_idx = idx // self.nr_pairs
        pair_idx = idx % self.nr_pairs

        # get the episode dataset
        ds = f[f"s-{episode_idx}"]
        da = f[f"a-{episode_idx}"]

        # calculate start and end indices for input and target sequences
        start_idx = pair_idx * self.shift
        end_idx = start_idx + 1 + self.target_steps

        # extract input and target sequences
        x = ds[start_idx]
        a = da[start_idx : end_idx - 1]
        y = ds[start_idx + 1 : end_idx]

        return (x, a), y
