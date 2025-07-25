from pathlib import Path
import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RBCDataset(Dataset[Tensor]):
    def __init__(
        self,
        path: Path,
        start: int = 0,
        end: int = 100,
        nr_episodes: int = None,
        input_steps: int = 8,
        target_steps: int = 8,
        shift: int = 1,
        stride: int = 1,
        pressure: bool = False,
    ):
        # parameters
        self.path = path
        self.start = start
        self.end = end
        self.input_steps = input_steps
        self.target_steps = target_steps
        self.shift = shift
        self.stride = stride
        self.pressure = pressure

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
        self.nr_channels = file["states0"].shape[1]

        # set other properties
        parameters = dict(file.attrs.items())
        self.nr_episodes = int(parameters["episodes"])
        self.episode_steps = int(parameters["steps"])
        self.shape = tuple(parameters["shape"])
        self.dt = float(parameters["dt"])
        self.episode_length = float(parameters["timesteps"])
        self.segments = int(parameters["segments"])
        self.limit = float(parameters["limit"])
        self.base_seed = int(parameters["base_seed"])

        # number of steps in dataset
        self.steps = self.end - self.start

        # number of sequence pairs per episode
        self.nr_pairs = (
            self.steps - self.input_steps - self.target_steps + 1
        ) // self.shift

    def _check_validity(self, nr_episodes):
        assert (self.end * self.dt) <= self.episode_length, (
            f"End steps {self.end} (={self.end / self.dt}[time]) exceeds episode length [time] {self.episode_length}"
        )
        assert self.start < self.end, (
            f"Start time {self.start} must be less than end time {self.end}"
        )
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
        ds = f[f"states{episode_idx}"]

        # calculate start and end indices for input and target sequences
        start_idx = self.start + pair_idx * self.shift
        mid_idx = start_idx + self.input_steps
        end_idx = mid_idx + self.target_steps

        # extract input and target sequences
        x = ds[start_idx : mid_idx : self.stride]
        y = ds[mid_idx : end_idx : self.stride]
        if not self.pressure:
            x = x[:, :-2]
            y = y[:, :-2]

        # change channel and time dimension for neuraloperator package
        x = torch.from_numpy(x).permute(1, 0, *range(2, x.ndim))
        y = torch.from_numpy(y).permute(1, 0, *range(2, y.ndim))
        return x, y


class RBCDataset2D(RBCDataset):
    def _set_data_properties(self, file):
        super()._set_data_properties(file)
        self.height, self.width = file["states0"].shape[2:4]


class RBCDataset3D(RBCDataset):
    def _set_data_properties(self, file):
        super()._set_data_properties(file)
        self.depth, self.height, self.width = file["states0"].shape[2:5]
