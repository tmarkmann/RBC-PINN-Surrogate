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
        with h5py.File(path, "r") as file:
            self._set_data_properties(file)
            if nr_episodes is not None:
                self.nr_episodes = nr_episodes
            else:
                nr_episodes = self.nr_episodes

            # check validity of parameters
            self._check_validity(nr_episodes)

            # get episode
            self.episodes = {}
            for episode in range(self.nr_episodes):
                ep = torch.tensor(file[f"states{episode}"][self.start : self.end])
                # Swap the channel dimension to the first dimension
                ep = self._permute(ep)
                # store episode data
                self.episodes[episode] = ep

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

    def _permute(self, ep: torch.Tensor) -> torch.Tensor:
        # (1, 0, 2, 3, 4, …) works for both 2‑D and 3‑D inputs
        order = (1, 0, *range(2, ep.ndim))
        return ep.permute(order)

    def __len__(self) -> int:
        return self.nr_pairs * self.nr_episodes

    def __getitem__(self, idx: int) -> Tensor:
        # calculate episode and pair index
        episode_idx = idx // self.nr_pairs
        pair_idx = idx % self.nr_pairs

        # get the episode data
        episode_data = self.episodes[episode_idx]

        # only include pressure channel if specified
        if not self.pressure:
            episode_data = episode_data[:3, :, :]

        # calculate start and end indices for input and target sequences
        start_idx = pair_idx * self.shift
        end_idx_input = start_idx + self.input_steps
        end_idx_target = end_idx_input + self.target_steps
        # extract input and target sequences
        x = episode_data[:, start_idx : end_idx_input : self.stride]
        y = episode_data[:, end_idx_input : end_idx_target : self.stride]

        return x, y


class RBCDataset2D(RBCDataset):
    def _set_data_properties(self, file):
        super()._set_data_properties(file)
        self.height, self.width = file["states0"].shape[2:4]


class RBCDataset3D(RBCDataset):
    def _set_data_properties(self, file):
        super()._set_data_properties(file)
        self.depth, self.height, self.width = file["states0"].shape[2:5]
