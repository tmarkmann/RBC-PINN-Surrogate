from pathlib import Path
import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RBCDataset(Dataset[Tensor]):
    def __init__(
        self,
        path: Path,
        start_time: int = 0,
        end_time: int = 100,
        nr_episodes: int = 50,
        input_length: int = 8,
        target_length: int = 8,
        shift_time: int = 1,
        stride_time: int = 1,
    ):
        # parameters
        self.path = path
        self.start_time = start_time
        self.end_time = end_time
        self.nr_episodes = nr_episodes
        self.input_length = input_length
        self.target_length = target_length
        self.shift_time = shift_time
        self.stride_time = stride_time

        # retrieve dataset parameters

        with h5py.File(path, "r") as file:
            self.set_data_properties(file)

            # check validity of parameters
            self.check_validity()

            # get episode
            self.episodes = {}
            for episode in range(self.nr_episodes):
                ep = torch.tensor(
                    file[f"states{episode}"][self.start_step : self.end_step]
                )
                # Swap the channel dimension to the first dimension
                ep = torch.permute(ep, (1, 0, 2, 3))
                # store episode data
                self.episodes[episode] = ep

    def set_data_properties(self, file):
        # sets general data dimensions properties that are true for all episodes
        self.nr_channels = file["states0"].shape[1]
        self.height = file["states0"].shape[2]
        self.width = file["states0"].shape[3]

        # set other properties
        parameters = dict(file.attrs.items())
        self.episodes = int(parameters["episodes"])
        self.episode_steps = int(parameters["steps"])
        self.shape = tuple(parameters["shape"])
        self.dt = float(parameters["dt"])
        self.episode_length = float(parameters["timesteps"])
        self.segments = int(parameters["segments"])
        self.limit = float(parameters["limit"])
        self.base_seed = int(parameters["base_seed"])

        # discrete steps between model inputs
        self.shift_steps = int(self.shift_time / self.dt)
        # discrete steps between the snapshots in the input and output sequence.
        self.stride_steps = int(self.stride_time / self.dt)

        self.start_step = int(self.start_time / self.dt)
        self.end_step = int(self.end_time / self.dt)
        self.steps = self.end_step - self.start_step

        # number of sequence pairs per episode
        self.nr_pairs = (
            self.steps - self.input_length - self.target_length + 1
        ) // self.shift_steps

    def check_validity(self):
        assert self.end_time <= self.episode_length, (
            f"End time {self.end_time} exceeds episode length {self.episode_length}"
        )
        assert self.start_time < self.end_time, (
            f"Start time {self.start_time} must be less than end time {self.end_time}"
        )
        assert self.nr_episodes <= self.episodes, (
            f"Number of episodes {self.nr_episodes} exceeds available episodes {self.episodes}"
        )

    def __len__(self) -> int:
        return self.nr_pairs * self.nr_episodes

    def __getitem__(self, idx: int) -> Tensor:
        # calculate episode and pair index
        episode_idx = idx // self.nr_pairs
        pair_idx = idx % self.nr_pairs

        # get the episode data
        episode_data = self.episodes[episode_idx]

        # TODO current dataset has flipped vertical axis. future datasets should not have this.
        episode_data = torch.flip(episode_data, dims=[2])

        # calculate start and end indices for input and target sequences
        start_idx = pair_idx * self.shift_steps
        end_idx_input = start_idx + self.input_length
        end_idx_target = end_idx_input + self.target_length

        # extract input and target sequences
        x = episode_data[:, start_idx : end_idx_input : self.stride_steps]
        y = episode_data[:, end_idx_input : end_idx_target : self.stride_steps]

        return x, y
