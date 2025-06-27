from typing import List

import lightning as L
from torch.utils.data import Dataset, DataLoader
from .dataset import RBCDataset


class RBCDatamodule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        ra: int = 1e4,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        start_time: int = 0,
        end_time: int = 100,
        train_length: int = 8,
        test_length: int = 8,
        shift_time: int = 1,
        stride_time: int = 1,
        nr_episodes_train: int = 15,
        nr_episodes_val: int = 5,
        nr_episodes_test: int = 5,
    ) -> None:
        super().__init__()
        # DataModule parameters
        self.save_hyperparameters()
        # Dataset
        self.datasets: dict[str, Dataset] = {}
        self.paths: dict[str, List[str]] = {}
        # Transform
        # TODO

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.datasets["val"] = RBCDataset(
                self.hparams.data_dir + f"/val/ra{int(self.hparams.ra)}.h5",
                start_time=self.hparams.start_time,
                end_time=self.hparams.end_time,
                nr_episodes=self.hparams.nr_episodes_val,
                length=self.hparams.train_length,
                shift_time=self.hparams.shift_time,
                stride_time=self.hparams.stride_time,
            )
            self.datasets["train"] = RBCDataset(
                self.hparams.data_dir + f"/train/ra{int(self.hparams.ra)}.h5",
                start_time=self.hparams.start_time,
                end_time=self.hparams.end_time,
                nr_episodes=self.hparams.nr_episodes_train,
                length=self.hparams.train_length,
                shift_time=self.hparams.shift_time,
                stride_time=self.hparams.stride_time,
            )
        # Assign test dataset for use in dataloaders
        elif stage == "test":
            self.datasets["test"] = RBCDataset(
                self.hparams.data_dir + f"/test/ra{int(self.hparams.ra)}.h5",
                start_time=self.hparams.start_time,
                end_time=self.hparams.end_time,
                nr_episodes=self.hparams.nr_episodes_test,
                length=self.hparams.test_length,
                shift_time=self.hparams.shift_time,
                stride_time=self.hparams.stride_time,
            )
        else:
            raise ValueError(f"Stage not implemented: {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )
