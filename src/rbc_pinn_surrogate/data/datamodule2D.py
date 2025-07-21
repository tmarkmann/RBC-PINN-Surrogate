from typing import List

import lightning as L
from torch.utils.data import Dataset, DataLoader
from .dataset import RBCDataset2D


class RBCDatamodule2D(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        ra: int = 1e4,
        pressure: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        start: int = 0,
        end: int = 100,
        train_steps: int = 8,
        test_steps: int = 8,
        train_shift: int = 1,
        test_shift: int = 1,
        stride: int = 1,
        nr_episodes_train: int | None = None,
        nr_episodes_val: int | None = None,
        nr_episodes_test: int | None = None,
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
            self.datasets["val"] = RBCDataset2D(
                self.hparams.data_dir + f"/val/ra{int(self.hparams.ra)}.h5",
                start=self.hparams.start,
                end=self.hparams.end,
                nr_episodes=self.hparams.nr_episodes_val,
                input_steps=self.hparams.train_steps,
                target_steps=self.hparams.train_steps,
                shift=self.hparams.train_shift,
                stride=self.hparams.stride,
                pressure=self.hparams.pressure,
            )
            self.datasets["train"] = RBCDataset2D(
                self.hparams.data_dir + f"/train/ra{int(self.hparams.ra)}.h5",
                start=self.hparams.start,
                end=self.hparams.end,
                nr_episodes=self.hparams.nr_episodes_train,
                input_steps=self.hparams.train_steps,
                target_steps=self.hparams.train_steps,
                shift=self.hparams.train_shift,
                stride=self.hparams.stride,
                pressure=self.hparams.pressure,
            )
        # Assign test dataset for use in dataloaders
        elif stage == "test":
            self.datasets["test"] = RBCDataset2D(
                self.hparams.data_dir + f"/test/ra{int(self.hparams.ra)}.h5",
                start=self.hparams.start,
                end=self.hparams.end,
                nr_episodes=self.hparams.nr_episodes_test,
                input_steps=self.hparams.train_steps,
                target_steps=self.hparams.test_steps,
                shift=self.hparams.test_shift,
                stride=self.hparams.stride,
                pressure=self.hparams.pressure,
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
