from typing import List

import lightning as L
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .dataset_control import RBCDataset2DControl


class RBCDatamodule2DControl(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        ra: int = 1e4,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        horizon: int = 6,
        shift: int = 1,
        nr_episodes_train: int | None = None,
        nr_episodes_val: int | None = None,
        nr_episodes_test: int | None = None,
    ) -> None:
        super().__init__()
        # DataModule parameters
        self.save_hyperparameters()
        # Dataset
        self.datasets: dict[str, Dataset] = {}
        self.path: dict[str, List[str]] = {}
        # Transform
        # TODO

    def get_dataset(self, stage: str, type: str) -> str:
        path = f"{self.hparams.data_dir}/{stage}/ra{int(self.hparams.ra)}/{type}.h5"

        if stage == "train":
            nr = self.hparams.nr_episodes_train
        elif stage == "val":
            nr = self.hparams.nr_episodes_val
        elif stage == "test":
            nr = self.hparams.nr_episodes_test
        else:
            raise ValueError(f"Stage not implemented: {stage}")

        return RBCDataset2DControl(
            path,
            nr_episodes=nr,
            target_steps=self.hparams.horizon,
            shift=self.hparams.shift,
        )

    def setup(self, stage: str):
        types = ["ppo", "random", "zero"]
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.datasets["val"] = ConcatDataset(
                [self.get_dataset("val", t) for t in types]
            )
            self.datasets["train"] = ConcatDataset(
                [self.get_dataset("train", t) for t in types]
            )
        # Assign test dataset for use in dataloaders
        elif stage == "test":
            self.datasets["test"] = ConcatDataset(
                [self.get_dataset("test", t) for t in types]
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
