from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms


@dataclass(frozen=True)
class MNISTDataConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = True


class MNISTDataModule:
    """
    Minimal datamodule abstraction for loading MNIST via torchvision.
    Keeps dataset construction and dataloader configuration in one place.
    """

    def __init__(self, cfg: MNISTDataConfig) -> None:
        self.cfg = cfg
        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None

    @property
    def data_dir(self) -> Path:
        return Path(self.cfg.data_dir)

    @property
    def train_dataset(self) -> Dataset:
        return self._require(self._train_ds, "setup() must be called before accessing train_dataset.")

    @property
    def val_dataset(self) -> Dataset:
        return self._require(self._val_ds, "setup() must be called before accessing val_dataset.")

    def prepare_data(self) -> None:
        # Download-only step (safe to call multiple times).
        # In distributed mode, call this on rank 0 only.
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self) -> None:
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self._train_ds = datasets.MNIST(root=self.data_dir, train=True, download=False, transform=tfm)
        self._val_ds = datasets.MNIST(root=self.data_dir, train=False, download=False, transform=tfm)

    def train_dataloader(self, sampler: Optional[Sampler] = None) -> DataLoader:
        # shuffle and sampler are mutually exclusive; sampler handles ordering in distributed mode.
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory and torch.cuda.is_available(),
        )

    @staticmethod
    def _require(value: Optional[object], msg: str) -> object:
        if value is None:
            raise RuntimeError(msg)
        return value
