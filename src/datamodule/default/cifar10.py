from typing import Optional

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl

from src.utils.data_utils import TransformWrapper


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, **config):
        super().__init__()

        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        
        self.persistent_workers = True if self.num_workers > 0 else False

        # Standard CIFAR-10 stats
        mean = (0.4914, 0.4882, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        self.train_xform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.test_xform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_train = datasets.CIFAR10(self.data_dir, train=True)
            train_len = int(0.7 * len(full_train))
            val_len = len(full_train) - train_len
            train, val = random_split(full_train, [train_len, val_len])
            self.train_dataset = TransformWrapper(train, self.train_xform)
            self.val_dataset = TransformWrapper(val, self.test_xform)

        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(
                self.data_dir, train=False, transform=self.test_xform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )