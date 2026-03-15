# Source: https://arxiv.org/pdf/2002.05709
from typing import Optional

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl

from src.utils.data_utils import TwoViewDataset

class SimCLRCIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, **config):
        super().__init__()

        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        
        self.persistent_workers = True if self.num_workers > 0 else False

        # Standard CIFAR-10 stats
        mean = (0.4914, 0.4882, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        s = 0.5          #
        image_size = 32  #
        self.train_xform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.08, 1.0),
                    ratio=(3.0/4, 4.0/3)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.train_dataset = None

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train = datasets.CIFAR10(self.data_dir, train=True)
            self.train_dataset = TwoViewDataset(train, self.train_xform)

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