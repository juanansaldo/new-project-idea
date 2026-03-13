from typing import Optional

from torch.utils.data import random_split
from torchvision import datasets, transforms

from src.datamodule.mnist import MNISTDataModule
from src.utils.data_utils import TransformWrapper


class CIFAR10DataModule(MNISTDataModule):
    def __init__(self, **config):
        super().__init__(**config)

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