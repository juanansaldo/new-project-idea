# Source: https://arxiv.org/pdf/2002.05709
from pathlib import Path
from typing import Optional

import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class SimCLRImageNetDataModule(pl.LightningDataModule):
    def __init__(self, **config):
        super().__init__()

        self.data_dir = config["data_dir"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        
        self.persistent_workers = True if self.num_workers > 0 else False

        # Standard ImageNet stats
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        s = 1.0           #
        image_size = 224  #
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
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
       pass

    def _make_dataset(self, split: str, transform):
        root = Path(self.data_dir).resolve()
        shard_paths = sorted(root.glob(f"imagenet-{split}-*.tar"))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in {root} matching imagenet-{split}-*.tar")
        shards = [f"file:{p.as_posix()}" for p in shard_paths]

        def _decode_sample(sample):
            img, _ = sample
            return transform(img), transform(img)

        dataset = (
            wds.WebDataset(shards)
            .shuffle(10000)
            .decode("pil")
            .to_tuple("jpg", "cls")
            .map(_decode_sample)
        )
        return dataset

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_dataset("train", self.train_xform)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )