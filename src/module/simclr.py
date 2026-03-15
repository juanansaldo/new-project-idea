# Sources:
# https://arxiv.org/pdf/2002.05709
# https://docs.lightly.ai/self-supervised-learning/examples/simclr.html
from typing import Any

import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()

        #
        self.optimizer = config["optimizer"]

        # Build backbone
        backbone = models.resnet50(weights=None)

        if config["data"] == "CIFAR10":
            # Replace first conv. layer to accomodate 32x32 images
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()  # Remove first maxpool
        
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        #
        self.projection_head = SimCLRProjectionHead(
            input_dim=2048, 
            hidden_dim=2048,
            output_dim=128,
            num_layers=2,
            batch_norm=True,
        )
        
        #
        self.criterion = NTXentLoss(temperature=config["temperature"])

        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).flatten(start_dim=1)  #
        z = self.projection_head(x)  #
        return z

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        (x0, x1) = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    
