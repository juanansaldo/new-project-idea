from typing import Any

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import classification_report


class SimpleMNISTClassifier(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer = config["optimizer"]
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        return loss, acc

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, acc = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, acc = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.test_step_outputs.append({
            "y_true": y.cpu().numpy(),
            "y_pred": preds.cpu().numpy(),
        })

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        y_true = np.concatenate([o["y_true"] for o in self.test_step_outputs])
        y_pred = np.concatenate([o["y_pred"] for o in self.test_step_outputs])
        print(classification_report(y_true, y_pred, digits=4))
        self.test_step_outputs.clear()
        
    def configure_optimizers(self):
        return self.optimizer(self.parameters())