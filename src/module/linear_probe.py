# Linear probe on top of a pretrained SimCLR backbone (CIFAR-10 Resnet-50).
from typing import Any, Dict, List

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl
from sklearn.metrics import classification_report, confusion_matrix


def _build_cifar10_resnet50_backbone() -> nn.Module:
    """"""
    backbone = models.resnet50(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    return backbone


def _load_backbone_from_simclr_ckpt(backbone: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    prefix = "backbone."
    sub = {k[len(prefix) : ]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = backbone.load_state_dict(sub, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected load_state_dict result: missing={missing}, unexpected={unexpected}")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False


class LinearProbeClassifier(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer = config["optimizer"]
        self.criterion = nn.CrossEntropyLoss()

        data = config.get("data", "CIFAR10")
        if data != "CIFAR10":
            raise ValueError("This module currently only implements CIFAR-10 + SimCLR-style ResNet50.")
        
        self.backbone = _build_cifar10_resnet50_backbone()
        _load_backbone_from_simclr_ckpt(self.backbone, config["pretrained_ckpt"])

        self.feature_dim = 2048
        self.classifier = nn.Linear(self.feature_dim, config["num_classes"])

        self.test_step_outputs: List[Dict[str, np.ndarray]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen backbone: eval + no grad; BN running stats fixed.
        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone(x)
            if h.dim() > 2:
                h = h.flatten(start_dim=1)
        return self.classifier(h)

    def _shared_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        return loss, acc, y, preds

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, acc, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss
    def validation_step(self, batch: Any, batch_idx: int):
        loss, acc, _, _ = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    def test_step(self, batch: Any, batch_idx: int):
        loss, acc, y, preds = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.test_step_outputs.append(
            {"y_true": y.cpu().numpy(), "y_pred": preds.cpu().numpy()}
        )

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        y_true = np.concatenate([o["y_true"] for o in self.test_step_outputs])
        y_pred = np.concatenate([o["y_pred"] for o in self.test_step_outputs])

        # Save metrics next to this run
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        out: Dict[str, Any] = {
            "test_accuracy": float((y_true == y_pred).mean()),
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        root = Path(self.trainer.default_root_dir) if self.trainer.default_root_dir else Path.cwd()
        metrics_path = root / "linear_probe_metrics.json"
        metrics_path.write_text(json.dumps(out, indent=2))
        print(f"Wrote {metrics_path}")

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return self.optimizer(self.classifier.parameters())