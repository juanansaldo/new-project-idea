from torch import nn
from torchvision import models

from src.module.simple_classifier import SimpleClassifier


class ResNet18Classifier(SimpleClassifier):
    def __init__(self, **config):
        super().__init__(**config)

        backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, config["num_classes"])
        self.model = backbone