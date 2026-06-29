"""ConvNeXt model builders."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_convnext_tiny(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build a ConvNeXt-Tiny classifier with a custom output head."""
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model
