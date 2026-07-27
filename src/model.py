"""Model builders for mouse impairment classification."""

from __future__ import annotations

import torch.nn as nn


def build_model(config) -> nn.Module:
    """Build ResNet18 or ConvNeXt-Tiny for binary impairment classification."""
    from torchvision import models

    model_name = config.model_name.lower()
    builders = {
        "resnet18": (models.resnet18, models.ResNet18_Weights, "fc"),
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights, "classifier"),
    }
    if model_name not in builders:
        raise ValueError("model_name must be one of: resnet18, convnext_tiny")

    builder, weights_enum, head_kind = builders[model_name]
    weights = weights_enum.DEFAULT if config.pretrained else None
    try:
        model = builder(weights=weights)
    except Exception as exc:
        if not config.pretrained:
            raise
        print(f"Could not load pretrained {model_name} weights ({exc}); using random initialization.")
        model = builder(weights=None)

    if config.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    if head_kind == "fc":
        model.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(model.fc.in_features, 2),
        )
    else:
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(in_features, 2),
        )
    return model
