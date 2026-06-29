"""Evaluation and checkpoint utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix

from data.labels import LABEL_MAPPING
from models.convnext import build_convnext_tiny
from training.engine import evaluate
from utils.device import get_device


def load_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: torch.device | str | None = None,
    num_classes: int = 2,
):
    """Load a ConvNeXt checkpoint and return model plus metadata."""
    if device is None:
        device = get_device()
    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_convnext_tiny(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(
    checkpoint_path: Path | str,
    val_loader,
    *,
    device: torch.device | str | None = None,
) -> dict:
    """Evaluate a saved checkpoint on a validation dataloader."""
    if device is None:
        device = get_device()
    device = torch.device(device)

    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    y_true, y_pred = evaluate(model, val_loader, device)
    class_names = list(LABEL_MAPPING.values())
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)
    return {
        "checkpoint": checkpoint,
        "classification_report": report,
        "confusion_matrix": matrix,
        "y_true": y_true,
        "y_pred": y_pred,
    }
