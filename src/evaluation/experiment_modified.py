"""Backward-compatible experiment API.

New code should import from `training.engine` and `evaluation.metrics`.
"""

from __future__ import annotations

import torch

from data.labels import LABEL_MAPPING
from models.convnext import build_convnext_tiny
from training.engine import (
    TrainConfig,
    build_dataloaders,
    build_transforms,
    get_primary_metric,
    save_checkpoint,
    train_with_early_stopping,
)
from utils.device import get_device


def save_model(model, save_path, label_mapping=None):
    """Save a simple checkpoint for old notebook compatibility."""
    if label_mapping is None:
        label_mapping = LABEL_MAPPING
    torch.save({"model_state_dict": model.state_dict(), "label_mapping": label_mapping}, save_path)


def train_model_with_lr_schedule_early_stop_best_model(
    model,
    train_loader,
    val_loader,
    train_one_epoch_fn,
    evaluate_fn,
    device,
    epochs=30,
    lr=1e-4,
    weight_decay=1e-4,
    target_names=None,
    save_path="outputs/models/best_model.pt",
    metric_name="macro_f1",
    scheduler_factor=0.5,
    scheduler_patience=2,
    early_stopping_patience=5,
    min_delta=1e-4,
):
    """Compatibility wrapper around `train_with_early_stopping`."""
    config = TrainConfig(
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        metric_name=metric_name,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
    )
    return train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        train_one_epoch_fn=train_one_epoch_fn,
        evaluate_fn=evaluate_fn,
        device=device,
        config=config,
        save_path=save_path,
    )


def run_training_with_lr_schedule_early_stop_best_model(
    train_df,
    val_df,
    dataset_class,
    train_one_epoch_fn,
    evaluate_fn,
    save_path="outputs/models/best_model.pt",
    epochs=30,
    batch_size=32,
    image_size=224,
    lr=1e-4,
    weight_decay=1e-4,
    num_workers=0,
    pretrained=True,
    metric_name="macro_f1",
    scheduler_factor=0.5,
    scheduler_patience=2,
    early_stopping_patience=5,
    min_delta=1e-4,
):
    """Compatibility wrapper for existing notebooks."""
    device = get_device()
    train_loader, val_loader = build_dataloaders(
        train_df,
        val_df,
        dataset_class,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    model = build_convnext_tiny(num_classes=2, pretrained=pretrained)
    config = TrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        lr=lr,
        weight_decay=weight_decay,
        num_workers=num_workers,
        pretrained=pretrained,
        metric_name=metric_name,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
    )
    return train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        train_one_epoch_fn=train_one_epoch_fn,
        evaluate_fn=evaluate_fn,
        device=device,
        config=config,
        save_path=save_path,
        train_df=train_df,
    )
