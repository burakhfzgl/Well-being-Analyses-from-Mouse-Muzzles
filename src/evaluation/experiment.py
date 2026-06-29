"""Compatibility wrapper for the older experiment module."""

from evaluation.experiment_modified import (
    LABEL_MAPPING,
    TrainConfig,
    build_dataloaders,
    build_transforms,
    get_device,
    get_primary_metric,
    run_training_with_lr_schedule_early_stop_best_model,
    save_checkpoint,
    save_model,
    train_model_with_lr_schedule_early_stop_best_model,
)

__all__ = [
    "LABEL_MAPPING",
    "TrainConfig",
    "build_dataloaders",
    "build_transforms",
    "get_device",
    "get_primary_metric",
    "run_training_with_lr_schedule_early_stop_best_model",
    "save_checkpoint",
    "save_model",
    "train_model_with_lr_schedule_early_stop_best_model",
]
