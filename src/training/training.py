"""Backward-compatible training API.

New code should import from `training.engine` and `training.split`.
"""

from training.engine import (
    TrainConfig,
    build_dataloaders,
    build_transforms,
    compute_class_weights,
    evaluate,
    evaluate_with_probs,
    get_primary_metric,
    run_training,
    save_checkpoint,
    train_one_epoch,
    train_with_early_stopping,
)
from training.split import make_group_split

train_one_epoch_v2 = train_one_epoch

__all__ = [
    "TrainConfig",
    "build_dataloaders",
    "build_transforms",
    "compute_class_weights",
    "evaluate",
    "evaluate_with_probs",
    "get_primary_metric",
    "make_group_split",
    "run_training",
    "save_checkpoint",
    "train_one_epoch",
    "train_one_epoch_v2",
    "train_with_early_stopping",
]
