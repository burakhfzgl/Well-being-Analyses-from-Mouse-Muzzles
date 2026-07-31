"""Part 1 / Part 2 article experiment definitions and runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.paths import FIGURES_DIR, IMAGES_MGS_DIR, MODELS_DIR, MOUSE_DATASET_DIR, REPORTS_DIR
from src.train import (
    BalancedDeviceConfig,
    build_balanced_split,
    config_from_dict,
    run_balanced_device_training,
)

ARTICLE_REPORTS_DIR = REPORTS_DIR / "article_experiments"
ARTICLE_FIGURES_DIR = FIGURES_DIR / "article_experiments"
ARTICLE_MODELS_DIR = MODELS_DIR / "article_experiments"
ARTICLE_SPLITS_DIR = MOUSE_DATASET_DIR / "article_experiment_splits"

BASE_SETTINGS = {
    "organized_dir": MOUSE_DATASET_DIR / "Cropped_images",
    "samples_per_bucket": None,
    "train_fraction": 0.70,
    "val_fraction": 0.15,
    "test_fraction": 0.15,
    "epochs": 35,
    "batch_size": 16,
    "image_size": 224,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.30,
    "early_stopping_patience": 7,
    "scheduler_patience": 3,
    "seed": 42,
    "pretrained": True,
    "freeze_backbone": False,
    "split_strategy": "subset_class",
    "reuse_split": True,
    "calibrate_threshold": False,
}

PART1_RUNS = [
    {
        "run_name": "resnet18_crop_light",
        "model_name": "resnet18",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "crop",
        "split_name": "heldout_subset_class_crop",
    },
    {
        "run_name": "resnet18_crop_noaug",
        "model_name": "resnet18",
        "augmentation": False,
        "augmentation_strength": "light",
        "augment_factor": 1,
        "source_mode": "crop",
        "split_name": "heldout_subset_class_crop",
    },
    {
        "run_name": "resnet18_original_noaug",
        "model_name": "resnet18",
        "augmentation": False,
        "augmentation_strength": "light",
        "augment_factor": 1,
        "source_mode": "original",
        "split_name": "heldout_subset_class_original",
    },
    {
        "run_name": "resnet18_original_light",
        "model_name": "resnet18",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "original",
        "split_name": "heldout_subset_class_original",
    },
    {
        "run_name": "convnext_tiny_crop_noaug",
        "model_name": "convnext_tiny",
        "augmentation": False,
        "augmentation_strength": "light",
        "augment_factor": 1,
        "source_mode": "crop",
        "split_name": "heldout_subset_class_crop",
        "batch_size": 8,
    },
    {
        "run_name": "convnext_tiny_crop_light",
        "model_name": "convnext_tiny",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "crop",
        "split_name": "heldout_subset_class_crop",
        "batch_size": 8,
    },
    {
        "run_name": "convnext_tiny_original_noaug",
        "model_name": "convnext_tiny",
        "augmentation": False,
        "augmentation_strength": "light",
        "augment_factor": 1,
        "source_mode": "original",
        "split_name": "heldout_subset_class_original",
        "batch_size": 8,
    },
    {
        "run_name": "convnext_tiny_original_light",
        "model_name": "convnext_tiny",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "original",
        "split_name": "heldout_subset_class_original",
        "batch_size": 8,
    },
]

PART2_RUNS = [
    {
        "run_name": "convnext_tiny_crop_light_lr5e5",
        "model_name": "convnext_tiny",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "crop",
        "split_name": "heldout_subset_class_crop",
        "batch_size": 8,
        "lr": 5e-5,
    },
    {
        "run_name": "convnext_tiny_crop_light_lr3e4",
        "model_name": "convnext_tiny",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "crop",
        "split_name": "heldout_subset_class_crop",
        "batch_size": 8,
        "lr": 3e-4,
    },
    {
        "run_name": "convnext_tiny_crop_light_dropout1",
        "model_name": "convnext_tiny",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "crop",
        "split_name": "heldout_subset_class_crop",
        "batch_size": 8,
        "dropout": 0.10,
    },
    {
        "run_name": "convnext_tiny_crop_light_dropout0",
        "model_name": "convnext_tiny",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "crop",
        "split_name": "heldout_subset_class_crop",
        "batch_size": 8,
        "dropout": 0.0,
    },
    {
        "run_name": "convnext_tiny_crop_light_frozen",
        "model_name": "convnext_tiny",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "crop",
        "split_name": "heldout_subset_class_crop",
        "batch_size": 8,
        "freeze_backbone": True,
        "lr": 3e-4,
    },
    {
        "run_name": "resnet18_original_light_lr5e5",
        "model_name": "resnet18",
        "augmentation": True,
        "augmentation_strength": "light",
        "augment_factor": 2,
        "source_mode": "original",
        "split_name": "heldout_subset_class_original",
        "lr": 5e-5,
    },
]


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    return value


def remap_to_original(split_df: pd.DataFrame) -> pd.DataFrame:
    """Use original MGS images for the same indices and labels."""
    mapped = split_df.copy()
    mapped["path"] = mapped["index"].map(lambda name: str(IMAGES_MGS_DIR / name))
    exists = mapped["path"].map(lambda value: Path(value).is_file())
    missing = mapped.loc[~exists, "index"].tolist()
    if missing:
        raise RuntimeError(
            f"{len(missing)} split images do not exist in images_mgs; first missing: {missing[:5]}"
        )
    return mapped


def write_split(split_df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False)
    return path


def prepare_heldout_splits(base_config: BalancedDeviceConfig) -> dict[str, Path]:
    """Prepare fixed crop/original held-out splits."""
    ARTICLE_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    crop_split_path = ARTICLE_SPLITS_DIR / "heldout_subset_class_crop.csv"
    original_split_path = ARTICLE_SPLITS_DIR / "heldout_subset_class_original.csv"

    if not crop_split_path.is_file():
        crop_df = build_balanced_split(base_config)
        write_split(crop_df, crop_split_path)
    else:
        crop_df = pd.read_csv(crop_split_path)

    if not original_split_path.is_file():
        write_split(remap_to_original(crop_df), original_split_path)

    return {
        "heldout_subset_class_crop": crop_split_path,
        "heldout_subset_class_original": original_split_path,
    }


def experiment_settings(run: dict, split_csv: Path, run_name: str, part: str) -> dict:
    settings = {**BASE_SETTINGS, **run}
    settings["run_name"] = run_name
    settings["split_csv"] = split_csv
    settings["part"] = part
    settings["checkpoint"] = ARTICLE_MODELS_DIR / part / f"{run_name}.pt"
    settings["output_dir"] = ARTICLE_REPORTS_DIR / part / run_name
    settings["figures_dir"] = ARTICLE_FIGURES_DIR / part / run_name
    settings.pop("split_name", None)
    return settings


def selected_runs(part: str) -> list[tuple[str, dict]]:
    part_map = {
        "part1": PART1_RUNS,
        "part2": PART2_RUNS,
    }
    if part == "all":
        return [(part_name, run) for part_name, runs in part_map.items() for run in runs]
    return [(part, run) for run in part_map[part]]


def iter_experiments(part: str, only: set[str] | None = None) -> Iterable[dict]:
    base_config = config_from_dict(
        BASE_SETTINGS | {"split_csv": ARTICLE_SPLITS_DIR / "heldout_subset_class_crop.csv"}
    )
    heldout_splits = prepare_heldout_splits(base_config)
    for part_name, run in selected_runs(part):
        run_name = run["run_name"]
        if only and run_name not in only:
            continue
        yield experiment_settings(run, heldout_splits[run["split_name"]], run_name, part_name)


def write_manifest(experiments: list[dict]) -> None:
    ARTICLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (ARTICLE_REPORTS_DIR / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(
            [{key: _json_ready(value) for key, value in item.items()} for item in experiments],
            file,
            indent=2,
        )
    pd.DataFrame(
        [{key: _json_ready(value) for key, value in item.items()} for item in experiments]
    ).to_csv(ARTICLE_REPORTS_DIR / "experiment_manifest.csv", index=False)


def run_experiments(
    part: str = "part1",
    *,
    dry_run: bool = False,
    only: list[str] | None = None,
    limit_runs: int | None = None,
) -> None:
    """Prepare and optionally train Part 1 / Part 2 experiments."""
    only_set = set(only) if only else None
    experiments = list(iter_experiments(part, only=only_set))
    if limit_runs is not None:
        experiments = experiments[:limit_runs]
    write_manifest(experiments)

    print(f"Prepared {len(experiments)} experiments.")
    print(f"Manifest: {(ARTICLE_REPORTS_DIR / 'experiment_manifest.csv').resolve()}")
    for settings in experiments:
        print(
            f"- {settings['run_name']}: {settings['model_name']} | "
            f"{settings['source_mode']} | {settings['split_csv']}"
        )

    if dry_run:
        return

    for settings in experiments:
        print(f"\n=== Running {settings['run_name']} ===")
        run_balanced_device_training(settings)


def build_train_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--part", choices=["part1", "part2", "all"], default="part1")
    parser.add_argument("--dry-run", action="store_true", help="Prepare splits without training.")
    parser.add_argument("--only", nargs="*", help="Run only these run names.")
    parser.add_argument("--limit-runs", type=int, default=None, help="Run only the first N experiments.")
    return parser
