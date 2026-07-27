"""Balanced subset/class training pipeline for mouse impairment classification."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from paths import FIGURES_DIR, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, MOUSE_DATASET_DIR, REPORTS_DIR
from utils.device import get_device
from utils.reproducibility import set_seed

SUBSETS = ("AW", "JW", "KH", "LW", "MR")
LABELS = {"not_impaired": 0, "impaired": 1}
LABEL_NAMES = {0: "not_impaired", 1: "impaired"}


@dataclass
class BalancedDeviceConfig:
    """Settings for the balanced device experiment."""

    organized_dir: Path = MOUSE_DATASET_DIR / "images_perfect_organized"
    split_csv: Path = MOUSE_DATASET_DIR / "balanced_device_split.csv"
    checkpoint: Path = MODELS_DIR / "resnet18_balanced_device.pt"
    output_dir: Path = REPORTS_DIR / "balanced_device"
    figures_dir: Path = FIGURES_DIR / "balanced_device"
    samples_per_bucket: int | None = None
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    epochs: int = 35
    batch_size: int = 16
    image_size: int = 224
    lr: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.30
    early_stopping_patience: int = 7
    min_delta: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    num_workers: int = 0
    seed: int = 42
    deterministic: bool = False
    augmentation: bool = True
    augment_factor: int = 2
    model_name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
    split_strategy: str = "subset_class"
    calibrate_threshold: bool = False
    reuse_split: bool = False
    run_name: str = "balanced_device"
    part: str = "default"
    source_mode: str = "crop"
    augmentation_strength: str = "light"


def apply_train_augmentation(image: Image.Image, strength: str = "light") -> Image.Image:
    """Apply random spatial and photometric augmentations for training."""
    strength = strength.lower()
    if strength not in {"light", "strong"}:
        raise ValueError("augmentation_strength must be one of: light, strong")
    max_rotation = 12 if strength == "light" else 20
    max_shift_fraction = 0.08 if strength == "light" else 0.12
    brightness_range = (0.85, 1.15) if strength == "light" else (0.75, 1.25)
    contrast_range = (0.85, 1.15) if strength == "light" else (0.75, 1.25)
    color_range = (0.9, 1.1) if strength == "light" else (0.8, 1.2)

    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    if random.random() < 0.5:
        angle = random.uniform(-max_rotation, max_rotation)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))

    if random.random() < 0.5:
        max_shift = int(image.size[0] * max_shift_fraction)
        if max_shift > 0:
            dx = random.randint(-max_shift, max_shift)
            dy = random.randint(-max_shift, max_shift)
            image = image.transform(
                image.size,
                Image.AFFINE,
                (1, 0, dx, 0, 1, dy),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(128, 128, 128),
            )

    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(*brightness_range))

    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(*contrast_range))

    if random.random() < 0.3:
        image = ImageEnhance.Color(image).enhance(random.uniform(*color_range))

    return image


def pil_to_tensor(image: Image.Image, image_size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Convert a PIL image to a normalized tensor without NumPy."""
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image_tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    image_tensor = image_tensor.view(image_size, image_size, 3)
    image_tensor = image_tensor.permute(2, 0, 1).float().div(255.0)
    return (image_tensor - mean) / std


class DeviceImageDataset(Dataset):
    """Dataset returning images, labels, subset names, and filenames."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int,
        *,
        augment: bool = False,
        augmentation_strength: str = "light",
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.augmentation_strength = augmentation_strength
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        if self.augment:
            image = apply_train_augmentation(image, self.augmentation_strength)
        image_tensor = pil_to_tensor(image, self.image_size, self.mean, self.std)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image_tensor, label, row["subset"], row["index"]


class RepeatedAugmentDataset(Dataset):
    """Repeat each base sample to increase effective training size with fresh augmentations."""

    def __init__(self, base: DeviceImageDataset, repeat_factor: int) -> None:
        if repeat_factor < 1:
            raise ValueError("repeat_factor must be >= 1")
        self.base = base
        self.repeat_factor = repeat_factor

    def __len__(self) -> int:
        return len(self.base) * self.repeat_factor

    def __getitem__(self, index: int):
        return self.base[index % len(self.base)]


def _as_path(value: Path | str) -> Path:
    return value if isinstance(value, Path) else Path(value)


def config_from_dict(settings: dict | None = None) -> BalancedDeviceConfig:
    """Create a training config from a script settings dictionary."""
    config = BalancedDeviceConfig()
    if not settings:
        return config

    for key, value in settings.items():
        if not hasattr(config, key):
            raise KeyError(f"Unknown balanced device setting: {key}")
        if key.endswith("_dir") or key in {"organized_dir", "split_csv", "checkpoint", "output_dir", "figures_dir"}:
            value = _as_path(value)
        setattr(config, key, value)
    return config


def _split_counts(total: int, train_fraction: float, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")
    test_count = max(1, round(total * test_fraction))
    val_count = max(1, round(total * val_fraction))
    train_count = total - val_count - test_count
    if train_count <= 0:
        raise ValueError(f"Not enough samples per bucket for this split: {total}")
    return train_count, val_count, test_count


def _add_split_rows(
    rows: list[dict],
    files: list[Path],
    split_names: list[str],
    label: int,
    class_name: str,
) -> None:
    for split, path in zip(split_names, files, strict=True):
        rows.append(
            {
                "index": path.name,
                "path": str(path),
                "subset": path.parent.parent.name,
                "label": label,
                "label_name": class_name,
                "split": split,
            }
        )


def _split_names(total: int, config: BalancedDeviceConfig) -> list[str]:
    train_count, val_count, test_count = _split_counts(
        total,
        config.train_fraction,
        config.val_fraction,
        config.test_fraction,
    )
    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def build_subset_class_split(config: BalancedDeviceConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Split each subset/class bucket independently."""
    rows: list[dict] = []
    missing: list[str] = []

    for subset in SUBSETS:
        for class_name, label in LABELS.items():
            folder = config.organized_dir / subset / class_name
            files = sorted(folder.glob("*.jpg"))
            if not files:
                missing.append(f"{subset}/{class_name}: have 0 images")
                continue

            bucket_size = len(files) if config.samples_per_bucket is None else config.samples_per_bucket
            if len(files) < bucket_size:
                missing.append(
                    f"{subset}/{class_name}: have {len(files)}, need {bucket_size}"
                )
                continue

            selected_indices = rng.permutation(len(files))[:bucket_size]
            selected_files = [files[int(i)] for i in selected_indices]
            _add_split_rows(rows, selected_files, _split_names(bucket_size, config), label, class_name)

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise ValueError(
            "The organized cropped dataset is not ready for a balanced split.\n"
            "Run scripts/data_processing/prepare_organized_dataset.py and ensure "
            "each subset/class folder under images_perfect_organized has images:\n"
            f"{details}"
        )

    return pd.DataFrame(rows)


def build_balanced_split(config: BalancedDeviceConfig) -> pd.DataFrame:
    """Build and save the subset/class stratified train/val/test split."""
    if config.reuse_split and config.split_csv.is_file():
        return pd.read_csv(config.split_csv)

    if config.split_strategy != "subset_class":
        raise ValueError("split_strategy must be 'subset_class'")

    rng = np.random.default_rng(config.seed)
    df = build_subset_class_split(config, rng)
    config.split_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.split_csv, index=False)
    return df


def build_model(config: BalancedDeviceConfig) -> nn.Module:
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


def build_loaders(split_df: pd.DataFrame, config: BalancedDeviceConfig) -> dict[str, DataLoader]:
    """Create dataloaders for train/val/test."""
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(config.seed)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        split_data = split_df[split_df["split"] == split].reset_index(drop=True)
        augment = split == "train" and config.augmentation
        dataset = DeviceImageDataset(
            split_data,
            image_size=config.image_size,
            augment=augment,
            augmentation_strength=config.augmentation_strength,
        )
        if split == "train" and config.augmentation and config.augment_factor > 1:
            dataset = RepeatedAugmentDataset(dataset, config.augment_factor)

        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            generator=generator if split == "train" else None,
        )
    return loaders


def config_to_json_dict(config: BalancedDeviceConfig) -> dict:
    """Return config values in JSON-serializable form."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def save_run_config(config: BalancedDeviceConfig) -> None:
    """Persist the exact run settings next to the metrics."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config_to_json_dict(config), file, indent=2)


def save_epoch_history(history: list[dict] | pd.DataFrame, output_dir: Path) -> None:
    """Flush per-epoch history so interrupted runs still leave usable metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)


def move_inputs_to_device(inputs: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move image batch tensor to the target device."""
    return inputs.to(device, non_blocking=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train one epoch and return loss plus classification metrics."""
    model.train()
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels, _, _ in tqdm(loader, desc="Train", leave=False):
        images = move_inputs_to_device(images, device)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def predict(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float | None = None) -> dict:
    """Return loss-free predictions with metadata."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    subsets: list[str] = []
    indices: list[str] = []

    with torch.no_grad():
        for images, labels, batch_subsets, batch_indices in tqdm(loader, desc="Predict", leave=False):
            images = move_inputs_to_device(images, device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            if threshold is None:
                preds = probs.argmax(dim=1)
            else:
                preds = (probs[:, 1] >= threshold).long()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())
            y_prob.extend(probs[:, 1].detach().cpu().tolist())
            subsets.extend(list(batch_subsets))
            indices.extend(list(batch_indices))

    return {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_prob": np.asarray(y_prob),
        "subset": np.asarray(subsets),
        "index": np.asarray(indices),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Select a decision threshold using macro-F1 on a validation split."""
    best_threshold = 0.5
    best_macro_f1 = float("-inf")
    for threshold in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= threshold).astype(int)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)
    return best_threshold, best_macro_f1


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float | None = None,
) -> dict:
    """Evaluate a split and return loss plus classification metrics."""
    model.eval()
    total_loss = 0.0
    predictions = predict(model, loader, device, threshold=threshold)

    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = move_inputs_to_device(images, device)
            labels = labels.to(device, non_blocking=True)
            total_loss += float(criterion(model(images), labels).item())

    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    y_prob = predictions["y_prob"]
    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["average_precision"] = average_precision_score(y_true, y_prob)
        metrics["sensitivity"] = tp / max(tp + fn, 1)
        metrics["specificity"] = tn / max(tn + fp, 1)
        metrics["brier_score"] = brier_score_loss(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
        metrics["sensitivity"] = np.nan
        metrics["specificity"] = np.nan
        metrics["brier_score"] = np.nan
    if threshold is not None:
        metrics["decision_threshold"] = threshold
    return {**predictions, **metrics}


def save_checkpoint(
    model: nn.Module,
    config: BalancedDeviceConfig,
    *,
    epoch: int,
    best_metric: float,
) -> None:
    """Save best model checkpoint."""
    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_macro_f1": best_metric,
            "config": config_to_json_dict(config),
            "label_names": LABEL_NAMES,
        },
        config.checkpoint,
    )


def train_model(split_df: pd.DataFrame, config: BalancedDeviceConfig) -> tuple[nn.Module, pd.DataFrame, dict]:
    """Train the configured model and return best model, history dataframe, and final test metrics."""
    set_seed(config.seed, deterministic=config.deterministic)
    device = get_device()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(config)
    print(f"\nUsing device: {device}")
    print(f"Balanced split CSV: {config.split_csv.resolve()}")
    print(
        f"Model: {config.model_name} | pretrained={config.pretrained} "
        f"| freeze_backbone={config.freeze_backbone} | source={config.source_mode}"
    )
    train_rows = len(split_df[split_df["split"] == "train"])
    if config.augmentation:
        effective_train = train_rows * config.augment_factor
        print(
            f"Training augmentation: ON "
            f"({train_rows} base images x {config.augment_factor} = {effective_train} views/epoch)"
        )
    else:
        print("Training augmentation: OFF")

    loaders = build_loaders(split_df, config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    history: list[dict] = []
    best_val_f1 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config.epochs} | lr={current_lr:.8f}")
        train = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val = evaluate_split(model, loaders["val"], criterion, device)

        print(
            "train_loss={:.4f} train_acc={:.4f} train_macro_f1={:.4f} | "
            "val_loss={:.4f} val_acc={:.4f} val_bal_acc={:.4f} val_macro_f1={:.4f}".format(
                train["loss"],
                train["accuracy"],
                train["macro_f1"],
                val["loss"],
                val["accuracy"],
                val["balanced_accuracy"],
                val["macro_f1"],
            )
        )

        history.append(
            {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train["loss"],
                "train_accuracy": train["accuracy"],
                "train_balanced_accuracy": train["balanced_accuracy"],
                "train_macro_f1": train["macro_f1"],
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_balanced_accuracy": val["balanced_accuracy"],
                "val_macro_f1": val["macro_f1"],
                "val_precision_macro": val["precision_macro"],
                "val_recall_macro": val["recall_macro"],
                "val_mcc": val["mcc"],
                "val_roc_auc": val["roc_auc"],
                "val_average_precision": val["average_precision"],
                "val_sensitivity": val["sensitivity"],
                "val_specificity": val["specificity"],
                "val_brier_score": val["brier_score"],
            }
        )
        save_epoch_history(history, config.output_dir)

        scheduler.step(val["macro_f1"])
        improved = val["macro_f1"] > best_val_f1 + config.min_delta
        if improved:
            best_val_f1 = val["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(model, config, epoch=epoch, best_metric=best_val_f1)
            print(f"Saved best checkpoint: {config.checkpoint.resolve()}")
        else:
            epochs_without_improvement += 1
            print(
                f"No val macro-F1 improvement "
                f"({epochs_without_improvement}/{config.early_stopping_patience})"
            )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    checkpoint = torch.load(config.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if config.calibrate_threshold:
        val_predictions = predict(model, loaders["val"], device)
        decision_threshold, val_threshold_f1 = find_best_threshold(
            val_predictions["y_true"],
            val_predictions["y_prob"],
        )
        print(
            f"\nSelected decision threshold from validation: "
            f"{decision_threshold:.2f} (val macro-F1={val_threshold_f1:.4f})"
        )
        test_metrics = evaluate_split(model, loaders["test"], criterion, device, threshold=decision_threshold)
    else:
        test_metrics = evaluate_split(model, loaders["test"], criterion, device)
    history_df = pd.DataFrame(history)
    return model, history_df, test_metrics


def split_distribution(split_df: pd.DataFrame) -> pd.DataFrame:
    """Return counts by split/subset/class."""
    return (
        split_df.groupby(["split", "subset", "label_name"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "subset", "label_name"])
    )


def per_subset_metrics(results: dict) -> pd.DataFrame:
    """Compute test metrics separately for each subset."""
    rows: list[dict] = []
    for subset in SUBSETS:
        mask = results["subset"] == subset
        y_true = results["y_true"][mask]
        y_pred = results["y_pred"][mask]
        if len(y_true) == 0:
            continue
        rows.append(
            {
                "subset": subset,
                "n": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def save_metrics(config: BalancedDeviceConfig, history: pd.DataFrame, split_df: pd.DataFrame, test: dict) -> None:
    """Save CSV/JSON metrics for article tables."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(config)
    history.to_csv(config.output_dir / "training_history.csv", index=False)
    split_distribution(split_df).to_csv(config.output_dir / "split_distribution.csv", index=False)

    y_true = test["y_true"]
    y_pred = test["y_pred"]
    report = classification_report(
        y_true,
        y_pred,
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(config.output_dir / "test_classification_report.csv")
    per_subset_metrics(test).to_csv(config.output_dir / "test_per_subset_metrics.csv", index=False)

    predictions = pd.DataFrame(
        {
            "index": test["index"],
            "subset": test["subset"],
            "true_label": test["y_true"],
            "predicted_label": test["y_pred"],
            "impaired_probability": test["y_prob"],
        }
    )
    predictions.to_csv(config.output_dir / "test_predictions.csv", index=False)

    summary = {
        "run_name": config.run_name,
        "part": config.part,
        "model_name": config.model_name,
        "source_mode": config.source_mode,
        "split_strategy": config.split_strategy,
        "augmentation": config.augmentation,
        "augmentation_strength": config.augmentation_strength,
        "augment_factor": config.augment_factor,
        "test_accuracy": test["accuracy"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_precision_macro": test["precision_macro"],
        "test_recall_macro": test["recall_macro"],
        "test_mcc": test["mcc"],
        "test_roc_auc": test["roc_auc"],
        "test_average_precision": test["average_precision"],
        "test_sensitivity": test["sensitivity"],
        "test_specificity": test["specificity"],
        "test_brier_score": test["brier_score"],
        "decision_threshold": test.get("decision_threshold", 0.5),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    with (config.output_dir / "test_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    pd.DataFrame([{key: value for key, value in summary.items() if key != "confusion_matrix"}]).to_csv(
        config.output_dir / "experiment_summary.csv",
        index=False,
    )


def plot_history(history: pd.DataFrame, figures_dir: Path) -> None:
    """Save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    axes[2].plot(history["epoch"], history["val_macro_f1"], label="Val macro-F1")
    axes[2].plot(history["epoch"], history["val_roc_auc"], label="Val ROC-AUC")
    axes[2].set_title("Validation Metrics")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "training_curves.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrix(test: dict, figures_dir: Path) -> None:
    """Save confusion matrix figure."""
    matrix = confusion_matrix(test["y_true"], test["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], [LABEL_NAMES[0], LABEL_NAMES[1]], rotation=20, ha="right")
    ax.set_yticks([0, 1], [LABEL_NAMES[0], LABEL_NAMES[1]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(figures_dir / "test_confusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_roc_pr(test: dict, figures_dir: Path) -> None:
    """Save ROC and precision-recall curves."""
    y_true = test["y_true"]
    y_prob = test["y_prob"]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"ROC-AUC={test['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"AP={test['average_precision']:.3f}")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "test_roc_pr_curves.png", dpi=300)
    plt.close(fig)


def plot_calibration_curve(test: dict, figures_dir: Path) -> None:
    """Save predicted-probability calibration curve."""
    y_true = test["y_true"]
    y_prob = test["y_prob"]
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted impaired probability")
    ax.set_ylabel("Fraction impaired")
    ax.set_title("Calibration Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "test_calibration_curve.png", dpi=300)
    plt.close(fig)


def plot_probability_histogram(test: dict, figures_dir: Path) -> None:
    """Save class-wise predicted probability histogram."""
    y_true = test["y_true"]
    y_prob = test["y_prob"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_prob[y_true == 0], bins=20, alpha=0.65, label=LABEL_NAMES[0])
    ax.hist(y_prob[y_true == 1], bins=20, alpha=0.65, label=LABEL_NAMES[1])
    ax.axvline(test.get("decision_threshold", 0.5), linestyle="--", color="black", label="Threshold")
    ax.set_xlabel("Predicted impaired probability")
    ax.set_ylabel("Images")
    ax.set_title("Prediction Probability Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "test_probability_histogram.png", dpi=300)
    plt.close(fig)


def plot_per_class_metrics(test: dict, figures_dir: Path) -> None:
    """Save precision/recall/F1 bars for each class."""
    report = classification_report(
        test["y_true"],
        test["y_pred"],
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    metrics = pd.DataFrame(report).transpose().loc[[LABEL_NAMES[0], LABEL_NAMES[1]], ["precision", "recall", "f1-score"]]
    ax = metrics.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Test Metrics")
    ax.legend(title="Metric")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figures_dir / "test_per_class_metrics.png", dpi=300)
    plt.close(fig)


def plot_per_subset_metrics(test: dict, figures_dir: Path) -> None:
    """Save per-subset test accuracy/macro-F1 bars."""
    metrics = per_subset_metrics(test)
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, metrics["accuracy"], width, label="Accuracy")
    ax.bar(x + width / 2, metrics["macro_f1"], width, label="Macro-F1")
    ax.set_xticks(x, metrics["subset"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Subset")
    ax.set_ylabel("Score")
    ax.set_title("Test Metrics by Subset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "test_per_subset_metrics.png", dpi=300)
    plt.close(fig)


def plot_split_distribution(split_df: pd.DataFrame, figures_dir: Path) -> None:
    """Save split distribution figure."""
    counts = split_df.groupby(["split", "label_name"]).size().unstack(fill_value=0)
    counts = counts.loc[["train", "val", "test"]]
    ax = counts.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_title("Balanced Split Distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Images")
    ax.legend(title="Class")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(figures_dir / "split_distribution.png", dpi=300)
    plt.close(fig)


def save_figures(config: BalancedDeviceConfig, history: pd.DataFrame, split_df: pd.DataFrame, test: dict) -> None:
    """Save article-ready figures."""
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_history(history, config.figures_dir)
    plot_confusion_matrix(test, config.figures_dir)
    plot_roc_pr(test, config.figures_dir)
    plot_calibration_curve(test, config.figures_dir)
    plot_probability_histogram(test, config.figures_dir)
    plot_per_class_metrics(test, config.figures_dir)
    plot_per_subset_metrics(test, config.figures_dir)
    plot_split_distribution(split_df, config.figures_dir)


def print_terminal_report(split_df: pd.DataFrame, test: dict) -> None:
    """Print key terminal metrics."""
    print("\nSplit distribution:")
    print(split_distribution(split_df).to_string(index=False))

    print("\nTest metrics:")
    if "decision_threshold" in test:
        print(f"Decision threshold: {test['decision_threshold']:.2f}")
    print(f"Accuracy:           {test['accuracy']:.4f}")
    print(f"Balanced accuracy:  {test['balanced_accuracy']:.4f}")
    print(f"Macro F1:           {test['macro_f1']:.4f}")
    print(f"Macro precision:    {test['precision_macro']:.4f}")
    print(f"Macro recall:       {test['recall_macro']:.4f}")
    print(f"MCC:                {test['mcc']:.4f}")
    print(f"ROC AUC:            {test['roc_auc']:.4f}")
    print(f"Average precision:  {test['average_precision']:.4f}")
    print(f"Sensitivity:        {test['sensitivity']:.4f}")
    print(f"Specificity:        {test['specificity']:.4f}")
    print(f"Brier score:        {test['brier_score']:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(test["y_true"], test["y_pred"]))
    print("\nClassification report:")
    print(
        classification_report(
            test["y_true"],
            test["y_pred"],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
            zero_division=0,
        )
    )
    print("\nPer-subset test metrics:")
    print(per_subset_metrics(test).to_string(index=False))


def run_balanced_device_training(settings: dict | None = None) -> None:
    """Build split, train model, evaluate test set, and save outputs."""
    config = config_from_dict(settings)
    split_df = build_balanced_split(config)
    _, history, test = train_model(split_df, config)
    save_metrics(config, history, split_df, test)
    save_figures(config, history, split_df, test)
    print_terminal_report(split_df, test)
    print(f"\nBest checkpoint: {config.checkpoint.resolve()}")
    print(f"Metrics folder:  {config.output_dir.resolve()}")
    print(f"Figures folder:  {config.figures_dir.resolve()}")
