"""Training and evaluation engine for binary image classification."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.labels import LABEL_MAPPING
from models.convnext import build_convnext_tiny
from paths import IMAGENET_MEAN, IMAGENET_STD
from utils.device import get_device


@dataclass
class TrainConfig:
    """Hyperparameters and runtime settings for one training run."""

    epochs: int = 30
    batch_size: int = 32
    image_size: int = 224
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    pretrained: bool = True
    metric_name: str = "macro_f1"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    seed: int = 42
    class_weights: bool = False


def build_transforms(image_size: int = 224):
    """Build ImageNet-normalized train and validation transforms."""
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tfms, val_tfms


def build_dataloaders(
    train_df,
    val_df,
    dataset_class,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_tfms, val_tfms = build_transforms(image_size=image_size)
    train_ds = dataset_class(train_df, transform=train_tfms)
    val_ds = dataset_class(val_df, transform=val_tfms)

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def compute_class_weights(train_df, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training split."""
    counts = train_df["label"].value_counts().sort_index()
    total = counts.sum()
    weights = [total / (len(counts) * counts.get(label, 1)) for label in sorted(LABEL_MAPPING)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float | None = 1.0,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if torch.isnan(loss):
            raise ValueError("Training stopped because loss became NaN.")

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return validation labels and predicted classes."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return all_labels, all_preds


def evaluate_with_probs(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return labels, predicted classes, and class probabilities."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

    return np.asarray(all_labels), np.asarray(all_preds), np.asarray(all_probs)


def get_primary_metric(report: dict, metric_name: str = "macro_f1") -> float:
    """Extract the validation metric used for scheduler and checkpointing."""
    if metric_name == "macro_f1":
        return float(report["macro avg"]["f1-score"])
    if metric_name == "weighted_f1":
        return float(report["weighted avg"]["f1-score"])
    if metric_name == "impaired_f1":
        return float(report["impaired"]["f1-score"])
    if metric_name == "impaired_recall":
        return float(report["impaired"]["recall"])
    if metric_name == "accuracy":
        return float(report["accuracy"])
    raise ValueError(
        "Unknown metric_name. Use one of: macro_f1, weighted_f1, "
        "impaired_f1, impaired_recall, accuracy."
    )


def save_checkpoint(
    model: nn.Module,
    save_path: Path | str,
    *,
    epoch: int,
    best_metric: float,
    metric_name: str,
    config: TrainConfig | dict,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save model state and training metadata."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config) if isinstance(config, TrainConfig) else dict(config)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_mapping": LABEL_MAPPING,
        "epoch": epoch,
        "best_metric": best_metric,
        "metric_name": metric_name,
        "config": config_dict,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, save_path)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    train_one_epoch_fn: Callable = train_one_epoch,
    evaluate_fn: Callable = evaluate,
    device: torch.device | None = None,
    config: TrainConfig | None = None,
    save_path: Path | str = "outputs/models/best_model.pt",
    train_df=None,
) -> tuple[nn.Module, list[dict]]:
    """Train a model with LR scheduling, early stopping, and best checkpoint saving."""
    if config is None:
        config = TrainConfig()
    if device is None:
        device = get_device()

    model = model.to(device)
    if config.class_weights:
        if train_df is None:
            raise ValueError("train_df is required when class_weights=True")
        criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_df, device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    history: list[dict] = []
    best_metric = float("-inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    class_names = list(LABEL_MAPPING.values())

    for epoch in range(1, config.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{config.epochs}")
        print(f"Current LR: {current_lr:.8f}")

        train_loss = train_one_epoch_fn(model, train_loader, criterion, optimizer, device)
        y_true, y_pred = evaluate_fn(model, val_loader, device)

        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred)
        current_metric = get_primary_metric(report, config.metric_name)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation {config.metric_name}: {current_metric:.4f}")
        print("\nConfusion matrix:")
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        scheduler.step(current_metric)

        improved = current_metric > best_metric + config.min_delta
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            save_checkpoint(
                model,
                save_path,
                epoch=epoch,
                best_metric=best_metric,
                metric_name=config.metric_name,
                config=config,
                optimizer=optimizer,
            )
            print(f"Saved new best checkpoint to {save_path}")
        else:
            epochs_without_improvement += 1
            print(
                "No improvement. "
                f"Early-stopping counter: {epochs_without_improvement}/"
                f"{config.early_stopping_patience}"
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": current_lr,
                "val_metric_name": config.metric_name,
                "val_metric": current_metric,
                "best_metric_so_far": best_metric,
                "best_epoch_so_far": best_epoch,
                "confusion_matrix": cm,
                "classification_report": report,
            }
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                "\nEarly stopping triggered. "
                f"Best epoch: {best_epoch}, best {config.metric_name}: {best_metric:.4f}"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history


def run_training(
    train_df,
    val_df,
    dataset_class,
    *,
    save_path: Path | str,
    config: TrainConfig | None = None,
) -> tuple[nn.Module, list[dict]]:
    """Build dataloaders, create ConvNeXt, and train."""
    if config is None:
        config = TrainConfig()
    device = get_device()
    print("\nUsing device:", device)

    train_loader, val_loader = build_dataloaders(
        train_df,
        val_df,
        dataset_class,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        seed=config.seed,
    )
    model = build_convnext_tiny(num_classes=len(LABEL_MAPPING), pretrained=config.pretrained)
    return train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        device=device,
        config=config,
        save_path=save_path,
        train_df=train_df,
    )
