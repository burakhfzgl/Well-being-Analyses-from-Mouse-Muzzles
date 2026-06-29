"""Quick end-to-end check: paths, labels, dataset load, one training step."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.labels import build_labels, print_label_summary
from data.dataset import MouseImageDataset
from training.engine import build_dataloaders, evaluate, train_one_epoch
from models.model import build_convnext_tiny
from paths import IMAGES_MGS_DIR, MAIN_CSV, MGS_CSV, MOUSE_DATASET_DIR
from training.split import make_group_split
from utils.device import get_device

import torch
import torch.nn as nn


def main() -> None:
    print("=== paths ===")
    print("MOUSE_DATASET_DIR:", MOUSE_DATASET_DIR, "exists:", MOUSE_DATASET_DIR.is_dir())
    print("MGS_CSV:", MGS_CSV.is_file())
    print("MAIN_CSV:", MAIN_CSV.is_file())
    print("IMAGES_MGS_DIR:", IMAGES_MGS_DIR.is_dir())

    print("\n=== build_labels ===")
    df, summary = build_labels(MGS_CSV, MAIN_CSV, IMAGES_MGS_DIR, return_summary=True)
    print_label_summary(df, summary)
    assert len(df) > 0, "label dataframe is empty"

    print("\n=== dataset sample ===")
    sample_path = Path(df.iloc[0]["path"])
    assert sample_path.is_file(), f"missing image: {sample_path}"
    ds = MouseImageDataset(df.head(8))
    img, label = ds[0]
    print("sample image:", sample_path.name, "label:", int(label))

    print("\n=== mini train/val (8 rows, 1 epoch, no pretrained weights) ===")
    mini = df.sample(n=min(80, len(df)), random_state=0).reset_index(drop=True)
    train_df, val_df = make_group_split(mini)
    print("train:", len(train_df), "val:", len(val_df))

    device = get_device()
    print("device:", device)

    train_loader, val_loader = build_dataloaders(
        train_df,
        val_df,
        MouseImageDataset,
        batch_size=8,
        num_workers=0,
        image_size=128,
    )

    model = build_convnext_tiny(num_classes=2, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    y_true, y_pred = evaluate(model, val_loader, device)
    print("train loss:", round(loss, 4))
    print("val preds:", len(y_pred), "labels:", len(y_true))
    print("\nOK - quinyun pipeline is working.")


if __name__ == "__main__":
    main()
