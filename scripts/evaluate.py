"""Evaluate a saved classifier checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import MouseImageDataset
from data.labels import LABEL_MAPPING, build_labels
from evaluation.metrics import evaluate_checkpoint
from paths import IMAGES_MGS_DIR, MAIN_CSV, MGS_CSV, MODELS_DIR
from training.engine import build_dataloaders
from training.split import make_group_split
from utils.device import get_device
from utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODELS_DIR / "best_model.pt")
    parser.add_argument("--mgs-csv", type=Path, default=MGS_CSV)
    parser.add_argument("--main-csv", type=Path, default=MAIN_CSV)
    parser.add_argument("--image-dir", type=Path, default=IMAGES_MGS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = build_labels(args.mgs_csv, args.main_csv, args.image_dir)
    if args.limit is not None:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)

    train_df, val_df = make_group_split(df, test_size=args.val_size, seed=args.seed)
    _, val_loader = build_dataloaders(
        train_df,
        val_df,
        MouseImageDataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
    )

    result = evaluate_checkpoint(args.checkpoint, val_loader, device=get_device())
    print("Confusion matrix:")
    print(result["confusion_matrix"])
    print("\nClassification report:")
    print(
        classification_report(
            result["y_true"],
            result["y_pred"],
            target_names=list(LABEL_MAPPING.values()),
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
