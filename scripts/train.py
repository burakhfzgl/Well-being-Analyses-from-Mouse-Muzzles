"""Train the binary Mouse Grimace image classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset import MouseImageDataset
from data.labels import build_labels, print_label_summary
from paths import IMAGES_MGS_DIR, MAIN_CSV, MGS_CSV, MODELS_DIR
from training.engine import TrainConfig, run_training
from training.split import make_group_split
from utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgs-csv", type=Path, default=MGS_CSV)
    parser.add_argument("--main-csv", type=Path, default=MAIN_CSV)
    parser.add_argument("--image-dir", type=Path, default=IMAGES_MGS_DIR)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "best_model.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    df, summary = build_labels(
        args.mgs_csv,
        args.main_csv,
        args.image_dir,
        return_summary=True,
    )
    print_label_summary(df, summary)

    if args.limit is not None:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
        print(f"\nLimited training dataframe to {len(df)} rows.")

    train_df, val_df = make_group_split(df, test_size=args.val_size, seed=args.seed)
    print(f"\nTrain rows: {len(train_df)}")
    print(f"Val rows:   {len(val_df)}")

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        seed=args.seed,
        class_weights=args.class_weights,
    )
    run_training(
        train_df,
        val_df,
        MouseImageDataset,
        save_path=args.output,
        config=config,
    )
    print(f"\nTraining complete. Best checkpoint: {args.output}")


if __name__ == "__main__":
    main()
