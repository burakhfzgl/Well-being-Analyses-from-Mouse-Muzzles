"""Generate training/validation epoch curves for the best crop and full-image models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports" / "article_experiments"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagrams" / "training_curves"

BEST_RUNS = {
    "best_full_resnet18": REPORTS_DIR / "part2" / "resnet18_original_light_lr5e5",
    "best_crop_convnext": REPORTS_DIR / "part2" / "convnext_tiny_crop_light_dropout0",
}


def plot_history(history_path: Path, output_path: Path, title: str) -> None:
    history = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    axes[2].plot(history["epoch"], history["train_macro_f1"], label="train")
    axes[2].plot(history["epoch"], history["val_macro_f1"], label="validation")
    axes[2].set_title("Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train/validation curves for best runs.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []

    for run_key, run_dir in BEST_RUNS.items():
        history_path = run_dir / "training_history.csv"
        if not history_path.is_file():
            missing.append(str(history_path))
            continue
        plot_history(
            history_path,
            args.output_dir / f"{run_key}_train_val_curves.png",
            title=f"{run_key.replace('_', ' ')} train/validation curves",
        )

    if missing:
        missing_path = args.output_dir / "missing_training_histories.txt"
        missing_path.write_text(
            "Training history files are required to plot epoch curves. "
            "Rerun the corresponding experiments to recreate them:\n\n"
            + "\n".join(missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Missing training histories listed in: {missing_path.resolve()}")
    else:
        print(f"Training curves saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
