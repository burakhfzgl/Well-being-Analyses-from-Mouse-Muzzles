"""Prepare organized crop dataset from MGS labels (optional rebuild step)."""

from __future__ import annotations

from src.paths import CROPPED_IMAGES_DIR, MOUSE_DATASET_DIR, MGS_CSV
from src.preprocess import print_perfect_analysis_summary, run_perfect_analysis

# Only needed when rebuilding Cropped_images from raw flat crops.
IMAGES_PERFECT_DIR = MOUSE_DATASET_DIR / "images_perfect"
LABELS_CSV = MOUSE_DATASET_DIR / "images_perfect_labels.csv"


def run_prepare() -> None:
    """Build labels CSV and subset/class folders under Cropped_images."""
    if not IMAGES_PERFECT_DIR.is_dir():
        raise FileNotFoundError(
            f"Raw crop source not found: {IMAGES_PERFECT_DIR}\n"
            "If you already downloaded Cropped_images, skip prepare and run:\n"
            "  python main.py --mode check"
        )

    df, summary = run_perfect_analysis(
        MGS_CSV,
        IMAGES_PERFECT_DIR,
        LABELS_CSV,
        CROPPED_IMAGES_DIR,
    )

    print(f"Analysis CSV: {LABELS_CSV.resolve()}")
    print(f"Organized dataset: {CROPPED_IMAGES_DIR.resolve()}")
    print()
    print_perfect_analysis_summary(summary)
    print()
    print("Subset breakdown:")
    print(
        df.groupby(["subset", "impaired_not_impaired"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "not_impaired", 1: "impaired"})
        .to_string()
    )
