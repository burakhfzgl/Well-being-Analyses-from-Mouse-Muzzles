"""Prepare organized crop dataset from MGS labels."""

from __future__ import annotations

from src.preprocess import print_perfect_analysis_summary, run_perfect_analysis
from src.paths import (
    IMAGES_PERFECT_DIR,
    IMAGES_PERFECT_LABELS_CSV,
    IMAGES_PERFECT_ORGANIZED_DIR,
    MGS_CSV,
)


def run_prepare() -> None:
    """Build labels CSV and subset/class organized folders."""
    df, summary = run_perfect_analysis(
        MGS_CSV,
        IMAGES_PERFECT_DIR,
        IMAGES_PERFECT_LABELS_CSV,
        IMAGES_PERFECT_ORGANIZED_DIR,
    )

    print(f"Analysis CSV: {IMAGES_PERFECT_LABELS_CSV.resolve()}")
    print(f"Organized dataset: {IMAGES_PERFECT_ORGANIZED_DIR.resolve()}")
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
