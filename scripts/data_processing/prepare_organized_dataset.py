"""Analyze images_perfect scores and build a subset/class-labeled folder dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.perfect_analysis import print_perfect_analysis_summary, run_perfect_analysis  # noqa: E402
from paths import IMAGES_PERFECT_DIR, IMAGES_PERFECT_LABELS_CSV, IMAGES_PERFECT_ORGANIZED_DIR, MGS_CSV  # noqa: E402


def main() -> None:
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


if __name__ == "__main__":
    main()
