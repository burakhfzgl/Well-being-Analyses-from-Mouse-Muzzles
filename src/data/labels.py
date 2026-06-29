"""Build training labels from Mouse Grimace Scale CSV files."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VALID_SCORES = {0, 1, 2}
FAUS = ("nb", "cb", "wc")
RATERS = ("1", "2", "3", "4", "5", "6", "7", "9", "10", "11", "12")
LABEL_MAPPING = {0: "well-being", 1: "impaired"}


@dataclass(frozen=True)
class LabelBuildSummary:
    """Summary of label construction and image filtering."""

    input_rows: int
    output_rows: int
    missing_images: int
    incomplete_labels: int


def round_half_up(value: float) -> int:
    """Round halves upward, matching the project label rule."""
    return int(np.floor(float(value) + 0.5))


def parse_score(value) -> int | None:
    """Parse a valid MGS score cell."""
    if pd.isna(value) or value == "" or value == "-":
        return None
    try:
        score = int(value)
    except (ValueError, TypeError):
        return None
    return score if score in VALID_SCORES else None


def fau_averages(row: pd.Series) -> dict[str, float]:
    """Return average valid score per facial action unit."""
    averages: dict[str, float] = {}
    for fau in FAUS:
        scores: list[int] = []
        for rater in RATERS:
            column = f"{fau}{rater}"
            if column not in row.index:
                continue
            score = parse_score(row[column])
            if score is not None:
                scores.append(score)
        if scores:
            averages[fau] = sum(scores) / len(scores)
    return averages


def welfare_label_from_mgs_row(row: pd.Series) -> int | None:
    """Return binary welfare label, or None when required FAU scores are missing."""
    averages = fau_averages(row)
    if len(averages) < len(FAUS):
        return None
    rounded = [round_half_up(averages[fau]) for fau in FAUS]
    average_decision = sum(rounded) / len(rounded)
    return 1 if round_half_up(average_decision) >= 1 else 0


def build_labels(
    mgs_csv: Path | str,
    main_csv: Path | str,
    image_dir: Path | str,
    *,
    require_image: bool = True,
    drop_incomplete_labels: bool = True,
    return_summary: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, LabelBuildSummary]:
    """Build a dataframe with columns: index, path, label, subset, id."""
    mgs = pd.read_csv(mgs_csv)
    main = pd.read_csv(main_csv)
    image_dir = Path(image_dir)

    merged = mgs.merge(
        main[["index", "id", "subset"]],
        on="index",
        how="left",
        suffixes=("_mgs", "_main"),
    )
    if "subset_mgs" in merged.columns:
        merged["subset"] = merged["subset_main"].fillna(merged["subset_mgs"])
    elif "subset" not in merged.columns:
        merged["subset"] = merged.get("subset_main", merged.get("subset_mgs"))

    rows: list[dict] = []
    missing_images = 0
    incomplete_labels = 0

    for _, row in merged.iterrows():
        label = welfare_label_from_mgs_row(row)
        if label is None:
            incomplete_labels += 1
            if drop_incomplete_labels:
                continue
            label = 0

        image_path = image_dir / row["index"]
        if require_image and not image_path.is_file():
            missing_images += 1
            continue

        rows.append(
            {
                "index": row["index"],
                "path": str(image_path),
                "label": int(label),
                "subset": row["subset"],
                "id": row["id"],
            }
        )

    df = pd.DataFrame(rows)
    summary = LabelBuildSummary(
        input_rows=len(merged),
        output_rows=len(df),
        missing_images=missing_images,
        incomplete_labels=incomplete_labels,
    )

    if return_summary:
        return df, summary
    return df


def print_label_summary(df: pd.DataFrame, summary: LabelBuildSummary) -> None:
    """Print a concise label-building summary for CLI scripts."""
    print(f"Rows in MGS CSV:            {summary.input_rows}")
    print(f"Rows after filtering:       {summary.output_rows}")
    print(f"Missing image files:        {summary.missing_images}")
    print(f"Incomplete label rows:      {summary.incomplete_labels}")
    if not df.empty:
        print("\nClass counts:")
        print(df["label"].map(LABEL_MAPPING).value_counts().to_string())


def copy_impaired_images(df: pd.DataFrame, output_dir: Path | str) -> int:
    """Copy rows with label 1 into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, row in df[df["label"] == 1].iterrows():
        source = Path(row["path"])
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)
            copied += 1
    return copied
