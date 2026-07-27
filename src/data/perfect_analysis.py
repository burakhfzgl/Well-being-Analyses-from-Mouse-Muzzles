"""Build binary impairment labels from MGS scores and organize crop folders."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Whisker scores are stored as wc* in the MGS CSV; exposed as mean_ws in outputs.
FAUS = ("nb", "cb", "wc")
FAU_OUTPUT_NAMES = {"nb": "mean_nb", "cb": "mean_cb", "wc": "mean_ws"}
RATERS = ("1", "2", "3", "4", "5", "6", "7", "9", "10", "11", "12")
EXCLUDED_SCORES = {9}


@dataclass(frozen=True)
class PerfectAnalysisSummary:
    """Summary of images_perfect label and folder organization."""

    images_in_folder: int
    rows_in_csv: int
    rows_with_labels: int
    rows_without_scores: int
    not_impaired: int
    impaired: int
    copied_images: int


def parse_score(value) -> int | None:
    """Parse a score cell, ignoring missing values and score 9."""
    if pd.isna(value) or value == "" or value == "-":
        return None
    try:
        score = int(value)
    except (ValueError, TypeError):
        return None
    if score in EXCLUDED_SCORES:
        return None
    return score


def collect_fau_scores(row: pd.Series, fau: str) -> list[int]:
    """Collect valid rater scores for one facial action unit."""
    scores: list[int] = []
    for rater in RATERS:
        column = f"{fau}{rater}"
        if column not in row.index:
            continue
        score = parse_score(row[column])
        if score is not None:
            scores.append(score)
    return scores


def impaired_label_from_general_mean(general_mean: float) -> int:
    """Map pooled mean score to binary label: 0 = not impaired, 1 = impaired."""
    return 1 if general_mean >= 0.5 else 0


def analyze_mgs_row(row: pd.Series) -> dict | None:
    """Build one analysis record from an MGS CSV row."""
    fau_scores: dict[str, list[int]] = {fau: collect_fau_scores(row, fau) for fau in FAUS}
    all_scores = [score for scores in fau_scores.values() for score in scores]
    if not all_scores:
        return None

    record = {
        "index": row["index"],
        "subset": row["subset"],
        "general_mean": sum(all_scores) / len(all_scores),
    }
    for fau in FAUS:
        scores = fau_scores[fau]
        record[FAU_OUTPUT_NAMES[fau]] = sum(scores) / len(scores) if scores else pd.NA

    record["impaired_not_impaired"] = impaired_label_from_general_mean(record["general_mean"])
    return record


def load_mgs_table(mgs_csv: Path | str) -> pd.DataFrame:
    """Load MGS scores from CSV."""
    return pd.read_csv(mgs_csv)


def build_analysis_df(
    mgs_table: pd.DataFrame,
    *,
    include_indices: set[str] | None = None,
) -> pd.DataFrame:
    """Build label analysis rows filtered by image index."""
    rows: list[dict] = []
    for _, row in mgs_table.iterrows():
        index = row["index"]
        if include_indices is not None and index not in include_indices:
            continue
        if pd.isna(row.get("subset")):
            continue

        record = analyze_mgs_row(row)
        if record is not None:
            rows.append(record)

    columns = [
        "index",
        "mean_nb",
        "mean_cb",
        "mean_ws",
        "subset",
        "general_mean",
        "impaired_not_impaired",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_perfect_analysis_df(mgs_csv: Path | str, image_dir: Path | str) -> pd.DataFrame:
    """Build analysis dataframe for cropped images that exist on disk."""
    image_dir = Path(image_dir)
    available = {path.name for path in image_dir.glob("*.jpg")}
    mgs = load_mgs_table(mgs_csv)
    return build_analysis_df(mgs, include_indices=available)


def save_perfect_analysis_csv(
    mgs_csv: Path | str,
    image_dir: Path | str,
    output_csv: Path | str,
) -> tuple[pd.DataFrame, PerfectAnalysisSummary]:
    """Write analysis CSV for images_perfect and return dataframe + summary."""
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    mgs = pd.read_csv(mgs_csv)
    available = sorted(path.name for path in image_dir.glob("*.jpg"))
    df = build_perfect_analysis_df(mgs_csv, image_dir)
    df.to_csv(output_csv, index=False)

    summary = PerfectAnalysisSummary(
        images_in_folder=len(available),
        rows_in_csv=len(mgs),
        rows_with_labels=len(df),
        rows_without_scores=len(available) - len(df),
        not_impaired=int((df["impaired_not_impaired"] == 0).sum()) if not df.empty else 0,
        impaired=int((df["impaired_not_impaired"] == 1).sum()) if not df.empty else 0,
        copied_images=0,
    )
    return df, summary


def organize_perfect_dataset(
    df: pd.DataFrame,
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    clear_existing: bool = True,
) -> PerfectAnalysisSummary:
    """Copy images into subset/impaired|not_impaired folders."""
    output_dir = Path(output_dir)
    source_dir = Path(source_dir)

    if clear_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_source = 0
    for _, row in df.iterrows():
        label = int(row["impaired_not_impaired"])
        folder_name = "impaired" if label == 1 else "not_impaired"
        target_dir = output_dir / str(row["subset"]) / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        source = source_dir / row["index"]
        if source.is_file():
            shutil.copy2(source, target_dir / source.name)
            copied += 1
        else:
            missing_source += 1

    return PerfectAnalysisSummary(
        images_in_folder=len(list(source_dir.glob("*.jpg"))),
        rows_in_csv=0,
        rows_with_labels=len(df),
        rows_without_scores=missing_source,
        not_impaired=int((df["impaired_not_impaired"] == 0).sum()) if not df.empty else 0,
        impaired=int((df["impaired_not_impaired"] == 1).sum()) if not df.empty else 0,
        copied_images=copied,
    )


def run_perfect_analysis(
    mgs_csv: Path | str,
    image_dir: Path | str,
    output_csv: Path | str,
    organized_dir: Path | str,
) -> tuple[pd.DataFrame, PerfectAnalysisSummary]:
    """Build CSV labels and organized folder dataset for images_perfect."""
    df, csv_summary = save_perfect_analysis_csv(mgs_csv, image_dir, output_csv)
    org_summary = organize_perfect_dataset(df, image_dir, organized_dir)

    summary = PerfectAnalysisSummary(
        images_in_folder=csv_summary.images_in_folder,
        rows_in_csv=csv_summary.rows_in_csv,
        rows_with_labels=csv_summary.rows_with_labels,
        rows_without_scores=csv_summary.rows_without_scores,
        not_impaired=org_summary.not_impaired,
        impaired=org_summary.impaired,
        copied_images=org_summary.copied_images,
    )
    return df, summary


def print_perfect_analysis_summary(summary: PerfectAnalysisSummary) -> None:
    """Print a concise CLI summary."""
    print(f"Images in images_perfect:   {summary.images_in_folder}")
    print(f"Labeled rows in CSV:        {summary.rows_with_labels}")
    print(f"Rows without valid scores:  {summary.rows_without_scores}")
    print(f"Not impaired (0):           {summary.not_impaired}")
    print(f"Impaired (1):               {summary.impaired}")
    print(f"Copied to organized set:    {summary.copied_images}")
