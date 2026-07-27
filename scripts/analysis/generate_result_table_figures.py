"""Generate report-ready table figures for Part 1 and Part 2 results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diagrams" / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures" / "tables"


def format_score(value: float) -> str:
    return f"{float(value):.3f}"


def draw_table(data: pd.DataFrame, output_path: Path, title: str) -> None:
    """Draw a styled table matching the report/presentation style."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(data)
    fig_height = max(3.4, rows * 0.42 + 1.15)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=data.values,
        colLabels=data.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.65)

    header_color = "#b5161b"
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#222222")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f2f2f2" if row % 2 == 0 else "white")

    # Emphasize the best row's key metrics.
    metric_columns = {"Accuracy", "Recall", "Macro F1", "ROC AUC"}
    for col_idx, name in enumerate(data.columns):
        if name in metric_columns:
            table[(1, col_idx)].get_text().set_weight("bold")
            table[(1, col_idx)].get_text().set_color("#b5161b")

    ax.set_title(title, fontsize=15, weight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def part1_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "part1_results.csv")
    return pd.DataFrame(
        {
            "Model": df["model"],
            "Input": df["input"],
            "Aug.": df["augmentation"],
            "Accuracy": df["accuracy"].map(format_score),
            "Recall": df["recall"].map(format_score),
            "Macro F1": df["macro_f1"].map(format_score),
            "ROC AUC": df["roc_auc"].map(format_score),
        }
    )


def part2_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "part2_results.csv")
    return pd.DataFrame(
        {
            "Model": df["model"],
            "Input": df["input"],
            "LR": df["lr"],
            "Dropout": df["dropout"].map(lambda value: f"{float(value):.1f}"),
            "Frozen": df["frozen_backbone"],
            "Accuracy": df["accuracy"].map(format_score),
            "Recall": df["recall"].map(format_score),
            "Macro F1": df["macro_f1"].map(format_score),
            "ROC AUC": df["roc_auc"].map(format_score),
        }
    )


def part1_model_metrics_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "part1_results.csv")
    return pd.DataFrame(
        {
            "Model": df["model"],
            "Variant": df["input"] + ", " + df["augmentation"] + " aug",
            "Accuracy": df["accuracy"].map(format_score),
            "Recall": df["recall"].map(format_score),
            "Macro F1": df["macro_f1"].map(format_score),
            "ROC AUC": df["roc_auc"].map(format_score),
        }
    )


def part2_model_metrics_table() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "part2_results.csv")
    variant = (
        df["input"]
        + ", lr="
        + df["lr"].astype(str)
        + ", dropout="
        + df["dropout"].map(lambda value: f"{float(value):.1f}")
        + ", frozen="
        + df["frozen_backbone"].astype(str)
    )
    return pd.DataFrame(
        {
            "Model": df["model"],
            "Variant": variant,
            "Accuracy": df["accuracy"].map(format_score),
            "Recall": df["recall"].map(format_score),
            "Macro F1": df["macro_f1"].map(format_score),
            "ROC AUC": df["roc_auc"].map(format_score),
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tables = [
        ("part1_results_table.png", part1_table(), "Part 1 Results"),
        ("part2_results_table.png", part2_table(), "Part 2 Results"),
        ("part1_model_metrics_table.png", part1_model_metrics_table(), "Part 1 Model Metrics"),
        ("part2_model_metrics_table.png", part2_model_metrics_table(), "Part 2 Model Metrics"),
    ]
    for filename, table_data, title in tables:
        csv_filename = filename.replace(".png", ".csv")
        table_data.to_csv(OUTPUT_DIR / csv_filename, index=False)
        table_data.to_csv(RESULTS_FIGURES_DIR / csv_filename, index=False)
        draw_table(table_data, OUTPUT_DIR / filename, title)
        draw_table(table_data, RESULTS_FIGURES_DIR / filename, title)
        print(f"Saved {OUTPUT_DIR / csv_filename}")
        print(f"Saved {RESULTS_FIGURES_DIR / csv_filename}")
        print(f"Saved {OUTPUT_DIR / filename}")
        print(f"Saved {RESULTS_FIGURES_DIR / filename}")


if __name__ == "__main__":
    main()
