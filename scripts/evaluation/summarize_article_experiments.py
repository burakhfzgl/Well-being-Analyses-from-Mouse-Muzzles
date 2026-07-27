"""Summarize article experiment outputs into Part 1 / Part 2 tables and figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import FIGURES_DIR, REPORTS_DIR, RESULTS_DIR  # noqa: E402

ARTICLE_REPORTS_DIR = REPORTS_DIR / "article_experiments"
ARTICLE_FIGURES_DIR = FIGURES_DIR / "article_experiments"


def load_run_rows(reports_dir: Path) -> pd.DataFrame:
    """Load one summary row per completed run."""
    rows: list[dict] = []
    for summary_path in sorted(reports_dir.rglob("test_summary.json")):
        run_dir = summary_path.parent
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        config_path = run_dir / "run_config.json"
        config = {}
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
        row = {**config, **summary}
        row["run_dir"] = str(run_dir)
        row["run_name"] = row.get("run_name", run_dir.name)
        row["part"] = row.get("part") or run_dir.parent.name
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def format_model_name(name: str) -> str:
    labels = {
        "resnet18": "ResNet18",
        "convnext_tiny": "ConvNeXt-Tiny",
    }
    return labels.get(str(name), str(name))


def format_lr(value: float) -> str:
    text = f"{float(value):.0e}".replace("E", "e")
    return text.replace("e-0", "e-")


def save_part_results_tables(df: pd.DataFrame, reports_dir: Path, results_dir: Path = RESULTS_DIR) -> None:
    """Save compact Part 1 and Part 2 result tables for article reporting."""
    if df.empty:
        return

    part1 = df[df["part"] == "part1"].sort_values("test_macro_f1", ascending=False)
    if not part1.empty:
        table = pd.DataFrame(
            {
                "run_name": part1["run_name"],
                "model": part1["model_name"].map(format_model_name),
                "input": part1["source_mode"].replace({"original": "full"}),
                "augmentation": part1["augmentation"].map({True: "light", False: "none"}),
                "accuracy": part1["test_accuracy"].round(3),
                "recall": part1["test_recall_macro"].round(3),
                "macro_f1": part1["test_macro_f1"].round(3),
                "roc_auc": part1["test_roc_auc"].round(3),
            }
        )
        table.to_csv(reports_dir / "part1_results.csv", index=False)
        results_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(results_dir / "part1_results.csv", index=False)
        part_dir = reports_dir / "part1"
        part_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(part_dir / "part1_results.csv", index=False)

    part2 = df[df["part"] == "part2"].sort_values("test_macro_f1", ascending=False)
    if not part2.empty:
        table = pd.DataFrame(
            {
                "run_name": part2["run_name"],
                "model": part2["model_name"].map(format_model_name),
                "input": part2["source_mode"].replace({"original": "full"}),
                "lr": part2["lr"].map(format_lr),
                "dropout": part2["dropout"],
                "frozen_backbone": part2["freeze_backbone"].map({True: "yes", False: "no"}),
                "accuracy": part2["test_accuracy"].round(3),
                "recall": part2["test_recall_macro"].round(3),
                "macro_f1": part2["test_macro_f1"].round(3),
                "roc_auc": part2["test_roc_auc"].round(3),
            }
        )
        table.to_csv(reports_dir / "part2_results.csv", index=False)
        results_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(results_dir / "part2_results.csv", index=False)
        part_dir = reports_dir / "part2"
        part_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(part_dir / "part2_results.csv", index=False)


def plot_metric_bars(df: pd.DataFrame, figures_dir: Path) -> None:
    if df.empty:
        return
    ranked = df.sort_values("test_macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(ranked) * 0.45)))
    ax.barh(ranked["run_name"], ranked["test_macro_f1"], label="Macro-F1")
    ax.barh(ranked["run_name"], ranked["test_roc_auc"], alpha=0.45, label="ROC-AUC")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title("Held-Out Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "model_comparison_macro_f1_roc_auc.png", dpi=300)
    plt.close(fig)


def plot_augmentation_ablation(df: pd.DataFrame, figures_dir: Path) -> None:
    candidates = df[df["run_name"].str.contains("resnet18_crop", na=False)].copy()
    if candidates.empty:
        return
    candidates = candidates.sort_values("run_name")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(candidates["run_name"], candidates["test_macro_f1"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Augmentation Ablation")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figures_dir / "augmentation_ablation_macro_f1.png", dpi=300)
    plt.close(fig)


def plot_crop_original_ablation(df: pd.DataFrame, figures_dir: Path) -> None:
    candidates = df[df["run_name"].isin(["resnet18_crop_light", "resnet18_original_light"])].copy()
    if len(candidates) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(candidates["source_mode"], candidates["test_macro_f1"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Input")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Crop vs Original Ablation")
    fig.tight_layout()
    fig.savefig(figures_dir / "crop_vs_original_macro_f1.png", dpi=300)
    plt.close(fig)


def plot_part1_heatmap(df: pd.DataFrame, figures_dir: Path) -> None:
    """Save a compact model/input/augmentation heatmap for Part 1."""
    part1 = df[df["part"] == "part1"].copy()
    if part1.empty:
        return
    part1["setting"] = part1["source_mode"].replace({"original": "full"}) + " / " + part1["augmentation"].map(
        {True: "light aug", False: "no aug"}
    )
    pivot = part1.pivot_table(index="model_name", columns="setting", values="test_macro_f1", aggfunc="max")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns) * 1.8), max(3, len(pivot) * 0.8)))
    image = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Part 1 Ablation: Macro-F1")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            if pd.notna(value):
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="Macro-F1")
    fig.tight_layout()
    fig.savefig(figures_dir / "part1_model_input_augmentation_heatmap.png", dpi=300)
    plt.close(fig)


def plot_part_metric_bars(df: pd.DataFrame, figures_dir: Path, part: str) -> None:
    """Save ranked Macro-F1 bars for one experiment part."""
    part_df = df[df["part"] == part].copy()
    if part_df.empty:
        return
    part_df = part_df.sort_values("test_macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(part_df) * 0.45)))
    ax.barh(part_df["run_name"], part_df["test_macro_f1"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1")
    ax.set_title(f"{part.upper()} Ranked Runs")
    fig.tight_layout()
    fig.savefig(figures_dir / f"{part}_ranked_macro_f1.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize article-grade experiment outputs.")
    parser.add_argument("--reports-dir", type=Path, default=ARTICLE_REPORTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=ARTICLE_FIGURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    df = load_run_rows(args.reports_dir)
    if df.empty:
        print(f"No completed runs found under {args.reports_dir}")
        return

    save_part_results_tables(df, args.reports_dir)
    plot_metric_bars(df, args.figures_dir)
    plot_augmentation_ablation(df, args.figures_dir)
    plot_crop_original_ablation(df, args.figures_dir)
    plot_part1_heatmap(df, args.figures_dir)
    for part in sorted(df["part"].dropna().unique()):
        part_figures_dir = args.figures_dir / str(part)
        part_figures_dir.mkdir(parents=True, exist_ok=True)
        plot_metric_bars(df[df["part"] == part], part_figures_dir)
        plot_part_metric_bars(df, part_figures_dir, str(part))
    print(f"Summarized {len(df)} completed runs.")
    print(f"Tables:  {args.reports_dir.resolve()}")
    print(f"Figures: {args.figures_dir.resolve()}")


if __name__ == "__main__":
    main()
