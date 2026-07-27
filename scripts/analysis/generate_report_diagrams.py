"""Generate report diagrams from the final Part 1 and Part 2 result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DIAGRAMS_DIR = PROJECT_ROOT / "outputs" / "diagrams"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports" / "article_experiments"

METRICS = ["accuracy", "recall", "macro_f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "recall": "Recall",
    "macro_f1": "Macro-F1",
    "roc_auc": "ROC-AUC",
}
BEST_RUN_DIRS = {
    "best_full_resnet18": REPORTS_DIR / "part2" / "resnet18_original_light_lr5e5",
    "best_crop_convnext": REPORTS_DIR / "part2" / "convnext_tiny_crop_light_dropout0",
}
CLASS_LABELS = ["not impaired", "impaired"]


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load and combine the final Part 1 and Part 2 result tables."""
    part1 = pd.read_csv(results_dir / "part1_results.csv")
    part1["part"] = "Part 1"
    part1["setting"] = part1["augmentation"].map(lambda value: f"aug={value}")

    part2 = pd.read_csv(results_dir / "part2_results.csv")
    part2["part"] = "Part 2"
    part2["setting"] = (
        "lr="
        + part2["lr"].astype(str)
        + ", dropout="
        + part2["dropout"].astype(str)
        + ", frozen="
        + part2["frozen_backbone"].astype(str)
    )

    combined = pd.concat([part1, part2], ignore_index=True, sort=False)
    combined["display_name"] = combined["run_name"].str.replace("_", "\n")
    return combined.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def save_metric_table(df: pd.DataFrame, diagrams_dir: Path) -> None:
    """Save the combined metric table used by all diagrams."""
    columns = ["part", "run_name", "model", "input", "accuracy", "recall", "macro_f1", "roc_auc"]
    df[columns].to_csv(diagrams_dir / "combined_metrics_for_diagrams.csv", index=False)


def plot_metric_bars(df: pd.DataFrame, diagrams_dir: Path) -> None:
    """Plot all four key metrics for every run."""
    plot_df = df.sort_values("macro_f1", ascending=True)
    y = range(len(plot_df))

    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.38)))
    offsets = [-0.27, -0.09, 0.09, 0.27]
    height = 0.16
    for metric, offset in zip(METRICS, offsets, strict=True):
        ax.barh([idx + offset for idx in y], plot_df[metric], height=height, label=METRIC_LABELS[metric])

    ax.set_yticks(list(y), plot_df["run_name"])
    ax.set_xlim(0.70, 0.92)
    ax.set_xlabel("Score")
    ax.set_title("Accuracy, Recall, Macro-F1, and ROC-AUC by Run")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(diagrams_dir / "all_runs_metric_comparison.png", dpi=300)
    plt.close(fig)


def plot_ranked_metric(df: pd.DataFrame, diagrams_dir: Path, metric: str) -> None:
    """Plot one ranked metric for all runs."""
    plot_df = df.sort_values(metric, ascending=True)
    colors = plot_df["part"].map({"Part 1": "#4C78A8", "Part 2": "#F58518"})

    fig, ax = plt.subplots(figsize=(9, max(5, len(plot_df) * 0.35)))
    ax.barh(plot_df["run_name"], plot_df[metric], color=colors)
    ax.set_xlim(0.70, 0.92)
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_title(f"Ranked {METRIC_LABELS[metric]} Across Part 1 and Part 2")
    for idx, value in enumerate(plot_df[metric]):
        ax.text(value + 0.002, idx, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(diagrams_dir / f"ranked_{metric}.png", dpi=300)
    plt.close(fig)


def plot_metric_heatmap(df: pd.DataFrame, diagrams_dir: Path) -> None:
    """Plot a metric heatmap for compact comparison."""
    heatmap = df.set_index("run_name")[METRICS]
    fig, ax = plt.subplots(figsize=(8, max(5, len(heatmap) * 0.35)))
    image = ax.imshow(heatmap.to_numpy(), vmin=0.70, vmax=0.92, cmap="viridis")
    ax.set_xticks(range(len(METRICS)), [METRIC_LABELS[m] for m in METRICS], rotation=25, ha="right")
    ax.set_yticks(range(len(heatmap.index)), heatmap.index)
    ax.set_title("Metric Heatmap for Final Part 1 and Part 2 Runs")
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            ax.text(col, row, f"{heatmap.iloc[row, col]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(diagrams_dir / "metrics_heatmap.png", dpi=300)
    plt.close(fig)


def plot_correlation_matrix(df: pd.DataFrame, diagrams_dir: Path) -> None:
    """Plot correlation matrix between the key metrics."""
    corr = df[METRICS].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    labels = [METRIC_LABELS[m] for m in METRICS]
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Correlation Matrix of Evaluation Metrics")
    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(diagrams_dir / "metric_correlation_matrix.png", dpi=300)
    corr.to_csv(diagrams_dir / "metric_correlation_matrix.csv")
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, diagrams_dir: Path) -> None:
    """Plot Macro-F1 against ROC-AUC with model/input annotations."""
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"crop": "o", "full": "s"}
    colors = {"ResNet18": "#4C78A8", "ConvNeXt-Tiny": "#F58518"}
    for _, row in df.iterrows():
        ax.scatter(
            row["macro_f1"],
            row["roc_auc"],
            marker=markers.get(row["input"], "o"),
            color=colors.get(row["model"], "#666666"),
            s=80,
            alpha=0.85,
        )
        ax.text(row["macro_f1"] + 0.001, row["roc_auc"] + 0.001, row["run_name"], fontsize=7)

    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Macro-F1 vs ROC-AUC")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(diagrams_dir / "macro_f1_vs_roc_auc.png", dpi=300)
    plt.close(fig)


def plot_input_summary(df: pd.DataFrame, diagrams_dir: Path) -> None:
    """Summarize mean metric values for crop and full-image settings."""
    summary = df.groupby("input", as_index=True)[METRICS].mean().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary.rename(columns=METRIC_LABELS).plot(kind="bar", ax=ax)
    ax.set_ylim(0.70, 0.90)
    ax.set_xlabel("Input type")
    ax.set_ylabel("Mean score")
    ax.set_title("Mean Metrics by Input Type")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(diagrams_dir / "mean_metrics_by_input_type.png", dpi=300)
    summary.to_csv(diagrams_dir / "mean_metrics_by_input_type.csv")
    plt.close(fig)


def load_prediction_table(run_dir: Path) -> pd.DataFrame:
    predictions_path = run_dir / "test_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing prediction table: {predictions_path}")
    return pd.read_csv(predictions_path)


def plot_confusion_matrix_for_run(predictions: pd.DataFrame, diagrams_dir: Path, run_key: str) -> None:
    matrix = confusion_matrix(predictions["true_label"], predictions["predicted_label"], labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_LABELS)
    ax.set_yticks([0, 1], CLASS_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix: {run_key.replace('_', ' ')}")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(image, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(diagrams_dir / f"{run_key}_confusion_matrix.png", dpi=300)
    pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS).to_csv(
        diagrams_dir / f"{run_key}_confusion_matrix.csv"
    )
    plt.close(fig)


def plot_probability_histogram(predictions: pd.DataFrame, diagrams_dir: Path, run_key: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, label_name, color in [(0, "not impaired true", "#4C78A8"), (1, "impaired true", "#F58518")]:
        values = predictions.loc[predictions["true_label"] == label, "impaired_probability"]
        ax.hist(values, bins=20, alpha=0.65, label=label_name, color=color)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="decision threshold")
    ax.set_xlabel("Predicted impaired probability")
    ax.set_ylabel("Number of test images")
    ax.set_title(f"Prediction Probability Distribution: {run_key.replace('_', ' ')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(diagrams_dir / f"{run_key}_probability_histogram.png", dpi=300)
    plt.close(fig)


def plot_threshold_curve(predictions: pd.DataFrame, diagrams_dir: Path, run_key: str) -> None:
    y_true = predictions["true_label"].to_numpy()
    y_prob = predictions["impaired_probability"].to_numpy()
    rows = []
    for threshold in [value / 100 for value in range(5, 96, 5)]:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }
        )
    curve = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric in ["accuracy", "recall", "macro_f1"]:
        ax.plot(curve["threshold"], curve[metric], marker="o", label=METRIC_LABELS.get(metric, metric))
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="used threshold")
    ax.set_xlabel("Decision threshold for impaired class")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Threshold Sensitivity: {run_key.replace('_', ' ')}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(diagrams_dir / f"{run_key}_threshold_curve.png", dpi=300)
    curve.to_csv(diagrams_dir / f"{run_key}_threshold_curve.csv", index=False)
    plt.close(fig)


def plot_class_recall(predictions: pd.DataFrame, diagrams_dir: Path, run_key: str) -> None:
    recalls = recall_score(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(CLASS_LABELS, recalls, color=["#4C78A8", "#F58518"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title(f"Per-Class Recall: {run_key.replace('_', ' ')}")
    for idx, value in enumerate(recalls):
        ax.text(idx, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(diagrams_dir / f"{run_key}_per_class_recall.png", dpi=300)
    pd.DataFrame({"class": CLASS_LABELS, "recall": recalls}).to_csv(
        diagrams_dir / f"{run_key}_per_class_recall.csv",
        index=False,
    )
    plt.close(fig)


def plot_best_run_diagnostics(diagrams_dir: Path) -> None:
    for run_key, run_dir in BEST_RUN_DIRS.items():
        predictions = load_prediction_table(run_dir)
        plot_confusion_matrix_for_run(predictions, diagrams_dir, run_key)
        plot_probability_histogram(predictions, diagrams_dir, run_key)
        plot_threshold_curve(predictions, diagrams_dir, run_key)
        plot_class_recall(predictions, diagrams_dir, run_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report diagrams from final result tables.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--diagrams-dir", type=Path, default=DIAGRAMS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.diagrams_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.results_dir)
    save_metric_table(df, args.diagrams_dir)
    plot_metric_bars(df, args.diagrams_dir)
    plot_metric_heatmap(df, args.diagrams_dir)
    plot_correlation_matrix(df, args.diagrams_dir)
    plot_scatter(df, args.diagrams_dir)
    plot_input_summary(df, args.diagrams_dir)
    plot_best_run_diagnostics(args.diagrams_dir)
    for metric in METRICS:
        plot_ranked_metric(df, args.diagrams_dir, metric)
    print(f"Diagrams saved to: {args.diagrams_dir.resolve()}")


if __name__ == "__main__":
    main()
