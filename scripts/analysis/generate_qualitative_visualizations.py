"""Generate saliency maps and qualitative examples for completed article runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = SRC_DIR / "training"
for path in (SRC_DIR, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from balanced_device import LABEL_NAMES, build_loaders, build_model, config_from_dict, pil_to_tensor, predict  # noqa: E402
from paths import FIGURES_DIR, IMAGENET_MEAN, IMAGENET_STD, REPORTS_DIR  # noqa: E402
from utils.device import get_device  # noqa: E402

ARTICLE_REPORTS_DIR = REPORTS_DIR / "article_experiments"
ARTICLE_FIGURES_DIR = FIGURES_DIR / "article_experiments"


def label_text(label: int) -> str:
    """Return human-readable class labels for report figures."""
    return LABEL_NAMES[int(label)].replace("_", " ")


def comparison_header(row: pd.Series) -> str:
    """Return a compact true-vs-predicted title for saliency figures."""
    return f"{label_text(row['true_label'])} true vs predicted {label_text(row['predicted_label'])}"


def load_run_config(run_dir: Path) -> dict:
    with (run_dir / "run_config.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def select_examples(predictions: pd.DataFrame, examples_per_group: int) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]
    predictions["confidence"] = predictions["impaired_probability"].where(
        predictions["predicted_label"] == 1,
        1.0 - predictions["impaired_probability"],
    )
    correct = predictions[predictions["correct"]].sort_values("confidence", ascending=False).head(examples_per_group)
    incorrect = predictions[~predictions["correct"]].sort_values("confidence", ascending=False).head(examples_per_group)
    return pd.concat([correct, incorrect], ignore_index=True)


def image_to_tensor(path: Path, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = pil_to_tensor(image, image_size, mean, std)
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR), tensor


def normalize_heatmap(gradient: torch.Tensor) -> torch.Tensor:
    heatmap = gradient.detach().abs().max(dim=1).values.squeeze(0).cpu()
    heatmap = heatmap - heatmap.min()
    return heatmap / heatmap.max().clamp_min(1e-8)


def saliency_for_single(model: torch.nn.Module, image_tensor: torch.Tensor, predicted_label: int, device: torch.device):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)
    logits = model(image_tensor)
    score = logits[0, predicted_label]
    model.zero_grad(set_to_none=True)
    score.backward()
    return normalize_heatmap(image_tensor.grad), torch.softmax(logits.detach(), dim=1)[0, 1].item()


def overlay_axis(ax, image: Image.Image, heatmap: torch.Tensor, title: str) -> None:
    ax.imshow(image)
    ax.imshow(heatmap.numpy(), cmap="magma", alpha=0.45)
    ax.set_title(title)
    ax.axis("off")


def plot_single_example(
    model: torch.nn.Module,
    row: pd.Series,
    image_path: Path,
    output_path: Path,
    image_size: int,
    device: torch.device,
) -> None:
    image, tensor = image_to_tensor(image_path, image_size)
    heatmap, _ = saliency_for_single(model, tensor, int(row["predicted_label"]), device)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")
    overlay_axis(axes[1], image, heatmap, "Gradient saliency")
    fig.suptitle(comparison_header(row))
    fig.tight_layout()
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def predictions_for_run(
    model: torch.nn.Module,
    split_path: Path,
    config,
    device: torch.device,
) -> pd.DataFrame:
    """Recompute test predictions when the saved per-image CSV is not available."""
    split_df = pd.read_csv(split_path)
    loaders = build_loaders(split_df, config)
    test = predict(model, loaders["test"], device, threshold=0.5)
    return pd.DataFrame(
        {
            "index": test["index"],
            "subset": test["subset"],
            "true_label": test["y_true"],
            "predicted_label": test["y_pred"],
            "impaired_probability": test["y_prob"],
        }
    )


def plot_report_panel(
    model: torch.nn.Module,
    examples: pd.DataFrame,
    output_path: Path,
    image_size: int,
    device: torch.device,
    title: str,
) -> None:
    """Create a compact report-ready panel of inputs and saliency overlays."""
    rows = len(examples)
    if rows == 0:
        return
    fig, axes = plt.subplots(rows, 2, figsize=(8, max(3, rows * 2.3)))
    if rows == 1:
        axes = axes.reshape(1, 2)

    for row_idx, (_, row) in enumerate(examples.iterrows()):
        image, tensor = image_to_tensor(Path(row["path"]), image_size)
        heatmap, _ = saliency_for_single(model, tensor, int(row["predicted_label"]), device)

        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(comparison_header(row), fontsize=9)
        axes[row_idx, 0].axis("off")
        overlay_axis(axes[row_idx, 1], image, heatmap, "Gradient saliency")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_for_run(run_dir: Path, examples_per_group: int) -> None:
    config_dict = load_run_config(run_dir)
    config = config_from_dict(config_dict)
    checkpoint = Path(config_dict["checkpoint"])
    predictions_path = run_dir / "test_predictions.csv"
    split_path = Path(config_dict["split_csv"])
    if not checkpoint.is_file() or not split_path.is_file():
        print(f"Skipping incomplete run: {run_dir}")
        return

    device = get_device()
    model = build_model(config).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    if predictions_path.is_file():
        predictions = pd.read_csv(predictions_path)
    else:
        predictions = predictions_for_run(model, split_path, config, device)
        predictions.to_csv(predictions_path, index=False)

    split_df = pd.read_csv(split_path)
    examples = select_examples(predictions, examples_per_group).merge(
        split_df,
        left_on=["index", "subset", "true_label"],
        right_on=["index", "subset", "label"],
        how="left",
        suffixes=("", "_split"),
    )
    output_dir = Path(config_dict["figures_dir"]) / "qualitative_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    export_columns = [
        "index",
        "subset",
        "true_label",
        "predicted_label",
        "impaired_probability",
        "path",
        "label_name",
        "split",
    ]
    examples[[col for col in export_columns if col in examples.columns]].to_csv(
        output_dir / "selected_examples.csv",
        index=False,
    )

    for _, row in examples.iterrows():
        filename = f"{row['index'].replace('.jpg', '')}_true{label_text(row['true_label']).replace(' ', '-')}_pred{label_text(row['predicted_label']).replace(' ', '-')}.png"
        output_path = output_dir / filename
        plot_single_example(model, row, Path(row["path"]), output_path, config.image_size, device)
    plot_report_panel(
        model,
        examples,
        output_dir / "report_saliency_panel.png",
        config.image_size,
        device,
        title=f"{config.run_name}: qualitative saliency examples",
    )
    print(f"Qualitative examples: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate saliency and qualitative examples for completed runs.")
    parser.add_argument("--part", choices=["part1", "part2", "all"], default="all")
    parser.add_argument("--only", nargs="*", help="Run names to visualize.")
    parser.add_argument("--examples-per-group", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parts = ["part1", "part2"] if args.part == "all" else [args.part]
    only = set(args.only) if args.only else None
    run_dirs: list[Path] = []
    for part in parts:
        part_dir = ARTICLE_REPORTS_DIR / part
        if not part_dir.is_dir():
            continue
        for run_dir in sorted(path for path in part_dir.iterdir() if path.is_dir()):
            if only and run_dir.name not in only:
                continue
            run_dirs.append(run_dir)
    if not run_dirs:
        print("No completed runs found for qualitative visualization.")
        return
    for run_dir in run_dirs:
        generate_for_run(run_dir, args.examples_per_group)


if __name__ == "__main__":
    main()
