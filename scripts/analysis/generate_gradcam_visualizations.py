"""Generate Grad-CAM visualizations for the best crop and full-image models."""

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
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = SRC_DIR / "training"
for path in (SRC_DIR, TRAINING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from balanced_device import LABEL_NAMES, build_loaders, build_model, config_from_dict, pil_to_tensor, predict  # noqa: E402
from paths import IMAGENET_MEAN, IMAGENET_STD, REPORTS_DIR, RESULTS_DIR  # noqa: E402
from utils.device import get_device  # noqa: E402

ARTICLE_REPORTS_DIR = REPORTS_DIR / "article_experiments"
GRADCAM_RESULTS_DIR = RESULTS_DIR / "figures" / "gradcam"

BEST_RUNS = {
    "best_full_resnet18": ARTICLE_REPORTS_DIR / "part2" / "resnet18_original_light_lr5e5",
    "best_crop_convnext": ARTICLE_REPORTS_DIR / "part2" / "convnext_tiny_crop_light_dropout0",
}


def label_text(label: int) -> str:
    return LABEL_NAMES[int(label)].replace("_", " ")


def comparison_header(row: pd.Series) -> str:
    return f"{label_text(row['true_label'])} true vs predicted {label_text(row['predicted_label'])}"


def gradcam_title(row: pd.Series) -> str:
    return f"Grad-CAM {label_text(row['predicted_label'])}"


def image_to_tensor(path: Path, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB")
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor = pil_to_tensor(image, image_size, mean, std)
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR), tensor


class GradCAM:
    """Minimal Grad-CAM implementation for torchvision CNN backbones."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, image_tensor: torch.Tensor, class_index: int, device: torch.device) -> tuple[torch.Tensor, float]:
        input_tensor = image_tensor.unsqueeze(0).to(device)
        logits = self.model(input_tensor)
        probability = torch.softmax(logits.detach(), dim=1)[0, 1].item()

        self.model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu()
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam, probability


def target_layer_for_model(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "convnext_tiny":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer configured for {model_name}")


def load_run(run_dir: Path, device: torch.device):
    with (run_dir / "run_config.json").open("r", encoding="utf-8") as file:
        config_dict = json.load(file)
    config = config_from_dict(config_dict)
    model = build_model(config).to(device)
    checkpoint = torch.load(Path(config_dict["checkpoint"]), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return config, config_dict, model


def predictions_for_run(model: torch.nn.Module, split_path: Path, config, device: torch.device) -> pd.DataFrame:
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


def select_examples(predictions: pd.DataFrame, examples_per_group: int) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["is_match"] = predictions["true_label"] == predictions["predicted_label"]
    predictions["confidence"] = predictions["impaired_probability"].where(
        predictions["predicted_label"] == 1,
        1.0 - predictions["impaired_probability"],
    )
    matches = predictions[predictions["is_match"]].sort_values("confidence", ascending=False).head(examples_per_group)
    mismatches = (
        predictions[~predictions["is_match"]].sort_values("confidence", ascending=False).head(examples_per_group)
    )
    return pd.concat([matches, mismatches], ignore_index=True)


def overlay_gradcam(ax, image: Image.Image, cam: torch.Tensor, title: str) -> None:
    ax.imshow(image)
    ax.imshow(cam.numpy(), cmap="jet", alpha=0.42)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_single_gradcam(
    gradcam: GradCAM,
    model: torch.nn.Module,
    row: pd.Series,
    output_path: Path,
    image_size: int,
    device: torch.device,
) -> None:
    del model
    image, tensor = image_to_tensor(Path(row["path"]), image_size)
    cam, _ = gradcam(tensor, int(row["predicted_label"]), device)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title(comparison_header(row), fontsize=9)
    axes[0].axis("off")
    overlay_gradcam(axes[1], image, cam, gradcam_title(row))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_gradcam_panel(
    gradcam: GradCAM,
    examples: pd.DataFrame,
    output_path: Path,
    image_size: int,
    device: torch.device,
    title: str,
) -> None:
    rows = len(examples)
    if rows == 0:
        return

    fig, axes = plt.subplots(rows, 2, figsize=(8, max(3, rows * 2.25)))
    if rows == 1:
        axes = axes.reshape(1, 2)

    for row_idx, (_, row) in enumerate(examples.iterrows()):
        image, tensor = image_to_tensor(Path(row["path"]), image_size)
        cam, _ = gradcam(tensor, int(row["predicted_label"]), device)
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title(comparison_header(row), fontsize=9)
        axes[row_idx, 0].axis("off")
        overlay_gradcam(axes[row_idx, 1], image, cam, gradcam_title(row))

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_for_run(run_key: str, run_dir: Path, examples_per_group: int, output_root: Path) -> None:
    device = get_device()
    config, config_dict, model = load_run(run_dir, device)
    split_path = Path(config_dict["split_csv"])
    predictions_path = run_dir / "test_predictions.csv"
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

    output_dir = output_root / run_key
    output_dir.mkdir(parents=True, exist_ok=True)
    examples[
        [
            "index",
            "subset",
            "true_label",
            "predicted_label",
            "impaired_probability",
            "path",
            "label_name",
            "split",
        ]
    ].to_csv(output_dir / "selected_gradcam_examples.csv", index=False)

    target_layer = target_layer_for_model(model, config.model_name)
    gradcam = GradCAM(model, target_layer)
    try:
        for _, row in examples.iterrows():
            filename = (
                f"{row['index'].replace('.jpg', '')}_true"
                f"{label_text(row['true_label']).replace(' ', '-')}_pred"
                f"{label_text(row['predicted_label']).replace(' ', '-')}_gradcam.png"
            )
            save_single_gradcam(gradcam, model, row, output_dir / filename, config.image_size, device)
        save_gradcam_panel(
            gradcam,
            examples,
            output_dir / "report_gradcam_panel.png",
            config.image_size,
            device,
            title=f"{config.run_name}: Grad-CAM examples",
        )
    finally:
        gradcam.close()

    print(f"Grad-CAM outputs: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples for best crop/full models.")
    parser.add_argument("--examples-per-group", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, default=GRADCAM_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run_key, run_dir in BEST_RUNS.items():
        generate_for_run(run_key, run_dir, args.examples_per_group, args.output_dir)


if __name__ == "__main__":
    main()
