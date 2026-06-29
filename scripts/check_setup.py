"""Check environment, CUDA, dataset paths, labels, and image loading."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from data.dataset import MouseImageDataset
from data.labels import build_labels, print_label_summary
from paths import IMAGES_MGS_DIR, MAIN_CSV, MGS_CSV, MOUSE_DATASET_DIR
from utils.device import describe_device, get_device


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Device:", describe_device(get_device()))

    required_paths = {
        "MOUSE_DATASET_DIR": MOUSE_DATASET_DIR,
        "MGS_CSV": MGS_CSV,
        "MAIN_CSV": MAIN_CSV,
        "IMAGES_MGS_DIR": IMAGES_MGS_DIR,
    }
    for name, path in required_paths.items():
        print(f"{name}: {path} exists={path.exists()}")

    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset paths: "
            + ", ".join(missing)
            + ". See dataset/README.md."
        )

    df, summary = build_labels(MGS_CSV, MAIN_CSV, IMAGES_MGS_DIR, return_summary=True)
    print()
    print_label_summary(df, summary)
    if df.empty:
        raise RuntimeError("No labeled images were found.")

    dataset = MouseImageDataset(df.head(1))
    image, label = dataset[0]
    print("\nSample loaded:")
    print("image type:", type(image).__name__)
    print("label:", int(label))
    print("\nOK - setup check passed.")


if __name__ == "__main__":
    main()
