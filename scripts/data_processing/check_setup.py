"""Check environment, CUDA, and dataset paths required for the experiments."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from paths import (
    IMAGES_MGS_DIR,
    IMAGES_PERFECT_DIR,
    IMAGES_PERFECT_ORGANIZED_DIR,
    MAIN_CSV,
    MGS_CSV,
    MOUSE_DATASET_DIR,
)
from utils.device import describe_device, get_device

SUBSETS = ("AW", "JW", "KH", "LW", "MR")
CLASSES = ("impaired", "not_impaired")


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
        "IMAGES_PERFECT_DIR": IMAGES_PERFECT_DIR,
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

    crop_count = len(list(IMAGES_PERFECT_DIR.glob("*.jpg")))
    full_count = len(list(IMAGES_MGS_DIR.glob("*.jpg")))
    print(f"Cropped images (images_perfect): {crop_count}")
    print(f"Full images (images_mgs):        {full_count}")
    if crop_count == 0 or full_count == 0:
        raise RuntimeError("Expected JPG images under images_perfect and images_mgs.")

    sample = next(IMAGES_PERFECT_DIR.glob("*.jpg"))
    with Image.open(sample) as image:
        image = image.convert("RGB")
        print(f"Sample crop: {sample.name} size={image.size}")

    if IMAGES_PERFECT_ORGANIZED_DIR.is_dir():
        bucket_counts = []
        for subset in SUBSETS:
            for class_name in CLASSES:
                folder = IMAGES_PERFECT_ORGANIZED_DIR / subset / class_name
                count = len(list(folder.glob("*.jpg"))) if folder.is_dir() else 0
                bucket_counts.append((f"{subset}/{class_name}", count))
        print("\nOrganized dataset buckets:")
        for name, count in bucket_counts:
            print(f"  {name}: {count}")
        empty = [name for name, count in bucket_counts if count == 0]
        if empty:
            print(
                "\nWarning: empty organized buckets found. "
                "Run scripts/data_processing/prepare_organized_dataset.py"
            )
    else:
        print(
            f"\nOrganized dataset not found at {IMAGES_PERFECT_ORGANIZED_DIR}. "
            "Run scripts/data_processing/prepare_organized_dataset.py before training."
        )

    print("\nOK - setup check passed.")


if __name__ == "__main__":
    main()
