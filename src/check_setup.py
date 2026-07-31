"""Environment and dataset setup checks."""

from __future__ import annotations

import sys

import torch
from PIL import Image

from src.paths import (
    IMAGES_MGS_DIR,
    IMAGES_PERFECT_DIR,
    CROPPED_IMAGES_DIR,
    MAIN_CSV,
    MGS_CSV,
    MOUSE_DATASET_DIR,
    PROJECT_ROOT,
)
from src.device import describe_device, get_device

SUBSETS = ("AW", "JW", "KH", "LW", "MR")
CLASSES = ("impaired", "not_impaired")


def run_check() -> None:
    """Validate Python/PyTorch, CUDA, and required dataset paths."""
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

    if CROPPED_IMAGES_DIR.is_dir():
        bucket_counts = []
        for subset in SUBSETS:
            for class_name in CLASSES:
                folder = CROPPED_IMAGES_DIR / subset / class_name
                count = len(list(folder.glob("*.jpg"))) if folder.is_dir() else 0
                bucket_counts.append((f"{subset}/{class_name}", count))
        print("\nCropped dataset buckets:")
        for name, count in bucket_counts:
            print(f"  {name}: {count}")
        empty = [name for name, count in bucket_counts if count == 0]
        if empty:
            print("\nWarning: empty cropped buckets found. Run: python main.py --mode prepare")
    else:
        print(
            f"\nCropped dataset not found at {CROPPED_IMAGES_DIR}. "
            "Run: python main.py --mode prepare"
        )

    print("\nOK - setup check passed.")
