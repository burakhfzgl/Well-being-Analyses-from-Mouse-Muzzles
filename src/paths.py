"""Central project paths."""

from pathlib import Path

# quinyun/ (repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local dataset. The folder contents are ignored by Git.
MOUSE_DATASET_DIR = PROJECT_ROOT / "dataset" / "mouse_dataset"

MGS_CSV = MOUSE_DATASET_DIR / "MouseGrimaceFaces_mgs.csv"
MAIN_CSV = MOUSE_DATASET_DIR / "MouseGrimaceFaces_main.csv"
IMAGES_MGS_DIR = MOUSE_DATASET_DIR / "images_mgs"
IMAGES_PERFECT_DIR = MOUSE_DATASET_DIR / "images_perfect"
MUZZLE_CROPS_DIR = MOUSE_DATASET_DIR / "muzzle_crops"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"

# Backward-compatible alias used in older scripts/notebooks.
mouse_dataset = MOUSE_DATASET_DIR

CLASS_NAMES = ("well-being", "impaired")
LABEL_MAPPING = {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
