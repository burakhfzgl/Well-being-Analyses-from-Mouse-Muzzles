"""Central project paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local dataset (contents ignored by Git; see dataset/README.md).
MOUSE_DATASET_DIR = PROJECT_ROOT / "dataset" / "mouse_dataset"

MGS_CSV = MOUSE_DATASET_DIR / "MouseGrimaceFaces_mgs.csv"
MAIN_CSV = MOUSE_DATASET_DIR / "MouseGrimaceFaces_main.csv"
IMAGES_MGS_DIR = MOUSE_DATASET_DIR / "images_mgs"
IMAGES_PERFECT_DIR = MOUSE_DATASET_DIR / "images_perfect"
IMAGES_PERFECT_LABELS_CSV = MOUSE_DATASET_DIR / "images_perfect_labels.csv"
IMAGES_PERFECT_ORGANIZED_DIR = MOUSE_DATASET_DIR / "images_perfect_organized"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"

CLASS_NAMES = ("not_impaired", "impaired")
LABEL_MAPPING = {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
