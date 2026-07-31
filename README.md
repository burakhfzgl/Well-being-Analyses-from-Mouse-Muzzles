# Mouse Muzzle Impairment Classification

Deep learning ablation study for binary mouse welfare classification from facial images. Models predict **impaired** vs **not impaired** using Mouse Grimace Scale (MGS) scores as labels, comparing cropped muzzle images against full-face images and testing how architecture, augmentation, and training choices affect performance.

## Motivation

Automated scoring of mouse facial expressions can support more consistent welfare assessment in laboratory settings. This repository asks a practical question:

> How well can modern CNNs classify impairment from mouse face images, and which input type and training choices matter most?

The study evaluates:

- **Input type:** cropped muzzle images vs full images
- **Architecture:** ResNet18 vs ConvNeXt-Tiny
- **Augmentation:** none vs light spatial/photometric augmentation
- **Tuning:** learning rate, dropout, and backbone freezing on the strongest baselines

## Key Results

Held-out test metrics from the final article tables (`results/part1_results.csv`, `results/part2_results.csv`).

### Best models

| Setting | Run | Macro-F1 | Accuracy | Recall | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Best overall | ResNet18, full image, light aug, lr=`5e-5` | **0.838** | 0.838 | 0.838 | 0.883 |
| Best crop-only | ConvNeXt-Tiny, crop, light aug, dropout=`0` | **0.812** | 0.812 | 0.812 | 0.857 |
| Best Part 1 baseline | ResNet18, full image, light aug | **0.830** | 0.831 | 0.830 | 0.890 |

### Main findings

- **Full images outperform crops for ResNet18.** The strongest overall run uses full-face inputs with light augmentation.
- **Light augmentation helps.** Especially for ResNet18 on full images (Macro-F1 `0.830` with light aug vs `0.773` without).
- **ConvNeXt-Tiny is strongest on crops.** Its best crop setting reaches Macro-F1 `0.812`, competitive with full-image ResNet18.
- **Lower learning rate improves the best ResNet18.** Reducing lr from `1e-4` to `5e-5` lifts Macro-F1 from `0.830` to `0.838`.
- **Freezing the backbone hurts.** ConvNeXt with a frozen backbone drops to Macro-F1 `0.778`.

### Part 1: architecture × input × augmentation

| Model | Input | Augmentation | Accuracy | Recall | Macro-F1 | ROC-AUC |
|---|---|---|---:|---:|---:|---:|
| ResNet18 | full | light | 0.831 | 0.830 | 0.830 | 0.890 |
| ConvNeXt-Tiny | crop | light | 0.797 | 0.797 | 0.797 | 0.849 |
| ConvNeXt-Tiny | full | none | 0.797 | 0.797 | 0.797 | 0.851 |
| ConvNeXt-Tiny | full | light | 0.797 | 0.797 | 0.796 | 0.859 |
| ConvNeXt-Tiny | crop | none | 0.789 | 0.790 | 0.789 | 0.846 |
| ResNet18 | full | none | 0.774 | 0.774 | 0.773 | 0.838 |
| ResNet18 | crop | light | 0.759 | 0.759 | 0.759 | 0.852 |
| ResNet18 | crop | none | 0.737 | 0.737 | 0.737 | 0.820 |

### Part 2: targeted tuning

| Model | Input | LR | Dropout | Frozen | Accuracy | Recall | Macro-F1 | ROC-AUC |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| ResNet18 | full | 5e-5 | 0.3 | no | 0.838 | 0.838 | 0.838 | 0.883 |
| ConvNeXt-Tiny | crop | 1e-4 | 0.0 | no | 0.812 | 0.812 | 0.812 | 0.857 |
| ConvNeXt-Tiny | crop | 5e-5 | 0.3 | no | 0.805 | 0.804 | 0.804 | 0.886 |
| ConvNeXt-Tiny | crop | 3e-4 | 0.3 | yes | 0.778 | 0.778 | 0.778 | 0.849 |
| ConvNeXt-Tiny | crop | 3e-4 | 0.3 | no | 0.771 | 0.771 | 0.771 | 0.825 |
| ConvNeXt-Tiny | crop | 1e-4 | 0.1 | no | 0.771 | 0.770 | 0.770 | 0.828 |

### Qualitative analysis

Saliency and Grad-CAM panels for the best full-image and crop models are available under `results/figures/`:

- `results/figures/gradcam/best_full_resnet18/report_gradcam_panel.png`
- `results/figures/gradcam/best_crop_convnext/report_gradcam_panel.png`
- `results/figures/tables/`

These figures show correct and incorrect test examples with model attention overlays.

## Method Overview

1. Build binary labels from MGS facial action unit scores (`nb`, `cb`, whisker/`wc`).
2. Organize cropped images into subset/class folders (`AW`, `JW`, `KH`, `LW`, `MR` × impaired / not impaired).
3. Create a fixed subset/class stratified train/val/test split, then optionally remap the same indices to full images.
4. Fine-tune ImageNet-pretrained ResNet18 or ConvNeXt-Tiny with early stopping on validation Macro-F1.
5. Report held-out accuracy, macro recall, Macro-F1, and ROC-AUC.

## Repository Layout

```text
.
├── main.py                   # CONFIG + pipeline entry point
├── requirements.txt
├── dataset/                  # Local data (not committed)
├── results/                  # Final Part 1 / Part 2 tables and figures
├── outputs/                  # Local run artifacts (ignored by Git)
├── scripts/analysis/         # Report figure generators
└── src/
    ├── paths.py              # Project paths
    ├── device.py             # CUDA / device helpers
    ├── reproducibility.py    # Seeding
    ├── preprocess.py         # MGS labels + organize crops
    ├── prepare.py            # Dataset preparation runner
    ├── check_setup.py        # Environment / data checks
    ├── model.py              # ResNet18 / ConvNeXt builders
    ├── train.py              # Training loop, metrics, plots
    ├── experiments.py        # Part 1 / Part 2 experiment grid
    └── summarize.py          # Result table aggregation
```

## Setup

### Hardware

- NVIDIA GPU with at least **8 GB VRAM** recommended for ConvNeXt-Tiny
- CPU works, but full Part 1 / Part 2 training is slow

### Install

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPUs, install a matching PyTorch CUDA wheel first:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

### Dataset

Image data is not committed. This project uses **two** image roots:

| Role | Download | Local path |
|---|---|---|
| Cropped muzzles | [Cropped_images.rar (TU Berlin Cloud)](https://tubcloud.tu-berlin.de/s/RR5qtwfatPn7bF2) | `dataset/mouse_dataset/Cropped_images/` |
| Full-face images | [DepositOnce](https://depositonce.tu-berlin.de/items/d2f955f2-8811-4976-86ae-00b04ebf37dd) | `dataset/mouse_dataset/images_mgs/` |

Expected layout (only what the training code uses):

```text
dataset/mouse_dataset/
├── MouseGrimaceFaces_mgs.csv
├── MouseGrimaceFaces_main.csv
├── Cropped_images/              # AW|JW|KH|LW|MR / impaired|not_impaired / *.jpg
├── images_mgs/                  # flat full-face JPGs (same filenames as crops)
└── article_experiment_splits/   # auto-created / reused held-out splits
```

**How crops and full images are linked:** crop runs read
`Cropped_images/<subset>/<class>/<id>.jpg`. Full-image runs remap the same
`<id>.jpg` into `images_mgs/<id>.jpg`. Filenames must match exactly.

More detail: [`dataset/README.md`](dataset/README.md).

Verify after placing the data:

```bash
python main.py --mode check
```

## Reproducing Experiments

Edit `CONFIG` in `main.py`, or override from the CLI:

```bash
python main.py --mode train --part all --dry-run
python main.py --mode train --part part1
python main.py --mode train --part part2
python main.py --mode train --part all --only resnet18_original_light_lr5e5
```

Regenerate tables and figures:

```bash
python main.py --mode summarize
python main.py --mode tables
python main.py --mode diagrams
python main.py --mode curves
python main.py --mode figures
python main.py --mode gradcam
```

Or use Make:

```bash
make check prepare dry-run
make part1 part2
make summarize tables diagrams curves figures gradcam
```

## Outputs

Tracked in Git:

- `results/part1_results.csv`
- `results/part2_results.csv`
- curated figures under `results/figures/`

Generated locally during training (ignored by Git):

- `outputs/models/article_experiments/`
- `outputs/reports/article_experiments/`
- `outputs/figures/article_experiments/`
- `outputs/diagrams/`
