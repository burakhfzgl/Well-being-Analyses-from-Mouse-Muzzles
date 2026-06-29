# Mouse Grimace Computer Vision

This repository contains a PyTorch computer vision pipeline for binary mouse welfare classification from Mouse Grimace Faces images.

The project is intentionally **code-only** for public GitHub. Images, model checkpoints, generated figures, and reports stay local and are ignored by Git. See `dataset/README.md` and `docs/dataset.md` for the expected local dataset layout.

## What This Project Does

- Builds image-level labels from Mouse Grimace Scale (MGS) rater scores.
- Uses group-aware train/validation splitting by `subset + id` to reduce mouse identity leakage.
- Trains a ConvNeXt-Tiny binary classifier (`well-being` vs `impaired`).
- Evaluates classification metrics and saves the best checkpoint.
- Provides visualization utilities for Grad-CAM, saliency maps, and dataset figures.

## Repository Layout

```text
.
├── dataset/
│   └── README.md                 # Dataset placement instructions, data ignored by Git
├── docs/
│   ├── dataset.md
│   ├── project_structure.md
│   └── reproducibility.md
├── notebooks/                    # Example notebooks, not the primary workflow
├── scripts/
│   ├── check_setup.py
│   ├── train.py
│   ├── evaluate.py
│   └── make_report.py
├── src/
│   ├── data/                     # Labels and Dataset classes
│   ├── evaluation/               # Metrics, checkpoints, reports
│   ├── models/                   # Model builders
│   ├── training/                 # Split and training loops
│   ├── utils/                    # Device, paths, reproducibility
│   └── visualization/            # Grad-CAM, saliency, plots
├── outputs/                      # Generated locally, ignored by Git
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA GPU such as an RTX 4060, install the CUDA PyTorch wheels:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

Verify the environment:

```powershell
python scripts/check_setup.py
```

## Dataset Setup

Place the dataset locally like this:

```text
dataset/
└── mouse_dataset/
    ├── MouseGrimaceFaces_main.csv
    ├── MouseGrimaceFaces_mgs.csv
    ├── images_mgs/
    ├── images_perfect/
    └── muzzle_crops/             # optional
```

The dataset folder is ignored by Git. Public users should obtain the Mouse Grimace Faces data from the original source and follow `docs/dataset.md` for placement and expected filenames.

## Quick Start

Check paths, dependencies, CUDA availability, label counts, and image loading:

```powershell
python scripts/check_setup.py
```

Run a short smoke training pass:

```powershell
python scripts/train.py --epochs 1 --batch-size 8 --image-size 128 --no-pretrained --limit 80
```

Run a full training job:

```powershell
python scripts/train.py --epochs 30 --batch-size 32 --image-size 224 --pretrained
```

Evaluate a saved checkpoint:

```powershell
python scripts/evaluate.py --checkpoint outputs/models/best_model.pt
```

Generate the dataset report:

```powershell
python scripts/make_report.py
```

## Outputs

Generated artifacts are written under `outputs/` and ignored by Git:

- `outputs/models/` for checkpoints
- `outputs/reports/` for CSV/PDF reports
- `outputs/figures/` for plots and visualizations
- `outputs/gradcam/` for Grad-CAM figures

## Notes for Contributors

- Keep data and checkpoints out of Git.
- Prefer CLI scripts for reproducible runs.
- Use notebooks only as readable examples or reports.
- Put reusable code in `src/`, not inside notebook cells.

## License and Dataset Terms

No project license has been selected yet. Before publishing, choose an appropriate code license and confirm the original dataset license/citation requirements separately.