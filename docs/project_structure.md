# Project Structure

The project is organized around reusable source modules and thin command-line scripts.

## Source Modules

```text
src/
├── paths.py
├── data/
│   ├── dataset.py
│   ├── labels.py
│   └── build_labels.py
├── models/
│   ├── convnext.py
│   └── model.py
├── training/
│   ├── engine.py
│   ├── split.py
│   ├── training.py
│   └── training_modified.py
├── evaluation/
│   ├── metrics.py
│   ├── experiment.py
│   ├── experiment_modified.py
│   └── generate_mouse_dataset_report.py
├── utils/
│   ├── device.py
│   └── reproducibility.py
└── visualization/
```

## CLI Scripts

- `scripts/check_setup.py` verifies paths, dependencies, CUDA, labels, and image loading.
- `scripts/train.py` builds labels, splits the data, trains a ConvNeXt model, and saves a checkpoint.
- `scripts/evaluate.py` loads a checkpoint and reports validation metrics.
- `scripts/make_report.py` creates the dataset exploration report.

## Notebooks

Notebooks are examples and reports. Reusable logic should live in `src/`; notebooks should import that logic instead of redefining training or preprocessing functions.

## Where to Change Common Settings

- Dataset paths: `src/paths.py`
- Label logic: `src/data/labels.py`
- Train/validation split: `src/training/split.py`
- Training loop and checkpoints: `src/training/engine.py`
- Model architecture: `src/models/convnext.py`
- Metrics and checkpoint loading: `src/evaluation/metrics.py`
