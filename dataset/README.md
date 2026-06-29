# Dataset Layout

This repository does not commit the Mouse Grimace Faces images or generated crops. Keep the data locally under this folder when running experiments.

Expected layout:

```text
dataset/
└── mouse_dataset/
    ├── MouseGrimaceFaces_main.csv
    ├── MouseGrimaceFaces_mgs.csv
    ├── images_mgs/          # MGS-labeled source images
    ├── images_perfect/      # optional full-face crops
    └── muzzle_crops/        # optional DLC crops
```

Only this `README.md` is intended to be tracked by Git. The image folders, zip files, model outputs, and generated reports are ignored.

## Python Usage

```python
from data.labels import build_labels
from paths import IMAGES_MGS_DIR, MAIN_CSV, MGS_CSV

df = build_labels(MGS_CSV, MAIN_CSV, IMAGES_MGS_DIR)
```

## Command-Line Checks

```powershell
python scripts/check_setup.py
python scripts/train.py --epochs 1 --batch-size 8 --image-size 128 --no-pretrained --limit 80
```

See `docs/dataset.md` for label construction details and dataset attribution notes.
