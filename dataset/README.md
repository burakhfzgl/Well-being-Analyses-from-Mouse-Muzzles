# Dataset Layout

This repository does not commit Mouse Grimace Faces images. Keep data locally under this folder when running experiments.

```text
dataset/mouse_dataset/
├── MouseGrimaceFaces_main.csv
├── MouseGrimaceFaces_mgs.csv
├── images_mgs/                 # Original full images
├── images_perfect/             # Curated cropped images
└── images_perfect_organized/   # Built by prepare_organized_dataset.py
```

Only this `README.md` is tracked by Git. Image folders and generated labels are ignored.

```bash
python scripts/data_processing/prepare_organized_dataset.py
python scripts/data_processing/check_setup.py
```
