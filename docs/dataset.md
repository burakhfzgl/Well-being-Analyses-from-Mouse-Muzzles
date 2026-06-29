# Dataset

The project expects the Mouse Grimace Faces data to be available locally. The data is not committed to Git because the image folders and archives are large and may have separate license or redistribution terms.

## Expected Layout

```text
dataset/
└── mouse_dataset/
    ├── MouseGrimaceFaces_main.csv
    ├── MouseGrimaceFaces_mgs.csv
    ├── images_mgs/
    ├── images_perfect/
    └── muzzle_crops/
```

`images_mgs/` is the default image directory used by the CLI scripts. `images_perfect/` and `muzzle_crops/` can be selected with `--image-dir` when you want to train on a different image representation.

## Label Construction

The binary target is derived from Mouse Grimace Scale action units:

- `nb`: nose bulge
- `cb`: cheek bulge
- `wc`: whisker change

For each image, valid rater scores are values `0`, `1`, or `2`. The pipeline averages valid rater scores per action unit, rounds half up, then averages the rounded action-unit scores:

- final rounded score `0` -> `well-being`
- final rounded score `1` or higher -> `impaired`

Rows missing one or more action-unit averages are excluded by default. This avoids treating incomplete labels as well-being.

## Data That Should Not Be Committed

The following are local artifacts and are ignored by Git:

- `dataset/mouse_dataset/`
- dataset archives such as `*.zip` or `*.rar`
- generated crops such as `muzzle_crops/`
- model checkpoints and reports under `outputs/`

## Attribution

Before publishing this repository, add the correct Mouse Grimace Faces dataset citation and confirm the dataset redistribution terms. Keep the code license separate from the data license.
