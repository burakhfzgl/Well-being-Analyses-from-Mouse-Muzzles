# Dataset

Image data is **not committed**. Download the two sources below and place them under
`dataset/mouse_dataset/` using the exact folder names expected by the code
(`src/paths.py`).

## Downloads

| Role | Source | Extract / place as |
|---|---|---|
| **Cropped muzzle images** | [Cropped_images.rar (TU Berlin Cloud)](https://tubcloud.tu-berlin.de/s/RR5qtwfatPn7bF2) | `dataset/mouse_dataset/Cropped_images/` |
| **Full-face images** | [DepositOnce record](https://depositonce.tu-berlin.de/items/d2f955f2-8811-4976-86ae-00b04ebf37dd) | `dataset/mouse_dataset/images_mgs/` |

Also keep the MGS label CSVs next to those folders (same directory):

- `MouseGrimaceFaces_mgs.csv`
- `MouseGrimaceFaces_main.csv` (metadata; optional for training if splits already exist)

## Layout used by this project

```text
dataset/mouse_dataset/
├── MouseGrimaceFaces_mgs.csv
├── MouseGrimaceFaces_main.csv
├── Cropped_images/                 # crop experiments
│   ├── AW/
│   │   ├── impaired/
│   │   └── not_impaired/
│   ├── JW/
│   ├── KH/
│   ├── LW/
│   └── MR/
├── images_mgs/                     # full-image experiments (flat JPGs)
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── article_experiment_splits/      # created/reused by the training code
    ├── heldout_subset_class_crop.csv
    └── heldout_subset_class_original.csv
```

## How the code links crops and full images

1. **Crop runs** load images from  
   `Cropped_images/<subset>/<class>/<filename>.jpg`  
   Example: `Cropped_images/KH/impaired/008821.jpg`

2. **Full-image runs** use the **same filenames** under  
   `images_mgs/<filename>.jpg`  
   Example: `images_mgs/008821.jpg`

3. The held-out split is built once on crops, then remapped to `images_mgs` by
   filename (`index` column in the split CSV).  
   So every crop file name must exist as a matching JPG in `images_mgs/`.

Subsets: `AW`, `JW`, `KH`, `LW`, `MR`  
Classes: `impaired`, `not_impaired`

## Setup steps

1. Create the folder:
   ```bash
   mkdir -p dataset/mouse_dataset
   ```
2. Download [Cropped_images.rar](https://tubcloud.tu-berlin.de/s/RR5qtwfatPn7bF2),
   extract it so you get `dataset/mouse_dataset/Cropped_images/...`
   (not `dataset/mouse_dataset/Cropped_images/Cropped_images/...`).
3. Download the full images from
   [DepositOnce](https://depositonce.tu-berlin.de/items/d2f955f2-8811-4976-86ae-00b04ebf37dd)
   and place the JPG files in `dataset/mouse_dataset/images_mgs/`.
4. Put `MouseGrimaceFaces_mgs.csv` (and `MouseGrimaceFaces_main.csv` if available)
   into `dataset/mouse_dataset/`.
5. Verify:
   ```bash
   python main.py --mode check
   ```

If crop/full splits do not exist yet, the first training run creates them under
`article_experiment_splits/`.

## Notes

- Do not rename `Cropped_images` or `images_mgs`; those names are hard-coded in
  `src/paths.py` and `src/experiments.py`.
- On Windows, a long path or nested extract is a common mistake. After
  extracting the RAR, confirm a path like  
  `dataset/mouse_dataset/Cropped_images/AW/not_impaired/*.jpg` exists.
- Full images must be flat JPG files in `images_mgs/` with names that match the
  crop basenames.
