"""PyTorch datasets for mouse image classification."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class MouseImageDataset(Dataset):
    """Dataset backed by a dataframe with `path` and `label` columns."""

    def __init__(self, dataframe, transform=None) -> None:
        required_columns = {"path", "label"}
        missing = required_columns.difference(dataframe.columns)
        if missing:
            raise ValueError(f"Dataframe is missing required columns: {sorted(missing)}")

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = Path(row["path"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label


# Backward-compatible class name used by existing notebooks.
MouseDataset = MouseImageDataset
