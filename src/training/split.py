"""Train/validation splitting utilities."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_group_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    group_columns: tuple[str, str] = ("subset", "id"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by mouse identity group to reduce leakage."""
    for column in (*group_columns, "label"):
        if column not in df.columns:
            raise ValueError(f"Dataframe is missing required column: {column}")

    groups = df[group_columns[0]].astype(str) + "_" + df[group_columns[1]].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, y=df["label"], groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df
