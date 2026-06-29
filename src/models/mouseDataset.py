"""Backward-compatible dataset import.

New code should import `MouseImageDataset` from `data.dataset`.
"""

from data.dataset import MouseDataset, MouseImageDataset

__all__ = ["MouseDataset", "MouseImageDataset"]