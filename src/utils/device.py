"""Device selection helpers."""

from __future__ import annotations

import torch


def get_device(prefer_mps: bool = False) -> torch.device:
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device | str | None = None) -> str:
    """Return a human-readable device description."""
    if device is None:
        device = get_device()
    device = torch.device(device)
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return f"cuda ({name}, {total_gb:.1f} GB)"
    return device.type
