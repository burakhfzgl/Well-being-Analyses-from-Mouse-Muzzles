#!/usr/bin/env python3
"""Mouse Muzzle Impairment Classification.

Binary computer-vision study: impaired vs not impaired from mouse face images.
Part 1 ablates architecture / input / augmentation.
Part 2 tunes learning rate, dropout, and backbone freezing.

Usage
-----
    python main.py
    python main.py --mode check
    python main.py --mode prepare
    python main.py --mode train --part part1
    python main.py --mode train --part all --dry-run
    python main.py --mode summarize
    python main.py --mode tables
"""

from __future__ import annotations

import argparse
import time

import torch

from src.check_setup import run_check
from src.device import describe_device, get_device
from src.experiments import run_experiments
from src.prepare import run_prepare
from src.reproducibility import set_seed
from src.summarize import main as run_summarize

## --------------- Config ------------ ##

CONFIG = {
    ## Pipeline mode when no CLI override is given.
    ## Options: "check", "prepare", "train", "summarize", "tables", "diagrams", "curves", "figures", "gradcam"
    "mode": "train",

    ## Part 1 = architecture × input × augmentation
    ## Part 2 = lr / dropout / freeze tuning
    ## all = Part 1 then Part 2
    "part": "part1",

    ## If True, prepare splits/manifest only (no GPU training).
    "dry_run": False,

    ## Optional: run only these experiment names (None = all selected part runs).
    "only": None,
    # "only": ["resnet18_original_light_lr5e5"],

    ## Optional: run only the first N experiments from the selected part.
    "limit_runs": None,

    ## Reproducibility
    "seed": 42,

    ## Default training hyperparameters (overridden per experiment in src/experiments.py)
    "epochs": 35,
    "batch_size": 16,
    "image_size": 224,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.30,
    "pretrained": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mouse muzzle impairment classification (ResNet18 / ConvNeXt-Tiny).",
    )
    parser.add_argument(
        "--mode",
        choices=["check", "prepare", "train", "summarize", "tables", "diagrams", "curves", "figures", "gradcam"],
        default=None,
        help="Override CONFIG['mode'].",
    )
    parser.add_argument(
        "--part",
        choices=["part1", "part2", "all"],
        default=None,
        help="Override CONFIG['part'] for training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare experiment plan without training.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these experiment names.",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help="Run only the first N experiments.",
    )
    return parser.parse_args()


def run_analysis_script(relative_path: str, script_argv: list[str] | None = None) -> None:
    """Run a report/figure script from scripts/analysis/."""
    import runpy
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent
    script = root / relative_path
    previous = sys.argv
    sys.argv = [str(script), *(script_argv or [])]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous


def main() -> None:
    args = parse_args()

    mode = args.mode or CONFIG["mode"]
    part = args.part or CONFIG["part"]
    dry_run = args.dry_run or CONFIG["dry_run"]
    only = args.only if args.only is not None else CONFIG["only"]
    limit_runs = args.limit_runs if args.limit_runs is not None else CONFIG["limit_runs"]

    set_seed(CONFIG["seed"])
    device = get_device()

    print("=" * 60)
    print("Mouse Muzzle Impairment Classification")
    print("=" * 60)
    print(f"Mode:   {mode}")
    print(f"Device: {describe_device(device)}")
    print(f"CUDA:   {torch.cuda.is_available()}")
    print(f"Seed:   {CONFIG['seed']}")
    print("=" * 60)

    start = time.time()

    if mode == "check":
        run_check()

    elif mode == "prepare":
        run_prepare()

    elif mode == "train":
        print(f"Part:      {part}")
        print(f"Dry run:   {dry_run}")
        print(f"Only:      {only}")
        print(f"Limit:     {limit_runs}")
        run_experiments(
            part=part,
            dry_run=dry_run,
            only=only,
            limit_runs=limit_runs,
        )

    elif mode == "summarize":
        run_summarize([])

    elif mode == "tables":
        run_analysis_script("scripts/analysis/generate_result_table_figures.py")

    elif mode == "diagrams":
        run_analysis_script("scripts/analysis/generate_report_diagrams.py")

    elif mode == "curves":
        run_analysis_script("scripts/analysis/generate_training_curves.py")

    elif mode == "figures":
        run_analysis_script(
            "scripts/analysis/generate_qualitative_visualizations.py",
            ["--part", "all"],
        )

    elif mode == "gradcam":
        run_analysis_script("scripts/analysis/generate_gradcam_visualizations.py")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
