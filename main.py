"""Project entry point.

The public workflow is CLI-first. Use:

    python scripts/check_setup.py
    python scripts/train.py
    python scripts/evaluate.py --checkpoint outputs/models/best_model.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Mouse grimace computer vision project")
    parser.add_argument(
        "command",
        choices=["check", "train", "evaluate", "report"],
        help="CLI command to run",
    )
    args, remainder = parser.parse_known_args()

    script_map = {
        "check": "check_setup.py",
        "train": "train.py",
        "evaluate": "evaluate.py",
        "report": "make_report.py",
    }
    script = Path(__file__).resolve().parent / "scripts" / script_map[args.command]
    subprocess.run([sys.executable, str(script), *remainder], check=True)


if __name__ == "__main__":
    main()
