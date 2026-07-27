"""Thin wrapper. Prefer: python main.py --mode train --part part1"""

from __future__ import annotations

import argparse

from src.experiments import build_train_parser, run_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run article-grade mouse impairment experiments.")
    build_train_parser(parser)
    args = parser.parse_args()
    run_experiments(
        part=args.part,
        dry_run=args.dry_run,
        only=args.only,
        limit_runs=args.limit_runs,
    )


if __name__ == "__main__":
    main()
