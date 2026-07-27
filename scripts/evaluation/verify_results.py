"""Compare regenerated result tables against expected README metrics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.paths import REPORTS_DIR, RESULTS_DIR

METRIC_COLS = ["accuracy", "recall", "macro_f1", "roc_auc"]
EXPECTED = {
    "resnet18_original_light_lr5e5": 0.838,
    "convnext_tiny_crop_light_dropout0": 0.812,
    "resnet18_original_light": 0.830,
}


def compare(name: str, left: pd.DataFrame, right: pd.DataFrame) -> bool:
    print(f"=== {name} ===")
    left = left.sort_values("run_name").reset_index(drop=True)
    right = right.sort_values("run_name").reset_index(drop=True)
    print(f"rows: {len(left)} vs {len(right)}")
    if set(left["run_name"]) != set(right["run_name"]):
        print("RUN NAME MISMATCH")
        print("only left:", set(left["run_name"]) - set(right["run_name"]))
        print("only right:", set(right["run_name"]) - set(left["run_name"]))
        return False

    merged = left.merge(right, on="run_name", suffixes=("_l", "_r"))
    ok = True
    for col in METRIC_COLS:
        diff = (merged[f"{col}_l"] - merged[f"{col}_r"]).abs()
        max_diff = float(diff.max())
        print(f"  {col}: max abs diff = {max_diff:.6f}")
        if max_diff > 1e-6:
            ok = False
            print(merged.loc[diff > 1e-6, ["run_name", f"{col}_l", f"{col}_r"]].to_string(index=False))
    print("PASS" if ok else "FAIL")
    return ok


def main() -> None:
    results_part1 = pd.read_csv(RESULTS_DIR / "part1_results.csv")
    results_part2 = pd.read_csv(RESULTS_DIR / "part2_results.csv")
    reports_part1 = pd.read_csv(REPORTS_DIR / "article_experiments" / "part1_results.csv")
    reports_part2 = pd.read_csv(REPORTS_DIR / "article_experiments" / "part2_results.csv")

    ok = True
    ok &= compare("part1 results/ vs outputs/reports/", results_part1, reports_part1)
    ok &= compare("part2 results/ vs outputs/reports/", results_part2, reports_part2)

    print("\nBest models:")
    best_part2 = results_part2.sort_values("macro_f1", ascending=False).iloc[0]
    best_part1 = results_part1.sort_values("macro_f1", ascending=False).iloc[0]
    print(
        f"  best overall: {best_part2['run_name']} macro_f1={best_part2['macro_f1']:.3f}"
    )
    print(
        f"  best part1:   {best_part1['run_name']} macro_f1={best_part1['macro_f1']:.3f}"
    )

    print("\nREADME Macro-F1 checks:")
    combined = pd.concat([results_part1, results_part2], ignore_index=True)
    for run_name, expected in EXPECTED.items():
        row = combined[combined["run_name"] == run_name]
        if row.empty:
            print(f"  {run_name}: MISSING")
            ok = False
            continue
        got = float(row.iloc[0]["macro_f1"])
        match = abs(got - expected) < 1e-3
        status = "OK" if match else "FAIL"
        print(f"  {run_name}: got={got:.3f} expected={expected:.3f} {status}")
        ok &= match

    # Full table dump for review
    print("\nPart 1 table:")
    print(results_part1.sort_values("macro_f1", ascending=False).to_string(index=False))
    print("\nPart 2 table:")
    print(results_part2.sort_values("macro_f1", ascending=False).to_string(index=False))

    print("\nOVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
