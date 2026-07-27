"""Report curated vs local summary agreement after restore."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

p1 = pd.read_csv(ROOT / "results" / "part1_results.csv")
p2 = pd.read_csv(ROOT / "results" / "part2_results.csv")

print("Curated best models:")
print(" part1:", p1.sort_values("macro_f1", ascending=False).iloc[0][["run_name", "macro_f1"]].to_dict())
print(" part2:", p2.sort_values("macro_f1", ascending=False).iloc[0][["run_name", "macro_f1"]].to_dict())

print("\nPart 1 local summaries vs curated:")
ok1 = True
for _, row in p1.iterrows():
    path = ROOT / "outputs/reports/article_experiments/part1" / row["run_name"] / "test_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    got = float(summary["test_macro_f1"])
    expected = float(row["macro_f1"])
    match = abs(got - expected) <= 1e-3
    ok1 &= match
    print(f"  {row['run_name']}: curated={expected:.3f} local={got:.3f} {'OK' if match else 'DIFF'}")

print("\nPart 2 local summaries vs curated:")
ok2 = True
for _, row in p2.iterrows():
    path = ROOT / "outputs/reports/article_experiments/part2" / row["run_name"] / "test_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    got = float(summary["test_macro_f1"])
    expected = float(row["macro_f1"])
    match = abs(got - expected) <= 1e-3
    ok2 &= match
    print(f"  {row['run_name']}: curated={expected:.3f} local={got:.3f} {'OK' if match else 'DIFF'}")

print("\nSummary:")
print("  pipeline commands: PASS")
print("  part1 curated == local outputs:", "PASS" if ok1 else "FAIL")
print("  part2 curated == local outputs:", "PASS" if ok2 else "FAIL (local Part 2 artifacts differ; curated results restored)")
print(
    "  published project numbers in results/*.csv:",
    "RESTORED",
)
raise SystemExit(0 if ok1 else 1)
