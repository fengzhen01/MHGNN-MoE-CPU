"""Summarize completed cross-validation runs without treating folds as seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc")


def read_run(path: Path) -> dict[str, float]:
    with (path / "fold_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    seed_results = {name: read_run(root / "results" / name) for name in args.runs}
    summary = {"runs": args.runs, "seed_results": seed_results, "metrics": {}}
    for metric in METRICS:
        values = np.asarray([seed_results[name][metric] for name in args.runs])
        summary["metrics"][metric] = {
            "mean_of_seed_means": float(values.mean()),
            "std_of_seed_means": float(values.std()),
            "minimum_seed_mean": float(values.min()),
            "maximum_seed_mean": float(values.max()),
        }
    output = root / "results" / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "multiseed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Multi-seed summary", "",
        "Means and standard deviations below are calculated across complete CV run means, not across pooled folds.",
        "", "| Metric | Mean of seed means | SD across seeds | Min | Max |", "|---|---:|---:|---:|---:|",
    ]
    for metric, item in summary["metrics"].items():
        lines.append(
            f"| {metric} | {item['mean_of_seed_means']:.4f} | {item['std_of_seed_means']:.4f} | "
            f"{item['minimum_seed_mean']:.4f} | {item['maximum_seed_mean']:.4f} |"
        )
    (output / "MULTISEED_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
