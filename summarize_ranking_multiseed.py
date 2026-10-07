"""Summarize full-candidate ranking runs across independent random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_run(path: Path) -> dict:
    with (path / "summary.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    runs = {name: read_run(root / "results" / name) for name in args.runs}
    metric_names = tuple(next(iter(runs.values()))["metrics"])
    seed_results = {
        name: {metric: float(item["metrics"][metric]["mean"]) for metric in metric_names}
        for name, item in runs.items()
    }

    summary = {
        "runs": args.runs,
        "aggregation": "mean and population SD across complete run macro means",
        "seed_results": seed_results,
        "metrics": {},
    }
    for metric in metric_names:
        values = np.asarray([seed_results[name][metric] for name in args.runs])
        summary["metrics"][metric] = {
            "mean_of_seed_means": float(values.mean()),
            "std_of_seed_means": float(values.std()),
            "minimum_seed_mean": float(values.min()),
            "maximum_seed_mean": float(values.max()),
        }

    output = root / "results" / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "multiseed_ranking_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Multi-seed full-ranking summary",
        "",
        "Means and standard deviations are calculated across complete-run macro means, not pooled folds or queries.",
        "",
        "| Metric | Mean of seed means | SD across seeds | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, item in summary["metrics"].items():
        lines.append(
            f"| {metric} | {item['mean_of_seed_means']:.4f} | "
            f"{item['std_of_seed_means']:.4f} | {item['minimum_seed_mean']:.4f} | "
            f"{item['maximum_seed_mean']:.4f} |"
        )
    (output / "MULTISEED_RANKING_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
