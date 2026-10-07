"""Cluster-bootstrap confidence intervals for filtered full-ranking results."""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np

METRICS = ("ap", "mrr", "recall@10", "ndcg@10")


def read_rows(path: Path, seed: int):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (str(seed), str(row["fold"]), str(row["herb"])): {
                metric: float(row[metric]) for metric in METRICS
            }
            for row in csv.DictReader(handle)
        }


def merge_method(root: Path, runs: list[str]):
    merged = {}
    for seed, run in enumerate(runs):
        merged.update(read_rows(root / "results" / run / "query_metrics.csv", seed))
    return merged


def bootstrap_mean(values, cluster_ids, rng, reps):
    clusters = np.unique(cluster_ids)
    by_cluster = [values[cluster_ids == cluster] for cluster in clusters]
    draws = np.empty(reps, dtype=float)
    for index in range(reps):
        selected = rng.integers(0, len(by_cluster), size=len(by_cluster))
        draws[index] = np.concatenate([by_cluster[item] for item in selected]).mean()
    return draws


def ci(values):
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--hybrid-runs", nargs=3, required=True)
    parser.add_argument("--deep-runs", nargs=3, required=True)
    parser.add_argument("--histgb-runs", nargs=3, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    methods = {name: merge_method(root, runs) for name, runs in (
        ("hybrid", args.hybrid_runs), ("deep", args.deep_runs), ("histgb", args.histgb_runs))}
    common = set(methods["hybrid"])
    common &= set(methods["deep"]) & set(methods["histgb"])
    if not common:
        raise RuntimeError("No common seed/fold/herb query clusters across methods")
    ordered = sorted(common)
    cluster_ids = np.asarray([f"{seed}:{herb}" for seed, _fold, herb in ordered])
    rng = np.random.default_rng(args.seed)
    summary = {"dataset": args.dataset, "clusters": int(len(np.unique(cluster_ids)),),
               "queries": len(ordered), "bootstrap_replicates": args.reps,
               "bootstrap_unit": "seed-herb query cluster", "methods": {},
               "paired_differences": {}}
    for method, rows in methods.items():
        summary["methods"][method] = {}
        for metric in METRICS:
            values = np.asarray([rows[key][metric] for key in ordered])
            summary["methods"][method][metric] = ci(bootstrap_mean(values, cluster_ids, rng, args.reps))
    for candidate in ("deep", "histgb"):
        name = f"hybrid_minus_{candidate}"
        summary["paired_differences"][name] = {}
        for metric in METRICS:
            values = np.asarray([methods["hybrid"][key][metric] - methods[candidate][key][metric] for key in ordered])
            draws = bootstrap_mean(values, cluster_ids, rng, args.reps)
            summary["paired_differences"][name][metric] = {
                **ci(draws), "observed_mean": float(values.mean()),
                "probability_positive": float(np.mean(draws > 0.0))}
    output = root / "results" / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "BOOTSTRAP_RANKING_CI.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# {args.dataset} full-ranking cluster bootstrap", "",
             "Bootstrap unit: seed–herb query cluster; intervals are descriptive and do not replace an external test set.", "",
             "| Method | Metric | Mean | 95% CI |", "|---|---|---:|---:|"]
    for method, metrics in summary["methods"].items():
        for metric, item in metrics.items():
            lines.append(f"| {method} | {metric} | {item['mean']:.4f} | [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] |")
    lines += ["", "| Paired difference | Metric | Mean difference | 95% CI | P(draw > 0) |", "|---|---|---:|---:|---:|"]
    for comparison, metrics in summary["paired_differences"].items():
        for metric, item in metrics.items():
            lines.append(f"| {comparison} | {metric} | {item['observed_mean']:.4f} | [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | {item['probability_positive']:.4f} |")
    (output / "BOOTSTRAP_RANKING_CI.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
