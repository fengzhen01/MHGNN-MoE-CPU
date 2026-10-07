"""Audit pair features for direct label proxies on saved cross-validation splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from mhgnn_moe.classical import (
    PROXIMITY_FEATURE_NAMES, STRUCTURAL_FEATURE_NAMES, load_split, pair_features, target_sets,
)
from mhgnn_moe.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["HIT", "TCMIO"], required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    reference = root / "results" / args.reference_run
    output = root / "results" / args.output
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(root / "data" / "processed", args.dataset, torch.device("cpu"))
    sets = target_sets(bundle)
    names = list(STRUCTURAL_FEATURE_NAMES) + list(PROXIMITY_FEATURE_NAMES)
    rows = []
    fold = 0
    while list((reference / "splits").glob(f"*_fold{fold}.npz")):
        split = load_split(reference, fold)
        edges = np.concatenate((split["valid_pos"], split["valid_neg"]))
        labels = np.concatenate((np.ones(len(split["valid_pos"])), np.zeros(len(split["valid_neg"]))))
        features = pair_features(bundle, sets, edges, include_proximity=True)
        row = {
            "fold": fold + 1,
            "positive_proximity_coverage": float(features[labels == 1, -1].mean()),
            "negative_proximity_coverage": float(features[labels == 0, -1].mean()),
        }
        for index, name in enumerate(names):
            auc = float(roc_auc_score(labels, features[:, index]))
            row[f"auc_{name}"] = max(auc, 1.0 - auc)
        rows.append(row)
        fold += 1
    if not rows:
        raise RuntimeError("No saved folds found")

    summary = {
        "dataset": args.dataset,
        "reference_run": args.reference_run,
        "folds": len(rows),
        "positive_proximity_coverage_mean": float(np.mean([r["positive_proximity_coverage"] for r in rows])),
        "negative_proximity_coverage_mean": float(np.mean([r["negative_proximity_coverage"] for r in rows])),
        "single_feature_absolute_auroc_mean": {
            name: float(np.mean([r[f"auc_{name}"] for r in rows])) for name in names
        },
        "interpretation": (
            "An absolute AUROC near 1.0 means the feature alone separates the saved labels. "
            "Proximity-derived features must not enter the predictive model when coverage differs by label."
        ),
    }
    (output / "leakage_audit.json").write_text(
        json.dumps({"summary": summary, "folds": rows}, indent=2), encoding="utf-8"
    )
    lines = [
        "# Feature leakage audit", "",
        f"- Dataset: {args.dataset}", f"- Reference split: `{args.reference_run}`",
        f"- Positive proximity coverage: {summary['positive_proximity_coverage_mean']:.4f}",
        f"- Negative proximity coverage: {summary['negative_proximity_coverage_mean']:.4f}",
        "", "| Feature | Mean absolute single-feature AUROC |", "|---|---:|",
    ]
    for name, value in summary["single_feature_absolute_auroc_mean"].items():
        lines.append(f"| {name} | {value:.4f} |")
    lines += [
        "", "Proximity-derived features are diagnostic-only in the leakage-safe protocol.",
    ]
    (output / "LEAKAGE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
