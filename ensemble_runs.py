"""Evaluate a pre-specified probability average of two same-split runs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from mhgnn_moe.metrics import binary_metrics


def read_predictions(path: Path) -> dict[tuple[int, int], tuple[int, float]]:
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            herb = int(row.get("herb_id", row.get("herb")))
            symptom = int(row.get("symptom_id", row.get("symptom")))
            result[(herb, symptom)] = (int(float(row["label"])), float(row["probability"]))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.weight_a <= 1.0:
        raise ValueError("weight-a must be between 0 and 1")

    root = Path(__file__).resolve().parent
    left = root / "results" / args.run_a / "predictions"
    right = root / "results" / args.run_b / "predictions"
    output = root / "results" / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    source_splits = root / "results" / args.run_a / "splits"
    if source_splits.exists():
        shutil.copytree(source_splits, output / "splits", dirs_exist_ok=True)
    metrics_rows = []
    fold = 1
    while (left / f"fold_{fold}.csv").exists() and (right / f"fold_{fold}.csv").exists():
        a = read_predictions(left / f"fold_{fold}.csv")
        b = read_predictions(right / f"fold_{fold}.csv")
        if set(a) != set(b):
            raise RuntimeError(f"Fold {fold}: pair sets differ between runs")
        prediction_rows = []
        for pair in sorted(a):
            label_a, probability_a = a[pair]
            label_b, probability_b = b[pair]
            if label_a != label_b:
                raise RuntimeError(f"Fold {fold}: label mismatch for {pair}")
            probability = args.weight_a * probability_a + (1.0 - args.weight_a) * probability_b
            prediction_rows.append({
                "herb": pair[0], "symptom": pair[1], "label": label_a,
                "probability": probability,
            })
        write_csv(output / "predictions" / f"fold_{fold}.csv", prediction_rows)
        labels = np.asarray([row["label"] for row in prediction_rows])
        probabilities = np.asarray([row["probability"] for row in prediction_rows])
        row = binary_metrics(labels, probabilities, args.threshold)
        row["fold"] = fold
        metrics_rows.append(row)
        fold += 1
    if not metrics_rows:
        raise RuntimeError("No aligned prediction folds found")
    write_csv(output / "fold_metrics.csv", metrics_rows)
    summary = {
        "run_a": args.run_a, "run_b": args.run_b, "weight_a": args.weight_a,
        "threshold": args.threshold, "folds": len(metrics_rows),
    }
    for metric in ("accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc"):
        values = np.asarray([row[metric] for row in metrics_rows])
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
