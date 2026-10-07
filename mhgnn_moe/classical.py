"""Classical same-split baselines using interpretable network features."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_dataset
from .metrics import binary_metrics


STRUCTURAL_FEATURE_NAMES = (
    "log_herb_targets", "log_symptom_targets", "target_intersection", "target_jaccard",
    "target_cosine",
)
PROXIMITY_FEATURE_NAMES = ("proximity_d", "proximity_z", "proximity_present")


def load_split(directory: Path, fold: int) -> dict[str, np.ndarray]:
    matches = sorted((directory / "splits").glob(f"*_fold{fold}.npz"))
    if not matches:
        raise FileNotFoundError(f"No split for fold {fold} in {directory}")
    payload = np.load(matches[-1])
    return {key: payload[key] for key in payload.files}


def target_sets(bundle) -> list[set[int]]:
    result = [set() for _ in range(bundle.num_hyperedges)]
    edges = bundle.incidence_edge.cpu().numpy()
    vertices = bundle.incidence_vertex.cpu().numpy()
    for edge, vertex in zip(edges, vertices):
        result[int(edge)].add(int(vertex))
    return result


def pair_features(
    bundle, sets: list[set[int]], edges: np.ndarray, include_proximity: bool = False,
) -> np.ndarray:
    finite_d = bundle.proximity_d[np.isfinite(bundle.proximity_d)]
    finite_z = bundle.proximity_z[np.isfinite(bundle.proximity_z)]
    d_mean, d_std = float(finite_d.mean()), max(float(finite_d.std()), 1e-8)
    z_mean, z_std = float(finite_z.mean()), max(float(finite_z.std()), 1e-8)
    rows = []
    for herb_value, symptom_value in edges:
        herb, symptom = int(herb_value), int(symptom_value)
        left, right = sets[herb], sets[bundle.num_herbs + symptom]
        intersection = len(left & right)
        union = len(left | right)
        denominator = max((len(left) * len(right)) ** 0.5, 1.0)
        d, z = bundle.proximity_d[herb, symptom], bundle.proximity_z[herb, symptom]
        present = float(np.isfinite(d))
        structural = (
            np.log1p(len(left)), np.log1p(len(right)), np.log1p(intersection),
            intersection / max(union, 1), intersection / denominator,
        )
        proximity = (
            (float(d) - d_mean) / d_std if np.isfinite(d) else 0.0,
            (float(z) - z_mean) / z_std if np.isfinite(z) else 0.0,
            present,
        )
        rows.append(structural + proximity if include_proximity else structural)
    return np.asarray(rows, dtype=np.float32)


def build_model(name: str, seed: int):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=seed
        )
    if name == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    raise ValueError(name)


def save_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["HIT", "TCMIO"], required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--model", choices=["logistic", "random_forest", "hist_gb"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-leaky-proximity", action="store_true",
        help="Diagnostic only: include label-revealing proximity coverage features",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    root = Path(__file__).resolve().parents[1]
    reference = root / "results" / args.reference_run
    output = root / "results" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions").mkdir(exist_ok=True)
    bundle = load_dataset(root / "data" / "processed", args.dataset, torch.device("cpu"))
    sets = target_sets(bundle)
    rows = []
    fold = 0
    while list((reference / "splits").glob(f"*_fold{fold}.npz")):
        split = load_split(reference, fold)
        train_edges = np.concatenate((split["train_pos"], split["train_neg"]))
        train_labels = np.concatenate((np.ones(len(split["train_pos"])), np.zeros(len(split["train_neg"]))))
        valid_edges = np.concatenate((split["valid_pos"], split["valid_neg"]))
        valid_labels = np.concatenate((np.ones(len(split["valid_pos"])), np.zeros(len(split["valid_neg"]))))
        start = time.time()
        model = build_model(args.model, args.seed + fold)
        model.fit(
            pair_features(bundle, sets, train_edges, args.include_leaky_proximity), train_labels
        )
        probability = model.predict_proba(
            pair_features(bundle, sets, valid_edges, args.include_leaky_proximity)
        )[:, 1]
        metrics = binary_metrics(valid_labels, probability, 0.5)
        metrics.update(fold=fold + 1, seconds=time.time() - start)
        rows.append(metrics)
        save_csv(
            output / "predictions" / f"fold_{fold + 1}.csv",
            [
                {
                    "herb": int(edge[0]), "symptom": int(edge[1]),
                    "label": int(label), "probability": float(score),
                }
                for edge, label, score in zip(valid_edges, valid_labels, probability)
            ],
        )
        fold += 1
    if not rows:
        raise RuntimeError("No folds found")
    save_csv(output / "fold_metrics.csv", rows)
    summary = {"dataset": args.dataset, "model": args.model, "reference_run": args.reference_run, "folds": len(rows)}
    for metric in ("accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc", "seconds"):
        values = np.asarray([row[metric] for row in rows])
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std())
    summary["feature_names"] = list(STRUCTURAL_FEATURE_NAMES) + (
        list(PROXIMITY_FEATURE_NAMES) if args.include_leaky_proximity else []
    )
    summary["leaky_proximity_included"] = bool(args.include_leaky_proximity)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
