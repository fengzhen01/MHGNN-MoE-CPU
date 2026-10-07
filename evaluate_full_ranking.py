"""Evaluate saved models by ranking all eligible symptoms for each query herb."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from mhgnn_moe.classical import build_model, load_split, pair_features, target_sets
from mhgnn_moe.data import load_dataset, normalized_ppi_adjacency
from mhgnn_moe.model import MHGNNMoE


def ranking_metrics(labels: np.ndarray, scores: np.ndarray, ks: tuple[int, ...]) -> dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    relevant = int(labels.sum())
    first = np.flatnonzero(ranked > 0)
    result = {
        "ap": float(average_precision_score(labels, scores)),
        "mrr": float(1.0 / (first[0] + 1)) if len(first) else 0.0,
    }
    for k in ks:
        top = ranked[:k]
        hits = int(top.sum())
        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        dcg = float(np.sum(top * discounts))
        ideal_count = min(relevant, k)
        idcg = float(np.sum(discounts[:ideal_count]))
        result[f"hit@{k}"] = float(hits > 0)
        result[f"recall@{k}"] = hits / relevant if relevant else 0.0
        result[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return result


def save_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_deep_model(bundle, checkpoint: dict, device: torch.device) -> MHGNNMoE:
    config = checkpoint["config"]
    model = MHGNNMoE(
        input_dim=bundle.features.shape[1], hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"], ppi_layers=config["ppi_layers"],
        hyper_layers=config["hyper_layers"], dropout=config["dropout"],
        num_herbs=bundle.num_herbs, num_hyperedges=bundle.num_hyperedges,
        num_experts=config.get("num_experts", 3),
        router_temperature=config.get("router_temperature", 1.0),
        initial_residual_strength=config.get("initial_residual_strength", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.moe_enabled = bool(checkpoint.get("moe_enabled", False))
    model.eval()
    return model


@torch.no_grad()
def deep_scores(model, bundle, adjacency, edges: np.ndarray, config: dict, batch_size: int) -> np.ndarray:
    low, high = model.encode(
        bundle.features, adjacency, bundle.incidence_edge, bundle.incidence_vertex
    )
    values = []
    for start in range(0, len(edges), batch_size):
        tensor = torch.as_tensor(edges[start:start + batch_size], dtype=torch.long, device=bundle.features.device)
        pair_stats = bundle.pair_stats(
            tensor, include_proximity=bool(config.get("allow_proximity_router", False))
        )
        output = model.forward_from_encoded(low, high, tensor, pair_stats)
        values.append(torch.sigmoid(output["logits"]).cpu().numpy())
    return np.concatenate(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["deep", "histgb", "hybrid"], default="hybrid")
    parser.add_argument("--deep-weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not 0.0 <= args.deep_weight <= 1.0:
        raise ValueError("deep-weight must be in [0, 1]")

    torch.set_num_threads(args.cpu_threads)
    root = Path(__file__).resolve().parent
    reference = root / "results" / args.reference_run
    output = root / "results" / args.output
    output.mkdir(parents=True, exist_ok=True)
    first_checkpoint = torch.load(reference / "checkpoints" / "fold_1.pt", map_location="cpu")
    dataset = first_checkpoint["config"]["dataset"]
    bundle = load_dataset(root / "data" / "processed", dataset, torch.device("cpu"))
    adjacency = normalized_ppi_adjacency(bundle.ppi_edges, bundle.num_proteins, torch.device("cpu"))
    sets = target_sets(bundle)
    all_known = set(map(tuple, bundle.positives_unique.tolist()))
    ks = (1, 5, 10, 20)
    query_rows, fold_rows = [], []
    fold = 0
    while list((reference / "splits").glob(f"*_fold{fold}.npz")):
        split = load_split(reference, fold)
        valid_set = set(map(tuple, split["valid_pos"].tolist()))
        query_herbs = sorted({herb for herb, _ in valid_set})
        candidate_edges, query_slices = [], []
        for herb in query_herbs:
            start = len(candidate_edges)
            eligible = [
                symptom for symptom in range(bundle.num_symptoms)
                if (herb, symptom) not in all_known or (herb, symptom) in valid_set
            ]
            candidate_edges.extend((herb, symptom) for symptom in eligible)
            query_slices.append((herb, start, len(candidate_edges)))
        candidates = np.asarray(candidate_edges, dtype=np.int64)

        deep = None
        if args.mode in {"deep", "hybrid"}:
            checkpoint = torch.load(
                reference / "checkpoints" / f"fold_{fold + 1}.pt", map_location="cpu"
            )
            if bool(checkpoint["config"].get("allow_proximity_router", False)):
                raise RuntimeError("Full-ranking evaluation refuses proximity-enabled checkpoints")
            model = build_deep_model(bundle, checkpoint, torch.device("cpu"))
            deep = deep_scores(model, bundle, adjacency, candidates, checkpoint["config"], args.batch_size)

        classical = None
        if args.mode in {"histgb", "hybrid"}:
            train_edges = np.concatenate((split["train_pos"], split["train_neg"]))
            train_labels = np.concatenate((np.ones(len(split["train_pos"])), np.zeros(len(split["train_neg"]))))
            estimator = build_model("hist_gb", int(first_checkpoint["config"]["seed"]) + fold)
            estimator.fit(pair_features(bundle, sets, train_edges), train_labels)
            classical = estimator.predict_proba(pair_features(bundle, sets, candidates))[:, 1]

        if args.mode == "deep":
            scores = deep
        elif args.mode == "histgb":
            scores = classical
        else:
            scores = args.deep_weight * deep + (1.0 - args.deep_weight) * classical

        current = []
        for herb, start, stop in query_slices:
            edges = candidates[start:stop]
            labels = np.asarray([(int(h), int(s)) in valid_set for h, s in edges], dtype=np.int64)
            item = ranking_metrics(labels, scores[start:stop], ks)
            item.update(fold=fold + 1, herb=int(herb), candidates=len(labels), positives=int(labels.sum()))
            query_rows.append(item)
            current.append(item)
        fold_row = {"fold": fold + 1, "queries": len(current)}
        for metric in ("ap", "mrr") + tuple(
            f"{prefix}@{k}" for k in ks for prefix in ("hit", "recall", "ndcg")
        ):
            fold_row[metric] = float(np.mean([row[metric] for row in current]))
        fold_rows.append(fold_row)
        fold += 1

    save_csv(output / "query_metrics.csv", query_rows)
    save_csv(output / "fold_metrics.csv", fold_rows)
    summary = {
        "dataset": dataset, "reference_run": args.reference_run, "mode": args.mode,
        "deep_weight": args.deep_weight if args.mode == "hybrid" else None,
        "folds": len(fold_rows), "queries": len(query_rows),
        "candidate_policy": "all symptoms excluding known non-validation positives",
        "metrics": {},
    }
    metric_names = [key for key in fold_rows[0] if key not in {"fold", "queries"}]
    for metric in metric_names:
        values = np.asarray([row[metric] for row in fold_rows])
        summary["metrics"][metric] = {"mean": float(values.mean()), "std": float(values.std())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
