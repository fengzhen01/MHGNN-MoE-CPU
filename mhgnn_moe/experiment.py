"""Training and cross-validation orchestration with auditable artifacts."""

from __future__ import annotations

import copy
import csv
import json
import logging
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F

from .data import (
    audit_split, load_dataset, load_or_create_split, normalized_ppi_adjacency,
    sample_uniform_pairs,
)
from .metrics import binary_metrics
from .model import MHGNNMoE


LOGGER = logging.getLogger("mhgnn_moe")


def router_pair_stats(bundle, edges: torch.Tensor, config: dict) -> torch.Tensor:
    """Build router features with label-revealing proximity masked by default."""
    return bundle.pair_stats(
        edges, include_proximity=bool(config.get("allow_proximity_router", False))
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_edges(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.long, device=device)


def combine_edges(
    pos: np.ndarray, neg: np.ndarray, device: torch.device, bundle=None, config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edges = np.concatenate((pos, neg), axis=0)
    labels = np.concatenate((np.ones(len(pos)), np.zeros(len(neg)))).astype(np.float32)
    weights = np.ones(len(edges), dtype=np.float32)
    config = config or {}
    if config.get("loss_mode", "bce") == "confidence_bce":
        if bundle is None:
            raise ValueError("bundle is required for confidence_bce")
        weights[len(pos):] = bundle.unlabeled_confidence(
            neg,
            minimum=float(config.get("negative_confidence_min", 0.25)),
            power=float(config.get("negative_confidence_power", 1.0)),
        )
    return (
        tensor_edges(edges, device),
        torch.as_tensor(labels, dtype=torch.float32, device=device),
        torch.as_tensor(weights, dtype=torch.float32, device=device),
    )


def classification_loss(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, config: dict,
) -> torch.Tensor:
    mode = config.get("loss_mode", "bce")
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if mode == "bce":
        return raw.mean()
    if mode == "confidence_bce":
        positive = labels > 0.5
        negative = ~positive
        positive_loss = raw[positive].mean()
        negative_loss = (raw[negative] * weights[negative]).sum() / weights[negative].sum().clamp_min(1e-8)
        return 0.5 * (positive_loss + negative_loss)
    if mode == "pairwise_bpr":
        positive = labels > 0.5
        negative = ~positive
        positive_logits = logits[positive]
        negative_logits = logits[negative]
        if not len(positive_logits) or len(negative_logits) % len(positive_logits):
            raise ValueError(
                "pairwise_bpr requires an integer number of ordered negatives per positive"
            )
        negatives_per_positive = len(negative_logits) // len(positive_logits)
        negative_logits = negative_logits.reshape(len(positive_logits), negatives_per_positive)
        ranking = F.softplus(-(positive_logits[:, None] - negative_logits)).mean()
        positive_bce = raw[positive].mean()
        negative_bce = raw[negative].mean()
        auxiliary_weight = float(config.get("ranking_bce_weight", 0.2))
        return ranking + auxiliary_weight * 0.5 * (positive_bce + negative_bce)
    if mode == "nnpu":
        positive = labels > 0.5
        unlabeled = ~positive
        prior = float(config["positive_class_prior"])
        beta = float(config.get("nnpu_beta", 0.0))
        gamma = float(config.get("nnpu_gamma", 1.0))
        positive_risk = prior * F.softplus(-logits[positive]).mean()
        negative_risk = F.softplus(logits[unlabeled]).mean() - prior * F.softplus(logits[positive]).mean()
        if float(negative_risk.detach()) < -beta:
            return -gamma * negative_risk
        return positive_risk + negative_risk
    raise ValueError(f"Unknown loss_mode: {mode}")


@torch.no_grad()
def evaluate(
    model, bundle, adjacency, edges, labels, threshold: float, config: dict,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    output = model(
        bundle.features, adjacency, bundle.incidence_edge, bundle.incidence_vertex,
        edges, router_pair_stats(bundle, edges, config),
    )
    probabilities = torch.sigmoid(output["logits"]).cpu().numpy()
    metrics = binary_metrics(labels.cpu().numpy(), probabilities, threshold)
    metrics["residual_strength"] = float(output["residual_strength"].detach().cpu())
    router = output["router_probs"].detach().cpu().numpy()
    if router.size:
        for index, value in enumerate(router.mean(axis=0)):
            metrics[f"expert_{index}_usage"] = float(value)
    return metrics, probabilities, router


def train_stage(
    model, stage: str, epochs: int, learning_rate: float, config: dict,
    bundle, adjacency, train_edges, train_labels, train_weights, valid_edges, valid_labels,
) -> list[dict]:
    if epochs <= 0:
        return []
    model.set_stage(stage)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=config["weight_decay"])
    use_early_stopping = bool(config.get("early_stopping", False))
    best_value = -float("inf")
    best_state = copy.deepcopy(model.state_dict()) if use_early_stopping else None
    patience = 0
    history: list[dict] = []
    metric_name = config.get("early_stopping_metric", "auprc")
    cached_train = None
    if stage == "moe":
        model.eval()
        with torch.no_grad():
            low, high = model.encode(
                bundle.features, adjacency, bundle.incidence_edge, bundle.incidence_vertex
            )
            base_logits = model.forward_from_encoded(
                low, high, train_edges, router_pair_stats(bundle, train_edges, config)
            )["base_logits"]
        cached_train = (low.detach(), high.detach(), base_logits.detach())

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if cached_train is None:
            output = model(
                bundle.features, adjacency, bundle.incidence_edge, bundle.incidence_vertex,
                train_edges, router_pair_stats(bundle, train_edges, config),
            )
        else:
            output = model.forward_from_encoded(
                cached_train[0], cached_train[1], train_edges,
                router_pair_stats(bundle, train_edges, config), base_logits_override=cached_train[2],
            )
        loss = classification_loss(
            output["logits"], train_labels, train_weights, config
        )
        if model.moe_enabled:
            loss = loss + config.get("balance_loss_weight", 0.0) * output["balance_loss"]
            teacher = torch.sigmoid(output["base_logits"].detach())
            student = torch.sigmoid(output["logits"])
            loss = loss + config.get("distillation_weight", 0.0) * F.mse_loss(student, teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()

        row = {"stage": stage, "epoch": epoch + 1, "loss": float(loss.detach().cpu())}
        if use_early_stopping:
            valid_metrics, _, _ = evaluate(
                model, bundle, adjacency, valid_edges, valid_labels, config["threshold"], config
            )
            row.update(valid_metrics)
        history.append(row)
        if use_early_stopping:
            score = valid_metrics[metric_name]
            if score > best_value + 1e-8:
                best_value = score
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            if use_early_stopping:
                LOGGER.info(
                    "%s epoch %d/%d loss=%.4f %s=%.4f",
                    stage, epoch + 1, epochs, row["loss"], metric_name, score,
                )
            else:
                LOGGER.info("%s epoch %d/%d loss=%.4f", stage, epoch + 1, epochs, row["loss"])
        if use_early_stopping and patience >= config.get("early_stopping_patience", epochs + 1):
            LOGGER.info("Early stopping %s at epoch %d", stage, epoch + 1)
            break
    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: dict, project_root: Path, run_dir: Path) -> dict:
    device_name = config.get("device", "cpu")
    if device_name != "cpu" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        device_name = "cpu"
    device = torch.device(device_name)
    threads = int(config.get("cpu_threads", min(os.cpu_count() or 1, 8)))
    if device.type == "cpu":
        torch.set_num_threads(max(1, threads))
    set_seed(int(config["seed"]))
    data_dir = project_root / "data" / "processed"
    bundle = load_dataset(data_dir, config["dataset"], device)
    adjacency = normalized_ppi_adjacency(bundle.ppi_edges, bundle.num_proteins, device)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "splits").mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "config.resolved.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "device": str(device),
        "cpu_threads": threads,
        "cuda_available": torch.cuda.is_available(),
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    folds_value = config.get("fold_indices", "all")
    fold_indices = list(range(config["folds"])) if folds_value == "all" else [int(x) for x in folds_value]
    fold_rows: list[dict] = []
    start_all = time.time()
    for fold in fold_indices:
        LOGGER.info("Dataset=%s fold=%d/%d model=%s", bundle.name, fold + 1, config["folds"], config["model"])
        set_seed(int(config["seed"]) + fold)
        reference_run = config.get("reference_split_run")
        if reference_run:
            reference_dir = project_root / "results" / reference_run / "splits"
            matches = sorted(reference_dir.glob(f"*_fold{fold}.npz"))
            if not matches:
                raise FileNotFoundError(f"No reference split for fold {fold}: {reference_dir}")
            payload = np.load(matches[-1])
            split = {key: payload[key] for key in payload.files}
            split.setdefault("heldout_herbs", np.empty(0, dtype=np.int64))
            split.setdefault("heldout_symptoms", np.empty(0, dtype=np.int64))
            split.setdefault("dropped_pos", np.empty((0, 2), dtype=np.int64))
            if config.get("train_negative_strategy") == "uniform":
                heldout_herbs = split["heldout_herbs"]
                heldout_symptoms = split["heldout_symptoms"]
                allowed_herbs = (
                    np.setdiff1d(np.arange(bundle.num_herbs), heldout_herbs)
                    if heldout_herbs.size else None
                )
                allowed_symptoms = (
                    np.setdiff1d(np.arange(bundle.num_symptoms), heldout_symptoms)
                    if heldout_symptoms.size else None
                )
                split["train_neg"] = sample_uniform_pairs(
                    bundle,
                    len(split["train_pos"]) * int(config.get("train_unlabeled_ratio", 1)),
                    bundle.positives_unique,
                    int(config["seed"]) + fold * 1009 + 31,
                    bool(config["unique_negatives"]),
                    reserved=map(tuple, split["valid_neg"].tolist()),
                    allowed_herbs=allowed_herbs,
                    allowed_symptoms=allowed_symptoms,
                )
            np.savez_compressed(
                run_dir / "splits" / f"reference_{reference_run}_fold{fold}.npz", **split
            )
        else:
            split = load_or_create_split(
                bundle=bundle,
                protocol=config["protocol"], folds=config["folds"], fold=fold,
                seed=config["seed"], top_n=config["negative_top_n"],
                network_probability=config["network_negative_probability"],
                unique_negatives=config["unique_negatives"], split_dir=run_dir / "splits",
                train_negative_strategy=config.get("train_negative_strategy", "mixed_proximity"),
                validation_negative_strategy=config.get("validation_negative_strategy", "mixed_proximity"),
                train_unlabeled_ratio=int(config.get("train_unlabeled_ratio", 1)),
            )
        split_audit = audit_split(bundle, split, config["protocol"])
        (run_dir / "splits" / f"audit_fold_{fold + 1}.json").write_text(
            json.dumps(split_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        train_edges, train_labels, train_weights = combine_edges(
            split["train_pos"], split["train_neg"], device, bundle, config
        )
        valid_edges, valid_labels, valid_weights = combine_edges(
            split["valid_pos"], split["valid_neg"], device, bundle, config
        )
        model = MHGNNMoE(
            input_dim=bundle.features.shape[1], hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"], ppi_layers=config["ppi_layers"],
            hyper_layers=config["hyper_layers"], dropout=config["dropout"],
            num_herbs=bundle.num_herbs, num_hyperedges=bundle.num_hyperedges,
            num_experts=config.get("num_experts", 3),
            router_temperature=config.get("router_temperature", 1.0),
            initial_residual_strength=config.get("initial_residual_strength", 0.1),
        ).to(device)
        history: list[dict] = []
        fold_start = time.time()
        if config["model"] == "baseline":
            history += train_stage(
                model, "pretrain", config["epochs"], config["learning_rate"], config,
                bundle, adjacency, train_edges, train_labels, train_weights, valid_edges, valid_labels,
            )
            model.moe_enabled = False
        elif config["model"] == "moe":
            baseline_run = config.get("baseline_run")
            if baseline_run:
                checkpoint_path = project_root / "results" / baseline_run / "checkpoints" / f"fold_{fold + 1}.pt"
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Baseline checkpoint not found: {checkpoint_path}. "
                        "Complete the paired baseline run before MoE."
                    )
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state"])
                strength = float(config.get("initial_residual_strength", 0.1))
                strength = min(max(strength, 1e-4), 1.0 - 1e-4)
                with torch.no_grad():
                    model.residual_logit.fill_(math.log(strength / (1.0 - strength)))
                LOGGER.info("Loaded paired baseline checkpoint: %s", checkpoint_path)
            else:
                history += train_stage(
                    model, "pretrain", config["pretrain_epochs"], config["learning_rate"], config,
                    bundle, adjacency, train_edges, train_labels, train_weights, valid_edges, valid_labels,
                )
            history += train_stage(
                model, "moe", config["moe_epochs"], config["moe_learning_rate"], config,
                bundle, adjacency, train_edges, train_labels, train_weights, valid_edges, valid_labels,
            )
            history += train_stage(
                model, "finetune", config["finetune_epochs"], config["finetune_learning_rate"], config,
                bundle, adjacency, train_edges, train_labels, train_weights, valid_edges, valid_labels,
            )
            model.moe_enabled = True
        else:
            raise ValueError("model must be baseline or moe")

        metrics, probabilities, router = evaluate(
            model, bundle, adjacency, valid_edges, valid_labels, config["threshold"], config
        )
        metrics.update(
            fold=fold + 1,
            seconds=time.time() - fold_start,
            train_positives=len(split["train_pos"]),
            valid_positives=len(split["valid_pos"]),
            heldout_herbs=split_audit["heldout_herbs"],
            heldout_symptoms=split_audit["heldout_symptoms"],
            dropped_positives=split_audit["dropped_positives"],
            train_negative_confidence=float(train_weights[train_labels < 0.5].mean().cpu()),
        )
        fold_rows.append(metrics)
        save_csv(run_dir / f"history_fold_{fold + 1}.csv", history)
        torch.save(
            {"model_state": model.state_dict(), "moe_enabled": model.moe_enabled, "config": config},
            run_dir / "checkpoints" / f"fold_{fold + 1}.pt",
        )
        if config.get("save_predictions", True):
            prediction_rows = []
            edge_np = valid_edges.cpu().numpy()
            label_np = valid_labels.cpu().numpy().astype(int)
            for index, (edge, label, probability) in enumerate(zip(edge_np, label_np, probabilities)):
                row = {
                    "herb_id": int(edge[0]), "symptom_id": int(edge[1]),
                    "label": int(label), "probability": float(probability),
                    "unlabeled_confidence": float(valid_weights[index].cpu()),
                }
                if router.size:
                    for expert, weight in enumerate(router[index]):
                        row[f"expert_{expert}_weight"] = float(weight)
                prediction_rows.append(row)
            save_csv(run_dir / "predictions" / f"fold_{fold + 1}.csv", prediction_rows)
        save_csv(run_dir / "fold_metrics.csv", fold_rows)

    numeric_keys = [
        "accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc",
        "residual_strength", "seconds", "expert_0_usage", "expert_1_usage", "expert_2_usage",
        "train_negative_confidence", "heldout_herbs", "heldout_symptoms", "dropped_positives",
    ]
    summary = {"dataset": config["dataset"], "model": config["model"], "folds_completed": len(fold_rows)}
    for key in numeric_keys:
        values = [row[key] for row in fold_rows if key in row]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    summary["total_seconds"] = time.time() - start_all
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary
