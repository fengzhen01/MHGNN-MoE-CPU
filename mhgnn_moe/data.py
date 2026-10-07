"""Portable data loading, leakage-safe folds, and auditable negative sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.model_selection import KFold


@dataclass
class DatasetBundle:
    name: str
    features: torch.Tensor
    ppi_edges: np.ndarray
    positives_raw: np.ndarray
    positives_unique: np.ndarray
    incidence_edge: torch.Tensor
    incidence_vertex: torch.Tensor
    num_herbs: int
    num_symptoms: int
    num_hyperedges: int
    proximity_d: np.ndarray
    proximity_z: np.ndarray
    hyperedge_sizes: np.ndarray

    @property
    def num_proteins(self) -> int:
        return int(self.features.shape[0])

    def positives(self, protocol: str) -> np.ndarray:
        if protocol in {"strict_unique", "cold_herb", "cold_symptom", "cold_both"}:
            return self.positives_unique.copy()
        if protocol == "legacy_rows":
            return self.positives_raw.copy()
        raise ValueError(f"Unknown protocol: {protocol}")

    def pair_stats(
        self, edges: torch.Tensor, include_proximity: bool = False,
    ) -> torch.Tensor:
        """Return auxiliary pair features for the MoE router.

        Proximity coverage is label-revealing in the released data: known
        positives are absent from the proximity table while sampled unlabeled
        pairs are normally present. The safe default therefore masks the two
        legacy proximity slots while retaining checkpoint-compatible shapes.
        """
        idx = edges.detach().cpu().numpy().astype(np.int64)
        herb = idx[:, 0]
        symptom = idx[:, 1]
        d = self.proximity_d[herb, symptom]
        present = np.isfinite(d).astype(np.float32)
        finite = self.proximity_d[np.isfinite(self.proximity_d)]
        mean = float(finite.mean()) if finite.size else 0.0
        std = float(finite.std()) if finite.size else 1.0
        std = max(std, 1e-6)
        d = np.nan_to_num((d - mean) / std, nan=0.0).astype(np.float32)
        herb_size = np.log1p(self.hyperedge_sizes[herb])
        symptom_size = np.log1p(self.hyperedge_sizes[self.num_herbs + symptom])
        scale = max(float(np.log1p(self.hyperedge_sizes.max())), 1.0)
        if not include_proximity:
            d = np.zeros_like(d)
            present = np.zeros_like(present)
        result = np.stack((herb_size / scale, symptom_size / scale, d, present), axis=1)
        return torch.as_tensor(result, dtype=torch.float32, device=edges.device)

    def unlabeled_confidence(
        self, edges: np.ndarray, minimum: float = 0.25, power: float = 1.0,
    ) -> np.ndarray:
        """Estimate how confidently an unlabeled pair can be treated as negative.

        Larger network distance receives a larger weight. Missing proximity evidence
        receives the minimum weight. This is a confidence-weighted unlabeled loss,
        not a claim that sampled pairs are verified biological negatives.
        """
        result = np.full(len(edges), minimum, dtype=np.float32)
        for index, (herb_value, symptom_value) in enumerate(np.asarray(edges, dtype=np.int64)):
            herb, symptom = int(herb_value), int(symptom_value)
            value = self.proximity_d[herb, symptom]
            finite = self.proximity_d[herb]
            finite = finite[np.isfinite(finite)]
            if np.isfinite(value) and finite.size:
                percentile = float(np.mean(finite <= value))
                result[index] = minimum + (1.0 - minimum) * (percentile ** power)
        return result


def load_dataset(data_dir: Path, name: str, device: torch.device) -> DatasetBundle:
    common_features = np.load(data_dir / "protein_features.npy", mmap_mode="r")
    features = torch.as_tensor(np.array(common_features), dtype=torch.float32, device=device)
    ppi_edges = np.load(data_dir / "ppi_edges.npy").astype(np.int64)
    payload = np.load(data_dir / f"{name}.npz")
    num_herbs = int(payload["num_herbs"])
    num_symptoms = int(payload["num_symptoms"])
    num_hyperedges = int(payload["num_hyperedges"])
    incidence_edge_np = payload["incidence_edge"].astype(np.int64)
    incidence_vertex_np = payload["incidence_vertex"].astype(np.int64)
    sizes = np.bincount(incidence_edge_np, minlength=num_hyperedges).astype(np.float32)
    prox_d = np.full((num_herbs, num_symptoms), np.nan, dtype=np.float32)
    prox_z = np.full((num_herbs, num_symptoms), np.nan, dtype=np.float32)
    ph = payload["proximity_herb"].astype(np.int64)
    ps = payload["proximity_symptom"].astype(np.int64)
    valid = (ph >= 0) & (ph < num_herbs) & (ps >= 0) & (ps < num_symptoms)
    prox_d[ph[valid], ps[valid]] = payload["proximity_d"][valid]
    prox_z[ph[valid], ps[valid]] = payload["proximity_z"][valid]
    return DatasetBundle(
        name=name,
        features=features,
        ppi_edges=ppi_edges,
        positives_raw=payload["positives_raw"].astype(np.int64),
        positives_unique=payload["positives_unique"].astype(np.int64),
        incidence_edge=torch.as_tensor(incidence_edge_np, dtype=torch.long, device=device),
        incidence_vertex=torch.as_tensor(incidence_vertex_np, dtype=torch.long, device=device),
        num_herbs=num_herbs,
        num_symptoms=num_symptoms,
        num_hyperedges=num_hyperedges,
        proximity_d=prox_d,
        proximity_z=prox_z,
        hyperedge_sizes=sizes,
    )


def normalized_ppi_adjacency(edges: np.ndarray, num_nodes: int, device: torch.device) -> torch.Tensor:
    src = np.concatenate((edges[:, 0], edges[:, 1], np.arange(num_nodes)))
    dst = np.concatenate((edges[:, 1], edges[:, 0], np.arange(num_nodes)))
    indices = torch.as_tensor(np.stack((src, dst)), dtype=torch.long, device=device)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    adjacency = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    row = adjacency.indices()[0]
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.index_add_(0, row, adjacency.values())
    normalized = adjacency.values() / degree[row].clamp_min(1.0)
    return torch.sparse_coo_tensor(adjacency.indices(), normalized, adjacency.shape).coalesce()


def make_folds(positives: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return [(positives[train], positives[valid]) for train, valid in splitter.split(positives)]


def _entity_fold_labels(values: np.ndarray, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    entities = np.unique(values)
    if len(entities) < folds:
        raise ValueError(f"Cannot create {folds} folds from only {len(entities)} entities")
    counts = np.asarray([np.sum(values == entity) for entity in entities], dtype=np.int64)
    rng = np.random.default_rng(seed)
    tie_break = rng.random(len(entities))
    order = np.lexsort((tie_break, -counts))
    fold_loads = np.zeros(folds, dtype=np.int64)
    fold_sizes = np.zeros(folds, dtype=np.int64)
    labels = np.empty(len(entities), dtype=np.int64)
    for index in order:
        candidates = np.flatnonzero(fold_loads == fold_loads.min())
        if len(candidates) > 1:
            min_size = fold_sizes[candidates].min()
            candidates = candidates[fold_sizes[candidates] == min_size]
        selected = int(candidates[int(rng.integers(len(candidates)))])
        labels[index] = selected
        fold_loads[selected] += counts[index]
        fold_sizes[selected] += 1
    return entities, labels


def _entity_partition(values: np.ndarray, folds: int, fold: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    entities, labels = _entity_fold_labels(values, folds, seed)
    return entities[labels != fold], entities[labels == fold]


def _balanced_double_cold_groups(
    positives: np.ndarray, folds: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    herbs, herb_labels = _entity_fold_labels(positives[:, 0], folds, seed)
    symptoms, symptom_labels = _entity_fold_labels(positives[:, 1], folds, seed + 7919)
    herb_lookup = dict(zip(herbs.tolist(), herb_labels.tolist()))
    symptom_lookup = dict(zip(symptoms.tolist(), symptom_labels.tolist()))
    matrix = np.zeros((folds, folds), dtype=np.int64)
    for herb, symptom in positives:
        matrix[herb_lookup[int(herb)], symptom_lookup[int(symptom)]] += 1

    import itertools

    best_permutation = None
    best_score = None
    for permutation in itertools.permutations(range(folds)):
        diagonal = np.asarray([matrix[index, permutation[index]] for index in range(folds)])
        score = (int(diagonal.min()), -int(diagonal.max() - diagonal.min()), int(diagonal.sum()))
        if best_score is None or score > best_score:
            best_score, best_permutation = score, permutation
    remap = np.empty(folds, dtype=np.int64)
    for herb_fold, symptom_fold in enumerate(best_permutation):
        remap[symptom_fold] = herb_fold
    symptom_labels = remap[symptom_labels]
    return herbs, herb_labels, symptoms, symptom_labels


def make_protocol_split(
    positives: np.ndarray, protocol: str, folds: int, fold: int, seed: int,
) -> dict[str, np.ndarray]:
    if protocol in {"strict_unique", "legacy_rows"}:
        train_pos, valid_pos = make_folds(positives, folds, seed)[fold]
        return {
            "train_pos": train_pos, "valid_pos": valid_pos,
            "heldout_herbs": np.empty(0, dtype=np.int64),
            "heldout_symptoms": np.empty(0, dtype=np.int64),
            "dropped_pos": np.empty((0, 2), dtype=np.int64),
        }
    if protocol == "cold_herb":
        _, heldout_herbs = _entity_partition(positives[:, 0], folds, fold, seed)
        valid_mask = np.isin(positives[:, 0], heldout_herbs)
        return {
            "train_pos": positives[~valid_mask], "valid_pos": positives[valid_mask],
            "heldout_herbs": heldout_herbs,
            "heldout_symptoms": np.empty(0, dtype=np.int64),
            "dropped_pos": np.empty((0, 2), dtype=np.int64),
        }
    if protocol == "cold_symptom":
        _, heldout_symptoms = _entity_partition(positives[:, 1], folds, fold, seed)
        valid_mask = np.isin(positives[:, 1], heldout_symptoms)
        return {
            "train_pos": positives[~valid_mask], "valid_pos": positives[valid_mask],
            "heldout_herbs": np.empty(0, dtype=np.int64),
            "heldout_symptoms": heldout_symptoms,
            "dropped_pos": np.empty((0, 2), dtype=np.int64),
        }
    if protocol == "cold_both":
        herbs, herb_labels, symptoms, symptom_labels = _balanced_double_cold_groups(positives, folds, seed)
        heldout_herbs = herbs[herb_labels == fold]
        heldout_symptoms = symptoms[symptom_labels == fold]
        herb_held = np.isin(positives[:, 0], heldout_herbs)
        symptom_held = np.isin(positives[:, 1], heldout_symptoms)
        train_mask = ~herb_held & ~symptom_held
        valid_mask = herb_held & symptom_held
        dropped_mask = ~(train_mask | valid_mask)
        train_pos, valid_pos = positives[train_mask], positives[valid_mask]
        if len(valid_pos) < 2:
            raise ValueError(
                f"cold_both fold {fold + 1} has only {len(valid_pos)} positives; "
                "use fewer folds"
            )
        return {
            "train_pos": train_pos, "valid_pos": valid_pos,
            "heldout_herbs": heldout_herbs, "heldout_symptoms": heldout_symptoms,
            "dropped_pos": positives[dropped_mask],
        }
    raise ValueError(f"Unknown protocol: {protocol}")


def _ranked_candidates(
    bundle: DatasetBundle, herb: int, known: set[tuple[int, int]], top_n: int,
    allowed_symptoms: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = bundle.proximity_d[herb]
    symptom_values = range(bundle.num_symptoms) if allowed_symptoms is None else allowed_symptoms
    candidates = np.asarray(
        [int(s) for s in symptom_values if (herb, int(s)) not in known], dtype=np.int64
    )
    if candidates.size == 0:
        return candidates, candidates, candidates
    values = distances[candidates]
    finite_mask = np.isfinite(values)
    finite_candidates = candidates[finite_mask]
    finite_values = values[finite_mask]
    if finite_candidates.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), candidates
    order = np.argsort(finite_values)
    count = min(top_n, len(order))
    closest = finite_candidates[order[:count]]
    farthest = finite_candidates[order[-count:]]
    ranked = np.union1d(closest, farthest)
    rest = np.setdiff1d(candidates, ranked, assume_unique=False)
    return closest, farthest, rest


def sample_negatives(
    bundle: DatasetBundle,
    positive_rows: np.ndarray,
    all_known: np.ndarray,
    seed: int,
    top_n: int,
    network_probability: float,
    unique: bool,
    reserved: Iterable[tuple[int, int]] = (),
    allowed_symptoms: np.ndarray | None = None,
    strategy: str = "mixed_proximity",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    known = set(map(tuple, all_known.tolist()))
    used = set(reserved)
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    negatives: list[tuple[int, int]] = []
    for herb_value in positive_rows[:, 0]:
        herb = int(herb_value)
        if herb not in cache:
            cache[herb] = _ranked_candidates(bundle, herb, known, top_n, allowed_symptoms)
        closest, farthest, rest = cache[herb]
        not_closest = np.union1d(rest, farthest)
        not_farthest = np.union1d(rest, closest)
        if strategy == "network_close":
            pools = (closest, not_closest)
        elif strategy == "network_far":
            pools = (farthest, not_farthest)
        elif strategy == "mixed_proximity":
            pools = ((farthest, not_farthest) if rng.random() < network_probability
                     else (not_farthest, farthest))
        else:
            raise ValueError(f"Unknown proximity negative strategy: {strategy}")
        chosen = None
        for pool in pools:
            if pool.size == 0:
                continue
            order = rng.permutation(pool)
            for symptom in order:
                pair = (herb, int(symptom))
                if not unique or pair not in used:
                    chosen = pair
                    break
            if chosen is not None:
                break
        if chosen is None:
            symptom_values = range(bundle.num_symptoms) if allowed_symptoms is None else allowed_symptoms
            fallback = [
                (herb, symptom)
                for symptom_value in symptom_values
                for symptom in (int(symptom_value),)
                if (herb, symptom) not in known and (not unique or (herb, symptom) not in used)
            ]
            if not fallback:
                raise RuntimeError(f"No negative candidate remains for herb {herb}")
            chosen = fallback[int(rng.integers(len(fallback)))]
        negatives.append(chosen)
        used.add(chosen)
    return np.asarray(negatives, dtype=np.int64)


def sample_uniform_pairs(
    bundle: DatasetBundle, count: int, all_known: np.ndarray, seed: int, unique: bool,
    reserved: Iterable[tuple[int, int]] = (), allowed_herbs: np.ndarray | None = None,
    allowed_symptoms: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    herbs = np.arange(bundle.num_herbs) if allowed_herbs is None else np.asarray(allowed_herbs)
    symptoms = np.arange(bundle.num_symptoms) if allowed_symptoms is None else np.asarray(allowed_symptoms)
    herb_set, symptom_set = set(herbs.tolist()), set(symptoms.tolist())
    known = set(map(tuple, all_known.tolist()))
    used = set(reserved)
    available = len(herbs) * len(symptoms) - sum(
        int(herb in herb_set and symptom in symptom_set) for herb, symptom in known
    )
    if unique and count > available - len(used):
        raise RuntimeError(f"Requested {count} unique unlabeled pairs but only {available - len(used)} remain")
    result: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(10000, count * 100)
    while len(result) < count and attempts < max_attempts:
        pair = (int(rng.choice(herbs)), int(rng.choice(symptoms)))
        attempts += 1
        if pair in known or (unique and pair in used):
            continue
        result.append(pair)
        used.add(pair)
    if len(result) != count:
        raise RuntimeError(f"Could sample only {len(result)} of {count} requested unlabeled pairs")
    return np.asarray(result, dtype=np.int64)


def load_or_create_split(
    bundle: DatasetBundle,
    protocol: str,
    folds: int,
    fold: int,
    seed: int,
    top_n: int,
    network_probability: float,
    unique_negatives: bool,
    split_dir: Path,
    train_negative_strategy: str = "mixed_proximity",
    validation_negative_strategy: str = "mixed_proximity",
    train_unlabeled_ratio: int = 1,
) -> dict[str, np.ndarray]:
    split_dir.mkdir(parents=True, exist_ok=True)
    probability_tag = str(network_probability).replace(".", "p")
    path = split_dir / (
        f"{bundle.name}_{protocol}_k{folds}_seed{seed}_n{top_n}_"
        f"p{probability_tag}_u{int(unique_negatives)}_tr{train_negative_strategy}_"
        f"vr{validation_negative_strategy}_r{train_unlabeled_ratio}_fold{fold}.npz"
    )
    if path.exists():
        return {key: value for key, value in np.load(path).items()}
    positives = bundle.positives(protocol)
    split = make_protocol_split(positives, protocol, folds, fold, seed)
    train_pos, valid_pos = split["train_pos"], split["valid_pos"]
    all_known = bundle.positives_unique
    heldout_symptoms = split["heldout_symptoms"]
    heldout_herbs = split["heldout_herbs"]
    if heldout_herbs.size:
        train_allowed_herbs = np.setdiff1d(np.arange(bundle.num_herbs), heldout_herbs)
        valid_allowed_herbs = heldout_herbs
    else:
        train_allowed_herbs = valid_allowed_herbs = None
    if heldout_symptoms.size:
        train_allowed_symptoms = np.setdiff1d(np.arange(bundle.num_symptoms), heldout_symptoms)
        valid_allowed_symptoms = heldout_symptoms
    else:
        train_allowed_symptoms = valid_allowed_symptoms = None
    if train_negative_strategy == "uniform":
        train_neg = sample_uniform_pairs(
            bundle, len(train_pos) * int(train_unlabeled_ratio), all_known,
            seed + fold * 1009 + 1, unique_negatives,
            allowed_herbs=train_allowed_herbs, allowed_symptoms=train_allowed_symptoms,
        )
    elif train_negative_strategy in {"mixed_proximity", "network_close", "network_far"}:
        source = np.repeat(train_pos, int(train_unlabeled_ratio), axis=0)
        train_neg = sample_negatives(
            bundle, source, all_known, seed + fold * 1009 + 1, top_n,
            network_probability, unique_negatives, allowed_symptoms=train_allowed_symptoms,
            strategy=train_negative_strategy,
        )
    else:
        raise ValueError(f"Unknown train_negative_strategy: {train_negative_strategy}")
    reserved = map(tuple, train_neg.tolist())
    if validation_negative_strategy == "uniform":
        valid_neg = sample_uniform_pairs(
            bundle, len(valid_pos), all_known, seed + fold * 1009 + 2, unique_negatives,
            reserved=reserved, allowed_herbs=valid_allowed_herbs,
            allowed_symptoms=valid_allowed_symptoms,
        )
    elif validation_negative_strategy in {"mixed_proximity", "network_close", "network_far"}:
        valid_neg = sample_negatives(
            bundle, valid_pos, all_known, seed + fold * 1009 + 2, top_n,
            network_probability, unique_negatives, reserved=reserved,
            allowed_symptoms=valid_allowed_symptoms, strategy=validation_negative_strategy,
        )
    else:
        raise ValueError(f"Unknown validation_negative_strategy: {validation_negative_strategy}")
    split.update(train_neg=train_neg, valid_neg=valid_neg)
    np.savez_compressed(path, **split)
    return split


def audit_split(bundle: DatasetBundle, split: dict[str, np.ndarray], protocol: str) -> dict[str, int | bool]:
    train_pos = set(map(tuple, split["train_pos"].tolist()))
    valid_pos = set(map(tuple, split["valid_pos"].tolist()))
    train_neg = set(map(tuple, split["train_neg"].tolist()))
    valid_neg = set(map(tuple, split["valid_neg"].tolist()))
    known = set(map(tuple, bundle.positives_unique.tolist()))
    checks = {
        "train_valid_positive_disjoint": train_pos.isdisjoint(valid_pos),
        "positive_negative_disjoint": (train_pos | valid_pos).isdisjoint(train_neg | valid_neg),
        "negative_not_known_positive": (train_neg | valid_neg).isdisjoint(known),
        "train_valid_negative_disjoint": train_neg.isdisjoint(valid_neg),
    }
    held_herbs = set(map(int, split.get("heldout_herbs", np.empty(0, dtype=np.int64)).tolist()))
    held_symptoms = set(map(int, split.get("heldout_symptoms", np.empty(0, dtype=np.int64)).tolist()))
    if protocol in {"cold_herb", "cold_both"}:
        checks["cold_herb_train_excludes_heldout"] = all(h not in held_herbs for h, _ in train_pos | train_neg)
        checks["cold_herb_valid_only_heldout"] = all(h in held_herbs for h, _ in valid_pos | valid_neg)
    if protocol in {"cold_symptom", "cold_both"}:
        checks["cold_symptom_train_excludes_heldout"] = all(s not in held_symptoms for _, s in train_pos | train_neg)
        checks["cold_symptom_valid_only_heldout"] = all(s in held_symptoms for _, s in valid_pos | valid_neg)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Split audit failed: {', '.join(failed)}")
    return {
        **checks,
        "train_positives": len(split["train_pos"]),
        "valid_positives": len(split["valid_pos"]),
        "train_negatives": len(split["train_neg"]),
        "valid_negatives": len(split["valid_neg"]),
        "heldout_herbs": len(held_herbs),
        "heldout_symptoms": len(held_symptoms),
        "dropped_positives": len(split.get("dropped_pos", ())),
    }
