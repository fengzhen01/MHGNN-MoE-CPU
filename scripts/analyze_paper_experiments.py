"""Generate paper-ready diagnostic, interpretability, and case-analysis artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_full_ranking import build_deep_model, deep_scores, ranking_metrics
from mhgnn_moe.classical import (
    STRUCTURAL_FEATURE_NAMES,
    build_model,
    load_split,
    pair_features,
    target_sets,
)
from mhgnn_moe.data import load_dataset, normalized_ppi_adjacency


RESULTS = ROOT / "results"
OUT = RESULTS / "paper_additional_experiments"
FIG = RESULTS / "figures"
SOURCE = ROOT / "data" / "source"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

COLORS = {"MHGNN-BCE": "#4C78A8", "HistGB": "#F2A541", "Fixed fusion": "#3A9D7D"}
FEATURE_LABELS = {
    "log_herb_targets": "Log herb target count",
    "log_symptom_targets": "Log symptom target count",
    "target_intersection": "Log shared-target count",
    "target_jaccard": "Target Jaccard similarity",
    "target_cosine": "Target cosine similarity",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def read_prediction_dir(run: str) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for path in sorted((RESULTS / run / "predictions").glob("fold_*.csv")):
        frame = pd.read_csv(path)
        rows.append(frame[["label", "probability"]])
    data = pd.concat(rows, ignore_index=True)
    return data["label"].to_numpy(int), data["probability"].to_numpy(float)


def plot_roc_pr() -> dict:
    runs = {
        "HIT": {
            "MHGNN-BCE": "hit_baseline_strict_full",
            "HistGB": "hit_histgb_structural_strict",
            "Fixed fusion": "hit_baseline_histgb_ensemble",
        },
        "TCMIO": {
            "MHGNN-BCE": "tcmio_baseline_strict_full",
            "HistGB": "tcmio_histgb_structural_strict",
            "Fixed fusion": "tcmio_baseline_histgb_ensemble",
        },
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.5))
    summary = {}
    for row, dataset in enumerate(("HIT", "TCMIO")):
        roc_ax, pr_ax = axes[row]
        summary[dataset] = {}
        prevalence = None
        for method, run in runs[dataset].items():
            labels, scores = read_prediction_dir(run)
            prevalence = float(labels.mean())
            fpr, tpr, _ = roc_curve(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
            summary[dataset][method] = {"auroc": float(auroc), "auprc": float(auprc)}
            roc_ax.plot(fpr, tpr, lw=1.8, color=COLORS[method], label=f"{method} ({auroc:.3f})")
            pr_ax.plot(recall, precision, lw=1.8, color=COLORS[method], label=f"{method} ({auprc:.3f})")
        roc_ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1)
        pr_ax.axhline(prevalence, ls="--", color="#777777", lw=1, label=f"Prevalence ({prevalence:.2f})")
        roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{dataset}: ROC")
        pr_ax.set(xlabel="Recall", ylabel="Precision", title=f"{dataset}: precision–recall")
        for ax in (roc_ax, pr_ax):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(ls="--", alpha=0.25)
            ax.legend(frameon=False, fontsize=8, loc="lower right" if ax is roc_ax else "lower left")
    fig.suptitle("Out-of-fold discrimination under balanced sampled evaluation", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "oof_roc_pr_curves")
    (OUT / "oof_roc_pr_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def association_degrees(bundle) -> tuple[np.ndarray, np.ndarray]:
    positives = bundle.positives_unique
    return (
        np.bincount(positives[:, 0], minlength=bundle.num_herbs),
        np.bincount(positives[:, 1], minlength=bundle.num_symptoms),
    )


def plot_biological_landscape() -> dict:
    bundles = {name: load_dataset(ROOT / "data" / "processed", name, torch.device("cpu")) for name in ("HIT", "TCMIO")}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    summary = {}
    for row, (name, bundle) in enumerate(bundles.items()):
        herb_degree, symptom_degree = association_degrees(bundle)
        summary[name] = {
            "herb_degree_median": float(np.median(herb_degree)),
            "herb_degree_max": int(herb_degree.max()),
            "symptom_degree_median": float(np.median(symptom_degree)),
            "symptom_degree_max": int(symptom_degree.max()),
            "zero_degree_herbs": int(np.sum(herb_degree == 0)),
        }
        for ax, values, label, color in (
            (axes[row, 0], herb_degree, "Herbs", "#4C78A8"),
            (axes[row, 1], symptom_degree, "Symptoms", "#D97941"),
        ):
            bins = np.arange(0, values.max() + 2) - 0.5
            ax.hist(values, bins=bins, color=color, alpha=0.88)
            ax.axvline(np.median(values), color="#333333", ls="--", lw=1, label=f"Median={np.median(values):.1f}")
            ax.set_yscale("log")
            ax.set_xlabel("Number of known associations")
            ax.set_ylabel("Entity count (log scale)")
            ax.set_title(f"{name}: {label.lower()}")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(axis="y", ls="--", alpha=0.25)
    fig.suptitle("Association-degree distributions in the two benchmark datasets", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "association_degree_distributions")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    rng = np.random.default_rng(42)
    feature_summary = {}
    for ax, (name, bundle) in zip(axes, bundles.items()):
        sets = target_sets(bundle)
        positives = bundle.positives_unique
        known = set(map(tuple, positives.tolist()))
        total = bundle.num_herbs * bundle.num_symptoms
        sample_count = min(10000, total - len(known))
        unlabeled = []
        while len(unlabeled) < sample_count:
            herbs = rng.integers(0, bundle.num_herbs, size=sample_count)
            symptoms = rng.integers(0, bundle.num_symptoms, size=sample_count)
            for pair in zip(herbs.tolist(), symptoms.tolist()):
                if pair not in known:
                    unlabeled.append(pair)
                if len(unlabeled) >= sample_count:
                    break
        unlabeled = np.asarray(unlabeled, dtype=np.int64)
        pos_x = pair_features(bundle, sets, positives)
        unl_x = pair_features(bundle, sets, unlabeled)
        # pair_features stores log1p(|T_h \u2229 T_s|), rather than the raw overlap count.
        labels = ["Log shared targets", "Jaccard", "Cosine"]
        indices = [2, 3, 4]
        positions = np.arange(3)
        width = 0.34
        pos_means = [pos_x[:, i].mean() for i in indices]
        unl_means = [unl_x[:, i].mean() for i in indices]
        ax.bar(positions - width / 2, pos_means, width, label="Known associations", color="#3A9D7D")
        ax.bar(positions + width / 2, unl_means, width, label="Random unlabeled pairs", color="#B7B7B7")
        ax.set_xticks(positions, labels)
        ax.set_title(name)
        ax.grid(axis="y", ls="--", alpha=0.25)
        feature_summary[name] = {
            labels[j]: {"positive_mean": float(pos_means[j]), "unlabeled_mean": float(unl_means[j])}
            for j in range(3)
        }
    axes[0].set_ylabel("Mean feature value")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Target-set overlap in known and randomly sampled unlabeled pairs", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "target_overlap_bioinformatics")
    summary["target_overlap"] = feature_summary
    (OUT / "bioinformatics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_hit_names() -> tuple[dict[int, str], dict[int, str]]:
    herb_index = pd.read_excel(SOURCE / "HIT" / "original_data" / "herb.xlsx")
    herb_names = pd.read_csv(SOURCE / "HIT" / "original_data" / "herb_chinese.csv", encoding="gb18030")
    herb = herb_index.merge(herb_names, on="tcm_id", how="left")
    herb_map = {
        int(row.node_id): f"{row.herb_name} ({row.tcm_id})" if pd.notna(row.herb_name) else str(row.tcm_id)
        for row in herb.itertuples()
    }
    symptom = pd.read_excel(SOURCE / "HIT" / "original_data" / "symptom.xlsx")
    symptom_map = {int(row.node_id): str(row.symptom) for row in symptom.itertuples()}
    return herb_map, symptom_map


def plot_hit_network() -> None:
    bundle = load_dataset(ROOT / "data" / "processed", "HIT", torch.device("cpu"))
    herb_map, symptom_map = load_hit_names()
    positives = bundle.positives_unique
    herb_degree, symptom_degree = association_degrees(bundle)
    top_h = set(np.argsort(-herb_degree)[:18].tolist())
    top_s = set(np.argsort(-symptom_degree)[:14].tolist())
    edges = [(int(h), int(s)) for h, s in positives if int(h) in top_h and int(s) in top_s]
    g = nx.Graph()
    for h in sorted(top_h):
        g.add_node(f"h{h}", kind="herb", label=herb_map.get(h, f"Herb {h}"), degree=int(herb_degree[h]))
    for s in sorted(top_s):
        g.add_node(f"s{s}", kind="symptom", label=symptom_map.get(s, f"Symptom {s}"), degree=int(symptom_degree[s]))
    g.add_edges_from((f"h{h}", f"s{s}") for h, s in edges)
    fig, ax = plt.subplots(figsize=(11, 8))
    herbs = sorted([n for n, d in g.nodes(data=True) if d["kind"] == "herb"], key=lambda n: -g.nodes[n]["degree"])
    symptoms = sorted([n for n, d in g.nodes(data=True) if d["kind"] == "symptom"], key=lambda n: -g.nodes[n]["degree"])
    pos = {n: (0, 1 - 2 * i / max(len(herbs) - 1, 1)) for i, n in enumerate(herbs)}
    pos.update({n: (1, 1 - 2 * i / max(len(symptoms) - 1, 1)) for i, n in enumerate(symptoms)})
    nx.draw_networkx_edges(g, pos, ax=ax, width=0.55, alpha=0.28, edge_color="#777777")
    herb_sizes = [35 + 4 * g.nodes[n]["degree"] for n in herbs]
    symptom_sizes = [35 + 2 * g.nodes[n]["degree"] for n in symptoms]
    nx.draw_networkx_nodes(g, pos, nodelist=herbs, node_color="#4C78A8", node_size=herb_sizes, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=symptoms, node_color="#D97941", node_shape="s", node_size=symptom_sizes, ax=ax)
    for n in herbs:
        ax.text(-0.025, pos[n][1], g.nodes[n]["label"], ha="right", va="center", fontsize=7.5)
    for n in symptoms:
        ax.text(1.025, pos[n][1], g.nodes[n]["label"], ha="left", va="center", fontsize=7.5)
    ax.text(0, 1.09, "High-degree herbs", ha="center", fontsize=10, fontweight="bold")
    ax.text(1, 1.09, "High-degree symptoms (UMLS IDs)", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Illustrative HIT herb–symptom subnetwork", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(-1.12, 1.14)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "hit_herb_symptom_subnetwork")


def candidate_payload(bundle, split) -> tuple[np.ndarray, list[tuple[int, int, int]], set[tuple[int, int]]]:
    all_known = set(map(tuple, bundle.positives_unique.tolist()))
    valid_set = set(map(tuple, split["valid_pos"].tolist()))
    candidates, slices = [], []
    for herb in sorted({int(h) for h, _ in valid_set}):
        start = len(candidates)
        candidates.extend(
            (herb, symptom)
            for symptom in range(bundle.num_symptoms)
            if (herb, symptom) not in all_known or (herb, symptom) in valid_set
        )
        slices.append((herb, start, len(candidates)))
    return np.asarray(candidates, dtype=np.int64), slices, valid_set


def macro_ap(labels: np.ndarray, scores: np.ndarray, slices: list[tuple[int, int, int]]) -> float:
    return float(np.mean([average_precision_score(labels[a:b], scores[a:b]) for _, a, b in slices]))


def permutation_feature_importance(dataset: str, run: str, repeats: int = 5) -> pd.DataFrame:
    bundle = load_dataset(ROOT / "data" / "processed", dataset, torch.device("cpu"))
    sets = target_sets(bundle)
    first_checkpoint = torch.load(RESULTS / run / "checkpoints" / "fold_1.pt", map_location="cpu")
    rows = []
    fold = 0
    while list((RESULTS / run / "splits").glob(f"*_fold{fold}.npz")):
        split = load_split(RESULTS / run, fold)
        candidates, slices, valid_set = candidate_payload(bundle, split)
        labels = np.asarray([tuple(x) in valid_set for x in candidates], dtype=np.int64)
        train_edges = np.concatenate((split["train_pos"], split["train_neg"]))
        train_labels = np.concatenate((np.ones(len(split["train_pos"])), np.zeros(len(split["train_neg"]))))
        estimator = build_model("hist_gb", int(first_checkpoint["config"]["seed"]) + fold)
        estimator.fit(pair_features(bundle, sets, train_edges), train_labels)
        x = pair_features(bundle, sets, candidates)
        base = macro_ap(labels, estimator.predict_proba(x)[:, 1], slices)
        rng = np.random.default_rng(4200 + fold)
        for feature, name in enumerate(STRUCTURAL_FEATURE_NAMES):
            for repeat in range(repeats):
                permuted = x.copy()
                for _, start, stop in slices:
                    permuted[start:stop, feature] = rng.permutation(permuted[start:stop, feature])
                changed = macro_ap(labels, estimator.predict_proba(permuted)[:, 1], slices)
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold + 1,
                        "feature": name,
                        "repeat": repeat + 1,
                        "baseline_macro_ap": base,
                        "permuted_macro_ap": changed,
                        "macro_ap_decrease": base - changed,
                    }
                )
        fold += 1
    return pd.DataFrame(rows)


def plot_feature_importance() -> pd.DataFrame:
    frames = [
        permutation_feature_importance("HIT", "hit_ranking_bpr_broad_seed0"),
        permutation_feature_importance("TCMIO", "tcmio_ranking_bpr_broad_seed0"),
    ]
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUT / "query_preserving_permutation_importance.csv", index=False, encoding="utf-8-sig")
    grouped = result.groupby(["dataset", "feature"])["macro_ap_decrease"].agg(["mean", "std"]).reset_index()
    grouped.to_csv(OUT / "query_preserving_permutation_importance_summary.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
    for ax, dataset in zip(axes, ("HIT", "TCMIO")):
        frame = grouped[grouped.dataset == dataset].sort_values("mean")
        y = np.arange(len(frame))
        ax.barh(y, frame["mean"], xerr=frame["std"], color="#3A9D7D", capsize=3)
        ax.axvline(0, color="#555555", lw=1)
        ax.set_yticks(y, [FEATURE_LABELS[x] for x in frame.feature])
        ax.set_xlabel("Decrease in Macro-AP after permutation")
        ax.set_title(dataset)
        ax.grid(axis="x", ls="--", alpha=0.25)
    fig.suptitle("Query-preserving permutation importance of HistGB features", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "histgb_query_permutation_importance")
    return grouped


def cold_start_summary() -> pd.DataFrame:
    protocols = {
        "Cold herb": "hit_cold_herb_ranking_bpr_broad_seed0_full_ranking_{}",
        "Cold symptom": "hit_cold_symptom_ranking_bpr_broad_seed0_full_ranking_{}",
        "Cold both": "hit_cold_both_ranking_bpr_broad_seed0_full_ranking_{}",
    }
    methods = {"deep": "MHGNN-BPR", "histgb": "HistGB", "hybrid": "Fixed fusion"}
    rows = []
    for protocol, pattern in protocols.items():
        for mode, method in methods.items():
            payload = json.loads((RESULTS / pattern.format(mode) / "summary.json").read_text())
            rows.append(
                {
                    "protocol": protocol,
                    "method": method,
                    "Macro-AP": payload["metrics"]["ap"]["mean"],
                    "MRR": payload["metrics"]["mrr"]["mean"],
                    "Recall@10": payload["metrics"]["recall@10"]["mean"],
                    "NDCG@10": payload["metrics"]["ndcg@10"]["mean"],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "cold_start_full_ranking.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3), sharey=True)
    colors = {"MHGNN-BPR": "#4C78A8", "HistGB": "#F2A541", "Fixed fusion": "#3A9D7D"}
    for ax, metric in zip(axes.ravel(), ("Macro-AP", "MRR", "Recall@10", "NDCG@10")):
        x = np.arange(3)
        for index, method in enumerate(methods.values()):
            values = [float(frame[(frame.protocol == p) & (frame.method == method)][metric].iloc[0]) for p in protocols]
            ax.bar(x + (index - 1) * 0.23, values, 0.23, label=method, color=colors[method])
        ax.set_xticks(x, list(protocols))
        ax.set_ylim(0, 0.8)
        ax.set_title(metric)
        ax.grid(axis="y", ls="--", alpha=0.25)
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Full-candidate ranking under HIT cold-start protocols", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save(fig, "hit_cold_start_full_ranking")
    return frame


def case_analysis() -> dict:
    dataset = "HIT"
    run = "hit_ranking_bpr_broad_seed0"
    ranking_run = RESULTS / "hit_full_ranking_bpr_broad_hybrid_seed0"
    query = pd.read_csv(ranking_run / "query_metrics.csv")
    eligible = query[query.positives >= 3].copy()
    success = eligible.sort_values(["ap", "positives", "herb"], ascending=[False, False, True]).iloc[0]
    difficult = eligible.sort_values(["ap", "positives", "herb"], ascending=[True, False, True]).iloc[0]
    chosen = [("Successful case", success), ("Challenging case", difficult)]

    bundle = load_dataset(ROOT / "data" / "processed", dataset, torch.device("cpu"))
    adjacency = normalized_ppi_adjacency(bundle.ppi_edges, bundle.num_proteins, torch.device("cpu"))
    sets = target_sets(bundle)
    herb_map, symptom_map = load_hit_names()
    all_known = set(map(tuple, bundle.positives_unique.tolist()))
    case_rows, summaries = [], []
    for case_type, row in chosen:
        fold = int(row["fold"]) - 1
        herb = int(row["herb"])
        split = load_split(RESULTS / run, fold)
        valid_set = set(map(tuple, split["valid_pos"].tolist()))
        candidates = np.asarray(
            [
                (herb, symptom)
                for symptom in range(bundle.num_symptoms)
                if (herb, symptom) not in all_known or (herb, symptom) in valid_set
            ],
            dtype=np.int64,
        )
        labels = np.asarray([tuple(x) in valid_set for x in candidates], dtype=np.int64)
        checkpoint = torch.load(RESULTS / run / "checkpoints" / f"fold_{fold + 1}.pt", map_location="cpu")
        model = build_deep_model(bundle, checkpoint, torch.device("cpu"))
        deep = deep_scores(model, bundle, adjacency, candidates, checkpoint["config"], 8192)
        train_edges = np.concatenate((split["train_pos"], split["train_neg"]))
        train_labels = np.concatenate((np.ones(len(split["train_pos"])), np.zeros(len(split["train_neg"]))))
        estimator = build_model("hist_gb", int(checkpoint["config"]["seed"]) + fold)
        estimator.fit(pair_features(bundle, sets, train_edges), train_labels)
        hist = estimator.predict_proba(pair_features(bundle, sets, candidates))[:, 1]
        fusion = 0.5 * deep + 0.5 * hist
        metrics = ranking_metrics(labels, fusion, (10,))
        summaries.append(
            {
                "case_type": case_type,
                "herb_id": herb,
                "herb_name": herb_map.get(herb, str(herb)),
                "fold": fold + 1,
                "positives": int(labels.sum()),
                "candidates": len(labels),
                "Macro_AP": metrics["ap"],
                "MRR": metrics["mrr"],
                "Recall@10": metrics["recall@10"],
                "NDCG@10": metrics["ndcg@10"],
            }
        )
        order = np.argsort(-fusion, kind="stable")
        for rank, index in enumerate(order, start=1):
            symptom = int(candidates[index, 1])
            case_rows.append(
                {
                    "case_type": case_type,
                    "herb_id": herb,
                    "herb_name": herb_map.get(herb, str(herb)),
                    "fold": fold + 1,
                    "rank": rank,
                    "symptom_id": symptom,
                    "symptom_umls": symptom_map.get(symptom, str(symptom)),
                    "heldout_positive": int(labels[index]),
                    "deep_score": float(deep[index]),
                    "histgb_score": float(hist[index]),
                    "fusion_score": float(fusion[index]),
                }
            )
    details = pd.DataFrame(case_rows)
    summary = pd.DataFrame(summaries)
    details.to_csv(OUT / "case_ranking_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "case_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=False)
    for ax, item in zip(axes, summaries):
        subset = details[details.case_type == item["case_type"]].sort_values("rank").head(20)
        x = np.arange(len(subset))
        ax.bar(x, subset.fusion_score, color=np.where(subset.heldout_positive == 1, "#3A9D7D", "#B7B7B7"))
        for xi, is_positive in zip(x, subset.heldout_positive):
            if is_positive:
                ax.scatter(xi, 1.03, marker="*", s=65, color="#C44E52", clip_on=False)
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("Fusion score")
        ax.set_title(
            f"{item['case_type']}: {item['herb_name']} | AP={item['Macro_AP']:.3f}, Recall@10={item['Recall@10']:.3f}",
            fontsize=10,
        )
        ax.set_xticks(x, subset.symptom_umls, rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", ls="--", alpha=0.25)
    fig.suptitle("Top-20 candidate symptoms in representative HIT queries", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "hit_case_analysis_top20")
    return {"summary": summaries}


def main() -> None:
    set_style()
    report = {
        "roc_pr": plot_roc_pr(),
        "bioinformatics": plot_biological_landscape(),
    }
    plot_hit_network()
    importance = plot_feature_importance()
    report["feature_importance"] = importance.to_dict("records")
    report["cold_start"] = cold_start_summary().to_dict("records")
    report["cases"] = case_analysis()
    (OUT / "analysis_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "figures": str(FIG)}, indent=2))


if __name__ == "__main__":
    main()
