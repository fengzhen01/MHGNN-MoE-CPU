"""Paired fold comparison with split audit and exact sign-flip inference."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc")
SPLIT_KEYS = ("train_pos", "valid_pos", "train_neg", "valid_neg")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fold_split(directory: Path, fold: int) -> dict[str, np.ndarray]:
    matches = list((directory / "splits").glob(f"*_fold{fold}.npz"))
    if not matches:
        raise FileNotFoundError(f"No split file for fold {fold} in {directory}")
    payload = np.load(matches[-1])
    return {key: payload[key] for key in SPLIT_KEYS}


def split_equal(left: Path, right: Path, folds: int, keys: tuple[str, ...]) -> bool:
    for fold in range(folds):
        first, second = fold_split(left, fold), fold_split(right, fold)
        if not all(np.array_equal(first[key], second[key]) for key in keys):
            return False
    return True


def exact_sign_flip_pvalue(deltas: np.ndarray) -> float:
    observed = abs(float(deltas.mean()))
    if len(deltas) > 20:
        raise ValueError("Exact sign-flip test is limited to 20 paired observations")
    extreme = 0
    total = 2 ** len(deltas)
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        value = abs(float(np.mean(deltas * np.asarray(signs))))
        extreme += value >= observed - 1e-15
    return extreme / total


def bootstrap_ci(deltas: np.ndarray, seed: int = 0, samples: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comparison-scope", choices=["all", "evaluation"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    baseline_dir = root / "results" / args.baseline
    candidate_dir = root / "results" / args.candidate
    baseline = read_rows(baseline_dir / "fold_metrics.csv")
    candidate = read_rows(candidate_dir / "fold_metrics.csv")
    if len(baseline) != len(candidate):
        raise RuntimeError("Runs have different numbers of completed folds")
    keys = SPLIT_KEYS if args.comparison_scope == "all" else ("train_pos", "valid_pos", "valid_neg")
    if not split_equal(baseline_dir, candidate_dir, len(baseline), keys):
        raise RuntimeError(f"Runs do not use identical paired {args.comparison_scope} splits")

    report = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "folds": len(baseline),
        "paired_splits_identical": True,
        "comparison_scope": args.comparison_scope,
        "metrics": {},
    }
    for metric in METRICS:
        first = np.asarray([float(row[metric]) for row in baseline])
        second = np.asarray([float(row[metric]) for row in candidate])
        delta = second - first
        lower, upper = bootstrap_ci(delta)
        report["metrics"][metric] = {
            "baseline_mean": float(first.mean()),
            "candidate_mean": float(second.mean()),
            "mean_delta": float(delta.mean()),
            "delta_std": float(delta.std()),
            "bootstrap_95ci": [lower, upper],
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(delta == 0)),
            "losses": int(np.sum(delta < 0)),
            "exact_sign_flip_p_two_sided": exact_sign_flip_pvalue(delta),
            "paired_effect_dz": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else 0.0,
            "fold_deltas": delta.tolist(),
        }

    output = root / "results" / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 配对实验比较", "", f"基线：`{args.baseline}`  ", f"候选：`{args.candidate}`  ",
        f"配对划分一致：是（{len(baseline)}折）", "",
        "| 指标 | 基线均值 | 候选均值 | 平均变化 | 95% bootstrap CI | 胜/平/负 | 精确符号置换p值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        item = report["metrics"][metric]
        ci = item["bootstrap_95ci"]
        lines.append(
            f"| {metric} | {item['baseline_mean']:.4f} | {item['candidate_mean']:.4f} | "
            f"{item['mean_delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{item['wins']}/{item['ties']}/{item['losses']} | {item['exact_sign_flip_p_two_sided']:.4f} |"
        )
    lines += [
        "", "说明：折数较少且交叉验证折并非完全独立，p值与置信区间仅作为稳定性证据，不能替代多随机种子和外部验证。", "",
    ]
    (output / "PAIRED_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
