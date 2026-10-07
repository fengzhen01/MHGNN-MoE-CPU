"""Create a reproducible baseline-versus-MoE report from completed folds."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RUNS = {
    "HIT": ("hit_baseline_strict_full", "hit_moe_strict_full_cached"),
    "TCMIO": ("tcmio_baseline_strict_full", "tcmio_moe_strict_full_final"),
}
METRICS = ("accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    comparison: list[dict] = []
    fold_deltas: list[dict] = []
    split_audit: list[dict] = []
    for dataset, (baseline_name, moe_name) in RUNS.items():
        baseline_dir, moe_dir = RESULTS / baseline_name, RESULTS / moe_name
        baseline = read_csv(baseline_dir / "fold_metrics.csv")
        moe = read_csv(moe_dir / "fold_metrics.csv")
        if len(baseline) != 10 or len(moe) != 10:
            raise RuntimeError(f"{dataset}: expected 10 completed folds")
        for index, (base_row, moe_row) in enumerate(zip(baseline, moe), start=1):
            base_split = next((baseline_dir / "splits").glob(f"*_fold{index - 1}.npz"))
            moe_split = next((moe_dir / "splits").glob(f"*_fold{index - 1}.npz"))
            equal = digest(base_split) == digest(moe_split)
            split_audit.append({"dataset": dataset, "fold": index, "identical": equal})
            if not equal:
                raise RuntimeError(f"{dataset} fold {index}: baseline/MoE splits differ")
            delta_row = {"dataset": dataset, "fold": index}
            for metric in METRICS:
                delta_row[f"{metric}_delta"] = float(moe_row[metric]) - float(base_row[metric])
            fold_deltas.append(delta_row)

        for model, rows in (("baseline", baseline), ("moe", moe)):
            row = {"dataset": dataset, "model": model}
            for metric in METRICS:
                values = np.asarray([float(item[metric]) for item in rows])
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std())
            comparison.append(row)

        base_summary = comparison[-2]
        moe_summary = comparison[-1]
        delta_summary = {"dataset": dataset, "model": "moe_minus_baseline"}
        for metric in METRICS:
            delta_summary[f"{metric}_mean"] = moe_summary[f"{metric}_mean"] - base_summary[f"{metric}_mean"]
            values = [row[f"{metric}_delta"] for row in fold_deltas if row["dataset"] == dataset]
            delta_summary[f"{metric}_std"] = float(np.std(values))
            delta_summary[f"{metric}_wins"] = int(np.sum(np.asarray(values) > 0))
        comparison.append(delta_summary)

    comparison_path = RESULTS / "experiment_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = sorted({key for row in comparison for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)
    with (RESULTS / "paired_fold_deltas.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_deltas[0]))
        writer.writeheader()
        writer.writerows(fold_deltas)
    (RESULTS / "split_audit.json").write_text(
        json.dumps(split_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 正式实验结果", "", "协议：strict_unique，固定种子0，10折交叉验证；数值为均值±总体标准差。", "",
        "| 数据集 | 模型 | F1 | MCC | AUROC | AUPRC |", "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in RUNS:
        rows = [row for row in comparison if row["dataset"] == dataset and row["model"] != "moe_minus_baseline"]
        for row in rows:
            values = [f"{row[f'{m}_mean']:.4f}±{row[f'{m}_std']:.4f}" for m in ("f1", "mcc", "auroc", "auprc")]
            lines.append(f"| {dataset} | {row['model']} | " + " | ".join(values) + " |")
    lines += ["", "## MoE相对基线的均值变化与胜出折数", "",
              "| 数据集 | ΔF1 | ΔMCC | ΔAUROC | ΔAUPRC | AUPRC胜出折 |",
              "|---|---:|---:|---:|---:|---:|"]
    for dataset in RUNS:
        row = next(item for item in comparison if item["dataset"] == dataset and item["model"] == "moe_minus_baseline")
        lines.append(
            f"| {dataset} | {row['f1_mean']:+.4f} | {row['mcc_mean']:+.4f} | "
            f"{row['auroc_mean']:+.4f} | {row['auprc_mean']:+.4f} | {row['auprc_wins']}/10 |"
        )
    lines += [
        "", "所有20组baseline/MoE逐折划分文件的SHA-256均一致。",
        "MoE改进幅度较小，应表述为在保持基线性能的基础上取得数值改善；未进行重复随机种子实验前，不宣称统计显著提升。", "",
    ]
    (RESULTS / "FINAL_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Result audit and comparison files written to", RESULTS)


if __name__ == "__main__":
    main()
