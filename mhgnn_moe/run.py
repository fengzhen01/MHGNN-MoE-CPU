"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU-portable MHGNN/MoE cross-validation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--fold-indices", nargs="*", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--moe-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--baseline-run", default=None)
    parser.add_argument("--protocol", choices=["strict_unique", "legacy_rows", "cold_herb", "cold_symptom", "cold_both"])
    parser.add_argument("--loss-mode", choices=["bce", "confidence_bce", "pairwise_bpr", "nnpu"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--negative-confidence-min", type=float)
    parser.add_argument("--negative-confidence-power", type=float)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for argument, key in (
        (args.folds, "folds"), (args.epochs, "epochs"),
        (args.pretrain_epochs, "pretrain_epochs"), (args.moe_epochs, "moe_epochs"),
        (args.finetune_epochs, "finetune_epochs"), (args.cpu_threads, "cpu_threads"),
        (args.protocol, "protocol"), (args.loss_mode, "loss_mode"), (args.seed, "seed"),
        (args.negative_confidence_min, "negative_confidence_min"),
        (args.negative_confidence_power, "negative_confidence_power"),
    ):
        if argument is not None:
            config[key] = argument
    if args.fold_indices is not None and len(args.fold_indices):
        config["fold_indices"] = args.fold_indices
    if args.baseline_run is not None:
        config["baseline_run"] = None if args.baseline_run.lower() == "none" else args.baseline_run
    run_name = args.run_name or (
        f"{config['dataset'].lower()}_{config['model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    summary = run_experiment(config, project_root, project_root / "results" / run_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
