#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
"$PYTHON" -m mhgnn_moe.audit --data-dir data/processed
"$PYTHON" -m mhgnn_moe.run --config configs/hit_baseline.json --run-name smoke_hit_baseline --folds 2 --fold-indices 0 --epochs 1 --cpu-threads 4
"$PYTHON" -m mhgnn_moe.run --config configs/hit_moe.json --run-name smoke_hit_moe --folds 2 --fold-indices 0 --pretrain-epochs 1 --moe-epochs 1 --finetune-epochs 1 --baseline-run none --cpu-threads 4
echo "Smoke tests passed."
