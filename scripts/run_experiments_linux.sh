#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
"$PYTHON" -m mhgnn_moe.run --config configs/hit_baseline.json --run-name hit_baseline_strict_full --cpu-threads 4
"$PYTHON" -m mhgnn_moe.run --config configs/tcmio_baseline.json --run-name tcmio_baseline_strict_full --cpu-threads 4
"$PYTHON" -m mhgnn_moe.run --config configs/hit_moe.json --run-name hit_moe_strict_full_cached --cpu-threads 4
"$PYTHON" -m mhgnn_moe.run --config configs/tcmio_moe.json --run-name tcmio_moe_strict_full_final --cpu-threads 4
"$PYTHON" analyze_results.py
