#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON=python; fi
cd "$PROJECT_ROOT"

"$PYTHON" -m unittest discover -s tests -v

for dataset in hit tcmio; do
  runs=()
  for seed in 0 17 42; do
    train_run="${dataset}_ranking_bpr_broad_seed${seed}"
    rank_run="${dataset}_full_ranking_bpr_broad_hybrid_seed${seed}"
    "$PYTHON" -m mhgnn_moe.run \
      --config "configs/${dataset}_ranking_bpr_broad.json" \
      --seed "$seed" --run-name "$train_run" --cpu-threads 4
    "$PYTHON" evaluate_full_ranking.py \
      --reference-run "$train_run" --mode hybrid --deep-weight 0.5 \
      --output "$rank_run" --cpu-threads 4
    runs+=("$rank_run")
  done
  "$PYTHON" summarize_ranking_multiseed.py \
    --runs "${runs[@]}" \
    --output "multiseed_${dataset}_full_ranking_bpr_broad_hybrid"
done
