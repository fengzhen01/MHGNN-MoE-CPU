$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
Set-Location -LiteralPath $ProjectRoot

& $Python -m unittest discover -s tests -v
& $Python audit_feature_leakage.py --dataset HIT --reference-run hit_baseline_strict_full --output audit_hit_feature_leakage
& $Python -m mhgnn_moe.classical --dataset HIT --reference-run hit_baseline_strict_full --model hist_gb --run-name hit_histgb_structural_strict
& $Python ensemble_runs.py --run-a hit_baseline_strict_full --run-b hit_histgb_structural_strict --weight-a 0.5 --output hit_baseline_histgb_ensemble
& $Python -m mhgnn_moe.run --config configs/hit_hard_baseline.json --run-name hit_hard_baseline_seed0 --cpu-threads 4
& $Python -m mhgnn_moe.classical --dataset HIT --reference-run hit_hard_baseline_seed0 --model hist_gb --run-name hit_hard_histgb_structural_seed0
& $Python ensemble_runs.py --run-a hit_hard_baseline_seed0 --run-b hit_hard_histgb_structural_seed0 --weight-a 0.5 --output hit_hard_baseline_histgb_ensemble_seed0
