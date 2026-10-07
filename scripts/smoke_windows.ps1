$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location -LiteralPath $ProjectRoot

& $Python -m mhgnn_moe.audit --data-dir data/processed
& $Python -m mhgnn_moe.run --config configs/hit_baseline.json --run-name smoke_hit_baseline --folds 2 --fold-indices 0 --epochs 1 --cpu-threads 4
& $Python -m mhgnn_moe.run --config configs/hit_moe.json --run-name smoke_hit_moe --folds 2 --fold-indices 0 --pretrain-epochs 1 --moe-epochs 1 --finetune-epochs 1 --baseline-run none --cpu-threads 4

Write-Host "Smoke tests passed. See results\smoke_hit_baseline and results\smoke_hit_moe."
