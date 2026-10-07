$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Set-Location -LiteralPath $ProjectRoot

& $Python -m mhgnn_moe.run --config configs/hit_baseline.json --run-name hit_baseline_strict_full --cpu-threads 4
& $Python -m mhgnn_moe.run --config configs/tcmio_baseline.json --run-name tcmio_baseline_strict_full --cpu-threads 4
& $Python -m mhgnn_moe.run --config configs/hit_moe.json --run-name hit_moe_strict_full_cached --cpu-threads 4
& $Python -m mhgnn_moe.run --config configs/tcmio_moe.json --run-name tcmio_moe_strict_full_final --cpu-threads 4
& $Python analyze_results.py
