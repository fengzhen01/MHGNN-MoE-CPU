$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}
Set-Location -LiteralPath $ProjectRoot

& $Python -m unittest discover -s tests -v
& $Python -m mhgnn_moe.run --config configs/hit_confidence.json --run-name hit_confidence_strict_seed0 --cpu-threads 4
& $Python compare_paired_runs.py --baseline hit_baseline_strict_full --candidate hit_confidence_strict_seed0 --output compare_hit_bce_vs_confidence_seed0
& $Python -m mhgnn_moe.run --config configs/hit_cold_herb_confidence.json --run-name hit_cold_herb_confidence_seed0 --cpu-threads 4
& $Python -m mhgnn_moe.run --config configs/hit_cold_symptom_confidence.json --run-name hit_cold_symptom_confidence_seed0 --cpu-threads 4
& $Python -m mhgnn_moe.run --config configs/hit_cold_both_confidence.json --run-name hit_cold_both_confidence_seed0 --cpu-threads 4
