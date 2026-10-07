$ErrorActionPreference = "Stop"
$python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

$experiments = @(
    @{ Config = "configs/hit_cold_herb_ranking_bpr_broad.json"; Run = "hit_cold_herb_ranking_bpr_broad_seed0" },
    @{ Config = "configs/hit_cold_symptom_ranking_bpr_broad.json"; Run = "hit_cold_symptom_ranking_bpr_broad_seed0" },
    @{ Config = "configs/hit_cold_both_ranking_bpr_broad.json"; Run = "hit_cold_both_ranking_bpr_broad_seed0" }
)

foreach ($experiment in $experiments) {
    & $python -m mhgnn_moe.run --config $experiment.Config --run-name $experiment.Run
    foreach ($mode in @("deep", "histgb", "hybrid")) {
        $output = "$($experiment.Run)_full_ranking_$mode"
        & $python evaluate_full_ranking.py --reference-run $experiment.Run --output $output --mode $mode --deep-weight 0.5 --cpu-threads 4
    }
}
