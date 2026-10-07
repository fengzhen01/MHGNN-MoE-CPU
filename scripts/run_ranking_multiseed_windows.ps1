$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
Set-Location -LiteralPath $ProjectRoot

& $Python -m unittest discover -s tests -v

$Datasets = @("hit", "tcmio")
$Seeds = @(0, 17, 42)
foreach ($Dataset in $Datasets) {
    $Config = "configs/${Dataset}_ranking_bpr_broad.json"
    foreach ($Seed in $Seeds) {
        $TrainRun = "${Dataset}_ranking_bpr_broad_seed${Seed}"
        $RankRun = "${Dataset}_full_ranking_bpr_broad_hybrid_seed${Seed}"
        & $Python -m mhgnn_moe.run --config $Config --seed $Seed --run-name $TrainRun --cpu-threads 4
        & $Python evaluate_full_ranking.py --reference-run $TrainRun --mode hybrid --deep-weight 0.5 --output $RankRun --cpu-threads 4
    }
    $Runs = $Seeds | ForEach-Object { "${Dataset}_full_ranking_bpr_broad_hybrid_seed$_" }
    & $Python summarize_ranking_multiseed.py --runs $Runs --output "multiseed_${Dataset}_full_ranking_bpr_broad_hybrid"
}
