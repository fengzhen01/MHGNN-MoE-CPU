param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

& $Python -m venv .venv
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
& $VenvPython -m pip install -r requirements.txt
& $VenvPython -m mhgnn_moe.audit --data-dir data/processed

Write-Host "CPU environment ready: $VenvPython"
