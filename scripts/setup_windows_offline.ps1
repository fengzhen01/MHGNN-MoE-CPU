param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Wheelhouse = Join-Path $ProjectRoot "wheelhouse"
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Wheelhouse)) {
    throw "wheelhouse not found. Generate it on a compatible online Windows machine first."
}

& $Python -m venv .venv
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $VenvPython -m pip install --no-index --find-links $Wheelhouse pip setuptools wheel
& $VenvPython -m pip install --no-index --find-links $Wheelhouse torch==2.3.1
& $VenvPython -m pip install --no-index --find-links $Wheelhouse -r requirements.txt
& $VenvPython -m mhgnn_moe.audit --data-dir data/processed

Write-Host "Offline CPU environment ready: $VenvPython"
