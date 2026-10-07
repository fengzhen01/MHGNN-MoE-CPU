param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Wheelhouse = Join-Path $ProjectRoot "wheelhouse"
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null
Set-Location -LiteralPath $ProjectRoot

& $Python -m pip download pip setuptools wheel -d $Wheelhouse
& $Python -m pip download torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu -d $Wheelhouse
& $Python -m pip download -r requirements.txt -d $Wheelhouse

Write-Host "Offline packages saved to: $Wheelhouse"
