<#
.SYNOPSIS
    Sets up a development environment for voxlab on Windows.

.DESCRIPTION
    Creates a virtual environment, installs the package with the extras you
    ask for, and finishes by running the environment check.

.PARAMETER Extras
    Which optional dependency groups to install. Defaults to dev only, which
    needs no GPU and no model weights.

.EXAMPLE
    .\scripts\setup-dev.ps1
    .\scripts\setup-dev.ps1 -Extras dev,stt,audio
#>
[CmdletBinding()]
param(
    [string[]]$Extras = @('dev'),
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host '==> Checking the Python version' -ForegroundColor Cyan
# Deliberately free of double quotes: PowerShell strips embedded quotes when
# passing arguments to a native executable, which silently corrupts any -c
# snippet that contains them.
$versionCode = 'import sys; print(sys.version_info[0] * 100 + sys.version_info[1])'
$versionNumber = [int](& $Python -c $versionCode)
if ($versionNumber -lt 311) {
    $shown = '{0}.{1}' -f [math]::Floor($versionNumber / 100), ($versionNumber % 100)
    throw "Python 3.11 or newer is required, found $shown."
}
Write-Host "    Python $([math]::Floor($versionNumber / 100)).$($versionNumber % 100)" -ForegroundColor DarkGray

if (-not (Test-Path '.venv')) {
    Write-Host '==> Creating the virtual environment' -ForegroundColor Cyan
    & $Python -m venv .venv
}

$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment is broken: $venvPython not found."
}

Write-Host '==> Upgrading pip' -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip --quiet

$extrasSpec = ($Extras -join ',')
Write-Host "==> Installing voxlab with extras: $extrasSpec" -ForegroundColor Cyan
# The bracket expression has to be quoted; PowerShell would otherwise treat it
# as an array index.
& $venvPython -m pip install -e ".[$extrasSpec]"

if (-not (Test-Path '.env')) {
    Write-Host '==> Creating .env from .env.example' -ForegroundColor Cyan
    Copy-Item '.env.example' '.env'
}

Write-Host '==> Environment check' -ForegroundColor Cyan
& $venvPython -m voxlab.cli doctor

Write-Host ''
Write-Host 'Done. Activate the environment with:' -ForegroundColor Green
Write-Host '    .\.venv\Scripts\Activate.ps1'
