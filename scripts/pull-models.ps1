<#
.SYNOPSIS
    Downloads the local models voxlab needs.

.DESCRIPTION
    Pulls the Qwen3 model into Ollama. The Whisper weights are not fetched here
    on purpose: faster-whisper downloads them on first use, into its own cache.
#>
[CmdletBinding()]
param(
    [string]$LlmModel = 'qwen3:8b'
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command 'ollama' -ErrorAction SilentlyContinue)) {
    throw 'ollama was not found on PATH. Install it from https://ollama.com and reopen the shell.'
}

Write-Host "==> Pulling $LlmModel" -ForegroundColor Cyan
& ollama pull $LlmModel

Write-Host ''
Write-Host 'Installed models:' -ForegroundColor Green
& ollama list
