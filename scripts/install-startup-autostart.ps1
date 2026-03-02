$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$launcherScript = Join-Path $PSScriptRoot "run-live-demo.ps1"
$startupDir = [Environment]::GetFolderPath("Startup")
$startupCmd = Join-Path $startupDir "MT5AI-Demo.cmd"

if (-not (Test-Path $launcherScript)) {
    throw "Launcher script not found: $launcherScript"
}

$content = @"
@echo off
start "" /min powershell.exe -ExecutionPolicy Bypass -WindowStyle Hidden -File "$launcherScript"
"@

Set-Content -Path $startupCmd -Value $content -Encoding ASCII
Write-Host "Installed startup launcher at $startupCmd"
