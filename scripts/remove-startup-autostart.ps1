$ErrorActionPreference = "Stop"

$startupDir = [Environment]::GetFolderPath("Startup")
$startupCmd = Join-Path $startupDir "MT5AI-Demo.cmd"

if (Test-Path $startupCmd) {
    Remove-Item -Force $startupCmd
    Write-Host "Removed $startupCmd"
}
else {
    Write-Host "No startup launcher found at $startupCmd"
}
