$ErrorActionPreference = "Stop"

$taskName = "MT5AI Demo Runner"
$launcherScript = Join-Path $PSScriptRoot "run-live-demo.ps1"

if (-not (Test-Path $launcherScript)) {
    throw "Launcher script not found: $launcherScript"
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcherScript`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Write-Host "Installed scheduled task: $taskName"
