param(
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$mainPy = Join-Path $repoRoot "main.py"
$envFile = Join-Path $repoRoot ".env"
$logDir = Join-Path $repoRoot "logs"
$runnerLog = Join-Path $logDir "run-live-launcher.log"
$mutexName = "Global\MT5AI-DemoRunner"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

function Write-RunnerLog {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$timestamp | $Message"
    Add-Content -Path $runnerLog -Value $line -Encoding UTF8
    Write-Host $line
}

function Invoke-LoggedCommand {
    param(
        [string]$Executable,
        [string[]]$Arguments
    )

    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Executable @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Add-Content -Path $runnerLog -Value $line -Encoding UTF8
            Write-Host $line
        }
        return $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Get-TerminalPath {
    if (-not (Test-Path $envFile)) {
        return $null
    }
    $match = Get-Content $envFile | Select-String -Pattern '^MT5__TERMINAL_PATH=(.+)$' | Select-Object -First 1
    if ($null -eq $match) {
        return $null
    }
    return $match.Matches[0].Groups[1].Value.Trim().Trim('"')
}

function Ensure-MT5Terminal {
    $terminalPath = Get-TerminalPath
    if ([string]::IsNullOrWhiteSpace($terminalPath) -or -not (Test-Path $terminalPath)) {
        Write-RunnerLog "MT5 terminal path is missing or invalid; relying on existing terminal session"
        return
    }

    $running = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue
    if ($running) {
        return
    }

    Write-RunnerLog "Starting MT5 terminal: $terminalPath"
    Start-Process -FilePath $terminalPath | Out-Null
    Start-Sleep -Seconds 8
}

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe"
}

if (-not (Test-Path $mainPy)) {
    throw "main.py not found at $mainPy"
}

$mutex = New-Object System.Threading.Mutex($false, $mutexName)
if (-not $mutex.WaitOne(0, $false)) {
    Write-RunnerLog "Another MT5AI demo launcher instance is already running"
    exit 0
}

try {
    while ($true) {
        Ensure-MT5Terminal

        Write-RunnerLog "Refreshing models before run-live"
        Invoke-LoggedCommand -Executable $pythonExe -Arguments @($mainPy, "refresh-models") | Out-Null

        Write-RunnerLog "Starting run-live"
        $exitCode = Invoke-LoggedCommand -Executable $pythonExe -Arguments @($mainPy, "run-live")
        Write-RunnerLog "run-live exited with code $exitCode"

        if ($Once) {
            exit $exitCode
        }

        Write-RunnerLog "Restarting run-live in 15 seconds"
        Start-Sleep -Seconds 15
    }
}
finally {
    $mutex.ReleaseMutex() | Out-Null
    $mutex.Dispose()
}
