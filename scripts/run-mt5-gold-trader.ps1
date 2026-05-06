param(
    [switch]$Once,
    [switch]$Preflight
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "mt5_gold_trader.log"

Set-Location $Root

if ($Preflight) {
    python -m invest_advisor_bot.mt5_gold_trader preflight 2>&1 | Tee-Object -FilePath $LogPath -Append
} elseif ($Once) {
    python -m invest_advisor_bot.mt5_gold_trader cycle 2>&1 | Tee-Object -FilePath $LogPath -Append
} else {
    python -m invest_advisor_bot.mt5_gold_trader run 2>&1 | Tee-Object -FilePath $LogPath -Append
}
