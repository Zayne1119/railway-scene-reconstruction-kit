$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot '.venv'

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
$pythonLauncher = Get-Command python -ErrorAction SilentlyContinue

if ($null -ne $pyLauncher) {
    $basePython = $pyLauncher.Source
    $basePythonArgs = @('-3.11')
}
elseif ($null -ne $pythonLauncher) {
    $basePython = $pythonLauncher.Source
    $basePythonArgs = @()
}
else {
    throw 'Python 3.11 was not found. Install it, reopen PowerShell, and rerun this script.'
}

$detectedVersion = & $basePython @basePythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $detectedVersion.Trim() -ne '3.11') {
    throw "Python 3.11 is required; detected $detectedVersion."
}

if (-not (Test-Path -LiteralPath $venvPath)) {
    & $basePython @basePythonArgs -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create .venv.'
    }
}

$pythonPath = Join-Path $venvPath 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Virtual-environment Python is missing: $pythonPath"
}

& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -e "$repoRoot[dev]"
& $pythonPath -m railway_recon --help

Write-Host "Ready. Activate with: $venvPath\Scripts\Activate.ps1"
