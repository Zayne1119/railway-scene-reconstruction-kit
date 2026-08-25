$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Run scripts/bootstrap.ps1 first.'
}

Push-Location $repoRoot
try {
    & $pythonPath scripts/smoke_test.py
    & $pythonPath scripts/run_tests.py
    & $pythonPath -m ruff check .
    & $pythonPath -m railway_recon safety-check --root .
}
finally {
    Pop-Location
}
