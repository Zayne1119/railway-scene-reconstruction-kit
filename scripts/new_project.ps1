param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Name
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Run scripts/bootstrap.ps1 first.'
}

$target = Join-Path (Join-Path $repoRoot 'projects') $ProjectId
& $pythonPath -m railway_recon init $target --project-id $ProjectId --name $Name

