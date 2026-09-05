param(
    [Parameter(Mandatory = $true)][string]$ProjectId,
    [Parameter(Mandatory = $true)][string]$Name
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'onboarding_common.ps1')
$repoRoot = Split-Path -Parent $PSScriptRoot
if ($ProjectId -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$') {
    throw 'ProjectId must use 1-80 letters, digits, underscores or hyphens; paths are not accepted.'
}
$pythonPath = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Run scripts/bootstrap.ps1 first.'
}

$target = Join-Path (Join-Path $repoRoot 'projects') $ProjectId
if (Test-Path -LiteralPath $target) { throw "Project already exists; refusing to overwrite: $target" }
Invoke-RailwayCommand $pythonPath @('-m', 'railway_recon', 'init', $target, '--project-id', $ProjectId, '--name', $Name) 'Initialize empty project'
