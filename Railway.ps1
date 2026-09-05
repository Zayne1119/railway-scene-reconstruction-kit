param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'doctor', 'demo', 'start', 'check', 'new')]
    [string]$Action = 'help',
    [switch]$CoreOnly,
    [switch]$Offline,
    [switch]$SkipSetup,
    [switch]$NoStart,
    [ValidateRange(1024, 65535)][int]$Port = 3010,
    [string]$UvPath,
    [string]$PythonPath,
    [string]$NodeDirectory,
    [string]$ProjectId,
    [string]$Name
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'scripts/onboarding_common.ps1')
$repoRoot = $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv/Scripts/python.exe'
try {
    if ($Action -eq 'help') {
        Write-Host @'
Railway handoff launcher (Windows x64)
  .\Railway.ps1 demo           Install, generate synthetic model, start localhost viewer
  .\Railway.ps1 setup          Install locked Python + web dependencies
  .\Railway.ps1 doctor         Diagnose environment without installing anything
  .\Railway.ps1 demo -SkipSetup Rebuild/verify demo without reinstalling
  .\Railway.ps1 demo -NoStart  Generate/check demo, do not start a server
  .\Railway.ps1 start          Start existing demo viewer (Ctrl+C to stop)
  .\Railway.ps1 check          Run Python tests, viewer tests, viewer build
  .\Railway.ps1 new -ProjectId my-site -Name "My Site"  Create an EMPTY project

Options: -CoreOnly (no viewer), -Port 3011, -Offline,
         -UvPath PATH -PythonPath PATH -NodeDirectory DIRECTORY
Start here: docs/NEWCOMER_HANDOFF_CN.md
The demo is synthetic, not a customer model or a field-accuracy result.
'@
        exit 0
    }
    if ($Action -eq 'setup' -or ($Action -eq 'demo' -and -not $SkipSetup)) {
        $setupArguments = @{ CoreOnly = $CoreOnly; Offline = $Offline }
        foreach ($option in @('UvPath', 'PythonPath', 'NodeDirectory')) {
            $value = Get-Variable -Name $option -ValueOnly
            if ($value) { $setupArguments[$option] = $value }
        }
        & (Join-Path $repoRoot 'scripts/bootstrap.ps1') @setupArguments
        if (-not $?) { throw 'Environment setup failed.' }
        if ($Action -eq 'setup') { exit 0 }
    }
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw 'Project Python is missing. Run: powershell -ExecutionPolicy Bypass -File .\Railway.ps1 setup'
    }
    $nodeExe = $null
    if (-not $CoreOnly) { $nodeExe = Find-RailwayNode -RepoRoot $repoRoot -NodeDirectory $NodeDirectory }
    if ($nodeExe) { $NodeDirectory = Split-Path -Parent $nodeExe }
    if ($Action -in @('doctor', 'demo', 'start')) {
        $doctorArgs = @((Join-Path $repoRoot 'scripts/check_environment.py'), '--scope', $(if ($CoreOnly) { 'core' } else { 'full' }))
        if ($NodeDirectory -and -not $CoreOnly) { $doctorArgs += @('--node-directory', $NodeDirectory) }
        Invoke-RailwayCommand $venvPython $doctorArgs 'Check environment'
        if ($Action -eq 'doctor') { exit 0 }
    }
    if ($Action -eq 'new') {
        if (-not $ProjectId -or -not $Name) { throw 'new requires -ProjectId and -Name.' }
        & (Join-Path $repoRoot 'scripts/new_project.ps1') -ProjectId $ProjectId -Name $Name
        if (-not $?) { throw 'New-project initialization failed.' }
        Write-Host 'Created a project scaffold, not a reconstructed model. Read docs/AI_HANDOFF_CN.md before importing real data.'
        exit 0
    }
    if ($Action -eq 'check') {
        Push-Location $repoRoot
        try { Invoke-RailwayCommand $venvPython @('-m', 'pytest', '-rs') 'Python regression tests' }
        finally { Pop-Location }
        if (-not $CoreOnly) {
            Invoke-RailwayNpm $repoRoot $NodeDirectory @('test')
            Invoke-RailwayNpm $repoRoot $NodeDirectory @('run', 'build')
        }
        Write-Host 'Requested regression checks passed.'
        exit 0
    }
    if ($Action -eq 'demo') {
        $demoArgs = @((Join-Path $repoRoot 'scripts/build_onboarding_demo.py'))
        if ($CoreOnly) { $demoArgs += @('--no-web') }
        Invoke-RailwayCommand $venvPython $demoArgs 'Build/verify independent synthetic demo'
        Write-Host ('Model directory: ' + (Join-Path $repoRoot 'projects/onboarding-demo'))
        if ($NoStart -or $CoreOnly) { exit 0 }
    }
    if ($Action -eq 'start' -or $Action -eq 'demo') {
        if ($CoreOnly) { throw 'start requires web dependencies; omit -CoreOnly.' }
        if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'web/public/demo/project.json'))) {
            throw 'Demo payload is missing. Run Railway.ps1 demo -SkipSetup -NoStart first.'
        }
        Write-Host "Open http://127.0.0.1:$Port/?demo=1 . Ctrl+C stops this server."
        Write-Host 'Localhost only. An occupied port fails explicitly; use -Port 3011 instead of stopping other applications.'
        Invoke-RailwayNpm $repoRoot $NodeDirectory @('run', 'dev', '--', '--host', '127.0.0.1', '--port', "$Port", '--strictPort')
    }
} catch {
    Write-Host ("RAILWAY FAILED: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'See docs/NEWCOMER_HANDOFF_CN.md. Existing production models are not replaced.'
    exit 1
}
