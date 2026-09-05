param(
    [switch]$CoreOnly,
    [switch]$Offline,
    [string]$UvPath,
    [string]$PythonPath,
    [string]$NodeDirectory
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'onboarding_common.ps1')
$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $repoRoot '.runtime'
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
$savedEnvironment = @{}
try {
    if (-not [Environment]::Is64BitOperatingSystem -or $env:OS -ne 'Windows_NT') {
        throw 'This installer supports Windows x64. See docs/NEWCOMER_HANDOFF_CN.md for other systems.'
    }
    New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
    foreach ($entry in @{
        UV_PYTHON_INSTALL_DIR = (Join-Path $runtimeRoot 'python')
        UV_PYTHON_BIN_DIR = (Join-Path $runtimeRoot 'bin')
        UV_CACHE_DIR = (Join-Path $runtimeRoot 'cache')
        UV_PROJECT_ENVIRONMENT = (Join-Path $repoRoot '.venv')
        UV_NO_PROGRESS = '1'
    }.GetEnumerator()) {
        $savedEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, 'Process')
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
    if (-not $UvPath) {
        $localUv = Join-Path $runtimeRoot 'uv-0.12.1\uv.exe'
        $pathUv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $localUv -PathType Leaf) { $UvPath = $localUv }
        elseif ($pathUv) { $UvPath = $pathUv.Source }
        else {
            if ($Offline) { throw 'Offline setup needs -UvPath or an existing uv installation.' }
            $UvPath = Install-RailwayPortableArchive -RuntimeRoot $runtimeRoot -Name 'uv-0.12.1' `
                -ArchiveName 'uv-x86_64-pc-windows-msvc.zip' `
                -Url 'https://releases.astral.sh/github/uv/releases/download/0.12.1/uv-x86_64-pc-windows-msvc.zip' `
                -ChecksumUrl 'https://releases.astral.sh/github/uv/releases/download/0.12.1/uv-x86_64-pc-windows-msvc.zip.sha256' `
                -ExecutableRelativePath 'uv.exe'
        }
    }
    $UvPath = (Resolve-Path -LiteralPath $UvPath).Path
    Invoke-RailwayCommand $UvPath @('--version') 'Check uv'
    $offlineArgs = @()
    if ($Offline) { $offlineArgs = @('--offline') }
    if (-not $PythonPath) {
        if (Test-Path -LiteralPath $venvPython -PathType Leaf) { $PythonPath = $venvPython }
        else {
            Invoke-RailwayCommand $UvPath (@('python', 'install', '3.11.15', '--no-bin', '--no-registry') + $offlineArgs) 'Install project-local Python 3.11.15'
            $PythonPath = & $UvPath python find --managed-python 3.11.15 @offlineArgs
            if ($LASTEXITCODE -ne 0 -or -not $PythonPath) { throw 'Could not locate project-local Python.' }
            $PythonPath = ($PythonPath | Select-Object -Last 1).Trim()
        }
    }
    $PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
    $pythonVersion = & $PythonPath -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))'
    if ($LASTEXITCODE -ne 0 -or ($pythonVersion -join '').Trim() -ne '3.11') {
        throw 'Python 3.11 is required. Existing environments are not deleted automatically.'
    }
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        $existingVersion = & $venvPython -c 'import sys; print(sys.version_info.major, sys.version_info.minor, sep=chr(46))'
        if ($LASTEXITCODE -ne 0 -or ($existingVersion -join '').Trim() -ne '3.11') {
            throw 'Existing .venv is not usable Python 3.11. Use a new source directory; it will not be replaced automatically.'
        }
    }
    Push-Location $repoRoot
    try {
        Invoke-RailwayCommand $UvPath (@('sync', '--locked', '--inexact', '--extra', 'dev', '--python', $PythonPath) + $offlineArgs) 'Install locked Python dependencies'
    } finally { Pop-Location }
    if (-not $CoreOnly) {
        $nodeExe = Find-RailwayNode -RepoRoot $repoRoot -NodeDirectory $NodeDirectory
        if (-not $nodeExe) {
            if ($Offline) { throw 'Offline setup needs -NodeDirectory or a project-local Node installation.' }
            $nodeExe = Install-RailwayPortableArchive -RuntimeRoot $runtimeRoot -Name 'node-v22.23.2-win-x64' `
                -ArchiveName 'node-v22.23.2-win-x64.zip' `
                -Url 'https://nodejs.org/dist/v22.23.2/node-v22.23.2-win-x64.zip' `
                -ChecksumUrl 'https://nodejs.org/dist/v22.23.2/SHASUMS256.txt' `
                -ExecutableRelativePath 'node-v22.23.2-win-x64/node.exe' -StripDirectory 'node-v22.23.2-win-x64' -ExtractionPython $venvPython
        }
        $NodeDirectory = Split-Path -Parent $nodeExe
        $nodeVersion = & $nodeExe --version
        $nodeCompatible = $false
        if ($LASTEXITCODE -eq 0 -and ($nodeVersion -join '') -match '^v(\d+)\.(\d+)\.(\d+)$') {
            $nodeCompatible = [int]$Matches[1] -gt 22 -or ([int]$Matches[1] -eq 22 -and [int]$Matches[2] -ge 12)
        }
        if (-not $nodeCompatible) { throw 'Selected Node is too old or unusable. Use Node 22.12+ (tested: 22.23.2) or omit -NodeDirectory to install locally.' }
        $npmArgs = @('ci', '--no-audit', '--no-fund')
        if ($Offline) { $npmArgs += '--offline' }
        Invoke-RailwayNpm -RepoRoot $repoRoot -NodeDirectory $NodeDirectory -Arguments $npmArgs
    }
    $doctorArgs = @((Join-Path $repoRoot 'scripts/check_environment.py'), '--scope', $(if ($CoreOnly) { 'core' } else { 'full' }))
    if ($NodeDirectory -and -not $CoreOnly) { $doctorArgs += @('--node-directory', $NodeDirectory) }
    Invoke-RailwayCommand $venvPython $doctorArgs 'Verify installed environment'
    if (-not $CoreOnly) {
        # Persist only a verified tool location. This local receipt is not handed off.
        @{ schema_version = 1; node_directory = $NodeDirectory } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeRoot 'environment.json') -Encoding UTF8
    }
    Write-Host 'Setup completed. Next: powershell -ExecutionPolicy Bypass -File .\Railway.ps1 demo -SkipSetup'
} catch {
    Write-Host ("SETUP FAILED: " + $_.Exception.Message) -ForegroundColor Red
    Write-Host 'No ready status was issued. Existing models and environments were not deleted.'
    exit 1
} finally {
    foreach ($entry in $savedEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
}
