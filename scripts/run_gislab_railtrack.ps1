param(
    [Parameter(Mandatory = $true)]
    [string]$InputCloud,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [string]$Image = "railway/gislab-railtrack:3557ecf",

    [UInt64]$MaximumPoints = [UInt64]::MaxValue,

    [ValidateSet("trace", "debug", "info", "warning", "error", "fatal")]
    [string]$LogLevel = "info",

    [UInt32]$TimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"

$inputPath = (Resolve-Path -LiteralPath $InputCloud).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputDirectory)
if (-not (Test-Path -LiteralPath $outputPath)) {
    New-Item -ItemType Directory -Path $outputPath | Out-Null
}
$outputPath = (Resolve-Path -LiteralPath $outputPath).Path

$inputDirectory = Split-Path -Parent $inputPath
$inputName = Split-Path -Leaf $inputPath
$benchmarkArguments = @(
    "--input", "/input/$inputName",
    "--output", "/output",
    "--algorithm", "RailTrack",
    "--shift",
    "--size", $MaximumPoints.ToString(),
    "--loglevel", $LogLevel
)
$arguments = @(
    "run", "--rm",
    "--mount", "type=bind,source=$inputDirectory,target=/input,readonly",
    "--mount", "type=bind,source=$outputPath,target=/output"
)
if ($TimeoutSeconds -gt 0) {
    $arguments += @(
        "--entrypoint", "/usr/bin/timeout",
        $Image,
        "${TimeoutSeconds}s",
        "/opt/railroad/railroad_benchmark"
    )
}
else {
    $arguments += $Image
}
$arguments += $benchmarkArguments

$startedAt = [DateTimeOffset]::UtcNow
$timer = [System.Diagnostics.Stopwatch]::StartNew()
$combinedLog = Join-Path $outputPath "combined.log"
$previousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& docker @arguments 2>&1 | Tee-Object -FilePath $combinedLog
$exitCode = $LASTEXITCODE
$ErrorActionPreference = $previousErrorActionPreference
$timer.Stop()
$finishedAt = [DateTimeOffset]::UtcNow

$run = [ordered]@{
    schema_version = "railway.external-baseline-run.v1"
    method = "GISLab-ELTE/railroad:RailTrack"
    upstream_commit = "3557ecff1bd284108c7a833cce2b50ec9fcd5189"
    lastools_commit = "9bdc92c73047b46be25e5c2ed4abda2521e30fba"
    image = $Image
    input = $inputPath
    input_bytes = (Get-Item -LiteralPath $inputPath).Length
    input_sha256 = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    output_directory = $outputPath
    maximum_points = $MaximumPoints
    log_level = $LogLevel
    timeout_seconds = $TimeoutSeconds
    started_at = $startedAt.ToString("o")
    finished_at = $finishedAt.ToString("o")
    elapsed_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 6)
    exit_code = $exitCode
    command = @("railroad_benchmark") + $benchmarkArguments
}

$run | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $outputPath "run.json") -Encoding utf8
if ($exitCode -ne 0) {
    throw "GISLab RailTrack exited with code $exitCode. See $outputPath"
}

Write-Output (Join-Path $outputPath "RailTrack.laz")
