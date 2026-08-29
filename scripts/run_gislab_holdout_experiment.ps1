param(
    [string]$BenchmarkRoot,
    [string]$ReferenceReportRoot,
    [string]$OutputName = "gislab_railtrack_v1_frozen",
    [UInt64]$MaximumPoints = [UInt64]::MaxValue
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BenchmarkRoot)) {
    $BenchmarkRoot = Join-Path $repositoryRoot "benchmarks\local\site-a"
}
if ([string]::IsNullOrWhiteSpace($ReferenceReportRoot)) {
    $ReferenceReportRoot = Join-Path $BenchmarkRoot "reference\rail_candidates"
}
$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$singleRun = Join-Path $PSScriptRoot "run_gislab_railtrack.ps1"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python environment not found: $python"
}

$holdoutRoot = Join-Path $BenchmarkRoot "holdouts\point_holdout_v1"
$manifest = Join-Path $holdoutRoot "manifest.json"
$baselineRoot = Join-Path $BenchmarkRoot "baselines\$OutputName"
$reportPath = Join-Path $BenchmarkRoot "reports\rail-holdout-gislab-v1.json"

# Start with the smallest spatial block so resource incompatibility is detected
# early without spending the full cost of the densest 0-50 m block.
$segments = @(
    "s0150_0200m",
    "s0100_0150m",
    "s0050_0100m",
    "s0000_0050m"
)
$splits = @("train", "holdout")

foreach ($split in $splits) {
    foreach ($segment in $segments) {
        $inputCloud = Join-Path $holdoutRoot "$segment.$split.laz"
        $reference = Join-Path $ReferenceReportRoot "${segment}_rail_candidates.json"
        $preparedInput = Join-Path $baselineRoot "preprocessed\$split\$segment.laz"
        $preparedManifest = Join-Path $baselineRoot "preprocessed\$split\$segment.manifest.json"
        $rawOutput = Join-Path $baselineRoot "raw\$split\$segment"
        $normalizedOutput = Join-Path $baselineRoot "$split\$segment.json"
        if (-not (Test-Path -LiteralPath $preparedInput)) {
            $prepareArguments = @(
                "-m", "railway_recon",
                "benchmark-prepare-gislab-input",
                "--source", $inputCloud,
                "--reference-report", $reference,
                "--segment", $segment,
                "--output", $preparedInput,
                "--manifest", $preparedManifest,
                "--coordinate-space", "corridor_local_xy"
            )
            & $python @prepareArguments
            if ($LASTEXITCODE -ne 0) {
                throw "GISLab shared preprocessing failed for $segment/$split"
            }
        }
        $railTrackOutput = Join-Path $rawOutput "RailTrack.laz"
        if (-not (Test-Path -LiteralPath $normalizedOutput)) {
            $runManifest = Join-Path $rawOutput "run.json"
            if (-not (Test-Path -LiteralPath $runManifest)) {
                $runParameters = @{
                    InputCloud = $preparedInput
                    OutputDirectory = $rawOutput
                    MaximumPoints = $MaximumPoints
                    LogLevel = "trace"
                }
                try {
                    & $singleRun @runParameters
                }
                catch {
                    Write-Warning $_
                }
            }
            if (Test-Path -LiteralPath $railTrackOutput) {
                $normalizeArguments = @(
                    "-m", "railway_recon",
                    "benchmark-normalize-gislab-rails",
                    "--source", $railTrackOutput,
                    "--reference-report", $reference,
                    "--segment", $segment,
                    "--output", $normalizedOutput,
                    "--source-coordinate-space", "corridor_local_xy"
                )
                & $python @normalizeArguments
            }
            else {
                $runRecord = Get-Content -LiteralPath $runManifest -Raw | ConvertFrom-Json
                if ([int]$runRecord.exit_code -in @(134, 139)) {
                    $failureCommand = "benchmark-record-gislab-no-detection"
                }
                elseif ([int]$runRecord.exit_code -in @(124, 137)) {
                    $failureCommand = "benchmark-record-gislab-timeout"
                }
                else {
                    throw "Unrecognized GISLab failure for $segment/$split"
                }
                $failureArguments = @(
                    "-m", "railway_recon",
                    $failureCommand,
                    "--run-manifest", $runManifest,
                    "--reference-report", $reference,
                    "--segment", $segment,
                    "--output", $normalizedOutput,
                    "--source-coordinate-space", "corridor_local_xy"
                )
                & $python @failureArguments
            }
            if ($LASTEXITCODE -ne 0) {
                throw "GISLab output normalization failed for $segment/$split"
            }
        }
    }
}

if (-not (Test-Path -LiteralPath $reportPath)) {
    $evaluationArguments = @(
        "-m", "railway_recon",
        "benchmark-evaluate-rail-holdout",
        "--holdout-manifest", $manifest,
        "--output", $reportPath
    )
    foreach ($segment in $segments) {
        $evaluationArguments += @(
            "--train-report", "$segment=$(Join-Path $baselineRoot "train\$segment.json")",
            "--holdout-report", "$segment=$(Join-Path $baselineRoot "holdout\$segment.json")",
            "--holdout-cloud", "$segment=$(Join-Path $holdoutRoot "$segment.holdout.laz")"
        )
    }
    & $python @evaluationArguments
    if ($LASTEXITCODE -ne 0) {
        throw "GISLab holdout evaluation failed"
    }
}

Write-Output $reportPath
