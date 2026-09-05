# Shared Windows onboarding helpers. No global installation, cleanup, or process killing.
function ConvertTo-RailwayNativeArgument {
    param([AllowEmptyString()][string]$Value)
    # Windows CommandLineToArgvW/CRT escaping, including empty arguments and trailing slashes.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Invoke-RailwayCommand {
    param([string]$Executable, [string[]]$Arguments, [string]$Stage)
    Write-Host ("[Railway] " + $Stage)
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Executable
    $startInfo.UseShellExecute = $false
    $startInfo.WorkingDirectory = (Get-Location).ProviderPath
    $startInfo.Arguments = (@($Arguments | ForEach-Object { ConvertTo-RailwayNativeArgument $_ }) -join ' ')
    $process = [System.Diagnostics.Process]::Start($startInfo)
    try {
        $process.WaitForExit()
        $global:LASTEXITCODE = $process.ExitCode
        if ($process.ExitCode -ne 0) { throw "$Stage failed (exit $($process.ExitCode))." }
    } finally { $process.Dispose() }
}

function Find-RailwayNode {
    param([string]$RepoRoot, [string]$NodeDirectory)
    if ($NodeDirectory) {
        $explicitNode = Join-Path $NodeDirectory 'node.exe'
        if (-not (Test-Path -LiteralPath $explicitNode -PathType Leaf)) { throw "Node not found: $explicitNode" }
        return (Resolve-Path -LiteralPath $explicitNode).Path
    }
    $localNode = Join-Path $RepoRoot '.runtime/node-v22.23.2-win-x64/node.exe'
    if (Test-Path -LiteralPath $localNode -PathType Leaf) { return $localNode }
    $receipt = Join-Path $RepoRoot '.runtime/environment.json'
    if (Test-Path -LiteralPath $receipt -PathType Leaf) {
        $settings = Get-Content -LiteralPath $receipt -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($settings.node_directory) {
            $savedNode = Join-Path $settings.node_directory 'node.exe'
            if (Test-Path -LiteralPath $savedNode -PathType Leaf) { return $savedNode }
        }
    }
    $pathNode = Get-Command node -CommandType Application -ErrorAction SilentlyContinue
    if ($pathNode) {
        $detectedNode = & $pathNode.Source --version
        if ($LASTEXITCODE -eq 0 -and ($detectedNode -join '') -match '^v(\d+)\.(\d+)\.(\d+)$') {
            if ([int]$Matches[1] -gt 22 -or ([int]$Matches[1] -eq 22 -and [int]$Matches[2] -ge 12)) {
                return $pathNode.Source
            }
        }
        # Preserve an old global installation; setup installs its own local version.
    }
    return $null
}

function Invoke-RailwayNpm {
    param([string]$RepoRoot, [string]$NodeDirectory, [string[]]$Arguments)
    $nodeExe = Find-RailwayNode -RepoRoot $RepoRoot -NodeDirectory $NodeDirectory
    if (-not $nodeExe) { throw 'Node is missing. Run Railway.ps1 setup.' }
    $npmCli = Join-Path (Split-Path -Parent $nodeExe) 'node_modules/npm/bin/npm-cli.js'
    if (-not (Test-Path -LiteralPath $npmCli -PathType Leaf)) { throw "npm is missing alongside Node: $npmCli" }
    $priorPath = $env:PATH
    Push-Location (Join-Path $RepoRoot 'web')
    try {
        $env:PATH = (Split-Path -Parent $nodeExe) + [IO.Path]::PathSeparator + $priorPath
        Invoke-RailwayCommand $nodeExe (@($npmCli) + $Arguments) ('npm ' + ($Arguments -join ' '))
    } finally { $env:PATH = $priorPath; Pop-Location }
}

function Install-RailwayPortableArchive {
    param(
        [string]$RuntimeRoot, [string]$Name, [string]$ArchiveName,
        [string]$Url, [string]$ChecksumUrl, [string]$ExecutableRelativePath,
        [string]$StripDirectory,
        [string]$ExtractionPython
    )
    foreach ($singleName in @($Name, $ArchiveName)) {
        if (-not $singleName -or $singleName -in @('.', '..') -or $singleName -match '[\\/:]') {
            throw 'Runtime and archive names must be single filenames.'
        }
    }
    foreach ($relativeName in @($ExecutableRelativePath, $StripDirectory)) {
        if ($relativeName -and ([IO.Path]::IsPathRooted($relativeName) -or $relativeName -match '(^|[\\/])\.\.([\\/]|$)' -or $relativeName.Contains(':'))) {
            throw 'Runtime archive subpaths must stay inside the extraction directory.'
        }
    }
    $resolvedRuntime = [IO.Path]::GetFullPath($RuntimeRoot)
    $finalDirectory = [IO.Path]::GetFullPath((Join-Path $resolvedRuntime $Name))
    if (-not $finalDirectory.StartsWith($resolvedRuntime.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Runtime target escapes the project runtime directory.'
    }
    if (Test-Path -LiteralPath $finalDirectory) { throw "Runtime destination already exists; refusing to overwrite: $finalDirectory" }
    # Retain failed attempts for diagnosis; never recursively delete a computed directory.
    $attempt = Join-Path $resolvedRuntime ('downloads/attempt-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $attempt -Force | Out-Null
    $archive = Join-Path $attempt $ArchiveName
    $checksumFile = Join-Path $attempt 'checksums.txt'
    $priorProtocol = [Net.ServicePointManager]::SecurityProtocol
    try {
        [Net.ServicePointManager]::SecurityProtocol = $priorProtocol -bor [Net.SecurityProtocolType]::Tls12
        Write-Host "[Railway] Download $Name (official HTTPS release + SHA-256 check)"
        Invoke-WebRequest -UseBasicParsing -Uri $ChecksumUrl -OutFile $checksumFile -TimeoutSec 60
        $checksumHashes = @(foreach ($line in (Get-Content -LiteralPath $checksumFile)) {
            if ($line.Trim() -match '^([a-fA-F0-9]{64})(?:\s+\*?(.+))?$') {
                if (-not $Matches[2] -or $Matches[2] -ceq $ArchiveName) { $Matches[1] }
            }
        })
        if ($checksumHashes.Count -ne 1) { throw 'Official checksum entry is missing or ambiguous.' }
        $expectedHash = $checksumHashes[0]
        Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $archive -TimeoutSec 180
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedHash) { throw 'Downloaded archive SHA-256 mismatch; archive will not be extracted.' }
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $extractRoot = Join-Path $attempt 'extracted'
        New-Item -ItemType Directory -Path $extractRoot | Out-Null
        $zip = [IO.Compression.ZipFile]::OpenRead($archive)
        try {
            foreach ($entry in $zip.Entries) {
                $entryTarget = [IO.Path]::GetFullPath((Join-Path $extractRoot $entry.FullName))
                if (-not $entryTarget.StartsWith($extractRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe ZIP entry path.' }
            }
        } finally { $zip.Dispose() }
        if ($ExtractionPython) {
            # Node's nested npm files exceed PS 5.1/.NET's legacy path limit.
            # Extended-length destination works without changing OS long-path policy.
            # All ZIP entries have already passed the boundary check above.
            $longExtractRoot = '\\?\' + $extractRoot
            if ($extractRoot.StartsWith('\\')) { $longExtractRoot = '\\?\UNC\' + $extractRoot.Substring(2) }
            Invoke-RailwayCommand $ExtractionPython @('-c', 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])', $archive, $longExtractRoot) 'Extract verified runtime archive'
        } else {
            [IO.Compression.ZipFile]::ExtractToDirectory($archive, $extractRoot)
        }
        if (-not (Test-Path -LiteralPath (Join-Path $extractRoot $ExecutableRelativePath) -PathType Leaf)) { throw 'Runtime executable is missing from downloaded archive.' }
        $sourceDirectory = $extractRoot
        if ($StripDirectory) { $sourceDirectory = Join-Path $extractRoot $StripDirectory }
        $resolvedSource = [IO.Path]::GetFullPath($sourceDirectory)
        if ($resolvedSource -ne $extractRoot -and -not $resolvedSource.StartsWith($extractRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Runtime staging directory escaped its boundary.' }
        Move-Item -LiteralPath $resolvedSource -Destination $finalDirectory
        if ($StripDirectory) { return (Join-Path $finalDirectory 'node.exe') }
        return (Join-Path $finalDirectory $ExecutableRelativePath)
    } catch {
        throw ("Runtime download/install failed: " + $_.Exception.Message + ". Check network access; retry, or supply -UvPath / -PythonPath / -NodeDirectory. No global settings changed.")
    } finally { [Net.ServicePointManager]::SecurityProtocol = $priorProtocol }
}
