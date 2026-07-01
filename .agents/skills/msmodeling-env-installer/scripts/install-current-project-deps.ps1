param(
    [string]$EnvName = ".venv",
    [string]$PythonVersion = "",
    [switch]$UseExistingEnv,
    [switch]$SetProjectEnv,
    [switch]$UseHFMirror,
    [switch]$UseProjectUvCache = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PypiMirror = "https://mirrors.ustc.edu.cn/pypi/web/simple"

function Test-CommandExists {
    param([string]$Name)
    try {
        Get-Command $Name -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Resolve-PythonLauncher {
    if (Test-CommandExists "python") {
        return @("python")
    }
    if (Test-CommandExists "py") {
        return @("py", "-3")
    }
    throw "No Python launcher found. Install Python 3.10+ first."
}

function Invoke-Python {
    param(
        [string[]]$Launcher,
        [string[]]$PythonArgs
    )
    if ($Launcher.Count -eq 1) {
        & $Launcher[0] @PythonArgs
    } else {
        & $Launcher[0] $Launcher[1] @PythonArgs
    }
}

function Get-PythonScriptsPath {
    param([string[]]$Launcher)
    $scriptsPath = (Invoke-Python -Launcher $Launcher -PythonArgs @("-c", "import sysconfig; print(sysconfig.get_path('scripts'))")) | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($scriptsPath)) {
        return $null
    }
    return $scriptsPath.Trim()
}

function Get-PythonVersion {
    param([string[]]$Launcher)
    $versionText = (Invoke-Python -Launcher $Launcher -PythonArgs @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")) | Select-Object -First 1
    if (-not $versionText) {
        throw "Unable to detect Python version."
    }
    return [Version]($versionText.Trim())
}

function Resolve-UvCommand {
    param([string[]]$Launcher)

    $uvCommand = Get-Command "uv" -ErrorAction SilentlyContinue
    if ($uvCommand) {
        return $uvCommand.Source
    }

    $scriptsPath = Get-PythonScriptsPath -Launcher $Launcher
    if ($scriptsPath) {
        foreach ($fileName in @("uv.exe", "uv")) {
            $candidate = Join-Path $scriptsPath $fileName
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }

    throw "uv executable not found after installation. Ensure Python Scripts directory is on PATH or reinstall uv."
}

function Enable-ProjectUvCache {
    if (-not $UseProjectUvCache) {
        return
    }

    if (-not [string]::IsNullOrWhiteSpace($env:UV_CACHE_DIR)) {
        Write-Host "Using existing UV_CACHE_DIR: $env:UV_CACHE_DIR"
        return
    }

    $cachePath = Join-Path (Resolve-Path ".").Path ".uv-cache"
    New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
    $env:UV_CACHE_DIR = $cachePath
    Write-Host "UV_CACHE_DIR set for current session: $env:UV_CACHE_DIR"
}

function Test-PythonModuleAvailable {
    param(
        [string[]]$Launcher,
        [string]$ModuleName
    )
    Invoke-Python -Launcher $Launcher -PythonArgs @("-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$ModuleName') else 1)") | Out-Null
    return $LASTEXITCODE -eq 0
}

function Test-PythonPackageInstalled {
    param(
        [string[]]$Launcher,
        [string]$PackageName
    )
    Invoke-Python -Launcher $Launcher -PythonArgs @("-m", "pip", "show", $PackageName) | Out-Null
    return $LASTEXITCODE -eq 0
}

function Assert-ExistingEnvironmentClean {
    param([string[]]$Launcher)

    $blockedPackages = @()
    if (Test-PythonModuleAvailable -Launcher $Launcher -ModuleName "torch_npu") {
        $blockedPackages += "torch_npu"
    }

    foreach ($packageName in @("torch-npu", "torch_npu", "cudatoolkit")) {
        if (Test-PythonPackageInstalled -Launcher $Launcher -PackageName $packageName) {
            $blockedPackages += $packageName
        }
    }

    $blockedPackages = @($blockedPackages | Select-Object -Unique)
    if ($blockedPackages.Count -gt 0) {
        $packageList = $blockedPackages -join ", "
        throw "Existing environment contains $packageList. README fallback requires an environment without torch_npu or cudatoolkit. Create a fresh environment by rerunning without -UseExistingEnv."
    }

    Write-Host "Existing environment check passed: torch_npu and cudatoolkit are absent."
}

if ((-not (Test-Path "README.md")) -or (-not (Test-Path "pyproject.toml"))) {
    throw "README.md or pyproject.toml not found. Run this script from msmodeling repository root."
}

$launcher = @(Resolve-PythonLauncher)
$detectedPython = Get-PythonVersion -Launcher $launcher
if ($detectedPython -lt [Version]"3.10.0") {
    throw "Detected Python $detectedPython. Python 3.10+ is required."
}
Write-Host "Detected Python version: $detectedPython"

if (-not (Test-CommandExists "uv")) {
    Write-Host "uv not found. Installing uv with pip..."
    Invoke-Python -Launcher $launcher -PythonArgs @("-m", "pip", "install", "uv", "-i", $PypiMirror)
}

$uv = Resolve-UvCommand -Launcher $launcher
Write-Host "Using uv executable: $uv"
Enable-ProjectUvCache

if ([string]::IsNullOrWhiteSpace($PythonVersion)) {
    $PythonVersion = "$($detectedPython.Major).$($detectedPython.Minor)"
    Write-Host "PythonVersion not specified. Using detected Python version for venv: $PythonVersion"
}

$venvPython = Join-Path (Get-Location) "$EnvName\Scripts\python.exe"

if (-not $UseExistingEnv) {
    Write-Host "Installing msmodeling with uv sync (env: $EnvName, Python: $PythonVersion)..."
    $env:UV_PROJECT_ENVIRONMENT = $EnvName
    $env:UV_PYTHON = $PythonVersion
    & $uv sync

    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment python not found after uv sync: $venvPython"
    }

    Write-Host "Verifying msmodeling CLI..."
    & $uv run msmodeling --help | Out-Null
} else {
    Write-Host "Using legacy fallback: pip install -r requirements.txt (does not install msmodeling CLI; prefer uv sync)"
    if (-not (Test-Path "requirements.txt")) {
        throw "requirements.txt not found for legacy fallback."
    }
    if (Test-Path $venvPython) {
        Assert-ExistingEnvironmentClean -Launcher @($venvPython)
        & $venvPython -m pip install -r requirements.txt
    } else {
        Assert-ExistingEnvironmentClean -Launcher $launcher
        Invoke-Python -Launcher $launcher -PythonArgs @("-m", "pip", "install", "-r", "requirements.txt")
    }
}

if (Test-Path $venvPython) {
    & $uv pip check --python $venvPython
} else {
    Invoke-Python -Launcher $launcher -PythonArgs @("-m", "pip", "check")
}

if ($SetProjectEnv) {
    $repoRoot = (Resolve-Path ".").Path
    if ([string]::IsNullOrEmpty($env:PYTHONPATH)) {
        $env:PYTHONPATH = $repoRoot
    } else {
        $env:PYTHONPATH = "$repoRoot;$env:PYTHONPATH"
    }
    Write-Host "PYTHONPATH set for current session: $env:PYTHONPATH"
}

if ($UseHFMirror) {
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    Write-Host "HF_ENDPOINT set for current session: $env:HF_ENDPOINT"
}

Write-Host "Done. Activation: $EnvName\Scripts\activate  |  Or: uv run <command>"
