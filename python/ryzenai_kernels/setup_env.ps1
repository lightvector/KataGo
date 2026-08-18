<#
.SYNOPSIS
Set up the mlir-aie ("iron") Python environment that make_artifacts.py needs.

.DESCRIPTION
Only kernel authors need this. Building katago itself needs nothing from here,
and running it needs nothing at all beyond the NPU driver - the .xclbin files
are already in the repository. This is for regenerating them.

What it does, in order:

  1. Checks the prerequisites (Python, the MSVC C++ toolchain, the XRT SDK) and
     stops with an explanation if one is missing. It never installs any of them.
  2. Creates a virtual environment at -Prefix.
  3. Installs the mlir-aie wheels plus numpy and ml_dtypes into it.
  4. Writes activate_iron.bat next to this script, which enters the environment
     with mlir_aie/bin and the XRT tools on PATH.
  5. Verifies by importing aie.iron and running make_artifacts.py --list.

Nothing is downloaded or installed unless -Execute is passed. Without it the
script prints the exact commands it would run, so the several hundred megabytes
of wheels are never fetched by surprise.

.PARAMETER Prefix
Where to create the environment. Required, deliberately: this puts a large
tree somewhere permanent, and guessing a location for that is not the script's
call to make.

.PARAMETER Version
mlir-aie release tag to install. Defaults to the newest release.

.PARAMETER Execute
Actually create the environment. Without this the script only reports.

.EXAMPLE
.\setup_env.ps1 -Prefix C:\Envs\mlir-aie
Report what would be done, changing nothing.

.EXAMPLE
.\setup_env.ps1 -Prefix C:\Envs\mlir-aie -Execute
Create the environment for real.
#>
[CmdletBinding()]
param(
  [string]$Prefix,
  [string]$Version = "latest",
  [switch]$Execute
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Note($msg) { Write-Host "   $msg" }

if (-not $Prefix) {
  Write-Host @"
ERROR: -Prefix is required.

Give the directory to create the mlir-aie environment in. It is not defaulted
because it holds a multi-gigabyte toolchain and where that belongs is your call,
not this script's.

  .\setup_env.ps1 -Prefix C:\Envs\mlir-aie              # report only
  .\setup_env.ps1 -Prefix C:\Envs\mlir-aie -Execute     # actually install
  .\setup_env.ps1 -Prefix D:\tools\mlir-aie -Execute    # anywhere else

See Compiling.md, "Regenerating the RyzenAI NPU kernels", for the full flow.
"@ -ForegroundColor Red
  exit 1
}

# ---- 1. prerequisites, checked but never installed --------------------------

Step "Checking prerequisites"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Fail "python is not on PATH. Install Python 3.10 or newer from python.org or via conda." }
$pyver = (& python -c "import sys;print('%d.%d'%sys.version_info[:2])").Trim()
if ([version]$pyver -lt [version]"3.10") {
  Fail "python $pyver is too old; mlir-aie needs 3.10 or newer. The one on PATH is $($py.Source)."
}
Note "python $pyver at $($py.Source)"

# vswhere sits at a documented fixed location under Program Files, so this
# neither hardcodes an edition nor a version.
$vswhere = $null
foreach ($pf in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
  if ($pf) {
    $cand = Join-Path $pf "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $cand) { $vswhere = $cand; break }
  }
}
if (-not $vswhere) {
  Fail "vswhere.exe not found. Install Visual Studio Build Tools with the 'Desktop development with C++' workload - the kernel compiler shells out to cl.exe."
}
$vsPath = & $vswhere -all -sort -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vsPath) {
  Fail "No Visual Studio C++ toolchain found. Install the 'Desktop development with C++' workload."
}
Note "MSVC toolchain at $vsPath"

# XRT: the env var if it points somewhere real, else the conventional install
# root. Probe for xclbinutil.exe itself rather than for a bin\ directory - the
# Windows SDK puts its tools straight in the install root, and xclbinutil is the
# one thing aiecc actually shells out to. Both layouts are accepted.
$xrt = $null
foreach ($cand in @($env:XILINX_XRT, "C:\Xilinx\XRT")) {
  if (-not $cand) { continue }
  foreach ($sub in @("", "bin")) {
    $dir = if ($sub) { Join-Path $cand $sub } else { $cand }
    if (Test-Path (Join-Path $dir "xclbinutil.exe")) { $xrt = $dir; break }
  }
  if ($xrt) { break }
}
if (-not $xrt) {
  Fail "xclbinutil.exe not found. Install AMD's XRT for Windows and either set XILINX_XRT to it or accept the default C:\Xilinx\XRT - aiecc shells out to xclbinutil to package each .xclbin."
}
Note "XRT at $xrt"

# ---- 2..5. the plan ---------------------------------------------------------

$venv = Join-Path $Prefix "ironenv"
$venvPy = Join-Path $venv "Scripts\python.exe"
$wheelIndex = "https://github.com/Xilinx/mlir-aie/releases"
$activateBat = Join-Path $Here "activate_iron.bat"

$plan = @(
  @{ What = "create venv";      Cmd = "python -m venv `"$venv`"" },
  @{ What = "upgrade pip";      Cmd = "`"$venvPy`" -m pip install --upgrade pip wheel" },
  @{ What = "install mlir-aie"; Cmd = "`"$venvPy`" -m pip install --find-links $wheelIndex/$Version mlir_aie mlir_aie_tools" },
  @{ What = "install deps";     Cmd = "`"$venvPy`" -m pip install numpy ml_dtypes" }
)

Step "Plan"
Note "environment  : $venv"
Note "mlir-aie     : $Version"
Note "activator    : $activateBat"
foreach ($p in $plan) { Note ("{0,-16} {1}" -f $p.What, $p.Cmd) }

if (-not $Execute) {
  Write-Host "`nDry run - nothing was downloaded or created. Re-run with -Execute to proceed." -ForegroundColor Yellow
  Write-Host "The mlir-aie wheels are several hundred megabytes." -ForegroundColor Yellow
  exit 0
}

Step "Creating the environment"
if (Test-Path $venvPy) {
  Note "already exists, reusing: $venv"
} else {
  New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
  & python -m venv $venv
  if ($LASTEXITCODE -ne 0) { Fail "venv creation failed." }
}

foreach ($p in $plan | Select-Object -Skip 1) {
  Step $p.What
  cmd /c $p.Cmd
  if ($LASTEXITCODE -ne 0) { Fail "$($p.What) failed. See the pip output above." }
}

# ---- the activator ----------------------------------------------------------

Step "Writing $activateBat"
$mlirBin = Join-Path $venv "Lib\site-packages\mlir_aie\bin"
@"
@echo off
rem Generated by setup_env.ps1. Enters the mlir-aie environment so that
rem make_artifacts.py can run:
rem
rem   activate_iron.bat
rem   python make_artifacts.py --for-model 512 8 --ffn-hidden 768
rem
rem aiecc needs mlir_aie\bin for its own tools and XRT for xclbinutil.
set "PATH=$mlirBin;$xrt;%PATH%"
call "$venv\Scripts\activate.bat"
"@ | Set-Content -Path $activateBat -Encoding ascii
Note "written"

# ---- verify -----------------------------------------------------------------

Step "Verifying"
$env:PATH = "$mlirBin;$xrt;$env:PATH"
& $venvPy -c "import aie.iron; print('   aie.iron imports OK')"
if ($LASTEXITCODE -ne 0) { Fail "the environment installed but 'import aie.iron' fails." }
& $venvPy (Join-Path $Here "make_artifacts.py") --list --for-model 384 6
if ($LASTEXITCODE -ne 0) { Fail "make_artifacts.py could not run." }

Write-Host "`nDone. Use it with:" -ForegroundColor Green
Write-Host "   cd `"$Here`""
Write-Host "   activate_iron.bat"
Write-Host "   python make_artifacts.py --for-model <trunk> <heads> [points] [--ffn-hidden K]"
