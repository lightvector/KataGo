<#
.SYNOPSIS
Set up the mlir-aie ("IRON") toolchain that make_artifacts.py needs.

.DESCRIPTION
Only kernel authors need this. Building katago needs nothing from here, and
running it needs nothing at all beyond the NPU driver - the .xclbin kernels are
committed to this repository. This is for regenerating them.

The toolchain is a git checkout, not just a set of wheels: the GEMM generators
compile against kernel sources that live in the repository's aie_kernels tree
and are not part of any wheel. So this clones mlir-aie and runs the installer
that ships inside it (utils/iron_setup.py), which builds the virtual
environment and writes the iron_env activation scripts. Doing it that way means
the environment always matches whatever that checkout expects, rather than a
second guess maintained here.

What it does, in order:

  1. Checks the prerequisites (git, Python, the MSVC C++ toolchain, XRT) and
     stops with an explanation if one is missing. It never installs any of them.
  2. Clones mlir-aie into -Prefix, or reuses the checkout already there.
  3. Runs its utils/iron_setup.py to create the environment.
  4. Writes activate_iron.bat next to this script, which enters that
     environment from anywhere.
  5. Verifies: aie.iron imports, the aie_kernels tree the GEMM generators need
     is present, and make_artifacts.py runs.

Nothing is cloned or installed unless -Execute is passed. Without it the script
only checks prerequisites and prints what it would do, so several gigabytes are
never fetched by surprise.

.PARAMETER Prefix
Where to put the mlir-aie checkout and its environment. Required, deliberately:
this is a multi-gigabyte tree and where it belongs is your call.

.PARAMETER Ref
Git ref (branch, tag or commit) to check out. Defaults to the default branch.

.PARAMETER Execute
Actually clone and install. Without this the script only reports.

.EXAMPLE
.\setup_env.ps1 -Prefix C:\Envs\mlir-aie
Report what would be done, changing nothing.

.EXAMPLE
.\setup_env.ps1 -Prefix C:\Envs\mlir-aie -Execute
Clone and install for real.
#>
[CmdletBinding()]
param(
  [string]$Prefix,
  [string]$Ref,
  [switch]$Execute
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoUrl = "https://github.com/Xilinx/mlir-aie.git"

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Note($msg) { Write-Host "   $msg" }

if (-not $Prefix) {
  Write-Host @"
ERROR: -Prefix is required.

Give the directory to put the mlir-aie checkout and its environment in. It is
not defaulted because it is a multi-gigabyte tree and where that belongs is your
call, not this script's.

  .\setup_env.ps1 -Prefix C:\Envs\mlir-aie              # report only
  .\setup_env.ps1 -Prefix C:\Envs\mlir-aie -Execute     # actually install
  .\setup_env.ps1 -Prefix D:\tools\mlir-aie -Execute    # anywhere else

See Compiling.md, "Regenerating the NPU kernels", for the full flow.
"@ -ForegroundColor Red
  exit 1
}

# ---- 1. prerequisites, checked but never installed --------------------------

Step "Checking prerequisites"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
  Fail "git is not on PATH. The toolchain is a git checkout, not just wheels."
}
Note "git present"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Fail "python is not on PATH. Install Python 3.10 or newer." }
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
if (-not $vsPath) { Fail "No Visual Studio C++ toolchain found. Install the 'Desktop development with C++' workload." }
Note "MSVC toolchain at $vsPath"

# XRT: probe for xclbinutil.exe rather than for a bin\ directory - the Windows
# SDK puts its tools straight in the install root, and xclbinutil is the one
# thing aiecc actually shells out to.
$xrtRoot = $null
foreach ($cand in @($env:XRT_ROOT, $env:XILINX_XRT, "C:\Xilinx\XRT")) {
  if (-not $cand) { continue }
  foreach ($sub in @("", "bin")) {
    $dir = if ($sub) { Join-Path $cand $sub } else { $cand }
    if (Test-Path (Join-Path $dir "xclbinutil.exe")) { $xrtRoot = $cand; break }
  }
  if ($xrtRoot) { break }
}
if (-not $xrtRoot) {
  Fail "xclbinutil.exe not found. Install AMD's XRT for Windows and either set XRT_ROOT to it or accept the default C:\Xilinx\XRT - aiecc shells out to xclbinutil to package each .xclbin."
}
Note "XRT at $xrtRoot"

# ---- the plan ---------------------------------------------------------------

$activateBat = Join-Path $Here "activate_iron.bat"
$ironPs1 = Join-Path $Prefix "iron_env.ps1"
$cloneArgs = if ($Ref) { "clone $RepoUrl `"$Prefix`" && git -C `"$Prefix`" checkout $Ref" } else { "clone $RepoUrl `"$Prefix`"" }

Step "Plan"
Note "checkout   : $Prefix   (from $RepoUrl$(if ($Ref) { ", ref $Ref" }))"
Note "installer  : $Prefix\utils\iron_setup.py --xrt-root `"$xrtRoot`""
Note "activator  : $activateBat  ->  $ironPs1"
if (Test-Path (Join-Path $Prefix ".git")) { Note "note       : a checkout already exists there and will be reused, not re-cloned" }

if (-not $Execute) {
  Write-Host "`nDry run - nothing was cloned or installed. Re-run with -Execute to proceed." -ForegroundColor Yellow
  Write-Host "The checkout plus its environment is several gigabytes." -ForegroundColor Yellow
  exit 0
}

# ---- 2. the checkout --------------------------------------------------------

Step "Getting mlir-aie"
if (Test-Path (Join-Path $Prefix ".git")) {
  Note "reusing the checkout already at $Prefix"
  if ($Ref) {
    & git -C $Prefix fetch --all --quiet
    & git -C $Prefix checkout $Ref
    if ($LASTEXITCODE -ne 0) { Fail "could not check out $Ref." }
  }
} else {
  if ((Test-Path $Prefix) -and (Get-ChildItem $Prefix -Force | Select-Object -First 1)) {
    Fail "$Prefix exists and is not empty, but is not a git checkout. Point -Prefix somewhere else, or clear it."
  }
  & git clone $RepoUrl $Prefix
  if ($LASTEXITCODE -ne 0) { Fail "git clone failed." }
  if ($Ref) {
    & git -C $Prefix checkout $Ref
    if ($LASTEXITCODE -ne 0) { Fail "could not check out $Ref." }
  }
}
Note ("at commit " + (& git -C $Prefix rev-parse --short HEAD))

# ---- 3. its own installer ---------------------------------------------------

Step "Running the checkout's own installer (utils/iron_setup.py)"
Push-Location $Prefix
try {
  & python (Join-Path $Prefix "utils\iron_setup.py") --xrt-root $xrtRoot
  if ($LASTEXITCODE -ne 0) { Fail "iron_setup.py failed. See its output above." }
} finally {
  Pop-Location
}
if (-not (Test-Path $ironPs1)) {
  Fail "iron_setup.py finished but did not write $ironPs1. The checkout's layout may have changed."
}

# ---- 4. the activator -------------------------------------------------------

Step "Writing $activateBat"
$ironCmd = Join-Path $Prefix "iron_env.cmd"
if (-not (Test-Path $ironCmd)) { Fail "expected $ironCmd from iron_setup.py, but it is not there." }
@"
@echo off
rem Generated by setup_env.ps1. Enters the mlir-aie environment so that
rem make_artifacts.py can run:
rem
rem   activate_iron.bat
rem   python make_artifacts.py --for-model 512 8 --ffn-hidden 768
rem
rem The real work is done by the checkout's own iron_env.cmd, which sets
rem MLIR_AIE_INSTALL_DIR, PEANO_INSTALL_DIR, XRT_ROOT and the rest.
call "$ironCmd"
"@ | Set-Content -Path $activateBat -Encoding ascii
Note "written"

# ---- 5. verify --------------------------------------------------------------

Step "Verifying"
$venvPy = Join-Path $Prefix "ironenv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Fail "no interpreter at $venvPy." }

& $venvPy -c "import aie.iron; print('   aie.iron imports OK')"
if ($LASTEXITCODE -ne 0) { Fail "the environment installed but 'import aie.iron' fails." }

# The GEMM generators compile against kernel sources from the checkout, not from
# any wheel, so a wheel-only environment would pass the import above and then
# fail at the first build.
$kernels = Join-Path $Prefix "aie_kernels\aie2p"
if (-not (Test-Path $kernels)) {
  Fail "$kernels is missing. The GEMM generators need the checkout's kernel sources; a wheels-only install is not enough."
}
Note "aie_kernels\aie2p present"

& $venvPy (Join-Path $Here "make_artifacts.py") --list --for-model 384 6
if ($LASTEXITCODE -ne 0) { Fail "make_artifacts.py could not run." }

Write-Host "`nDone. Use it with:" -ForegroundColor Green
Write-Host "   cd `"$Here`""
Write-Host "   activate_iron.bat"
Write-Host "   python make_artifacts.py --for-model <trunk> <heads> [points] [--ffn-hidden K]"
