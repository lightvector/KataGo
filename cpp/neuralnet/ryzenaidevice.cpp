#include "../neuralnet/ryzenaidevice.h"

#include <algorithm>
#include <cstdlib>

#include "../core/global.h"
#include "../core/os.h"
#include "../dataio/homedata.h"

#ifdef OS_IS_WINDOWS
#include <windows.h>
#endif

// XRT's headers are noisy under MSVC's default warning level and pull in a lot
// of Windows machinery; keep them isolated to this translation unit.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100 4245 4267 4996)
#endif
#include "xrt/xrt_device.h"
#include "xrt/experimental/xrt_system.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "../external/filesystem-1.5.8/include/ghc/filesystem.hpp"

namespace gfs = ghc::filesystem;

using namespace std;

int RyzenAIDevice::maxColumns(Arch arch) {
  switch(arch) {
    case Arch::NPU1: return 4;
    case Arch::NPU2: return 8;
    default: return 0;
  }
}

const char* RyzenAIDevice::archName(Arch arch) {
  switch(arch) {
    case Arch::NPU1: return "npu1";
    case Arch::NPU2: return "npu2";
    default: return "unknown";
  }
}

RyzenAIDevice::Arch RyzenAIDevice::archOfDeviceName(const string& name) {
  const string lowered = Global::toLower(name);
  auto has = [&lowered](const char* needle) { return lowered.find(needle) != string::npos; };

  // XDNA2 / aie2p
  if(has("strix") || has("krackan") || has("halo") || has("npu2"))
    return Arch::NPU2;
  // XDNA1 / aie2
  if(has("phoenix") || has("hawk") || has("npu1"))
    return Arch::NPU1;
  return Arch::Unknown;
}

namespace {

string repointXrtIfNeeded() {
#ifdef OS_IS_WINDOWS
  static const char* const kRuntimeLib = "xrt_core.dll";
#else
  static const char* const kRuntimeLib = "libxrt_core.so";
#endif

  auto hasRuntime = [](const string& dir) {
    if(dir.size() <= 0)
      return false;
    std::error_code ec;
    return gfs::exists(gfs::u8path(dir) / kRuntimeLib, ec);
  };

  string current;
  const char* env = std::getenv("XILINX_XRT");
  if(env != NULL)
    current = env;
  if(hasRuntime(current))
    return "XILINX_XRT = " + current;

  string exeDir;
  try {
    const vector<string> dirs = HomeData::getDefaultFilesDirs();
    if(dirs.size() > 0)
      exeDir = dirs[0];
  }
  catch(const std::exception&) {
    // Leave exeDir empty; handled below.
  }
  if(!hasRuntime(exeDir)) {
    if(current.size() > 0)
      return "XILINX_XRT = " + current + " (no " + kRuntimeLib +
             " there, and none next to the executable either)";
    return string("XILINX_XRT is unset and no ") + kRuntimeLib +
           " sits next to the executable; relying on the system library path";
  }

  // Both the Win32 environment block and the CRT copy need updating: XRT lives
  // in its own DLL with its own CRT, and which one it reads is not contractual.
#ifdef OS_IS_WINDOWS
  SetEnvironmentVariableA("XILINX_XRT", exeDir.c_str());
  _putenv_s("XILINX_XRT", exeDir.c_str());
#else
  setenv("XILINX_XRT", exeDir.c_str(), 1);
#endif

  if(current.size() > 0)
    return "XILINX_XRT was " + current + ", which has no " + kRuntimeLib +
           "; repointed at " + exeDir;
  return "XILINX_XRT set to " + exeDir;
}

}  // namespace

string RyzenAIDevice::ensureRuntimeLibraryPath() {
  // Only the first call changes anything. Later callers are logging paths, and
  // caching means they report the repair that actually happened rather than the
  // already-repaired state, which reads as if nothing had been wrong.
  static const string description = repointXrtIfNeeded();
  return description;
}

RyzenAIDevice::Arch RyzenAIDevice::archOfDevice(int deviceIdx) {
  try {
    xrt::device device((unsigned int)(deviceIdx < 0 ? 0 : deviceIdx));
    return archOfDeviceName(device.get_info<xrt::info::device::name>());
  }
  catch(const std::exception&) {
    return Arch::Unknown;
  }
}

vector<RyzenAIDevice::Info> RyzenAIDevice::enumerate() {
  vector<Info> devices;
  unsigned int numDevices = 0;
  try {
    numDevices = xrt::system::enumerate_devices();
  }
  catch(const std::exception&) {
    // No driver, no device, or an XRT that cannot talk to it. Not an error
    // here - the caller reports "none found".
    return devices;
  }

  for(unsigned int i = 0; i < numDevices; i++) {
    Info info;
    info.index = (int)i;
    info.arch = Arch::Unknown;
    try {
      xrt::device device(i);
      info.name = device.get_info<xrt::info::device::name>();
      info.bdf = device.get_info<xrt::info::device::bdf>();
      info.arch = archOfDeviceName(info.name);
    }
    catch(const std::exception& e) {
      // Report the device as present but undescribable rather than dropping it,
      // so that a device that exists-but-is-busy is still visible to the user.
      info.name = string("(could not query: ") + e.what() + ")";
    }
    devices.push_back(info);
  }
  return devices;
}

string RyzenAIDevice::describeRuntime() {
  vector<Info> devices = enumerate();
  if(devices.size() <= 0)
    return string(
      "No NPU devices found by XRT. Check that the AMD NPU driver is installed and that the NPU is "
      "enabled in the BIOS."
    );

  string ret = "Found " + Global::uint64ToString((uint64_t)devices.size()) + " NPU device(s):";
  for(size_t i = 0; i < devices.size(); i++) {
    ret += "\n  Device " + Global::intToString(devices[i].index) + ": " + devices[i].name +
           (devices[i].bdf.size() > 0 ? (" [" + devices[i].bdf + "]") : string(""));
    if(devices[i].arch == Arch::Unknown)
      ret += " - unrecognized architecture, NPU acceleration unavailable (will use the CPU reference path)";
    else
      ret += " - " + string(archName(devices[i].arch)) + ", up to " +
             Global::intToString(maxColumns(devices[i].arch)) + " columns";
  }
  return ret;
}

string RyzenAIDevice::resolveArtifactDir(const string& configuredDir) {
  if(configuredDir.size() > 0)
    return configuredDir;

  // Kernel artifacts are deployed next to the executable, so that an install is
  // just exe + dlls + this directory.
  vector<string> dirs = HomeData::getDefaultFilesDirs();
  if(dirs.size() > 0) {
    gfs::path p = gfs::u8path(dirs[0]) / "ryzenai";
    return p.u8string();
  }
  return string("ryzenai");
}

vector<string> RyzenAIDevice::listXclbins(const string& dir) {
  vector<string> names;
  try {
    gfs::path p = gfs::u8path(dir);
    if(!gfs::exists(p) || !gfs::is_directory(p))
      return names;
    // Artifacts are grouped into per-dtype subdirectories (bf16/, ...), so
    // recurse and report paths relative to dir rather than bare filenames.
    for(const auto& entry : gfs::recursive_directory_iterator(p)) {
      if(!gfs::is_regular_file(entry.path()))
        continue;
      string filename = entry.path().filename().u8string();
      if(Global::isSuffix(Global::toLower(filename), ".xclbin"))
        names.push_back(gfs::relative(entry.path(), p).generic_u8string());
    }
  }
  catch(const std::exception&) {
    // An unreadable directory is equivalent to an empty one for our purposes.
    return names;
  }
  std::sort(names.begin(), names.end());
  return names;
}
