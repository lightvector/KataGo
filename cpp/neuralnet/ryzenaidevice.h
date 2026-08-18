/*
 * XRT device discovery and NPU kernel-artifact (xclbin) location for the
 * RyzenAI backend.
 *
 * This layer is deliberately the ONLY place that includes XRT headers, so that
 * the rest of the backend can be built and reasoned about without them.
 */

#ifndef NEURALNET_RYZENAI_DEVICE_H_
#define NEURALNET_RYZENAI_DEVICE_H_

#include <string>
#include <vector>

namespace RyzenAIDevice {

  // AIE architecture family. Kernel binaries are per-family and NOT
  // interchangeable: XDNA2 cores run the aie2p ISA, XDNA1 cores run aie2, so an
  // xclbin built for one will not load on the other. The families also differ in
  // width - XDNA1 exposes 4 columns (16 compute tiles) to the toolchain, XDNA2
  // exposes 8 (32 tiles) - and only XDNA2 supports the BFP16 path.
  enum class Arch { Unknown, NPU1, NPU2 };

  // Largest column count the toolchain can target for a family, which is also
  // the widest artifact variant we ship for it.
  int maxColumns(Arch arch);
  const char* archName(Arch arch);  // "npu1" / "npu2" / "unknown"

  struct Info {
    int index;
    std::string name;  // e.g. "NPU Strix"
    std::string bdf;   // e.g. "00c2:00:01.1"
    Arch arch;
  };

  // Classifies a device by the product name XRT reports. Substring matching is
  // all that is available: XRT exposes no architecture field, so new silicon
  // will read as Unknown until its name is added here (which degrades to the CPU
  // reference path rather than loading a wrong-ISA xclbin).
  Arch archOfDeviceName(const std::string& name);

  // Architecture of one device, or Unknown if it cannot be opened/classified.
  // deviceIdx < 0 selects the default device.
  Arch archOfDevice(int deviceIdx);

  // Makes XRT able to find its own runtime libraries, and must be called before
  // any other function here.
  //
  // XRT resolves xrt_core from $XILINX_XRT. The Windows XRT *SDK* installs only
  // headers and import libraries under that path - the runtime DLLs live in the
  // driver store and are shipped next to katago.exe - so a machine where
  // XILINX_XRT points at the SDK (the natural thing to do when building) has
  // every XRT call fail with "No such library ...\\xrt_core.dll", which surfaces
  // as "no NPU devices found" rather than as anything pointing at the cause.
  //
  // If the variable is unset or points somewhere without xrt_core, and the DLL
  // is present beside the executable, this repoints it there. Returns a
  // description of what it did, for logging; never throws.
  std::string ensureRuntimeLibraryPath();

  // Enumerates NPU devices visible to XRT. Never throws: on any XRT failure it
  // returns an empty vector, since callers (notably printDevices) must stay
  // usable on machines with no NPU or no driver.
  std::vector<Info> enumerate();

  // Human-readable one-line summary of the XRT/driver stack, e.g. for logging.
  // Returns an explanatory string rather than throwing if XRT is unavailable.
  std::string describeRuntime();

  // Directory holding the NPU kernel binaries. Resolution order:
  //   1. configuredDir, if non-empty (the ryzenaiArtifactDir config key)
  //   2. "<directory containing the running executable>/ryzenai"
  // The directory is not required to exist; callers decide how to react.
  std::string resolveArtifactDir(const std::string& configuredDir);

  // Names (not full paths) of the *.xclbin files found directly under dir, or
  // an empty vector if dir does not exist. Sorted, for reproducible logging.
  std::vector<std::string> listXclbins(const std::string& dir);

}  // namespace RyzenAIDevice

#endif  // NEURALNET_RYZENAI_DEVICE_H_
