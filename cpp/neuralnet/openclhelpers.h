#ifndef NEURALNET_OPENCL_HELPERS_H_
#define NEURALNET_OPENCL_HELPERS_H_

#include "../core/global.h"
#include "../core/logger.h"
#include "../neuralnet/openclincludes.h"

#define CHECK_ERR(x) { OpenCLHelpers::checkErrors((x),__FILE__,#x,__LINE__); }

struct OpenCLTuneParams;

struct DeviceInfo {
  //Indexes whatever order that OpenCL gives us devices, across all platforms.
  int gpuIdx;

  cl_device_id deviceId;
  std::string name;
  std::string vendor;
  cl_device_type deviceType;
  std::string openCLVersion;

  int defaultDesirability;

  static constexpr int MAX_PLATFORMS = 32;
  static constexpr int MAX_DEVICES = 512;
  static std::vector<DeviceInfo> getAllDeviceInfosOnSystem(Logger* logger);
};

struct InitializedDevice {
  DeviceInfo info;
  cl_command_queue commandQueue;
};

//Wrapper around cl_context for sharing initialization code
struct DevicesContext {
  cl_context context;

  //Index of the default device to use if not specified (user-provided gpuIdx == -1)
  int defaultGpuIdx;

  //Filtered and initialized subset of allDeviceInfos
  std::vector<InitializedDevice> devicesToUse;
  //All unique names of devices being used
  std::vector<std::string> uniqueDeviceNamesToUse;

  DevicesContext(const std::vector<DeviceInfo>& allDeviceInfos, const std::vector<int>& gpuIdxsToUse, Logger* logger, bool enableProfiling);
  ~DevicesContext();

  DevicesContext() = delete;
  DevicesContext(const DevicesContext&) = delete;
  DevicesContext& operator=(const DevicesContext&) = delete;

  //Given the gpuIdx, find the initialized device of that GPU. Fails if it was not a gpuIdx provided in
  //gpuIdxsToUse upon creation of this DevicesContext.
  const InitializedDevice& findGpuExn(int gpuIdx) const;
  //Find devices being used with a given name
  std::vector<InitializedDevice> findDevicesToUseWithName(const std::string& name) const;
  std::vector<cl_device_id> findDeviceIdsToUseWithName(const std::string& name) const;
};

namespace OpenCLHelpers {
  const char* getErrorMessage(cl_int error);
  void checkErrors(cl_int error, const char* file, const char* func, int line);

  struct CompileError final : public StringError { CompileError(const char* msg):StringError(msg) {}; CompileError(const std::string& msg):StringError(msg) {}; };
  cl_program compileProgram(
    const std::string& name,
    cl_context context,
    const std::vector<cl_device_id>& devices,
    const std::string& str,
    const std::string& options
  );
  bool tryCompileProgram(
    const std::string& name,
    cl_context context,
    const std::vector<cl_device_id>& devices,
    const std::string& str,
    const std::string& options,
    cl_program& buf
  );

  cl_mem createReadOnlyBuffer(cl_context context, std::vector<float>& data);
  cl_mem createReadWriteBuffer(cl_context context, std::vector<float>& data);
  cl_mem createReadWriteBuffer(cl_context context, size_t numFloats);

  void blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numFloats, std::vector<float>& dstBuf);

  size_t powerOf2ify(size_t size);
  size_t roundUpToMultiple(size_t size, size_t ofThis);

  cl_int doBatchedXGemm_KM_KN_MN(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int M, int N, int K,
    cl_mem A, cl_mem B, cl_mem C,
    int numBatchElts,
    cl_event* eventBuf
  );

  cl_int doStridedBatchedXGemm_KM_KN_MN(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int M, int N, int K,
    int aStride, int bStride, int cStride,
    cl_mem A, cl_mem B, cl_mem C,
    int numBatchElts,
    cl_event* eventBuf
  );

  cl_int doBatchedXGemm_MK_NK_MN(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int M, int N, int K,
    cl_mem A, cl_mem B, cl_mem C,
    int numBatchElts,
    cl_event* eventBuf
  );

  cl_int doWinogradTransform(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    cl_mem input, cl_mem convWorkspace,
    int batchSize, int nnXLen, int nnYLen,
    int numTilesX, int numTilesY,
    int inChannels,
    int convSize,
    cl_event* eventBuf
  );

  cl_int doWinogradTransformWithBNRelu(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    cl_mem input, cl_mem convWorkspace,
    cl_mem scaleBuf, cl_mem biasBuf, cl_mem mask,
    int batchSize, int nnXLen, int nnYLen,
    int numTilesX, int numTilesY,
    int inChannels,
    int convSize,
    cl_event* eventBuf
  );

  cl_int doWinogradUntransform(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    cl_mem convWorkspace2, cl_mem output,
    int batchSize, int nnXLen, int nnYLen,
    int numTilesX, int numTilesY,
    int outChannels,
    int convSize,
    cl_event* eventBuf
  );

  cl_int performGPool(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int batchSize, int gpoolChannels, int nnXYLen,
    cl_mem gpoolConvOut, cl_mem gpoolConcat, cl_mem maskSum,
    cl_event* eventBuf
  );

  cl_int performValueHeadPool(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int batchSize, int gpoolChannels, int nnXYLen,
    cl_mem gpoolConvOut, cl_mem gpoolConcat, cl_mem maskSum,
    cl_event* eventBuf
  );

  cl_int computeMaskSums(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    cl_mem mask,
    cl_mem maskSum,
    int batchSize,
    int nnXLen,
    int nnYLen,
    cl_event* eventBuf
  );

  cl_int transposeNCHW(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int batchSize, int cSize, int nnXLen, int nnYLen,
    cl_mem input, cl_mem output,
    cl_event* eventBuf
  );

}


#endif //NEURALNET_OPENCL_HELPERS_H_
