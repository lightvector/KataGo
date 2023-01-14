#ifndef NEURALNET_OPENCL_HELPERS_H_
#define NEURALNET_OPENCL_HELPERS_H_

#include "../core/global.h"
#include "../core/logger.h"
#include "../neuralnet/openclincludes.h"

#include "../external/half-2.1.0/include/half.hpp"

#define CHECK_ERR(x) { OpenCLHelpers::checkErrors((x),__FILE__,#x,__LINE__); }

struct OpenCLTuneParams;
namespace OpenCLParams {
  struct XGemmParams;
}

struct DeviceInfo {
  //Indexes whatever order that OpenCL gives us devices, across all platforms.
  int gpuIdx;

  cl_device_id deviceId;
  cl_platform_id platformId;
  std::string platformDesc;
  std::string name;
  std::string vendor;
  cl_device_type deviceType;
  std::string openCLVersion;
  std::string extensions;

  int defaultDesirability;
  bool supportsFP16Compute;

  static constexpr int MAX_PLATFORMS = 32;
  static constexpr int MAX_DEVICES = 512;
  static std::vector<DeviceInfo> getAllDeviceInfosOnSystem(Logger* logger);
};

struct InitializedPlatform {
  cl_context context;
  cl_platform_id platformId;
  std::string platformDesc;
  std::vector<cl_context_properties> properties;
  std::vector<cl_device_id> deviceIdsToUseForThisPlatform;
};

struct InitializedDevice {
  DeviceInfo info;
  cl_context context;
  cl_command_queue commandQueue;
};

//Wrapper around cl_context for sharing initialization code
struct DevicesContext {
  //Index of the default device to use if not specified (user-provided gpuIdx == -1)
  int defaultGpuIdx;
  //All platforms with for which we made a context
  //A platform might be in here more than once, with different contexts
  std::vector<InitializedPlatform*> initializedPlatforms;

  //Filtered and initialized subset of allDeviceInfos
  std::vector<InitializedDevice*> devicesToUse;
  //All unique names of devices being used
  std::vector<std::string> uniqueDeviceNamesToUse;

  DevicesContext(const std::vector<DeviceInfo>& allDeviceInfos, const std::vector<int>& gpuIdxsToUse, Logger* logger, bool enableProfiling);
  ~DevicesContext();

  DevicesContext() = delete;
  DevicesContext(const DevicesContext&) = delete;
  DevicesContext& operator=(const DevicesContext&) = delete;

  //Given the gpuIdx, find the initialized device of that GPU. Fails if it was not a gpuIdx provided in
  //gpuIdxsToUse upon creation of this DevicesContext.
  const InitializedDevice* findGpuExn(int gpuIdx) const;
  //Find devices being used with a given name
  std::vector<const InitializedDevice*> findDevicesToUseWithName(const std::string& name) const;
  std::vector<cl_device_id> findDeviceIdsToUseWithName(const std::string& name) const;
};

namespace OpenCLHelpers {
  const std::string getErrorMessage(cl_int error);
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
    cl_program& buf,
    std::string& errorMessage
  );

  cl_mem createReadOnlyBuffer(cl_context context, std::vector<float>& data);
  cl_mem createReadOnlyBuffer(cl_context context, std::vector<half_float::half>& data);
  cl_mem createReadWriteBuffer(cl_context context, std::vector<float>& data);
  cl_mem createReadWriteBuffer(cl_context context, std::vector<half_float::half>& data);
  cl_mem createReadWriteBufferFloat(cl_context context, size_t numElts);
  cl_mem createReadWriteBufferHalf(cl_context context, size_t numElts);
  cl_mem createReadWriteBufferFloatZeros(cl_context context, size_t numElts);
  cl_mem createReadWriteBufferHalfZeros(cl_context context, size_t numElts);
  cl_mem createReadWriteBufferBytes(cl_context clContext, size_t numBytes);

  void blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, std::vector<float>& dstBuf);
  void blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, float* dstBuf);
  void blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, std::vector<half_float::half>& dstBuf);
  void blockingReadBufferHalfToFloat(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, std::vector<float>& dstBuf);
  void blockingReadBufferHalfToFloat(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, float* dstBuf);
  void blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, std::vector<float>& dstBuf, bool useFP16);
  void blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, float* dstBuf, bool useFP16);

  size_t powerOf2ify(size_t size);
  size_t roundUpToMultiple(size_t size, size_t ofThis);
  int roundUpToMultipleInt(size_t size, size_t ofThis);

  cl_int doBatchedXGemm_KM_KN_NM(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLParams::XGemmParams& tuneParams,
    int M, int N, int K,
    cl_mem A, cl_mem B, cl_mem C,
    int numBatchElts,
    cl_event* eventBuf
  );

  cl_int doBatchedHGemmWmma_KM_KN_NM(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int M, int N, int K,
    cl_mem A, cl_mem B, cl_mem C,
    int numBatchElts,
    cl_event* eventBuf
  );

  cl_int doHGemmWmma_NCHW_ICOC(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int batchSize, int cSize, int hwSize, int ocSize,
    cl_mem A, cl_mem B, cl_mem C,
    cl_event* eventBuf
  );

  cl_int doBatchedXGemmDirect_KM_KN_NM(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int M, int N, int K,
    cl_mem A, cl_mem B, cl_mem C,
    int numBatchElts,
    cl_event* eventBuf
  );

  cl_int doStridedBatchedXGemmDirect_KM_KN_NM(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int M, int N, int K,
    int aStride, int bStride, int cStride,
    cl_mem A, cl_mem B, cl_mem C,
    int numBatchElts,
    cl_event* eventBuf
  );

  cl_int doBatchedXGemmDirect_MK_NK_MN(
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
    int nnXLen, int nnYLen,
    int batchSize, int numTilesX, int numTilesY, int batchNumTilesPadMultiple,
    int inChannels, int inChannelsPadMultiple,
    int convSize,
    cl_event* eventBuf
  );

  cl_int doWinogradTransformWithBNAct(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    cl_mem input, cl_mem convWorkspace,
    cl_mem scaleBuf, cl_mem biasBuf, cl_mem mask,
    int nnXLen, int nnYLen,
    int batchSize, int numTilesX, int numTilesY, int batchNumTilesPadMultiple,
    int inChannels, int inChannelsPadMultiple,
    int convSize,
    cl_event* eventBuf
  );

  cl_int doWinogradUntransform(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    cl_mem convWorkspace2, cl_mem output,
    int nnXLen, int nnYLen,
    int batchSize, int numTilesX, int numTilesY, int batchNumTilesPadMultiple,
    int outChannels, int outChannelsPadMultiple,
    int convSize,
    cl_event* eventBuf
  );

  cl_int performGPoolMask(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    const OpenCLTuneParams& tuneParams,
    int batchSize, int gpoolChannels, int nnXYLen,
    cl_mem gpoolConvOut, cl_mem gpoolConcat, cl_mem mask, cl_mem maskSum,
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

  cl_int doAddPointWise(
    cl_kernel kernel,
    cl_command_queue commandQueue,
    cl_mem acc,
    cl_mem value,
    int totalSize,
    cl_event* eventBuf
  );
}


#endif //NEURALNET_OPENCL_HELPERS_H_
