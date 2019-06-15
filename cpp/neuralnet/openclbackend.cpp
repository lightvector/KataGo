
#include "../neuralnet/nninterface.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../neuralnet/nninputs.h"
#include "../neuralnet/openclkernels.h"

using namespace std;

static const char* getErrorString(cl_int error)
{
  switch(error){
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  default: return "Unknown OpenCL error";
  }
}

static void checkErrors(cl_int error, const char* file, const char* func, int line) {
  if(error != 0)
    throw StringError(string("OpenCL error at ") + file + ", func " + func + ", line " + Global::intToString(line) + ", error " + getErrorString(error));
}
#define CHECK_ERR(x) { checkErrors((x),__FILE__,#x,__LINE__); }

static size_t powerOf2ify(size_t size) {
  if(size <= 2)
    return size;
  if(size <= 4)
    return 4;
  size_t s = 1;
  while(s * 4 < size)
    s *= 2;

  if(s >= size)
    return s;
  if(s * 2 >= size)
    return s * 2;
  if(s * 3 >= size)
    return s * 3;
  assert(s * 4 >= size);
  return s * 4;
}

template<typename T>
static size_t byteSizeofVectorContents(const typename std::vector<T>& vec) {
    return sizeof(T) * vec.size();
}

//---------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
}

void NeuralNet::globalCleanup() {
}


//---------------------------------------------------------------------------------------------------------

static cl_program compileProgram(const string& name, cl_context context, const vector<cl_device_id>& devices, const string& str) {
  const char* lines[1] = {str.c_str()};
  const size_t sizes[1] = {str.size()};
  cl_int err;
  cl_program program = clCreateProgramWithSource(context,1,lines,sizes,&err);
  CHECK_ERR(err);
  const char* options = NULL;
  //TODO test
  // const char* options = "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero";
  err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
  if(err != 0) {
    for(int i = 0; i<devices.size(); i++) {
      cl_int err2;
      vector<char> buf(100000);
      size_t retSize;
      err2 = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, byteSizeofVectorContents(buf), buf.data(), &retSize);
      CHECK_ERR(err2);
      cout << "BUILD LOG FOR " << name << " ON DEVICE " << i << endl;
      cout << buf.data() << endl;
    }
    CHECK_ERR(err);
  }

  CHECK_ERR(err);
  return program;
}

struct ComputeContext {
  cl_context context;
  vector<cl_platform_id> platformIds;
  vector<cl_device_id> deviceIds;
  vector<string> deviceNames;
  vector<string> deviceVendors;

  vector<int> gpuIdxsToUse;
  vector<cl_device_id> deviceIdsToUse;
  vector<cl_command_queue> commandQueues;

  cl_program conv2dNCHWProgram;

  ComputeContext(const vector<int>& gIdxs, Logger* logger)
    : platformIds(32),
      deviceIds(512),
      deviceNames(),
      deviceVendors(),
      gpuIdxsToUse(gIdxs),
      deviceIdsToUse(),
      commandQueues()
  {
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(platformIds.size(), platformIds.data(), &numPlatforms);
    CHECK_ERR(err);
    assert(numPlatforms <= platformIds.size());
    platformIds.resize(numPlatforms);

    int numDevicesTotal = 0;
    for(int i = 0; i<numPlatforms && numDevicesTotal < deviceIds.size(); i++) {
      cl_uint numDevices;
      err = clGetDeviceIDs(
        platformIds[i], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, deviceIds.size() - numDevicesTotal,
        deviceIds.data() + numDevicesTotal, &numDevices);
      CHECK_ERR(err);
      assert(numDevices <= deviceIds.size());
      numDevicesTotal += numDevices;
    }
    deviceIds.resize(numDevicesTotal);

    constexpr int bufLen = 2048;
    char buf[bufLen];
    for(int i = 0; i<bufLen; i++)
      buf[i] = '\0';

    for(int i = 0; i<numDevicesTotal; i++) {
      size_t sizeRet;
      err = clGetDeviceInfo(deviceIds[i], CL_DEVICE_NAME, bufLen, buf, &sizeRet);
      CHECK_ERR(err);
      deviceNames.push_back(string(buf));
      err = clGetDeviceInfo(deviceIds[i], CL_DEVICE_VENDOR, bufLen, buf, &sizeRet);
      CHECK_ERR(err);
      deviceVendors.push_back(string(buf));
      if(logger != NULL)
        logger->write("Found OpenCL Device " + Global::intToString(i) + ": " + deviceNames[i] + " (" + deviceVendors[i] + ")");
    }

    for(size_t i = 0; i<gpuIdxsToUse.size(); i++) {
      int gpuIdx = gpuIdxsToUse[i];
      if(gpuIdx < 0 || gpuIdx >= numDevicesTotal)
        throw StringError("Requested gpuIdx/device " + Global::intToString(gpuIdx) + " was not found");
      deviceIdsToUse.push_back(deviceIds[gpuIdx]);
    }

    cl_context_properties* properties = NULL;
    cl_uint numDevicesToUse = (cl_uint)deviceIdsToUse.size();
    context = clCreateContext(properties, numDevicesToUse, deviceIdsToUse.data(), NULL, NULL, &err);
    CHECK_ERR(err);

    for(size_t i = 0; i<gpuIdxsToUse.size(); i++) {
      //TODO consider CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
      cl_command_queue commandQueue = clCreateCommandQueue(context, deviceIdsToUse[i], 0, &err);
      CHECK_ERR(err);
      commandQueues.push_back(commandQueue);
    }

    conv2dNCHWProgram = compileProgram("conv2dNCHWProgram", context, deviceIdsToUse, OpenCLKernels::conv2dNCHW);
  }

  ~ComputeContext() {
    clReleaseProgram(conv2dNCHWProgram);
    for(int i = 0; i<commandQueues.size(); i++) {
      clFlush(commandQueues[i]);
      clFinish(commandQueues[i]);
      clReleaseCommandQueue(commandQueues[i]);
    }
    clReleaseContext(context);
  }

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

  int findWhichGpu(int gpuIdx) const {
    for(int i = 0; i<gpuIdxsToUse.size(); i++) {
      if(gpuIdxsToUse[i] == gpuIdx)
        return i;
    }
    throw StringError("Attempted to create ComputeHandle for a gpuIdx that was not part of the ComputeContext: " + Global::intToString(gpuIdx));
  }
};


ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger
) {
  return new ComputeContext(gpuIdxs,logger);
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}


//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  (void)modelFileIdx;
  LoadedModel* loadedModel = new LoadedModel(file);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

//--------------------------------------------------------------

struct ComputeHandle {
  cl_context clContext;
  cl_command_queue commandQueue;

  cl_kernel conv2dNCHWKernel;

  ComputeHandle(ComputeContext* context, const LoadedModel* loadedModel, int gpuIdx, int maxBatchSize, int nnXLen, int nnYLen) {
    clContext = context->context;
    int which = context->findWhichGpu(gpuIdx);
    commandQueue = context->commandQueues[which];

    cl_int err;
    conv2dNCHWKernel = clCreateKernel(context->conv2dNCHWProgram, "conv2dNCHW", &err);
    CHECK_ERR(err);

    //TODO note that loaded model can be null, in which case we're just testing one thing
    (void)loadedModel;
    (void)maxBatchSize;
    (void)nnXLen;
    (void)nnYLen;
  }

  ~ComputeHandle() {
    clReleaseKernel(conv2dNCHWKernel);
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  int nnXLen,
  int nnYLen,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  bool useFP16,
  bool cudaUseNHWC
) {
  (void)cudaUseNHWC;

  if(logger != NULL)
    logger->write("OpenCL backend: Model version " + Global::intToString(loadedModel->modelDesc.version));

  //Current implementation always tolerates excess nn len
  (void)requireExactNNLen;

  if(inputsUseNHWC != false)
    throw StringError("OpenCL backend: inputsUseNHWC = false required, other configurations not supported");
  if(useFP16 != false)
    throw StringError("OpenCL backend: useFP16 = false required, other configurations not supported");

  return new ComputeHandle(context,loadedModel,gpuIdxForThisThread,maxBatchSize,nnXLen,nnYLen);
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

//--------------------------------------------------------------

struct ConvLayer {
  int convYSize;
  int convXSize;
  int convYRadius;
  int convXRadius;
  int inChannels;
  int outChannels;
  int dilationY;
  int dilationX;

  int nnXLen;
  int nnYLen;
  cl_mem filter;

  static constexpr int nKernelDims = 3;
  size_t globalSizes[nKernelDims];

  ConvLayer(ComputeHandle* handle, const ConvLayerDesc* desc, int nnX, int nnY) {
    convYSize = desc->convYSize;
    convXSize = desc->convXSize;
    convYRadius = convYSize / 2;
    convXRadius = convXSize / 2;
    inChannels = desc->inChannels;
    outChannels = desc->outChannels;
    dilationY = desc->dilationY;
    dilationX = desc->dilationX;

    nnXLen = nnX;
    nnYLen = nnY;

    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);
    if(dilationX != 1 || dilationY != 1)
      throw StringError("OpenCL backend: Encountered convolution dilation factors other than 1, not supported");

    cl_int err;
    vector<float> weights = desc->weights;
    filter = clCreateBuffer(
      handle->clContext,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      byteSizeofVectorContents(weights),
      weights.data(),
      &err
    );
    CHECK_ERR(err);

    globalSizes[0] = powerOf2ify(nnXLen);
    globalSizes[1] = powerOf2ify(nnYLen);
    globalSizes[2] = powerOf2ify(outChannels);
  }

  ~ConvLayer() {
    clReleaseMemObject(filter);
  }

  void apply(ComputeHandle* handle, cl_mem input, cl_mem output, int batchSize) {
    cl_kernel kernel = handle->conv2dNCHWKernel;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&batchSize);
    clSetKernelArg(kernel, 4, sizeof(int), (void *)&nnYLen);
    clSetKernelArg(kernel, 5, sizeof(int), (void *)&nnXLen);
    clSetKernelArg(kernel, 6, sizeof(int), (void *)&outChannels);
    clSetKernelArg(kernel, 7, sizeof(int), (void *)&inChannels);
    clSetKernelArg(kernel, 8, sizeof(int), (void *)&convYRadius);
    clSetKernelArg(kernel, 9, sizeof(int), (void *)&convXRadius);

    cl_int err;
    size_t* localSizes = NULL; //TODO actually pick these
    err = clEnqueueNDRangeKernel(
      handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
    );
    CHECK_ERR(err);
  }

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;
};


//--------------------------------------------------------------

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  (void)loadedModel;
  (void)maxBatchSize;
  (void)nnXLen;
  (void)nnYLen;
  throw StringError("Dummy neural net backend: NeuralNet::createInputBuffers unimplemented");
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  if(buffers != NULL)
    throw StringError("Dummy neural net backend: NeuralNet::freeInputBuffers unimplemented");
}

//--------------------------------------------------------------

float* NeuralNet::getBatchEltSpatialInplace(InputBuffers* buffers, int nIdx) {
  (void)buffers;
  (void)nIdx;
  throw StringError("Dummy neural net backend: NeuralNet::getBatchEltSpatialInplace unimplemented");
}

float* NeuralNet::getBatchEltGlobalInplace(InputBuffers* buffers, int nIdx) {
  (void)buffers;
  (void)nIdx;
  throw StringError("Dummy neural net backend: NeuralNet::getBatchEltGlobalInplace unimplemented");
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* buffers) {
  (void)buffers;
  throw StringError("Dummy neural net backend: NeuralNet::getSymmetriesInplace unimplemented");
}

int NeuralNet::getBatchEltSpatialLen(const InputBuffers* buffers) {
  (void)buffers;
  throw StringError("Dummy neural net backend: NeuralNet::getBatchEltSpatialLen unimplemented");
}

int NeuralNet::getBatchEltGlobalLen(const InputBuffers* buffers) {
  (void)buffers;
  throw StringError("Dummy neural net backend: NeuralNet::getBatchEltGlobalLen unimplemented");
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* buffers,
  int numBatchEltsFilled,
  vector<NNOutput*>& outputs
) {
  (void)gpuHandle;
  (void)buffers;
  (void)numBatchEltsFilled;
  (void)outputs;
  throw StringError("Dummy neural net backend: NeuralNet::getOutput unimplemented");
}



bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer
) {
  Logger* logger = NULL;
  cl_int err;
  int gpuIdx = 0;
  ComputeContext* context = createComputeContext({gpuIdx}, logger);

  if(useFP16 != false)
    return false;
  if(useNHWC != false)
    return false;

  LoadedModel* loadedModel = NULL;
  bool requireExactNNLen = false;
  bool inputsUseNHWC = useNHWC;
  ComputeHandle* handle = createComputeHandle(
    context, loadedModel, logger, batchSize, nnXLen, nnYLen, requireExactNNLen, inputsUseNHWC, gpuIdx, useFP16, useNHWC
  );

  ConvLayer* layer = new ConvLayer(handle, desc, nnXLen, nnYLen);

  size_t numInputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->inChannels;
  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateConv: unexpected input buffer size");
  outputBuffer.resize(numOutputFloats);

  vector<float> inputTmp = inputBuffer;
  cl_mem input = clCreateBuffer(
    handle->clContext,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    byteSizeofVectorContents(inputTmp),
    inputTmp.data(),
    &err
  );
  CHECK_ERR(err);

  cl_mem output = clCreateBuffer(handle->clContext, CL_MEM_WRITE_ONLY, byteSizeofVectorContents(outputBuffer), NULL, &err);
  CHECK_ERR(err);
  layer->apply(handle, input, output, batchSize);

  cl_bool blocking = CL_TRUE;
  err = clEnqueueReadBuffer(handle->commandQueue, output, blocking, 0, byteSizeofVectorContents(outputBuffer), outputBuffer.data(), 0, NULL, NULL);

  clReleaseMemObject(output);
  clReleaseMemObject(input);
  delete layer;
  freeComputeHandle(handle);
  freeComputeContext(context);

  return true;
}

//Mask should be in 'NHW' format (no "C" channel).
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}
