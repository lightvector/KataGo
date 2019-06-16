
#include "../neuralnet/nninterface.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
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

static void checkBufferSize(int batchSize, int nnXLen, int nnYLen, int channels) {
  if((int64_t)batchSize * nnXLen * nnYLen * channels >= (int64_t)1 << 31)
    throw StringError("Batch size too large, resulting GPU buffers might exceed 2^31 entries which is not currently supported");
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


//---------------------------------------------------------------------------------------------------------


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
  cl_program scaleBiasMaskNCHWProgram;
  cl_program scaleBiasMaskReluNCHWProgram;
  cl_program addPointWiseProgram;
  cl_program matMulProgram;
  cl_program sumChannelsNCHWProgram;
  cl_program gPoolChannelsNCHWProgram;
  cl_program valueHeadPoolChannelsNCHWProgram;
  cl_program addChannelBiasesNCHWProgram;
  cl_program addCBiasesNCProgram;
  cl_program addCBiasesNCReluProgram;
  cl_program transposeNCHWProgram;
  cl_program mirrorProgram;
  cl_program extractChannel0NCHWProgram;

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
    scaleBiasMaskNCHWProgram = compileProgram("scaleBiasMaskNCHWProgram", context, deviceIdsToUse, OpenCLKernels::scaleBiasMaskNCHW);
    scaleBiasMaskReluNCHWProgram = compileProgram("scaleBiasMaskReluNCHWProgram", context, deviceIdsToUse, OpenCLKernels::scaleBiasMaskReluNCHW);
    addPointWiseProgram = compileProgram("addPointWiseProgram", context, deviceIdsToUse, OpenCLKernels::addPointWise);
    matMulProgram = compileProgram("matMulProgram", context, deviceIdsToUse, OpenCLKernels::matMul);
    sumChannelsNCHWProgram = compileProgram("sumChannelsNCHWProgram", context, deviceIdsToUse, OpenCLKernels::sumChannelsNCHW);
    gPoolChannelsNCHWProgram = compileProgram("gPoolChannelsNCHWProgram", context, deviceIdsToUse, OpenCLKernels::gPoolChannelsNCHW);
    valueHeadPoolChannelsNCHWProgram = compileProgram("valueHeadPoolChannelsNCHWProgram", context, deviceIdsToUse, OpenCLKernels::valueHeadPoolChannelsNCHW);
    addChannelBiasesNCHWProgram = compileProgram("addChannelBiasesNCHWProgram", context, deviceIdsToUse, OpenCLKernels::addChannelBiasesNCHW);
    addCBiasesNCProgram = compileProgram("addCBiasesNCProgram", context, deviceIdsToUse, OpenCLKernels::addCBiasesNC);
    addCBiasesNCReluProgram = compileProgram("addCBiasesNCReluProgram", context, deviceIdsToUse, OpenCLKernels::addCBiasesNCRelu);
    transposeNCHWProgram = compileProgram("transposeNCHWProgram", context, deviceIdsToUse, OpenCLKernels::transposeNCHW);
    mirrorProgram = compileProgram("mirrorProgram", context, deviceIdsToUse, OpenCLKernels::mirror);
    extractChannel0NCHWProgram = compileProgram("extractChannel0NCHWProgram", context, deviceIdsToUse, OpenCLKernels::extractChannel0NCHW);
  }

  ~ComputeContext() {
    clReleaseProgram(conv2dNCHWProgram);
    clReleaseProgram(scaleBiasMaskNCHWProgram);
    clReleaseProgram(scaleBiasMaskReluNCHWProgram);
    clReleaseProgram(addPointWiseProgram);
    clReleaseProgram(matMulProgram);
    clReleaseProgram(sumChannelsNCHWProgram);
    clReleaseProgram(gPoolChannelsNCHWProgram);
    clReleaseProgram(valueHeadPoolChannelsNCHWProgram);
    clReleaseProgram(addChannelBiasesNCHWProgram);
    clReleaseProgram(addCBiasesNCProgram);
    clReleaseProgram(addCBiasesNCReluProgram);
    clReleaseProgram(transposeNCHWProgram);
    clReleaseProgram(mirrorProgram);
    clReleaseProgram(extractChannel0NCHWProgram);
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
  cl_kernel scaleBiasMaskNCHWKernel;
  cl_kernel scaleBiasMaskReluNCHWKernel;
  cl_kernel addPointWiseKernel;
  cl_kernel matMulKernel;
  cl_kernel sumChannelsNCHWKernel;
  cl_kernel gPoolChannelsNCHWKernel;
  cl_kernel valueHeadPoolChannelsNCHWKernel;
  cl_kernel addChannelBiasesNCHWKernel;
  cl_kernel addCBiasesNCKernel;
  cl_kernel addCBiasesNCReluKernel;
  cl_kernel transposeNCHWKernel;
  cl_kernel mirrorKernel;
  cl_kernel extractChannel0NCHWKernel;

  ComputeHandle(ComputeContext* context, const LoadedModel* loadedModel, int gpuIdx, int maxBatchSize, int nnXLen, int nnYLen) {
    clContext = context->context;
    int which = context->findWhichGpu(gpuIdx);
    commandQueue = context->commandQueues[which];

    cl_int err;
    conv2dNCHWKernel = clCreateKernel(context->conv2dNCHWProgram, "conv2dNCHW", &err);
    CHECK_ERR(err);
    scaleBiasMaskNCHWKernel = clCreateKernel(context->scaleBiasMaskNCHWProgram, "scaleBiasMaskNCHW", &err);
    CHECK_ERR(err);
    scaleBiasMaskReluNCHWKernel = clCreateKernel(context->scaleBiasMaskReluNCHWProgram, "scaleBiasMaskReluNCHW", &err);
    CHECK_ERR(err);
    addPointWiseKernel = clCreateKernel(context->addPointWiseProgram, "addPointWise", &err);
    CHECK_ERR(err);
    matMulKernel = clCreateKernel(context->matMulProgram, "matMul", &err);
    CHECK_ERR(err);
    sumChannelsNCHWKernel = clCreateKernel(context->sumChannelsNCHWProgram, "sumChannelsNCHW", &err);
    CHECK_ERR(err);
    gPoolChannelsNCHWKernel = clCreateKernel(context->gPoolChannelsNCHWProgram, "gPoolChannelsNCHW", &err);
    CHECK_ERR(err);
    valueHeadPoolChannelsNCHWKernel = clCreateKernel(context->valueHeadPoolChannelsNCHWProgram, "valueHeadPoolChannelsNCHW", &err);
    CHECK_ERR(err);
    addChannelBiasesNCHWKernel = clCreateKernel(context->addChannelBiasesNCHWProgram, "addChannelBiasesNCHW", &err);
    CHECK_ERR(err);
    addCBiasesNCKernel = clCreateKernel(context->addCBiasesNCProgram, "addCBiasesNC", &err);
    CHECK_ERR(err);
    addCBiasesNCReluKernel = clCreateKernel(context->addCBiasesNCReluProgram, "addCBiasesNCRelu", &err);
    CHECK_ERR(err);
    transposeNCHWKernel = clCreateKernel(context->transposeNCHWProgram, "transposeNCHW", &err);
    CHECK_ERR(err);
    mirrorKernel = clCreateKernel(context->mirrorProgram, "mirror", &err);
    CHECK_ERR(err);
    extractChannel0NCHWKernel = clCreateKernel(context->extractChannel0NCHWProgram, "extractChannel0NCHW", &err);
    CHECK_ERR(err);

    //TODO note that loaded model can be null, in which case we're just testing one thing
    (void)loadedModel;
    (void)maxBatchSize;
    (void)nnXLen;
    (void)nnYLen;
  }

  ~ComputeHandle() {
    clReleaseKernel(conv2dNCHWKernel);
    clReleaseKernel(scaleBiasMaskNCHWKernel);
    clReleaseKernel(scaleBiasMaskReluNCHWKernel);
    clReleaseKernel(addPointWiseKernel);
    clReleaseKernel(matMulKernel);
    clReleaseKernel(sumChannelsNCHWKernel);
    clReleaseKernel(gPoolChannelsNCHWKernel);
    clReleaseKernel(valueHeadPoolChannelsNCHWKernel);
    clReleaseKernel(addChannelBiasesNCHWKernel);
    clReleaseKernel(addCBiasesNCKernel);
    clReleaseKernel(addCBiasesNCReluKernel);
    clReleaseKernel(transposeNCHWKernel);
    clReleaseKernel(mirrorKernel);
    clReleaseKernel(extractChannel0NCHWKernel);
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

static cl_mem createReadOnlyBuffer(ComputeHandle* handle, vector<float>& data) {
  cl_int err;
  cl_mem buf = clCreateBuffer(
    handle->clContext,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    byteSizeofVectorContents(data),
    data.data(),
    &err
  );
  CHECK_ERR(err);
  return buf;
}

static cl_mem createReadWriteBuffer(ComputeHandle* handle, vector<float>& data) {
  cl_int err;
  cl_mem buf = clCreateBuffer(
    handle->clContext,
    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
    byteSizeofVectorContents(data),
    data.data(),
    &err
  );
  CHECK_ERR(err);
  return buf;
}

static cl_mem createReadWriteBuffer(ComputeHandle* handle, size_t numFloats) {
  cl_int err;
  cl_mem buf = clCreateBuffer(
    handle->clContext,
    CL_MEM_READ_WRITE,
    numFloats * sizeof(float),
    NULL,
    &err
  );
  CHECK_ERR(err);
  return buf;
}

static void addChannelBiases(ComputeHandle* handle, cl_mem src, cl_mem bias, int ncSize, int nnXYLen) {
  cl_int err;
  static constexpr int nKernelDims = 2;
  size_t globalSizes[nKernelDims] = {powerOf2ify(nnXYLen),powerOf2ify(ncSize)};
  size_t* localSizes = NULL; //TODO actually pick these

  cl_kernel kernel = handle->addChannelBiasesNCHWKernel;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&src);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bias);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&ncSize);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&nnXYLen);

  err = clEnqueueNDRangeKernel(
    handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
  );
  CHECK_ERR(err);
}

static void performGPool(ComputeHandle* handle, int batchSize, int gpoolChannels, int nnXYLen, cl_mem gpoolConvOut, cl_mem gpoolConcat, cl_mem maskSum) {
  cl_int err;
  static constexpr int nKernelDims = 3;
  //TODO optimize/tune, dehardcode numbers
  size_t globalSizes[nKernelDims] = {32,powerOf2ify(gpoolChannels),powerOf2ify(batchSize)};
  size_t localSizes[nKernelDims] = {32,std::min((size_t)8,globalSizes[1]),1};

  cl_kernel kernel = handle->gPoolChannelsNCHWKernel;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpoolConvOut);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpoolConcat);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&maskSum);
  clSetKernelArg(kernel, 3, sizeof(float) * localSizes[0] * localSizes[1] * localSizes[2], NULL);
  clSetKernelArg(kernel, 4, sizeof(float) * localSizes[0] * localSizes[1] * localSizes[2], NULL);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&gpoolChannels);
  clSetKernelArg(kernel, 7, sizeof(int), (void *)&nnXYLen);

  err = clEnqueueNDRangeKernel(
    handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
  );
  CHECK_ERR(err);
}

static void performValueHeadPool(ComputeHandle* handle, int batchSize, int gpoolChannels, int nnXYLen, cl_mem gpoolConvOut, cl_mem gpoolConcat, cl_mem maskSum) {
  cl_int err;
  static constexpr int nKernelDims = 3;
  //TODO optimize/tune, dehardcode numbers
  size_t globalSizes[nKernelDims] = {32,powerOf2ify(gpoolChannels),powerOf2ify(batchSize)};
  size_t localSizes[nKernelDims] = {32,std::min((size_t)8,globalSizes[1]),1};

  cl_kernel kernel = handle->valueHeadPoolChannelsNCHWKernel;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpoolConvOut);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpoolConcat);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&maskSum);
  clSetKernelArg(kernel, 3, sizeof(float) * localSizes[0] * localSizes[1] * localSizes[2], NULL);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&gpoolChannels);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&nnXYLen);

  err = clEnqueueNDRangeKernel(
    handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
  );
  CHECK_ERR(err);
}

static void transposeNCHW(ComputeHandle* handle, int batchSize, int cSize, int nnXLen, int nnYLen, cl_mem input, cl_mem output) {
  cl_int err;
  static constexpr int nKernelDims = 3;
  //TODO optimize/tune, dehardcode numbers
  int tileDim = 32;
  int targetNumThreads = 256;
  int tileStride = targetNumThreads/tileDim;
  int ncLen = batchSize*cSize;
  size_t globalSizes[nKernelDims] = {(size_t)(nnXLen+tileDim-1)/tileDim*tileDim,(size_t)(nnYLen+tileDim-1)/tileDim*tileDim,powerOf2ify(ncLen)};
  size_t localSizes[nKernelDims] = {(size_t)tileDim,std::min((size_t)8,globalSizes[1]),1};

  cl_kernel kernel = handle->transposeNCHWKernel;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output);
  clSetKernelArg(kernel, 2, sizeof(float) * tileDim * (tileDim+1) * localSizes[2], NULL);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&nnXLen);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&nnYLen);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&tileDim);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&tileStride);
  clSetKernelArg(kernel, 7, sizeof(int), (void *)&ncLen);

  err = clEnqueueNDRangeKernel(
    handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
  );
  CHECK_ERR(err);
}

static void doMirror(ComputeHandle* handle, int batchSize, int mSize, int subSize, cl_mem input, cl_mem output) {
  cl_int err;
  static constexpr int nKernelDims = 3;
  size_t globalSizes[nKernelDims] = {powerOf2ify(subSize),powerOf2ify(mSize),powerOf2ify(batchSize)};
  size_t* localSizes = NULL; //TODO actually pick these

  cl_kernel kernel = handle->mirrorKernel;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&mSize);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&subSize);

  err = clEnqueueNDRangeKernel(
    handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
  );
  CHECK_ERR(err);
}

static void doMirrorNCHW(ComputeHandle* handle, int batchSize, int cSize, int nnXLen, int nnYLen, bool mirrorY, bool mirrorX, cl_mem input, cl_mem output) {
  if(mirrorY && mirrorX)
    doMirror(handle,batchSize*cSize,nnYLen*nnXLen,1,input,output);
  else if(mirrorY)
    doMirror(handle,batchSize*cSize,nnYLen,nnXLen,input,output);
  else if(mirrorX)
    doMirror(handle,batchSize*cSize*nnYLen,nnXLen,1,input,output);
  else {
    cl_int err;
    err = clEnqueueCopyBuffer(handle->commandQueue, input, output, 0, 0, sizeof(float)*batchSize*cSize*nnYLen*nnXLen, 0, NULL, NULL);
    CHECK_ERR(err);
  }
}

static void applySymmetriesNCHW(
  ComputeHandle* handle,
  const bool* symmetriesBuffer, bool inverse, int batchSize, int cSize, int nnXLen, int nnYLen,
  cl_mem input, cl_mem inputScratch
) {
  if(!symmetriesBuffer[0] && !symmetriesBuffer[1] && !symmetriesBuffer[2])
    return;

  cl_int err;
  if(inverse) {
    if(symmetriesBuffer[2] && nnXLen == nnYLen)
      transposeNCHW(handle, batchSize, cSize, nnXLen, nnYLen, input, inputScratch);
    else {
      err = clEnqueueCopyBuffer(handle->commandQueue, input, inputScratch, 0, 0, sizeof(float)*batchSize*cSize*nnYLen*nnXLen, 0, NULL, NULL);
      CHECK_ERR(err);
    }
    doMirrorNCHW(handle, batchSize, cSize, nnYLen, nnXLen, symmetriesBuffer[0], symmetriesBuffer[1], inputScratch, input);
  }
  else {
    doMirrorNCHW(handle, batchSize, cSize, nnYLen, nnXLen, symmetriesBuffer[0], symmetriesBuffer[1], input, inputScratch);

    if(symmetriesBuffer[2] && nnXLen == nnYLen)
      transposeNCHW(handle, batchSize, cSize, nnXLen, nnYLen, inputScratch, input);
    else {
      err = clEnqueueCopyBuffer(handle->commandQueue, inputScratch, input, 0, 0, sizeof(float)*batchSize*cSize*nnYLen*nnXLen, 0, NULL, NULL);
      CHECK_ERR(err);
    }
  }
}


//--------------------------------------------------------------

struct ConvLayer {
  string name;
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
    name = desc->name;
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

    vector<float> weights = desc->weights;
    filter = createReadOnlyBuffer(handle,weights);

    globalSizes[0] = powerOf2ify(nnXLen);
    globalSizes[1] = powerOf2ify(nnYLen);
    globalSizes[2] = powerOf2ify(outChannels);
  }

  ~ConvLayer() {
    clReleaseMemObject(filter);
  }

  void apply(ComputeHandle* handle, int batchSize, cl_mem input, cl_mem output) {
    cl_kernel kernel = handle->conv2dNCHWKernel;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&batchSize);
    clSetKernelArg(kernel, 4, sizeof(int), (void *)&nnXLen);
    clSetKernelArg(kernel, 5, sizeof(int), (void *)&nnYLen);
    clSetKernelArg(kernel, 6, sizeof(int), (void *)&outChannels);
    clSetKernelArg(kernel, 7, sizeof(int), (void *)&inChannels);
    clSetKernelArg(kernel, 8, sizeof(int), (void *)&convXRadius);
    clSetKernelArg(kernel, 9, sizeof(int), (void *)&convYRadius);

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

struct BatchNormLayer {
  string name;
  int numChannels;
  float epsilon;

  int nnXLen;
  int nnYLen;
  int nnXYLen;
  cl_mem mergedScaleBuf;
  cl_mem mergedBiasBuf;

  static constexpr int nKernelDims = 2;
  size_t globalSizes[nKernelDims];

  BatchNormLayer(ComputeHandle* handle, const BatchNormLayerDesc* desc, int nnX, int nnY) {
    name = desc->name;
    numChannels = desc->numChannels;
    epsilon = desc->epsilon;

    nnXLen = nnX;
    nnYLen = nnY;
    nnXYLen = nnX * nnY;

    assert(desc->mean.size() == numChannels);
    assert(desc->variance.size() == numChannels);
    assert(desc->scale.size() == numChannels);
    assert(desc->bias.size() == numChannels);

    vector<float> mergedScale(numChannels);
    vector<float> mergedBias(numChannels);
    for(int i = 0; i<numChannels; i++) {
      mergedScale[i] = desc->scale[i] / sqrt(desc->variance[i] + epsilon);
      mergedBias[i] = desc->bias[i] - mergedScale[i] * desc->mean[i];
    }

    mergedScaleBuf = createReadOnlyBuffer(handle,mergedScale);
    mergedBiasBuf = createReadOnlyBuffer(handle,mergedBias);

    globalSizes[0] = powerOf2ify(nnXLen * nnYLen);
    globalSizes[1] = powerOf2ify(numChannels);
  }

  ~BatchNormLayer() {
    clReleaseMemObject(mergedScaleBuf);
    clReleaseMemObject(mergedBiasBuf);
  }

  void apply(ComputeHandle* handle, int batchSize, bool applyRelu, cl_mem input, cl_mem output, cl_mem mask) {
    cl_kernel kernel;
    if(!applyRelu)
      kernel = handle->scaleBiasMaskNCHWKernel;
    else
      kernel = handle->scaleBiasMaskReluNCHWKernel;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&mergedScaleBuf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&mergedBiasBuf);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&mask);
    clSetKernelArg(kernel, 5, sizeof(int), (void *)&batchSize);
    clSetKernelArg(kernel, 6, sizeof(int), (void *)&numChannels);
    clSetKernelArg(kernel, 7, sizeof(int), (void *)&nnXYLen);

    cl_int err;
    size_t* localSizes = NULL; //TODO actually pick these
    err = clEnqueueNDRangeKernel(
      handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
    );
    CHECK_ERR(err);
  }

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;
};

//--------------------------------------------------------------

struct ActivationLayer {
  string name;

  ActivationLayer(
    ComputeHandle* handle, const ActivationLayerDesc* desc
  ) {
    (void)handle;
    name = desc->name;
  }

  ~ActivationLayer() {
  }

  ActivationLayer() = delete;
  ActivationLayer(const ActivationLayer&) = delete;
  ActivationLayer& operator=(const ActivationLayer&) = delete;
};

//--------------------------------------------------------------

struct MatMulLayer {
  string name;
  int inChannels;
  int outChannels;

  cl_mem matBuf;

  MatMulLayer(ComputeHandle* handle, const MatMulLayerDesc* desc) {
    name = desc->name;
    inChannels = desc->inChannels;
    outChannels = desc->outChannels;

    assert(desc->weights.size() == inChannels * outChannels);
    vector<float> weights = desc->weights;
    matBuf = createReadOnlyBuffer(handle,weights);
  }

  ~MatMulLayer() {
    clReleaseMemObject(matBuf);
  }

  void apply(ComputeHandle* handle, int batchSize, cl_mem input, cl_mem output) {
    cl_kernel kernel = handle->matMulKernel;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&matBuf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&output);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&batchSize);
    clSetKernelArg(kernel, 4, sizeof(int), (void *)&inChannels);
    clSetKernelArg(kernel, 5, sizeof(int), (void *)&outChannels);

    cl_int err;
    static constexpr int nKernelDims = 2;
    size_t globalSizes[nKernelDims] = {powerOf2ify((size_t)outChannels), powerOf2ify((size_t)batchSize)};
    size_t* localSizes = NULL; //TODO actually pick these
    err = clEnqueueNDRangeKernel(
      handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
    );
    CHECK_ERR(err);
  }

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;
};

//--------------------------------------------------------------

struct MatBiasLayer {
  string name;
  int numChannels;

  cl_mem biasBuf;

  MatBiasLayer(ComputeHandle* handle, const MatBiasLayerDesc* desc) {
    name = desc->name;
    numChannels = desc->numChannels;

    assert(desc->weights.size() == numChannels);
    vector<float> weights = desc->weights;
    biasBuf = createReadOnlyBuffer(handle,weights);
  }

  ~MatBiasLayer() {
    clReleaseMemObject(biasBuf);
  }

  void apply(ComputeHandle* handle, int batchSize, bool applyRelu, cl_mem input) {
    cl_kernel kernel = applyRelu ? handle->addCBiasesNCReluKernel : handle->addCBiasesNCKernel;

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&biasBuf);
    clSetKernelArg(kernel, 2, sizeof(int), (void *)&batchSize);
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&numChannels);

    cl_int err;
    static constexpr int nKernelDims = 2;
    size_t globalSizes[nKernelDims] = {powerOf2ify((size_t)numChannels), powerOf2ify((size_t)batchSize)};
    size_t* localSizes = NULL; //TODO actually pick these
    err = clEnqueueNDRangeKernel(
      handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
    );
    CHECK_ERR(err);
  }

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;
};


//--------------------------------------------------------------

struct ResidualBlock {
  string name;
  BatchNormLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  BatchNormLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  int nnXLen;
  int nnYLen;
  int regularChannels;

  ResidualBlock(
    ComputeHandle* handle,
    const ResidualBlockDesc* desc,
    int nnX, int nnY
  ): name(desc->name),
     preBN(handle,&desc->preBN,nnX,nnY),
     preActivation(handle,&desc->preActivation),
     regularConv(handle,&desc->regularConv,nnX,nnY),
     midBN(handle,&desc->midBN,nnX,nnY),
     midActivation(handle,&desc->midActivation),
     finalConv(handle,&desc->finalConv,nnX,nnY),
     nnXLen(nnX),
     nnYLen(nnY),
     regularChannels(desc->regularConv.outChannels)
  {
  }

  ~ResidualBlock() {
  }

  void apply(
    ComputeHandle* handle,
    int batchSize,
    cl_mem trunk,
    cl_mem trunkScratch,
    cl_mem mid,
    cl_mem midScratch,
    cl_mem mask
  ) {
    preBN.apply(handle,batchSize,true,trunk,trunkScratch,mask);
    regularConv.apply(handle,batchSize,trunkScratch,mid);
    midBN.apply(handle,batchSize,true,mid,midScratch,mask);
    finalConv.apply(handle,batchSize,midScratch,trunkScratch);

    cl_kernel kernel = handle->addPointWiseKernel;
    int totalSize = batchSize * finalConv.outChannels * nnYLen * nnXLen;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&trunk);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&trunkScratch);
    clSetKernelArg(kernel, 2, sizeof(int), (void *)&totalSize);

    cl_int err;
    static constexpr int nKernelDims = 1;
    size_t globalSizes[nKernelDims] = {powerOf2ify((size_t)totalSize)};
    size_t* localSizes = NULL; //TODO actually pick these
    err = clEnqueueNDRangeKernel(
      handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
    );
    CHECK_ERR(err);
  }

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

};

//--------------------------------------------------------------

struct GlobalPoolingResidualBlock {
  string name;
  BatchNormLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  ConvLayer gpoolConv;
  BatchNormLayer gpoolBN;
  ActivationLayer gpoolActivation;
  MatMulLayer gpoolToBiasMul;
  BatchNormLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  int nnXLen;
  int nnYLen;
  int nnXYLen;
  int regularChannels;
  int gpoolChannels;

  GlobalPoolingResidualBlock(
    ComputeHandle* handle,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX, int nnY
  ): name(desc->name),
     preBN(handle,&desc->preBN,nnX,nnY),
     preActivation(handle,&desc->preActivation),
     regularConv(handle,&desc->regularConv,nnX,nnY),
     gpoolConv(handle,&desc->gpoolConv,nnX,nnY),
     gpoolBN(handle,&desc->gpoolBN,nnX,nnY),
     gpoolActivation(handle,&desc->gpoolActivation),
     gpoolToBiasMul(handle,&desc->gpoolToBiasMul),
     midBN(handle,&desc->midBN,nnX,nnY),
     midActivation(handle,&desc->midActivation),
     finalConv(handle,&desc->finalConv,nnX,nnY),
     nnXLen(nnX),
     nnYLen(nnY),
     nnXYLen(nnX*nnY),
     regularChannels(desc->regularConv.outChannels),
     gpoolChannels(desc->gpoolConv.outChannels)
  {
  }

  ~GlobalPoolingResidualBlock() {
  }

  void apply(
    ComputeHandle* handle,
    int batchSize,
    cl_mem trunk,
    cl_mem trunkScratch,
    cl_mem mid,
    cl_mem midScratch,
    cl_mem gpoolOut,
    cl_mem gpoolOut2,
    cl_mem gpoolConcat,
    cl_mem gpoolBias,
    cl_mem mask,
    cl_mem maskSum
  ) {
    preBN.apply(handle,batchSize,true,trunk,trunkScratch,mask);
    regularConv.apply(handle,batchSize,trunkScratch,mid);
    gpoolConv.apply(handle,batchSize,trunkScratch,gpoolOut);
    gpoolBN.apply(handle,batchSize,true,gpoolOut,gpoolOut2,mask);

    performGPool(handle, batchSize, gpoolChannels, nnXYLen, gpoolOut2, gpoolConcat, maskSum);

    gpoolToBiasMul.apply(handle,batchSize,gpoolConcat,gpoolBias);
    addChannelBiases(handle, mid, gpoolBias, batchSize * regularChannels, nnXYLen);

    // vector<float> tmp(batchSize*regularChannels);
    // clEnqueueReadBuffer(handle->commandQueue, gpoolBias, CL_TRUE, 0, byteSizeofVectorContents(tmp), tmp.data(), 0, NULL, NULL);
    // cout << "TEST" << endl;
    // for(int i = 0; i<tmp.size(); i++)
    //   cout << tmp[i] << endl;

    midBN.apply(handle,batchSize,true,mid,midScratch,mask);
    finalConv.apply(handle,batchSize,midScratch,trunkScratch);

    {
      cl_kernel kernel = handle->addPointWiseKernel;
      int totalSize = batchSize * finalConv.outChannels * nnYLen * nnXLen;
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&trunk);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&trunkScratch);
      clSetKernelArg(kernel, 2, sizeof(int), (void *)&totalSize);

      cl_int err;
      static constexpr int nKernelDims = 1;
      size_t globalSizes[nKernelDims] = {powerOf2ify((size_t)totalSize)};
      size_t* localSizes = NULL; //TODO actually pick these
      err = clEnqueueNDRangeKernel(
        handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
      );
      CHECK_ERR(err);
    }
  }

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

};

//--------------------------------------------------------------

struct Trunk {
  string name;
  int version;
  int numBlocks;
  int trunkNumChannels;
  int midNumChannels;
  int regularNumChannels;
  int dilatedNumChannels;
  int gpoolNumChannels;

  int maxBatchSize;
  int nnXLen;
  int nnYLen;

  ConvLayer* initialConv;
  MatMulLayer* initialMatMul;
  vector<pair<int,void*>> blocks;
  BatchNormLayer* trunkTipBN;
  ActivationLayer* trunkTipActivation;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    ComputeHandle* handle,
    const TrunkDesc* desc,
    int maxBatchSz,
    int nnX,
    int nnY
  ) {
    name = desc->name;
    version = desc->version;
    numBlocks = desc->numBlocks;
    trunkNumChannels = desc->trunkNumChannels;
    midNumChannels = desc->midNumChannels;
    regularNumChannels = desc->regularNumChannels;
    dilatedNumChannels = desc->dilatedNumChannels;
    gpoolNumChannels = desc->gpoolNumChannels;

    maxBatchSize = maxBatchSz;
    nnXLen = nnX;
    nnYLen = nnY;

    checkBufferSize(maxBatchSize,nnXLen,nnYLen,trunkNumChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,midNumChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,regularNumChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,dilatedNumChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,gpoolNumChannels);

    initialConv = new ConvLayer(handle,&desc->initialConv,nnXLen,nnYLen);
    initialMatMul = new MatMulLayer(handle,&desc->initialMatMul);

    trunkTipBN = new BatchNormLayer(handle,&desc->trunkTipBN,nnXLen,nnYLen);
    trunkTipActivation = new ActivationLayer(handle,&desc->trunkTipActivation);

    assert(desc->blocks.size() == numBlocks);
    for(int i = 0; i<numBlocks; i++) {
      if(desc->blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)desc->blocks[i].second;
        ResidualBlock* block = new ResidualBlock(
          handle,
          blockDesc,
          nnXLen,
          nnYLen
        );
        blocks.push_back(make_pair(ORDINARY_BLOCK_KIND,(void*)block));
      }
      else if(desc->blocks[i].first == DILATED_BLOCK_KIND) {
        throw StringError("Neural net use dilated convolutions but OpenCL implementation dues not currently support them");
      }
      else if(desc->blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)desc->blocks[i].second;
        GlobalPoolingResidualBlock* block = new GlobalPoolingResidualBlock(
          handle,
          blockDesc,
          nnXLen,
          nnYLen
        );
        blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND,(void*)block));
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
  }

  ~Trunk()
  {
    for(int i = 0; i<blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlock* block = (ResidualBlock*)blocks[i].second;
        delete block;
      }
      else if(blocks[i].first == DILATED_BLOCK_KIND) {
        ASSERT_UNREACHABLE;
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second;
        delete block;
      }
    }

    delete initialConv;
    delete initialMatMul;
    delete trunkTipBN;
    delete trunkTipActivation;
  }

  void apply(
    ComputeHandle* handle,
    int batchSize,
    cl_mem input,
    cl_mem inputGlobal,
    cl_mem trunk,
    cl_mem trunkScratch,
    cl_mem mid,
    cl_mem midScratch,
    cl_mem gpoolOut,
    cl_mem gpoolOut2,
    cl_mem gpoolConcat,
    cl_mem gpoolBias,
    cl_mem mask,
    cl_mem maskSum
  ) const {

    //Feed the conv into trunkScratch, not trunk
    initialConv->apply(handle,batchSize,input,trunkScratch);

    if(initialMatMul != NULL) {
      //Feed the matmul into trunk
      initialMatMul->apply(handle,batchSize,inputGlobal,trunk);
      //Then accumulate it into trunkScratch, broadcasting during the process
      addChannelBiases(handle, trunkScratch, trunk, batchSize * trunkNumChannels, nnXLen*nnYLen);
    }

    for(int i = 0; i<blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlock* block = (ResidualBlock*)blocks[i].second;
        block->apply(
          handle,
          batchSize,
          trunkScratch, //Flip trunk and trunkScratch so that the result gets accumulated in trunkScratch
          trunk,
          mid,
          midScratch,
          mask
        );
      }
      else if(blocks[i].first == DILATED_BLOCK_KIND) {
        ASSERT_UNREACHABLE;
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second;
        block->apply(
          handle,
          batchSize,
          trunkScratch, //Flip trunk and trunkScratch so that the result gets accumulated in trunkScratch
          trunk,
          mid,
          midScratch,
          gpoolOut,
          gpoolOut2,
          gpoolConcat,
          gpoolBias,
          mask,
          maskSum
        );
      }
      else {
        ASSERT_UNREACHABLE;
      }

    }

    //And now with the final BN port it from trunkScratch to trunk.
    bool applyBNRelu = true;
    trunkTipBN->apply(handle,batchSize,applyBNRelu,trunkScratch,trunk,mask);
  }

};

//--------------------------------------------------------------

struct PolicyHead {
  string name;
  int version;
  int nnXLen;
  int nnYLen;
  int p1Channels;
  int g1Channels;
  int p2Channels;

  ConvLayer* p1Conv;
  ConvLayer* g1Conv;
  BatchNormLayer* g1BN;
  ActivationLayer* g1Activation;
  MatMulLayer* gpoolToBiasMul;
  BatchNormLayer* p1BN;
  ActivationLayer* p1Activation;
  ConvLayer* p2Conv;
  MatMulLayer* gpoolToPassMul;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    ComputeHandle* handle,
    const PolicyHeadDesc* desc,
    int nnX,
    int nnY
  ) {
    name = desc->name;
    version = desc->version;
    nnXLen = nnX;
    nnYLen = nnY;
    p1Channels = desc->p1Conv.outChannels;
    g1Channels = desc->g1Conv.outChannels;
    p2Channels = desc->p2Conv.outChannels;

    p1Conv = new ConvLayer(handle,&desc->p1Conv,nnXLen,nnYLen);
    g1Conv = new ConvLayer(handle,&desc->g1Conv,nnXLen,nnYLen);
    g1BN = new BatchNormLayer(handle,&desc->g1BN,nnXLen,nnYLen);
    g1Activation = new ActivationLayer(handle,&desc->g1Activation);
    gpoolToBiasMul = new MatMulLayer(handle,&desc->gpoolToBiasMul);
    p1BN = new BatchNormLayer(handle,&desc->p1BN,nnXLen,nnYLen);
    p1Activation = new ActivationLayer(handle,&desc->p1Activation);
    p2Conv = new ConvLayer(handle,&desc->p2Conv,nnXLen,nnYLen);
    gpoolToPassMul = new MatMulLayer(handle,&desc->gpoolToPassMul);
  }

  ~PolicyHead()
  {
    delete p1Conv;
    delete g1Conv;
    delete g1BN;
    delete g1Activation;
    delete gpoolToBiasMul;
    delete p1BN;
    delete p1Activation;
    delete p2Conv;
    delete gpoolToPassMul;
  }

  void apply(
    ComputeHandle* handle,
    const bool* symmetriesBuffer,
    int batchSize,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem trunk,
    cl_mem p1Out,
    cl_mem p1Out2,
    cl_mem gpoolOut,
    cl_mem gpoolOut2,
    cl_mem gpoolConcat,
    cl_mem gpoolBias,
    cl_mem p2Out,
    cl_mem policyPass,
    cl_mem policy
  ) const {

    bool applyBNRelu = true;
    p1Conv->apply(handle,batchSize,trunk,p1Out);
    g1Conv->apply(handle,batchSize,trunk,gpoolOut);
    g1BN->apply(handle,batchSize,applyBNRelu,gpoolOut,gpoolOut2,mask);

    performGPool(handle, batchSize, g1Channels, nnXLen*nnYLen, gpoolOut2, gpoolConcat, maskSum);

    gpoolToBiasMul->apply(handle,batchSize,gpoolConcat,gpoolBias);

    cl_mem p1OutA;
    cl_mem p1OutB;
    p1OutA = p1Out;
    p1OutB = p1Out2;

    addChannelBiases(handle, p1OutA, gpoolBias, batchSize * p1Channels, nnXLen*nnYLen);

    p1BN->apply(handle,batchSize,true,p1OutA,p1OutB,mask);
    p2Conv->apply(handle,batchSize,p1OutB,p2Out);

    bool inverse = true;
    applySymmetriesNCHW(handle, symmetriesBuffer, inverse, batchSize, p2Channels, nnXLen, nnYLen, p2Out, policy);

    gpoolToPassMul->apply(handle,batchSize,gpoolConcat,policyPass);
  }

};

//--------------------------------------------------------------

struct ValueHead {
  string name;
  int version;
  int nnXLen;
  int nnYLen;
  int v1Channels;
  int v2Channels;
  int valueChannels;
  int scoreValueChannels;
  int ownershipChannels;

  ConvLayer* v1Conv;
  BatchNormLayer* v1BN;
  ActivationLayer* v1Activation;
  MatMulLayer* v2Mul;
  MatBiasLayer* v2Bias;
  ActivationLayer* v2Activation;
  MatMulLayer* v3Mul;
  MatBiasLayer* v3Bias;
  MatMulLayer* sv3Mul;
  MatBiasLayer* sv3Bias;
  ConvLayer* vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    ComputeHandle* handle,
    const ValueHeadDesc* desc,
    int nnX,
    int nnY
  ) {
    name = desc->name;
    version = desc->version;
    nnXLen = nnX;
    nnYLen = nnY;
    v1Channels = desc->v1Conv.outChannels;
    v2Channels = desc->v2Mul.outChannels;
    valueChannels = desc->v3Mul.outChannels;
    scoreValueChannels = desc->sv3Mul.outChannels;
    ownershipChannels = desc->vOwnershipConv.outChannels;

    v1Conv = new ConvLayer(handle,&desc->v1Conv,nnXLen,nnYLen);
    v1BN = new BatchNormLayer(handle,&desc->v1BN,nnXLen,nnYLen);
    v1Activation = new ActivationLayer(handle,&desc->v1Activation);
    v2Mul = new MatMulLayer(handle,&desc->v2Mul);
    v2Bias = new MatBiasLayer(handle,&desc->v2Bias);
    v2Activation = new ActivationLayer(handle,&desc->v2Activation);
    v3Mul = new MatMulLayer(handle,&desc->v3Mul);
    v3Bias = new MatBiasLayer(handle,&desc->v3Bias);
    sv3Mul = new MatMulLayer(handle,&desc->sv3Mul);
    sv3Bias = new MatBiasLayer(handle,&desc->sv3Bias);
    vOwnershipConv = new ConvLayer(handle,&desc->vOwnershipConv,nnXLen,nnYLen);
  }

  ~ValueHead()
  {
    delete v1Conv;
    delete v1BN;
    delete v1Activation;
    delete v2Mul;
    delete v2Bias;
    delete v2Activation;
    delete v3Mul;
    delete v3Bias;
    delete sv3Mul;
    delete sv3Bias;
    delete vOwnershipConv;
  }


  void apply(
    ComputeHandle* handle,
    const bool* symmetriesBuffer,
    int batchSize,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem trunk,
    cl_mem v1Out,
    cl_mem v1Out2,
    cl_mem v1Mean,
    cl_mem v2Out,
    cl_mem value,
    cl_mem scoreValue,
    cl_mem ownership,
    cl_mem ownershipScratch
  ) const {

    bool applyBNRelu = true;
    v1Conv->apply(handle,batchSize,trunk,v1Out);
    v1BN->apply(handle,batchSize,applyBNRelu,v1Out,v1Out2,mask);

    performValueHeadPool(handle, batchSize, v1Channels, nnXLen*nnYLen, v1Out2, v1Mean, maskSum);

    v2Mul->apply(handle,batchSize,v1Mean,v2Out);
    v2Bias->apply(handle,batchSize,true,v2Out);
    v3Mul->apply(handle,batchSize,v2Out,value);
    v3Bias->apply(handle,batchSize,false,value);

    sv3Mul->apply(handle,batchSize,v2Out,scoreValue);
    sv3Bias->apply(handle,batchSize,false,scoreValue);

    vOwnershipConv->apply(handle,batchSize,v1Out2,ownership);

    bool inverse = true;
    applySymmetriesNCHW(handle, symmetriesBuffer, inverse, batchSize, ownershipChannels, nnXLen, nnYLen, ownership, ownershipScratch);
  }

};

//--------------------------------------------------------------

static void computeMaskSums(
  ComputeHandle* handle,
  cl_mem mask,
  cl_mem maskSum,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  cl_int err;
  static constexpr int nKernelDims = 3;
  //TODO optimize/tune, dehardcode numbers
  size_t globalSizes[nKernelDims] = {32,1,powerOf2ify(batchSize)};
  size_t localSizes[nKernelDims] = {32,1,std::min((size_t)8,powerOf2ify(batchSize))};

  cl_kernel kernel = handle->sumChannelsNCHWKernel;
  int numChannels = 1;
  int nnXYLen = nnXLen * nnYLen;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mask);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&maskSum);
  clSetKernelArg(kernel, 2, sizeof(float) * localSizes[0] * localSizes[1] * localSizes[2], NULL);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&numChannels);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&nnXYLen);

  err = clEnqueueNDRangeKernel(
    handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
  );
  CHECK_ERR(err);
}


//--------------------------------------------------------------

struct Model {
  string name;
  int version;
  int maxBatchSize;
  int nnXLen;
  int nnYLen;
  int numInputChannels;
  int numInputGlobalChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  Trunk* trunk;
  PolicyHead* policyHead;
  ValueHead* valueHead;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    ComputeHandle* handle,
    const ModelDesc* desc,
    int maxBatchSz,
    int nnX,
    int nnY
  ) {
    name = desc->name;
    version = desc->version;
    maxBatchSize = maxBatchSz;

    nnXLen = nnX;
    nnYLen = nnY;
    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnXLen, NNPos::MAX_BOARD_LEN
      ));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnYLen, NNPos::MAX_BOARD_LEN
      ));

    numInputChannels = desc->numInputChannels;
    numInputGlobalChannels = desc->numInputGlobalChannels;
    numValueChannels = desc->numValueChannels;
    numScoreValueChannels = desc->numScoreValueChannels;
    numOwnershipChannels = desc->numOwnershipChannels;

    int numFeatures = NNModelVersion::getNumSpatialFeatures(version);
    if(numInputChannels != numFeatures)
      throw StringError(Global::strprintf("Neural net numInputChannels (%d) was not the expected number based on version (%d)",
        numInputChannels, numFeatures
      ));
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
    if(numInputGlobalChannels != numGlobalFeatures)
      throw StringError(Global::strprintf("Neural net numInputGlobalChannels (%d) was not the expected number based on version (%d)",
        numInputGlobalChannels, numGlobalFeatures
      ));

    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputGlobalChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numValueChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numScoreValueChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numOwnershipChannels);

    trunk = new Trunk(handle,&desc->trunk,maxBatchSize,nnXLen,nnYLen);
    policyHead = new PolicyHead(handle,&desc->policyHead,nnXLen,nnYLen);
    valueHead = new ValueHead(handle,&desc->valueHead,nnXLen,nnYLen);
  }

  ~Model()
  {
    delete valueHead;
    delete policyHead;
    delete trunk;
  }

  void apply(
    ComputeHandle* handle,
    int batchSize,
    bool* symmetriesBuffer,

    cl_mem input,
    cl_mem inputScratch,
    cl_mem inputGlobal,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem trunkBuf,
    cl_mem trunkScratch,
    cl_mem mid,
    cl_mem midScratch,
    cl_mem gpoolOut,
    cl_mem gpoolOut2,
    cl_mem gpoolConcat,
    cl_mem gpoolBias,

    cl_mem p1Out,
    cl_mem p1Out2,
    cl_mem p2Out,
    cl_mem policyPass,
    cl_mem policy,

    cl_mem v1Out,
    cl_mem v1Out2,
    cl_mem v1Mean,
    cl_mem v2Out,
    cl_mem value,
    cl_mem scoreValue,
    cl_mem ownership,
    cl_mem ownershipScratch
  ) {

    bool inverse = false;
    applySymmetriesNCHW(handle, symmetriesBuffer, inverse, batchSize, numInputChannels, nnXLen, nnYLen, input, inputScratch);

    {
      cl_kernel kernel = handle->extractChannel0NCHWKernel;
      int nnXYLen = nnXLen * nnYLen;
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&mask);
      clSetKernelArg(kernel, 2, sizeof(int), (void *)&batchSize);
      clSetKernelArg(kernel, 3, sizeof(int), (void *)&numInputChannels);
      clSetKernelArg(kernel, 4, sizeof(int), (void *)&nnXYLen);

      cl_int err;
      static constexpr int nKernelDims = 2;
      size_t globalSizes[nKernelDims] = {powerOf2ify((size_t)nnXYLen), powerOf2ify((size_t)batchSize)};
      size_t* localSizes = NULL; //TODO actually pick these
      err = clEnqueueNDRangeKernel(
        handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, NULL
      );
      CHECK_ERR(err);
    }

    computeMaskSums(handle,mask,maskSum,batchSize,nnXLen,nnYLen);

    trunk->apply(
      handle,
      batchSize,
      input,
      inputGlobal,
      trunkBuf,
      trunkScratch,
      mid,
      midScratch,
      gpoolOut,
      gpoolOut2,
      gpoolConcat,
      gpoolBias,
      mask,
      maskSum
    );
    policyHead->apply(
      handle,
      symmetriesBuffer,
      batchSize,
      mask,
      maskSum,
      trunkBuf,
      p1Out,
      p1Out2,
      gpoolOut,
      gpoolOut2,
      gpoolConcat,
      gpoolBias,
      p2Out,
      policyPass,
      policy
    );
    valueHead->apply(
      handle,
      symmetriesBuffer,
      batchSize,
      mask,
      maskSum,
      trunkBuf,
      v1Out,
      v1Out2,
      v1Mean,
      v2Out,
      value,
      scoreValue,
      ownership,
      ownershipScratch
    );
  }

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

  if(useFP16 != false)
    return false;
  if(useNHWC != false)
    return false;

  LoadedModel* loadedModel = NULL;
  bool requireExactNNLen = false;
  bool inputsUseNHWC = useNHWC;
  ComputeContext* context = createComputeContext({gpuIdx}, logger);
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
  cl_mem input = createReadOnlyBuffer(handle,inputTmp);

  cl_mem output = clCreateBuffer(handle->clContext, CL_MEM_WRITE_ONLY, byteSizeofVectorContents(outputBuffer), NULL, &err);
  CHECK_ERR(err);
  layer->apply(handle, batchSize, input, output);

  cl_bool blocking = CL_TRUE;
  err = clEnqueueReadBuffer(handle->commandQueue, output, blocking, 0, byteSizeofVectorContents(outputBuffer), outputBuffer.data(), 0, NULL, NULL);
  CHECK_ERR(err);

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
  Logger* logger = NULL;
  cl_int err;
  int gpuIdx = 0;

  if(useFP16 != false)
    return false;
  if(useNHWC != false)
    return false;

  LoadedModel* loadedModel = NULL;
  bool requireExactNNLen = false;
  bool inputsUseNHWC = useNHWC;
  ComputeContext* context = createComputeContext({gpuIdx}, logger);
  ComputeHandle* handle = createComputeHandle(
    context, loadedModel, logger, batchSize, nnXLen, nnYLen, requireExactNNLen, inputsUseNHWC, gpuIdx, useFP16, useNHWC
  );

  BatchNormLayer* layer = new BatchNormLayer(handle, desc, nnXLen, nnYLen);

  size_t numInputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;
  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected input buffer size");
  outputBuffer.resize(numOutputFloats);

  vector<float> inputTmp = inputBuffer;
  vector<float> maskTmp = maskBuffer;
  cl_mem input = createReadOnlyBuffer(handle,inputTmp);
  cl_mem mask = createReadOnlyBuffer(handle,maskTmp);

  cl_mem output = clCreateBuffer(handle->clContext, CL_MEM_WRITE_ONLY, byteSizeofVectorContents(outputBuffer), NULL, &err);
  CHECK_ERR(err);
  bool applyRelu = false;
  layer->apply(handle, batchSize, applyRelu, input, output, mask);

  cl_bool blocking = CL_TRUE;
  err = clEnqueueReadBuffer(handle->commandQueue, output, blocking, 0, byteSizeofVectorContents(outputBuffer), outputBuffer.data(), 0, NULL, NULL);
  CHECK_ERR(err);

  clReleaseMemObject(input);
  clReleaseMemObject(mask);
  clReleaseMemObject(output);
  delete layer;
  freeComputeHandle(handle);
  freeComputeContext(context);

  return true;
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
  Logger* logger = NULL;
  cl_int err;
  int gpuIdx = 0;

  if(useFP16 != false)
    return false;
  if(useNHWC != false)
    return false;

  LoadedModel* loadedModel = NULL;
  bool requireExactNNLen = false;
  bool inputsUseNHWC = useNHWC;
  ComputeContext* context = createComputeContext({gpuIdx}, logger);
  ComputeHandle* handle = createComputeHandle(
    context, loadedModel, logger, batchSize, nnXLen, nnYLen, requireExactNNLen, inputsUseNHWC, gpuIdx, useFP16, useNHWC
  );

  ResidualBlock* layer = new ResidualBlock(handle, desc, nnXLen, nnYLen);

  size_t numTrunkFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)batchSize * nnXLen * nnYLen;
  size_t numMidFloats = (size_t)batchSize * nnXLen * nnYLen * desc->finalConv.inChannels;
  if(numTrunkFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");
  outputBuffer.resize(numTrunkFloats);

  vector<float> inputTmp = inputBuffer;
  vector<float> maskTmp = maskBuffer;
  cl_mem trunk = createReadWriteBuffer(handle,inputTmp);
  cl_mem mask = createReadOnlyBuffer(handle,maskTmp);
  cl_mem trunkScratch = createReadWriteBuffer(handle,numTrunkFloats);
  cl_mem mid = createReadWriteBuffer(handle,numMidFloats);
  cl_mem midScratch = createReadWriteBuffer(handle,numMidFloats);

  layer->apply(handle, batchSize, trunk, trunkScratch, mid, midScratch, mask);

  cl_bool blocking = CL_TRUE;
  err = clEnqueueReadBuffer(handle->commandQueue, trunk, blocking, 0, byteSizeofVectorContents(outputBuffer), outputBuffer.data(), 0, NULL, NULL);
  CHECK_ERR(err);

  clReleaseMemObject(trunk);
  clReleaseMemObject(mask);
  clReleaseMemObject(trunkScratch);
  clReleaseMemObject(mid);
  clReleaseMemObject(midScratch);
  delete layer;
  freeComputeHandle(handle);
  freeComputeContext(context);

  return true;
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
  Logger* logger = NULL;
  cl_int err;
  int gpuIdx = 0;

  if(useFP16 != false)
    return false;
  if(useNHWC != false)
    return false;

  LoadedModel* loadedModel = NULL;
  bool requireExactNNLen = false;
  bool inputsUseNHWC = useNHWC;
  ComputeContext* context = createComputeContext({gpuIdx}, logger);
  ComputeHandle* handle = createComputeHandle(
    context, loadedModel, logger, batchSize, nnXLen, nnYLen, requireExactNNLen, inputsUseNHWC, gpuIdx, useFP16, useNHWC
  );

  GlobalPoolingResidualBlock* layer = new GlobalPoolingResidualBlock(handle, desc, nnXLen, nnYLen);

  size_t numTrunkFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)batchSize * nnXLen * nnYLen;
  size_t numMaskSumFloats = (size_t)batchSize;
  size_t numMidFloats = (size_t)batchSize * nnXLen * nnYLen * desc->finalConv.inChannels;
  size_t numGPoolOutFloats = (size_t)batchSize * nnXLen * nnYLen * desc->gpoolConv.outChannels;
  size_t numGPoolConcatFloats = (size_t)batchSize * 3 * desc->gpoolConv.outChannels;
  size_t numGPoolBiasFloats = (size_t)batchSize * desc->regularConv.outChannels;

  if(numTrunkFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");
  outputBuffer.resize(numTrunkFloats);

  vector<float> inputTmp = inputBuffer;
  vector<float> maskTmp = maskBuffer;
  cl_mem trunk = createReadWriteBuffer(handle,inputTmp);
  cl_mem mask = createReadOnlyBuffer(handle,maskTmp);
  cl_mem maskSum = createReadWriteBuffer(handle,numMaskSumFloats);
  cl_mem trunkScratch = createReadWriteBuffer(handle,numTrunkFloats);
  cl_mem mid = createReadWriteBuffer(handle,numMidFloats);
  cl_mem midScratch = createReadWriteBuffer(handle,numMidFloats);
  cl_mem gpoolOut = createReadWriteBuffer(handle,numGPoolOutFloats);
  cl_mem gpoolOut2 = createReadWriteBuffer(handle,numGPoolOutFloats);
  cl_mem gpoolConcat = createReadWriteBuffer(handle,numGPoolConcatFloats);
  cl_mem gpoolBias = createReadWriteBuffer(handle,numGPoolBiasFloats);

  computeMaskSums(handle,mask,maskSum,batchSize,nnXLen,nnYLen);

  layer->apply(
    handle,
    batchSize,
    trunk,
    trunkScratch,
    mid,
    midScratch,
    gpoolOut,
    gpoolOut2,
    gpoolConcat,
    gpoolBias,
    mask,
    maskSum
  );

  cl_bool blocking = CL_TRUE;
  err = clEnqueueReadBuffer(handle->commandQueue, trunk, blocking, 0, byteSizeofVectorContents(outputBuffer), outputBuffer.data(), 0, NULL, NULL);
  CHECK_ERR(err);

  clReleaseMemObject(trunk);
  clReleaseMemObject(mask);
  clReleaseMemObject(maskSum);
  clReleaseMemObject(trunkScratch);
  clReleaseMemObject(mid);
  clReleaseMemObject(midScratch);
  clReleaseMemObject(gpoolOut);
  clReleaseMemObject(gpoolOut2);
  clReleaseMemObject(gpoolConcat);
  clReleaseMemObject(gpoolBias);
  delete layer;
  freeComputeHandle(handle);
  freeComputeContext(context);

  return true;
}
