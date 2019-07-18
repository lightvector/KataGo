#ifdef USE_OPENCL_BACKEND

#include "../neuralnet/openclhelpers.h"
#include "../neuralnet/opencltuner.h"

using namespace std;

const char* OpenCLHelpers::getErrorMessage(cl_int error)
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

void OpenCLHelpers::checkErrors(cl_int error, const char* file, const char* func, int line) {
  if(error != 0)
    throw StringError(string("OpenCL error at ") + file + ", func " + func + ", line " + Global::intToString(line) + ", error " + getErrorMessage(error));
}

template<typename T>
static size_t byteSizeofVectorContents(const typename std::vector<T>& vec) {
    return sizeof(T) * vec.size();
}

cl_program OpenCLHelpers::compileProgram(const string& name, cl_context context, const vector<cl_device_id>& devices, const string& str, const string& options) {
  const char* lines[1] = {str.c_str()};
  const size_t sizes[1] = {str.size()};
  cl_int err;
  cl_program program = clCreateProgramWithSource(context,1,lines,sizes,&err);
  CHECK_ERR(err);

  const string opts = options + " -cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero";

  err = clBuildProgram(program, 0, NULL, opts.c_str(), NULL, NULL);
  if(err != 0) {
    string s;
    s += OpenCLHelpers::getErrorMessage(err) + string("\n");
    for(int i = 0; i<devices.size(); i++) {
      cl_int err2;
      vector<char> buf(100000);
      size_t retSize;
      err2 = clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG, byteSizeofVectorContents(buf), buf.data(), &retSize);
      CHECK_ERR(err2);
      s += "BUILD LOG FOR " + name + " ON DEVICE " + Global::intToString(i) + "\n";
      s += buf.data() + string("\n");
    }
    clReleaseProgram(program);
    throw CompileError(s);
  }
  return program;
}

bool OpenCLHelpers::tryCompileProgram(const string& name, cl_context context, const vector<cl_device_id>& devices, const string& str, const string& options, cl_program& buf) {
  try {
    buf = compileProgram(name,context,devices,str,options);
  }
  catch(CompileError& e) {
    (void)e;
    return false;
  }
  return true;
}

cl_mem OpenCLHelpers::createReadOnlyBuffer(cl_context clContext, vector<float>& data) {
  cl_int err;
  cl_mem buf = clCreateBuffer(
    clContext,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    byteSizeofVectorContents(data),
    data.data(),
    &err
  );
  CHECK_ERR(err);
  return buf;
}

cl_mem OpenCLHelpers::createReadWriteBuffer(cl_context clContext, vector<float>& data) {
  cl_int err;
  cl_mem buf = clCreateBuffer(
    clContext,
    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
    byteSizeofVectorContents(data),
    data.data(),
    &err
  );
  CHECK_ERR(err);
  return buf;
}

cl_mem OpenCLHelpers::createReadWriteBuffer(cl_context clContext, size_t numFloats) {
  //Minimum allocation size, just in case, to avoid allocations of size 0
  if(numFloats < 32)
    numFloats = 32;

  cl_int err;
  cl_mem buf = clCreateBuffer(
    clContext,
    CL_MEM_READ_WRITE,
    numFloats * sizeof(float),
    NULL,
    &err
  );
  CHECK_ERR(err);
  return buf;
}


void OpenCLHelpers::blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numFloats, std::vector<float>& dstBuf) {
  dstBuf.resize(numFloats);
  cl_bool blocking = CL_TRUE;
  cl_int err;
  err = clEnqueueReadBuffer(commandQueue, srcBuf, blocking, 0, byteSizeofVectorContents(dstBuf), dstBuf.data(), 0, NULL, NULL);
  CHECK_ERR(err);
}

vector<DeviceInfo> DeviceInfo::getAllDeviceInfosOnSystem(Logger* logger) {
  cl_int err;
  cl_uint numPlatforms;
  vector<cl_platform_id> platformIds(MAX_PLATFORMS);
  err = clGetPlatformIDs(platformIds.size(), platformIds.data(), &numPlatforms);
  CHECK_ERR(err);
  assert(numPlatforms <= platformIds.size());
  platformIds.resize(numPlatforms);

  constexpr int bufLen = 2048;
  char buf[bufLen];
  for(int i = 0; i<bufLen; i++)
    buf[i] = '\0';

  int numDevicesTotal = 0;
  vector<cl_device_id> deviceIds(MAX_DEVICES);
  for(int platformIdx = 0; platformIdx < numPlatforms && numDevicesTotal < deviceIds.size(); platformIdx++) {
    size_t sizeRet;

    err = clGetPlatformInfo(platformIds[platformIdx], CL_PLATFORM_NAME, bufLen, buf, &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string name = string(buf);

    err = clGetPlatformInfo(platformIds[platformIdx], CL_PLATFORM_VENDOR, bufLen, buf, &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string vendor = string(buf);

    err = clGetPlatformInfo(platformIds[platformIdx], CL_PLATFORM_VERSION, bufLen, buf, &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string version = string(buf);

    if(logger != NULL)
      logger->write("Found OpenCL Platform " + Global::intToString(platformIdx) + ": " + name + " (" + vendor + ") (" + version + ")");

    cl_uint numDevices;
    err = clGetDeviceIDs(
      platformIds[platformIdx], CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, deviceIds.size() - numDevicesTotal,
      deviceIds.data() + numDevicesTotal, &numDevices);
    //Allow there to be 0 devices on this platform, just move on to the next
    if(err == CL_DEVICE_NOT_FOUND) {
      if(logger != NULL)
        logger->write("Found 0 device(s) on platform " + Global::intToString(platformIdx) + " with type GPU or Accelerator, skipping");
      continue;
    }

    CHECK_ERR(err);
    assert(numDevices <= deviceIds.size());
    numDevicesTotal += numDevices;
    if(logger != NULL)
      logger->write("Found " + Global::intToString(numDevices) + " device(s) on platform " + Global::intToString(platformIdx) + " with type GPU or Accelerator");
  }
  deviceIds.resize(numDevicesTotal);

  vector<DeviceInfo> allDeviceInfos;
  for(int gpuIdx = 0; gpuIdx<numDevicesTotal; gpuIdx++) {
    size_t sizeRet;

    err = clGetDeviceInfo(deviceIds[gpuIdx], CL_DEVICE_NAME, bufLen, buf, &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string name = string(buf);

    err = clGetDeviceInfo(deviceIds[gpuIdx], CL_DEVICE_VENDOR, bufLen, buf, &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string vendor = string(buf);

    if(logger != NULL)
      logger->write("Found OpenCL Device " + Global::intToString(gpuIdx) + ": " + name + " (" + vendor + ")");

    DeviceInfo info;
    info.gpuIdx = gpuIdx;
    info.deviceId = deviceIds[gpuIdx];
    info.name = name;
    info.vendor = vendor;
    allDeviceInfos.push_back(info);
  }

  return allDeviceInfos;
}

//----------------------------------------------------------------------------------------


DevicesContext::DevicesContext(const vector<DeviceInfo>& allDeviceInfos, const vector<int>& gIdxsToUse, Logger* logger, bool enableProfiling)
  : devicesToUse(),
    uniqueDeviceNamesToUse()
{
  //Sort and ensure no duplicates
  vector<int> gpuIdxsToUse = gIdxsToUse;
  std::sort(gpuIdxsToUse.begin(),gpuIdxsToUse.end());
  for(size_t i = 1; i<gpuIdxsToUse.size(); i++) {
    if(gpuIdxsToUse[i-1] == gpuIdxsToUse[i])
      throw StringError("Requested gpuIdx/device more than once: " + Global::intToString(gpuIdxsToUse[i]));
  }

  vector<cl_device_id> deviceIdsToUse;
  for(size_t i = 0; i<gpuIdxsToUse.size(); i++) {
    int gpuIdx = gpuIdxsToUse[i];
    if(gpuIdx < 0 || gpuIdx >= allDeviceInfos.size())
      throw StringError(
        "Requested gpuIdx/device " + Global::intToString(gpuIdx) +
        " was not found, valid devices range from 0 to " + Global::intToString((int)allDeviceInfos.size() - 1)
      );
    deviceIdsToUse.push_back(allDeviceInfos[gpuIdx].deviceId);
  }

  cl_int err;
  cl_context_properties* properties = NULL;
  cl_uint numDevicesToUse = (cl_uint)deviceIdsToUse.size();
  context = clCreateContext(properties, numDevicesToUse, deviceIdsToUse.data(), NULL, NULL, &err);
  CHECK_ERR(err);

  for(size_t i = 0; i<gpuIdxsToUse.size(); i++) {
    //TODO - someday, maybe consider CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    cl_command_queue commandQueue;
    if(enableProfiling)
      commandQueue = clCreateCommandQueue(context, deviceIdsToUse[i], CL_QUEUE_PROFILING_ENABLE, &err);
    else
      commandQueue = clCreateCommandQueue(context, deviceIdsToUse[i], 0, &err);

    CHECK_ERR(err);
    InitializedDevice device;
    device.info = allDeviceInfos[gpuIdxsToUse[i]];
    device.commandQueue = commandQueue;
    devicesToUse.push_back(device);

    string message = ("Using OpenCL Device " + Global::intToString(gpuIdxsToUse[i]) + ": " + device.info.name + " (" + device.info.vendor + ")");
    if(logger != NULL) {
      logger->write(message);
      if(!logger->isLoggingToStdout() && !logger->isLoggingToStderr())
        cerr << message << endl;
    }
  }

  for(size_t i = 0; i<gpuIdxsToUse.size(); i++) {
    if(contains(uniqueDeviceNamesToUse, devicesToUse[i].info.name))
      continue;
    uniqueDeviceNamesToUse.push_back(devicesToUse[i].info.name);
  }
}

DevicesContext::~DevicesContext() {
  for(int i = 0; i<devicesToUse.size(); i++) {
    clFlush(devicesToUse[i].commandQueue);
    clFinish(devicesToUse[i].commandQueue);
    clReleaseCommandQueue(devicesToUse[i].commandQueue);
  }
  clReleaseContext(context);
}

const InitializedDevice& DevicesContext::findGpuExn(int gpuIdx) const {
  for(int i = 0; i<devicesToUse.size(); i++) {
    if(devicesToUse[i].info.gpuIdx == gpuIdx)
      return devicesToUse[i];
  }
  throw StringError("BUG? Attempted to create ComputeHandle for a gpuIdx that was not part of the DevicesContext: " + Global::intToString(gpuIdx));
}

vector<InitializedDevice> DevicesContext::findDevicesToUseWithName(const string& name) const {
  vector<InitializedDevice> devices;
  for(int i = 0; i<devicesToUse.size(); i++) {
    if(devicesToUse[i].info.name == name)
      devices.push_back(devicesToUse[i]);
  }
  return devices;
}
vector<cl_device_id> DevicesContext::findDeviceIdsToUseWithName(const string& name) const {
  vector<cl_device_id> deviceIds;
  for(int i = 0; i<devicesToUse.size(); i++) {
    if(devicesToUse[i].info.name == name)
      deviceIds.push_back(devicesToUse[i].info.deviceId);
  }
  return deviceIds;
}


//----------------------------------------------------------------------------------------

size_t OpenCLHelpers::powerOf2ify(size_t size) {
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

size_t OpenCLHelpers::roundUpToMultiple(size_t size, size_t ofThis) {
  return (size + ofThis - 1) / ofThis * ofThis;
}

cl_int OpenCLHelpers::doBatchedXGemm_KM_KN_MN(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  int M, int N, int K,
  cl_mem A, cl_mem B, cl_mem C,
  int numBatchElts,
  cl_event* eventBuf
) {
  int cTranspose = 1;

  clSetKernelArg(kernel, 0, sizeof(int), (void *)&M);
  clSetKernelArg(kernel, 1, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&A);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&M);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&B);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&C);
  clSetKernelArg(kernel, 8, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 9, sizeof(int), (void *)&cTranspose);

  static constexpr int nKernelDims = 3;
  const size_t WGD = tuneParams.xGemmDirect.WGD;
  const size_t MDIMCD = tuneParams.xGemmDirect.MDIMCD;
  const size_t NDIMCD = tuneParams.xGemmDirect.NDIMCD;

  size_t mCeiled = roundUpToMultiple(M,WGD);
  size_t nCeiled = roundUpToMultiple(N,WGD);

  size_t globalSizes[nKernelDims] = {mCeiled * MDIMCD / WGD, nCeiled * NDIMCD / WGD, (size_t)numBatchElts};
  size_t localSizes[nKernelDims] = {MDIMCD, NDIMCD, 1};

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}

cl_int OpenCLHelpers::doStridedBatchedXGemm_KM_KN_MN(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  int M, int N, int K,
  int aStride, int bStride, int cStride,
  cl_mem A, cl_mem B, cl_mem C,
  int numBatchElts,
  cl_event* eventBuf
) {
  int cTranspose = 1;

  clSetKernelArg(kernel, 0, sizeof(int), (void *)&M);
  clSetKernelArg(kernel, 1, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&A);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&M);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&aStride);
  clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&B);
  clSetKernelArg(kernel, 7, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 8, sizeof(int), (void *)&bStride);
  clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&C);
  clSetKernelArg(kernel,10, sizeof(int), (void *)&N);
  clSetKernelArg(kernel,11, sizeof(int), (void *)&cStride);
  clSetKernelArg(kernel,12, sizeof(int), (void *)&cTranspose);

  static constexpr int nKernelDims = 3;
  const size_t WGD = tuneParams.xGemmDirect.WGD;
  const size_t MDIMCD = tuneParams.xGemmDirect.MDIMCD;
  const size_t NDIMCD = tuneParams.xGemmDirect.NDIMCD;

  size_t mCeiled = roundUpToMultiple(M,WGD);
  size_t nCeiled = roundUpToMultiple(N,WGD);

  size_t globalSizes[nKernelDims] = {mCeiled * MDIMCD / WGD, nCeiled * NDIMCD / WGD, (size_t)numBatchElts};
  size_t localSizes[nKernelDims] = {MDIMCD, NDIMCD, 1};

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}

cl_int OpenCLHelpers::doBatchedXGemm_MK_NK_MN(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  int M, int N, int K,
  cl_mem A, cl_mem B, cl_mem C,
  int numBatchElts,
  cl_event* eventBuf
) {
  int cTranspose = 1;

  clSetKernelArg(kernel, 0, sizeof(int), (void *)&M);
  clSetKernelArg(kernel, 1, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&A);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&K);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&B);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&K);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&C);
  clSetKernelArg(kernel, 8, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 9, sizeof(int), (void *)&cTranspose);

  static constexpr int nKernelDims = 3;
  const size_t WGD = tuneParams.xGemmDirect.WGD;
  const size_t MDIMCD = tuneParams.xGemmDirect.MDIMCD;
  const size_t NDIMCD = tuneParams.xGemmDirect.NDIMCD;

  size_t mCeiled = roundUpToMultiple(M,WGD);
  size_t nCeiled = roundUpToMultiple(N,WGD);

  size_t globalSizes[nKernelDims] = {mCeiled * MDIMCD / WGD, nCeiled * NDIMCD / WGD, (size_t)numBatchElts};
  size_t localSizes[nKernelDims] = {MDIMCD, NDIMCD, 1};

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}

cl_int OpenCLHelpers::doWinogradTransform(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  cl_mem input, cl_mem convWorkspace,
  int batchSize, int nnXLen, int nnYLen,
  int numTilesX, int numTilesY,
  int inChannels,
  int convSize,
  cl_event* eventBuf
) {
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&convWorkspace);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&nnXLen);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&nnYLen);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&numTilesX);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&numTilesY);
  clSetKernelArg(kernel, 7, sizeof(int), (void *)&inChannels);

  static constexpr int nKernelDims = 3;
  size_t localSizes[nKernelDims] = {
    (size_t)(convSize == 3 ? tuneParams.conv3x3.transLocalSize0 : tuneParams.conv5x5.transLocalSize0),
    (size_t)(convSize == 3 ? tuneParams.conv3x3.transLocalSize1 : tuneParams.conv5x5.transLocalSize1),
    (size_t)(convSize == 3 ? tuneParams.conv3x3.transLocalSize2 : tuneParams.conv5x5.transLocalSize2)
  };

  size_t globalSizes[nKernelDims] = {
    roundUpToMultiple(powerOf2ify(numTilesX),localSizes[0]),
    roundUpToMultiple(powerOf2ify(numTilesY),localSizes[1]),
    roundUpToMultiple(batchSize * inChannels,localSizes[2])
  };

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}

cl_int OpenCLHelpers::doWinogradUntransform(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  cl_mem convWorkspace2, cl_mem output,
  int batchSize, int nnXLen, int nnYLen,
  int numTilesX, int numTilesY,
  int outChannels,
  int convSize,
  cl_event* eventBuf
) {
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&convWorkspace2);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&nnXLen);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&nnYLen);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&numTilesX);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&numTilesY);
  clSetKernelArg(kernel, 7, sizeof(int), (void *)&outChannels);

  static constexpr int nKernelDims = 3;
  size_t localSizes[nKernelDims] = {
    (size_t)(convSize == 3 ? tuneParams.conv3x3.untransLocalSize0 : tuneParams.conv5x5.untransLocalSize0),
    (size_t)(convSize == 3 ? tuneParams.conv3x3.untransLocalSize1 : tuneParams.conv5x5.untransLocalSize1),
    (size_t)(convSize == 3 ? tuneParams.conv3x3.untransLocalSize2 : tuneParams.conv5x5.untransLocalSize2)
  };

  size_t globalSizes[nKernelDims] = {
    roundUpToMultiple(powerOf2ify(numTilesX),localSizes[0]),
    roundUpToMultiple(powerOf2ify(numTilesY),localSizes[1]),
    roundUpToMultiple(batchSize * outChannels,localSizes[2])
  };

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}



cl_int OpenCLHelpers::performGPool(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  int batchSize, int gpoolChannels, int nnXYLen,
  cl_mem gpoolConvOut, cl_mem gpoolConcat, cl_mem maskSum,
  cl_event* eventBuf
) {
  static constexpr int nKernelDims = 3;
  size_t localSizes[nKernelDims] = {
    (size_t)tuneParams.gPool.XYSTRIDE,
    std::min((size_t)tuneParams.gPool.CHANNELSTRIDE,powerOf2ify(gpoolChannels)),
    std::min((size_t)tuneParams.gPool.BATCHSTRIDE,powerOf2ify(batchSize))
  };
  size_t globalSizes[nKernelDims] = {
    (size_t)tuneParams.gPool.XYSTRIDE,
    roundUpToMultiple(gpoolChannels,localSizes[1]),
    roundUpToMultiple(batchSize,localSizes[2])
  };

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpoolConvOut);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpoolConcat);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&maskSum);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&gpoolChannels);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&nnXYLen);

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}

cl_int OpenCLHelpers::performValueHeadPool(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  int batchSize, int gpoolChannels, int nnXYLen,
  cl_mem gpoolConvOut, cl_mem gpoolConcat, cl_mem maskSum,
  cl_event* eventBuf
) {
  static constexpr int nKernelDims = 3;
  size_t localSizes[nKernelDims] = {
    (size_t)tuneParams.gPool.XYSTRIDE,
    std::min((size_t)tuneParams.gPool.CHANNELSTRIDE,powerOf2ify(gpoolChannels)),
    std::min((size_t)tuneParams.gPool.BATCHSTRIDE,powerOf2ify(batchSize))
  };
  size_t globalSizes[nKernelDims] = {
    (size_t)tuneParams.gPool.XYSTRIDE,
    roundUpToMultiple(gpoolChannels,localSizes[1]),
    roundUpToMultiple(batchSize,localSizes[2])
  };

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&gpoolConvOut);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&gpoolConcat);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&maskSum);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&gpoolChannels);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&nnXYLen);

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}

cl_int OpenCLHelpers::computeMaskSums(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  cl_mem mask,
  cl_mem maskSum,
  int batchSize,
  int nnXLen,
  int nnYLen,
  cl_event* eventBuf
) {
  static constexpr int nKernelDims = 3;
  size_t localSizes[nKernelDims] = {
    (size_t)tuneParams.gPool.XYSTRIDE,
    1,
    std::min((size_t)tuneParams.gPool.BATCHSTRIDE,powerOf2ify(batchSize))
  };
  size_t globalSizes[nKernelDims] = {
    (size_t)tuneParams.gPool.XYSTRIDE,
    1,
    roundUpToMultiple(batchSize,localSizes[2])
  };

  int numChannels = 1;
  int nnXYLen = nnXLen * nnYLen;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mask);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&maskSum);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&numChannels);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&nnXYLen);

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}


cl_int OpenCLHelpers::transposeNCHW(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  int batchSize, int cSize, int nnXLen, int nnYLen,
  cl_mem input, cl_mem output,
  cl_event* eventBuf
) {
  static constexpr int nKernelDims = 3;
  int TILEDIM = tuneParams.transpose.TILEDIM;
  int TILESTRIDE = tuneParams.transpose.TILESTRIDE;
  size_t localSizes[nKernelDims] = {
    (size_t)TILEDIM,
    (size_t)TILESTRIDE,
    std::min((size_t)tuneParams.transpose.NCSTRIDE,powerOf2ify(batchSize*cSize))
  };
  size_t globalSizes[nKernelDims] = {
    (size_t)(nnXLen+TILEDIM-1)/TILEDIM*localSizes[0],
    (size_t)(nnYLen+TILEDIM-1)/TILEDIM*localSizes[1],
    roundUpToMultiple(batchSize*cSize,localSizes[2])
  };

  int ncLen = batchSize*cSize;

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&nnXLen);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&nnYLen);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&ncLen);

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}

#endif
