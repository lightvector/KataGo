#ifdef USE_OPENCL_BACKEND

#include "../neuralnet/openclhelpers.h"
#include "../neuralnet/opencltuner.h"

using namespace std;

using half_t = half_float::half;

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
    cout << e.what() << endl;
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
cl_mem OpenCLHelpers::createReadOnlyBuffer(cl_context clContext, vector<half_t>& data) {
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
cl_mem OpenCLHelpers::createReadWriteBuffer(cl_context clContext, vector<half_t>& data) {
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

cl_mem OpenCLHelpers::createReadWriteBufferFloat(cl_context clContext, size_t numElts) {
  //Minimum allocation size, just in case, to avoid allocations of size 0
  if(numElts < 32)
    numElts = 32;

  cl_int err;
  cl_mem buf = clCreateBuffer(
    clContext,
    CL_MEM_READ_WRITE,
    numElts * sizeof(float),
    NULL,
    &err
  );
  CHECK_ERR(err);
  return buf;
}
cl_mem OpenCLHelpers::createReadWriteBufferHalf(cl_context clContext, size_t numElts) {
  //Minimum allocation size, just in case, to avoid allocations of size 0
  if(numElts < 32)
    numElts = 32;

  cl_int err;
  cl_mem buf = clCreateBuffer(
    clContext,
    CL_MEM_READ_WRITE,
    numElts * sizeof(half_t),
    NULL,
    &err
  );
  CHECK_ERR(err);
  return buf;
}


void OpenCLHelpers::blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, std::vector<float>& dstBuf) {
  dstBuf.resize(numElts);
  cl_bool blocking = CL_TRUE;
  cl_int err;
  err = clEnqueueReadBuffer(commandQueue, srcBuf, blocking, 0, byteSizeofVectorContents(dstBuf), dstBuf.data(), 0, NULL, NULL);
  CHECK_ERR(err);
}
void OpenCLHelpers::blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, std::vector<half_t>& dstBuf) {
  dstBuf.resize(numElts);
  cl_bool blocking = CL_TRUE;
  cl_int err;
  err = clEnqueueReadBuffer(commandQueue, srcBuf, blocking, 0, byteSizeofVectorContents(dstBuf), dstBuf.data(), 0, NULL, NULL);
  CHECK_ERR(err);
}
void OpenCLHelpers::blockingReadBufferHalfToFloat(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, std::vector<float>& dstBuf) {
  vector<half_t> tmpHalf;
  blockingReadBuffer(commandQueue, srcBuf, numElts, tmpHalf);
   dstBuf.resize(numElts);
  for(size_t i = 0; i<numElts; i++)
    dstBuf[i] = tmpHalf[i];
}
void OpenCLHelpers::blockingReadBuffer(cl_command_queue commandQueue, cl_mem srcBuf, size_t numElts, std::vector<float>& dstBuf, bool useFP16) {
  if(useFP16)
    blockingReadBufferHalfToFloat(commandQueue, srcBuf, numElts, dstBuf);
  else
    blockingReadBuffer(commandQueue, srcBuf, numElts, dstBuf);
}

vector<DeviceInfo> DeviceInfo::getAllDeviceInfosOnSystem(Logger* logger) {
  //Some opencl headers/implementations are buggy and have more platforms or more devices than they
  //say their maximum is, so just add a buffer.
  static constexpr size_t maxPlatforms = MAX_PLATFORMS + 128;
  static constexpr size_t maxDevices = MAX_DEVICES + 1024;

  cl_int err;
  cl_uint numPlatforms;
  vector<cl_platform_id> platformIds(maxPlatforms);
  err = clGetPlatformIDs(platformIds.size(), platformIds.data(), &numPlatforms);
  CHECK_ERR(err);
  assert(numPlatforms <= platformIds.size());
  platformIds.resize(numPlatforms);

  constexpr int bufLen = 16384;
  vector<char> buf(bufLen);
  for(int i = 0; i<bufLen; i++)
    buf[i] = '\0';

  int numDevicesTotal = 0;
  vector<cl_device_id> deviceIds(maxDevices);
  vector<cl_platform_id> platformIdsForDevices;
  vector<string> platformDescsForDevices;
  for(int platformIdx = 0; platformIdx < numPlatforms && numDevicesTotal < deviceIds.size(); platformIdx++) {
    size_t sizeRet;
    cl_platform_id platformId = platformIds[platformIdx];

    err = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, bufLen, buf.data(), &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string name = string(buf.data());

    err = clGetPlatformInfo(platformId, CL_PLATFORM_VENDOR, bufLen, buf.data(), &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string vendor = string(buf.data());

    err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, bufLen, buf.data(), &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string version = string(buf.data());

    string desc =  name + " (" + vendor + ") (" + version + ")";
    if(logger != NULL)
      logger->write("Found OpenCL Platform " + Global::intToString(platformIdx) + ": " + desc);

    cl_uint numDevices;
    err = clGetDeviceIDs(
      platformId, CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, deviceIds.size() - numDevicesTotal,
      deviceIds.data() + numDevicesTotal, &numDevices);
    //Allow there to be 0 devices on this platform, just move on to the next
    if(err == CL_DEVICE_NOT_FOUND) {
      if(logger != NULL)
        logger->write("Found 0 device(s) on platform " + Global::intToString(platformIdx) + " with type CPU or GPU or Accelerator, skipping");
      continue;
    }

    for(size_t i = 0; i < numDevices; i++) {
      platformIdsForDevices.push_back(platformId);
      platformDescsForDevices.push_back(desc);
    }

    CHECK_ERR(err);
    numDevicesTotal += numDevices;
    assert(numDevicesTotal <= deviceIds.size());
    if(logger != NULL)
      logger->write("Found " + Global::intToString(numDevices) + " device(s) on platform " + Global::intToString(platformIdx) + " with type CPU or GPU or Accelerator");
  }
  deviceIds.resize(numDevicesTotal);

  vector<DeviceInfo> allDeviceInfos;
  for(int gpuIdx = 0; gpuIdx<numDevicesTotal; gpuIdx++) {
    size_t sizeRet;

    err = clGetDeviceInfo(deviceIds[gpuIdx], CL_DEVICE_NAME, bufLen, buf.data(), &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string name = string(buf.data());

    err = clGetDeviceInfo(deviceIds[gpuIdx], CL_DEVICE_VENDOR, bufLen, buf.data(), &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string vendor = string(buf.data());

    cl_device_type deviceType;
    err = clGetDeviceInfo(deviceIds[gpuIdx], CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, &sizeRet);
    assert(sizeRet <= sizeof(cl_device_type));
    CHECK_ERR(err);

    err = clGetDeviceInfo(deviceIds[gpuIdx], CL_DEVICE_VERSION, bufLen, buf.data(), &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string openCLVersion = string(buf.data());

    err = clGetDeviceInfo(deviceIds[gpuIdx], CL_DEVICE_EXTENSIONS, bufLen, buf.data(), &sizeRet);
    assert(sizeRet < bufLen-1);
    CHECK_ERR(err);
    string extensions = string(buf.data());

    int defaultDesirability = 0;
    //Compute desirability for this device for default device selection
    {
      //We should make sure CPUs don't get ranked above GPUs even if they have good vendor
      bool isCPU = ((deviceType & CL_DEVICE_TYPE_CPU) != 0);

      string lowercaseVendor = Global::toLower(vendor);
      if(lowercaseVendor.find("advanced micro devices") != string::npos && !isCPU) defaultDesirability += 10000000;
      else if(lowercaseVendor.find("amd") != string::npos && !isCPU) defaultDesirability += 10000000;
      else if(lowercaseVendor.find("nvidia") != string::npos && !isCPU) defaultDesirability += 10000000;
      else if(lowercaseVendor.find("intel") != string::npos && !isCPU) defaultDesirability += 5000000;

      if(deviceType == CL_DEVICE_TYPE_GPU) defaultDesirability += 1000000;
      else if(deviceType == CL_DEVICE_TYPE_ACCELERATOR) defaultDesirability += 500000;
      else if((deviceType & CL_DEVICE_TYPE_GPU) != 0) defaultDesirability += 200000;
      else if((deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0) defaultDesirability += 100000;
      else if(deviceType == CL_DEVICE_TYPE_DEFAULT) defaultDesirability += 50000;

      vector<string> versionPieces = Global::split(Global::trim(openCLVersion));
      if(versionPieces.size() >= 2) {
        vector<string> majorMinor = Global::split(Global::trim(versionPieces[1]),'.');
        if(majorMinor.size() == 2) {
          int major = 0;
          int minor = 0;
          bool sucMajor = Global::tryStringToInt(majorMinor[0],major);
          bool sucMinor = Global::tryStringToInt(majorMinor[1],minor);
          if(sucMajor && sucMinor && major >= 0 && major < 100 && minor >= 0 && minor < 100) {
            defaultDesirability += major * 100 + minor;
          }
        }
      }
    }

    if(logger != NULL)
      logger->write(
        "Found OpenCL Device " + Global::intToString(gpuIdx) +
        ": " + name + " (" + vendor + ")" + " (score " +
        Global::intToString(defaultDesirability) + ")"
      );

    DeviceInfo info;
    info.gpuIdx = gpuIdx;
    info.deviceId = deviceIds[gpuIdx];
    info.platformId = platformIdsForDevices[gpuIdx];
    info.platformDesc = platformDescsForDevices[gpuIdx];
    info.name = name;
    info.vendor = vendor;
    info.deviceType = deviceType;
    info.openCLVersion = openCLVersion;
    info.extensions = extensions;
    info.defaultDesirability = defaultDesirability;
    info.supportsFP16Compute = (extensions.find("cl_khr_fp16") != string::npos);
    allDeviceInfos.push_back(info);
  }

  return allDeviceInfos;
}

//----------------------------------------------------------------------------------------


DevicesContext::DevicesContext(const vector<DeviceInfo>& allDeviceInfos, const vector<int>& gIdxsToUse, Logger* logger, bool enableProfiling)
  : initializedPlatforms(),
    devicesToUse(),
    uniqueDeviceNamesToUse()
{
  defaultGpuIdx = 0;
  int bestDesirability = 0;
  for(int gpuIdx = 0; gpuIdx<allDeviceInfos.size(); gpuIdx++) {
    if(allDeviceInfos[gpuIdx].defaultDesirability > bestDesirability) {
      defaultGpuIdx = gpuIdx;
      bestDesirability = allDeviceInfos[gpuIdx].defaultDesirability;
    }
  }

  //Sort and ensure no duplicates
  vector<int> gpuIdxsToUse = gIdxsToUse;
  std::sort(gpuIdxsToUse.begin(),gpuIdxsToUse.end());
  for(size_t i = 1; i<gpuIdxsToUse.size(); i++) {
    if(gpuIdxsToUse[i-1] == gpuIdxsToUse[i])
      throw StringError("Requested gpuIdx/device more than once: " + Global::intToString(gpuIdxsToUse[i]));
  }

  //Handle default gpu idx
  if(gpuIdxsToUse.size() > 0 && gpuIdxsToUse[0] == -1) {
    if(contains(gpuIdxsToUse,defaultGpuIdx))
      gpuIdxsToUse.erase(gpuIdxsToUse.begin());
    else
      gpuIdxsToUse[0] = defaultGpuIdx;
    std::sort(gpuIdxsToUse.begin(),gpuIdxsToUse.end());
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

  for(size_t i = 0; i<gpuIdxsToUse.size(); i++) {
    int gpuIdx = gpuIdxsToUse[i];
    const DeviceInfo& deviceInfo = allDeviceInfos[gpuIdx];
    cl_device_id deviceId = deviceInfo.deviceId;
    cl_platform_id platformId = deviceInfo.platformId;
    if(!contains(initializedPlatforms,platformId)) {
      InitializedPlatform* initializedPlatform = new InitializedPlatform();
      initializedPlatform->platformId = platformId;
      initializedPlatform->platformDesc = deviceInfo.platformDesc;
      initializedPlatforms[platformId] = initializedPlatform;
    }
    InitializedPlatform* initializedPlatform = initializedPlatforms[platformId];
    initializedPlatform->deviceIdsToUseForThisPlatform.push_back(deviceId);
  }

  for(auto iter = initializedPlatforms.begin(); iter != initializedPlatforms.end(); ++iter) {
    InitializedPlatform* initializedPlatform = iter->second;
    cl_platform_id platformId = initializedPlatform->platformId;
    initializedPlatform->properties.push_back(CL_CONTEXT_PLATFORM);
    initializedPlatform->properties.push_back((cl_context_properties)platformId);
    initializedPlatform->properties.push_back(0);

    string message =
      "Creating context for OpenCL Platform: " + initializedPlatform->platformDesc;
    if(logger != NULL) {
      logger->write(message);
      if(!logger->isLoggingToStdout() && !logger->isLoggingToStderr())
        cerr << message << endl;
    }

    cl_int err;
    initializedPlatform->context = clCreateContext(
      initializedPlatform->properties.data(),
      initializedPlatform->deviceIdsToUseForThisPlatform.size(),
      initializedPlatform->deviceIdsToUseForThisPlatform.data(),
      NULL,
      NULL,
      &err
    );
    CHECK_ERR(err);
  }

  for(size_t i = 0; i<gpuIdxsToUse.size(); i++) {
    int gpuIdx = gpuIdxsToUse[i];
    const DeviceInfo& deviceInfo = allDeviceInfos[gpuIdx];
    cl_device_id deviceId = deviceInfo.deviceId;
    cl_platform_id platformId = deviceInfo.platformId;
    cl_context context = initializedPlatforms[platformId]->context;

    //TODO - someday, maybe consider CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    cl_int err;
    cl_command_queue commandQueue;
    if(enableProfiling)
      commandQueue = clCreateCommandQueue(context, deviceId, CL_QUEUE_PROFILING_ENABLE, &err);
    else
      commandQueue = clCreateCommandQueue(context, deviceId, 0, &err);

    CHECK_ERR(err);
    InitializedDevice* device = new InitializedDevice();
    device->info = deviceInfo;
    device->context = context;
    device->commandQueue = commandQueue;
    devicesToUse.push_back(device);

    string message =
      "Using OpenCL Device " + Global::intToString(gpuIdxsToUse[i]) + ": " + device->info.name +
      " (" + device->info.vendor + ") " +
      device->info.openCLVersion + " (Extensions: " + device->info.extensions + ")";
    if(logger != NULL) {
      logger->write(message);
      if(!logger->isLoggingToStdout() && !logger->isLoggingToStderr())
        cerr << message << endl;
    }
  }

  for(size_t i = 0; i<gpuIdxsToUse.size(); i++) {
    if(contains(uniqueDeviceNamesToUse, devicesToUse[i]->info.name))
      continue;
    uniqueDeviceNamesToUse.push_back(devicesToUse[i]->info.name);
  }
}

DevicesContext::~DevicesContext() {
  for(int i = 0; i<devicesToUse.size(); i++) {
    InitializedDevice* device = devicesToUse[i];
    clFlush(device->commandQueue);
    clFinish(device->commandQueue);
    clReleaseCommandQueue(device->commandQueue);
    delete device;
  }

  for(auto iter = initializedPlatforms.begin(); iter != initializedPlatforms.end(); ++iter) {
    InitializedPlatform* initializedPlatform = iter->second;
    clReleaseContext(initializedPlatform->context);
    delete initializedPlatform;
  }
}

const InitializedDevice* DevicesContext::findGpuExn(int gpuIdx) const {
  if(gpuIdx == -1)
    gpuIdx = defaultGpuIdx;
  for(int i = 0; i<devicesToUse.size(); i++) {
    if(devicesToUse[i]->info.gpuIdx == gpuIdx)
      return devicesToUse[i];
  }
  throw StringError("BUG? Attempted to create ComputeHandle for a gpuIdx that was not part of the DevicesContext: " + Global::intToString(gpuIdx));
}

vector<const InitializedDevice*> DevicesContext::findDevicesToUseWithName(const string& name) const {
  vector<const InitializedDevice*> devices;
  for(int i = 0; i<devicesToUse.size(); i++) {
    if(devicesToUse[i]->info.name == name) {
      devices.push_back(devicesToUse[i]);
    }
  }
  return devices;
}
vector<cl_device_id> DevicesContext::findDeviceIdsToUseWithName(const string& name) const {
  vector<cl_device_id> deviceIds;
  for(int i = 0; i<devicesToUse.size(); i++) {
    if(devicesToUse[i]->info.name == name)
      deviceIds.push_back(devicesToUse[i]->info.deviceId);
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

cl_int OpenCLHelpers::doBatchedXGemm_KM_KN_NM(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  int M, int N, int K,
  cl_mem A, cl_mem B, cl_mem C,
  int numBatchElts,
  cl_event* eventBuf
) {
  clSetKernelArg(kernel, 0, sizeof(int), (void *)&M);
  clSetKernelArg(kernel, 1, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&A);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&M);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&K);
  clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&B);
  clSetKernelArg(kernel, 7, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 8, sizeof(int), (void *)&K);
  clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&C);
  clSetKernelArg(kernel,10, sizeof(int), (void *)&M);
  clSetKernelArg(kernel,11, sizeof(int), (void *)&N);

  assert(M % tuneParams.xGemm.MWG == 0);
  assert(N % tuneParams.xGemm.NWG == 0);
  assert(K % tuneParams.xGemm.KWG == 0);

  static constexpr int nKernelDims = 3;
  const size_t MDIMC = tuneParams.xGemm.MDIMC;
  const size_t NDIMC = tuneParams.xGemm.NDIMC;
  const size_t MWG = tuneParams.xGemm.MWG;
  const size_t NWG = tuneParams.xGemm.NWG;

  size_t globalSizes[nKernelDims] = {M * MDIMC / MWG, N * NDIMC / NWG, (size_t)numBatchElts};
  size_t localSizes[nKernelDims] = {MDIMC, NDIMC, 1};

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}

cl_int OpenCLHelpers::doBatchedXGemmDirect_KM_KN_NM(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  int M, int N, int K,
  cl_mem A, cl_mem B, cl_mem C,
  int numBatchElts,
  cl_event* eventBuf
) {
  int cTranspose = 0;

  clSetKernelArg(kernel, 0, sizeof(int), (void *)&M);
  clSetKernelArg(kernel, 1, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&K);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&A);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&M);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&B);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&N);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&C);
  clSetKernelArg(kernel, 8, sizeof(int), (void *)&M);
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

cl_int OpenCLHelpers::doStridedBatchedXGemmDirect_KM_KN_NM(
  cl_kernel kernel,
  cl_command_queue commandQueue,
  const OpenCLTuneParams& tuneParams,
  int M, int N, int K,
  int aStride, int bStride, int cStride,
  cl_mem A, cl_mem B, cl_mem C,
  int numBatchElts,
  cl_event* eventBuf
) {
  int cTranspose = 0;

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
  clSetKernelArg(kernel,10, sizeof(int), (void *)&M);
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

cl_int OpenCLHelpers::doBatchedXGemmDirect_MK_NK_MN(
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
  int nnXLen, int nnYLen,
  int batchSize, int numTilesX, int numTilesY, int batchNumTilesPadMultiple,
  int inChannels, int inChannelsPadMultiple,
  int convSize,
  cl_event* eventBuf
) {
  int inChannelsPadded = roundUpToMultiple(inChannels, inChannelsPadMultiple);
  int batchNumTilesPadded = roundUpToMultiple(batchSize * numTilesX * numTilesY, batchNumTilesPadMultiple);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&convWorkspace);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&nnXLen);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&nnYLen);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&numTilesX);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&numTilesY);
  clSetKernelArg(kernel, 7, sizeof(int), (void *)&inChannels);
  clSetKernelArg(kernel, 8, sizeof(int), (void *)&inChannelsPadded);
  clSetKernelArg(kernel, 9, sizeof(int), (void *)&batchNumTilesPadded);

  static constexpr int nKernelDims = 2;
  size_t localSizes[nKernelDims] = {
    (size_t)(convSize == 3 ? tuneParams.conv3x3.transLocalSize0 : tuneParams.conv5x5.transLocalSize0),
    (size_t)(convSize == 3 ? tuneParams.conv3x3.transLocalSize1 : tuneParams.conv5x5.transLocalSize1),
  };

  size_t globalSizes[nKernelDims] = {
    roundUpToMultiple(batchNumTilesPadded, localSizes[0]),
    roundUpToMultiple(inChannelsPadded,localSizes[1])
  };

  cl_int err;
  err = clEnqueueNDRangeKernel(
    commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, eventBuf
  );
  return err;
}

cl_int OpenCLHelpers::doWinogradTransformWithBNRelu(
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
) {
  int inChannelsPadded = roundUpToMultiple(inChannels, inChannelsPadMultiple);
  int batchNumTilesPadded = roundUpToMultiple(batchSize * numTilesX * numTilesY, batchNumTilesPadMultiple);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&convWorkspace);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&scaleBuf);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&biasBuf);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&mask);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&nnXLen);
  clSetKernelArg(kernel, 7, sizeof(int), (void *)&nnYLen);
  clSetKernelArg(kernel, 8, sizeof(int), (void *)&numTilesX);
  clSetKernelArg(kernel, 9, sizeof(int), (void *)&numTilesY);
  clSetKernelArg(kernel, 10, sizeof(int), (void *)&inChannels);
  clSetKernelArg(kernel, 11, sizeof(int), (void *)&inChannelsPadded);
  clSetKernelArg(kernel, 12, sizeof(int), (void *)&batchNumTilesPadded);

  static constexpr int nKernelDims = 2;
  size_t localSizes[nKernelDims] = {
    (size_t)(convSize == 3 ? tuneParams.conv3x3.transLocalSize0 : tuneParams.conv5x5.transLocalSize0),
    (size_t)(convSize == 3 ? tuneParams.conv3x3.transLocalSize1 : tuneParams.conv5x5.transLocalSize1),
  };

  size_t globalSizes[nKernelDims] = {
    roundUpToMultiple(batchNumTilesPadded, localSizes[0]),
    roundUpToMultiple(inChannelsPadded,localSizes[1])
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
  int nnXLen, int nnYLen,
  int batchSize, int numTilesX, int numTilesY, int batchNumTilesPadMultiple,
  int outChannels, int outChannelsPadMultiple,
  int convSize,
  cl_event* eventBuf
) {
  int outChannelsPadded = roundUpToMultiple(outChannels, outChannelsPadMultiple);
  int batchNumTilesPadded = roundUpToMultiple(batchSize * numTilesX * numTilesY, batchNumTilesPadMultiple);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&convWorkspace2);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output);
  clSetKernelArg(kernel, 2, sizeof(int), (void *)&batchSize);
  clSetKernelArg(kernel, 3, sizeof(int), (void *)&nnXLen);
  clSetKernelArg(kernel, 4, sizeof(int), (void *)&nnYLen);
  clSetKernelArg(kernel, 5, sizeof(int), (void *)&numTilesX);
  clSetKernelArg(kernel, 6, sizeof(int), (void *)&numTilesY);
  clSetKernelArg(kernel, 7, sizeof(int), (void *)&outChannels);
  clSetKernelArg(kernel, 8, sizeof(int), (void *)&outChannelsPadded);
  clSetKernelArg(kernel, 9, sizeof(int), (void *)&batchNumTilesPadded);

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


#endif
