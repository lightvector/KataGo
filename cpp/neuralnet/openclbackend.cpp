#ifdef USE_OPENCL_BACKEND

#include "../neuralnet/nninterface.h"
#include "../neuralnet/openclincludes.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/openclkernels.h"
#include "../neuralnet/opencltuner.h"
#include "../neuralnet/activations.h"

#include "../neuralnet/openclhelpers.h"

#include "../core/simpleallocator.h"

using namespace std;
using namespace OpenCLHelpers;

using half_t = half_float::half;

//======================================================================================================
/*
  FP16 CONVENTIONS.

  When using FP16...
  - Every "spatial" tensor is in FP16.
  -- So, the NHWC tensors for the trunk, and the NHW tensor for the mask are FP16.
  - Additionally, batch norm scales and biases are in FP16.
  - But everything else is NOT in FP16. In particular:
  -- The initial matmul for the global features are FP32
  -- Global pooling an FP16 tensor produces FP32 pooled values
  -- Value head and policy head's global pooling produce FP32 pooled values.
  -- This means that every MatMul layer and MatBias layer is operating in FP32.
  -- Basically, everything non-spatial (except for batch norm) is FP32.

*/

//Define this to print out some of the intermediate values of the neural net
//#define DEBUG_INTERMEDIATE_VALUES

//Define this to try profiling some kernels
//#define PROFILE_KERNELS

#ifdef PROFILE_KERNELS
#define MAYBE_EVENT cl_event event
#define MAYBE_EVENTREF &event
#define MAYBE_FREE_EVENT (void)0

#define MAYBE_PROFILE(_name) {                                          \
    const char* _profileName = (_name);                                 \
    bool isNew = false;                                                 \
    if(!contains(handle->profileAddeds,_profileName)) {                         \
      handle->profileAddeds.push_back(_profileName);                            \
      handle->counters[_profileName] = new int[1];                              \
      handle->timeTakens[_profileName] = new double[1];                         \
      handle->counters[_profileName][0] = 0;                            \
      handle->timeTakens[_profileName][0] = 0.0;                        \
      isNew = true;                                                     \
    }                                                                   \
    int* counter = handle->counters[_profileName];                              \
    double* timeTaken = handle->timeTakens[_profileName];                       \
    handle->profileEvents.push_back(event);                             \
    handle->profileCallbacks.push_back(std::function<void()>([event,counter,timeTaken,_profileName]() { \
          cl_int profileErr;                                            \
          cl_ulong time_start, time_end;                                \
          profileErr = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); CHECK_ERR(profileErr); \
          profileErr = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); CHECK_ERR(profileErr) ; \
          *timeTaken += (time_end - time_start) * 1e-9;                 \
          (*counter)++;                                                 \
        }));                                                            \
    if(isNew) {                                                         \
      handle->profileResultPrinters.push_back(std::function<void()>([_profileName,counter,timeTaken]() { \
            cout << _profileName << " " << *counter << " " << *timeTaken/ *counter << " " << *timeTaken << "\n"; \
          }));                                                          \
    }                                                                   \
  }
#else
#define MAYBE_EVENT (void)0
#define MAYBE_EVENTREF NULL
#define MAYBE_FREE_EVENT (void)0
#define MAYBE_PROFILE(name) (void)0
#endif

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
  // If int is only 2 bytes, this implementation won't work right now.
  static_assert(sizeof(int) >= 4, "");
}

void NeuralNet::globalCleanup() {
}

//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc,expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file,expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

ModelPostProcessParams NeuralNet::getPostProcessParams(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.postProcessParams;
}

//---------------------------------------------------------------------------------------------------------

// Wraps cl_program with a destructor that calls clReleaseProgram
using CLProgram = WrappedWithDeleter<cl_program,int,clReleaseProgram>;

struct CompiledPrograms {
  const OpenCLTuneParams tuneParams;

  const bool usingFP16Storage;
  const bool usingFP16Compute;
  const bool usingFP16TensorCores;
  const bool usingFP16TensorCoresFor1x1;

  CLProgram conv2dNCHWProgram;
  CLProgram winogradConv3x3NCHWTransformProgram;
  CLProgram winogradConv3x3NCHWBNReluTransformProgram;
  CLProgram winogradConv3x3NCHWBNMishTransformProgram;
  CLProgram winogradConv3x3NCHWUntransformProgram;
  CLProgram winogradConv5x5NCHWTransformProgram;
  CLProgram winogradConv5x5NCHWBNReluTransformProgram;
  CLProgram winogradConv5x5NCHWBNMishTransformProgram;
  CLProgram winogradConv5x5NCHWUntransformProgram;
  CLProgram scaleBiasMaskNCHWProgram;
  CLProgram scaleBiasMaskReluNCHWProgram;
  CLProgram scaleBiasMaskMishNCHWProgram;
  CLProgram addPointWiseProgram;
  CLProgram sumChannelsNCHWProgram;
  CLProgram gPoolChannelsNCHWMaskProgram;
  CLProgram valueHeadPoolChannelsNCHWProgram;
  CLProgram addChannelBiasesNCHWProgram;
  CLProgram addCBiasesNCProgram;
  CLProgram addCBiasesNCReluProgram;
  CLProgram addCBiasesNCMishProgram;
  CLProgram extractChannel0NCHWProgram;
  CLProgram xgemmDirectProgram;
  CLProgram xgemmDirectProgramAlwaysFP32;
  CLProgram xgemmProgram;
  CLProgram hgemmWmmaNCHWProgram;

  CompiledPrograms(
    const cl_context& context,
    const vector<cl_device_id>& deviceIdsToUse,
    const OpenCLTuneParams& tParams,
    bool useFP16Storage,
    bool useFP16Compute,
    bool useFP16TensorCores,
    bool useFP16TensorCoresFor1x1
  ) :
    tuneParams(tParams),
    usingFP16Storage(useFP16Storage),
    usingFP16Compute(useFP16Compute),
    usingFP16TensorCores(useFP16TensorCores),
    usingFP16TensorCoresFor1x1(useFP16TensorCoresFor1x1)
  {
    string maybeFP16CompileOptions = "";
    if(useFP16Storage)
      maybeFP16CompileOptions += OpenCLKernels::fp16StorageDefine;
    if(useFP16Compute)
      maybeFP16CompileOptions += OpenCLKernels::fp16ComputeDefine;

    conv2dNCHWProgram = compileProgram(
      "conv2dNCHWProgram", context, deviceIdsToUse, OpenCLKernels::conv2dNCHW,
      maybeFP16CompileOptions
    );
    winogradConv3x3NCHWTransformProgram = compileProgram(
      "winogradConv3x3NCHWTransformProgram", context, deviceIdsToUse, OpenCLKernels::winogradTransformNCHW,
      tuneParams.conv3x3.compileOptions() + maybeFP16CompileOptions
    );
    winogradConv3x3NCHWBNReluTransformProgram = compileProgram(
      "winogradConv3x3NCHWBNReluTransformProgram", context, deviceIdsToUse, OpenCLKernels::winogradBNActTransformNCHW,
      tuneParams.conv3x3.compileOptions() + maybeFP16CompileOptions + OpenCLKernels::actReluDefine
    );
    winogradConv3x3NCHWBNMishTransformProgram = compileProgram(
      "winogradConv3x3NCHWBNMishTransformProgram", context, deviceIdsToUse, OpenCLKernels::winogradBNActTransformNCHW,
      tuneParams.conv3x3.compileOptions() + maybeFP16CompileOptions + OpenCLKernels::actMishDefine
    );
    winogradConv3x3NCHWUntransformProgram = compileProgram(
      "winogradConv3x3NCHWUntransformProgram", context, deviceIdsToUse, OpenCLKernels::winogradUntransformNCHW,
      tuneParams.conv3x3.compileOptions() + maybeFP16CompileOptions
    );
    winogradConv5x5NCHWTransformProgram = compileProgram(
      "winogradConv5x5NCHWTransformProgram", context, deviceIdsToUse, OpenCLKernels::winogradTransformNCHW,
      tuneParams.conv5x5.compileOptions() + maybeFP16CompileOptions
    );
    winogradConv5x5NCHWBNReluTransformProgram = compileProgram(
      "winogradConv5x5NCHWBNReluTransformProgram", context, deviceIdsToUse, OpenCLKernels::winogradBNActTransformNCHW,
      tuneParams.conv5x5.compileOptions() + maybeFP16CompileOptions + OpenCLKernels::actReluDefine
    );
    winogradConv5x5NCHWBNMishTransformProgram = compileProgram(
      "winogradConv5x5NCHWBNMishTransformProgram", context, deviceIdsToUse, OpenCLKernels::winogradBNActTransformNCHW,
      tuneParams.conv5x5.compileOptions() + maybeFP16CompileOptions + OpenCLKernels::actMishDefine
    );
    winogradConv5x5NCHWUntransformProgram = compileProgram(
      "winogradConv5x5NCHWUntransformProgram", context, deviceIdsToUse, OpenCLKernels::winogradUntransformNCHW,
      tuneParams.conv5x5.compileOptions() + maybeFP16CompileOptions
    );

    scaleBiasMaskNCHWProgram = compileProgram(
      "scaleBiasMaskNCHWProgram", context, deviceIdsToUse, OpenCLKernels::scaleBiasMaskActNCHW,
      maybeFP16CompileOptions + OpenCLKernels::actIdenDefine
    );
    scaleBiasMaskReluNCHWProgram = compileProgram(
      "scaleBiasMaskReluNCHWProgram", context, deviceIdsToUse, OpenCLKernels::scaleBiasMaskActNCHW,
      maybeFP16CompileOptions + OpenCLKernels::actReluDefine
    );
    scaleBiasMaskMishNCHWProgram = compileProgram(
      "scaleBiasMaskMishNCHWProgram", context, deviceIdsToUse, OpenCLKernels::scaleBiasMaskActNCHW,
      maybeFP16CompileOptions + OpenCLKernels::actMishDefine
    );
    addPointWiseProgram = compileProgram(
      "addPointWiseProgram", context, deviceIdsToUse, OpenCLKernels::addPointWise,
      maybeFP16CompileOptions
    );
    sumChannelsNCHWProgram = compileProgram(
      "sumChannelsNCHWProgram", context, deviceIdsToUse, OpenCLKernels::sumChannelsNCHW,
      tuneParams.gPool.compileOptions() + maybeFP16CompileOptions
    );
    gPoolChannelsNCHWMaskProgram = compileProgram(
      "gPoolChannelsNCHWMaskProgram", context, deviceIdsToUse, OpenCLKernels::gPoolChannelsNCHWMask,
      tuneParams.gPool.compileOptions() + maybeFP16CompileOptions
    );
    valueHeadPoolChannelsNCHWProgram = compileProgram(
      "valueHeadPoolChannelsNCHWProgram", context, deviceIdsToUse, OpenCLKernels::valueHeadPoolChannelsNCHW,
      tuneParams.gPool.compileOptions() + maybeFP16CompileOptions
    );
    addChannelBiasesNCHWProgram = compileProgram(
      "addChannelBiasesNCHWProgram", context, deviceIdsToUse, OpenCLKernels::addChannelBiasesNCHW,
      maybeFP16CompileOptions
    );
    addCBiasesNCProgram = compileProgram(
      "addCBiasesNCProgram", context, deviceIdsToUse, OpenCLKernels::addCBiasesNCAct,
      maybeFP16CompileOptions + OpenCLKernels::actIdenDefine
    );
    addCBiasesNCReluProgram = compileProgram(
      "addCBiasesNCReluProgram", context, deviceIdsToUse, OpenCLKernels::addCBiasesNCAct,
      maybeFP16CompileOptions + OpenCLKernels::actReluDefine
    );
    addCBiasesNCMishProgram = compileProgram(
      "addCBiasesNCMishProgram", context, deviceIdsToUse, OpenCLKernels::addCBiasesNCAct,
      maybeFP16CompileOptions + OpenCLKernels::actMishDefine
    );
    extractChannel0NCHWProgram = compileProgram(
      "extractChannel0NCHWProgram", context, deviceIdsToUse, OpenCLKernels::extractChannel0NCHW,
      maybeFP16CompileOptions
    );
    xgemmDirectProgram = compileProgram(
      "xgemmDirectProgram", context, deviceIdsToUse, OpenCLKernels::xgemmDirect,
      tuneParams.xGemmDirect.compileOptions() + maybeFP16CompileOptions + " -DROUTINE_GEMMSTRIDEDBATCHED"
    );
    xgemmDirectProgramAlwaysFP32 = compileProgram(
      "xgemmDirectProgramAlwaysFP32", context, deviceIdsToUse, OpenCLKernels::xgemmDirect,
      tuneParams.xGemmDirect.compileOptions() + " -DROUTINE_GEMMBATCHED"
    );
    if(usingFP16TensorCores) {
      xgemmProgram = compileProgram(
        "hgemmWmmaProgram", context, deviceIdsToUse, OpenCLKernels::hgemmWmma,
        tuneParams.hGemmWmma.compileOptions() + maybeFP16CompileOptions
      );
      if(usingFP16TensorCoresFor1x1) {
        hgemmWmmaNCHWProgram = compileProgram(
          "hgemmWmmaNCHWProgram", context, deviceIdsToUse, OpenCLKernels::hgemmWmmaNCHW,
          tuneParams.hGemmWmmaNCHW.compileOptions() + maybeFP16CompileOptions
        );
      }
    }
    else if(usingFP16Compute) {
      xgemmProgram = compileProgram(
        "xgemmProgram", context, deviceIdsToUse, OpenCLKernels::xgemm,
        tuneParams.xGemm16.compileOptions() + maybeFP16CompileOptions
      );
    }
    else {
      xgemmProgram = compileProgram(
        "xgemmProgram", context, deviceIdsToUse, OpenCLKernels::xgemm,
        tuneParams.xGemm.compileOptions() + maybeFP16CompileOptions
      );
    }
  }

  ~CompiledPrograms() {
  }

  CompiledPrograms() = delete;
  CompiledPrograms(const CompiledPrograms&) = delete;
  CompiledPrograms& operator=(const CompiledPrograms&) = delete;
};

//---------------------------------------------------------------------------------------------------------

struct ComputeContext {
  const int nnXLen;
  const int nnYLen;
  const enabled_t usingFP16Mode;
  const enabled_t usingNHWCMode;

  DevicesContext* devicesContext;
  map<cl_device_id,CompiledPrograms*> compiledProgramsByDeviceId;

#ifdef PROFILE_KERNELS
  static constexpr bool liveProfilingKernels = true;
#else
  static constexpr bool liveProfilingKernels = false;
#endif

  ComputeContext(
    const vector<int>& gIdxs,
    Logger* logger,
    int nnX,
    int nnY,
    enabled_t useFP16Mode,
    enabled_t useNHWCMode,
    std::function<OpenCLTuneParams(const string&,int)> getParamsForDeviceName
  ) :
    nnXLen(nnX),
    nnYLen(nnY),
    usingFP16Mode(useFP16Mode),
    usingNHWCMode(useNHWCMode)
  {
    vector<DeviceInfo> allDeviceInfos = DeviceInfo::getAllDeviceInfosOnSystem(logger);
    devicesContext = new DevicesContext(allDeviceInfos,gIdxs,logger,liveProfilingKernels);

    for(int i = 0; i<devicesContext->devicesToUse.size(); i++) {
      const InitializedDevice* device = devicesContext->devicesToUse[i];
      const string& name = device->info.name;
      vector<cl_device_id> deviceIds = { device->info.deviceId };

      OpenCLTuneParams tuneParams = getParamsForDeviceName(name, device->info.gpuIdx);

      bool useFP16Storage = false;
      bool useFP16Compute = false;
      bool useFP16TensorCores = false;
      bool useFP16TensorCoresFor1x1 = false;

      if(useFP16Mode == enabled_t::True) {
        if(!tuneParams.canUseFP16Storage && !tuneParams.canUseFP16Compute && !tuneParams.canUseFP16TensorCores && !tuneParams.canUseFP16TensorCoresFor1x1) {
          if(logger)
            logger->write("Warning: No FP16 support found at all on this device during tuning, but useFP16 is true, trying fp16 storage");
          useFP16Storage = true;
        }
        useFP16Storage = tuneParams.canUseFP16Storage;
        // Only use FP16 compute if not using tensor cores
        useFP16Compute = tuneParams.canUseFP16Storage && !tuneParams.canUseFP16TensorCores;
        useFP16TensorCores = tuneParams.canUseFP16TensorCores;
        useFP16TensorCoresFor1x1 = tuneParams.canUseFP16TensorCoresFor1x1;
      }
      else if(useFP16Mode == enabled_t::Auto) {
        useFP16Storage = tuneParams.shouldUseFP16Storage;
        useFP16Compute = tuneParams.shouldUseFP16Compute;
        useFP16TensorCores = tuneParams.shouldUseFP16TensorCores;
        useFP16TensorCoresFor1x1 = tuneParams.shouldUseFP16TensorCoresFor1x1;
      }

      CompiledPrograms* compiledPrograms = new CompiledPrograms(
        device->context, deviceIds, tuneParams,
        useFP16Storage, useFP16Compute, useFP16TensorCores, useFP16TensorCoresFor1x1
      );
      compiledProgramsByDeviceId[device->info.deviceId] = compiledPrograms;
    }
  }

  ~ComputeContext() {
    for(auto it = compiledProgramsByDeviceId.begin(); it != compiledProgramsByDeviceId.end(); ++it) {
      CompiledPrograms* compiledPrograms = it->second;
      delete compiledPrograms;
    }
    delete devicesContext;
  }

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

};

static ComputeContext* createComputeContextForTesting(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC
) {
  enabled_t useFP16Mode = useFP16 ? enabled_t::True : enabled_t::False;
  enabled_t useNHWCMode = useNHWC ? enabled_t::True : enabled_t::False;

  std::function<OpenCLTuneParams(const string&,int)> getParamsForDeviceName =
    [](const string& name, int gpuIdxForTuning) {
    (void)name;
    (void)gpuIdxForTuning;
    //Just use default values
    OpenCLTuneParams params = OpenCLTuneParams();
    //params.shouldUseFP16TensorCores = true;
    return params;
  };
  return new ComputeContext(gpuIdxs,logger,nnXLen,nnYLen,useFP16Mode,useNHWCMode,getParamsForDeviceName);
}

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  if(gpuIdxs.size() <= 0)
    throw StringError("NeuralNet::createComputeContext - specified no gpus to use");

  std::function<OpenCLTuneParams(const string&,int)> getParamsForDeviceName =
    [&openCLTunerFile,&homeDataDirOverride,openCLReTunePerBoardSize,logger,nnXLen,nnYLen,useFP16Mode,loadedModel](const string& name, int gpuIdxForTuning) {
    bool full = false;
    enabled_t testFP16Mode = useFP16Mode;
    enabled_t testFP16StorageMode = useFP16Mode;
    enabled_t testFP16ComputeMode = enabled_t::Auto;
    enabled_t testFP16TensorCoresMode = enabled_t::Auto;

    return OpenCLTuner::loadOrAutoTune(
      openCLTunerFile,homeDataDirOverride,name,gpuIdxForTuning,logger,openCLReTunePerBoardSize,
      nnXLen,nnYLen,
      testFP16Mode,testFP16StorageMode,testFP16ComputeMode,testFP16TensorCoresMode,
      OpenCLTuner::ModelInfoForTuning::ofDesc(&(loadedModel->modelDesc)),
      full
    );
  };
  return new ComputeContext(gpuIdxs,logger,nnXLen,nnYLen,useFP16Mode,useNHWCMode,getParamsForDeviceName);
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}


//--------------------------------------------------------------

// Wraps cl_kernel with a destructor that calls clReleaseKernel
using CLKernel = WrappedWithDeleter<cl_kernel,int,clReleaseKernel>;

struct ComputeHandleInternal {
  ComputeContext* computeContext;
  cl_context clContext;
  cl_command_queue commandQueue;
  OpenCLTuneParams tuneParams;

  bool usingFP16Storage;
  bool usingFP16Compute;
  bool usingFP16TensorCores;
  bool usingFP16TensorCoresFor1x1;

  CLKernel conv2dNCHWKernel;
  CLKernel winogradConv3x3NCHWTransformKernel;
  CLKernel winogradConv3x3NCHWBNReluTransformKernel;
  CLKernel winogradConv3x3NCHWBNMishTransformKernel;
  CLKernel winogradConv3x3NCHWUntransformKernel;
  CLKernel winogradConv5x5NCHWTransformKernel;
  CLKernel winogradConv5x5NCHWBNReluTransformKernel;
  CLKernel winogradConv5x5NCHWBNMishTransformKernel;
  CLKernel winogradConv5x5NCHWUntransformKernel;
  CLKernel scaleBiasMaskNCHWKernel;
  CLKernel scaleBiasMaskReluNCHWKernel;
  CLKernel scaleBiasMaskMishNCHWKernel;
  CLKernel addPointWiseKernel;
  CLKernel sumChannelsNCHWKernel;
  CLKernel gPoolChannelsNCHWMaskKernel;
  CLKernel valueHeadPoolChannelsNCHWKernel;
  CLKernel addChannelBiasesNCHWKernel;
  CLKernel addCBiasesNCKernel;
  CLKernel addCBiasesNCReluKernel;
  CLKernel addCBiasesNCMishKernel;
  CLKernel extractChannel0NCHWKernel;
  CLKernel xgemmDirectBatchedTTKernelAlwaysFP32;
  CLKernel xgemmDirectStridedBatchedNNKernel;
  CLKernel xgemmBatchedNNKernel;
  CLKernel hgemmWmmaNCHWKernel;

  std::vector<const char*> profileAddeds;
  std::map<const char*, int*> counters;
  std::map<const char*, double*> timeTakens;
  vector<cl_event> profileEvents;
  vector<std::function<void()>> profileCallbacks;
  vector<std::function<void()>> profileResultPrinters;

  ComputeHandleInternal(ComputeContext* ctx, int gpuIdx, bool inputsUseNHWC, bool useNHWC) {
    computeContext = ctx;

    const InitializedDevice* device = computeContext->devicesContext->findGpuExn(gpuIdx);
    clContext = device->context;
    commandQueue = device->commandQueue;
    CompiledPrograms* progs = computeContext->compiledProgramsByDeviceId[device->info.deviceId];
    assert(progs != NULL);
    tuneParams = progs->tuneParams;

    if(inputsUseNHWC != false)
      throw StringError("OpenCL backend: inputsUseNHWC = false required, other configurations not supported");
    if(useNHWC != false)
      throw StringError("OpenCL backend: useNHWC = false required, other configurations not supported");

    usingFP16Storage = progs->usingFP16Storage;
    usingFP16Compute = progs->usingFP16Compute;
    usingFP16TensorCores = progs->usingFP16TensorCores;
    usingFP16TensorCoresFor1x1 = progs->usingFP16TensorCoresFor1x1;

    cl_int err;
    conv2dNCHWKernel = clCreateKernel(progs->conv2dNCHWProgram, "conv2dNCHW", &err);
    CHECK_ERR(err);

    winogradConv3x3NCHWTransformKernel = clCreateKernel(progs->winogradConv3x3NCHWTransformProgram, "transform", &err);
    CHECK_ERR(err);
    winogradConv3x3NCHWBNReluTransformKernel = clCreateKernel(progs->winogradConv3x3NCHWBNReluTransformProgram, "bnActTransform", &err);
    CHECK_ERR(err);
    winogradConv3x3NCHWBNMishTransformKernel = clCreateKernel(progs->winogradConv3x3NCHWBNMishTransformProgram, "bnActTransform", &err);
    CHECK_ERR(err);
    winogradConv3x3NCHWUntransformKernel = clCreateKernel(progs->winogradConv3x3NCHWUntransformProgram, "untransform", &err);
    CHECK_ERR(err);

    winogradConv5x5NCHWTransformKernel = clCreateKernel(progs->winogradConv5x5NCHWTransformProgram, "transform", &err);
    CHECK_ERR(err);
    winogradConv5x5NCHWBNReluTransformKernel = clCreateKernel(progs->winogradConv5x5NCHWBNReluTransformProgram, "bnActTransform", &err);
    CHECK_ERR(err);
    winogradConv5x5NCHWBNMishTransformKernel = clCreateKernel(progs->winogradConv5x5NCHWBNMishTransformProgram, "bnActTransform", &err);
    CHECK_ERR(err);
    winogradConv5x5NCHWUntransformKernel = clCreateKernel(progs->winogradConv5x5NCHWUntransformProgram, "untransform", &err);
    CHECK_ERR(err);

    scaleBiasMaskNCHWKernel = clCreateKernel(progs->scaleBiasMaskNCHWProgram, "scaleBiasMaskActNCHW", &err);
    CHECK_ERR(err);
    scaleBiasMaskReluNCHWKernel = clCreateKernel(progs->scaleBiasMaskReluNCHWProgram, "scaleBiasMaskActNCHW", &err);
    CHECK_ERR(err);
    scaleBiasMaskMishNCHWKernel = clCreateKernel(progs->scaleBiasMaskMishNCHWProgram, "scaleBiasMaskActNCHW", &err);
    CHECK_ERR(err);
    addPointWiseKernel = clCreateKernel(progs->addPointWiseProgram, "addPointWise", &err);
    CHECK_ERR(err);
    sumChannelsNCHWKernel = clCreateKernel(progs->sumChannelsNCHWProgram, "sumChannelsNCHW", &err);
    CHECK_ERR(err);
    gPoolChannelsNCHWMaskKernel = clCreateKernel(progs->gPoolChannelsNCHWMaskProgram, "gPoolChannelsNCHWMask", &err);
    CHECK_ERR(err);
    valueHeadPoolChannelsNCHWKernel = clCreateKernel(progs->valueHeadPoolChannelsNCHWProgram, "valueHeadPoolChannelsNCHW", &err);
    CHECK_ERR(err);
    addChannelBiasesNCHWKernel = clCreateKernel(progs->addChannelBiasesNCHWProgram, "addChannelBiasesNCHW", &err);
    CHECK_ERR(err);
    addCBiasesNCKernel = clCreateKernel(progs->addCBiasesNCProgram, "addCBiasesNCAct", &err);
    CHECK_ERR(err);
    addCBiasesNCReluKernel = clCreateKernel(progs->addCBiasesNCReluProgram, "addCBiasesNCAct", &err);
    CHECK_ERR(err);
    addCBiasesNCMishKernel = clCreateKernel(progs->addCBiasesNCMishProgram, "addCBiasesNCAct", &err);
    CHECK_ERR(err);
    extractChannel0NCHWKernel = clCreateKernel(progs->extractChannel0NCHWProgram, "extractChannel0NCHW", &err);
    CHECK_ERR(err);
    xgemmDirectBatchedTTKernelAlwaysFP32 = clCreateKernel(progs->xgemmDirectProgramAlwaysFP32, "XgemmDirectBatchedTT", &err);
    CHECK_ERR(err);
    xgemmDirectStridedBatchedNNKernel = clCreateKernel(progs->xgemmDirectProgram, "XgemmDirectStridedBatchedNN", &err);
    CHECK_ERR(err);
    if(usingFP16TensorCores)
      xgemmBatchedNNKernel = clCreateKernel(progs->xgemmProgram, "hgemmWmmaBatched", &err);
    else
      xgemmBatchedNNKernel = clCreateKernel(progs->xgemmProgram, "XgemmBatched", &err);
    if(usingFP16TensorCoresFor1x1)
      hgemmWmmaNCHWKernel = clCreateKernel(progs->hgemmWmmaNCHWProgram, "hgemmWmmaNCHW", &err);
    CHECK_ERR(err);
  }

  ~ComputeHandleInternal() {
    for(int i = 0; i<profileEvents.size(); i++) {
      if(profileEvents[i] != NULL)
        clReleaseEvent(profileEvents[i]);
    }
  }

  ComputeHandleInternal() = delete;
  ComputeHandleInternal(const ComputeHandleInternal&) = delete;
  ComputeHandleInternal& operator=(const ComputeHandleInternal&) = delete;

  int getXGemmMPaddingMult() const {
    return tuneParams.getXGemmMPaddingMult(usingFP16Compute,usingFP16TensorCores);
  }
  int getXGemmNPaddingMult() const {
    return tuneParams.getXGemmNPaddingMult(usingFP16Compute,usingFP16TensorCores);
  }
  int getXGemmKPaddingMult() const {
    return tuneParams.getXGemmKPaddingMult(usingFP16Compute,usingFP16TensorCores);
  }
  int getHGemmWmmaNCHWRequiredCDivisor() const {
    return tuneParams.hGemmWmmaNCHW.getRequiredCDivisor();
  }

};

static cl_mem createReadOnlyBuffer(ComputeHandleInternal* handle, vector<float>& data, bool useFP16) {
  if(useFP16) {
    vector<half_t> dataHalf(data.size());
    for(size_t i = 0; i<data.size(); i++)
      dataHalf[i] = half_float::half_cast<half_t>(data[i]);
    return createReadOnlyBuffer(handle->clContext,dataHalf);
  }
  else
    return createReadOnlyBuffer(handle->clContext,data);
}
static cl_mem createReadWriteBuffer(ComputeHandleInternal* handle, vector<float>& data, bool useFP16) {
  if(useFP16) {
    vector<half_t> dataHalf(data.size());
    for(size_t i = 0; i<data.size(); i++)
      dataHalf[i] = half_float::half_cast<half_t>(data[i]);
    return createReadWriteBuffer(handle->clContext,dataHalf);
  }
  else
    return createReadWriteBuffer(handle->clContext,data);
}
static cl_mem createReadWriteBuffer(ComputeHandleInternal* handle, size_t numElts, bool useFP16) {
  if(useFP16)
    return createReadWriteBufferHalf(handle->clContext,numElts);
  else
    return createReadWriteBufferFloat(handle->clContext,numElts);
}

static void addChannelBiases(ComputeHandleInternal* handle, cl_mem src, cl_mem bias, int ncSize, int nnXYLen) {
  cl_int err;
  static constexpr int nKernelDims = 2;
  size_t globalSizes[nKernelDims] = {powerOf2ify(nnXYLen),powerOf2ify(ncSize)};
  size_t* localSizes = NULL;

  cl_kernel kernel = handle->addChannelBiasesNCHWKernel;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), (const void *)&src);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *)&bias);
  clSetKernelArg(kernel, 2, sizeof(int), (const void *)&ncSize);
  clSetKernelArg(kernel, 3, sizeof(int), (const void *)&nnXYLen);

  MAYBE_EVENT;
  err = clEnqueueNDRangeKernel(
    handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, MAYBE_EVENTREF
  );
  CHECK_ERR(err);
  MAYBE_PROFILE("AddChannelBiases");
  MAYBE_FREE_EVENT;
}

static void addPointWise(ComputeHandleInternal* handle, cl_mem acc, cl_mem value, int totalSize) {
  cl_int err;
  MAYBE_EVENT;
  err = OpenCLHelpers::doAddPointWise(handle->addPointWiseKernel, handle->commandQueue, acc, value, totalSize, MAYBE_EVENTREF);
  CHECK_ERR(err);
  MAYBE_PROFILE("AddPointWise");
  MAYBE_FREE_EVENT;
}

static void performGPoolMask(ComputeHandleInternal* handle, int batchSize, int gpoolChannels, int nnXYLen, cl_mem gpoolConvOut, cl_mem gpoolConcat, cl_mem mask, cl_mem maskSum) {
  cl_int err;
  MAYBE_EVENT;
  err = OpenCLHelpers::performGPoolMask(
    handle->gPoolChannelsNCHWMaskKernel,
    handle->commandQueue,
    handle->tuneParams,
    batchSize, gpoolChannels, nnXYLen,
    gpoolConvOut, gpoolConcat, mask, maskSum,
    MAYBE_EVENTREF
  );
  CHECK_ERR(err);
  MAYBE_PROFILE("PerformGPoolMask");
  MAYBE_FREE_EVENT;
}

static void performValueHeadPool(ComputeHandleInternal* handle, int batchSize, int gpoolChannels, int nnXYLen, cl_mem gpoolConvOut, cl_mem gpoolConcat, cl_mem maskSum) {
  cl_int err;
  MAYBE_EVENT;
  err = OpenCLHelpers::performValueHeadPool(
    handle->valueHeadPoolChannelsNCHWKernel,
    handle->commandQueue,
    handle->tuneParams,
    batchSize, gpoolChannels, nnXYLen,
    gpoolConvOut, gpoolConcat, maskSum,
    MAYBE_EVENTREF
  );
  CHECK_ERR(err);
  MAYBE_PROFILE("PerformVHPool");
  MAYBE_FREE_EVENT;
}


#ifdef DEBUG_INTERMEDIATE_VALUES
static void debugPrint2D(const string& name, ComputeHandleInternal* handle, cl_mem deviceBuf, int batchSize, int cSize) {
  vector<float> values;
  blockingReadBuffer(handle->commandQueue, deviceBuf, batchSize * cSize, values);
  cout << "=========================================================" << endl;
  cout << name << endl;
  int i = 0;
  for(int n = 0; n<batchSize; n++) {
    cout << "-(n=" << n << ")--------------------" << endl;
    for(int c = 0; c<cSize; c++)
      cout << values[i++] << " ";
    cout << endl;
  }
  cout << endl;
  cout << "=========================================================" << endl;
}

static void debugPrint4D(const string& name, ComputeHandleInternal* handle, cl_mem deviceBuf, int batchSize, int cSize, int nnXLen, int nnYLen, bool useNHWC) {
  vector<float> values;
  blockingReadBuffer(handle->commandQueue, deviceBuf, batchSize * cSize * nnXLen * nnYLen, values);
  cout << "=========================================================" << endl;
  cout << name << endl;
  int i = 0;
  for(int n = 0; n<batchSize; n++) {
    cout << "-(n=" << n << ")--------------------" << endl;
    if(useNHWC) {
      for(int y = 0; y<nnYLen; y++) {
        cout << "(y=" << y << ")" << endl;
        for(int x = 0; x<nnXLen; x++) {
          for(int c = 0; c<cSize; c++)
            cout << values[i++] << " ";
          cout << endl;
        }
        cout << endl;
      }
    }
    else {
      for(int c = 0; c<cSize; c++) {
        cout << "(c=" << c << ")" << endl;
        for(int y = 0; y<nnYLen; y++) {
          for(int x = 0; x<nnXLen; x++)
            cout << values[i++] << " ";
          cout << endl;
        }
        cout << endl;
      }
    }
  }
  cout << "=========================================================" << endl;
}
#endif

//--------------------------------------------------------------

struct ScratchBuffers {

  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  ComputeHandleInternal* handle;
  SimpleAllocator<cl_mem>* allocator;

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(ComputeHandleInternal* handle_, int maxBatchSize, int nnXLen, int nnYLen)
    : batchXYFloatBytes((size_t)maxBatchSize * nnXLen * nnYLen * sizeof(float)),
      batchFloatBytes((size_t)maxBatchSize * sizeof(float)),
      batchXYBytes((size_t)maxBatchSize * nnXLen * nnYLen * (handle_->usingFP16Storage ? sizeof(half_t) : sizeof(float))),
      batchBytes((size_t)maxBatchSize * (handle_->usingFP16Storage ? sizeof(half_t) : sizeof(float))),
      handle(handle_)
  {
    std::function<cl_mem(size_t)> allocateFunc = [this](size_t size) {
      return createReadWriteBufferBytes(handle->clContext, size);
    };
    std::function<void(cl_mem)> releaseFunc = [this](cl_mem buf) {
      clReleaseMemObject(buf);
    };

    allocator = new SimpleAllocator<cl_mem>(allocateFunc, releaseFunc);
  }
  ~ScratchBuffers() {
    delete allocator;
  }

  size_t getBufSizeXY(int channels) const {
    return channels * batchXYBytes;
  }
  size_t getBufSizeXYFloat(int channels) const {
    return channels * batchXYFloatBytes;
  }
  size_t getBufSizeFloat(int channels) const {
    return channels * batchFloatBytes;
  }
  size_t getBufSize(int channels) const {
    return channels * batchBytes;
  }

};

//--------------------------------------------------------------

struct ConvWorkspaceEltsNeeded {
  size_t size1;
  size_t size2;
  ConvWorkspaceEltsNeeded()
    :size1(0),size2(0)
  {}
  ConvWorkspaceEltsNeeded(size_t s1, size_t s2)
    :size1(s1),size2(s2)
  {}
  static ConvWorkspaceEltsNeeded getMax(ConvWorkspaceEltsNeeded a, ConvWorkspaceEltsNeeded b) {
    return ConvWorkspaceEltsNeeded(std::max(a.size1,b.size1),std::max(a.size2,b.size2));
  }
};

//--------------------------------------------------------------

struct BatchNormLayer {
  const string name;
  const int numChannels;
  const float epsilon;
  const int activation;

  const int nnXLen;
  const int nnYLen;
  const int nnXYLen;
  cl_mem mergedScaleBuf;
  cl_mem mergedBiasBuf;

  static constexpr int nKernelDims = 2;
  size_t globalSizes[nKernelDims];

  BatchNormLayer(
    ComputeHandleInternal* handle,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    int nnX,
    int nnY,
    bool useFP16
  ) :
    name(desc->name),
    numChannels(desc->numChannels),
    epsilon(desc->epsilon),
    activation(actDesc->activation),
    nnXLen(nnX),
    nnYLen(nnY),
    nnXYLen(nnX * nnY)
  {
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

    mergedScaleBuf = createReadOnlyBuffer(handle,mergedScale,useFP16);
    mergedBiasBuf = createReadOnlyBuffer(handle,mergedBias,useFP16);

    globalSizes[0] = powerOf2ify(nnXLen * nnYLen);
    globalSizes[1] = powerOf2ify(numChannels);
  }

  ~BatchNormLayer() {
    clReleaseMemObject(mergedScaleBuf);
    clReleaseMemObject(mergedBiasBuf);
  }

  void apply(ComputeHandleInternal* handle, int batchSize, cl_mem input, cl_mem output, cl_mem mask) const {
    cl_kernel kernel;
    if(activation == ACTIVATION_IDENTITY)
      kernel = handle->scaleBiasMaskNCHWKernel;
    else if(activation == ACTIVATION_RELU)
      kernel = handle->scaleBiasMaskReluNCHWKernel;
    else if(activation == ACTIVATION_MISH)
      kernel = handle->scaleBiasMaskMishNCHWKernel;
    else
      assert(false);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (const void *)&input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *)&output);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *)&mergedScaleBuf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), (const void *)&mergedBiasBuf);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (const void *)&mask);
    clSetKernelArg(kernel, 5, sizeof(int), (const void *)&batchSize);
    clSetKernelArg(kernel, 6, sizeof(int), (const void *)&numChannels);
    clSetKernelArg(kernel, 7, sizeof(int), (const void *)&nnXYLen);

    cl_int err;
    size_t* localSizes = NULL; //TODO actually pick these with tuning? Or fuse with conv untransform?
    MAYBE_EVENT;
    err = clEnqueueNDRangeKernel(
      handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, MAYBE_EVENTREF
    );
    CHECK_ERR(err);
    MAYBE_PROFILE("BatchNorm");
    MAYBE_FREE_EVENT;
  }

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;
};

//--------------------------------------------------------------

struct ConvLayer {
  const string name;
  const int convYSize;
  const int convXSize;
  const int convYRadius;
  const int convXRadius;
  const int inChannels;
  const int outChannels;
  const int dilationY;
  const int dilationX;

  const int nnXLen;
  const int nnYLen;

  bool usingHGemmWmmaNHCW; // For 1x1 convs

  int numTilesX;
  int numTilesY;
  int inTileXYSize;
  int outTileXYSize;

  cl_mem filter;

  static constexpr int nKernelDims = 3;

  ConvLayer(
    ComputeHandleInternal* handle,
    const ConvLayerDesc* desc,
    int nnX,
    int nnY,
    bool useFP16
  ) :
    name(desc->name),
    convYSize(desc->convYSize),
    convXSize(desc->convXSize),
    convYRadius(desc->convYSize / 2),
    convXRadius(desc->convXSize / 2),
    inChannels(desc->inChannels),
    outChannels(desc->outChannels),
    dilationY(desc->dilationY),
    dilationX(desc->dilationX),
    nnXLen(nnX),
    nnYLen(nnY)
  {
    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);
    if(dilationX != 1 || dilationY != 1)
      throw StringError("OpenCL backend: Encountered convolution dilation factors other than 1, not supported");

    //Initial values unless overrided below
    usingHGemmWmmaNHCW = false;
    numTilesX = 0;
    numTilesY = 0;
    inTileXYSize = 0;
    outTileXYSize = 0;

    if(convXSize == 1 && convYSize == 1) {
      //ic,oc
      vector<float> transWeights(inChannels * outChannels);
      for(int oc = 0; oc < outChannels; oc++) {
        for(int ic = 0; ic < inChannels; ic++) {
          transWeights[ic * outChannels + oc] = desc->weights[oc * inChannels + ic];
        }
      }
      filter = createReadOnlyBuffer(handle,transWeights,useFP16);
      if(handle->usingFP16TensorCoresFor1x1) {
        if(inChannels % handle->getHGemmWmmaNCHWRequiredCDivisor() == 0 && outChannels % handle->getHGemmWmmaNCHWRequiredCDivisor() == 0) {
          usingHGemmWmmaNHCW = true;
        }
      }
    }
    else if((convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)) {
      int inTileXSize = convXSize == 3 ? handle->tuneParams.conv3x3.INTILE_XSIZE : handle->tuneParams.conv5x5.INTILE_XSIZE;
      int inTileYSize = convYSize == 3 ? handle->tuneParams.conv3x3.INTILE_YSIZE : handle->tuneParams.conv5x5.INTILE_YSIZE;
      int outTileXSize = convXSize == 3 ? handle->tuneParams.conv3x3.OUTTILE_XSIZE : handle->tuneParams.conv5x5.OUTTILE_XSIZE;
      int outTileYSize = convYSize == 3 ? handle->tuneParams.conv3x3.OUTTILE_YSIZE : handle->tuneParams.conv5x5.OUTTILE_YSIZE;

      int outChannelsPadded = roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
      int inChannelsPadded = roundUpToMultipleInt(inChannels, handle->getXGemmKPaddingMult());

      numTilesX = (nnXLen + outTileXSize - 1) / outTileXSize;
      numTilesY = (nnYLen + outTileYSize - 1) / outTileYSize;
      inTileXYSize = inTileXSize * inTileYSize;
      outTileXYSize = outTileXSize * outTileYSize;

      static constexpr int maxTileXSize = 6;
      static constexpr int maxTileYSize = 6;

      assert((convXSize == 3 && convYSize == 3) ? (inTileXSize == 4 && outTileXSize == 2) || (inTileXSize == 6 && outTileXSize == 4) : true);
      assert((convXSize == 5 && convYSize == 5) ? (inTileYSize == 6 && outTileYSize == 2) : true);

      //INTILE_YSIZE, INTILE_XSIZE, ic, oc
      vector<float> transWeights(inTileXYSize * inChannelsPadded * outChannelsPadded);
      auto transform3x3_4 = [](float& a0, float& a1, float& a2, float& a3) {
        float z0 = a0; float z1 = a1; float z2 = a2;
        a0 = z0;
        a1 = 0.5f * (z0 + z1 + z2);
        a2 = 0.5f * (z0 - z1 + z2);
        a3 = z2;
      };
      auto transform3x3_6 = [](float& a0, float& a1, float& a2, float& a3, float& a4, float& a5) {
        float z0 = a0; float z1 = a1; float z2 = a2;
        // Low error winograd
        // double sqrt2 = sqrt(2.0);
        // a0 = z0;
        // a1 = (float)( (1.0 / 3.0) * (-2.0*z0 - sqrt2*z1 - z2) );
        // a2 = (float)( (1.0 / 3.0) * (-2.0*z0 + sqrt2*z1 - z2) );
        // a3 = (float)( (1.0 / 6.0) * (z0 + sqrt2*z1 + 2.0*z2) );
        // a4 = (float)( (1.0 / 6.0) * (z0 - sqrt2*z1 + 2.0*z2) );
        // a5 = z2;
        a0 = 0.25f * z0;
        a1 = (float)( (1.0 / 6.0) * (-z0 - z1 - z2) );
        a2 = (float)( (1.0 / 6.0) * (-z0 + z1 - z2) );
        a3 = (float)( (1.0 / 24.0) * (z0 + 2.0*z1 + 4.0*z2) );
        a4 = (float)( (1.0 / 24.0) * (z0 - 2.0*z1 + 4.0*z2) );
        a5 = 1.0f * z2;
      };
      auto transform5x5_6 = [](float& a0, float& a1, float& a2, float& a3, float& a4, float& a5) {
        float z0 = a0; float z1 = a1; float z2 = a2; float z3 = a3; float z4 = a4;
        a0 = 0.25f * z0;
        a1 = (float)( (1.0 / 6.0) * (-z0 - z1 - z2 - z3 - z4) );
        a2 = (float)( (1.0 / 6.0) * (-z0 + z1 - z2 + z3 - z4) );
        a3 = (float)( (1.0 / 24.0) * (z0 + 2.0*z1 + 4.0*z2 + 8.0*z3 + 16.0*z4) );
        a4 = (float)( (1.0 / 24.0) * (z0 - 2.0*z1 + 4.0*z2 - 8.0*z3 + 16.0*z4) );
        a5 = 1.0f * z4;
      };

      for(int oc = 0; oc < outChannelsPadded; oc++) {
        for(int ic = 0; ic < inChannelsPadded; ic++) {
          float tmp[maxTileYSize][maxTileXSize];
          for(int subY = 0; subY < convYSize; subY++) {
            for(int subX = 0; subX < convXSize; subX++) {
              if(oc < outChannels && ic < inChannels)
                tmp[subY][subX] = desc->weights[((oc * inChannels + ic) * convYSize + subY) * convXSize + subX];
              else
                tmp[subY][subX] = 0.0f;
            }
          }

          if(convXSize == 3 && inTileXSize == 4) {
            for(int subY = 0; subY < convYSize; subY++)
              transform3x3_4(tmp[subY][0], tmp[subY][1], tmp[subY][2], tmp[subY][3]);
          }
          else if(convXSize == 3 && inTileXSize == 6) {
            for(int subY = 0; subY < convYSize; subY++)
              transform3x3_6(tmp[subY][0], tmp[subY][1], tmp[subY][2], tmp[subY][3], tmp[subY][4], tmp[subY][5]);
          }
          else if(convXSize == 5 && inTileXSize == 6) {
            for(int subY = 0; subY < convYSize; subY++)
              transform5x5_6(tmp[subY][0], tmp[subY][1], tmp[subY][2], tmp[subY][3], tmp[subY][4], tmp[subY][5]);
          }

          if(convYSize == 3 && inTileYSize == 4) {
            for(int subX = 0; subX < inTileXSize; subX++)
              transform3x3_4(tmp[0][subX], tmp[1][subX], tmp[2][subX], tmp[3][subX]);
          }
          else if(convYSize == 3 && inTileYSize == 6) {
            for(int subX = 0; subX < inTileXSize; subX++)
              transform3x3_6(tmp[0][subX], tmp[1][subX], tmp[2][subX], tmp[3][subX], tmp[4][subX], tmp[5][subX]);
          }
          else if(convYSize == 5 && inTileYSize == 6) {
            for(int subX = 0; subX < inTileXSize; subX++)
              transform5x5_6(tmp[0][subX], tmp[1][subX], tmp[2][subX], tmp[3][subX], tmp[4][subX], tmp[5][subX]);
          }

          for(int subY = 0; subY < inTileYSize; subY++) {
            for(int subX = 0; subX < inTileXSize; subX++) {
              transWeights[((subY*inTileXSize + subX)*inChannelsPadded + ic)*outChannelsPadded + oc] = tmp[subY][subX];
            }
          }
        }
      }

      filter = createReadOnlyBuffer(handle,transWeights,useFP16);
    }
    else {
      vector<float> weights = desc->weights;
      filter = createReadOnlyBuffer(handle,weights,useFP16);
    }
  }

  ~ConvLayer() {
    clReleaseMemObject(filter);
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    int numTilesTotalPadded = roundUpToMultipleInt(maxBatchSize * numTilesX * numTilesY, handle->getXGemmMPaddingMult());
    int outChannelsPadded = roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
    int inChannelsPadded = roundUpToMultipleInt(inChannels, handle->getXGemmKPaddingMult());
    return
      ConvWorkspaceEltsNeeded(
        numTilesTotalPadded * inChannelsPadded * inTileXYSize,
        numTilesTotalPadded * outChannelsPadded * inTileXYSize
      );
  }

  void apply(ComputeHandleInternal* handle, int batchSize, cl_mem input, cl_mem output, cl_mem convWorkspace, cl_mem convWorkspace2) const {
    if(convXSize == 1 && convYSize == 1) {
      if(!usingHGemmWmmaNHCW) {
        int filterStride = 0; //Reuse same filter for all matrices in batch
        int inputStride = nnXLen*nnYLen * inChannels;
        int outputStride = nnXLen*nnYLen * outChannels;
        cl_int err;
        MAYBE_EVENT;
        err = doStridedBatchedXGemmDirect_KM_KN_NM(
          handle->xgemmDirectStridedBatchedNNKernel,
          handle->commandQueue,
          handle->tuneParams,
          nnXLen*nnYLen, outChannels, inChannels,
          inputStride, filterStride, outputStride,
          input, filter, output,
          batchSize,
          MAYBE_EVENTREF
        );
        CHECK_ERR(err);
        MAYBE_PROFILE("MATMULCONV1x1");
        MAYBE_FREE_EVENT;
      }
      else {
        cl_int err;
        MAYBE_EVENT;
        err = doHGemmWmma_NCHW_ICOC(
          handle->hgemmWmmaNCHWKernel,
          handle->commandQueue,
          handle->tuneParams,
          batchSize, inChannels, nnXLen*nnYLen, outChannels,
          input, filter, output,
          MAYBE_EVENTREF
        );
        CHECK_ERR(err);
        MAYBE_PROFILE("HGEMM1x1");
        MAYBE_FREE_EVENT;
      }
    }
    else if((convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)) {

      {
        cl_int err;
        MAYBE_EVENT;
        err = doWinogradTransform(
          (convXSize == 3 && convYSize == 3) ?
          handle->winogradConv3x3NCHWTransformKernel :
          handle->winogradConv5x5NCHWTransformKernel,
          handle->commandQueue,
          handle->tuneParams,
          input,convWorkspace,
          nnXLen,nnYLen,
          batchSize,numTilesX,numTilesY,handle->getXGemmMPaddingMult(), //M in gemm
          inChannels,handle->getXGemmKPaddingMult(),                    //K in gemm
          convXSize,
          MAYBE_EVENTREF
        );
        CHECK_ERR(err);
        if(convXSize == 3 && convYSize == 3) { MAYBE_PROFILE("3x3TRANSFORM"); }
        else { MAYBE_PROFILE("5x5TRANSFORM"); }
        MAYBE_FREE_EVENT;
      }

      {
        int numTilesTotalPadded = roundUpToMultipleInt(batchSize * numTilesX * numTilesY, handle->getXGemmMPaddingMult());
        int outChannelsPadded = roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
        int inChannelsPadded = roundUpToMultipleInt(inChannels, handle->getXGemmKPaddingMult());

        cl_int err;
        MAYBE_EVENT;
        if(handle->usingFP16TensorCores) {
          err = doBatchedHGemmWmma_KM_KN_NM(
            handle->xgemmBatchedNNKernel,
            handle->commandQueue,
            handle->tuneParams,
            numTilesTotalPadded, outChannelsPadded, inChannelsPadded,
            convWorkspace, filter, convWorkspace2,
            inTileXYSize,
            MAYBE_EVENTREF
          );
        }
        else {
          err = doBatchedXGemm_KM_KN_NM(
            handle->xgemmBatchedNNKernel,
            handle->commandQueue,
            handle->usingFP16Compute ? handle->tuneParams.xGemm16 : handle->tuneParams.xGemm,
            numTilesTotalPadded, outChannelsPadded, inChannelsPadded,
            convWorkspace, filter, convWorkspace2,
            inTileXYSize,
            MAYBE_EVENTREF
          );
        }
        CHECK_ERR(err);
        if(convXSize == 3 && convYSize == 3) { MAYBE_PROFILE("MATMULCONV3x3"); }
        else { MAYBE_PROFILE("MATMULCONV5x5"); }
        MAYBE_FREE_EVENT;
      }

      {
        cl_int err;
        MAYBE_EVENT;
        err = doWinogradUntransform(
          (convXSize == 3 && convYSize == 3) ?
          handle->winogradConv3x3NCHWUntransformKernel :
          handle->winogradConv5x5NCHWUntransformKernel,
          handle->commandQueue,
          handle->tuneParams,
          convWorkspace2,output,
          nnXLen,nnYLen,
          batchSize,numTilesX,numTilesY,handle->getXGemmMPaddingMult(), //M in gemm
          outChannels,handle->getXGemmNPaddingMult(),                   //N in gemm
          convXSize,
          MAYBE_EVENTREF
        );
        CHECK_ERR(err);
        if(convXSize == 3 && convYSize == 3) { MAYBE_PROFILE("3x3UNTRANSFORM"); }
        else { MAYBE_PROFILE("5x5UNTRANSFORM"); }
        MAYBE_FREE_EVENT;
      }

    }

    else {
      cl_kernel kernel = handle->conv2dNCHWKernel;
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (const void *)&input);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *)&filter);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (const void *)&output);

      //TODO throw this all away and just use winograd entirely
      static const size_t TILE_XSIZE = 32;
      static const size_t TILE_YSIZE = 4;
      static const size_t TILE_CHANNELS = 4;
      const size_t inputTileXSize = TILE_XSIZE + 2*convXRadius;
      const size_t inputTileYSize = TILE_YSIZE + 2*convYRadius;
      clSetKernelArg(kernel, 3, sizeof(float) * TILE_CHANNELS * inputTileXSize * inputTileYSize, NULL);
      clSetKernelArg(kernel, 4, sizeof(float) * TILE_XSIZE * TILE_YSIZE, NULL);
      clSetKernelArg(kernel, 5, sizeof(int), (const void *)&batchSize);
      clSetKernelArg(kernel, 6, sizeof(int), (const void *)&nnXLen);
      clSetKernelArg(kernel, 7, sizeof(int), (const void *)&nnYLen);
      clSetKernelArg(kernel, 8, sizeof(int), (const void *)&outChannels);
      clSetKernelArg(kernel, 9, sizeof(int), (const void *)&inChannels);
      clSetKernelArg(kernel, 10, sizeof(int), (const void *)&convXRadius);
      clSetKernelArg(kernel, 11, sizeof(int), (const void *)&convYRadius);

      static const int workPerThreadX = 1;
      static const int workPerThreadY = 1;
      size_t localSizes[nKernelDims];
      localSizes[0] = TILE_XSIZE / workPerThreadX;
      localSizes[1] = TILE_YSIZE / workPerThreadY;
      localSizes[2] = 1;

      size_t globalSizes[nKernelDims];
      globalSizes[0] = roundUpToMultiple(nnXLen,TILE_XSIZE);
      globalSizes[1] = roundUpToMultiple(nnYLen,TILE_YSIZE);
      globalSizes[2] = outChannels;

      cl_int err;
      MAYBE_EVENT;
      err = clEnqueueNDRangeKernel(
        handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, MAYBE_EVENTREF
      );
      CHECK_ERR(err);
      if(convXRadius == 2 && convYRadius == 2) {
        MAYBE_PROFILE("CONV5");
      }
      else {
        MAYBE_PROFILE("CONV");
      }
      MAYBE_FREE_EVENT;
    }
  }

  bool canApplyWithBNAct() const {
    return (convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5);
  }

  void applyWithBNAct(
    ComputeHandleInternal* handle, const BatchNormLayer* bnLayer, int batchSize,
    cl_mem input, cl_mem output, cl_mem mask, cl_mem convWorkspace, cl_mem convWorkspace2
  ) const {
    if((convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)) {
      {
        assert(bnLayer->activation == ACTIVATION_RELU || bnLayer->activation == ACTIVATION_MISH);
        cl_int err;
        MAYBE_EVENT;
        err = doWinogradTransformWithBNAct(
          (convXSize == 3 && convYSize == 3) ?
          (bnLayer->activation == ACTIVATION_RELU ? handle->winogradConv3x3NCHWBNReluTransformKernel : handle->winogradConv3x3NCHWBNMishTransformKernel) :
          (bnLayer->activation == ACTIVATION_RELU ? handle->winogradConv5x5NCHWBNReluTransformKernel : handle->winogradConv5x5NCHWBNMishTransformKernel),
          handle->commandQueue,
          handle->tuneParams,
          input,convWorkspace,
          bnLayer->mergedScaleBuf,
          bnLayer->mergedBiasBuf,
          mask,
          nnXLen,nnYLen,
          batchSize,numTilesX,numTilesY,handle->getXGemmMPaddingMult(), //M in gemm
          inChannels,handle->getXGemmKPaddingMult(),                    //K in gemm
          convXSize,
          MAYBE_EVENTREF
        );
        CHECK_ERR(err);
        if(convXSize == 3 && convYSize == 3) { MAYBE_PROFILE("3x3TRANSFORMBNACT"); }
        else { MAYBE_PROFILE("5x5TRANSFORMBNACT"); }
        MAYBE_FREE_EVENT;
      }

      {
        int numTilesTotalPadded = roundUpToMultipleInt(batchSize * numTilesX * numTilesY, handle->getXGemmMPaddingMult());
        int outChannelsPadded = roundUpToMultipleInt(outChannels, handle->getXGemmNPaddingMult());
        int inChannelsPadded = roundUpToMultipleInt(inChannels, handle->getXGemmKPaddingMult());

        cl_int err;
        MAYBE_EVENT;
        if(handle->usingFP16TensorCores) {
          err = doBatchedHGemmWmma_KM_KN_NM(
            handle->xgemmBatchedNNKernel,
            handle->commandQueue,
            handle->tuneParams,
            numTilesTotalPadded, outChannelsPadded, inChannelsPadded,
            convWorkspace, filter, convWorkspace2,
            inTileXYSize,
            MAYBE_EVENTREF
          );
        }
        else {
          err = doBatchedXGemm_KM_KN_NM(
            handle->xgemmBatchedNNKernel,
            handle->commandQueue,
            handle->usingFP16Compute ? handle->tuneParams.xGemm16 : handle->tuneParams.xGemm,
            numTilesTotalPadded, outChannelsPadded, inChannelsPadded,
            convWorkspace, filter, convWorkspace2,
            inTileXYSize,
            MAYBE_EVENTREF
          );
        }
        CHECK_ERR(err);
        if(convXSize == 3 && convYSize == 3) { MAYBE_PROFILE("MATMULCONV3x3BNACT"); }
        else { MAYBE_PROFILE("MATMULCONV5x5BNACT"); }
        MAYBE_FREE_EVENT;
      }

      {
        cl_int err;
        MAYBE_EVENT;
        err = doWinogradUntransform(
          (convXSize == 3 && convYSize == 3) ?
          handle->winogradConv3x3NCHWUntransformKernel :
          handle->winogradConv5x5NCHWUntransformKernel,
          handle->commandQueue,
          handle->tuneParams,
          convWorkspace2,output,
          nnXLen,nnYLen,
          batchSize,numTilesX,numTilesY,handle->getXGemmMPaddingMult(), //M in gemm
          outChannels,handle->getXGemmNPaddingMult(),                   //N in gemm
          convXSize,
          MAYBE_EVENTREF
        );
        CHECK_ERR(err);
        if(convXSize == 3 && convYSize == 3) { MAYBE_PROFILE("3x3UNTRANSFORMBNACT"); }
        else { MAYBE_PROFILE("5x5UNTRANSFORMBNACT"); }
        MAYBE_FREE_EVENT;
      }

    }
    else {
      throw StringError("Attempted ConvLayer::applyWithBNAct on non-3x3 or non-5x5 conv, implementation dues not currently support this");
    }
  }

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;
};

//--------------------------------------------------------------

struct MatMulLayer {
  const string name;
  const int inChannels;
  const int outChannels;

  cl_mem matBuf;

  MatMulLayer(ComputeHandleInternal* handle, const MatMulLayerDesc* desc)
    : name(desc->name),
      inChannels(desc->inChannels),
      outChannels(desc->outChannels)
  {
    assert(desc->weights.size() == inChannels * outChannels);
    vector<float> weights(desc->weights.size());
    //Transpose weights, we implemented the opencl kernel to expect oc,ic
    for(int oc = 0; oc < outChannels; oc++) {
      for(int ic = 0; ic < inChannels; ic++) {
        weights[oc * inChannels + ic] = desc->weights[ic * outChannels + oc];
      }
    }
    //See notes about FP16 conventions at the top of file
    bool useFP16 = false;
    matBuf = createReadOnlyBuffer(handle,weights,useFP16);
  }

  ~MatMulLayer() {
    clReleaseMemObject(matBuf);
  }

  void apply(ComputeHandleInternal* handle, int batchSize, cl_mem input, cl_mem output) const {
    MAYBE_EVENT;
    cl_int err = doBatchedXGemmDirect_MK_NK_MN(
      handle->xgemmDirectBatchedTTKernelAlwaysFP32,
      handle->commandQueue,
      handle->tuneParams,
      batchSize, outChannels, inChannels,
      input, matBuf, output,
      1,
      MAYBE_EVENTREF

    );
    CHECK_ERR(err);
    MAYBE_PROFILE("PLAINMATMUL");
    MAYBE_FREE_EVENT;
  }

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;
};

//--------------------------------------------------------------

struct MatBiasLayer {
  const string name;
  const int numChannels;
  const int activation;

  cl_mem biasBuf;

  MatBiasLayer(ComputeHandleInternal* handle, const MatBiasLayerDesc* desc, int activation_)
    : name(desc->name),
      numChannels(desc->numChannels),
      activation(activation_)
  {
    assert(desc->weights.size() == numChannels);
    vector<float> weights = desc->weights;
    //See notes about FP16 conventions at the top of file
    bool useFP16 = false;
    biasBuf = createReadOnlyBuffer(handle,weights,useFP16);
  }

  ~MatBiasLayer() {
    clReleaseMemObject(biasBuf);
  }

  void apply(ComputeHandleInternal* handle, int batchSize, cl_mem input) const {
    cl_kernel kernel;
    if(activation == ACTIVATION_IDENTITY)
      kernel = handle->addCBiasesNCKernel;
    else if(activation == ACTIVATION_RELU)
      kernel = handle->addCBiasesNCReluKernel;
    else if(activation == ACTIVATION_MISH)
      kernel = handle->addCBiasesNCMishKernel;
    else
      assert(false);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), (const void *)&input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *)&biasBuf);
    clSetKernelArg(kernel, 2, sizeof(int), (const void *)&batchSize);
    clSetKernelArg(kernel, 3, sizeof(int), (const void *)&numChannels);

    cl_int err;
    static constexpr int nKernelDims = 2;
    size_t globalSizes[nKernelDims] = {powerOf2ify((size_t)numChannels), powerOf2ify((size_t)batchSize)};
    size_t* localSizes = NULL;
    MAYBE_EVENT;
    err = clEnqueueNDRangeKernel(
      handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, MAYBE_EVENTREF
    );
    CHECK_ERR(err);
    MAYBE_PROFILE("MatBias");
    MAYBE_FREE_EVENT;
  }

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;
};

//--------------------------------------------------------------

struct NormActConv {
  const BatchNormLayer norm;
  const ConvLayer conv;
  const int inChannels;
  const int outChannels;

  NormActConv(
    ComputeHandleInternal* handle,
    const BatchNormLayerDesc* normDesc,
    const ActivationLayerDesc* actDesc,
    const ConvLayerDesc* convDesc,
    int nnX,
    int nnY,
    bool useFP16
  ) :
    norm(handle,normDesc,actDesc,nnX,nnY,useFP16),
    conv(handle,convDesc,nnX,nnY,useFP16),
    inChannels(norm.numChannels),
    outChannels(conv.outChannels)
  {
    assert(norm.numChannels == conv.inChannels);
  }

  ~NormActConv() {
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    return conv.requiredConvWorkspaceElts(handle,maxBatchSize);
  }

  void apply(
    ComputeHandleInternal* handle,
    int batchSize,
    cl_mem input,
    cl_mem inputScratchOrInput, //It's okay if this is the same as input, if it's okay to mutate input.
    cl_mem output,
    cl_mem mask,
    cl_mem convWorkspace,
    cl_mem convWorkspace2
  ) const {
    if(conv.canApplyWithBNAct())
      conv.applyWithBNAct(handle,&norm,batchSize,input,output,mask,convWorkspace,convWorkspace2);
    else {
      norm.apply(handle,batchSize,input,inputScratchOrInput,mask);
      conv.apply(handle,batchSize,inputScratchOrInput,output,convWorkspace,convWorkspace2);
    }
  }

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;

};


//--------------------------------------------------------------

struct ResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const NormActConv normActConv2;
  const int nnXLen;
  const int nnYLen;

  ResidualBlock(
    ComputeHandleInternal* handle,
    const ResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16
  ) :
    name(desc->name),
    normActConv1(handle,&desc->preBN,&desc->preActivation,&desc->regularConv,nnX,nnY,useFP16),
    normActConv2(handle,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16),
    nnXLen(nnX),
    nnYLen(nnY)
  {
  }

  ~ResidualBlock() {
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    return ConvWorkspaceEltsNeeded::getMax(
      normActConv1.requiredConvWorkspaceElts(handle,maxBatchSize),
      normActConv2.requiredConvWorkspaceElts(handle,maxBatchSize)
    );
  }

  void apply(
    ComputeHandleInternal* handle,
    ScratchBuffers* scratch,
    int batchSize,
    cl_mem trunk,
    cl_mem trunkScratch,
    cl_mem mask,
    cl_mem convWorkspace,
    cl_mem convWorkspace2
  ) const {
    SizedBuf<cl_mem> mid(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    normActConv1.apply(handle,batchSize,trunk,trunkScratch,mid.buf,mask,convWorkspace,convWorkspace2);
    normActConv2.apply(handle,batchSize,mid.buf,mid.buf,trunkScratch,mask,convWorkspace,convWorkspace2);
    addPointWise(handle, trunk, trunkScratch, batchSize * normActConv2.outChannels * nnYLen * nnXLen);
  }

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

};

//--------------------------------------------------------------

struct GlobalPoolingResidualBlock {
  const string name;
  const BatchNormLayer preBN;
  const ConvLayer regularConv;
  const ConvLayer gpoolConv;
  const BatchNormLayer gpoolBN;
  const MatMulLayer gpoolToBiasMul;
  const NormActConv normActConv2;

  const int nnXLen;
  const int nnYLen;
  const int nnXYLen;
  const int regularChannels;
  const int gpoolChannels;

  GlobalPoolingResidualBlock(
    ComputeHandleInternal* handle,
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16
  ) :
    name(desc->name),
    preBN(handle,&desc->preBN,&desc->preActivation,nnX,nnY,useFP16),
    regularConv(handle,&desc->regularConv,nnX,nnY,useFP16),
    gpoolConv(handle,&desc->gpoolConv,nnX,nnY,useFP16),
    gpoolBN(handle,&desc->gpoolBN,&desc->gpoolActivation,nnX,nnY,useFP16),
    gpoolToBiasMul(handle,&desc->gpoolToBiasMul),
    normActConv2(handle,&desc->midBN,&desc->midActivation,&desc->finalConv,nnX,nnY,useFP16),
    nnXLen(nnX),
    nnYLen(nnY),
    nnXYLen(nnX*nnY),
    regularChannels(desc->regularConv.outChannels),
    gpoolChannels(desc->gpoolConv.outChannels)
  {
  }

  ~GlobalPoolingResidualBlock() {
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,regularConv.requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,gpoolConv.requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,normActConv2.requiredConvWorkspaceElts(handle,maxBatchSize));
    return maxElts;
  }

  void apply(
    ComputeHandleInternal* handle,
    ScratchBuffers* scratch,
    int batchSize,
    cl_mem trunk,
    cl_mem trunkScratch,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem convWorkspace,
    cl_mem convWorkspace2
  ) const {
    SizedBuf<cl_mem> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<cl_mem> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<cl_mem> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(gpoolChannels*3));
    SizedBuf<cl_mem> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(regularChannels));

    preBN.apply(handle,batchSize,trunk,trunkScratch,mask);
    regularConv.apply(handle,batchSize,trunkScratch,regularOut.buf,convWorkspace,convWorkspace2);
    gpoolConv.apply(handle,batchSize,trunkScratch,gpoolOut.buf,convWorkspace,convWorkspace2);
    gpoolBN.apply(handle,batchSize,gpoolOut.buf,gpoolOut.buf,mask);

    performGPoolMask(handle, batchSize, gpoolChannels, nnXYLen, gpoolOut.buf, gpoolConcat.buf, mask, maskSum);

    gpoolToBiasMul.apply(handle,batchSize,gpoolConcat.buf,gpoolBias.buf);
    addChannelBiases(handle, regularOut.buf, gpoolBias.buf, batchSize * regularChannels, nnXYLen);

    // vector<float> tmp(batchSize*regularChannels);
    // clEnqueueReadBuffer(handle->commandQueue, gpoolBias, CL_TRUE, 0, byteSizeofVectorContents(tmp), tmp.data(), 0, NULL, NULL);
    // cout << "TEST" << endl;
    // for(int i = 0; i<tmp.size(); i++)
    //   cout << tmp[i] << endl;

    normActConv2.apply(handle,batchSize,regularOut.buf,regularOut.buf,trunkScratch,mask,convWorkspace,convWorkspace2);
    addPointWise(handle, trunk, trunkScratch, batchSize * normActConv2.outChannels * nnYLen * nnXLen);
  }

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

};

//--------------------------------------------------------------

struct BlockStack {
  const int numBlocks;
  const int trunkNumChannels;
  const int nnXLen;
  const int nnYLen;
  vector<pair<int,unique_ptr_void>> blocks;

  BlockStack() = delete;
  BlockStack(const BlockStack&) = delete;
  BlockStack& operator=(const BlockStack&) = delete;

  BlockStack(
    ComputeHandleInternal* handle,
    const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
    int nBlocks,
    int trunkChannels,
    int nnX,
    int nnY,
    bool useFP16
  );
  ~BlockStack();

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const;

  void apply(
    ComputeHandleInternal* handle,
    ScratchBuffers* scratch,
    int batchSize,
    cl_mem trunk,
    cl_mem trunkScratch,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem convWorkspace,
    cl_mem convWorkspace2
  ) const;

};

//--------------------------------------------------------------

struct NestedBottleneckResidualBlock {
  const string name;
  const NormActConv normActConv1;
  const BlockStack blocks;
  const NormActConv normActConv2;
  const int nnXLen;
  const int nnYLen;

  NestedBottleneckResidualBlock(
    ComputeHandleInternal* handle,
    const NestedBottleneckResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16
  ) :
    name(desc->name),
    normActConv1(handle,&desc->preBN,&desc->preActivation,&desc->preConv,nnX,nnY,useFP16),
    blocks(handle,desc->blocks,desc->numBlocks,desc->preConv.outChannels,nnX,nnY,useFP16),
    normActConv2(handle,&desc->postBN,&desc->postActivation,&desc->postConv,nnX,nnY,useFP16),
    nnXLen(nnX),
    nnYLen(nnY)
  {
  }

  ~NestedBottleneckResidualBlock() {
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    return ConvWorkspaceEltsNeeded::getMax(
      normActConv1.requiredConvWorkspaceElts(handle,maxBatchSize),
      ConvWorkspaceEltsNeeded::getMax(
        blocks.requiredConvWorkspaceElts(handle,maxBatchSize),
        normActConv2.requiredConvWorkspaceElts(handle,maxBatchSize)
      )
    );
  }

  void apply(
    ComputeHandleInternal* handle,
    ScratchBuffers* scratch,
    int batchSize,
    cl_mem trunk,
    cl_mem trunkScratch,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem convWorkspace,
    cl_mem convWorkspace2
  ) const {
    SizedBuf<cl_mem> mid(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<cl_mem> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    normActConv1.apply(handle,batchSize,trunk,trunkScratch,mid.buf,mask,convWorkspace,convWorkspace2);
    blocks.apply(handle,scratch,batchSize,mid.buf,midScratch.buf,mask,maskSum,convWorkspace,convWorkspace2);
    normActConv2.apply(handle,batchSize,mid.buf,mid.buf,trunkScratch,mask,convWorkspace,convWorkspace2);
    addPointWise(handle, trunk, trunkScratch, batchSize * normActConv2.outChannels * nnYLen * nnXLen);
  }

  NestedBottleneckResidualBlock() = delete;
  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlock&) = delete;
  NestedBottleneckResidualBlock& operator=(const NestedBottleneckResidualBlock&) = delete;

};

//--------------------------------------------------------------

BlockStack::BlockStack(
  ComputeHandleInternal* handle,
  const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
  int nBlocks,
  int trunkChannels,
  int nnX,
  int nnY,
  bool useFP16
) :
  numBlocks(nBlocks),
  trunkNumChannels(trunkChannels),
  nnXLen(nnX),
  nnYLen(nnY)
{
  assert(descBlocks.size() == numBlocks);
  for(int i = 0; i<numBlocks; i++) {
    if(descBlocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new ResidualBlock(
          handle,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16
        )
      );
      blocks.push_back(make_pair(ORDINARY_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new GlobalPoolingResidualBlock(
          handle,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16
        )
      );
      blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* blockDesc = (NestedBottleneckResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new NestedBottleneckResidualBlock(
          handle,
          blockDesc,
          nnXLen,
          nnYLen,
          useFP16
        )
      );
      blocks.push_back(make_pair(NESTED_BOTTLENECK_BLOCK_KIND,std::move(blockPtr)));
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}

BlockStack::~BlockStack() {
}

ConvWorkspaceEltsNeeded BlockStack::requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
  ConvWorkspaceEltsNeeded maxElts;

  for(int i = 0; i<blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,block->requiredConvWorkspaceElts(handle,maxBatchSize));
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,block->requiredConvWorkspaceElts(handle,maxBatchSize));
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,block->requiredConvWorkspaceElts(handle,maxBatchSize));
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  return maxElts;
}

void BlockStack::apply(
  ComputeHandleInternal* handle,
  ScratchBuffers* scratch,
  int batchSize,
  cl_mem trunk,
  cl_mem trunkScratch,
  cl_mem mask,
  cl_mem maskSum,
  cl_mem convWorkspace,
  cl_mem convWorkspace2
) const {
  for(int i = 0; i<blocks.size(); i++) {
#ifdef DEBUG_INTERMEDIATE_VALUES
    bool usingNHWC = false;
    debugPrint4D(string("Blockstack before block " + Global::intToString(i)), handle, trunkScratch, batchSize, trunkNumChannels, nnXLen, nnYLen, usingNHWC);
#endif

    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      block->apply(
        handle,
        scratch,
        batchSize,
        trunk,
        trunkScratch,
        mask,
        convWorkspace,
        convWorkspace2
      );
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      block->apply(
        handle,
        scratch,
        batchSize,
        trunk,
        trunkScratch,
        mask,
        maskSum,
        convWorkspace,
        convWorkspace2
      );
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      block->apply(
        handle,
        scratch,
        batchSize,
        trunk,
        trunkScratch,
        mask,
        maskSum,
        convWorkspace,
        convWorkspace2
      );
    }
    else {
      ASSERT_UNREACHABLE;
    }

  }
}


//--------------------------------------------------------------

struct Trunk {
  const string name;
  const int version;
  const int trunkNumChannels;
  const int midNumChannels;
  const int regularNumChannels;
  const int gpoolNumChannels;

  const int nnXLen;
  const int nnYLen;

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatMulLayer> initialMatMul;
  const BlockStack blocks;
  std::unique_ptr<BatchNormLayer> trunkTipBN;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    ComputeHandleInternal* handle,
    const TrunkDesc* desc,
    int maxBatchSize,
    int nnX,
    int nnY,
    bool useFP16
  ) :
    name(desc->name),
    version(desc->version),
    trunkNumChannels(desc->trunkNumChannels),
    midNumChannels(desc->midNumChannels),
    regularNumChannels(desc->regularNumChannels),
    gpoolNumChannels(desc->gpoolNumChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    blocks(handle,desc->blocks,desc->numBlocks,desc->trunkNumChannels,nnX,nnY,useFP16)
  {
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,trunkNumChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,midNumChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,regularNumChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,gpoolNumChannels);

    initialConv = std::make_unique<ConvLayer>(handle,&desc->initialConv,nnXLen,nnYLen,useFP16);
    initialMatMul = std::make_unique<MatMulLayer>(handle,&desc->initialMatMul);
    trunkTipBN = std::make_unique<BatchNormLayer>(handle,&desc->trunkTipBN,&desc->trunkTipActivation,nnXLen,nnYLen,useFP16);
  }

  ~Trunk() {
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    return ConvWorkspaceEltsNeeded::getMax(
      initialConv->requiredConvWorkspaceElts(handle,maxBatchSize),
      blocks.requiredConvWorkspaceElts(handle,maxBatchSize)
    );
  }

  void apply(
    ComputeHandleInternal* handle,
    ScratchBuffers* scratch,
    int batchSize,
    cl_mem input,
    cl_mem inputGlobal,
    cl_mem trunk,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem convWorkspace,
    cl_mem convWorkspace2
  ) const {

    SizedBuf<cl_mem> trunkScratch(scratch->allocator, scratch->getBufSizeXY(trunkNumChannels));

    initialConv->apply(handle,batchSize,input,trunk,convWorkspace,convWorkspace2);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    bool usingNHWC = false;
    debugPrint4D(string("Initial bin features"), handle, input, batchSize, initialConv->inChannels, nnXLen, nnYLen, usingNHWC);
    debugPrint4D(string("After initial conv"), handle, trunk, batchSize, trunkNumChannels, nnXLen, nnYLen, usingNHWC);
    #endif

    //Feed the matmul into trunkScratch, which will certainly be a big enough buffer
    initialMatMul->apply(handle,batchSize,inputGlobal,trunkScratch.buf);
    //Then accumulate it into trunk, broadcasting during the process
    addChannelBiases(handle, trunk, trunkScratch.buf, batchSize * trunkNumChannels, nnXLen*nnYLen);

    blocks.apply(handle,scratch,batchSize,trunk,trunkScratch.buf,mask,maskSum,convWorkspace,convWorkspace2);
    trunkTipBN->apply(handle,batchSize,trunk,trunk,mask);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    debugPrint4D(string("Trunk tip"), handle, trunk, batchSize, trunkNumChannels, nnXLen, nnYLen, usingNHWC);
    #endif
  }

};

//--------------------------------------------------------------

struct PolicyHead {
  const string name;
  const int version;
  const int nnXLen;
  const int nnYLen;
  const int p1Channels;
  const int g1Channels;
  const int p2Channels;

  std::unique_ptr<ConvLayer> p1Conv;
  std::unique_ptr<ConvLayer> g1Conv;
  std::unique_ptr<BatchNormLayer> g1BN;
  std::unique_ptr<MatMulLayer> gpoolToBiasMul;
  std::unique_ptr<BatchNormLayer> p1BN;
  std::unique_ptr<ConvLayer> p2Conv;
  std::unique_ptr<MatMulLayer> gpoolToPassMul;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    ComputeHandleInternal* handle,
    const PolicyHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16
  ) :
    name(desc->name),
    version(desc->version),
    nnXLen(nnX),
    nnYLen(nnY),
    p1Channels(desc->p1Conv.outChannels),
    g1Channels(desc->g1Conv.outChannels),
    p2Channels(desc->p2Conv.outChannels)
  {
    p1Conv = std::make_unique<ConvLayer>(handle,&desc->p1Conv,nnXLen,nnYLen,useFP16);
    g1Conv = std::make_unique<ConvLayer>(handle,&desc->g1Conv,nnXLen,nnYLen,useFP16);
    g1BN = std::make_unique<BatchNormLayer>(handle,&desc->g1BN,&desc->g1Activation,nnXLen,nnYLen,useFP16);
    gpoolToBiasMul = std::make_unique<MatMulLayer>(handle,&desc->gpoolToBiasMul);
    p1BN = std::make_unique<BatchNormLayer>(handle,&desc->p1BN,&desc->p1Activation,nnXLen,nnYLen,useFP16);
    p2Conv = std::make_unique<ConvLayer>(handle,&desc->p2Conv,nnXLen,nnYLen,useFP16);
    gpoolToPassMul = std::make_unique<MatMulLayer>(handle,&desc->gpoolToPassMul);
  }

  ~PolicyHead() {
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,p1Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,g1Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,p2Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    return maxElts;
  }

  void apply(
    ComputeHandleInternal* handle,
    ScratchBuffers* scratch,
    int batchSize,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem trunk,
    cl_mem policyPass,
    cl_mem policy,
    cl_mem convWorkspace,
    cl_mem convWorkspace2
  ) const {
    SizedBuf<cl_mem> p1Out(scratch->allocator, scratch->getBufSizeXY(p1Channels));
    SizedBuf<cl_mem> gpoolOut(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<cl_mem> gpoolConcat(scratch->allocator, scratch->getBufSizeFloat(g1Channels*3));
    SizedBuf<cl_mem> gpoolBias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));

    p1Conv->apply(handle,batchSize,trunk,p1Out.buf,convWorkspace,convWorkspace2);
    g1Conv->apply(handle,batchSize,trunk,gpoolOut.buf,convWorkspace,convWorkspace2);
    g1BN->apply(handle,batchSize,gpoolOut.buf,gpoolOut.buf,mask);

    performGPoolMask(handle, batchSize, g1Channels, nnXLen*nnYLen, gpoolOut.buf, gpoolConcat.buf, mask, maskSum);

    gpoolToBiasMul->apply(handle,batchSize,gpoolConcat.buf,gpoolBias.buf);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    bool usingNHWC = false;
    debugPrint4D(string("p1 pre-gpool-sum"), handle, p1Out.buf, batchSize, p1Channels, nnXLen, nnYLen, usingNHWC);
    debugPrint4D(string("g1 pre-gpool"), handle, gpoolOut.buf, batchSize, g1Channels, nnXLen, nnYLen, usingNHWC);
    debugPrint2D(string("g1 pooled"), handle, gpoolConcat.buf, batchSize, g1Channels*3);
    debugPrint2D(string("g1 biases"), handle, gpoolBias.buf, batchSize, p1Channels);
    #endif

    addChannelBiases(handle, p1Out.buf, gpoolBias.buf, batchSize * p1Channels, nnXLen*nnYLen);

    p1BN->apply(handle,batchSize,p1Out.buf,p1Out.buf,mask);
    p2Conv->apply(handle,batchSize,p1Out.buf,policy,convWorkspace,convWorkspace2);
    gpoolToPassMul->apply(handle,batchSize,gpoolConcat.buf,policyPass);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    debugPrint4D(string("p1 after-gpool-sum"), handle, p1Out.buf, batchSize, p1Channels, nnXLen, nnYLen, usingNHWC);
    debugPrint4D(string("p2"), handle, policy, batchSize, p2Channels, nnXLen, nnYLen, usingNHWC);
    debugPrint2D(string("p2pass"), handle, policyPass, batchSize, 1);
    #endif
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

  std::unique_ptr<ConvLayer> v1Conv;
  std::unique_ptr<BatchNormLayer> v1BN;
  std::unique_ptr<MatMulLayer> v2Mul;
  std::unique_ptr<MatBiasLayer> v2Bias;
  std::unique_ptr<MatMulLayer> v3Mul;
  std::unique_ptr<MatBiasLayer> v3Bias;
  std::unique_ptr<MatMulLayer> sv3Mul;
  std::unique_ptr<MatBiasLayer> sv3Bias;
  std::unique_ptr<ConvLayer> vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    ComputeHandleInternal* handle,
    const ValueHeadDesc* desc,
    int nnX,
    int nnY,
    bool useFP16
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

    v1Conv = std::make_unique<ConvLayer>(handle,&desc->v1Conv,nnXLen,nnYLen,useFP16);
    v1BN = std::make_unique<BatchNormLayer>(handle,&desc->v1BN,&desc->v1Activation,nnXLen,nnYLen,useFP16);
    v2Mul = std::make_unique<MatMulLayer>(handle,&desc->v2Mul);
    v2Bias = std::make_unique<MatBiasLayer>(handle,&desc->v2Bias,desc->v2Activation.activation);
    v3Mul = std::make_unique<MatMulLayer>(handle,&desc->v3Mul);
    v3Bias = std::make_unique<MatBiasLayer>(handle,&desc->v3Bias,ACTIVATION_IDENTITY);
    sv3Mul = std::make_unique<MatMulLayer>(handle,&desc->sv3Mul);
    sv3Bias = std::make_unique<MatBiasLayer>(handle,&desc->sv3Bias,ACTIVATION_IDENTITY);
    vOwnershipConv = std::make_unique<ConvLayer>(handle,&desc->vOwnershipConv,nnXLen,nnYLen,useFP16);
  }

  ~ValueHead() {
  }

  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle, size_t maxBatchSize) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,v1Conv->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,vOwnershipConv->requiredConvWorkspaceElts(handle,maxBatchSize));
    return maxElts;
  }

  void apply(
    ComputeHandleInternal* handle,
    ScratchBuffers* scratch,
    int batchSize,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem trunk,
    cl_mem value,
    cl_mem scoreValue,
    cl_mem ownership,
    cl_mem convWorkspace,
    cl_mem convWorkspace2
  ) const {
    SizedBuf<cl_mem> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<cl_mem> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<cl_mem> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));

    v1Conv->apply(handle,batchSize,trunk,v1Out.buf,convWorkspace,convWorkspace2);
    v1BN->apply(handle,batchSize,v1Out.buf,v1Out.buf,mask);

    performValueHeadPool(handle, batchSize, v1Channels, nnXLen*nnYLen, v1Out.buf, v1Mean.buf, maskSum);

    v2Mul->apply(handle,batchSize,v1Mean.buf,v2Out.buf);
    v2Bias->apply(handle,batchSize,v2Out.buf);
    v3Mul->apply(handle,batchSize,v2Out.buf,value);
    v3Bias->apply(handle,batchSize,value);

    sv3Mul->apply(handle,batchSize,v2Out.buf,scoreValue);
    sv3Bias->apply(handle,batchSize,scoreValue);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    bool usingNHWC = false;
    debugPrint4D(string("v1"), handle, v1Out.buf, batchSize, v1Channels, nnXLen, nnYLen, usingNHWC);
    debugPrint2D(string("v1 pooled"), handle, v1Mean.buf, batchSize, v1Channels);
    debugPrint2D(string("v2"), handle, v2Out.buf, batchSize, v1Channels);
    #endif

    vOwnershipConv->apply(handle,batchSize,v1Out.buf,ownership,convWorkspace,convWorkspace2);
  }

};

//--------------------------------------------------------------

static void computeMaskSums(
  ComputeHandleInternal* handle,
  cl_mem mask,
  cl_mem maskSum,
  int batchSize,
  int nnXLen,
  int nnYLen
) {
  cl_int err;
  MAYBE_EVENT;
  err = OpenCLHelpers::computeMaskSums(
    handle->sumChannelsNCHWKernel,
    handle->commandQueue,
    handle->tuneParams,
    mask,
    maskSum,
    batchSize,
    nnXLen,
    nnYLen,
    MAYBE_EVENTREF
  );
  CHECK_ERR(err);
  MAYBE_PROFILE("MaskSums");
  MAYBE_FREE_EVENT;
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

  std::unique_ptr<Trunk> trunk;
  std::unique_ptr<PolicyHead> policyHead;
  std::unique_ptr<ValueHead> valueHead;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    ComputeHandleInternal* handle,
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

    bool useFP16 = handle->usingFP16Storage;
    trunk = std::make_unique<Trunk>(handle,&desc->trunk,maxBatchSize,nnXLen,nnYLen,useFP16);
    policyHead = std::make_unique<PolicyHead>(handle,&desc->policyHead,nnXLen,nnYLen,useFP16);
    valueHead = std::make_unique<ValueHead>(handle,&desc->valueHead,nnXLen,nnYLen,useFP16);
  }

  ~Model() {
  }


  ConvWorkspaceEltsNeeded requiredConvWorkspaceElts(ComputeHandleInternal* handle) const {
    ConvWorkspaceEltsNeeded maxElts;
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,trunk->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,policyHead->requiredConvWorkspaceElts(handle,maxBatchSize));
    maxElts = ConvWorkspaceEltsNeeded::getMax(maxElts,valueHead->requiredConvWorkspaceElts(handle,maxBatchSize));
    return maxElts;
  }


  void apply(
    ComputeHandleInternal* handle,
    ScratchBuffers* scratch,
    int batchSize,

    cl_mem input,
    cl_mem inputGlobal,
    cl_mem mask,
    cl_mem maskSum,
    cl_mem trunkBuf,

    cl_mem policyPass,
    cl_mem policy,

    cl_mem value,
    cl_mem scoreValue,
    cl_mem ownership,

    cl_mem convWorkspace,
    cl_mem convWorkspace2
  ) const {

    {
      cl_kernel kernel = handle->extractChannel0NCHWKernel;
      int nnXYLen = nnXLen * nnYLen;
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (const void *)&input);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (const void *)&mask);
      clSetKernelArg(kernel, 2, sizeof(int), (const void *)&batchSize);
      clSetKernelArg(kernel, 3, sizeof(int), (const void *)&numInputChannels);
      clSetKernelArg(kernel, 4, sizeof(int), (const void *)&nnXYLen);

      cl_int err;
      static constexpr int nKernelDims = 2;
      size_t globalSizes[nKernelDims] = {powerOf2ify((size_t)nnXYLen), powerOf2ify((size_t)batchSize)};
      size_t* localSizes = NULL;
      MAYBE_EVENT;
      err = clEnqueueNDRangeKernel(
        handle->commandQueue, kernel, nKernelDims, NULL, globalSizes, localSizes, 0, NULL, MAYBE_EVENTREF
      );
      CHECK_ERR(err);
      MAYBE_PROFILE("ExtractMask");
      MAYBE_FREE_EVENT;
    }

    computeMaskSums(handle,mask,maskSum,batchSize,nnXLen,nnYLen);

    trunk->apply(
      handle,
      scratch,
      batchSize,
      input,
      inputGlobal,
      trunkBuf,
      mask,
      maskSum,
      convWorkspace,
      convWorkspace2
    );
    policyHead->apply(
      handle,
      scratch,
      batchSize,
      mask,
      maskSum,
      trunkBuf,
      policyPass,
      policy,
      convWorkspace,
      convWorkspace2
    );
    valueHead->apply(
      handle,
      scratch,
      batchSize,
      mask,
      maskSum,
      trunkBuf,
      value,
      scoreValue,
      ownership,
      convWorkspace,
      convWorkspace2
    );
  }

};

//--------------------------------------------------------------

struct Buffers {
  cl_mem input;
  cl_mem inputGlobal;
  size_t inputElts;
  size_t inputGlobalElts;

  cl_mem mask;
  cl_mem maskSum;

  cl_mem trunk;

  cl_mem policyPass;
  cl_mem policy;
  size_t policyPassElts;
  size_t policyElts;

  cl_mem value;
  size_t valueElts;
  cl_mem scoreValue;
  size_t scoreValueElts;
  cl_mem ownership;
  size_t ownershipElts;

  cl_mem convWorkspace;
  cl_mem convWorkspace2;

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(ComputeHandleInternal* handle, const Model& m) {
    size_t batchXYElts = (size_t)m.maxBatchSize * m.nnXLen * m.nnYLen;
    size_t batchElts = (size_t)m.maxBatchSize;

    bool useFP16 = handle->usingFP16Storage;

    inputElts = m.numInputChannels * batchXYElts;
    inputGlobalElts = m.numInputGlobalChannels * batchElts;

    input = createReadWriteBuffer(handle, inputElts, useFP16);
    inputGlobal = createReadWriteBuffer(handle, inputGlobalElts, false);

    mask = createReadWriteBuffer(handle, batchXYElts, useFP16);
    maskSum = createReadWriteBuffer(handle, batchElts, false);

    trunk = createReadWriteBuffer(handle, m.trunk->trunkNumChannels * batchXYElts, useFP16);

    policyPassElts = m.policyHead->p2Channels * batchElts;
    policyPass = createReadWriteBuffer(handle, policyPassElts, false);
    policyElts = m.policyHead->p2Channels * batchXYElts;
    policy = createReadWriteBuffer(handle, policyElts, useFP16);
    assert(m.policyHead->p2Channels == 1);

    valueElts = m.valueHead->valueChannels * batchElts;
    value = createReadWriteBuffer(handle, valueElts, false);

    scoreValueElts = m.valueHead->scoreValueChannels * batchElts;
    scoreValue = createReadWriteBuffer(handle, scoreValueElts, false);

    ownershipElts = m.valueHead->ownershipChannels * batchXYElts;
    ownership = createReadWriteBuffer(handle, ownershipElts, useFP16);

    ConvWorkspaceEltsNeeded convWorkspaceElts = m.requiredConvWorkspaceElts(handle);
    convWorkspace = createReadWriteBuffer(handle, convWorkspaceElts.size1, useFP16);
    convWorkspace2 = createReadWriteBuffer(handle, convWorkspaceElts.size2, useFP16);
  }

  ~Buffers() {
    clReleaseMemObject(input);
    clReleaseMemObject(inputGlobal);

    clReleaseMemObject(mask);
    clReleaseMemObject(maskSum);

    clReleaseMemObject(trunk);

    clReleaseMemObject(policyPass);
    clReleaseMemObject(policy);

    clReleaseMemObject(value);
    clReleaseMemObject(scoreValue);
    clReleaseMemObject(ownership);

    clReleaseMemObject(convWorkspace);
    clReleaseMemObject(convWorkspace2);

  }

};



//--------------------------------------------------------------

struct ComputeHandle {
  std::unique_ptr<ComputeHandleInternal> handle;
  std::unique_ptr<Model> model;
  std::unique_ptr<ScratchBuffers> scratch;
  std::unique_ptr<Buffers> buffers;
  const int nnXLen;
  const int nnYLen;
  const int policySize;
  const bool inputsUseNHWC;

  ComputeHandle(
    ComputeContext* context, const LoadedModel* loadedModel, int maxBatchSize, int gpuIdx, bool inputsUseNHWC_
  ):
    nnXLen(context->nnXLen),
    nnYLen(context->nnYLen),
    policySize(NNPos::getPolicySize(nnXLen, nnYLen)),
    inputsUseNHWC(inputsUseNHWC_)
  {
    bool useNHWC = context->usingNHWCMode == enabled_t::True ? true : false;
    handle = std::make_unique<ComputeHandleInternal>(context, gpuIdx, inputsUseNHWC, useNHWC);
    model = std::make_unique<Model>(handle.get(), &(loadedModel->modelDesc), maxBatchSize, nnXLen, nnYLen);
    scratch = std::make_unique<ScratchBuffers>(handle.get(), maxBatchSize, nnXLen, nnYLen);
    buffers = std::make_unique<Buffers>(handle.get(), *model);
  }

  ~ComputeHandle() {
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
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  auto deviceStr = [&]() {
    if(gpuIdxForThisThread < 0)
      return string("");
    return " Device " + Global::intToString(gpuIdxForThisThread);
  };

  if(logger != NULL) {
    logger->write("OpenCL backend thread " + Global::intToString(serverThreadIdx) + ":" + deviceStr() + " Model version " + Global::intToString(loadedModel->modelDesc.version));
    logger->write("OpenCL backend thread " + Global::intToString(serverThreadIdx) + ":" + deviceStr() + " Model name: " + loadedModel->modelDesc.name);
  }

  //Current implementation always tolerates excess nn len
  (void)requireExactNNLen;
  ComputeHandle* handle = new ComputeHandle(context,loadedModel,maxBatchSize,gpuIdxForThisThread,inputsUseNHWC);

  if(logger != NULL) {
    logger->write(
      "OpenCL backend thread " + Global::intToString(serverThreadIdx) + ":" + deviceStr() +
      " FP16Storage " + Global::boolToString(handle->handle->usingFP16Storage) +
      " FP16Compute " + Global::boolToString(handle->handle->usingFP16Compute) +
      " FP16TensorCores " + Global::boolToString(handle->handle->usingFP16TensorCores) +
      " FP16TensorCoresFor1x1 " + Global::boolToString(handle->handle->usingFP16TensorCoresFor1x1)
    );
  }
  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  return (
    handle->handle->usingFP16Storage ||
    handle->handle->usingFP16Compute ||
    handle->handle->usingFP16TensorCores ||
    handle->handle->usingFP16TensorCoresFor1x1
  );
}

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  vector<DeviceInfo> devices = DeviceInfo::getAllDeviceInfosOnSystem(NULL);
  for(int i = 0; i<devices.size(); i++) {
    const DeviceInfo& device = devices[i];
    string msg =
      "Found OpenCL Device " + Global::intToString(device.gpuIdx) + ": " + device.name + " (" + device.vendor + ")" +
      " (score " + Global::intToString(device.defaultDesirability) + ")";
    cout << msg << endl;
  }
}

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  size_t userInputBufferElts;
  size_t userInputGlobalBufferElts;
  size_t policyPassResultBufferElts;
  size_t policyResultBufferElts;
  size_t valueResultBufferElts;
  size_t scoreValueResultBufferElts;
  size_t ownershipResultBufferElts;

  float* userInputBuffer; //Host pointer
  half_t* userInputBufferHalf; //Host pointer
  float* userInputGlobalBuffer; //Host pointer

  float* policyPassResults; //Host pointer
  float* policyResults; //Host pointer
  half_t* policyResultsHalf; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer
  half_t* ownershipResultsHalf; //Host pointer

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singlePolicyPassResultElts = (size_t)(1);
    singlePolicyResultElts = (size_t)(nnXLen * nnYLen);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    userInputBufferElts = (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen;
    userInputGlobalBufferElts = (size_t)m.numInputGlobalChannels * maxBatchSize;
    policyPassResultBufferElts = (size_t)maxBatchSize * (1);
    policyResultBufferElts = (size_t)maxBatchSize * (nnXLen * nnYLen);
    valueResultBufferElts = (size_t)maxBatchSize * m.numValueChannels;
    scoreValueResultBufferElts = (size_t)maxBatchSize * m.numScoreValueChannels;
    ownershipResultBufferElts = (size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels;

    userInputBuffer = new float[(size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen];
    userInputBufferHalf = new half_t[(size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen];
    userInputGlobalBuffer = new float[(size_t)m.numInputGlobalChannels * maxBatchSize];

    policyPassResults = new float[(size_t)maxBatchSize * 1];
    policyResults = new float[(size_t)maxBatchSize * nnXLen * nnYLen];
    policyResultsHalf = new half_t[(size_t)maxBatchSize * nnXLen * nnYLen];
    valueResults = new float[(size_t)maxBatchSize * m.numValueChannels];

    scoreValueResults = new float[(size_t)maxBatchSize * m.numScoreValueChannels];
    ownershipResults = new float[(size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels];
    ownershipResultsHalf = new half_t[(size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputBufferHalf;
    delete[] userInputGlobalBuffer;
    delete[] policyPassResults;
    delete[] policyResults;
    delete[] policyResultsHalf;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
    delete[] ownershipResultsHalf;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;

};


InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel,maxBatchSize,nnXLen,nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}


void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int version = gpuHandle->model->version;

  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
  assert(numSpatialFeatures == gpuHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  for(int nIdx = 0; nIdx<batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobal;
    const float* rowSpatial = inputBufs[nIdx]->rowSpatial;
    std::copy(rowGlobal,rowGlobal+numGlobalFeatures,rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);
  }

  Buffers* buffers = gpuHandle->buffers.get();

  assert(inputBuffers->userInputBufferElts == buffers->inputElts);
  assert(inputBuffers->userInputGlobalBufferElts == buffers->inputGlobalElts);
  assert(inputBuffers->policyResultBufferElts == buffers->policyElts);
  assert(inputBuffers->valueResultBufferElts == buffers->valueElts);
  assert(inputBuffers->singlePolicyResultElts + inputBuffers->singlePolicyPassResultElts == gpuHandle->policySize);
  assert(inputBuffers->scoreValueResultBufferElts == buffers->scoreValueElts);
  assert(inputBuffers->ownershipResultBufferElts == buffers->ownershipElts);
  assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);

  ComputeHandleInternal* handle = gpuHandle->handle.get();
  bool useFP16Storage = gpuHandle->handle->usingFP16Storage;

  cl_int err;

  if(useFP16Storage) {
    size_t numElts = inputBuffers->singleInputElts * batchSize;
    for(size_t i = 0; i<numElts; i++)
      inputBuffers->userInputBufferHalf[i] = half_float::half_cast<half_t>(inputBuffers->userInputBuffer[i]);

    err = clEnqueueWriteBuffer(
      handle->commandQueue,
      buffers->input,
      CL_FALSE,
      0,
      inputBuffers->singleInputElts * sizeof(half_t) * batchSize,
      inputBuffers->userInputBufferHalf,
      0,
      NULL,
      NULL
    );
    CHECK_ERR(err);
  }
  else {
    err = clEnqueueWriteBuffer(
      handle->commandQueue,
      buffers->input,
      CL_FALSE,
      0,
      inputBuffers->singleInputElts * sizeof(float) * batchSize,
      inputBuffers->userInputBuffer,
      0,
      NULL,
      NULL
    );
    CHECK_ERR(err);
  }

  err = clEnqueueWriteBuffer(
    handle->commandQueue,
    buffers->inputGlobal,
    CL_FALSE,
    0,
    inputBuffers->singleInputGlobalElts * sizeof(float) * batchSize,
    inputBuffers->userInputGlobalBuffer,
    0,
    NULL,
    NULL
  );
  CHECK_ERR(err);

  gpuHandle->model->apply(
    handle,
    gpuHandle->scratch.get(),
    batchSize,

    buffers->input,
    buffers->inputGlobal,

    buffers->mask,
    buffers->maskSum,

    buffers->trunk,

    buffers->policyPass,
    buffers->policy,

    buffers->value,
    buffers->scoreValue,
    buffers->ownership,

    buffers->convWorkspace,
    buffers->convWorkspace2
  );

  cl_bool blocking = CL_TRUE;
  err = clEnqueueReadBuffer(
    handle->commandQueue, buffers->policyPass, blocking, 0,
    inputBuffers->singlePolicyPassResultElts*sizeof(float)*batchSize, inputBuffers->policyPassResults, 0, NULL, NULL
  );
  CHECK_ERR(err);
  if(useFP16Storage) {
    err = clEnqueueReadBuffer(
      handle->commandQueue, buffers->policy, blocking, 0,
      inputBuffers->singlePolicyResultElts*sizeof(half_t)*batchSize, inputBuffers->policyResultsHalf, 0, NULL, NULL
    );
    CHECK_ERR(err);
    size_t numElts = inputBuffers->singlePolicyResultElts * batchSize;
    for(size_t i = 0; i<numElts; i++)
      inputBuffers->policyResults[i] = inputBuffers->policyResultsHalf[i];
  }
  else {
    err = clEnqueueReadBuffer(
      handle->commandQueue, buffers->policy, blocking, 0,
      inputBuffers->singlePolicyResultElts*sizeof(float)*batchSize, inputBuffers->policyResults, 0, NULL, NULL
    );
    CHECK_ERR(err);
  }
  err = clEnqueueReadBuffer(
    handle->commandQueue, buffers->value, blocking, 0,
    inputBuffers->singleValueResultElts*sizeof(float)*batchSize, inputBuffers->valueResults, 0, NULL, NULL
  );
  CHECK_ERR(err);
  err = clEnqueueReadBuffer(
    handle->commandQueue, buffers->scoreValue, blocking, 0,
    inputBuffers->singleScoreValueResultElts*sizeof(float)*batchSize, inputBuffers->scoreValueResults, 0, NULL, NULL
  );
  CHECK_ERR(err);
  if(useFP16Storage) {
    err = clEnqueueReadBuffer(
      handle->commandQueue, buffers->ownership, blocking, 0,
      inputBuffers->singleOwnershipResultElts*sizeof(half_t)*batchSize, inputBuffers->ownershipResultsHalf, 0, NULL, NULL
    );
    CHECK_ERR(err);
    size_t numElts = inputBuffers->singleOwnershipResultElts * batchSize;
    for(size_t i = 0; i<numElts; i++)
      inputBuffers->ownershipResults[i] = inputBuffers->ownershipResultsHalf[i];
  }
  else {
    err = clEnqueueReadBuffer(
      handle->commandQueue, buffers->ownership, blocking, 0,
      inputBuffers->singleOwnershipResultElts*sizeof(float)*batchSize, inputBuffers->ownershipResults, 0, NULL, NULL
    );
    CHECK_ERR(err);
  }

  #ifdef PROFILE_KERNELS
  {
    cl_int profileErr;
    profileErr = clWaitForEvents(handle->profileEvents.size(), handle->profileEvents.data());
    CHECK_ERR(profileErr);
    for(int i = 0; i<handle->profileCallbacks.size(); i++) {
      handle->profileCallbacks[i]();
    }
    for(int i = 0; i<handle->profileEvents.size(); i++) {
      clReleaseEvent(handle->profileEvents[i]);
    }
    handle->profileEvents.clear();
    handle->profileCallbacks.clear();

    static int profileResultPrintCounter = 0;
    profileResultPrintCounter += 1;
    if(profileResultPrintCounter % 100 == 0) {
      for(int i = 0; i<handle->profileResultPrinters.size(); i++) {
        handle->profileResultPrinters[i]();
      }
    }
  }
  #else
  assert(handle->profileEvents.size() == 0);
  assert(handle->profileCallbacks.size() == 0);
  assert(handle->profileResultPrinters.size() == 0);
  #endif

  assert(outputs.size() == batchSize);

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    const float* policySrcBuf = inputBuffers->policyResults + row * inputBuffers->singlePolicyResultElts;
    float* policyProbs = output->policyProbs;

    //These are not actually correct, the client does the postprocessing to turn them into
    //policy probabilities and white game outcome probabilities
    //Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    policyProbs[inputBuffers->singlePolicyResultElts] = inputBuffers->policyPassResults[row];

    int numValueChannels = gpuHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * nnXLen * nnYLen;
      assert(gpuHandle->model->numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(version >= 9) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    }
    else if(version >= 8) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(version >= 4) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(version >= 3) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }

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

  if(useNHWC != false)
    return false;

  ComputeContext* context = createComputeContextForTesting({gpuIdx}, logger, nnXLen, nnYLen, useFP16, useNHWC);
  ComputeHandleInternal* handle = new ComputeHandleInternal(context, gpuIdx, useNHWC, useNHWC);

  ConvLayer* layer = new ConvLayer(handle, desc, nnXLen, nnYLen, useFP16);

  size_t numInputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->inChannels;
  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateConv: unexpected input buffer size");
  outputBuffer.resize(numOutputFloats);

  vector<float> inputTmp = inputBuffer;
  cl_mem input = createReadOnlyBuffer(handle,inputTmp,useFP16);
  ConvWorkspaceEltsNeeded convWorkspaceElts = layer->requiredConvWorkspaceElts(handle,batchSize);
  cl_mem convWorkspace = createReadWriteBuffer(handle, convWorkspaceElts.size1, useFP16);
  cl_mem convWorkspace2 = createReadWriteBuffer(handle, convWorkspaceElts.size2, useFP16);

  cl_mem output = clCreateBuffer(handle->clContext, CL_MEM_READ_WRITE, byteSizeofVectorContents(outputBuffer), NULL, &err);
  CHECK_ERR(err);
  layer->apply(handle, batchSize, input, output, convWorkspace, convWorkspace2);

  blockingReadBuffer(handle->commandQueue, output, numOutputFloats, outputBuffer, useFP16);

  clReleaseMemObject(output);
  clReleaseMemObject(convWorkspace);
  clReleaseMemObject(convWorkspace2);
  clReleaseMemObject(input);
  delete layer;
  delete handle;
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

  if(useNHWC != false)
    return false;

  ComputeContext* context = createComputeContextForTesting({gpuIdx}, logger, nnXLen, nnYLen, useFP16, useNHWC);
  ComputeHandleInternal* handle = new ComputeHandleInternal(context, gpuIdx, useNHWC, useNHWC);

  ActivationLayerDesc actDesc;
  actDesc.activation = ACTIVATION_IDENTITY;

  BatchNormLayer* layer = new BatchNormLayer(handle, desc, &actDesc, nnXLen, nnYLen, useFP16);

  size_t numInputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;
  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected input buffer size");
  outputBuffer.resize(numOutputFloats);

  vector<float> inputTmp = inputBuffer;
  vector<float> maskTmp = maskBuffer;
  cl_mem input = createReadOnlyBuffer(handle,inputTmp,useFP16);
  cl_mem mask = createReadOnlyBuffer(handle,maskTmp,useFP16);

  cl_mem output = clCreateBuffer(handle->clContext, CL_MEM_WRITE_ONLY, byteSizeofVectorContents(outputBuffer), NULL, &err);
  CHECK_ERR(err);
  layer->apply(handle, batchSize, input, output, mask);

  blockingReadBuffer(handle->commandQueue, output, numOutputFloats, outputBuffer, useFP16);

  clReleaseMemObject(input);
  clReleaseMemObject(mask);
  clReleaseMemObject(output);
  delete layer;
  delete handle;
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
  int gpuIdx = 0;

  if(useNHWC != false)
    return false;

  ComputeContext* context = createComputeContextForTesting({gpuIdx}, logger, nnXLen, nnYLen, useFP16, useNHWC);
  ComputeHandleInternal* handle = new ComputeHandleInternal(context, gpuIdx, useNHWC, useNHWC);

  ResidualBlock* layer = new ResidualBlock(handle, desc, nnXLen, nnYLen, useFP16);

  size_t numTrunkFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)batchSize * nnXLen * nnYLen;
  if(numTrunkFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");
  outputBuffer.resize(numTrunkFloats);

  ScratchBuffers* scratch = new ScratchBuffers(handle, batchSize, nnXLen, nnYLen);

  vector<float> inputTmp = inputBuffer;
  vector<float> maskTmp = maskBuffer;
  cl_mem trunk = createReadWriteBuffer(handle,inputTmp,useFP16);
  cl_mem mask = createReadOnlyBuffer(handle,maskTmp,useFP16);
  cl_mem trunkScratch = createReadWriteBuffer(handle,numTrunkFloats,useFP16);

  ConvWorkspaceEltsNeeded convWorkspaceElts = layer->requiredConvWorkspaceElts(handle,batchSize);
  cl_mem convWorkspace = createReadWriteBuffer(handle, convWorkspaceElts.size1, useFP16);
  cl_mem convWorkspace2 = createReadWriteBuffer(handle, convWorkspaceElts.size2, useFP16);

  layer->apply(handle, scratch, batchSize, trunk, trunkScratch, mask, convWorkspace, convWorkspace2);

  blockingReadBuffer(handle->commandQueue, trunk, numTrunkFloats, outputBuffer, useFP16);

  clReleaseMemObject(trunk);
  clReleaseMemObject(mask);
  clReleaseMemObject(trunkScratch);
  clReleaseMemObject(convWorkspace);
  clReleaseMemObject(convWorkspace2);
  delete scratch;
  delete layer;
  delete handle;
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
  int gpuIdx = 0;

  if(useNHWC != false)
    return false;

  ComputeContext* context = createComputeContextForTesting({gpuIdx}, logger, nnXLen, nnYLen, useFP16, useNHWC);
  ComputeHandleInternal* handle = new ComputeHandleInternal(context, gpuIdx, useNHWC, useNHWC);

  GlobalPoolingResidualBlock* layer = new GlobalPoolingResidualBlock(handle, desc, nnXLen, nnYLen, useFP16);

  size_t numTrunkFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)batchSize * nnXLen * nnYLen;
  size_t numMaskSumFloats = (size_t)batchSize;

  if(numTrunkFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");
  outputBuffer.resize(numTrunkFloats);

  ScratchBuffers* scratch = new ScratchBuffers(handle, batchSize, nnXLen, nnYLen);

  vector<float> inputTmp = inputBuffer;
  vector<float> maskTmp = maskBuffer;
  cl_mem trunk = createReadWriteBuffer(handle,inputTmp,useFP16);
  cl_mem mask = createReadOnlyBuffer(handle,maskTmp,useFP16);
  cl_mem maskSum = createReadWriteBuffer(handle,numMaskSumFloats,false);
  cl_mem trunkScratch = createReadWriteBuffer(handle,numTrunkFloats,useFP16);

  ConvWorkspaceEltsNeeded convWorkspaceElts = layer->requiredConvWorkspaceElts(handle,batchSize);
  cl_mem convWorkspace = createReadWriteBuffer(handle, convWorkspaceElts.size1, useFP16);
  cl_mem convWorkspace2 = createReadWriteBuffer(handle, convWorkspaceElts.size2, useFP16);

  computeMaskSums(handle,mask,maskSum,batchSize,nnXLen,nnYLen);

  layer->apply(
    handle,
    scratch,
    batchSize,
    trunk,
    trunkScratch,
    mask,
    maskSum,
    convWorkspace,
    convWorkspace2
  );

  blockingReadBuffer(handle->commandQueue, trunk, numTrunkFloats, outputBuffer, useFP16);

  clReleaseMemObject(trunk);
  clReleaseMemObject(mask);
  clReleaseMemObject(maskSum);
  clReleaseMemObject(trunkScratch);
  clReleaseMemObject(convWorkspace);
  clReleaseMemObject(convWorkspace2);
  delete scratch;
  delete layer;
  delete handle;
  freeComputeContext(context);

  return true;
}


#endif  // USE_OPENCL_BACKEND
