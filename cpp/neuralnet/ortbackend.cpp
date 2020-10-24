#ifdef USE_ONNXRUNTIME_BACKEND

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../core/makedir.h"
#include "../dataio/homedata.h"

#include "../external/half-2.1.0/include/half.hpp"
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#if defined(USE_ORT_CUDA) || defined(USE_ORT_TENSORRT)
  #include <cuda_provider_factory.h>
  #include "../neuralnet/cudaincludes.h"
#endif
#ifdef USE_ORT_TENSORRT
  #include <tensorrt_provider_factory.h>
#endif
#ifdef USE_ORT_DIRECTML
  #include <dml_provider_factory.h>
  #include <Dxgi.h>
#endif
#ifdef USE_ORT_MIGRAPHX
  #include <migraphx_provider_factory.h>
  #include "../neuralnet/openclincludes.h"
  #include "../neuralnet/openclhelpers.h"
#endif

using namespace std;

//------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
}

void NeuralNet::globalCleanup() {
}

//------------------------------------------------------------------------------

// Model itself is loaded in ComputeHandle instead
struct LoadedModel {
  ModelDesc modelDesc;

  // This is not optimal of course, we can probably tar .json and .onnx together?
  // Or the ONNX file itself can be parsed.
  LoadedModel(const string& fileName) {
    modelDesc.name = fileName;
    modelDesc.version = 8;
    modelDesc.numInputChannels = 22;
    modelDesc.numInputGlobalChannels = 19;
    modelDesc.numValueChannels = 3;
    modelDesc.numOwnershipChannels = 1;
    modelDesc.numScoreValueChannels = 4;
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file) {
  LoadedModel* loadedModel = new LoadedModel(file);
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

//------------------------------------------------------------------------------

std::unique_ptr < Ort::Env> env = nullptr;

struct Model {
  string name;
  int version;
  int numInputChannels;
  int numInputGlobalChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  Ort::Session* session;

  Model(
    const ModelDesc* desc,
    int gpuIdx,
    const string& onnxOptModelFile,
    const string& onnxRuntimeExecutionProvider,
    const string& homeDataDirOverride
  ) {
    name = desc->name;
    version = desc->version;
    numInputChannels = desc->numInputChannels;
    numInputGlobalChannels = desc->numInputGlobalChannels;
    numValueChannels = desc->numValueChannels;
    numScoreValueChannels = desc->numScoreValueChannels;
    numOwnershipChannels = desc->numOwnershipChannels;

    auto envLocal = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "Default");
    env = std::move(envLocal);
    Ort::SessionOptions sf;
    sf.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    sf.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    string dir = HomeData::getHomeDataDir(true, homeDataDirOverride);
    MakeDir::make(dir);
    string optModelPath = dir + "/" + onnxOptModelFile;
#ifdef _WIN32
    std::wstring optModelFile = std::wstring(optModelPath.begin(), optModelPath.end());
    sf.SetOptimizedModelFilePath(optModelFile.data());
#else
    sf.SetOptimizedModelFilePath(optModelPath.data());
#endif

    if(onnxRuntimeExecutionProvider == "CUDA") {
      #ifdef USE_ORT_CUDA
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, gpuIdx));
      #else
        throw StringError("KataGo was not compiled with CUDA support.");
      #endif
    }
    else if(onnxRuntimeExecutionProvider == "TensorRT") {
      #ifdef USE_ORT_TENSORRT
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, gpuIdx));
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, gpuIdx));
      #else
        throw StringError("KataGo was not compiled with TensorRT support.");
      #endif
    }
    else if(onnxRuntimeExecutionProvider == "DirectML") {
      #ifdef USE_ORT_DIRECTML
        sf.DisableMemPattern();
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sf, gpuIdx));
      #else
        throw StringError("KataGo was not compiled with DirectML support.");
      #endif
    }
    else if(onnxRuntimeExecutionProvider == "MIGraphX") {
      #ifdef USE_ORT_MIGRAPHX
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(sf, gpuIdx));
      #else
        throw StringError("KataGo was not compiled with MIGraphX support.");
      #endif
    }
    else {
      throw StringError("Invalid ONNXRuntime backend");
    }

#ifdef _WIN32
    std::wstring modelName = std::wstring(name.begin(), name.end());
    session = new Ort::Session(*env, modelName.data(), sf);
#else
    session = new Ort::Session(*env, name.data(), sf);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    // input nodes
    numInputNodes = session->GetInputCount();
    assert(numInputNodes == 2);

    for(int inputIdx = 0; inputIdx < numInputNodes; inputIdx++) {
      inputNodeNames.emplace_back(session->GetInputName(inputIdx, allocator));
    }
    
    // output nodes
    numOutputNodes = session->GetOutputCount();

    for(int outputIdx = 0; outputIdx < numOutputNodes; outputIdx++) {
      outputNodeNames.emplace_back(session->GetOutputName(outputIdx, allocator));
    }
  }

  bool getUsingFP16() {
    Ort::TypeInfo typeInfo = session->GetInputTypeInfo(0);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    auto type = tensorInfo.GetElementType();
    return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }

  vector<Ort::Value> evaluate(vector<Ort::Value>& inputTensors) {
    auto outputTensors = session->Run(
      Ort::RunOptions{nullptr},
      inputNodeNames.data(),
      inputTensors.data(),
      inputTensors.size(),
      outputNodeNames.data(),
      outputNodeNames.size()
    );

    return outputTensors;
  }

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  ~Model() {
    session->release();
  }

private:
  size_t numInputNodes;
  size_t numOutputNodes;
  vector<const char*> inputNodeNames;
  vector<const char*> outputNodeNames;
};

//------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t usingFP16;
  string onnxOptModelFile;
  string onnxRuntimeExecutionProvider;
  string homeDataDirOverride;

  ComputeContext(
    int nnX,
    int nnY,
    const string& optModelFile,
    const string& runtimeExecutionProvider,
    const string& homeDataDir,
    enabled_t useFP16
  ) {
    nnXLen = nnX;
    nnYLen = nnY;
    onnxOptModelFile = optModelFile;
    onnxRuntimeExecutionProvider = runtimeExecutionProvider;
    homeDataDirOverride = homeDataDir;
    usingFP16 = useFP16;
  }
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& onnxOptModelFile,
  const string& onnxRuntimeExecutionProvider,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)openCLReTunePerBoardSize;
  (void)useNHWCMode;
  (void)loadedModel;

  return new ComputeContext(
    nnXLen, nnYLen, onnxOptModelFile, onnxRuntimeExecutionProvider, homeDataDirOverride, useFP16Mode
  );
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

struct ComputeHandle {
  int nnXLen;
  int nnYLen;
  int policySize;
  bool usingFP16;
  Model* model;

  ComputeHandle(
    ComputeContext* context,
    const LoadedModel* loadedModel, 
    int gpuIdx
  ) {
    nnXLen = context->nnXLen;
    nnYLen = context->nnYLen;
    policySize = NNPos::getPolicySize(nnXLen, nnYLen);
    model = new Model(
      &(loadedModel->modelDesc),
      gpuIdx,
      context->onnxOptModelFile,
      context->onnxRuntimeExecutionProvider,
      context->homeDataDirOverride
    );
    usingFP16 = model->getUsingFP16();
  }
  ~ComputeHandle() {
    delete model;
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
  (void)maxBatchSize;
  (void)requireExactNNLen;
  (void)inputsUseNHWC;
  (void)serverThreadIdx;

  auto deviceStr = [&]() {
    if(gpuIdxForThisThread < 0)
      return string("");
    return " Device " + Global::intToString(gpuIdxForThisThread);
  };

  if(logger != NULL) {
    logger->write("ONNXRuntime backend thread " + Global::intToString(serverThreadIdx) + ":" + deviceStr() + " Model version " + Global::intToString(loadedModel->modelDesc.version));
    logger->write("ONNXRuntime backend thread " + Global::intToString(serverThreadIdx) + ":" + deviceStr() + " Model name: " + loadedModel->modelDesc.name);
  }

  ComputeHandle* handle = new ComputeHandle(context, loadedModel, gpuIdxForThisThread);

  if(logger != NULL) {
    if(context->onnxRuntimeExecutionProvider == "CUDA") {
      logger->write("ONNXRuntime: CUDA backend");
    }
    else if(context->onnxRuntimeExecutionProvider == "TensorRT") {
      logger->write("ONNXRuntime: TensorRT backend");
    }
    else if(context->onnxRuntimeExecutionProvider == "DirectML") {
      logger->write("ONNXRuntime: DirectML backend");
    }
    else if(context->onnxRuntimeExecutionProvider == "MIGraphX") {
      logger->write("ONNXRuntime: MIGraphX backend");
    }
    else {
      throw StringError("Invalid ONNXRuntime backend");
    }
  }

  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {

}

#if defined(USE_ORT_CUDA) || defined(USE_ORT_TENSORRT)
void NeuralNet::printCUDADevices() {
  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  for(int i = 0; i<numDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout << "Found CUDA device " << i << ": " << prop.name << endl;
  }
}
#endif

#ifdef USE_ORT_DIRECTML
#pragma comment(lib, "dxgi")
void NeuralNet::printDirectMLDevices() {
  IDXGIFactory* pFactory = NULL;
  IDXGIAdapter* pAdapter;
  vector<IDXGIAdapter*> vAdapters;

  if(FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory)))
  {
    throw StringError("Unable to create IDXGIFactory.");
  }

  for(int i = 0; pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND; ++i) {
    vAdapters.push_back(pAdapter);
  }

  if(pFactory) {
    pFactory->Release();
  }

  for(int i = 0; i < vAdapters.size(); i++) {
    DXGI_ADAPTER_DESC adapterDescription;
    vAdapters[i]->GetDesc(&adapterDescription);
    wstring wsDeviceName(adapterDescription.Description);
    string deviceName(wsDeviceName.begin(), wsDeviceName.end());
    if(deviceName != "Microsoft Basic Render Driver"){
      cout << "Found DirectML device " << i << ": " << deviceName.c_str() << endl;
    }
  }
}
#endif

#ifdef USE_ORT_MIGRAPHX
void NeuralNet::printOpenCLDevices() {
  vector<DeviceInfo> devices = DeviceInfo::getAllDeviceInfosOnSystem(NULL);
  for(int i = 0; i<devices.size(); i++) {
    const DeviceInfo& device = devices[i];
    string msg =
      "Found OpenCL Device " + Global::intToString(device.gpuIdx) + ": " + device.name + " (" + device.vendor + ")" +
      " (score " + Global::intToString(device.defaultDesirability) + ")";
    cout << msg << endl;
  }
}
#endif

void NeuralNet::printDevices(const string& ortExecutionProvider) {
  if(ortExecutionProvider == "CUDA") {
    #ifdef USE_ORT_CUDA
      NeuralNet::printCUDADevices();
    #endif
  }
  else if (ortExecutionProvider == "TensorRT") {
    #ifdef USE_ORT_TENSORRT
      NeuralNet::printCUDADevices();
    #endif
  }
  else if(ortExecutionProvider == "DirectML") {
    #ifdef USE_ORT_DIRECTML
      NeuralNet::printDirectMLDevices();
    #endif
  }
  else if(ortExecutionProvider == "MIGraphX") {
    #ifdef USE_ORT_MIGRAPHX
      NeuralNet::printOpenCLDevices();
    #endif
  }
}

//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleBinaryInputElts;
  size_t singleGlobalInputElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreResultElts;
  size_t singleOwnershipResultElts;

  // Host pointers
  float* userBinaryInputBuffer;
  float* userGlobalInputBuffer;
  float* policyResults;
  float* valueResults;
  float* scoreResults;
  float* ownershipResults;

  uint16_t* userBinaryInputBufferFP16;
  uint16_t* userGlobalInputBufferFP16;
  uint16_t* policyResultsFP16;
  uint16_t* valueResultsFP16;
  uint16_t* scoreResultsFP16;
  uint16_t* ownershipResultsFP16;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = nnXLen;
    int ySize = nnYLen;
    maxBatchSize = maxBatchSz;

    singleBinaryInputElts = (size_t)m.numInputChannels * xSize * ySize;
    singleGlobalInputElts = (size_t)m.numInputGlobalChannels;
    singlePolicyResultElts = (size_t)(1 + xSize * ySize);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * xSize * ySize;

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    userBinaryInputBuffer = new float[(size_t)maxBatchSize * singleBinaryInputElts];
    userGlobalInputBuffer = new float[(size_t)maxBatchSize * singleGlobalInputElts];
    policyResults = new float[(size_t)maxBatchSize * singlePolicyResultElts];
    valueResults = new float[(size_t)maxBatchSize * singleValueResultElts];
    scoreResults = new float[(size_t)maxBatchSize * singleScoreResultElts];
    ownershipResults = new float[(size_t)maxBatchSize * singleOwnershipResultElts];

    userBinaryInputBufferFP16 = new uint16_t[(size_t)maxBatchSize * singleBinaryInputElts];
    userGlobalInputBufferFP16 = new uint16_t[(size_t)maxBatchSize * singleGlobalInputElts];
    policyResultsFP16 = new uint16_t[(size_t)maxBatchSize * singlePolicyResultElts];
    valueResultsFP16 = new uint16_t[(size_t)maxBatchSize * singleValueResultElts];
    scoreResultsFP16 = new uint16_t[(size_t)maxBatchSize * singleScoreResultElts];
    ownershipResultsFP16 = new uint16_t[(size_t)maxBatchSize * singleOwnershipResultElts];
  }

  void copyInputFloatToHalf(size_t batchSize) {
    for(int i = 0; i < batchSize * singleBinaryInputElts; i++) {
      userBinaryInputBufferFP16[i] = half_float::detail::float2half<std::round_to_nearest>(userBinaryInputBuffer[i]);
    }
    for(int i = 0; i < batchSize * singleGlobalInputElts; i++) {
      userGlobalInputBufferFP16[i] = half_float::detail::float2half<std::round_to_nearest>(userGlobalInputBuffer[i]);
    }
  }

  void copyOutputHalfToFloat(size_t batchSize) {
    for(int i = 0; i < batchSize * singlePolicyResultElts; i++) {
      policyResults[i] = half_float::detail::half2float<float>(policyResultsFP16[i]);
    }
    for(int i = 0; i < batchSize * singleValueResultElts; i++) {
      valueResults[i] = half_float::detail::half2float<float>(valueResultsFP16[i]);
    }
    for(int i = 0; i < batchSize * singleScoreResultElts; i++) {
      scoreResults[i] = half_float::detail::half2float<float>(scoreResultsFP16[i]);
    }
    for(int i = 0; i < batchSize * singleOwnershipResultElts; i++) {
      ownershipResults[i] = half_float::detail::half2float<float>(ownershipResultsFP16[i]);
    }
  }

  ~InputBuffers() {
    delete[] userBinaryInputBuffer;
    delete[] userGlobalInputBuffer;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreResults;
    delete[] ownershipResults;
    delete[] userBinaryInputBufferFP16;
    delete[] userGlobalInputBufferFP16;
    delete[] policyResultsFP16;
    delete[] valueResultsFP16;
    delete[] scoreResultsFP16;
    delete[] ownershipResultsFP16;
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

//------------------------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  int symmetry,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int version = gpuHandle->model->version;
  bool usingFP16 = gpuHandle->usingFP16;

  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
  assert(numSpatialFeatures == gpuHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleBinaryInputElts);
  assert(numGlobalFeatures == inputBuffers->singleGlobalInputElts);

  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userBinaryInputBuffer + (inputBuffers->singleBinaryInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userGlobalInputBuffer + (inputBuffers->singleGlobalInputElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobal;
    const float* rowSpatial = inputBufs[nIdx]->rowSpatial;
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, symmetry);
  }

  const int policySize = nnXLen * nnYLen + 1;
  const int valueSize = gpuHandle->model->numValueChannels;
  const int scoreSize = gpuHandle->model->numScoreValueChannels;
  const int ownershipSize = nnXLen * nnYLen;

  assert(valueSize == 3);
  assert(gpuHandle->model->numOwnershipChannels == 1);

  // input
  vector<vector<int64_t>> inputNodeShape(2);
  vector<int64_t> inputNodeSizes(2);
  vector<Ort::Value> inputTensors;

  for(int i = 0; i < 2; i++) {
    Ort::TypeInfo typeInfo = gpuHandle->model->session->GetInputTypeInfo(i);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

    // input node dimensions
    inputNodeShape[i] = tensorInfo.GetShape();
    // This is -1, so should be manually assigned
    inputNodeShape[i][0] = (int64_t)batchSize;
  }
  assert(inputNodeShape[0].size() == 4);
  assert(inputNodeShape[0][1] == numSpatialFeatures);
  // Dynamic input shape for onnx models without masking
  inputNodeShape[0][2] = nnYLen;
  inputNodeShape[0][3] = nnXLen;
  assert(inputNodeShape[1].size() == 2);
  assert(inputNodeShape[1][1] == numGlobalFeatures);

  inputNodeSizes[0] = (int64_t)(batchSize * inputBuffers->singleBinaryInputElts);
  inputNodeSizes[1] = (int64_t)(batchSize * inputBuffers->singleGlobalInputElts);

  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  if(!usingFP16) {
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
      memoryInfo,
      inputBuffers->userBinaryInputBuffer,
      inputNodeSizes[0],
      inputNodeShape[0].data(),
      inputNodeShape[0].size()
    ));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
      memoryInfo,
      inputBuffers->userGlobalInputBuffer,
      inputNodeSizes[1],
      inputNodeShape[1].data(),
      inputNodeShape[1].size()
    ));
  }
  else {
    inputBuffers->copyInputFloatToHalf(batchSize);
    inputTensors.emplace_back(Ort::Value::CreateTensor(
      memoryInfo,
      inputBuffers->userBinaryInputBufferFP16,
      inputNodeSizes[0] * sizeof(uint16_t),
      inputNodeShape[0].data(),
      inputNodeShape[0].size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
    ));
    inputTensors.emplace_back(Ort::Value::CreateTensor(
      memoryInfo,
      inputBuffers->userGlobalInputBufferFP16,
      inputNodeSizes[1] * sizeof(uint16_t),
      inputNodeShape[1].data(),
      inputNodeShape[1].size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
    ));
  }

  vector<float> inputVector(inputNodeSizes[1]);
  for(int i = 0; i < inputNodeSizes[1]; i++) {
    inputVector[i] = (inputBuffers->userGlobalInputBuffer[i]);
  }

  // Evaluate
  auto outputTensors = gpuHandle->model->evaluate(inputTensors);

  // collect outputs to vectors
  if(!usingFP16){
    auto policy = outputTensors[0].GetTensorMutableData<float>();
    auto value = outputTensors[1].GetTensorMutableData<float>();
    auto score= outputTensors[2].GetTensorMutableData<float>();
    auto ownership = outputTensors[3].GetTensorMutableData<float>();
    std::copy(policy, policy + batchSize * policySize, inputBuffers->policyResults);
    std::copy(value, value + batchSize * valueSize, inputBuffers->valueResults);
    std::copy(score, score + batchSize * scoreSize, inputBuffers->scoreResults);
    std::copy(ownership, ownership + batchSize * ownershipSize, inputBuffers->ownershipResults);
  }
  else {
    auto policy = outputTensors[0].GetTensorMutableData<uint16_t>();
    auto value = outputTensors[1].GetTensorMutableData<uint16_t>();
    auto score = outputTensors[2].GetTensorMutableData<uint16_t>();
    auto ownership = outputTensors[3].GetTensorMutableData<uint16_t>();
    std::copy(policy, policy + batchSize * policySize, inputBuffers->policyResultsFP16);
    std::copy(value, value + batchSize * valueSize, inputBuffers->valueResultsFP16);
    std::copy(score, score + batchSize * scoreSize, inputBuffers->scoreResultsFP16);
    std::copy(ownership, ownership + batchSize * ownershipSize, inputBuffers->ownershipResultsFP16);
    inputBuffers->copyOutputHalfToFloat(batchSize);
  }

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    // Policy
    const float* policySrcBuf = inputBuffers->policyResults + row * policySize;
    float* policyProbs = output->policyProbs;

    //These are not actually correct, the client does the postprocessing to turn them into
    //policy probabilities and white game outcome probabilities
    //Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, symmetry);
    policyProbs[policySize-1] = policySrcBuf[policySize-1];

    // Value
    output->whiteWinProb = inputBuffers->valueResults[row * valueSize];
    output->whiteLossProb = inputBuffers->valueResults[row * valueSize + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * valueSize + 2];

    // Score
    if(version >= 8) {
      assert(scoreSize == 4);
      output->whiteScoreMean = inputBuffers->scoreResults[row * scoreSize];
      output->whiteScoreMeanSq = inputBuffers->scoreResults[row * scoreSize + 1];
      output->whiteLead = inputBuffers->scoreResults[row * scoreSize + 2];
      output->varTimeLeft = inputBuffers->scoreResults[row * scoreSize + 3];
    }
    else if(version >= 4) {
      assert(scoreSize== 2);
      output->whiteScoreMean = inputBuffers->scoreResults[row * scoreSize];
      output->whiteScoreMeanSq = inputBuffers->scoreResults[row * scoreSize + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
    }
    else if(version >= 3) {
      assert(scoreSize == 1);
      output->whiteScoreMean = inputBuffers->scoreResults[row * scoreSize];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
    }
    else {
      ASSERT_UNREACHABLE;
    }

    // Ownership
    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * ownershipSize;
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, symmetry);
    }
  }
}

//------------------------------------------------------------------------------

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)desiredBatchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)desiredBatchSize;
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
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc;
  (void)desiredBatchSize;
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
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)desiredBatchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

#endif  // USE_ONNXRUNTIME_BACKEND