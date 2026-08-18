#ifdef USE_RYZENAI_BACKEND

/** RyzenAI (AMD NPU) backend.
 *
 * Structured after cpp/neuralnet/eigenbackend.cpp. Currently all evaluation
 * runs on the pure C++ CPU reference forward path (neuralnet/ryzenaireference.h);
 * NPU kernel dispatch via XRT will be layered on top of it. Inputs and all
 * internal activations are NHWC float32.
 */

#include "../neuralnet/nninterface.h"

#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"

#include "../core/test.h"

#include "../neuralnet/ryzenaidevice.h"
#include "../neuralnet/ryzenaikernel.h"
#include "../neuralnet/ryzenaimatmul.h"
#include "../neuralnet/ryzenaireference.h"
#include "../neuralnet/ryzenaishapes.h"

using namespace std;

// LoadedModel / ModelDesc ---------------------------------------------------------------------------------------------

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
  // NOTE: modelDesc weights must still be intact here if any Workspace still
  // references them - but all ComputeHandles are always freed before this.
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

// --------------------------------------------------------------------------------------------------------------

struct ComputeContext {
  const int nnXLen;
  const int nnYLen;

  const enabled_t useFP16Mode;

  // Backend-specific configuration, read off of cfg in createComputeContext.
  // Device indices themselves (ryzenaiDeviceToUse / ryzenaiDeviceToUseThread<N>)
  // are parsed uniformly by setup.cpp and arrive via gpuIdxs / gpuIdxForThisThread.
  const string artifactDir;
  const string dtype;
  const bool forceNpuOnly;
  const bool verboseDispatch;
  const int maxCols;
  const int forceK;
  const vector<int> gpuIdxs;
  const vector<string> xclbins;

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

  ComputeContext(
    int nnX,
    int nnY,
    enabled_t fp16Mode,
    const string& aDir,
    const string& dt,
    bool forceNpu,
    bool verbose,
    int maxColumns,
    int forceReduceDim,
    const vector<int>& gIdxs,
    const vector<string>& xcl
  )
    : nnXLen(nnX),
      nnYLen(nnY),
      useFP16Mode(fp16Mode),
      artifactDir(aDir),
      dtype(dt),
      forceNpuOnly(forceNpu),
      verboseDispatch(verbose),
      maxCols(maxColumns),
      forceK(forceReduceDim),
      gpuIdxs(gIdxs),
      xclbins(xcl)
  {}
  ~ComputeContext() {}
};

// --------------------------------------------------------------------------------------------------------------

struct ComputeHandle {
  ComputeContext* context;
  bool inputsUseNHWC;
  int gpuIdxForThisThread;

  // NOT owned - owned by the LoadedModel, which always outlives this handle.
  // Do not call releaseWeights() on it: the workspace references its weights.
  const ModelDesc& modelDesc;

  RyzenAIRef::Workspace* workspace;

  // Dense-layer accelerator. Null when no NPU/artifact is usable, in which case
  // the workspace runs entirely on the CPU reference path.
  RyzenAIMatMul::Accel* accel;
  string accelInitError;
  Logger* logger;  // not owned; may be null

  // Output buffers for RyzenAIRef::forward(), allocated once per handle.
  vector<float> policyBuf;
  vector<float> policyPassBuf;
  vector<float> valueBuf;
  vector<float> scoreValueBuf;
  vector<float> ownershipBuf;

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  ComputeHandle(
    ComputeContext* ctx,
    const LoadedModel& loadedModel,
    int maxBatchSize,
    bool iNHWC,
    int gpuIdx,
    Logger* lg
  )
    : context(ctx),
      inputsUseNHWC(iNHWC),
      gpuIdxForThisThread(gpuIdx),
      modelDesc(loadedModel.modelDesc),
      workspace(NULL),
      accel(NULL),
      logger(lg)
  {
    workspace = RyzenAIRef::createWorkspace(modelDesc,maxBatchSize,context->nnXLen,context->nnYLen);
    RyzenAIRef::setProfileEnabled(context->verboseDispatch);

    RyzenAIMatMul::Options options;
    options.artifactDir = context->artifactDir;
    options.deviceIdx = gpuIdx;
    options.dtype = context->dtype;
    options.maxCols = context->maxCols;
    options.forceK = context->forceK;
    string accelErr;
    accel = RyzenAIMatMul::create(options,accelErr);
    if(accel != NULL)
      RyzenAIRef::setMatMulAccel(workspace,accel);
    else
      accelInitError = accelErr;

    const int nnXLen = context->nnXLen;
    const int nnYLen = context->nnYLen;
    policyBuf = vector<float>((size_t)maxBatchSize * nnXLen * nnYLen * modelDesc.numPolicyChannels);
    policyPassBuf = vector<float>((size_t)maxBatchSize * modelDesc.numPolicyChannels);
    valueBuf = vector<float>((size_t)maxBatchSize * modelDesc.numValueChannels);
    scoreValueBuf = vector<float>((size_t)maxBatchSize * modelDesc.numScoreValueChannels);
    ownershipBuf = vector<float>((size_t)maxBatchSize * nnXLen * nnYLen * modelDesc.numOwnershipChannels);
  }

  ~ComputeHandle() {
    // Report the NPU/CPU split here rather than at startup: how many dense
    // layers were actually accelerated is only known after evaluating.
    if(logger != NULL && accel != NULL && context->verboseDispatch)
      logger->write(RyzenAIMatMul::report(accel));
    if(logger != NULL && context->verboseDispatch)
      logger->write(RyzenAIRef::profileReport());

    // The workspace holds a bare pointer to the accelerator, so it has to stop
    // referring to it before the accelerator's device buffers go away.
    if(workspace != NULL)
      RyzenAIRef::setMatMulAccel(workspace,NULL);
    if(accel != NULL)
      RyzenAIMatMul::free(accel);
    accel = NULL;
    if(workspace != NULL)
      RyzenAIRef::freeWorkspace(workspace);
    workspace = NULL;
  }
};

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  std::vector<float> spatialInput;
  std::vector<float> globalInput;
  std::vector<float> metaInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = m.numInputGlobalChannels;
    singleInputMetaElts = m.numInputMetaChannels;

    singlePolicyPassResultElts = (size_t)(m.numPolicyChannels);
    singlePolicyResultElts = (size_t)(m.numPolicyChannels * nnXLen * nnYLen);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;

    testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0) {
      testAssert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);
    }

    spatialInput = vector<float>(m.numInputChannels * nnXLen * nnYLen * maxBatchSize);
    globalInput = vector<float>(m.numInputGlobalChannels * maxBatchSize);
    if(m.numInputMetaChannels > 0)
      metaInput = vector<float>(m.numInputMetaChannels * maxBatchSize);
    else
      metaInput = vector<float>(1);
  }

  ~InputBuffers() { }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

// NeuralNet -----------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // Must happen before the first XRT call of the process. printDevices() also
  // runs before any compute context exists, so it calls this too.
  RyzenAIDevice::ensureRuntimeLibraryPath();
}

void NeuralNet::globalCleanup() {
  // no-op
}

//------------------------------------------------------------------------------

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  string artifactDirConfig = cfg.contains("ryzenaiArtifactDir") ? cfg.getString("ryzenaiArtifactDir") : "";
  // auto | bf16 | bfp16. Both bf16 and bfp16 hand the NPU bfloat16 operands and
  // read float32 back - the difference is entirely inside the kernel - so this
  // only selects which artifact directory is loaded. See resolveDtype() for why
  // auto currently means bf16 even on hardware that supports bfp16.
  string dtypeStr = cfg.contains("ryzenaiDtype") ? cfg.getString("ryzenaiDtype") : "auto";
  RyzenAIKernel::Dtype dtype = RyzenAIKernel::Dtype::Auto;
  if(!RyzenAIKernel::parseDtype(dtypeStr, dtype))
    throw StringError(
      "RyzenAI backend: unrecognized ryzenaiDtype = '" + dtypeStr + "', expected auto, bf16 or bfp16"
    );
  bool forceNpuOnly = cfg.contains("ryzenaiForceNpuOnly") ? cfg.getBool("ryzenaiForceNpuOnly") : false;
  bool verboseDispatch = cfg.contains("ryzenaiVerboseDispatch") ? cfg.getBool("ryzenaiVerboseDispatch") : false;
  // Wider is not automatically faster: at one board per evaluation the array
  // is starved and the fixed dispatch cost dominates, so 4 columns measured
  // faster than 8. 0 means "as wide as the device allows".
  int maxCols = cfg.contains("ryzenaiMaxColumns") ? cfg.getInt("ryzenaiMaxColumns",0,64) : 4;
  // Run every layer from one xclbin, trading zero-padded arithmetic for not
  // switching hardware contexts. 0 = pick the closest K per layer.
  // -1 (the default) means decide from the model: see RyzenAIShapes::chooseSingleK.
  // 0 disables it, a positive value forces that reduction dim.
  int forceK = cfg.contains("ryzenaiForceK") ? cfg.getInt("ryzenaiForceK",-1,65536) : -1;
  if(forceK < 0) {
    // 4x is where the measured trade turns: below it the extra zero-padded
    // arithmetic is lost in the noise (NPU compute is ~1% of a forward pass),
    // above it the padding is real work on models whose GEMMs are already big.
    forceK = loadedModel != NULL ? RyzenAIShapes::chooseSingleK(loadedModel->modelDesc, nnXLen, nnYLen, 4.0) : 0;
  }

  string artifactDir = RyzenAIDevice::resolveArtifactDir(artifactDirConfig);

  if(logger != NULL) {
    logger->write("RyzenAI backend: " + RyzenAIDevice::ensureRuntimeLibraryPath());
    logger->write("RyzenAI backend: " + RyzenAIDevice::describeRuntime());
    logger->write("RyzenAI backend: artifact dir = " + artifactDir);
    logger->write("RyzenAI backend: dtype = " + dtypeStr + " (resolved: " +
                  RyzenAIKernel::dtypeName(RyzenAIKernel::resolveDtype(dtype, RyzenAIDevice::archOfDevice(-1))) + ")");
    if(useFP16Mode == enabled_t::True)
      logger->write("RyzenAI backend: useFP16 = true was requested, but the current reference path always computes in fp32");
  }

  vector<string> xclbins = RyzenAIDevice::listXclbins(artifactDir);
  if(xclbins.empty()) {
    if(forceNpuOnly) {
      throw StringError(
        "RyzenAI backend: ryzenaiForceNpuOnly = true but no .xclbin NPU kernel artifacts were found in " +
        artifactDir + " - NPU dispatch is not possible"
      );
    }
    if(logger != NULL) {
      logger->write(
        "RyzenAI backend: no .xclbin NPU kernel artifacts found in " + artifactDir +
        " - running on the CPU reference forward path, NPU acceleration is not enabled yet"
      );
    }
  }
  else if(cfg.contains("ryzenaiSelfTest") && cfg.getBool("ryzenaiSelfTest")) {
    // Loads every artifact, dispatches it on the NPU and checks it against a
    // plain-C++ GEMM. Opt-in because it costs a few seconds of startup, but it
    // is the fastest way to tell whether a machine's NPU/driver/artifact set is
    // actually working.
    if(logger != NULL)
      logger->write(RyzenAIKernel::selfTest(artifactDir, -1));
  }

  // Which GEMM shapes this model actually needs. An xclbin bakes in the
  // reduction dim K (and only K -- M and N ride in the instruction stream), so
  // this is what decides which artifacts must exist and which layers fall back
  // to the CPU reference path.
  if(cfg.contains("ryzenaiShapeReport") && cfg.getBool("ryzenaiShapeReport")) {
    if(logger != NULL && loadedModel != NULL)
      logger->write(RyzenAIShapes::report(loadedModel->modelDesc, nnXLen, nnYLen));
  }

  if(!xclbins.empty()) {
    if(logger != NULL) {
      // Count only: the shipped grid is a few hundred files, and naming them
      // all put tens of thousands of characters on one line of every log.
      // Which ones actually got loaded is what matters, and verbose dispatch
      // reports that per engine.
      logger->write(
        "RyzenAI backend: found " + Global::uint64ToString((uint64_t)xclbins.size()) +
        " NPU kernel artifacts in " + artifactDir
      );
    }
  }

  ComputeContext* context = new ComputeContext(
    nnXLen,nnYLen,useFP16Mode,artifactDir,dtypeStr,forceNpuOnly,verboseDispatch,maxCols,forceK,gpuIdxs,xclbins
  );
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

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
  if(logger != NULL) {
    logger->write("RyzenAI (AMD NPU) backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("RyzenAI (AMD NPU) backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name + " (" + loadedModel->modelDesc.getShortInfoString() + ")");
  }

  (void)requireExactNNLen; //We don't bother with mask optimizations if we know exact sizes right now.

  if(!inputsUseNHWC)
    throw StringError("RyzenAI backend: inputsUseNHWC = false unsupported (the reference path requires NHWC input)");

  ComputeHandle* handle = new ComputeHandle(context, *loadedModel, maxBatchSize, inputsUseNHWC, gpuIdxForThisThread, logger);
  if(logger != NULL) {
    const string prefix = "RyzenAI (AMD NPU) backend thread " + Global::intToString(serverThreadIdx) + ": ";
    if(handle->accel == NULL)
      logger->write(prefix + "dense layers stay on the CPU (" + handle->accelInitError + ")");
    else if(context->verboseDispatch)
      logger->write(prefix + "device index " + Global::intToString(gpuIdxForThisThread) +
                    ", dense layers offered to the NPU");
  }
  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}


bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  return false;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* handle, bool isWarmup) {
  (void)handle;
  (void)isWarmup;
  return false;
}

//------------------------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = computeHandle->context->nnXLen;
  const int nnYLen = computeHandle->context->nnYLen;
  const ModelDesc& modelDesc = computeHandle->modelDesc;
  const int modelVersion = modelDesc.modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures == modelDesc.numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  const int numPolicyChannels = modelDesc.numPolicyChannels;

  for(int nIdx = 0; nIdx<batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);
    float* rowMetaInput = inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    const bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;
    std::copy(rowGlobal,rowGlobal+numGlobalFeatures,rowGlobalInput);
    if(numMetaFeatures > 0) {
      testAssert(rowMeta != NULL);
      testAssert(hasRowMeta);
      std::copy(rowMeta,rowMeta+numMetaFeatures,rowMetaInput);
    }
    else {
      testAssert(!hasRowMeta);
    }
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, computeHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);
  }

  RyzenAIRef::forward(
    computeHandle->workspace,
    batchSize,
    inputBuffers->spatialInput.data(),
    inputBuffers->globalInput.data(),
    (numMetaFeatures > 0 ? inputBuffers->metaInput.data() : NULL),
    computeHandle->policyBuf.data(),
    computeHandle->policyPassBuf.data(),
    computeHandle->valueBuf.data(),
    computeHandle->scoreValueBuf.data(),
    computeHandle->ownershipBuf.data()
  );

  assert(inputBuffers->singlePolicyPassResultElts == numPolicyChannels);
  assert(inputBuffers->singlePolicyResultElts == numPolicyChannels * nnXLen * nnYLen);

  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  const float* policyData = computeHandle->policyBuf.data();
  const float* policyPassData = computeHandle->policyPassBuf.data();
  const float* valueData = computeHandle->valueBuf.data();
  const float* scoreValueData = computeHandle->scoreValueBuf.data();
  const float* ownershipData = computeHandle->ownershipBuf.data();

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = policyPassData + row * numPolicyChannels;
    const float* policySrcBuf = policyData + row * numPolicyChannels * nnXLen * nnYLen;
    float* policyProbs = output->policyProbs;

    // These are in logits, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    // Handle version >= 12 policy optimism
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
      // NHWC
      for(int i = 0; i<nnXLen*nnYLen; i++) {
        float p = policySrcBuf[i*numPolicyChannels];
        float pOpt = policySrcBuf[i*numPolicyChannels+1];
        policyProbsTmp[i] = p + (pOpt-p) * policyOptimism;
      }
      SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen*nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
    }
    else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[inputBuffers->singlePolicyResultElts] = policyPassSrcBuf[0];
    }

    int numValueChannels = modelDesc.numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = valueData[row * numValueChannels];
    output->whiteLossProb = valueData[row * numValueChannels + 1];
    output->whiteNoResultProb = valueData[row * numValueChannels + 2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = ownershipData + row * nnXLen * nnYLen;
      assert(modelDesc.numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(modelVersion >= 9) {
      int numScoreValueChannels = modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
      output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = scoreValueData[row * numScoreValueChannels + 4];
      output->shorttermScoreError = scoreValueData[row * numScoreValueChannels + 5];
    }
    else if(modelVersion >= 8) {
      int numScoreValueChannels = modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
      output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 4) {
      int numScoreValueChannels = modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 3) {
      int numScoreValueChannels = modelDesc.numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
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

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  cout << RyzenAIDevice::ensureRuntimeLibraryPath() << endl;
  cout << RyzenAIDevice::describeRuntime() << endl;
}

// FOR TESTING ---------------------------------------------------------------------------------------------------------
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
  (void)desc;
  (void)batchSize;
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

#endif  // USE_RYZENAI_BACKEND
