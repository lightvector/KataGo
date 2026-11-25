// Distributed inference client backend for KataGo, following PhoenixGo's dist/ design.
//
// This backend performs no local neural net computation. The engine (MCTS, GTP loop, ...)
// runs on a machine without GPUs, and every batch of neural net evals is forwarded over
// TCP to a fleet of remote "katago distserver" processes, each of which serves the model
// with a normal local backend (CUDA/TensorRT/OpenCL/Eigen/...). Multiple servers can be
// listed in the config (distServers = host1:port1,host2:port2,...); each NN server thread
// of the engine connects to one of them in round-robin fashion, like PhoenixGo's
// AsyncDistZeroModelClient rotating over its gRPC stubs.
//
// The wire format (neuralnet/distprotocol.h) exchanges canonical NNOutput-shaped
// tensors: the server runs the full local backend pipeline (symmetry, policy optimism
// blending, layout handling) via NeuralNet::getOutput, so this backend is pure
// marshalling.
//
// Config keys (in the usual KataGo config file):
//   distServers = host1:port1,host2:port2,...   (required)
//   distTimeoutSec = 600                        (optional, per-request socket timeout, 0 = none)
//
// Build with -DUSE_BACKEND=DISTRIBUTED.

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/distprotocol.h"

using namespace std;

//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    // Only the model descriptor is needed locally: the actual weights are computed
    // remotely, but we must parse the header to know tensor shapes and to verify the
    // client and server loaded the identical model (sha256 check in the handshake).
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file, expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

void NeuralNet::globalInitialize() {
  DistProto::initNetworking();
}

void NeuralNet::globalCleanup() {
  DistProto::cleanupNetworking();
}

//------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  Logger* logger;
  vector<pair<string,int>> servers; // (host, port)
  int timeoutSeconds;

  ComputeContext(int xLen, int yLen, Logger* lg, const vector<pair<string,int>>& svrs, int timeoutSec)
    : nnXLen(xLen), nnYLen(yLen), logger(lg), servers(svrs), timeoutSeconds(timeoutSec) {}
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;
};

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
  (void)gpuIdxs;
  (void)homeDataDirOverride;

  if(useFP16Mode == enabled_t::True && logger != NULL)
    logger->write("Distributed backend: useFP16 = true ignored; the remote servers decide precision.");

  if(!cfg.contains("distServers"))
    throw StringError("Distributed backend: config must specify distServers = host1:port1,host2:port2,...");

  vector<string> serverStrs = cfg.getStringsNonEmptyTrim("distServers");
  if(serverStrs.empty())
    throw StringError("Distributed backend: distServers is empty");

  vector<pair<string,int>> servers;
  for(const string& s: serverStrs) {
    string host;
    int port;
    DistProto::parseAddress(s, host, port);
    servers.push_back(make_pair(host, port));
  }

  int timeoutSeconds = cfg.contains("distTimeoutSec") ? cfg.getInt("distTimeoutSec", 0, 86400) : 600;

  ComputeContext* context = new ComputeContext(nnXLen, nnYLen, logger, servers, timeoutSeconds);
  (void)loadedModel;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

struct ComputeHandle {
  ComputeContext* context;
  int maxBatchSize;
  int modelVersion;
  DistProto::ModelDims dims;
  std::unique_ptr<DistProto::Connection> conn;
  Logger* logger;
  // Backends may replace the model with an activation-scaled variant and bump
  // postProcessParams.outputScaleMultiplier (e.g. CUDA/ROCm/OpenCL/TensorRT/ONNX scale8).
  // The server reports its effective value in the handshake; the client parses only the
  // model header, so it has to rescale received tensors by serverScale/clientScale to make
  // its own postprocessing reproduce the server's numbers.
  float outputScaleRatio;

  ComputeHandle(ComputeContext* ctx, int maxBatchSz, int mVer, const DistProto::ModelDims& d,
                float scaleRatio)
    : context(ctx), maxBatchSize(maxBatchSz), modelVersion(mVer), dims(d), conn(nullptr),
      logger(ctx->logger), outputScaleRatio(scaleRatio) {}
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
  (void)requireExactNNLen;
  (void)gpuIdxForThisThread;

  const ModelDesc& m = loadedModel->modelDesc;

  DistProto::ModelDims dims;
  dims.nnXLen = context->nnXLen;
  dims.nnYLen = context->nnYLen;
  dims.numInputChannels = m.numInputChannels;
  dims.numInputGlobalChannels = m.numInputGlobalChannels;
  dims.numInputMetaChannels = m.numInputMetaChannels;
  dims.numPolicyChannels = m.numPolicyChannels;
  dims.numValueChannels = m.numValueChannels;
  dims.numScoreValueChannels = m.numScoreValueChannels;
  dims.numOwnershipChannels = m.numOwnershipChannels;

  testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
  testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
  if(m.numInputMetaChannels > 0)
    testAssert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);

  // Round-robin over the configured servers, like PhoenixGo's AsyncDistZeroModelClient
  // rotating requests across its stubs.
  size_t serverIdx = (size_t)serverThreadIdx % context->servers.size();
  const string& host = context->servers[serverIdx].first;
  int port = context->servers[serverIdx].second;

  // The client's own model header still has the untransformed postprocessing scale (the
  // client never builds the net). The server reports what its backend actually ended up
  // with, and every value/policy tensor it sends is in that scale.
  const float clientOutputScale = m.postProcessParams.outputScaleMultiplier;

  float serverOutputScale = clientOutputScale;
  unique_ptr<ComputeHandle> handle(new ComputeHandle(context, maxBatchSize, m.modelVersion, dims, 1.0f));

  handle->conn = DistProto::connect(host, port, context->timeoutSeconds);
  DistProto::clientHandshake(
    *handle->conn,
    m.sha256,
    context->nnXLen,
    context->nnYLen,
    maxBatchSize,
    inputsUseNHWC,
    serverOutputScale
  );
  if(clientOutputScale > 0.0f && serverOutputScale > 0.0f)
    handle->outputScaleRatio = serverOutputScale / clientOutputScale;

  if(logger != NULL) {
    logger->write(
      "Distributed backend thread " + Global::intToString(serverThreadIdx) +
      ": connected to " + host + ":" + Global::intToString(port) +
      " (model " + m.name + " " + m.getShortInfoString() + ", nnXLen " + Global::intToString(context->nnXLen) +
      ", nnYLen " + Global::intToString(context->nnYLen) +
      ", outputScale " + Global::doubleToString(serverOutputScale) +
      "/" + Global::doubleToString(clientOutputScale) + ")"
    );
  }

  return handle.release();
}

void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
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

void NeuralNet::printDevices() {
  cout << "Distributed backend: computation is performed by remote katago distserver processes." << endl;
  cout << "Specify them in the config with distServers = host1:port1,host2:port2,..." << endl;
}

std::string NeuralNet::getRuntimeBackendDetail(ConfigParser& cfg) {
  (void)cfg;
  return std::string("distributed");
}

NeuralNet::BatchPolicy NeuralNet::getBatchPolicy(ConfigParser& cfg) {
  (void)cfg;
  return NeuralNet::BatchPolicy::Dynamic;
}

int NeuralNet::getNumEffectiveDevices(ConfigParser& cfg, const std::vector<int>& gpuIdxByServerThread) {
  (void)gpuIdxByServerThread;
  if(cfg.contains("distServers"))
    return std::max(1, (int)cfg.getStringsNonEmptyTrim("distServers").size());
  return 1;
}

//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    (void)loadedModel;
    (void)nnXLen;
    (void)nnYLen;
    maxBatchSize = maxBatchSz;
  }

  ~InputBuffers() {}

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  delete buffers;
}

//------------------------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  testAssert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  testAssert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  ComputeHandle* handle = computeHandle;
  const DistProto::ModelDims& dims = handle->dims;
  const int nnXLen = dims.nnXLen;
  const int nnYLen = dims.nnYLen;
  const int modelVersion = handle->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = dims.numInputMetaChannels;
  const int xy = nnXLen * nnYLen;

  // The rows arrive from NNEvaluator in canonical orientation in the per-request bufs;
  // the server applies symmetry, policy optimism and layout handling, so this backend
  // is pure marshalling: send the row bufs as-is, receive NNOutput-shaped tensors.
  vector<DistProto::RowInput> rows;
  rows.reserve(batchSize);
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;

    testAssert(rowGlobal != NULL);
    testAssert(rowSpatial != NULL);
    testAssert((int)inputBufs[nIdx]->rowGlobalBuf.size() >= numGlobalFeatures);
    testAssert((int)inputBufs[nIdx]->rowSpatialBuf.size() >= numSpatialFeatures * xy);
    if(numMetaFeatures > 0) {
      testAssert(hasRowMeta);
      testAssert(rowMeta != NULL);
    }
    else {
      testAssert(!hasRowMeta);
    }

    DistProto::RowInput row;
    row.symmetry = inputBufs[nIdx]->symmetry;
    row.policyOptimism = (float)inputBufs[nIdx]->policyOptimism;
    row.includeOwnerMap = inputBufs[nIdx]->includeOwnerMap;
    row.spatial = rowSpatial;
    row.global_ = rowGlobal;
    row.meta = numMetaFeatures > 0 ? rowMeta : NULL;
    rows.push_back(row);
  }

  DistProto::sendEvalRequest(*handle->conn, batchSize, dims, rows);
  DistProto::recvEvalResponseHeader(*handle->conn);

  assert(outputs.size() == (size_t)batchSize);

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    DistProto::RowOutput rowOut;
    rowOut.policyProbs = output->policyProbs;
    float valueBuf[3];
    rowOut.value = valueBuf;
    rowOut.ownership = output->whiteOwnerMap;
    vector<float> scoreValueBuf(dims.numScoreValueChannels);
    rowOut.scoreValue = scoreValueBuf.data();
    DistProto::recvEvalRow(*handle->conn, dims, rowOut);

    // Undo the server-side activation scaling so that the client's own postprocessing
    // (which uses the client model's scale) yields the same numbers as a local eval.
    if(handle->outputScaleRatio != 1.0f) {
      for(int i = 0; i < xy + 1; i++)
        rowOut.policyProbs[i] *= handle->outputScaleRatio;
      for(int i = 0; i < 3; i++)
        valueBuf[i] *= handle->outputScaleRatio;
      for(int i = 0; i < dims.numScoreValueChannels; i++)
        scoreValueBuf[i] *= handle->outputScaleRatio;
    }

    testAssert(dims.numValueChannels == 3);
    output->whiteWinProb = valueBuf[0];
    output->whiteLossProb = valueBuf[1];
    output->whiteNoResultProb = valueBuf[2];

    if(modelVersion >= 9) {
      testAssert(dims.numScoreValueChannels == 6);
      output->whiteScoreMean = scoreValueBuf[0];
      output->whiteScoreMeanSq = scoreValueBuf[1];
      output->whiteLead = scoreValueBuf[2];
      output->varTimeLeft = scoreValueBuf[3];
      output->shorttermWinlossError = scoreValueBuf[4];
      output->shorttermScoreError = scoreValueBuf[5];
    }
    else if(modelVersion >= 8) {
      testAssert(dims.numScoreValueChannels == 4);
      output->whiteScoreMean = scoreValueBuf[0];
      output->whiteScoreMeanSq = scoreValueBuf[1];
      output->whiteLead = scoreValueBuf[2];
      output->varTimeLeft = scoreValueBuf[3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 4) {
      testAssert(dims.numScoreValueChannels == 2);
      output->whiteScoreMean = scoreValueBuf[0];
      output->whiteScoreMeanSq = scoreValueBuf[1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(modelVersion >= 3) {
      testAssert(dims.numScoreValueChannels == 1);
      output->whiteScoreMean = scoreValueBuf[0];
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

//TESTING ----------------------------------------------------------------------------------

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
