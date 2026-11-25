// Distributed inference server for KataGo, following PhoenixGo's dist_zero_model_server design.
//
// This subcommand loads a model with the backend compiled into this binary
// (CUDA/TensorRT/OpenCL/Eigen/...) and serves neural net evals over TCP to remote
// KataGo engines built with -DUSE_BACKEND=DISTRIBUTED. Run one distserver per GPU
// machine (or one per GPU, or several for load balancing); the engine's config lists
// all of them in distServers = host1:port1,host2:port2,...
//
//   katago distserver -model model.bin.gz [-listen 0.0.0.0:17731] [-gpus 0,1]
//                     [-max-batch 256] [-boardsize 19] [-timeout 0]
//
// Each client connection gets its own compute handle and is served on its own thread,
// so multiple engine threads (and multiple engines) can use one server concurrently,
// and -gpus round-robins connections across the listed GPUs.

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/logger.h"
#include "../core/timer.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/distprotocol.h"
#include "../command/commandline.h"
#include "../main.h"

#include <atomic>
#include <thread>

using namespace std;

int MainCmds::distserver(const vector<string>& args) {
#ifdef USE_DISTRIBUTED_BACKEND
  // This binary is a client: its "neural net" is a TCP client to some other distserver.
  // Serving evals out of it would just forward to yet another server (or loop forever).
  (void)args;
  cerr << "Error: this binary was built with -DUSE_BACKEND=DISTRIBUTED, which has no local neural net "
          "backend and therefore cannot serve evals." << endl;
  cerr << "Run 'distserver' from a build made with the backend that owns the GPUs "
          "(CUDA/TENSORRT/OPENCL/EIGEN/ONNX/ROCM), e.g. -DUSE_BACKEND=CUDA." << endl;
  return 1;
#else

  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string listenAddress;
  string gpusStr;
  int maxBatchSize;
  int boardSize;
  int timeoutSeconds;
  try {
    KataGoCommandLine cmd("Run a KataGo distributed inference server");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> listenArg("", "listen", "Address to listen on, host:port", false, "0.0.0.0:17731", "ADDR");
    TCLAP::ValueArg<string> gpusArg("", "gpus", "Comma-separated gpu indices for this server (empty = default device). Connections are round-robined across them", false, "", "GPUS");
    TCLAP::ValueArg<int> maxBatchArg("", "max-batch", "Maximum eval batch size per connection", false, 256, "N");
    TCLAP::ValueArg<int> boardSizeArg("", "boardsize", "Board size (nnXLen/nnYLen) this server serves", false, 19, "N");
    TCLAP::ValueArg<int> timeoutArg("", "timeout", "Per-connection socket timeout in seconds (0 = none)", false, 0, "N");
    cmd.add(listenArg);
    cmd.add(gpusArg);
    cmd.add(maxBatchArg);
    cmd.add(boardSizeArg);
    cmd.add(timeoutArg);
    cmd.setShortUsageArgLimit();

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    listenAddress = listenArg.getValue();
    gpusStr = gpusArg.getValue();
    maxBatchSize = maxBatchArg.getValue();
    boardSize = boardSizeArg.getValue();
    timeoutSeconds = timeoutArg.getValue();

    if(maxBatchSize <= 0 || maxBatchSize > 65536)
      throw StringError("Invalid -max-batch: " + Global::intToString(maxBatchSize));
    if(boardSize <= 0 || boardSize > 50)
      throw StringError("Invalid -boardsize: " + Global::intToString(boardSize));
    if(timeoutSeconds < 0)
      throw StringError("Invalid -timeout: " + Global::intToString(timeoutSeconds));
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for arg " << e.argId() << endl;
    return 1;
  }
  catch(StringError& e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  Logger logger(nullptr, true);

  vector<int> gpuIdxs;
  if(gpusStr.empty()) {
    gpuIdxs.push_back(-1);
  }
  else {
    for(const string& s: Global::split(gpusStr, ',')) {
      string t = Global::trim(s);
      if(t.empty())
        continue;
      gpuIdxs.push_back(Global::stringToInt(t));
    }
    if(gpuIdxs.empty())
      gpuIdxs.push_back(-1);
  }

  string listenHost;
  int listenPort;
  try {
    DistProto::parseAddress(listenAddress, listenHost, listenPort);
  }
  catch(StringError& e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }

  try {
    DistProto::initNetworking();
    NeuralNet::globalInitialize();

    logger.write("Distserver: loading model " + modelFile);
    LoadedModel* loadedModel = NeuralNet::loadModelFile(modelFile, "");
    const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
    logger.write("Distserver: model " + desc.name + " " + desc.getShortInfoString() + " sha256 " + desc.sha256);

    DistProto::ModelDims dims;
    dims.nnXLen = boardSize;
    dims.nnYLen = boardSize;
    dims.numInputChannels = desc.numInputChannels;
    dims.numInputGlobalChannels = desc.numInputGlobalChannels;
    dims.numInputMetaChannels = desc.numInputMetaChannels;
    dims.numPolicyChannels = desc.numPolicyChannels;
    dims.numValueChannels = desc.numValueChannels;
    dims.numScoreValueChannels = desc.numScoreValueChannels;
    dims.numOwnershipChannels = desc.numOwnershipChannels;

    const int nnXLen = dims.nnXLen;
    const int nnYLen = dims.nnYLen;
    const int xy = nnXLen * nnYLen;
    const size_t spatialElts = dims.spatialElts();

    // ComputeContext is shared across all connection threads of this process.
    ComputeContext* computeContext = NeuralNet::createComputeContext(
      gpuIdxs,
      &logger,
      nnXLen,
      nnYLen,
      "",
      enabled_t::False,
      loadedModel,
      cfg
    );

    // Some backends (CUDA/ROCm/OpenCL/TensorRT/ONNX) swap in an activation-scaled
    // variant of the model inside createComputeContext and bump the postprocessing
    // scale to compensate (scale8). desc is a reference into loadedModel, so this read
    // happens after the transform and reports the value the backend will actually use.
    // The client rescales what it receives by serverScale / its own model scale.
    const float outputScaleMultiplier = desc.postProcessParams.outputScaleMultiplier;
    logger.write("Distserver: postprocess output scale " + Global::doubleToString(outputScaleMultiplier));

    std::atomic<int> nextGpuIdx(0);
    std::atomic<long long> numConnections(0);

    string gpusLogStr;
    for(size_t i = 0; i < gpuIdxs.size(); i++) {
      if(i > 0)
        gpusLogStr += ",";
      gpusLogStr += gpuIdxs[i] < 0 ? "default" : Global::intToString(gpuIdxs[i]);
    }

    logger.write(
      "Distserver: listening on " + listenHost + ":" + Global::intToString(listenPort) +
      ", boardsize " + Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
      ", max-batch " + Global::intToString(maxBatchSize) +
      ", gpus " + gpusLogStr
    );

    DistProto::Listener listener(listenHost, listenPort);

    while(true) {
      int clientSock = listener.acceptRaw();
      long long connId = numConnections.fetch_add(1);
      int gpuIdx = gpuIdxs[(size_t)(nextGpuIdx.fetch_add(1)) % gpuIdxs.size()];

      std::thread([&, clientSock, connId, gpuIdx]() {
        std::unique_ptr<DistProto::Connection> conn;
        ComputeHandle* computeHandle = NULL;
        InputBuffers* inputBuffers = NULL;
        bool handshakeDone = false;
        bool clientUseNHWC = false;
        int connMaxBatch = maxBatchSize;
        try {
          conn = DistProto::wrapRawSocket(clientSock, timeoutSeconds);
          DistProto::serverHandshake(
            *conn,
            desc.sha256,
            nnXLen,
            nnYLen,
            maxBatchSize,
            outputScaleMultiplier,
            connMaxBatch,
            clientUseNHWC
          );
          handshakeDone = true;
          if(connMaxBatch > maxBatchSize)
            connMaxBatch = maxBatchSize;

          logger.write(
            "Distserver: connection " + Global::intToString(connId) + " from thread " +
            "(gpu " + (gpuIdx < 0 ? string("default") : Global::intToString(gpuIdx)) + ", max-batch " +
            Global::intToString(connMaxBatch) + ", clientNHWC " + (clientUseNHWC ? string("1") : string("0")) + ")"
          );

          // Create the handle and buffers on this thread: for GPU backends, the handle
          // binds the GPU to the calling thread. Try NCHW first (the classic layout for
          // most GPU backends), fall back to NHWC for backends that require it (e.g.
          // Eigen only supports NHWC).
          bool serverUseNHWC = false;
          try {
            computeHandle = NeuralNet::createComputeHandle(
              computeContext,
              loadedModel,
              &logger,
              connMaxBatch,
              false, // requireExactNNLen
              false, // inputsUseNHWC
              gpuIdx,
              (int)connId
            );
          }
          catch(StringError&) {
            computeHandle = NeuralNet::createComputeHandle(
              computeContext,
              loadedModel,
              &logger,
              connMaxBatch,
              false, // requireExactNNLen
              true,  // inputsUseNHWC
              gpuIdx,
              (int)connId
            );
            serverUseNHWC = true;
          }
          inputBuffers = NeuralNet::createInputBuffers(loadedModel, connMaxBatch, nnXLen, nnYLen);

          const bool needSpatialTranspose = clientUseNHWC != serverUseNHWC;
          vector<float> spatialTransposeBuf(needSpatialTranspose ? spatialElts : 0);

          while(true) {
            int32_t batchSize = conn->readInt32();
            if(batchSize <= 0 || batchSize > connMaxBatch)
              throw StringError("Distserver: invalid batchSize " + Global::intToString(batchSize));

            vector<std::unique_ptr<NNResultBuf>> resultBufsOwned;
            vector<NNResultBuf*> resultBufs;
            vector<std::unique_ptr<NNOutput>> outputsOwned;
            vector<NNOutput*> outputs;
            resultBufsOwned.reserve(batchSize);
            outputsOwned.reserve(batchSize);
            resultBufs.reserve(batchSize);
            outputs.reserve(batchSize);

            for(int row = 0; row < batchSize; row++) {
              int32_t symmetry = conn->readInt32();
              float policyOptimism = conn->readFloat();
              int32_t includeOwnerMap = conn->readInt32();
              if(symmetry < 0 || symmetry >= SymmetryHelpers::NUM_SYMMETRIES)
                throw StringError("Distserver: invalid symmetry " + Global::intToString(symmetry));

              std::unique_ptr<NNResultBuf> buf = std::make_unique<NNResultBuf>();
              buf->symmetry = symmetry;
              buf->policyOptimism = policyOptimism;
              buf->includeOwnerMap = includeOwnerMap != 0;
              buf->rowSpatialBuf.resize(spatialElts);
              buf->rowGlobalBuf.resize((size_t)dims.numInputGlobalChannels);
              buf->hasRowMeta = dims.numInputMetaChannels > 0;
              if(dims.numInputMetaChannels > 0)
                buf->rowMetaBuf.resize((size_t)dims.numInputMetaChannels);

              if(needSpatialTranspose) {
                conn->readFloats(spatialTransposeBuf.data(), spatialElts);
                // src is in the CLIENT's layout, which is only NHWC if the client said so.
                DistProto::transposeTensor(
                  spatialTransposeBuf.data(), buf->rowSpatialBuf.data(), dims.numInputChannels, xy, clientUseNHWC
                );
              }
              else {
                conn->readFloats(buf->rowSpatialBuf.data(), spatialElts);
              }
              conn->readFloats(buf->rowGlobalBuf.data(), (size_t)dims.numInputGlobalChannels);
              if(dims.numInputMetaChannels > 0)
                conn->readFloats(buf->rowMetaBuf.data(), (size_t)dims.numInputMetaChannels);

              resultBufsOwned.push_back(std::move(buf));
              resultBufs.push_back(resultBufsOwned.back().get());

              std::unique_ptr<NNOutput> output = std::make_unique<NNOutput>();
              output->nnXLen = nnXLen;
              output->nnYLen = nnYLen;
              output->whiteOwnerMap = includeOwnerMap != 0 ? new float[xy] : NULL;
              outputsOwned.push_back(std::move(output));
              outputs.push_back(outputsOwned.back().get());
            }

            ClockTimer timer;
            NeuralNet::getOutput(computeHandle, inputBuffers, batchSize, resultBufs.data(), outputs);

            conn->writeInt32(0); // status ok
            for(int row = 0; row < batchSize; row++) {
              const NNOutput* output = outputs[row];
              conn->writeFloats(output->policyProbs, (size_t)(xy + 1));

              float valueBuf[3];
              valueBuf[0] = output->whiteWinProb;
              valueBuf[1] = output->whiteLossProb;
              valueBuf[2] = output->whiteNoResultProb;
              conn->writeFloats(valueBuf, 3);

              float scoreValueBuf[6];
              scoreValueBuf[0] = output->whiteScoreMean;
              scoreValueBuf[1] = output->whiteScoreMeanSq;
              scoreValueBuf[2] = output->whiteLead;
              scoreValueBuf[3] = output->varTimeLeft;
              scoreValueBuf[4] = output->shorttermWinlossError;
              scoreValueBuf[5] = output->shorttermScoreError;
              int numScoreValueChannels = desc.numScoreValueChannels;
              if(desc.modelVersion >= 9)
                testAssert(numScoreValueChannels == 6);
              else if(desc.modelVersion >= 8)
                testAssert(numScoreValueChannels == 4);
              else if(desc.modelVersion >= 4)
                testAssert(numScoreValueChannels == 2);
              else if(desc.modelVersion >= 3)
                testAssert(numScoreValueChannels == 1);
              conn->writeFloats(scoreValueBuf, (size_t)numScoreValueChannels);

              if(outputs[row]->whiteOwnerMap != NULL)
                conn->writeFloats(outputs[row]->whiteOwnerMap, (size_t)xy);
            }
            double elapsed = timer.getSeconds();
            logger.write(
              "Distserver: connection " + Global::intToString(connId) + " eval batch " +
              Global::intToString(batchSize) + " in " + Global::doubleToString(elapsed) + "s"
            );
          }
        }
        catch(StringError& e) {
          logger.write(string("Distserver: connection ") + Global::intToString(connId) + " closed: " + e.what());
          if(handshakeDone && conn != nullptr)
            DistProto::sendError(*conn, e.what());
        }
        catch(...) {
          logger.write(string("Distserver: connection ") + Global::intToString(connId) + " closed: unknown error");
        }

        if(computeHandle != NULL)
          NeuralNet::freeComputeHandle(computeHandle);
        if(inputBuffers != NULL)
          NeuralNet::freeInputBuffers(inputBuffers);
        // conn unique_ptr closes the socket.
      }).detach();
    }

    NeuralNet::freeComputeContext(computeContext);
    NeuralNet::freeLoadedModel(loadedModel);
    NeuralNet::globalCleanup();
    DistProto::cleanupNetworking();
  }
  catch(StringError& e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }
  catch(...) {
    cerr << "Error: unknown exception" << endl;
    return 1;
  }
  return 0;
#endif
}
