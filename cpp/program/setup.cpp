#include "../program/setup.h"
#include "../neuralnet/nninterface.h"

void Setup::initializeSession(ConfigParser& cfg) {

  string tensorflowGpuVisibleDeviceList;
  if(cfg.contains("tensorflowGpuVisibleDeviceList"))
    tensorflowGpuVisibleDeviceList = cfg.getString("tensorflowGpuVisibleDeviceList");

  double tensorflowPerProcessGpuMemoryFraction = -1;
  if(cfg.contains("tensorflowPerProcessGpuMemoryFraction"))
    tensorflowPerProcessGpuMemoryFraction = cfg.getDouble("tensorflowPerProcessGpuMemoryFraction",0.0,1.0);

  NeuralNet::globalInitialize(tensorflowGpuVisibleDeviceList,tensorflowPerProcessGpuMemoryFraction);
}

vector<NNEvaluator*> Setup::initializeNNEvaluators(
  const vector<string>& nnModelFiles,
  ConfigParser& cfg,
  Logger& logger,
  Rand& seedRand,
  bool debugSkipNeuralNetDefault
) {
  vector<NNEvaluator*> nnEvals;
  for(size_t i = 0; i<nnModelFiles.size(); i++) {
    string idxStr = Global::intToString(i);
    const string& nnModelFile = nnModelFiles[i];

    bool debugSkipNeuralNet = cfg.contains("debugSkipNeuralNet") ? cfg.getBool("debugSkipNeuralNet") : debugSkipNeuralNetDefault;
    int modelFileIdx = i;

    int posLen;
    if(cfg.contains("maxBoardSizeForNNBuffer" + idxStr))
      posLen = cfg.getInt("maxBoardSizeForNNBuffer" + idxStr, 1, NNPos::MAX_BOARD_LEN);
    else if(cfg.contains("maxBoardSizeForNNBuffer"))
      posLen = cfg.getInt("maxBoardSizeForNNBuffer", 1, NNPos::MAX_BOARD_LEN);
    else
      posLen = NNPos::MAX_BOARD_LEN;

    bool inputsUseNHWC = true;
    if(cfg.contains("inputsUseNHWC"+idxStr))
      inputsUseNHWC = cfg.getBool("inputsUseNHWC"+idxStr);
    else if(cfg.contains("inputsUseNHWC"))
      inputsUseNHWC = cfg.getBool("inputsUseNHWC");

    NNEvaluator* nnEval = new NNEvaluator(
      nnModelFile,
      modelFileIdx,
      cfg.getInt("nnMaxBatchSize", 1, 65536),
      posLen,
      inputsUseNHWC,
      cfg.getInt("nnCacheSizePowerOfTwo", -1, 48),
      debugSkipNeuralNet
    );

    bool nnRandomize = cfg.getBool("nnRandomize");
    string nnRandSeed;
    if(cfg.contains("nnRandSeed" + idxStr))
      nnRandSeed = cfg.getString("nnRandSeed" + idxStr);
    else if(cfg.contains("nnRandSeed"))
      nnRandSeed = cfg.getString("nnRandSeed");
    else
      nnRandSeed = Global::uint64ToString(seedRand.nextUInt64());
    logger.write("nnRandSeed" + idxStr + " = " + nnRandSeed);

    int numNNServerThreadsPerModel = cfg.getInt("numNNServerThreadsPerModel",1,1024);
    vector<int> cudaGpuIdxByServerThread;
    for(int j = 0; j<numNNServerThreadsPerModel; j++) {
      string threadIdxStr = Global::intToString(j);
      if(cfg.contains("cudaGpuToUseModel"+idxStr+"Thread"+threadIdxStr))
        cudaGpuIdxByServerThread.push_back(cfg.getInt("cudaGpuToUseModel"+idxStr+"Thread"+threadIdxStr,0,1023));
      else if(cfg.contains("cudaGpuToUseModel"+idxStr))
        cudaGpuIdxByServerThread.push_back(cfg.getInt("cudaGpuToUseModel"+idxStr,0,1023));
      else if(cfg.contains("cudaGpuToUse"))
        cudaGpuIdxByServerThread.push_back(cfg.getInt("cudaGpuToUse",0,1023));
      else
        cudaGpuIdxByServerThread.push_back(0);
    }

    bool cudaUseFP16 = false;
    if(cfg.contains("cudaUseFP16-"+idxStr))
      cudaUseFP16 = cfg.getBool("cudaUseFP16-"+idxStr);
    else if(cfg.contains("cudaUseFP16"))
      cudaUseFP16 = cfg.getBool("cudaUseFP16");

    bool cudaUseNHWC = false;
    if(cfg.contains("cudaUseNHWC"+idxStr))
      cudaUseNHWC = cfg.getBool("cudaUseNHWC"+idxStr);
    else if(cfg.contains("cudaUseNHWC"))
      cudaUseNHWC = cfg.getBool("cudaUseNHWC");

    logger.write(
      "After dedups: nnModelFile" + idxStr + " = " + nnModelFile
      + " useFP16 " + Global::boolToString(cudaUseFP16)
      + " useNHWC " + Global::boolToString(cudaUseNHWC)
    );

    int defaultSymmetry = 0;
    nnEval->spawnServerThreads(
      numNNServerThreadsPerModel,
      nnRandomize,
      nnRandSeed,
      defaultSymmetry,
      logger,
      cudaGpuIdxByServerThread,
      cudaUseFP16,
      cudaUseNHWC
    );

    nnEvals.push_back(nnEval);
  }

  return nnEvals;
}


vector<SearchParams> Setup::loadParams(
  ConfigParser& cfg
) {

  vector<SearchParams> paramss;
  int numBots = 1;
  if(cfg.contains("numBots"))
    numBots = cfg.getInt("numBots",1,1024);

  for(int i = 0; i<numBots; i++) {
    SearchParams params;

    string idxStr = Global::intToString(i);

    if(cfg.contains("maxPlayouts"+idxStr)) params.maxPlayouts = cfg.getUInt64("maxPlayouts"+idxStr, (uint64_t)1, (uint64_t)1 << 62);
    else if(cfg.contains("maxPlayouts"))   params.maxPlayouts = cfg.getUInt64("maxPlayouts",        (uint64_t)1, (uint64_t)1 << 62);
    if(cfg.contains("maxVisits"+idxStr)) params.maxVisits = cfg.getUInt64("maxVisits"+idxStr, (uint64_t)1, (uint64_t)1 << 62);
    else if(cfg.contains("maxVisits"))   params.maxVisits = cfg.getUInt64("maxVisits",        (uint64_t)1, (uint64_t)1 << 62);
    if(cfg.contains("maxTime"+idxStr)) params.maxTime = cfg.getDouble("maxTime"+idxStr, 0.0, 1.0e20);
    else if(cfg.contains("maxTime"))   params.maxTime = cfg.getDouble("maxTime",        0.0, 1.0e20);

    if(cfg.contains("numSearchThreads"+idxStr)) params.numThreads = cfg.getInt("numSearchThreads"+idxStr, 1, 1024);
    else                                        params.numThreads = cfg.getInt("numSearchThreads",        1, 1024);

    if(cfg.contains("winLossUtilityFactor"+idxStr)) params.winLossUtilityFactor = cfg.getDouble("winLossUtilityFactor"+idxStr, 0.0, 1.0);
    else                                            params.winLossUtilityFactor = cfg.getDouble("winLossUtilityFactor",        0.0, 1.0);
    if(cfg.contains("scoreUtilityFactor"+idxStr)) params.scoreUtilityFactor = cfg.getDouble("scoreUtilityFactor"+idxStr, 0.0, 1.0);
    else                                          params.scoreUtilityFactor = cfg.getDouble("scoreUtilityFactor",        0.0, 1.0);
    if(cfg.contains("noResultUtilityForWhite"+idxStr)) params.noResultUtilityForWhite = cfg.getDouble("noResultUtilityForWhite"+idxStr, -1.0, 1.0);
    else                                               params.noResultUtilityForWhite = cfg.getDouble("noResultUtilityForWhite",        -1.0, 1.0);
    if(cfg.contains("drawEquivalentWinsForWhite"+idxStr)) params.drawEquivalentWinsForWhite = cfg.getDouble("drawEquivalentWinsForWhite"+idxStr, 0.0, 1.0);
    else                                           params.drawEquivalentWinsForWhite = cfg.getDouble("drawEquivalentWinsForWhite",        0.0, 1.0);

    if(cfg.contains("cpuctExploration"+idxStr)) params.cpuctExploration = cfg.getDouble("cpuctExploration"+idxStr, 0.0, 10.0);
    else                                        params.cpuctExploration = cfg.getDouble("cpuctExploration",        0.0, 10.0);
    if(cfg.contains("fpuReductionMax"+idxStr)) params.fpuReductionMax = cfg.getDouble("fpuReductionMax"+idxStr, 0.0, 2.0);
    else                                       params.fpuReductionMax = cfg.getDouble("fpuReductionMax",        0.0, 2.0);
    if(cfg.contains("fpuUseParentAverage"+idxStr)) params.fpuUseParentAverage = cfg.getBool("fpuUseParentAverage"+idxStr);
    else if(cfg.contains("fpuUseParentAverage")) params.fpuUseParentAverage = cfg.getBool("fpuUseParentAverage");

    if(cfg.contains("valueWeightExponent"+idxStr)) params.valueWeightExponent = cfg.getDouble("valueWeightExponent"+idxStr, 0.0, 1.0);
    else if(cfg.contains("valueWeightExponent")) params.valueWeightExponent = cfg.getDouble("valueWeightExponent", 0.0, 1.0);
    else params.valueWeightExponent = 0.0;

    if(cfg.contains("visitsExponent"+idxStr)) params.visitsExponent = cfg.getDouble("visitsExponent"+idxStr, 0.0, 1.0);
    else if(cfg.contains("visitsExponent")) params.visitsExponent = cfg.getDouble("visitsExponent", 0.0, 1.0);
    else params.visitsExponent = 1.0;

    if(cfg.contains("scaleParentWeight"+idxStr)) params.scaleParentWeight = cfg.getBool("scaleParentWeight"+idxStr);
    else if(cfg.contains("scaleParentWeight")) params.scaleParentWeight = cfg.getBool("scaleParentWeight");
    else params.scaleParentWeight = true;

    if(cfg.contains("rootNoiseEnabled"+idxStr)) params.rootNoiseEnabled = cfg.getBool("rootNoiseEnabled"+idxStr);
    else                                        params.rootNoiseEnabled = cfg.getBool("rootNoiseEnabled");
    if(cfg.contains("rootDirichletNoiseTotalConcentration"+idxStr))
      params.rootDirichletNoiseTotalConcentration = cfg.getDouble("rootDirichletNoiseTotalConcentration"+idxStr, 0.001, 10000.0);
    else
      params.rootDirichletNoiseTotalConcentration = cfg.getDouble("rootDirichletNoiseTotalConcentration", 0.001, 10000.0);

    if(cfg.contains("rootDirichletNoiseWeight"+idxStr)) params.rootDirichletNoiseWeight = cfg.getDouble("rootDirichletNoiseWeight"+idxStr, 0.0, 1.0);
    else                                                params.rootDirichletNoiseWeight = cfg.getDouble("rootDirichletNoiseWeight",        0.0, 1.0);
    if(cfg.contains("chosenMoveTemperature"+idxStr)) params.chosenMoveTemperature = cfg.getDouble("chosenMoveTemperature"+idxStr, 0.0, 5.0);
    else                                             params.chosenMoveTemperature = cfg.getDouble("chosenMoveTemperature",        0.0, 5.0);
    if(cfg.contains("chosenMoveTemperatureEarly"+idxStr))
      params.chosenMoveTemperatureEarly = cfg.getDouble("chosenMoveTemperatureEarly"+idxStr, 0.0, 5.0);
    else
      params.chosenMoveTemperatureEarly = cfg.getDouble("chosenMoveTemperatureEarly",        0.0, 5.0);
    if(cfg.contains("chosenMoveTemperatureHalflife"+idxStr))
      params.chosenMoveTemperatureHalflife = cfg.getDouble("chosenMoveTemperatureHalflife"+idxStr, 0.1, 100000.0);
    else
      params.chosenMoveTemperatureHalflife = cfg.getDouble("chosenMoveTemperatureHalflife",        0.1, 100000.0);
    if(cfg.contains("chosenMoveSubtract"+idxStr)) params.chosenMoveSubtract = cfg.getDouble("chosenMoveSubtract"+idxStr, 0.0, 1.0e10);
    else                                          params.chosenMoveSubtract = cfg.getDouble("chosenMoveSubtract",        0.0, 1.0e10);
    if(cfg.contains("chosenMovePrune"+idxStr)) params.chosenMovePrune = cfg.getDouble("chosenMovePrune"+idxStr, 0.0, 1.0e10);
    else                                       params.chosenMovePrune = cfg.getDouble("chosenMovePrune",        0.0, 1.0e10);

    if(cfg.contains("rootEndingBonusPoints"+idxStr)) params.rootEndingBonusPoints = cfg.getDouble("rootEndingBonusPoints"+idxStr, -1.0, 1.0);
    else if(cfg.contains("rootEndingBonusPoints"))   params.rootEndingBonusPoints = cfg.getDouble("rootEndingBonusPoints",        -1.0, 1.0);
    else                                             params.rootEndingBonusPoints = 0.0;

    if(cfg.contains("mutexPoolSize"+idxStr)) params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize"+idxStr, 1, 1 << 24);
    else                                     params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize",        1, 1 << 24);
    if(cfg.contains("numVirtualLossesPerThread"+idxStr)) params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread"+idxStr, 1, 1000);
    else                                                 params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread",        1, 1000);

    paramss.push_back(params);
  }

  return paramss;
}
