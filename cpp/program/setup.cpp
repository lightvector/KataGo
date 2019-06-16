#include "../program/setup.h"

#include "../neuralnet/nninterface.h"

using namespace std;

void Setup::initializeSession(ConfigParser& cfg) {
  (void)cfg;
  NeuralNet::globalInitialize();
}

//TODO allow the same config to specify differing cuda and opencl options
vector<NNEvaluator*> Setup::initializeNNEvaluators(
  const vector<string>& nnModelNames,
  const vector<string>& nnModelFiles,
  ConfigParser& cfg,
  Logger& logger,
  Rand& seedRand,
  int maxConcurrentEvals,
  bool debugSkipNeuralNetDefault,
  bool alwaysIncludeOwnerMap,
  int defaultNNXLen,
  int defaultNNYLen,
  int forcedSymmetry
) {
  vector<NNEvaluator*> nnEvals;
  assert(nnModelNames.size() == nnModelFiles.size());
  for(size_t i = 0; i<nnModelFiles.size(); i++) {
    string idxStr = Global::intToString(i);
    const string& nnModelName = nnModelNames[i];
    const string& nnModelFile = nnModelFiles[i];

    bool debugSkipNeuralNet = cfg.contains("debugSkipNeuralNet") ? cfg.getBool("debugSkipNeuralNet") : debugSkipNeuralNetDefault;
    int modelFileIdx = i;

    int nnXLen = std::max(defaultNNXLen,7);
    int nnYLen = std::max(defaultNNYLen,7);
    if(cfg.contains("maxBoardXSizeForNNBuffer" + idxStr))
      nnXLen = cfg.getInt("maxBoardXSizeForNNBuffer" + idxStr, 7, NNPos::MAX_BOARD_LEN);
    else if(cfg.contains("maxBoardXSizeForNNBuffer"))
      nnXLen = cfg.getInt("maxBoardXSizeForNNBuffer", 7, NNPos::MAX_BOARD_LEN);
    else if(cfg.contains("maxBoardSizeForNNBuffer" + idxStr))
      nnXLen = cfg.getInt("maxBoardSizeForNNBuffer" + idxStr, 7, NNPos::MAX_BOARD_LEN);
    else if(cfg.contains("maxBoardSizeForNNBuffer"))
      nnXLen = cfg.getInt("maxBoardSizeForNNBuffer", 7, NNPos::MAX_BOARD_LEN);

    if(cfg.contains("maxBoardYSizeForNNBuffer" + idxStr))
      nnYLen = cfg.getInt("maxBoardYSizeForNNBuffer" + idxStr, 7, NNPos::MAX_BOARD_LEN);
    else if(cfg.contains("maxBoardYSizeForNNBuffer"))
      nnYLen = cfg.getInt("maxBoardYSizeForNNBuffer", 7, NNPos::MAX_BOARD_LEN);
    else if(cfg.contains("maxBoardSizeForNNBuffer" + idxStr))
      nnYLen = cfg.getInt("maxBoardSizeForNNBuffer" + idxStr, 7, NNPos::MAX_BOARD_LEN);
    else if(cfg.contains("maxBoardSizeForNNBuffer"))
      nnYLen = cfg.getInt("maxBoardSizeForNNBuffer", 7, NNPos::MAX_BOARD_LEN);

    bool requireExactNNLen = false;
    if(cfg.contains("requireMaxBoardSize" + idxStr))
      requireExactNNLen = cfg.getBool("requireMaxBoardSize" + idxStr);
    else if(cfg.contains("requireMaxBoardSize"))
      requireExactNNLen = cfg.getBool("requireMaxBoardSize");

    bool inputsUseNHWC = true;
    if(cfg.contains("inputsUseNHWC"+idxStr))
      inputsUseNHWC = cfg.getBool("inputsUseNHWC"+idxStr);
    else if(cfg.contains("inputsUseNHWC"))
      inputsUseNHWC = cfg.getBool("inputsUseNHWC");

    float nnPolicyTemperature = 1.0f;
    if(cfg.contains("nnPolicyTemperature"+idxStr))
      nnPolicyTemperature = cfg.getFloat("nnPolicyTemperature"+idxStr,0.01f,5.0f);
    else if(cfg.contains("nnPolicyTemperature"))
      nnPolicyTemperature = cfg.getFloat("nnPolicyTemperature",0.01f,5.0f);

    bool nnRandomize = cfg.getBool("nnRandomize");
    string nnRandSeed;
    if(cfg.contains("nnRandSeed" + idxStr))
      nnRandSeed = cfg.getString("nnRandSeed" + idxStr);
    else if(cfg.contains("nnRandSeed"))
      nnRandSeed = cfg.getString("nnRandSeed");
    else
      nnRandSeed = Global::uint64ToString(seedRand.nextUInt64());
    logger.write("nnRandSeed" + idxStr + " = " + nnRandSeed);

    //Parse cuda versions of the various cfg arguments below
    //as a way to support legacy configs. They are equivalent to non-cuda versions of those args.

    int numNNServerThreadsPerModel = cfg.getInt("numNNServerThreadsPerModel",1,1024);
    vector<int> gpuIdxByServerThread;
    for(int j = 0; j<numNNServerThreadsPerModel; j++) {
      string threadIdxStr = Global::intToString(j);
      if(cfg.contains("gpuToUseModel"+idxStr+"Thread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("gpuToUseModel"+idxStr+"Thread"+threadIdxStr,0,1023));
      else if(cfg.contains("cudaGpuToUseModel"+idxStr+"Thread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("cudaGpuToUseModel"+idxStr+"Thread"+threadIdxStr,0,1023));
      else if(cfg.contains("gpuToUseModel"+idxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("gpuToUseModel"+idxStr,0,1023));
      else if(cfg.contains("cudaGpuToUseModel"+idxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("cudaGpuToUseModel"+idxStr,0,1023));
      else if(cfg.contains("gpuToUseThread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("gpuToUseThread"+threadIdxStr,0,1023));
      else if(cfg.contains("cudaGpuToUseThread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("cudaGpuToUseThread"+threadIdxStr,0,1023));
      else if(cfg.contains("gpuToUse"))
        gpuIdxByServerThread.push_back(cfg.getInt("gpuToUse",0,1023));
      else if(cfg.contains("cudaGpuToUse"))
        gpuIdxByServerThread.push_back(cfg.getInt("cudaGpuToUse",0,1023));
      else
        gpuIdxByServerThread.push_back(0);
    }

    vector<int> gpuIdxs = gpuIdxByServerThread;
    std::sort(gpuIdxs.begin(), gpuIdxs.end());
    std::unique(gpuIdxs.begin(), gpuIdxs.end());

    bool useFP16 = false;
    if(cfg.contains("useFP16-"+idxStr))
      useFP16 = cfg.getBool("useFP16-"+idxStr);
    else if(cfg.contains("cudaUseFP16-"+idxStr))
      useFP16 = cfg.getBool("cudaUseFP16-"+idxStr);
    else if(cfg.contains("useFP16"))
      useFP16 = cfg.getBool("useFP16");
    else if(cfg.contains("cudaUseFP16"))
      useFP16 = cfg.getBool("cudaUseFP16");

    bool cudaUseNHWC = false;
    if(cfg.contains("cudaUseNHWC"+idxStr))
      cudaUseNHWC = cfg.getBool("cudaUseNHWC"+idxStr);
    else if(cfg.contains("cudaUseNHWC"))
      cudaUseNHWC = cfg.getBool("cudaUseNHWC");

    logger.write(
      "After dedups: nnModelFile" + idxStr + " = " + nnModelFile
      + " useFP16 " + Global::boolToString(useFP16)
      + " cudaUseNHWC " + Global::boolToString(cudaUseNHWC)
    );

    NNEvaluator* nnEval = new NNEvaluator(
      nnModelName,
      nnModelFile,
      gpuIdxs,
      &logger,
      modelFileIdx,
      cfg.getInt("nnMaxBatchSize", 1, 65536),
      maxConcurrentEvals,
      nnXLen,
      nnYLen,
      requireExactNNLen,
      inputsUseNHWC,
      cfg.getInt("nnCacheSizePowerOfTwo", -1, 48),
      cfg.getInt("nnMutexPoolSizePowerOfTwo", -1, 24),
      debugSkipNeuralNet,
      alwaysIncludeOwnerMap,
      nnPolicyTemperature
    );

    int defaultSymmetry = forcedSymmetry >= 0 ? forcedSymmetry : 0;
    nnEval->spawnServerThreads(
      numNNServerThreadsPerModel,
      (forcedSymmetry >= 0 ? false : nnRandomize),
      nnRandSeed,
      defaultSymmetry,
      logger,
      gpuIdxByServerThread,
      useFP16,
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

    if(cfg.contains("maxPlayouts"+idxStr)) params.maxPlayouts = cfg.getInt64("maxPlayouts"+idxStr, (int64_t)1, (int64_t)1 << 50);
    else if(cfg.contains("maxPlayouts"))   params.maxPlayouts = cfg.getInt64("maxPlayouts",        (int64_t)1, (int64_t)1 << 50);
    if(cfg.contains("maxVisits"+idxStr)) params.maxVisits = cfg.getInt64("maxVisits"+idxStr, (int64_t)1, (int64_t)1 << 50);
    else if(cfg.contains("maxVisits"))   params.maxVisits = cfg.getInt64("maxVisits",        (int64_t)1, (int64_t)1 << 50);
    if(cfg.contains("maxTime"+idxStr)) params.maxTime = cfg.getDouble("maxTime"+idxStr, 0.0, 1.0e20);
    else if(cfg.contains("maxTime"))   params.maxTime = cfg.getDouble("maxTime",        0.0, 1.0e20);

    if(cfg.contains("maxPlayoutsPondering"+idxStr)) params.maxPlayoutsPondering = cfg.getInt64("maxPlayoutsPondering"+idxStr, (int64_t)1, (int64_t)1 << 50);
    else if(cfg.contains("maxPlayoutsPondering"))   params.maxPlayoutsPondering = cfg.getInt64("maxPlayoutsPondering",        (int64_t)1, (int64_t)1 << 50);
    else                                            params.maxPlayoutsPondering = params.maxPlayouts;
    if(cfg.contains("maxVisitsPondering"+idxStr)) params.maxVisitsPondering = cfg.getInt64("maxVisitsPondering"+idxStr, (int64_t)1, (int64_t)1 << 50);
    else if(cfg.contains("maxVisitsPondering"))   params.maxVisitsPondering = cfg.getInt64("maxVisitsPondering",        (int64_t)1, (int64_t)1 << 50);
    else                                          params.maxVisitsPondering = params.maxVisits;
    if(cfg.contains("maxTimePondering"+idxStr)) params.maxTimePondering = cfg.getDouble("maxTimePondering"+idxStr, 0.0, 1.0e20);
    else if(cfg.contains("maxTimePondering"))   params.maxTimePondering = cfg.getDouble("maxTimePondering",        0.0, 1.0e20);
    else                                        params.maxTimePondering = params.maxTime;

    if(cfg.contains("lagBuffer"+idxStr)) params.lagBuffer = cfg.getDouble("lagBuffer"+idxStr, 0.0, 3600.0);
    else if(cfg.contains("lagBuffer"))   params.lagBuffer = cfg.getDouble("lagBuffer",        0.0, 3600.0);
    else                                 params.lagBuffer = 0.0;

    if(cfg.contains("searchFactorAfterOnePass"+idxStr)) params.searchFactorAfterOnePass = cfg.getDouble("searchFactorAfterOnePass"+idxStr, 0.0, 1.0);
    else if(cfg.contains("searchFactorAfterOnePass"))   params.searchFactorAfterOnePass = cfg.getDouble("searchFactorAfterOnePass",        0.0, 1.0);
    if(cfg.contains("searchFactorAfterTwoPass"+idxStr)) params.searchFactorAfterTwoPass = cfg.getDouble("searchFactorAfterTwoPass"+idxStr, 0.0, 1.0);
    else if(cfg.contains("searchFactorAfterTwoPass"))   params.searchFactorAfterTwoPass = cfg.getDouble("searchFactorAfterTwoPass",        0.0, 1.0);

    if(cfg.contains("numSearchThreads"+idxStr)) params.numThreads = cfg.getInt("numSearchThreads"+idxStr, 1, 1024);
    else                                        params.numThreads = cfg.getInt("numSearchThreads",        1, 1024);

    if(cfg.contains("winLossUtilityFactor"+idxStr)) params.winLossUtilityFactor = cfg.getDouble("winLossUtilityFactor"+idxStr, 0.0, 1.0);
    else                                            params.winLossUtilityFactor = cfg.getDouble("winLossUtilityFactor",        0.0, 1.0);
    if(cfg.contains("staticScoreUtilityFactor"+idxStr)) params.staticScoreUtilityFactor = cfg.getDouble("staticScoreUtilityFactor"+idxStr, 0.0, 1.0);
    else                                                params.staticScoreUtilityFactor = cfg.getDouble("staticScoreUtilityFactor",        0.0, 1.0);
    if(cfg.contains("dynamicScoreUtilityFactor"+idxStr)) params.dynamicScoreUtilityFactor = cfg.getDouble("dynamicScoreUtilityFactor"+idxStr, 0.0, 1.0);
    else                                                 params.dynamicScoreUtilityFactor = cfg.getDouble("dynamicScoreUtilityFactor",        0.0, 1.0);
    if(cfg.contains("noResultUtilityForWhite"+idxStr)) params.noResultUtilityForWhite = cfg.getDouble("noResultUtilityForWhite"+idxStr, -1.0, 1.0);
    else                                               params.noResultUtilityForWhite = cfg.getDouble("noResultUtilityForWhite",        -1.0, 1.0);
    if(cfg.contains("drawEquivalentWinsForWhite"+idxStr)) params.drawEquivalentWinsForWhite = cfg.getDouble("drawEquivalentWinsForWhite"+idxStr, 0.0, 1.0);
    else                                           params.drawEquivalentWinsForWhite = cfg.getDouble("drawEquivalentWinsForWhite",        0.0, 1.0);

    if(cfg.contains("cpuctExploration"+idxStr)) params.cpuctExploration = cfg.getDouble("cpuctExploration"+idxStr, 0.0, 10.0);
    else                                        params.cpuctExploration = cfg.getDouble("cpuctExploration",        0.0, 10.0);
    if(cfg.contains("fpuReductionMax"+idxStr)) params.fpuReductionMax = cfg.getDouble("fpuReductionMax"+idxStr, 0.0, 2.0);
    else                                       params.fpuReductionMax = cfg.getDouble("fpuReductionMax",        0.0, 2.0);
    if(cfg.contains("fpuLossProp"+idxStr)) params.fpuLossProp = cfg.getDouble("fpuLossProp"+idxStr, 0.0, 1.0);
    else if(cfg.contains("fpuLossProp"))   params.fpuLossProp = cfg.getDouble("fpuLossProp",        0.0, 1.0);
    else                                   params.fpuLossProp = 0.0;
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
    if(cfg.contains("rootPolicyTemperature"+idxStr)) params.rootPolicyTemperature = cfg.getDouble("rootPolicyTemperature"+idxStr, 0.01, 5.0);
    else if(cfg.contains("rootPolicyTemperature"))   params.rootPolicyTemperature = cfg.getDouble("rootPolicyTemperature",        0.01, 5.0);
    else                                             params.rootPolicyTemperature = 1.0;
    if(cfg.contains("rootFpuReductionMax"+idxStr)) params.rootFpuReductionMax = cfg.getDouble("rootFpuReductionMax"+idxStr, 0.0, 2.0);
    else if(cfg.contains("rootFpuReductionMax"))   params.rootFpuReductionMax = cfg.getDouble("rootFpuReductionMax",        0.0, 2.0);
    else                                           params.rootFpuReductionMax = params.rootNoiseEnabled ? 0.0 : params.fpuReductionMax;
    if(cfg.contains("rootFpuLossProp"+idxStr)) params.rootFpuLossProp = cfg.getDouble("rootFpuLossProp"+idxStr, 0.0, 1.0);
    else if(cfg.contains("rootFpuLossProp"))   params.rootFpuLossProp = cfg.getDouble("rootFpuLossProp",        0.0, 1.0);
    else                                       params.rootFpuLossProp = params.fpuLossProp;

    if(cfg.contains("rootDesiredPerChildVisitsCoeff"+idxStr)) params.rootDesiredPerChildVisitsCoeff = cfg.getDouble("rootDesiredPerChildVisitsCoeff"+idxStr, 0.0, 100.0);
    else if(cfg.contains("rootDesiredPerChildVisitsCoeff"))   params.rootDesiredPerChildVisitsCoeff = cfg.getDouble("rootDesiredPerChildVisitsCoeff",        0.0, 100.0);
    else                                                      params.rootDesiredPerChildVisitsCoeff = 0.0;

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

    if(cfg.contains("useLcbForSelection"+idxStr)) params.useLcbForSelection = cfg.getBool("useLcbForSelection"+idxStr);
    else if(cfg.contains("useLcbForSelection"))   params.useLcbForSelection = cfg.getBool("useLcbForSelection");
    else                                          params.useLcbForSelection = false;
    if(cfg.contains("lcbStdevs"+idxStr)) params.lcbStdevs = cfg.getDouble("lcbStdevs"+idxStr, 1.0, 12.0);
    else if(cfg.contains("lcbStdevs"))   params.lcbStdevs = cfg.getDouble("lcbStdevs",        1.0, 12.0);
    else                                 params.lcbStdevs = 4.0;
    if(cfg.contains("minVisitPropForLCB"+idxStr)) params.minVisitPropForLCB = cfg.getDouble("minVisitPropForLCB"+idxStr, 0.0, 1.0);
    else if(cfg.contains("minVisitPropForLCB"))   params.minVisitPropForLCB = cfg.getDouble("minVisitPropForLCB",        0.0, 1.0);
    else                                          params.minVisitPropForLCB = 0.05;

    if(cfg.contains("rootEndingBonusPoints"+idxStr)) params.rootEndingBonusPoints = cfg.getDouble("rootEndingBonusPoints"+idxStr, -1.0, 1.0);
    else if(cfg.contains("rootEndingBonusPoints"))   params.rootEndingBonusPoints = cfg.getDouble("rootEndingBonusPoints",        -1.0, 1.0);
    else                                             params.rootEndingBonusPoints = 0.0;
    if(cfg.contains("rootPruneUselessMoves"+idxStr)) params.rootPruneUselessMoves = cfg.getBool("rootPruneUselessMoves"+idxStr);
    else if(cfg.contains("rootPruneUselessMoves"))   params.rootPruneUselessMoves = cfg.getBool("rootPruneUselessMoves");
    else                                             params.rootPruneUselessMoves = false;

    if(cfg.contains("mutexPoolSize"+idxStr)) params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize"+idxStr, 1, 1 << 24);
    else                                     params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize",        1, 1 << 24);
    if(cfg.contains("numVirtualLossesPerThread"+idxStr)) params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread"+idxStr, 1, 1000);
    else                                                 params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread",        1, 1000);

    paramss.push_back(params);
  }

  return paramss;
}
