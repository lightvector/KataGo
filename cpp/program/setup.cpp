#include "../program/setup.h"

#include "../neuralnet/nninterface.h"

using namespace std;

void Setup::initializeSession(ConfigParser& cfg) {
  (void)cfg;
  NeuralNet::globalInitialize();
}

NNEvaluator* Setup::initializeNNEvaluator(
  const string& nnModelName,
  const string& nnModelFile,
  ConfigParser& cfg,
  Logger& logger,
  Rand& seedRand,
  int maxConcurrentEvals,
  int defaultNNXLen,
  int defaultNNYLen
) {
  vector<NNEvaluator*> nnEvals =
    initializeNNEvaluators(
      {nnModelName},{nnModelFile},cfg,logger,seedRand,maxConcurrentEvals,defaultNNXLen,defaultNNYLen
    );
  assert(nnEvals.size() == 1);
  return nnEvals[0];
}

vector<NNEvaluator*> Setup::initializeNNEvaluators(
  const vector<string>& nnModelNames,
  const vector<string>& nnModelFiles,
  ConfigParser& cfg,
  Logger& logger,
  Rand& seedRand,
  int maxConcurrentEvals,
  int defaultNNXLen,
  int defaultNNYLen
) {
  vector<NNEvaluator*> nnEvals;
  assert(nnModelNames.size() == nnModelFiles.size());

  #if defined(USE_CUDA_BACKEND)
  string backendPrefix = "cuda";
  #elif defined(USE_OPENCL_BACKEND)
  string backendPrefix = "opencl";
  #else
  string backendPrefix = "dummybackend";
  #endif

  //Automatically flag keys that are for other backends as used so that we don't warn about unused keys
  //for those options
  if(backendPrefix != "cuda")
    cfg.markAllKeysUsedWithPrefix("cuda");
  if(backendPrefix != "opencl")
    cfg.markAllKeysUsedWithPrefix("opencl");
  if(backendPrefix != "dummybackend")
    cfg.markAllKeysUsedWithPrefix("dummybackend");

  for(size_t i = 0; i<nnModelFiles.size(); i++) {
    string idxStr = Global::intToString(i);
    const string& nnModelName = nnModelNames[i];
    const string& nnModelFile = nnModelFiles[i];

    bool debugSkipNeuralNetDefault = (nnModelFile == "/dev/null");
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

    bool inputsUseNHWC = backendPrefix == "opencl" ? false : true;
    if(cfg.contains(backendPrefix+"InputsUseNHWC"+idxStr))
      inputsUseNHWC = cfg.getBool(backendPrefix+"InputsUseNHWC"+idxStr);
    else if(cfg.contains("inputsUseNHWC"+idxStr))
      inputsUseNHWC = cfg.getBool("inputsUseNHWC"+idxStr);
    else if(cfg.contains(backendPrefix+"InputsUseNHWC"))
      inputsUseNHWC = cfg.getBool(backendPrefix+"InputsUseNHWC");
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


    int numNNServerThreadsPerModel = cfg.getInt("numNNServerThreadsPerModel",1,1024);

    vector<int> gpuIdxByServerThread;
    for(int j = 0; j<numNNServerThreadsPerModel; j++) {
      string threadIdxStr = Global::intToString(j);
      if(cfg.contains(backendPrefix+"DeviceToUseModel"+idxStr+"Thread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt(backendPrefix+"DeviceToUseModel"+idxStr+"Thread"+threadIdxStr,0,1023));
      else if(cfg.contains(backendPrefix+"GpuToUseModel"+idxStr+"Thread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt(backendPrefix+"GpuToUseModel"+idxStr+"Thread"+threadIdxStr,0,1023));
      else if(cfg.contains("deviceToUseModel"+idxStr+"Thread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("deviceToUseModel"+idxStr+"Thread"+threadIdxStr,0,1023));
      else if(cfg.contains("gpuToUseModel"+idxStr+"Thread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("gpuToUseModel"+idxStr+"Thread"+threadIdxStr,0,1023));
      else if(cfg.contains(backendPrefix+"DeviceToUseModel"+idxStr))
        gpuIdxByServerThread.push_back(cfg.getInt(backendPrefix+"DeviceToUseModel"+idxStr,0,1023));
      else if(cfg.contains(backendPrefix+"GpuToUseModel"+idxStr))
        gpuIdxByServerThread.push_back(cfg.getInt(backendPrefix+"GpuToUseModel"+idxStr,0,1023));
      else if(cfg.contains("deviceToUseModel"+idxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("deviceToUseModel"+idxStr,0,1023));
      else if(cfg.contains("gpuToUseModel"+idxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("gpuToUseModel"+idxStr,0,1023));
      else if(cfg.contains(backendPrefix+"DeviceToUseThread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt(backendPrefix+"DeviceToUseThread"+threadIdxStr,0,1023));
      else if(cfg.contains(backendPrefix+"GpuToUseThread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt(backendPrefix+"GpuToUseThread"+threadIdxStr,0,1023));
      else if(cfg.contains("deviceToUseThread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("deviceToUseThread"+threadIdxStr,0,1023));
      else if(cfg.contains("gpuToUseThread"+threadIdxStr))
        gpuIdxByServerThread.push_back(cfg.getInt("gpuToUseThread"+threadIdxStr,0,1023));
      else if(cfg.contains(backendPrefix+"DeviceToUse"))
        gpuIdxByServerThread.push_back(cfg.getInt(backendPrefix+"DeviceToUse",0,1023));
      else if(cfg.contains(backendPrefix+"GpuToUse"))
        gpuIdxByServerThread.push_back(cfg.getInt(backendPrefix+"GpuToUse",0,1023));
      else if(cfg.contains("deviceToUse"))
        gpuIdxByServerThread.push_back(cfg.getInt("deviceToUse",0,1023));
      else if(cfg.contains("gpuToUse"))
        gpuIdxByServerThread.push_back(cfg.getInt("gpuToUse",0,1023));
      else
        gpuIdxByServerThread.push_back(-1);
    }

    string openCLTunerFile;
    if(cfg.contains("openclTunerFile"))
      openCLTunerFile = cfg.getString("openclTunerFile");
    bool openCLReTunePerBoardSize = false;
    if(cfg.contains("openclReTunePerBoardSize"))
      openCLReTunePerBoardSize = cfg.getBool("openclReTunePerBoardSize");

    vector<int> gpuIdxs = gpuIdxByServerThread;
    std::sort(gpuIdxs.begin(), gpuIdxs.end());
    std::unique(gpuIdxs.begin(), gpuIdxs.end());

    bool useFP16 = false;
    if(cfg.contains(backendPrefix+"UseFP16-"+idxStr))
      useFP16 = cfg.getBool(backendPrefix+"UseFP16-"+idxStr);
    else if(cfg.contains("useFP16-"+idxStr))
      useFP16 = cfg.getBool("useFP16-"+idxStr);
    else if(cfg.contains(backendPrefix+"UseFP16"))
      useFP16 = cfg.getBool(backendPrefix+"UseFP16");
    else if(cfg.contains("useFP16"))
      useFP16 = cfg.getBool("useFP16");

    bool useNHWC = false;
    if(cfg.contains(backendPrefix+"UseNHWC"+idxStr))
      useNHWC = cfg.getBool(backendPrefix+"UseNHWC"+idxStr);
    else if(cfg.contains("useNHWC"+idxStr))
      useNHWC = cfg.getBool("useNHWC"+idxStr);
    else if(cfg.contains(backendPrefix+"UseNHWC"))
      useNHWC = cfg.getBool(backendPrefix+"UseNHWC");
    else if(cfg.contains("useNHWC"))
      useNHWC = cfg.getBool("useNHWC");

    int forcedSymmetry = -1;
    if(cfg.contains("nnForcedSymmetry"))
      forcedSymmetry = cfg.getInt("nnForcedSymmetry",0,7);

    logger.write(
      "After dedups: nnModelFile" + idxStr + " = " + nnModelFile
      + " useFP16 " + Global::boolToString(useFP16)
      + " useNHWC " + Global::boolToString(useNHWC)
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
      nnPolicyTemperature,
      openCLTunerFile,
      openCLReTunePerBoardSize
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
      useNHWC
    );

    nnEvals.push_back(nnEval);
  }

  return nnEvals;
}

SearchParams Setup::loadSingleParams(
  ConfigParser& cfg
) {
  vector<SearchParams> paramss = loadParams(cfg);
  if(paramss.size() != 1)
    throw StringError("Config contains parameters for multiple bot configurations, but this KataGo command only supports a single configuration");
  return paramss[0];
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
    else                                                  params.drawEquivalentWinsForWhite = cfg.getDouble("drawEquivalentWinsForWhite",        0.0, 1.0);

    if(cfg.contains("dynamicScoreCenterZeroWeight"+idxStr)) params.dynamicScoreCenterZeroWeight = cfg.getDouble("dynamicScoreCenterZeroWeight"+idxStr, 0.0, 1.0);
    else if(cfg.contains("dynamicScoreCenterZeroWeight"))   params.dynamicScoreCenterZeroWeight = cfg.getDouble("dynamicScoreCenterZeroWeight",        0.0, 1.0);
    else params.dynamicScoreCenterZeroWeight = 0.0;
    if(cfg.contains("dynamicScoreCenterScale"+idxStr)) params.dynamicScoreCenterScale = cfg.getDouble("dynamicScoreCenterScale"+idxStr, 0.2, 5.0);
    else if(cfg.contains("dynamicScoreCenterScale"))   params.dynamicScoreCenterScale = cfg.getDouble("dynamicScoreCenterScale",        0.2, 5.0);
    else params.dynamicScoreCenterScale = 1.0;

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
    if(cfg.contains("rootPolicyTemperature"+idxStr)) params.rootPolicyTemperature = cfg.getDouble("rootPolicyTemperature"+idxStr, 0.01, 100.0);
    else if(cfg.contains("rootPolicyTemperature"))   params.rootPolicyTemperature = cfg.getDouble("rootPolicyTemperature",        0.01, 100.0);
    else                                             params.rootPolicyTemperature = 1.0;
    if(cfg.contains("rootPolicyTemperatureEarly"+idxStr)) params.rootPolicyTemperatureEarly = cfg.getDouble("rootPolicyTemperatureEarly"+idxStr, 0.01, 100.0);
    else if(cfg.contains("rootPolicyTemperatureEarly"))   params.rootPolicyTemperatureEarly = cfg.getDouble("rootPolicyTemperatureEarly",        0.01, 100.0);
    else                                                  params.rootPolicyTemperatureEarly = params.rootPolicyTemperature;
    if(cfg.contains("rootFpuReductionMax"+idxStr)) params.rootFpuReductionMax = cfg.getDouble("rootFpuReductionMax"+idxStr, 0.0, 2.0);
    else if(cfg.contains("rootFpuReductionMax"))   params.rootFpuReductionMax = cfg.getDouble("rootFpuReductionMax",        0.0, 2.0);
    else                                           params.rootFpuReductionMax = params.rootNoiseEnabled ? 0.0 : params.fpuReductionMax;
    if(cfg.contains("rootFpuLossProp"+idxStr)) params.rootFpuLossProp = cfg.getDouble("rootFpuLossProp"+idxStr, 0.0, 1.0);
    else if(cfg.contains("rootFpuLossProp"))   params.rootFpuLossProp = cfg.getDouble("rootFpuLossProp",        0.0, 1.0);
    else                                       params.rootFpuLossProp = params.fpuLossProp;
    if(cfg.contains("rootNumSymmetriesToSample"+idxStr)) params.rootNumSymmetriesToSample = cfg.getInt("rootNumSymmetriesToSample"+idxStr, 1, 16);
    else if(cfg.contains("rootNumSymmetriesToSample"))   params.rootNumSymmetriesToSample = cfg.getInt("rootNumSymmetriesToSample",        1, 16);
    else                                                 params.rootNumSymmetriesToSample = 1;

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
    if(cfg.contains("conservativePass"+idxStr)) params.conservativePass = cfg.getBool("conservativePass"+idxStr);
    else if(cfg.contains("conservativePass"))   params.conservativePass = cfg.getBool("conservativePass");
    else                                        params.conservativePass = false;
    if(cfg.contains("localExplore"+idxStr)) params.localExplore = cfg.getBool("localExplore"+idxStr);
    else if(cfg.contains("localExplore"))   params.localExplore = cfg.getBool("localExplore");
    else                                    params.localExplore = false;

    if(cfg.contains("playoutDoublingAdvantage"+idxStr)) params.playoutDoublingAdvantage = cfg.getDouble("playoutDoublingAdvantage"+idxStr,-3.0,3.0);
    else if(cfg.contains("playoutDoublingAdvantage"))   params.playoutDoublingAdvantage = cfg.getDouble("playoutDoublingAdvantage",-3.0,3.0);
    else                                                params.playoutDoublingAdvantage = 0.0;

    if(cfg.contains("mutexPoolSize"+idxStr)) params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize"+idxStr, 1, 1 << 24);
    else                                     params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize",        1, 1 << 24);
    if(cfg.contains("numVirtualLossesPerThread"+idxStr)) params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread"+idxStr, 1, 1000);
    else                                                 params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread",        1, 1000);

    paramss.push_back(params);
  }

  return paramss;
}

Player Setup::parseReportAnalysisWinrates(
  ConfigParser& cfg, Player defaultPerspective
) {
  if(!cfg.contains("reportAnalysisWinratesAs"))
    return defaultPerspective;

  string sOrig = cfg.getString("reportAnalysisWinratesAs");
  string s = Global::toLower(sOrig);
  if(s == "b" || s == "black")
    return P_BLACK;
  else if(s == "w" || s == "white")
    return P_WHITE;
  else if(s == "sidetomove")
    return C_EMPTY;

  throw StringError("Could not parse config value for reportAnalysisWinratesAs: " + sOrig);
}

Rules Setup::loadSingleRulesExceptForKomi(
  ConfigParser& cfg
) {
  Rules rules;

  string koRule = cfg.getString("koRule", Rules::koRuleStrings());
  string scoringRule = cfg.getString("scoringRule", Rules::scoringRuleStrings());
  bool multiStoneSuicideLegal = cfg.getBool("multiStoneSuicideLegal");
  bool hasButton = cfg.contains("hasButton") ? cfg.getBool("hasButton") : false;
  float komi = 7.5f;

  rules.koRule = Rules::parseKoRule(koRule);
  rules.scoringRule = Rules::parseScoringRule(scoringRule);
  rules.multiStoneSuicideLegal = multiStoneSuicideLegal;
  rules.hasButton = hasButton;
  rules.komi = komi;

  if(cfg.contains("taxRule")) {
    string taxRule = cfg.getString("taxRule", Rules::taxRuleStrings());
    rules.taxRule = Rules::parseTaxRule(taxRule);
  }
  else {
    rules.taxRule = (rules.scoringRule == Rules::SCORING_TERRITORY ? Rules::TAX_SEKI : Rules::TAX_NONE);
  }

  if(rules.hasButton && rules.scoringRule != Rules::SCORING_AREA)
    throw StringError("Config specifies hasButton=true on a scoring system other than AREA");

  //Also handles parsing of legacy option whiteBonusPerHandicapStone
  if(cfg.contains("whiteBonusPerHandicapStone") && cfg.contains("whiteHandicapBonus"))
    throw StringError("May specify only one of whiteBonusPerHandicapStone and whiteHandicapBonus in config");
  else if(cfg.contains("whiteHandicapBonus"))
    rules.whiteHandicapBonusRule = Rules::parseWhiteHandicapBonusRule(cfg.getString("whiteHandicapBonus", Rules::whiteHandicapBonusRuleStrings()));
  else if(cfg.contains("whiteBonusPerHandicapStone")) {
    int whiteBonusPerHandicapStone = cfg.getInt("whiteBonusPerHandicapStone",0,1);
    if(whiteBonusPerHandicapStone == 0)
      rules.whiteHandicapBonusRule = Rules::WHB_ZERO;
    else
      rules.whiteHandicapBonusRule = Rules::WHB_N;
  }
  else
    rules.whiteHandicapBonusRule = Rules::WHB_ZERO;

  return rules;
}
