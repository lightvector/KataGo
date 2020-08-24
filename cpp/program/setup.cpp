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
  int defaultNNYLen,
  int defaultMaxBatchSize,
  setup_for_t setupFor
) {
  vector<NNEvaluator*> nnEvals =
    initializeNNEvaluators(
      {nnModelName},{nnModelFile},cfg,logger,seedRand,maxConcurrentEvals,defaultNNXLen,defaultNNYLen,defaultMaxBatchSize,setupFor
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
  int defaultNNYLen,
  int defaultMaxBatchSize,
  setup_for_t setupFor
) {
  vector<NNEvaluator*> nnEvals;
  assert(nnModelNames.size() == nnModelFiles.size());

  #if defined(USE_CUDA_BACKEND)
  string backendPrefix = "cuda";
  #elif defined(USE_OPENCL_BACKEND)
  string backendPrefix = "opencl";
  #elif defined(USE_EIGEN_BACKEND)
  string backendPrefix = "eigen";
  #else
  string backendPrefix = "dummybackend";
  #endif

  //Automatically flag keys that are for other backends as used so that we don't warn about unused keys
  //for those options
  if(backendPrefix != "cuda")
    cfg.markAllKeysUsedWithPrefix("cuda");
  if(backendPrefix != "opencl")
    cfg.markAllKeysUsedWithPrefix("opencl");
  if(backendPrefix != "eigen")
    cfg.markAllKeysUsedWithPrefix("eigen");
  if(backendPrefix != "dummybackend")
    cfg.markAllKeysUsedWithPrefix("dummybackend");

  for(size_t i = 0; i<nnModelFiles.size(); i++) {
    string idxStr = Global::intToString(i);
    const string& nnModelName = nnModelNames[i];
    const string& nnModelFile = nnModelFiles[i];

    bool debugSkipNeuralNetDefault = (nnModelFile == "/dev/null");
    bool debugSkipNeuralNet =
      setupFor == SETUP_FOR_DISTRIBUTED ? false :
      cfg.contains("debugSkipNeuralNet") ? cfg.getBool("debugSkipNeuralNet") :
      debugSkipNeuralNetDefault;

    int nnXLen = std::max(defaultNNXLen,7);
    int nnYLen = std::max(defaultNNYLen,7);
    if(setupFor != SETUP_FOR_DISTRIBUTED) {
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
    }

    bool requireExactNNLen = false;
    if(setupFor != SETUP_FOR_DISTRIBUTED) {
      if(cfg.contains("requireMaxBoardSize" + idxStr))
        requireExactNNLen = cfg.getBool("requireMaxBoardSize" + idxStr);
      else if(cfg.contains("requireMaxBoardSize"))
        requireExactNNLen = cfg.getBool("requireMaxBoardSize");
    }

    bool inputsUseNHWC = backendPrefix == "opencl" ? false : true;
    if(cfg.contains(backendPrefix+"InputsUseNHWC"+idxStr))
      inputsUseNHWC = cfg.getBool(backendPrefix+"InputsUseNHWC"+idxStr);
    else if(cfg.contains("inputsUseNHWC"+idxStr))
      inputsUseNHWC = cfg.getBool("inputsUseNHWC"+idxStr);
    else if(cfg.contains(backendPrefix+"InputsUseNHWC"))
      inputsUseNHWC = cfg.getBool(backendPrefix+"InputsUseNHWC");
    else if(cfg.contains("inputsUseNHWC"))
      inputsUseNHWC = cfg.getBool("inputsUseNHWC");

    bool nnRandomize =
      setupFor == SETUP_FOR_DISTRIBUTED ? true :
      cfg.contains("nnRandomize") ? cfg.getBool("nnRandomize") :
      true;

    string nnRandSeed;
    if(setupFor == SETUP_FOR_DISTRIBUTED)
      nnRandSeed = Global::uint64ToString(seedRand.nextUInt64());
    else if(cfg.contains("nnRandSeed" + idxStr))
      nnRandSeed = cfg.getString("nnRandSeed" + idxStr);
    else if(cfg.contains("nnRandSeed"))
      nnRandSeed = cfg.getString("nnRandSeed");
    else
      nnRandSeed = Global::uint64ToString(seedRand.nextUInt64());
    logger.write("nnRandSeed" + idxStr + " = " + nnRandSeed);

#ifndef USE_EIGEN_BACKEND
    int numNNServerThreadsPerModel =
      cfg.contains("numNNServerThreadsPerModel") ? cfg.getInt("numNNServerThreadsPerModel",1,1024) : 1;
#else
    int numSearchThreads =
      cfg.contains("numSearchThreads0") ? cfg.getInt("numSearchThreads0", 1, 4096) :
      cfg.contains("numSearchThreads") ? cfg.getInt("numSearchThreads", 1, 4096) :
      1;
    int numNNServerThreadsPerModel =
      cfg.contains("numNNServerThreadsPerModel") ? cfg.getInt("numNNServerThreadsPerModel",1,1024) :
      setupFor == SETUP_FOR_DISTRIBUTED ? 16 :
      setupFor == SETUP_FOR_MATCH ? std::max(numSearchThreads*2,16) :
      setupFor == SETUP_FOR_ANALYSIS ? std::max(numSearchThreads*2,16) :
      setupFor == SETUP_FOR_GTP ? numSearchThreads :
      setupFor == SETUP_FOR_BENCHMARK ? numSearchThreads :
      cfg.getInt("numNNServerThreadsPerModel",1,1024);
#endif

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

    string homeDataDirOverride = loadHomeDataDirOverride(cfg);

    string openCLTunerFile;
    if(cfg.contains("openclTunerFile"))
      openCLTunerFile = cfg.getString("openclTunerFile");
    bool openCLReTunePerBoardSize = false;
    if(cfg.contains("openclReTunePerBoardSize"))
      openCLReTunePerBoardSize = cfg.getBool("openclReTunePerBoardSize");

    enabled_t useFP16Mode = enabled_t::Auto;
    if(cfg.contains(backendPrefix+"UseFP16-"+idxStr))
      useFP16Mode = cfg.getEnabled(backendPrefix+"UseFP16-"+idxStr);
    else if(cfg.contains("useFP16-"+idxStr))
      useFP16Mode = cfg.getEnabled("useFP16-"+idxStr);
    else if(cfg.contains(backendPrefix+"UseFP16"))
      useFP16Mode = cfg.getEnabled(backendPrefix+"UseFP16");
    else if(cfg.contains("useFP16"))
      useFP16Mode = cfg.getEnabled("useFP16");

    enabled_t useNHWCMode = enabled_t::Auto;
    if(cfg.contains(backendPrefix+"UseNHWC"+idxStr))
      useNHWCMode = cfg.getEnabled(backendPrefix+"UseNHWC"+idxStr);
    else if(cfg.contains("useNHWC"+idxStr))
      useNHWCMode = cfg.getEnabled("useNHWC"+idxStr);
    else if(cfg.contains(backendPrefix+"UseNHWC"))
      useNHWCMode = cfg.getEnabled(backendPrefix+"UseNHWC");
    else if(cfg.contains("useNHWC"))
      useNHWCMode = cfg.getEnabled("useNHWC");

    int forcedSymmetry = -1;
    if(setupFor != SETUP_FOR_DISTRIBUTED && cfg.contains("nnForcedSymmetry"))
      forcedSymmetry = cfg.getInt("nnForcedSymmetry",0,7);

    logger.write(
      "After dedups: nnModelFile" + idxStr + " = " + nnModelFile
      + " useFP16 " + useFP16Mode.toString()
      + " useNHWC " + useNHWCMode.toString()
    );

    int nnCacheSizePowerOfTwo =
      cfg.contains("nnCacheSizePowerOfTwo") ? cfg.getInt("nnCacheSizePowerOfTwo", -1, 48) :
      setupFor == SETUP_FOR_GTP ? 20 :
      setupFor == SETUP_FOR_BENCHMARK ? 20 :
      setupFor == SETUP_FOR_DISTRIBUTED ? 21 :
      setupFor == SETUP_FOR_MATCH ? 21 :
      setupFor == SETUP_FOR_ANALYSIS ? 23 :
      cfg.getInt("nnCacheSizePowerOfTwo", -1, 48);

    int nnMutexPoolSizePowerOfTwo =
      cfg.contains("nnMutexPoolSizePowerOfTwo") ? cfg.getInt("nnMutexPoolSizePowerOfTwo", -1, 24) :
      setupFor == SETUP_FOR_GTP ? 16 :
      setupFor == SETUP_FOR_BENCHMARK ? 16 :
      setupFor == SETUP_FOR_DISTRIBUTED ? 17 :
      setupFor == SETUP_FOR_MATCH ? 17 :
      setupFor == SETUP_FOR_ANALYSIS ? 17 :
      cfg.getInt("nnMutexPoolSizePowerOfTwo", -1, 24);

#ifndef USE_EIGEN_BACKEND
    int nnMaxBatchSize;
    if(setupFor == SETUP_FOR_BENCHMARK || setupFor == SETUP_FOR_DISTRIBUTED) {
      nnMaxBatchSize = defaultMaxBatchSize;
    }
    else if(defaultMaxBatchSize > 0) {
      nnMaxBatchSize =
        cfg.contains("nnMaxBatchSize") ? cfg.getInt("nnMaxBatchSize", 1, 65536) :
        defaultMaxBatchSize;
    }
    else {
      nnMaxBatchSize = cfg.getInt("nnMaxBatchSize", 1, 65536);
    }
#else
    //Large batches don't really help CPUs the way they do GPUs because a single CPU on its own is single-threaded
    //and doesn't greatly benefit from having a bigger chunk of parallelizable work to do on the large scale.
    //So we just fix a size here that isn't crazy and saves memory, completely ignore what the user would have
    //specified for GPUs.
    int nnMaxBatchSize = 4;
    cfg.markAllKeysUsedWithPrefix("nnMaxBatchSize");
    (void)defaultMaxBatchSize;
#endif

    int defaultSymmetry = forcedSymmetry >= 0 ? forcedSymmetry : 0;

    NNEvaluator* nnEval = new NNEvaluator(
      nnModelName,
      nnModelFile,
      &logger,
      nnMaxBatchSize,
      maxConcurrentEvals,
      nnXLen,
      nnYLen,
      requireExactNNLen,
      inputsUseNHWC,
      nnCacheSizePowerOfTwo,
      nnMutexPoolSizePowerOfTwo,
      debugSkipNeuralNet,
      openCLTunerFile,
      homeDataDirOverride,
      openCLReTunePerBoardSize,
      useFP16Mode,
      useNHWCMode,
      numNNServerThreadsPerModel,
      gpuIdxByServerThread,
      nnRandSeed,
      (forcedSymmetry >= 0 ? false : nnRandomize),
      defaultSymmetry
    );

    nnEval->spawnServerThreads();

    nnEvals.push_back(nnEval);
  }

  return nnEvals;
}

string Setup::loadHomeDataDirOverride(
  ConfigParser& cfg
){
  string homeDataDirOverride;
  if(cfg.contains("homeDataDir"))
    homeDataDirOverride = cfg.getString("homeDataDir");
  return homeDataDirOverride;
}

SearchParams Setup::loadSingleParams(
  ConfigParser& cfg
) {
  vector<SearchParams> paramss = loadParams(cfg);
  if(paramss.size() != 1)
    throw StringError("Config contains parameters for multiple bot configurations, but this KataGo command only supports a single configuration");
  return paramss[0];
}

static Player parsePlayer(const char* field, const string& s) {
  Player pla = C_EMPTY;
  bool suc = PlayerIO::tryParsePlayer(s,pla);
  if(!suc)
    throw StringError("Could not parse player in field " + string(field) + ", should be BLACK or WHITE");
  return pla;
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
    else                                            params.maxPlayoutsPondering = (int64_t)1 << 50;
    if(cfg.contains("maxVisitsPondering"+idxStr)) params.maxVisitsPondering = cfg.getInt64("maxVisitsPondering"+idxStr, (int64_t)1, (int64_t)1 << 50);
    else if(cfg.contains("maxVisitsPondering"))   params.maxVisitsPondering = cfg.getInt64("maxVisitsPondering",        (int64_t)1, (int64_t)1 << 50);
    else                                          params.maxVisitsPondering = (int64_t)1 << 50;
    if(cfg.contains("maxTimePondering"+idxStr)) params.maxTimePondering = cfg.getDouble("maxTimePondering"+idxStr, 0.0, 1.0e20);
    else if(cfg.contains("maxTimePondering"))   params.maxTimePondering = cfg.getDouble("maxTimePondering",        0.0, 1.0e20);
    else                                        params.maxTimePondering = 1.0e20;

    if(cfg.contains("lagBuffer"+idxStr)) params.lagBuffer = cfg.getDouble("lagBuffer"+idxStr, 0.0, 3600.0);
    else if(cfg.contains("lagBuffer"))   params.lagBuffer = cfg.getDouble("lagBuffer",        0.0, 3600.0);
    else                                 params.lagBuffer = 0.0;

    if(cfg.contains("searchFactorAfterOnePass"+idxStr)) params.searchFactorAfterOnePass = cfg.getDouble("searchFactorAfterOnePass"+idxStr, 0.0, 1.0);
    else if(cfg.contains("searchFactorAfterOnePass"))   params.searchFactorAfterOnePass = cfg.getDouble("searchFactorAfterOnePass",        0.0, 1.0);
    if(cfg.contains("searchFactorAfterTwoPass"+idxStr)) params.searchFactorAfterTwoPass = cfg.getDouble("searchFactorAfterTwoPass"+idxStr, 0.0, 1.0);
    else if(cfg.contains("searchFactorAfterTwoPass"))   params.searchFactorAfterTwoPass = cfg.getDouble("searchFactorAfterTwoPass",        0.0, 1.0);

    if(cfg.contains("numSearchThreads"+idxStr)) params.numThreads = cfg.getInt("numSearchThreads"+idxStr, 1, 4096);
    else                                        params.numThreads = cfg.getInt("numSearchThreads",        1, 4096);

    if(cfg.contains("winLossUtilityFactor"+idxStr)) params.winLossUtilityFactor = cfg.getDouble("winLossUtilityFactor"+idxStr, 0.0, 1.0);
    else if(cfg.contains("winLossUtilityFactor"))   params.winLossUtilityFactor = cfg.getDouble("winLossUtilityFactor",        0.0, 1.0);
    else                                            params.winLossUtilityFactor = 1.0;
    if(cfg.contains("staticScoreUtilityFactor"+idxStr)) params.staticScoreUtilityFactor = cfg.getDouble("staticScoreUtilityFactor"+idxStr, 0.0, 1.0);
    else if(cfg.contains("staticScoreUtilityFactor"))   params.staticScoreUtilityFactor = cfg.getDouble("staticScoreUtilityFactor",        0.0, 1.0);
    else                                                params.staticScoreUtilityFactor = 0.1;
    if(cfg.contains("dynamicScoreUtilityFactor"+idxStr)) params.dynamicScoreUtilityFactor = cfg.getDouble("dynamicScoreUtilityFactor"+idxStr, 0.0, 1.0);
    else if(cfg.contains("dynamicScoreUtilityFactor"))   params.dynamicScoreUtilityFactor = cfg.getDouble("dynamicScoreUtilityFactor",        0.0, 1.0);
    else                                                 params.dynamicScoreUtilityFactor = 0.3;
    if(cfg.contains("noResultUtilityForWhite"+idxStr)) params.noResultUtilityForWhite = cfg.getDouble("noResultUtilityForWhite"+idxStr, -1.0, 1.0);
    else if(cfg.contains("noResultUtilityForWhite"))   params.noResultUtilityForWhite = cfg.getDouble("noResultUtilityForWhite",        -1.0, 1.0);
    else                                               params.noResultUtilityForWhite = 0.0;
    if(cfg.contains("drawEquivalentWinsForWhite"+idxStr)) params.drawEquivalentWinsForWhite = cfg.getDouble("drawEquivalentWinsForWhite"+idxStr, 0.0, 1.0);
    else if(cfg.contains("drawEquivalentWinsForWhite"))   params.drawEquivalentWinsForWhite = cfg.getDouble("drawEquivalentWinsForWhite",        0.0, 1.0);
    else                                                  params.drawEquivalentWinsForWhite = 0.5;

    if(cfg.contains("dynamicScoreCenterZeroWeight"+idxStr)) params.dynamicScoreCenterZeroWeight = cfg.getDouble("dynamicScoreCenterZeroWeight"+idxStr, 0.0, 1.0);
    else if(cfg.contains("dynamicScoreCenterZeroWeight"))   params.dynamicScoreCenterZeroWeight = cfg.getDouble("dynamicScoreCenterZeroWeight",        0.0, 1.0);
    else params.dynamicScoreCenterZeroWeight = 0.20;
    if(cfg.contains("dynamicScoreCenterScale"+idxStr)) params.dynamicScoreCenterScale = cfg.getDouble("dynamicScoreCenterScale"+idxStr, 0.2, 5.0);
    else if(cfg.contains("dynamicScoreCenterScale"))   params.dynamicScoreCenterScale = cfg.getDouble("dynamicScoreCenterScale",        0.2, 5.0);
    else params.dynamicScoreCenterScale = 0.75;

    if(cfg.contains("cpuctExploration"+idxStr)) params.cpuctExploration = cfg.getDouble("cpuctExploration"+idxStr, 0.0, 10.0);
    else if(cfg.contains("cpuctExploration"))   params.cpuctExploration = cfg.getDouble("cpuctExploration",        0.0, 10.0);
    else                                        params.cpuctExploration = 0.9;
    if(cfg.contains("cpuctExplorationLog"+idxStr)) params.cpuctExplorationLog = cfg.getDouble("cpuctExplorationLog"+idxStr, 0.0, 10.0);
    else if(cfg.contains("cpuctExplorationLog"))   params.cpuctExplorationLog = cfg.getDouble("cpuctExplorationLog",        0.0, 10.0);
    else                                           params.cpuctExplorationLog = 0.4;
    if(cfg.contains("cpuctExplorationBase"+idxStr)) params.cpuctExplorationBase = cfg.getDouble("cpuctExplorationBase"+idxStr, 10.0, 100000.0);
    else if(cfg.contains("cpuctExplorationBase"))   params.cpuctExplorationBase = cfg.getDouble("cpuctExplorationBase",        10.0, 100000.0);
    else                                            params.cpuctExplorationBase = 500.0;

    if(cfg.contains("fpuReductionMax"+idxStr)) params.fpuReductionMax = cfg.getDouble("fpuReductionMax"+idxStr, 0.0, 2.0);
    else if(cfg.contains("fpuReductionMax"))   params.fpuReductionMax = cfg.getDouble("fpuReductionMax",        0.0, 2.0);
    else params.fpuReductionMax = 0.2;
    if(cfg.contains("fpuLossProp"+idxStr)) params.fpuLossProp = cfg.getDouble("fpuLossProp"+idxStr, 0.0, 1.0);
    else if(cfg.contains("fpuLossProp"))   params.fpuLossProp = cfg.getDouble("fpuLossProp",        0.0, 1.0);
    else                                   params.fpuLossProp = 0.0;
    if(cfg.contains("fpuUseParentAverage"+idxStr)) params.fpuUseParentAverage = cfg.getBool("fpuUseParentAverage"+idxStr);
    else if(cfg.contains("fpuUseParentAverage"))   params.fpuUseParentAverage = cfg.getBool("fpuUseParentAverage");
    else                                           params.fpuUseParentAverage = true;

    if(cfg.contains("valueWeightExponent"+idxStr)) params.valueWeightExponent = cfg.getDouble("valueWeightExponent"+idxStr, 0.0, 1.0);
    else if(cfg.contains("valueWeightExponent")) params.valueWeightExponent = cfg.getDouble("valueWeightExponent", 0.0, 1.0);
    else params.valueWeightExponent = 0.5;

    if(cfg.contains("rootNoiseEnabled"+idxStr)) params.rootNoiseEnabled = cfg.getBool("rootNoiseEnabled"+idxStr);
    else if(cfg.contains("rootNoiseEnabled"))   params.rootNoiseEnabled = cfg.getBool("rootNoiseEnabled");
    else                                        params.rootNoiseEnabled = false;
    if(cfg.contains("rootDirichletNoiseTotalConcentration"+idxStr))
      params.rootDirichletNoiseTotalConcentration = cfg.getDouble("rootDirichletNoiseTotalConcentration"+idxStr, 0.001, 10000.0);
    else if(cfg.contains("rootDirichletNoiseTotalConcentration"))
      params.rootDirichletNoiseTotalConcentration = cfg.getDouble("rootDirichletNoiseTotalConcentration", 0.001, 10000.0);
    else
      params.rootDirichletNoiseTotalConcentration = 10.83;
    if(cfg.contains("rootDirichletNoiseWeight"+idxStr)) params.rootDirichletNoiseWeight = cfg.getDouble("rootDirichletNoiseWeight"+idxStr, 0.0, 1.0);
    else if(cfg.contains("rootDirichletNoiseWeight"))   params.rootDirichletNoiseWeight = cfg.getDouble("rootDirichletNoiseWeight",        0.0, 1.0);
    else                                                params.rootDirichletNoiseWeight = 0.25;

    if(cfg.contains("rootPolicyTemperature"+idxStr)) params.rootPolicyTemperature = cfg.getDouble("rootPolicyTemperature"+idxStr, 0.01, 100.0);
    else if(cfg.contains("rootPolicyTemperature"))   params.rootPolicyTemperature = cfg.getDouble("rootPolicyTemperature",        0.01, 100.0);
    else                                             params.rootPolicyTemperature = 1.0;
    if(cfg.contains("rootPolicyTemperatureEarly"+idxStr)) params.rootPolicyTemperatureEarly = cfg.getDouble("rootPolicyTemperatureEarly"+idxStr, 0.01, 100.0);
    else if(cfg.contains("rootPolicyTemperatureEarly"))   params.rootPolicyTemperatureEarly = cfg.getDouble("rootPolicyTemperatureEarly",        0.01, 100.0);
    else                                                  params.rootPolicyTemperatureEarly = params.rootPolicyTemperature;
    if(cfg.contains("rootFpuReductionMax"+idxStr)) params.rootFpuReductionMax = cfg.getDouble("rootFpuReductionMax"+idxStr, 0.0, 2.0);
    else if(cfg.contains("rootFpuReductionMax"))   params.rootFpuReductionMax = cfg.getDouble("rootFpuReductionMax",        0.0, 2.0);
    else                                           params.rootFpuReductionMax = params.rootNoiseEnabled ? 0.0 : 0.1;
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
    else if(cfg.contains("chosenMoveTemperature"))   params.chosenMoveTemperature = cfg.getDouble("chosenMoveTemperature",        0.0, 5.0);
    else                                             params.chosenMoveTemperature = 0.1;
    if(cfg.contains("chosenMoveTemperatureEarly"+idxStr))
      params.chosenMoveTemperatureEarly = cfg.getDouble("chosenMoveTemperatureEarly"+idxStr, 0.0, 5.0);
    else if(cfg.contains("chosenMoveTemperatureEarly"))
      params.chosenMoveTemperatureEarly = cfg.getDouble("chosenMoveTemperatureEarly",        0.0, 5.0);
    else
      params.chosenMoveTemperatureEarly = 0.5;
    if(cfg.contains("chosenMoveTemperatureHalflife"+idxStr))
      params.chosenMoveTemperatureHalflife = cfg.getDouble("chosenMoveTemperatureHalflife"+idxStr, 0.1, 100000.0);
    else if(cfg.contains("chosenMoveTemperatureHalflife"))
      params.chosenMoveTemperatureHalflife = cfg.getDouble("chosenMoveTemperatureHalflife",        0.1, 100000.0);
    else
      params.chosenMoveTemperatureHalflife = 19;
    if(cfg.contains("chosenMoveSubtract"+idxStr)) params.chosenMoveSubtract = cfg.getDouble("chosenMoveSubtract"+idxStr, 0.0, 1.0e10);
    else if(cfg.contains("chosenMoveSubtract"))   params.chosenMoveSubtract = cfg.getDouble("chosenMoveSubtract",        0.0, 1.0e10);
    else                                          params.chosenMoveSubtract = 0.0;
    if(cfg.contains("chosenMovePrune"+idxStr)) params.chosenMovePrune = cfg.getDouble("chosenMovePrune"+idxStr, 0.0, 1.0e10);
    else if(cfg.contains("chosenMovePrune"))   params.chosenMovePrune = cfg.getDouble("chosenMovePrune",        0.0, 1.0e10);
    else                                       params.chosenMovePrune = 1.0;

    if(cfg.contains("useLcbForSelection"+idxStr)) params.useLcbForSelection = cfg.getBool("useLcbForSelection"+idxStr);
    else if(cfg.contains("useLcbForSelection"))   params.useLcbForSelection = cfg.getBool("useLcbForSelection");
    else                                          params.useLcbForSelection = true;
    if(cfg.contains("lcbStdevs"+idxStr)) params.lcbStdevs = cfg.getDouble("lcbStdevs"+idxStr, 1.0, 12.0);
    else if(cfg.contains("lcbStdevs"))   params.lcbStdevs = cfg.getDouble("lcbStdevs",        1.0, 12.0);
    else                                 params.lcbStdevs = 5.0;
    if(cfg.contains("minVisitPropForLCB"+idxStr)) params.minVisitPropForLCB = cfg.getDouble("minVisitPropForLCB"+idxStr, 0.0, 1.0);
    else if(cfg.contains("minVisitPropForLCB"))   params.minVisitPropForLCB = cfg.getDouble("minVisitPropForLCB",        0.0, 1.0);
    else                                          params.minVisitPropForLCB = 0.15;

    if(cfg.contains("rootEndingBonusPoints"+idxStr)) params.rootEndingBonusPoints = cfg.getDouble("rootEndingBonusPoints"+idxStr, -1.0, 1.0);
    else if(cfg.contains("rootEndingBonusPoints"))   params.rootEndingBonusPoints = cfg.getDouble("rootEndingBonusPoints",        -1.0, 1.0);
    else                                             params.rootEndingBonusPoints = 0.5;
    if(cfg.contains("rootPruneUselessMoves"+idxStr)) params.rootPruneUselessMoves = cfg.getBool("rootPruneUselessMoves"+idxStr);
    else if(cfg.contains("rootPruneUselessMoves"))   params.rootPruneUselessMoves = cfg.getBool("rootPruneUselessMoves");
    else                                             params.rootPruneUselessMoves = true;
    if(cfg.contains("conservativePass"+idxStr)) params.conservativePass = cfg.getBool("conservativePass"+idxStr);
    else if(cfg.contains("conservativePass"))   params.conservativePass = cfg.getBool("conservativePass");
    else                                        params.conservativePass = false;
    if(cfg.contains("fillDameBeforePass"+idxStr)) params.fillDameBeforePass = cfg.getBool("fillDameBeforePass"+idxStr);
    else if(cfg.contains("fillDameBeforePass"))   params.fillDameBeforePass = cfg.getBool("fillDameBeforePass");
    else                                          params.fillDameBeforePass = false;
    //Controlled by GTP directly, not used in any other mode
    params.avoidMYTDaggerHackPla = C_EMPTY;
    if(cfg.contains("wideRootNoise"+idxStr)) params.wideRootNoise = cfg.getDouble("wideRootNoise"+idxStr, 0.0, 5.0);
    else if(cfg.contains("wideRootNoise"))   params.wideRootNoise = cfg.getDouble("wideRootNoise", 0.0, 5.0);
    else                                     params.wideRootNoise = 0.0;

    if(cfg.contains("playoutDoublingAdvantage"+idxStr)) params.playoutDoublingAdvantage = cfg.getDouble("playoutDoublingAdvantage"+idxStr,-3.0,3.0);
    else if(cfg.contains("playoutDoublingAdvantage"))   params.playoutDoublingAdvantage = cfg.getDouble("playoutDoublingAdvantage",-3.0,3.0);
    else                                                params.playoutDoublingAdvantage = 0.0;
    if(cfg.contains("playoutDoublingAdvantagePla"+idxStr)) params.playoutDoublingAdvantagePla = parsePlayer("playoutDoublingAdvantagePla",cfg.getString("playoutDoublingAdvantagePla"+idxStr));
    else if(cfg.contains("playoutDoublingAdvantagePla"))   params.playoutDoublingAdvantagePla = parsePlayer("playoutDoublingAdvantagePla",cfg.getString("playoutDoublingAdvantagePla"));
    else                                                   params.playoutDoublingAdvantagePla = C_EMPTY;

    if(cfg.contains("nnPolicyTemperature"+idxStr))
      params.nnPolicyTemperature = cfg.getFloat("nnPolicyTemperature"+idxStr,0.01f,5.0f);
    else if(cfg.contains("nnPolicyTemperature"))
      params.nnPolicyTemperature = cfg.getFloat("nnPolicyTemperature",0.01f,5.0f);
    else
      params.nnPolicyTemperature = 1.0f;

    if(cfg.contains("antiMirror"+idxStr)) params.antiMirror = cfg.getBool("antiMirror"+idxStr);
    else if(cfg.contains("antiMirror"))   params.antiMirror = cfg.getBool("antiMirror");
    else                                  params.antiMirror = false;

    if(cfg.contains("mutexPoolSize"+idxStr)) params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize"+idxStr, 1, 1 << 24);
    else if(cfg.contains("mutexPoolSize"))   params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize",        1, 1 << 24);
    else                                     params.mutexPoolSize = 16384;
    if(cfg.contains("numVirtualLossesPerThread"+idxStr)) params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread"+idxStr, 1, 1000);
    else if(cfg.contains("numVirtualLossesPerThread"))   params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread",        1, 1000);
    else                                                 params.numVirtualLossesPerThread = 1;

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

  if(cfg.contains("rules")) {
    if(cfg.contains("koRule")) throw StringError("Cannot both specify 'rules' and individual rules like koRule");
    if(cfg.contains("scoringRule")) throw StringError("Cannot both specify 'rules' and individual rules like scoringRule");
    if(cfg.contains("multiStoneSuicideLegal")) throw StringError("Cannot both specify 'rules' and individual rules like multiStoneSuicideLegal");
    if(cfg.contains("hasButton")) throw StringError("Cannot both specify 'rules' and individual rules like hasButton");
    if(cfg.contains("taxRule")) throw StringError("Cannot both specify 'rules' and individual rules like taxRule");
    if(cfg.contains("whiteHandicapBonus")) throw StringError("Cannot both specify 'rules' and individual rules like whiteHandicapBonus");
    if(cfg.contains("whiteBonusPerHandicapStone")) throw StringError("Cannot both specify 'rules' and individual rules like whiteBonusPerHandicapStone");

    rules = Rules::parseRules(cfg.getString("rules"));
  }
  else {
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
  }

  return rules;
}

vector<pair<set<string>,set<string>>> Setup::getMutexKeySets() {
  vector<pair<set<string>,set<string>>> mutexKeySets = {
    std::make_pair<set<string>,set<string>>(
    {"rules"},{"koRule","scoringRule","multiStoneSuicideLegal","taxRule","hasButton","whiteBonusPerHandicapStone","whiteHandicapBonus"}
    ),
  };
  return mutexKeySets;
}
