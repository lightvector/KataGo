#include "../program/setup.h"

#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../core/fileutils.h"
#include "../neuralnet/nninterface.h"
#include "../search/patternbonustable.h"

using namespace std;

void Setup::initializeSession(ConfigParser& cfg) {
  (void)cfg;
  NeuralNet::globalInitialize();
}

std::vector<std::string> Setup::getBackendPrefixes() {
  std::vector<std::string> prefixes;
  prefixes.push_back("cuda");
  prefixes.push_back("trt");
  prefixes.push_back("metal");
  prefixes.push_back("opencl");
  prefixes.push_back("eigen");
  prefixes.push_back("dummybackend");
  return prefixes;
}

NNEvaluator* Setup::initializeNNEvaluator(
  const string& nnModelName,
  const string& nnModelFile,
  const string& expectedSha256,
  ConfigParser& cfg,
  Logger& logger,
  Rand& seedRand,
  int expectedConcurrentEvals,
  int defaultNNXLen,
  int defaultNNYLen,
  int defaultMaxBatchSize,
  bool defaultRequireExactNNLen,
  bool disableFP16,
  setup_for_t setupFor
) {
  vector<NNEvaluator*> nnEvals =
    initializeNNEvaluators(
      {nnModelName},
      {nnModelFile},
      {expectedSha256},
      cfg,
      logger,
      seedRand,
      expectedConcurrentEvals,
      defaultNNXLen,
      defaultNNYLen,
      defaultMaxBatchSize,
      defaultRequireExactNNLen,
      disableFP16,
      setupFor
    );
  assert(nnEvals.size() == 1);
  return nnEvals[0];
}

vector<NNEvaluator*> Setup::initializeNNEvaluators(
  const vector<string>& nnModelNames,
  const vector<string>& nnModelFiles,
  const vector<string>& expectedSha256s,
  ConfigParser& cfg,
  Logger& logger,
  Rand& seedRand,
  int expectedConcurrentEvals,
  int defaultNNXLen,
  int defaultNNYLen,
  int defaultMaxBatchSize,
  bool defaultRequireExactNNLen,
  bool disableFP16,
  setup_for_t setupFor
) {
  vector<NNEvaluator*> nnEvals;
  assert(nnModelNames.size() == nnModelFiles.size());
  assert(expectedSha256s.size() == 0 || expectedSha256s.size() == nnModelFiles.size());

  #if defined(USE_CUDA_BACKEND)
  string backendPrefix = "cuda";
  #elif defined(USE_TENSORRT_BACKEND)
  string backendPrefix = "trt";
  #elif defined(USE_METAL_BACKEND)
  string backendPrefix = "metal";
  #elif defined(USE_OPENCL_BACKEND)
  string backendPrefix = "opencl";
  #elif defined(USE_EIGEN_BACKEND)
  string backendPrefix = "eigen";
  #else
  string backendPrefix = "dummybackend";
  #endif

  //Automatically flag keys that are for other backends as used so that we don't warn about unused keys
  //for those options
  for(const string& prefix: getBackendPrefixes()) {
    if(prefix != backendPrefix)
      cfg.markAllKeysUsedWithPrefix(prefix);
  }

  for(size_t i = 0; i<nnModelFiles.size(); i++) {
    string idxStr = Global::uint64ToString(i);
    const string& nnModelName = nnModelNames[i];
    const string& nnModelFile = nnModelFiles[i];
    const string& expectedSha256 = expectedSha256s.size() > 0 ? expectedSha256s[i]: "";

    bool debugSkipNeuralNetDefault = (nnModelFile == "/dev/null");
    bool debugSkipNeuralNet =
      setupFor == SETUP_FOR_DISTRIBUTED ? debugSkipNeuralNetDefault :
      cfg.contains("debugSkipNeuralNet") ? cfg.getBool("debugSkipNeuralNet") :
      debugSkipNeuralNetDefault;

    int nnXLen = std::max(defaultNNXLen,2);
    int nnYLen = std::max(defaultNNYLen,2);
    if(setupFor != SETUP_FOR_DISTRIBUTED) {
      if(cfg.contains("maxBoardXSizeForNNBuffer" + idxStr))
        nnXLen = cfg.getInt("maxBoardXSizeForNNBuffer" + idxStr, 2, NNPos::MAX_BOARD_LEN);
      else if(cfg.contains("maxBoardXSizeForNNBuffer"))
        nnXLen = cfg.getInt("maxBoardXSizeForNNBuffer", 2, NNPos::MAX_BOARD_LEN);
      else if(cfg.contains("maxBoardSizeForNNBuffer" + idxStr))
        nnXLen = cfg.getInt("maxBoardSizeForNNBuffer" + idxStr, 2, NNPos::MAX_BOARD_LEN);
      else if(cfg.contains("maxBoardSizeForNNBuffer"))
        nnXLen = cfg.getInt("maxBoardSizeForNNBuffer", 2, NNPos::MAX_BOARD_LEN);

      if(cfg.contains("maxBoardYSizeForNNBuffer" + idxStr))
        nnYLen = cfg.getInt("maxBoardYSizeForNNBuffer" + idxStr, 2, NNPos::MAX_BOARD_LEN);
      else if(cfg.contains("maxBoardYSizeForNNBuffer"))
        nnYLen = cfg.getInt("maxBoardYSizeForNNBuffer", 2, NNPos::MAX_BOARD_LEN);
      else if(cfg.contains("maxBoardSizeForNNBuffer" + idxStr))
        nnYLen = cfg.getInt("maxBoardSizeForNNBuffer" + idxStr, 2, NNPos::MAX_BOARD_LEN);
      else if(cfg.contains("maxBoardSizeForNNBuffer"))
        nnYLen = cfg.getInt("maxBoardSizeForNNBuffer", 2, NNPos::MAX_BOARD_LEN);
    }

    bool requireExactNNLen = defaultRequireExactNNLen;
    if(setupFor != SETUP_FOR_DISTRIBUTED) {
      if(cfg.contains("requireMaxBoardSize" + idxStr))
        requireExactNNLen = cfg.getBool("requireMaxBoardSize" + idxStr);
      else if(cfg.contains("requireMaxBoardSize"))
        requireExactNNLen = cfg.getBool("requireMaxBoardSize");
    }

    bool inputsUseNHWC = backendPrefix == "opencl" || backendPrefix == "trt" || backendPrefix == "metal" ? false : true;
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
    (void)expectedConcurrentEvals;
    cfg.markAllKeysUsedWithPrefix("numEigenThreadsPerModel");
    int numNNServerThreadsPerModel =
      cfg.contains("numNNServerThreadsPerModel") ? cfg.getInt("numNNServerThreadsPerModel",1,1024) : 1;
#else
    cfg.markAllKeysUsedWithPrefix("numNNServerThreadsPerModel");
    int numNNServerThreadsPerModel =
      cfg.contains("numEigenThreadsPerModel") ? cfg.getInt("numEigenThreadsPerModel",1,1024) :
      computeDefaultEigenBackendThreads(expectedConcurrentEvals,logger);
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

    enabled_t useINT8Mode = enabled_t::Auto;
    if(cfg.contains(backendPrefix+"UseINT8"+idxStr))
      useINT8Mode = cfg.getEnabled(backendPrefix+"UseINT8"+idxStr);
    else if(cfg.contains("useINT8"+idxStr))
      useINT8Mode = cfg.getEnabled("useINT8"+idxStr);
    else if(cfg.contains(backendPrefix+"UseINT8"))
      useINT8Mode = cfg.getEnabled(backendPrefix+"UseINT8");
    else if(cfg.contains("useINT8"))
      useINT8Mode = cfg.getEnabled("useINT8");

    string int8CalibrationCacheFile;
    if(cfg.contains("int8CalibrationCacheFile"+idxStr))
      int8CalibrationCacheFile = cfg.getString("int8CalibrationCacheFile"+idxStr);
    else if(cfg.contains("int8CalibrationCacheFile"))
      int8CalibrationCacheFile = cfg.getString("int8CalibrationCacheFile");

    enabled_t useFP8Mode = enabled_t::Auto;
    if(cfg.contains(backendPrefix+"UseFP8"+idxStr))
      useFP8Mode = cfg.getEnabled(backendPrefix+"UseFP8"+idxStr);
    else if(cfg.contains("useFP8"+idxStr))
      useFP8Mode = cfg.getEnabled("useFP8"+idxStr);
    else if(cfg.contains(backendPrefix+"UseFP8"))
      useFP8Mode = cfg.getEnabled(backendPrefix+"UseFP8");
    else if(cfg.contains("useFP8"))
      useFP8Mode = cfg.getEnabled("useFP8");

    int forcedSymmetry = -1;
    if(setupFor != SETUP_FOR_DISTRIBUTED && cfg.contains("nnForcedSymmetry"))
      forcedSymmetry = cfg.getInt("nnForcedSymmetry",0,SymmetryHelpers::NUM_SYMMETRIES-1);

    logger.write(
      "After dedups: nnModelFile" + idxStr + " = " + nnModelFile
      + " useFP16 " + useFP16Mode.toString()
      + " useNHWC " + useNHWCMode.toString()
      + " useINT8 " + useINT8Mode.toString()
      + " useFP8 " + useFP8Mode.toString()
      + " int8Calib " + int8CalibrationCacheFile
    );

    int nnCacheSizePowerOfTwo =
      cfg.contains("nnCacheSizePowerOfTwo") ? cfg.getInt("nnCacheSizePowerOfTwo", -1, 48) :
      setupFor == SETUP_FOR_GTP ? 20 :
      setupFor == SETUP_FOR_BENCHMARK ? 20 :
      setupFor == SETUP_FOR_DISTRIBUTED ? 19 :
      setupFor == SETUP_FOR_MATCH ? 21 :
      setupFor == SETUP_FOR_ANALYSIS ? 23 :
      cfg.getInt("nnCacheSizePowerOfTwo", -1, 48);

    int nnMutexPoolSizePowerOfTwo =
      cfg.contains("nnMutexPoolSizePowerOfTwo") ? cfg.getInt("nnMutexPoolSizePowerOfTwo", -1, 24) :
      setupFor == SETUP_FOR_GTP ? 16 :
      setupFor == SETUP_FOR_BENCHMARK ? 16 :
      setupFor == SETUP_FOR_DISTRIBUTED ? 16 :
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
    int nnMaxBatchSize = 2;
    cfg.markAllKeysUsedWithPrefix("nnMaxBatchSize");
    (void)defaultMaxBatchSize;
#endif

    int defaultSymmetry = forcedSymmetry >= 0 ? forcedSymmetry : 0;
    if(disableFP16)
      useFP16Mode = enabled_t::False;

    NNEvaluator* nnEval = new NNEvaluator(
      nnModelName,
      nnModelFile,
      expectedSha256,
      &logger,
      nnMaxBatchSize,
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
      useINT8Mode,
      useFP8Mode,
      int8CalibrationCacheFile,
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

int Setup::computeDefaultEigenBackendThreads(int expectedConcurrentEvals, Logger& logger) {
  auto getNumCores = [&logger]() {
    int numCores = (int)std::thread::hardware_concurrency();
    if(numCores <= 0) {
      logger.write("Could not determine number of cores on this machine, choosing eigen backend threads as if it were 8");
      numCores = 8;
    }
    return numCores;
  };
  return std::min(expectedConcurrentEvals,getNumCores());
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
  ConfigParser& cfg,
  setup_for_t setupFor
) {
  const bool hasHumanModel = false;
  return loadSingleParams(cfg,setupFor,hasHumanModel);
}
SearchParams Setup::loadSingleParams(
  ConfigParser& cfg,
  setup_for_t setupFor,
  bool hasHumanModel
) {
  const bool loadSingleConfigOnly = true;
  vector<SearchParams> paramss = loadParams(cfg, setupFor, hasHumanModel, loadSingleConfigOnly);
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
  ConfigParser& cfg,
  setup_for_t setupFor
) {
  const bool hasHumanModel = false;
  const bool loadSingleConfigOnly = false;
  return loadParams(cfg,setupFor,hasHumanModel,loadSingleConfigOnly);
}

vector<SearchParams> Setup::loadParams(
  ConfigParser& cfg,
  setup_for_t setupFor,
  bool hasHumanModel
) {
  const bool loadSingleConfigOnly = false;
  return loadParams(cfg,setupFor,hasHumanModel,loadSingleConfigOnly);
}

vector<SearchParams> Setup::loadParams(
  ConfigParser& cfg,
  setup_for_t setupFor,
  bool hasHumanModel,
  bool loadSingleConfigOnly
) {

  vector<SearchParams> paramss;
  int numBots = 1;
  if(cfg.contains("numBots"))
    numBots = cfg.getInt("numBots",1,MAX_BOT_PARAMS_FROM_CFG);

  if(loadSingleConfigOnly) {
    if(numBots != 1)
      throw ConfigParsingError("The config for this command cannot have numBots > 0");
  }

  for(int i = 0; i<numBots; i++) {
    SearchParams params;

    string idxStr = loadSingleConfigOnly ? "" : Global::intToString(i);

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

    if(cfg.contains("minPlayoutsPerThread"+idxStr)) params.minPlayoutsPerThread = cfg.getDouble("minPlayoutsPerThread"+idxStr, 0.0, 1.0e20);
    else if(cfg.contains("minPlayoutsPerThread"))   params.minPlayoutsPerThread = cfg.getDouble("minPlayoutsPerThread",        0.0, 1.0e20);
    else                                            params.minPlayoutsPerThread = (setupFor == SETUP_FOR_ANALYSIS || setupFor == SETUP_FOR_GTP) ? 8.0 : 0.0;

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
    else                                        params.cpuctExploration = 1.0;
    if(cfg.contains("cpuctExplorationLog"+idxStr)) params.cpuctExplorationLog = cfg.getDouble("cpuctExplorationLog"+idxStr, 0.0, 10.0);
    else if(cfg.contains("cpuctExplorationLog"))   params.cpuctExplorationLog = cfg.getDouble("cpuctExplorationLog",        0.0, 10.0);
    else                                           params.cpuctExplorationLog = 0.45;
    if(cfg.contains("cpuctExplorationBase"+idxStr)) params.cpuctExplorationBase = cfg.getDouble("cpuctExplorationBase"+idxStr, 10.0, 100000.0);
    else if(cfg.contains("cpuctExplorationBase"))   params.cpuctExplorationBase = cfg.getDouble("cpuctExplorationBase",        10.0, 100000.0);
    else                                            params.cpuctExplorationBase = 500.0;

    if(cfg.contains("cpuctUtilityStdevPrior"+idxStr)) params.cpuctUtilityStdevPrior = cfg.getDouble("cpuctUtilityStdevPrior"+idxStr, 0.0, 10.0);
    else if(cfg.contains("cpuctUtilityStdevPrior"))   params.cpuctUtilityStdevPrior = cfg.getDouble("cpuctUtilityStdevPrior",        0.0, 10.0);
    else                                              params.cpuctUtilityStdevPrior = 0.40;
    if(cfg.contains("cpuctUtilityStdevPriorWeight"+idxStr)) params.cpuctUtilityStdevPriorWeight = cfg.getDouble("cpuctUtilityStdevPriorWeight"+idxStr, 0.0, 100.0);
    else if(cfg.contains("cpuctUtilityStdevPriorWeight"))   params.cpuctUtilityStdevPriorWeight = cfg.getDouble("cpuctUtilityStdevPriorWeight",        0.0, 100.0);
    else                                                    params.cpuctUtilityStdevPriorWeight = 2.0;
    if(cfg.contains("cpuctUtilityStdevScale"+idxStr)) params.cpuctUtilityStdevScale = cfg.getDouble("cpuctUtilityStdevScale"+idxStr, 0.0, 1.0);
    else if(cfg.contains("cpuctUtilityStdevScale"))   params.cpuctUtilityStdevScale = cfg.getDouble("cpuctUtilityStdevScale",        0.0, 1.0);
    else                                              params.cpuctUtilityStdevScale = ((setupFor != SETUP_FOR_DISTRIBUTED && setupFor != SETUP_FOR_OTHER) ? 0.85 : 0.0);


    if(cfg.contains("fpuReductionMax"+idxStr)) params.fpuReductionMax = cfg.getDouble("fpuReductionMax"+idxStr, 0.0, 2.0);
    else if(cfg.contains("fpuReductionMax"))   params.fpuReductionMax = cfg.getDouble("fpuReductionMax",        0.0, 2.0);
    else params.fpuReductionMax = 0.2;
    if(cfg.contains("fpuLossProp"+idxStr)) params.fpuLossProp = cfg.getDouble("fpuLossProp"+idxStr, 0.0, 1.0);
    else if(cfg.contains("fpuLossProp"))   params.fpuLossProp = cfg.getDouble("fpuLossProp",        0.0, 1.0);
    else                                   params.fpuLossProp = 0.0;
    if(cfg.contains("fpuParentWeightByVisitedPolicy"+idxStr)) params.fpuParentWeightByVisitedPolicy = cfg.getBool("fpuParentWeightByVisitedPolicy"+idxStr);
    else if(cfg.contains("fpuParentWeightByVisitedPolicy"))   params.fpuParentWeightByVisitedPolicy = cfg.getBool("fpuParentWeightByVisitedPolicy");
    else                                                      params.fpuParentWeightByVisitedPolicy = (setupFor != SETUP_FOR_DISTRIBUTED);
    if(params.fpuParentWeightByVisitedPolicy) {
      if(cfg.contains("fpuParentWeightByVisitedPolicyPow"+idxStr)) params.fpuParentWeightByVisitedPolicyPow = cfg.getDouble("fpuParentWeightByVisitedPolicyPow"+idxStr, 0.0, 5.0);
      else if(cfg.contains("fpuParentWeightByVisitedPolicyPow"))   params.fpuParentWeightByVisitedPolicyPow = cfg.getDouble("fpuParentWeightByVisitedPolicyPow",        0.0, 5.0);
      else                                                         params.fpuParentWeightByVisitedPolicyPow = 2.0;
    }
    else {
      if(cfg.contains("fpuParentWeight"+idxStr)) params.fpuParentWeight = cfg.getDouble("fpuParentWeight"+idxStr,        0.0, 1.0);
      else if(cfg.contains("fpuParentWeight"))   params.fpuParentWeight = cfg.getDouble("fpuParentWeight",        0.0, 1.0);
      else                                       params.fpuParentWeight = 0.0;
    }

    if(cfg.contains("policyOptimism"+idxStr)) params.policyOptimism = cfg.getDouble("policyOptimism"+idxStr, 0.0, 1.0);
    else if(cfg.contains("policyOptimism"))   params.policyOptimism = cfg.getDouble("policyOptimism",        0.0, 1.0);
    else params.policyOptimism = (setupFor != SETUP_FOR_DISTRIBUTED && setupFor != SETUP_FOR_OTHER) ? 1.0 : 0.0;

    if(cfg.contains("valueWeightExponent"+idxStr)) params.valueWeightExponent = cfg.getDouble("valueWeightExponent"+idxStr, 0.0, 1.0);
    else if(cfg.contains("valueWeightExponent")) params.valueWeightExponent = cfg.getDouble("valueWeightExponent", 0.0, 1.0);
    else params.valueWeightExponent = 0.25;
    if(cfg.contains("useNoisePruning"+idxStr)) params.useNoisePruning = cfg.getBool("useNoisePruning"+idxStr);
    else if(cfg.contains("useNoisePruning"))   params.useNoisePruning = cfg.getBool("useNoisePruning");
    else                                       params.useNoisePruning = (setupFor != SETUP_FOR_DISTRIBUTED && setupFor != SETUP_FOR_OTHER);
    if(cfg.contains("noisePruneUtilityScale"+idxStr)) params.noisePruneUtilityScale = cfg.getDouble("noisePruneUtilityScale"+idxStr, 0.001, 10.0);
    else if(cfg.contains("noisePruneUtilityScale"))   params.noisePruneUtilityScale = cfg.getDouble("noisePruneUtilityScale", 0.001, 10.0);
    else                                              params.noisePruneUtilityScale = 0.15;
    if(cfg.contains("noisePruningCap"+idxStr)) params.noisePruningCap = cfg.getDouble("noisePruningCap"+idxStr, 0.0, 1e50);
    else if(cfg.contains("noisePruningCap"))   params.noisePruningCap = cfg.getDouble("noisePruningCap", 0.0, 1e50);
    else                                       params.noisePruningCap = 1e50;


    if(cfg.contains("useUncertainty"+idxStr)) params.useUncertainty = cfg.getBool("useUncertainty"+idxStr);
    else if(cfg.contains("useUncertainty"))   params.useUncertainty = cfg.getBool("useUncertainty");
    else                                      params.useUncertainty = (setupFor != SETUP_FOR_DISTRIBUTED && setupFor != SETUP_FOR_OTHER);
    if(cfg.contains("uncertaintyCoeff"+idxStr)) params.uncertaintyCoeff = cfg.getDouble("uncertaintyCoeff"+idxStr, 0.0001, 1.0);
    else if(cfg.contains("uncertaintyCoeff"))   params.uncertaintyCoeff = cfg.getDouble("uncertaintyCoeff", 0.0001, 1.0);
    else                                        params.uncertaintyCoeff = 0.25;
    if(cfg.contains("uncertaintyExponent"+idxStr)) params.uncertaintyExponent = cfg.getDouble("uncertaintyExponent"+idxStr, 0.0, 2.0);
    else if(cfg.contains("uncertaintyExponent"))   params.uncertaintyExponent = cfg.getDouble("uncertaintyExponent", 0.0, 2.0);
    else                                           params.uncertaintyExponent = 1.0;
    if(cfg.contains("uncertaintyMaxWeight"+idxStr)) params.uncertaintyMaxWeight = cfg.getDouble("uncertaintyMaxWeight"+idxStr, 1.0, 100.0);
    else if(cfg.contains("uncertaintyMaxWeight"))   params.uncertaintyMaxWeight = cfg.getDouble("uncertaintyMaxWeight", 1.0, 100.0);
    else                                            params.uncertaintyMaxWeight = 8.0;

    if(cfg.contains("useGraphSearch"+idxStr)) params.useGraphSearch = cfg.getBool("useGraphSearch"+idxStr);
    else if(cfg.contains("useGraphSearch"))   params.useGraphSearch = cfg.getBool("useGraphSearch");
    else                                      params.useGraphSearch = (setupFor != SETUP_FOR_DISTRIBUTED);
    if(cfg.contains("graphSearchRepBound"+idxStr)) params.graphSearchRepBound = cfg.getInt("graphSearchRepBound"+idxStr, 3, 50);
    else if(cfg.contains("graphSearchRepBound"))   params.graphSearchRepBound = cfg.getInt("graphSearchRepBound",        3, 50);
    else                                           params.graphSearchRepBound = 11;
    if(cfg.contains("graphSearchCatchUpLeakProb"+idxStr)) params.graphSearchCatchUpLeakProb = cfg.getDouble("graphSearchCatchUpLeakProb"+idxStr, 0.0, 1.0);
    else if(cfg.contains("graphSearchCatchUpLeakProb"))   params.graphSearchCatchUpLeakProb = cfg.getDouble("graphSearchCatchUpLeakProb", 0.0, 1.0);
    else                                                  params.graphSearchCatchUpLeakProb = 0.0;
    // if(cfg.contains("graphSearchCatchUpProp"+idxStr)) params.graphSearchCatchUpProp = cfg.getDouble("graphSearchCatchUpProp"+idxStr, 0.0, 1.0);
    // else if(cfg.contains("graphSearchCatchUpProp"))   params.graphSearchCatchUpProp = cfg.getDouble("graphSearchCatchUpProp", 0.0, 1.0);
    // else                                              params.graphSearchCatchUpProp = 0.0;

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
    if(cfg.contains("rootNumSymmetriesToSample"+idxStr)) params.rootNumSymmetriesToSample = cfg.getInt("rootNumSymmetriesToSample"+idxStr, 1, SymmetryHelpers::NUM_SYMMETRIES);
    else if(cfg.contains("rootNumSymmetriesToSample"))   params.rootNumSymmetriesToSample = cfg.getInt("rootNumSymmetriesToSample",        1, SymmetryHelpers::NUM_SYMMETRIES);
    else                                                 params.rootNumSymmetriesToSample = 1;
    if(cfg.contains("rootSymmetryPruning"+idxStr)) params.rootSymmetryPruning = cfg.getBool("rootSymmetryPruning"+idxStr);
    else if(cfg.contains("rootSymmetryPruning"))   params.rootSymmetryPruning = cfg.getBool("rootSymmetryPruning");
    else                                           params.rootSymmetryPruning = (setupFor == SETUP_FOR_ANALYSIS || setupFor == SETUP_FOR_GTP);

    if(cfg.contains("rootDesiredPerChildVisitsCoeff"+idxStr)) params.rootDesiredPerChildVisitsCoeff = cfg.getDouble("rootDesiredPerChildVisitsCoeff"+idxStr, 0.0, 100.0);
    else if(cfg.contains("rootDesiredPerChildVisitsCoeff"))   params.rootDesiredPerChildVisitsCoeff = cfg.getDouble("rootDesiredPerChildVisitsCoeff",        0.0, 100.0);
    else                                                      params.rootDesiredPerChildVisitsCoeff = 0.0;

    if(cfg.contains("rootPolicyOptimism"+idxStr)) params.rootPolicyOptimism = cfg.getDouble("rootPolicyOptimism"+idxStr, 0.0, 1.0);
    else if(cfg.contains("rootPolicyOptimism"))   params.rootPolicyOptimism = cfg.getDouble("rootPolicyOptimism",        0.0, 1.0);
    else params.rootPolicyOptimism = (setupFor != SETUP_FOR_DISTRIBUTED && setupFor != SETUP_FOR_OTHER) ? std::min(params.policyOptimism, 0.2) : 0.0;

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
    if(cfg.contains("chosenMoveTemperatureOnlyBelowProb"+idxStr)) params.chosenMoveTemperatureOnlyBelowProb = cfg.getDouble("chosenMoveTemperatureOnlyBelowProb"+idxStr, 0.0, 1.0);
    else if(cfg.contains("chosenMoveTemperatureOnlyBelowProb"))   params.chosenMoveTemperatureOnlyBelowProb = cfg.getDouble("chosenMoveTemperatureOnlyBelowProb",        0.0, 1.0);
    else                                                          params.chosenMoveTemperatureOnlyBelowProb = 1.0;
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
    //For distributed and selfplay, we default to buggy LCB for the moment since it has effects on the policy training target.
    if(cfg.contains("useNonBuggyLcb"+idxStr)) params.useNonBuggyLcb = cfg.getBool("useNonBuggyLcb"+idxStr);
    else if(cfg.contains("useNonBuggyLcb"))   params.useNonBuggyLcb = cfg.getBool("useNonBuggyLcb");
    else                                      params.useNonBuggyLcb = (setupFor != SETUP_FOR_DISTRIBUTED && setupFor != SETUP_FOR_OTHER);


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
    else                                     params.wideRootNoise = (setupFor == SETUP_FOR_ANALYSIS ? Setup::DEFAULT_ANALYSIS_WIDE_ROOT_NOISE : 0.00);

    if(cfg.contains("enablePassingHacks"+idxStr)) params.enablePassingHacks = cfg.getBool("enablePassingHacks"+idxStr);
    else if(cfg.contains("enablePassingHacks")) params.enablePassingHacks = cfg.getBool("enablePassingHacks");
    else params.enablePassingHacks = (setupFor == SETUP_FOR_GTP || setupFor == SETUP_FOR_ANALYSIS) ? true : false;

    if(cfg.contains("enableMorePassingHacks"+idxStr)) params.enableMorePassingHacks = cfg.getBool("enableMorePassingHacks"+idxStr);
    else if(cfg.contains("enableMorePassingHacks")) params.enableMorePassingHacks = cfg.getBool("enableMorePassingHacks");
    else params.enableMorePassingHacks = (setupFor == SETUP_FOR_GTP || setupFor == SETUP_FOR_ANALYSIS) ? true : false;

    if(cfg.contains("playoutDoublingAdvantage"+idxStr)) params.playoutDoublingAdvantage = cfg.getDouble("playoutDoublingAdvantage"+idxStr,-3.0,3.0);
    else if(cfg.contains("playoutDoublingAdvantage"))   params.playoutDoublingAdvantage = cfg.getDouble("playoutDoublingAdvantage",-3.0,3.0);
    else                                                params.playoutDoublingAdvantage = 0.0;
    if(cfg.contains("playoutDoublingAdvantagePla"+idxStr)) params.playoutDoublingAdvantagePla = parsePlayer("playoutDoublingAdvantagePla",cfg.getString("playoutDoublingAdvantagePla"+idxStr));
    else if(cfg.contains("playoutDoublingAdvantagePla"))   params.playoutDoublingAdvantagePla = parsePlayer("playoutDoublingAdvantagePla",cfg.getString("playoutDoublingAdvantagePla"));
    else                                                   params.playoutDoublingAdvantagePla = C_EMPTY;

    if(cfg.contains("avoidRepeatedPatternUtility"+idxStr)) params.avoidRepeatedPatternUtility = cfg.getDouble("avoidRepeatedPatternUtility"+idxStr, -3.0, 3.0);
    else if(cfg.contains("avoidRepeatedPatternUtility"))   params.avoidRepeatedPatternUtility = cfg.getDouble("avoidRepeatedPatternUtility", -3.0, 3.0);
    else                                                   params.avoidRepeatedPatternUtility = 0.0;

    if(cfg.contains("nnPolicyTemperature"+idxStr))
      params.nnPolicyTemperature = cfg.getFloat("nnPolicyTemperature"+idxStr,0.01f,5.0f);
    else if(cfg.contains("nnPolicyTemperature"))
      params.nnPolicyTemperature = cfg.getFloat("nnPolicyTemperature",0.01f,5.0f);
    else
      params.nnPolicyTemperature = 1.0f;

    if(cfg.contains("antiMirror"+idxStr)) params.antiMirror = cfg.getBool("antiMirror"+idxStr);
    else if(cfg.contains("antiMirror"))   params.antiMirror = cfg.getBool("antiMirror");
    else                                  params.antiMirror = false;

    if(cfg.contains("ignorePreRootHistory"+idxStr)) params.ignorePreRootHistory = cfg.getBool("ignorePreRootHistory"+idxStr);
    else if(cfg.contains("ignorePreRootHistory"))   params.ignorePreRootHistory = cfg.getBool("ignorePreRootHistory");
    else                                            params.ignorePreRootHistory = (setupFor == SETUP_FOR_ANALYSIS ? Setup::DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY : false);
    if(cfg.contains("ignoreAllHistory"+idxStr)) params.ignoreAllHistory = cfg.getBool("ignoreAllHistory"+idxStr);
    else if(cfg.contains("ignoreAllHistory"))   params.ignoreAllHistory = cfg.getBool("ignoreAllHistory");
    else                                        params.ignoreAllHistory = false;

    if(cfg.contains("subtreeValueBiasFactor"+idxStr)) params.subtreeValueBiasFactor = cfg.getDouble("subtreeValueBiasFactor"+idxStr, 0.0, 1.0);
    else if(cfg.contains("subtreeValueBiasFactor")) params.subtreeValueBiasFactor = cfg.getDouble("subtreeValueBiasFactor", 0.0, 1.0);
    else params.subtreeValueBiasFactor = 0.45;
    if(cfg.contains("subtreeValueBiasFreeProp"+idxStr)) params.subtreeValueBiasFreeProp = cfg.getDouble("subtreeValueBiasFreeProp"+idxStr, 0.0, 1.0);
    else if(cfg.contains("subtreeValueBiasFreeProp")) params.subtreeValueBiasFreeProp = cfg.getDouble("subtreeValueBiasFreeProp", 0.0, 1.0);
    else params.subtreeValueBiasFreeProp = 0.8;
    if(cfg.contains("subtreeValueBiasWeightExponent"+idxStr)) params.subtreeValueBiasWeightExponent = cfg.getDouble("subtreeValueBiasWeightExponent"+idxStr, 0.0, 1.0);
    else if(cfg.contains("subtreeValueBiasWeightExponent")) params.subtreeValueBiasWeightExponent = cfg.getDouble("subtreeValueBiasWeightExponent", 0.0, 1.0);
    else params.subtreeValueBiasWeightExponent = 0.85;

    if(cfg.contains("nodeTableShardsPowerOfTwo"+idxStr)) params.nodeTableShardsPowerOfTwo = cfg.getInt("nodeTableShardsPowerOfTwo"+idxStr, 8, 24);
    else if(cfg.contains("nodeTableShardsPowerOfTwo"))   params.nodeTableShardsPowerOfTwo = cfg.getInt("nodeTableShardsPowerOfTwo",        8, 24);
    else                                                 params.nodeTableShardsPowerOfTwo = 16;
    if(cfg.contains("numVirtualLossesPerThread"+idxStr)) params.numVirtualLossesPerThread = cfg.getDouble("numVirtualLossesPerThread"+idxStr, 0.01, 1000.0);
    else if(cfg.contains("numVirtualLossesPerThread"))   params.numVirtualLossesPerThread = cfg.getDouble("numVirtualLossesPerThread",        0.01, 1000.0);
    else                                                 params.numVirtualLossesPerThread = 1.0;

    if(cfg.contains("treeReuseCarryOverTimeFactor"+idxStr)) params.treeReuseCarryOverTimeFactor = cfg.getDouble("treeReuseCarryOverTimeFactor"+idxStr,0.0,1.0);
    else if(cfg.contains("treeReuseCarryOverTimeFactor"))   params.treeReuseCarryOverTimeFactor = cfg.getDouble("treeReuseCarryOverTimeFactor",0.0,1.0);
    else                                                    params.treeReuseCarryOverTimeFactor = 0.0;
    if(cfg.contains("overallocateTimeFactor"+idxStr)) params.overallocateTimeFactor = cfg.getDouble("overallocateTimeFactor"+idxStr,0.01,100.0);
    else if(cfg.contains("overallocateTimeFactor"))   params.overallocateTimeFactor = cfg.getDouble("overallocateTimeFactor",0.01,100.0);
    else                                              params.overallocateTimeFactor = 1.0;
    if(cfg.contains("midgameTimeFactor"+idxStr)) params.midgameTimeFactor = cfg.getDouble("midgameTimeFactor"+idxStr,0.01,100.0);
    else if(cfg.contains("midgameTimeFactor"))   params.midgameTimeFactor = cfg.getDouble("midgameTimeFactor",0.01,100.0);
    else                                         params.midgameTimeFactor = 1.0;
    if(cfg.contains("midgameTurnPeakTime"+idxStr)) params.midgameTurnPeakTime = cfg.getDouble("midgameTurnPeakTime"+idxStr,0.0,1000.0);
    else if(cfg.contains("midgameTurnPeakTime"))   params.midgameTurnPeakTime = cfg.getDouble("midgameTurnPeakTime",0.0,1000.0);
    else                                           params.midgameTurnPeakTime = 130.0;
    if(cfg.contains("endgameTurnTimeDecay"+idxStr)) params.endgameTurnTimeDecay = cfg.getDouble("endgameTurnTimeDecay"+idxStr,0.0,1000.0);
    else if(cfg.contains("endgameTurnTimeDecay"))   params.endgameTurnTimeDecay = cfg.getDouble("endgameTurnTimeDecay",0.0,1000.0);
    else                                            params.endgameTurnTimeDecay = 100.0;
    if(cfg.contains("obviousMovesTimeFactor"+idxStr)) params.obviousMovesTimeFactor = cfg.getDouble("obviousMovesTimeFactor"+idxStr,0.01,1.0);
    else if(cfg.contains("obviousMovesTimeFactor"))   params.obviousMovesTimeFactor = cfg.getDouble("obviousMovesTimeFactor",0.01,1.0);
    else                                              params.obviousMovesTimeFactor = 1.0;
    if(cfg.contains("obviousMovesPolicyEntropyTolerance"+idxStr)) params.obviousMovesPolicyEntropyTolerance = cfg.getDouble("obviousMovesPolicyEntropyTolerance"+idxStr,0.001,2.0);
    else if(cfg.contains("obviousMovesPolicyEntropyTolerance"))   params.obviousMovesPolicyEntropyTolerance = cfg.getDouble("obviousMovesPolicyEntropyTolerance",0.001,2.0);
    else                                                          params.obviousMovesPolicyEntropyTolerance = 0.30;
    if(cfg.contains("obviousMovesPolicySurpriseTolerance"+idxStr)) params.obviousMovesPolicySurpriseTolerance = cfg.getDouble("obviousMovesPolicySurpriseTolerance"+idxStr,0.001,2.0);
    else if(cfg.contains("obviousMovesPolicySurpriseTolerance"))   params.obviousMovesPolicySurpriseTolerance = cfg.getDouble("obviousMovesPolicySurpriseTolerance",0.001,2.0);
    else                                                           params.obviousMovesPolicySurpriseTolerance = 0.15;
    if(cfg.contains("futileVisitsThreshold"+idxStr)) params.futileVisitsThreshold = cfg.getDouble("futileVisitsThreshold"+idxStr,0.01,1.0);
    else if(cfg.contains("futileVisitsThreshold"))   params.futileVisitsThreshold = cfg.getDouble("futileVisitsThreshold",0.01,1.0);
    else                                             params.futileVisitsThreshold = 0.0;

    // This does NOT report an error under throwHumanParsingError like the parameters below that expect a second model
    // because the user might be providing the human model as the MAIN model. In which case humanSLProfile is still a
    // valid param but the others are not.
    if(setupFor != SETUP_FOR_DISTRIBUTED) {
      string humanSLProfileName;
      if(cfg.contains("humanSLProfile"+idxStr)) humanSLProfileName = cfg.getString("humanSLProfile"+idxStr);
      else if(cfg.contains("humanSLProfile"))   humanSLProfileName = cfg.getString("humanSLProfile");
      params.humanSLProfile = SGFMetadata::getProfile(humanSLProfileName);
    }

    auto throwHumanParsingError = [](const string& param) {
      throw ConfigParsingError(
        string("Provided parameter ") + param + string(" but no human model was specified (e.g -human-model b18c384nbt-humanv0.bin.gz)")
      );
    };

    if(!hasHumanModel && cfg.contains("humanSLCpuctExploration"+idxStr)) throwHumanParsingError("humanSLCpuctExploration"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLCpuctExploration")) throwHumanParsingError("humanSLCpuctExploration");
    else if(cfg.contains("humanSLCpuctExploration"+idxStr)) params.humanSLCpuctExploration = cfg.getDouble("humanSLCpuctExploration"+idxStr, 0.0, 1000.0);
    else if(cfg.contains("humanSLCpuctExploration"))   params.humanSLCpuctExploration = cfg.getDouble("humanSLCpuctExploration",        0.0, 1000.0);
    else                                               params.humanSLCpuctExploration = 1.0;
    if(!hasHumanModel && cfg.contains("humanSLCpuctPermanent"+idxStr)) throwHumanParsingError("humanSLCpuctPermanent"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLCpuctPermanent")) throwHumanParsingError("humanSLCpuctPermanent");
    else if(cfg.contains("humanSLCpuctPermanent"+idxStr)) params.humanSLCpuctPermanent = cfg.getDouble("humanSLCpuctPermanent"+idxStr, 0.0, 1000.0);
    else if(cfg.contains("humanSLCpuctPermanent"))   params.humanSLCpuctPermanent = cfg.getDouble("humanSLCpuctPermanent",        0.0, 1000.0);
    else                                             params.humanSLCpuctPermanent = 0.0;
    if(!hasHumanModel && cfg.contains("humanSLRootExploreProbWeightless"+idxStr)) throwHumanParsingError("humanSLRootExploreProbWeightless"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLRootExploreProbWeightless")) throwHumanParsingError("humanSLRootExploreProbWeightless");
    else if(cfg.contains("humanSLRootExploreProbWeightless"+idxStr)) params.humanSLRootExploreProbWeightless = cfg.getDouble("humanSLRootExploreProbWeightless"+idxStr, 0.0, 1.0);
    else if(cfg.contains("humanSLRootExploreProbWeightless"))   params.humanSLRootExploreProbWeightless = cfg.getDouble("humanSLRootExploreProbWeightless",        0.0, 1.0);
    else                                                        params.humanSLRootExploreProbWeightless = 0.0;
    if(!hasHumanModel && cfg.contains("humanSLRootExploreProbWeightful"+idxStr)) throwHumanParsingError("humanSLRootExploreProbWeightful"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLRootExploreProbWeightful")) throwHumanParsingError("humanSLRootExploreProbWeightful");
    else if(cfg.contains("humanSLRootExploreProbWeightful"+idxStr)) params.humanSLRootExploreProbWeightful = cfg.getDouble("humanSLRootExploreProbWeightful"+idxStr, 0.0, 1.0);
    else if(cfg.contains("humanSLRootExploreProbWeightful"))   params.humanSLRootExploreProbWeightful = cfg.getDouble("humanSLRootExploreProbWeightful",        0.0, 1.0);
    else                                                       params.humanSLRootExploreProbWeightful = 0.0;
    if(!hasHumanModel && cfg.contains("humanSLPlaExploreProbWeightless"+idxStr)) throwHumanParsingError("humanSLPlaExploreProbWeightless"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLPlaExploreProbWeightless")) throwHumanParsingError("humanSLPlaExploreProbWeightless");
    else if(cfg.contains("humanSLPlaExploreProbWeightless"+idxStr)) params.humanSLPlaExploreProbWeightless = cfg.getDouble("humanSLPlaExploreProbWeightless"+idxStr, 0.0, 1.0);
    else if(cfg.contains("humanSLPlaExploreProbWeightless"))   params.humanSLPlaExploreProbWeightless = cfg.getDouble("humanSLPlaExploreProbWeightless",        0.0, 1.0);
    else                                                       params.humanSLPlaExploreProbWeightless = 0.0;
    if(!hasHumanModel && cfg.contains("humanSLPlaExploreProbWeightful"+idxStr)) throwHumanParsingError("humanSLPlaExploreProbWeightful"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLPlaExploreProbWeightful")) throwHumanParsingError("humanSLPlaExploreProbWeightful");
    else if(cfg.contains("humanSLPlaExploreProbWeightful"+idxStr)) params.humanSLPlaExploreProbWeightful = cfg.getDouble("humanSLPlaExploreProbWeightful"+idxStr, 0.0, 1.0);
    else if(cfg.contains("humanSLPlaExploreProbWeightful"))   params.humanSLPlaExploreProbWeightful = cfg.getDouble("humanSLPlaExploreProbWeightful",        0.0, 1.0);
    else                                                      params.humanSLPlaExploreProbWeightful = 0.0;
    if(!hasHumanModel && cfg.contains("humanSLOppExploreProbWeightless"+idxStr)) throwHumanParsingError("humanSLOppExploreProbWeightless"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLOppExploreProbWeightless")) throwHumanParsingError("humanSLOppExploreProbWeightless");
    else if(cfg.contains("humanSLOppExploreProbWeightless"+idxStr)) params.humanSLOppExploreProbWeightless = cfg.getDouble("humanSLOppExploreProbWeightless"+idxStr, 0.0, 1.0);
    else if(cfg.contains("humanSLOppExploreProbWeightless"))   params.humanSLOppExploreProbWeightless = cfg.getDouble("humanSLOppExploreProbWeightless",        0.0, 1.0);
    else                                                       params.humanSLOppExploreProbWeightless = 0.0;
    if(!hasHumanModel && cfg.contains("humanSLOppExploreProbWeightful"+idxStr)) throwHumanParsingError("humanSLOppExploreProbWeightful"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLOppExploreProbWeightful")) throwHumanParsingError("humanSLOppExploreProbWeightful");
    else if(cfg.contains("humanSLOppExploreProbWeightful"+idxStr)) params.humanSLOppExploreProbWeightful = cfg.getDouble("humanSLOppExploreProbWeightful"+idxStr, 0.0, 1.0);
    else if(cfg.contains("humanSLOppExploreProbWeightful"))   params.humanSLOppExploreProbWeightful = cfg.getDouble("humanSLOppExploreProbWeightful",        0.0, 1.0);
    else                                                      params.humanSLOppExploreProbWeightful = 0.0;
    if(!hasHumanModel && cfg.contains("humanSLChosenMoveProp"+idxStr)) throwHumanParsingError("humanSLChosenMoveProp"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLChosenMoveProp")) throwHumanParsingError("humanSLChosenMoveProp");
    else if(cfg.contains("humanSLChosenMoveProp"+idxStr)) params.humanSLChosenMoveProp = cfg.getDouble("humanSLChosenMoveProp"+idxStr, 0.0, 1.0);
    else if(cfg.contains("humanSLChosenMoveProp"))   params.humanSLChosenMoveProp = cfg.getDouble("humanSLChosenMoveProp",        0.0, 1.0);
    else                                             params.humanSLChosenMoveProp = 0.0;
    if(!hasHumanModel && cfg.contains("humanSLChosenMoveIgnorePass"+idxStr)) throwHumanParsingError("humanSLChosenMoveIgnorePass"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLChosenMoveIgnorePass")) throwHumanParsingError("humanSLChosenMoveIgnorePass");
    else if(cfg.contains("humanSLChosenMoveIgnorePass"+idxStr)) params.humanSLChosenMoveIgnorePass = cfg.getBool("humanSLChosenMoveIgnorePass"+idxStr);
    else if(cfg.contains("humanSLChosenMoveIgnorePass"))   params.humanSLChosenMoveIgnorePass = cfg.getBool("humanSLChosenMoveIgnorePass");
    else                                                   params.humanSLChosenMoveIgnorePass = false;
    if(!hasHumanModel && cfg.contains("humanSLChosenMovePiklLambda"+idxStr)) throwHumanParsingError("humanSLChosenMovePiklLambda"+idxStr);
    else if(!hasHumanModel && cfg.contains("humanSLChosenMovePiklLambda")) throwHumanParsingError("humanSLChosenMovePiklLambda");
    else if(cfg.contains("humanSLChosenMovePiklLambda"+idxStr)) params.humanSLChosenMovePiklLambda = cfg.getDouble("humanSLChosenMovePiklLambda"+idxStr, 0.0, 1000000000.0);
    else if(cfg.contains("humanSLChosenMovePiklLambda"))   params.humanSLChosenMovePiklLambda = cfg.getDouble("humanSLChosenMovePiklLambda",        0.0, 1000000000.0);
    else                                                   params.humanSLChosenMovePiklLambda = 1000000000.0;

    //On distributed, tolerate reading mutexPoolSize since older version configs use it.
    if(setupFor == SETUP_FOR_DISTRIBUTED)
      cfg.markAllKeysUsedWithPrefix("mutexPoolSize");

    paramss.push_back(params);
  }

  return paramss;
}


bool Setup::maybeWarnHumanSLParams(
  const SearchParams& params,
  const NNEvaluator* nnEval,
  const NNEvaluator* humanEval,
  std::ostream& out,
  Logger* logger
) {
  if(params.humanSLProfile.initialized) {
    bool hasAnySGFMetaUse =
      (nnEval != NULL && nnEval->requiresSGFMetadata()) ||
      (humanEval != NULL && humanEval->requiresSGFMetadata());
    if(!hasAnySGFMetaUse) {
      string modelNames;
      if(nnEval != NULL)
        modelNames += nnEval->getModelName();
      if(humanEval != NULL) {
        if(modelNames.size() > 0)
          modelNames += " and ";
        modelNames += humanEval->getModelName();
      }
      if(logger != NULL)
        logger->write("WARNING: humanSLProfile is specified as config param but model(s) don't use it: " + modelNames);
      out << "WARNING: humanSLProfile is specified as config param but model(s) don't use it: " << modelNames << endl;
      return true;
    }
  }
  return false;
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

Rules Setup::loadSingleRules(
  ConfigParser& cfg,
  bool loadKomi
) {
  Rules rules;

  if(cfg.contains("rules")) {
    if(cfg.contains("koRule")) throw StringError("Cannot both specify 'rules' and individual rules like koRule");
    if(cfg.contains("scoringRule")) throw StringError("Cannot both specify 'rules' and individual rules like scoringRule");
    if(cfg.contains("multiStoneSuicideLegal")) throw StringError("Cannot both specify 'rules' and individual rules like multiStoneSuicideLegal");
    if(cfg.contains("hasButton")) throw StringError("Cannot both specify 'rules' and individual rules like hasButton");
    if(cfg.contains("taxRule")) throw StringError("Cannot both specify 'rules' and individual rules like taxRule");
    if(cfg.contains("whiteHandicapBonus")) throw StringError("Cannot both specify 'rules' and individual rules like whiteHandicapBonus");
    if(cfg.contains("friendlyPassOk")) throw StringError("Cannot both specify 'rules' and individual rules like friendlyPassOk");
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

    if(cfg.contains("friendlyPassOk")) {
      rules.friendlyPassOk = cfg.getBool("friendlyPassOk");
    }

    //Drop default komi to 6.5 for territory rules, and to 7.0 for button
    if(rules.scoringRule == Rules::SCORING_TERRITORY)
      rules.komi = 6.5f;
    else if(rules.hasButton)
      rules.komi = 7.0f;
  }

  if(loadKomi) {
    rules.komi = cfg.getFloat("komi",Rules::MIN_USER_KOMI,Rules::MAX_USER_KOMI);
  }

  return rules;
}

bool Setup::loadDefaultBoardXYSize(
  ConfigParser& cfg,
  Logger& logger,
  int& defaultBoardXSizeRet,
  int& defaultBoardYSizeRet
) {
  const int defaultBoardXSize =
    cfg.contains("defaultBoardXSize") ? cfg.getInt("defaultBoardXSize",2,Board::MAX_LEN) :
    cfg.contains("defaultBoardSize") ? cfg.getInt("defaultBoardSize",2,Board::MAX_LEN) :
    -1;
  const int defaultBoardYSize =
    cfg.contains("defaultBoardYSize") ? cfg.getInt("defaultBoardYSize",2,Board::MAX_LEN) :
    cfg.contains("defaultBoardSize") ? cfg.getInt("defaultBoardSize",2,Board::MAX_LEN) :
    -1;
  if((defaultBoardXSize == -1) != (defaultBoardYSize == -1))
    logger.write("Warning: Config specified only one of defaultBoardXSize or defaultBoardYSize and no other board size parameter, ignoring it");

  if(defaultBoardXSize == -1 || defaultBoardYSize == -1) {
    return false;
  }
  defaultBoardXSizeRet = defaultBoardXSize;
  defaultBoardYSizeRet = defaultBoardYSize;
  return true;
}

vector<pair<set<string>,set<string>>> Setup::getMutexKeySets() {
  vector<pair<set<string>,set<string>>> mutexKeySets = {
    std::make_pair<set<string>,set<string>>(
    {"rules"},{"koRule","scoringRule","multiStoneSuicideLegal","taxRule","hasButton","whiteBonusPerHandicapStone","friendlyPassOk","whiteHandicapBonus"}
    ),
  };
  return mutexKeySets;
}

std::vector<std::unique_ptr<PatternBonusTable>> Setup::loadAvoidSgfPatternBonusTables(ConfigParser& cfg, Logger& logger) {
  int numBots = 1;
  if(cfg.contains("numBots"))
    numBots = cfg.getInt("numBots",1,MAX_BOT_PARAMS_FROM_CFG);

  std::vector<std::unique_ptr<PatternBonusTable>> tables;
  for(int i = 0; i<numBots; i++) {
    //Indexes different bots, such as in a match config
    const string idxStr = Global::intToString(i);

    std::unique_ptr<PatternBonusTable> patternBonusTable = nullptr;
    for(int j = 1; j<100000; j++) {
      //Indexes different sets of params for different sets of files, to combine into one bot.
      const string setStr = j == 1 ? string() : Global::intToString(j);
      const string prefix = "avoidSgf"+setStr;

      //Tries to find prefix+suffix+optional index
      //E.g. "avoidSgf"+"PatternUtility"+(optional integer indexing which bot for match)
      auto contains = [&cfg,&idxStr,&prefix](const string& suffix) {
        return cfg.containsAny({prefix+suffix+idxStr,prefix+suffix});
      };
      auto find = [&cfg,&idxStr,&prefix](const string& suffix) {
        return cfg.firstFoundOrFail({prefix+suffix+idxStr,prefix+suffix});
      };

      if(contains("PatternUtility")) {
        double penalty = cfg.getDouble(find("PatternUtility"),-3.0,3.0);
        double lambda = contains("PatternLambda") ? cfg.getDouble(find("PatternLambda"),0.0,1.0) : 1.0;
        int minTurnNumber = contains("PatternMinTurnNumber") ? cfg.getInt(find("PatternMinTurnNumber"),0,1000000) : 0;
        size_t maxFiles = contains("PatternMaxFiles") ? (size_t)cfg.getInt(find("PatternMaxFiles"),1,1000000) : 1000000;
        vector<string> allowedPlayerNames = contains("PatternAllowedNames") ? cfg.getStringsNonEmptyTrim(find("PatternAllowedNames")) : vector<string>();
        vector<string> sgfDirs = cfg.getStrings(find("PatternDirs"));
        if(patternBonusTable == nullptr)
          patternBonusTable = std::make_unique<PatternBonusTable>();
        string logSource = "bot " + idxStr;
        patternBonusTable->avoidRepeatedSgfMoves(sgfDirs,penalty,lambda,minTurnNumber,maxFiles,allowedPlayerNames,logger,logSource);
      }
    }
    tables.push_back(std::move(patternBonusTable));
  }
  return tables;
}

static string boardSizeToStr(int boardXSize, int boardYSize) {
  return Global::intToString(boardXSize) + "x" + Global::intToString(boardYSize);
}

static int getAutoPatternIntParam(ConfigParser& cfg, const string& param, int boardXSize, int boardYSize, int min, int max) {
  if(cfg.contains(param + boardSizeToStr(boardXSize,boardYSize)))
    return cfg.getInt(param + boardSizeToStr(boardXSize,boardYSize), min, max);
  if(!cfg.contains(param))
    throw ConfigParsingError(param + " was not specified in the config");
  return cfg.getInt(param, min, max);
}
static int64_t getAutoPatternInt64Param(ConfigParser& cfg, const string& param, int boardXSize, int boardYSize, int64_t min, int64_t max) {
  if(cfg.contains(param + boardSizeToStr(boardXSize,boardYSize)))
    return cfg.getInt64(param + boardSizeToStr(boardXSize,boardYSize), min, max);
  if(!cfg.contains(param))
    throw ConfigParsingError(param + " was not specified in the config");
  return cfg.getInt64(param, min, max);
}
static double getAutoPatternDoubleParam(ConfigParser& cfg, const string& param, int boardXSize, int boardYSize, double min, double max) {
  if(cfg.contains(param + boardSizeToStr(boardXSize,boardYSize)))
    return cfg.getDouble(param + boardSizeToStr(boardXSize,boardYSize), min, max);
  if(!cfg.contains(param))
    throw ConfigParsingError(param + " was not specified in the config");
  return cfg.getDouble(param, min, max);
}

bool Setup::saveAutoPatternBonusData(const std::vector<Sgf::PositionSample>& genmoveSamples, ConfigParser& cfg, Logger& logger, Rand& rand) {
  if(genmoveSamples.size() <= 0)
    return false;
  if(!cfg.contains("autoAvoidRepeatDir"))
    return false;

  string autoAvoidPatternsDir = cfg.getString("autoAvoidRepeatDir");
  MakeDir::make(autoAvoidPatternsDir);

  std::map<std::pair<int,int>, std::unique_ptr<ofstream>> outByBoardSize;
  string fileName = Global::uint64ToHexString(rand.nextUInt64()) + "_poses.txt";
  for(const Sgf::PositionSample& sampleToWrite : genmoveSamples) {
    int boardXSize = sampleToWrite.board.x_size;
    int boardYSize = sampleToWrite.board.y_size;
    std::pair<int,int> boardSize = std::make_pair(boardXSize, boardYSize);

    int minTurnNumber = getAutoPatternIntParam(cfg,"autoAvoidRepeatMinTurnNumber",boardXSize,boardYSize,0,1000000);
    int maxTurnNumber = getAutoPatternIntParam(cfg,"autoAvoidRepeatMaxTurnNumber",boardXSize,boardYSize,0,1000000);
    if(sampleToWrite.initialTurnNumber < minTurnNumber || sampleToWrite.initialTurnNumber > maxTurnNumber)
      continue;
    assert(sampleToWrite.moves.size() == 0);
    if(!contains(outByBoardSize,boardSize)) {
      MakeDir::make(autoAvoidPatternsDir + "/" + boardSizeToStr(boardXSize, boardYSize));
      outByBoardSize[boardSize] = std::make_unique<ofstream>();
      string filePath = autoAvoidPatternsDir + "/" + boardSizeToStr(boardXSize, boardYSize) + "/" + fileName;
      bool suc = FileUtils::tryOpen(*(outByBoardSize[boardSize]), filePath);
      if(!suc) {
        logger.write("ERROR: could not open " + filePath);
        return false;
      }
    }
    *(outByBoardSize[boardSize]) << Sgf::PositionSample::toJsonLine(sampleToWrite) << "\n";
  }
  for(auto iter = outByBoardSize.begin(); iter != outByBoardSize.end(); ++iter) {
    iter->second->close();
  }
  logger.write("Saved " + Global::uint64ToString(genmoveSamples.size()) + " avoid poses to " + autoAvoidPatternsDir);
  return true;
}

std::unique_ptr<PatternBonusTable> Setup::loadAndPruneAutoPatternBonusTables(ConfigParser& cfg, Logger& logger) {
  std::unique_ptr<PatternBonusTable> patternBonusTable = nullptr;

  if(cfg.contains("autoAvoidRepeatDir")) {
    string baseDir = cfg.getString("autoAvoidRepeatDir");
    std::vector<string> boardSizeDirs = FileUtils::listFiles(baseDir);

    patternBonusTable = std::make_unique<PatternBonusTable>();

    for(const string& dirName: boardSizeDirs) {
      std::vector<string> pieces = Global::split(dirName,'x');
      if(pieces.size() != 2)
        continue;
      int boardXSize;
      int boardYSize;
      bool suc = Global::tryStringToInt(pieces[0],boardXSize) && Global::tryStringToInt(pieces[1],boardYSize);
      if(!suc)
        continue;
      if(boardXSize < 2 || boardXSize > Board::MAX_LEN || boardYSize < 2 || boardYSize > Board::MAX_LEN)
        continue;

      string dirPath = baseDir + "/" + dirName;
      if(!FileUtils::isDirectory(dirPath))
        continue;

      double penalty = getAutoPatternDoubleParam(cfg,"autoAvoidRepeatUtility",boardXSize,boardYSize,-3.0,3.0);
      double lambda = getAutoPatternDoubleParam(cfg,"autoAvoidRepeatLambda",boardXSize,boardYSize,0.0,1.0);
      int minTurnNumber = getAutoPatternIntParam(cfg,"autoAvoidRepeatMinTurnNumber",boardXSize,boardYSize,0,1000000);
      int maxTurnNumber = getAutoPatternIntParam(cfg,"autoAvoidRepeatMaxTurnNumber",boardXSize,boardYSize,0,1000000);
      size_t maxPoses = getAutoPatternInt64Param(cfg,"autoAvoidRepeatMaxPoses",boardXSize,boardYSize,0,(int64_t)1000000000000LL);

      string logSource = dirPath;
      patternBonusTable->avoidRepeatedPosMovesAndDeleteExcessFiles({baseDir + "/" + dirName},penalty,lambda,minTurnNumber,maxTurnNumber,maxPoses,logger,logSource);
    }


    cfg.markAllKeysUsedWithPrefix("autoAvoidRepeatUtility");
    cfg.markAllKeysUsedWithPrefix("autoAvoidRepeatLambda");
    cfg.markAllKeysUsedWithPrefix("autoAvoidRepeatMinTurnNumber");
    cfg.markAllKeysUsedWithPrefix("autoAvoidRepeatMaxTurnNumber");
    cfg.markAllKeysUsedWithPrefix("autoAvoidRepeatMaxPoses");
    cfg.markAllKeysUsedWithPrefix("autoAvoidRepeatSaveChunkSize");
  }
  return patternBonusTable;
}
