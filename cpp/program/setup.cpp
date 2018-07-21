#include "../program/setup.h"

vector<NNEvaluator*> Setup::initializeNNEvaluators(
  const vector<string>& nnModelFiles,
  ConfigParser& cfg,
  Logger& logger,
  Rand& seedRand
) {
  vector<NNEvaluator*> nnEvals;
  for(size_t i = 0; i<nnModelFiles.size(); i++) {
    string idxStr = Global::intToString(i);
    const string& nnModelFile = nnModelFiles[i];

    bool debugSkipNeuralNet = false;
    NNEvaluator* nnEval = new NNEvaluator(
      nnModelFile,
      cfg.getInt("nnMaxBatchSize", 1, 65536),
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

    vector<string> gpuVisibleDeviceListByThread;
    string gpuVisibleDeviceListByThreadStr;
    if(cfg.contains("gpuVisibleDeviceListByThread" + idxStr))
      gpuVisibleDeviceListByThreadStr = cfg.getString("gpuVisibleDeviceListByThread" + idxStr);
    else if(cfg.contains("gpuVisibleDeviceListByThread"))
      gpuVisibleDeviceListByThreadStr = cfg.getString("gpuVisibleDeviceListByThread");

    if(gpuVisibleDeviceListByThreadStr.length() > 0) {
      vector<string> pieces = Global::split(gpuVisibleDeviceListByThreadStr,';');
      for(size_t j = 0; j < pieces.size(); j++)
        gpuVisibleDeviceListByThread.push_back(Global::trim(pieces[j]));
    }
    if(gpuVisibleDeviceListByThread.size() > 1024)
      throw IOError("Too many values for gpuVisibleDeviceListByThread");

    double perProcessGPUMemoryFraction = -1;
    if(cfg.contains("perProcessGPUMemoryFraction"))
      perProcessGPUMemoryFraction = cfg.getDouble("perProcessGPUMemoryFraction",0.0,1.0);

    int numNNServerThreads = gpuVisibleDeviceListByThreadStr.length() == 0 ? 1 : gpuVisibleDeviceListByThread.size();
    int defaultSymmetry = 0;
    nnEval->spawnServerThreads(
      numNNServerThreads,
      nnRandomize,
      nnRandSeed,
      defaultSymmetry,
      logger,
      gpuVisibleDeviceListByThread,
      perProcessGPUMemoryFraction
    );

    nnEvals.push_back(nnEval);
  }

  return nnEvals;
}


vector<SearchParams> Setup::loadParams(
  ConfigParser& cfg,
  Rand& seedRand
) {

  SearchParams baseParams;
  {
    if(cfg.contains("maxPlayouts"))
      baseParams.maxPlayouts = cfg.getUInt64("maxPlayouts", (uint64_t)1, (uint64_t)1 << 62);
    if(cfg.contains("maxVisits"))
      baseParams.maxVisits = cfg.getUInt64("maxVisits", (uint64_t)1, (uint64_t)1 << 62);
    if(cfg.contains("maxTime"))
      baseParams.maxTime = cfg.getDouble("maxTime", 0.0, 1.0e20);
    baseParams.numThreads = cfg.getInt("numSearchThreads", 1, 1024);

    baseParams.winLossUtilityFactor = cfg.getDouble("winLossUtilityFactor", 0.0, 1.0);
    baseParams.scoreUtilityFactor = cfg.getDouble("scoreUtilityFactor", 0.0, 1.0);
    baseParams.noResultUtilityForWhite = cfg.getDouble("noResultUtilityForWhite", -2.0, 2.0);
    baseParams.drawUtilityForWhite = cfg.getDouble("drawUtilityForWhite", -2.0, 2.0);

    baseParams.cpuctExploration = cfg.getDouble("cpuctExploration", 0.0, 10.0);
    baseParams.fpuReductionMax = cfg.getDouble("fpuReductionMax", 0.0, 2.0);

    baseParams.rootNoiseEnabled = cfg.getBool("rootNoiseEnabled");
    baseParams.rootDirichletNoiseTotalConcentration = cfg.getDouble("rootDirichletNoiseTotalConcentration", 0.001, 10000.0);
    baseParams.rootDirichletNoiseWeight = cfg.getDouble("rootDirichletNoiseWeight", 0.0, 1.0);

    baseParams.chosenMoveTemperature = cfg.getDouble("chosenMoveTemperature", 0.0, 5.0);
    baseParams.chosenMoveSubtract = cfg.getDouble("chosenMoveSubtract", 0.0, 1.0e10);

    baseParams.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize", 1, 1 << 24);
    baseParams.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread", 1, 1000);
  }

  vector<SearchParams> paramss;
  int numBots = 1;
  if(cfg.contains("numBots"))
    numBots = cfg.getInt("numBots",1,1024);

  for(int i = 0; i<numBots; i++) {
    SearchParams params = baseParams;

    if(cfg.contains("searchRandSeed"))
      params.randSeed = cfg.getString("searchRandSeed");
    else
      params.randSeed = Global::uint64ToString(seedRand.nextUInt64());

    string idxStr = Global::intToString(i);

    if(cfg.contains("maxPlayouts"+idxStr))
      params.maxPlayouts = cfg.getUInt64("maxPlayouts"+idxStr, (uint64_t)1, (uint64_t)1 << 62);
    if(cfg.contains("maxVisits"+idxStr))
      params.maxVisits = cfg.getUInt64("maxVisits"+idxStr, (uint64_t)1, (uint64_t)1 << 62);
    if(cfg.contains("maxTime"+idxStr))
      params.maxTime = cfg.getDouble("maxTime"+idxStr, 0.0, 1.0e20);
    if(cfg.contains("numSearchThreads"+idxStr))
      params.numThreads = cfg.getInt("numSearchThreads"+idxStr, 1, 1024);
    if(cfg.contains("winLossUtilityFactor"+idxStr))
      params.winLossUtilityFactor = cfg.getDouble("winLossUtilityFactor"+idxStr, 0.0, 1.0);
    if(cfg.contains("scoreUtilityFactor"+idxStr))
      params.scoreUtilityFactor = cfg.getDouble("scoreUtilityFactor"+idxStr, 0.0, 1.0);
    if(cfg.contains("noResultUtilityForWhite"+idxStr))
      params.noResultUtilityForWhite = cfg.getDouble("noResultUtilityForWhite"+idxStr, -2.0, 2.0);
    if(cfg.contains("drawUtilityForWhite"+idxStr))
      params.drawUtilityForWhite = cfg.getDouble("drawUtilityForWhite"+idxStr, -2.0, 2.0);
    if(cfg.contains("cpuctExploration"+idxStr))
      params.cpuctExploration = cfg.getDouble("cpuctExploration"+idxStr, 0.0, 10.0);
    if(cfg.contains("fpuReductionMax"+idxStr))
      params.fpuReductionMax = cfg.getDouble("fpuReductionMax"+idxStr, 0.0, 2.0);
    if(cfg.contains("rootNoiseEnabled"+idxStr))
      params.rootNoiseEnabled = cfg.getBool("rootNoiseEnabled"+idxStr);
    if(cfg.contains("rootDirichletNoiseTotalConcentration"+idxStr))
      params.rootDirichletNoiseTotalConcentration = cfg.getDouble("rootDirichletNoiseTotalConcentration"+idxStr, 0.001, 10000.0);
    if(cfg.contains("rootDirichletNoiseWeight"+idxStr))
      params.rootDirichletNoiseWeight = cfg.getDouble("rootDirichletNoiseWeight"+idxStr, 0.0, 1.0);
    if(cfg.contains("chosenMoveTemperature"+idxStr))
      params.chosenMoveTemperature = cfg.getDouble("chosenMoveTemperature"+idxStr, 0.0, 5.0);
    if(cfg.contains("chosenMoveSubtract"+idxStr))
      params.chosenMoveSubtract = cfg.getDouble("chosenMoveSubtract"+idxStr, 0.0, 1.0e10);
    if(cfg.contains("mutexPoolSize"+idxStr))
      params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize"+idxStr, 1, 1 << 24);
    if(cfg.contains("numVirtualLossesPerThread"+idxStr))
      params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread"+idxStr, 1, 1000);

    paramss.push_back(params);
  }

  return paramss;
}
