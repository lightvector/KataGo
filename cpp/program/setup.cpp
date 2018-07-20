#include "../program/setup.h"

vector<NNEvaluator*> Setup::initializeNNEvaluators(
  const vector<string>& nnModelFiles,
  ConfigParser& cfg,
  Logger& logger,
  Rand& seedRand
) {
  vector<NNEvaluator*> nnEvals;
  for(size_t i = 0; i<nnModelFiles.size(); i++) {
    string idxstr = Global::intToString(i);
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
    if(cfg.contains("nnRandSeed" + idxstr))
      nnRandSeed = cfg.getString("nnRandSeed" + idxstr);
    else if(cfg.contains("nnRandSeed"))
      nnRandSeed = cfg.getString("nnRandSeed");
    else
      nnRandSeed = Global::uint64ToString(seedRand.nextUInt64());
    logger.write("nnRandSeed" + idxstr + " = " + nnRandSeed);

    vector<string> gpuVisibleDeviceListByThread;
    string gpuVisibleDeviceListByThreadStr;
    if(cfg.contains("gpuVisibleDeviceListByThread" + idxstr))
      gpuVisibleDeviceListByThreadStr = cfg.getString("gpuVisibleDeviceListByThread" + idxstr);
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
