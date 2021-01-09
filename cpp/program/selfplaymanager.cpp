#include "../program/selfplaymanager.h"

using namespace std;

SelfplayManager::ModelData::ModelData(
  const string& name, NNEvaluator* neval, int maxDQueueSize,
  TrainingDataWriter* tdWriter, TrainingDataWriter* vdWriter, ofstream* sOut,
  double initialTime,
  bool hasDataLoop
):
  modelName(name),
  nnEval(neval),
  gameStartedCount(0),
  lastReleaseTime(initialTime),
  hasDataWriteLoop(hasDataLoop),
  finishedGameQueue(maxDQueueSize),
  acquireCount(0),
  tdataWriter(tdWriter),
  vdataWriter(vdWriter),
  sgfOut(sOut)
{
}

SelfplayManager::ModelData::~ModelData() {
  delete nnEval;
  delete tdataWriter;
  if(vdataWriter != NULL)
    delete vdataWriter;
  if(sgfOut != NULL)
    delete sgfOut;
}

//------------------------------------------------------------------------------------

SelfplayManager::SelfplayManager(
  double vProp,
  int maxDQueueSize,
  Logger* lg,
  int64_t logEvery,
  bool autoCleanup
):
  validationProp(vProp),
  maxDataQueueSize(maxDQueueSize),
  logger(lg),
  logGamesEvery(logEvery),
  autoCleanupAllButLatestIfUnused(autoCleanup),
  timer(),
  managerMutex(),
  modelDatas(),
  numDataWriteLoopsActive(0),
  dataWriteLoopsAreDone(),
  totalNumRowsProcessed(0)
{
}

SelfplayManager::~SelfplayManager() {
  std::unique_lock<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    //If a client tries to delete this while something is still acquired, there's something wrong.
    assert(modelDatas[i]->acquireCount == 0);
    //Trigger data writing loop to quit once it reaches end of its queue
    modelDatas[i]->finishedGameQueue.setReadOnly();
    totalNumRowsProcessed += modelDatas[i]->nnEval->numRowsProcessed();
    //Data write loop is responsible for deleting ModelData, if it exists
    if(!modelDatas[i]->hasDataWriteLoop)
      delete modelDatas[i];
  }
  modelDatas.clear();
  while(numDataWriteLoopsActive > 0) {
    dataWriteLoopsAreDone.wait(lock);
  }
}

uint64_t SelfplayManager::getTotalNumRowsProcessed() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  uint64_t total = totalNumRowsProcessed;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    total += modelDatas[i]->nnEval->numRowsProcessed();
  }
  return total;
}


static void dataWriteLoop(SelfplayManager* manager, SelfplayManager::ModelData* modelData) {
  manager->runDataWriteLoop(modelData);
}

void SelfplayManager::maybeAutoCleanupAlreadyLocked() {
  if(autoCleanupAllButLatestIfUnused && modelDatas.size() > 0) {
    for(size_t i = 0; i<modelDatas.size()-1; i++) {
      ModelData* foundData = modelDatas[i];
      if(foundData->acquireCount <= 0) {
        assert(foundData->acquireCount == 0);
        //Trigger data writing loop to quit once it reaches end of its queue
        foundData->finishedGameQueue.setReadOnly();
        totalNumRowsProcessed += foundData->nnEval->numRowsProcessed();
        //Data write loop is responsible for deleting ModelData, if it exists
        if(!foundData->hasDataWriteLoop)
          delete foundData;
        modelDatas.erase(modelDatas.begin()+i);
        i--;
      }
    }
  }
}


void SelfplayManager::cleanupUnusedModelsOlderThan(double seconds) {
  std::lock_guard<std::mutex> lock(managerMutex);
  double now = timer.getSeconds();
  for(size_t i = 0; i<modelDatas.size(); i++) {
    ModelData* foundData = modelDatas[i];
    if(foundData->acquireCount <= 0 && now - foundData->lastReleaseTime > seconds) {
      assert(foundData->acquireCount == 0);
      logger->write("Unloading network that hasn't been used in a while: " + foundData->modelName);
      //Trigger data writing loop to quit once it reaches end of its queue
      foundData->finishedGameQueue.setReadOnly();
      totalNumRowsProcessed += foundData->nnEval->numRowsProcessed();
      //Data write loop is responsible for deleting ModelData, if it exists
      if(!foundData->hasDataWriteLoop)
        delete foundData;
      modelDatas.erase(modelDatas.begin()+i);
      i--;
    }
  }
}

void SelfplayManager::clearUnusedModelCaches() {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    ModelData* foundData = modelDatas[i];
    if(foundData->acquireCount <= 0) {
      foundData->nnEval->clearCache();
    }
  }
}


void SelfplayManager::loadModelAndStartDataWriting(
  NNEvaluator* nnEval,
  TrainingDataWriter* tdataWriter,
  TrainingDataWriter* vdataWriter,
  ofstream* sgfOut
) {
  string modelName = nnEval->getModelName();
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      throw StringError("SelfplayManager::loadModelAndStartDataWriting: Duplicate model name: " + modelName);
    }
  }

  double initialTime = timer.getSeconds();
  bool hasDataWriteLoop = true;
  ModelData* newModel = new ModelData(modelName,nnEval,maxDataQueueSize,tdataWriter,vdataWriter,sgfOut,initialTime,hasDataWriteLoop);
  modelDatas.push_back(newModel);
  numDataWriteLoopsActive++;
  std::thread newThread(dataWriteLoop,this,newModel);
  newThread.detach();

  maybeAutoCleanupAlreadyLocked();
}

void SelfplayManager::loadModelNoDataWritingLoop(
  NNEvaluator* nnEval,
  TrainingDataWriter* tdataWriter,
  TrainingDataWriter* vdataWriter,
  ofstream* sgfOut
) {
  string modelName = nnEval->getModelName();
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      throw StringError("SelfplayManager::loadModelAndStartDataWriting: Duplicate model name: " + modelName);
    }
  }

  double initialTime = timer.getSeconds();
  bool hasDataWriteLoop = false;
  ModelData* newModel = new ModelData(modelName,nnEval,maxDataQueueSize,tdataWriter,vdataWriter,sgfOut,initialTime,hasDataWriteLoop);
  modelDatas.push_back(newModel);
  maybeAutoCleanupAlreadyLocked();
}

size_t SelfplayManager::numModels() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  return modelDatas.size();
}

vector<string> SelfplayManager::modelNames() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  vector<string> names;
  for(size_t i = 0; i<modelDatas.size(); i++)
    names.push_back(modelDatas[i]->modelName);
  return names;
}

string SelfplayManager::getLatestModelName() const {
  std::lock_guard<std::mutex> lock(managerMutex);
  if(modelDatas.size() <= 0)
    throw StringError("SelfplayManager::getLatestModelName: no models loaded");
  return modelDatas[modelDatas.size()-1]->modelName;
}

bool SelfplayManager::hasModel(const std::string& modelName) const {
  std::lock_guard<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName)
      return true;
  }
  return false;
}


NNEvaluator* SelfplayManager::acquireModelAlreadyLocked(ModelData* foundData) {
  foundData->acquireCount += 1;
  return foundData->nnEval;
}
void SelfplayManager::releaseAlreadyLocked(ModelData* foundData) {
  foundData->lastReleaseTime = timer.getSeconds();
  foundData->acquireCount -= 1;
}

NNEvaluator* SelfplayManager::acquireModel(const string& modelName) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData != NULL)
    return acquireModelAlreadyLocked(foundData);
  return NULL;
}

NNEvaluator* SelfplayManager::acquireLatest() {
  std::lock_guard<std::mutex> lock(managerMutex);
  if(modelDatas.size() <= 0)
    return NULL;
  ModelData* foundData = modelDatas[modelDatas.size()-1];
  return acquireModelAlreadyLocked(foundData);
}

void SelfplayManager::release(const string& modelName) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData != NULL) {
    releaseAlreadyLocked(foundData);
    maybeAutoCleanupAlreadyLocked();
  }
}

void SelfplayManager::release(NNEvaluator* nnEval) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData != NULL) {
    releaseAlreadyLocked(foundData);
    maybeAutoCleanupAlreadyLocked();
  }
}

void SelfplayManager::countOneGameStarted(NNEvaluator* nnEval) {
  std::unique_lock<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::countOneGameStarted: could not find model. Possible bug - client did not acquire model?");

  foundData->gameStartedCount += 1;
  int64_t gameStartedCount = foundData->gameStartedCount;
  lock.unlock();

  if(logger != NULL && gameStartedCount % logGamesEvery == 0) {
    logger->write("Started " + Global::int64ToString(gameStartedCount) + " games with " + nnEval->getModelName());
  }
  int64_t logNNEvery = logGamesEvery*100 > 1000 ? logGamesEvery*100 : 1000;
  if(logger != NULL && gameStartedCount % logNNEvery == 0) {
    logger->write(nnEval->getModelFileName());
    logger->write("NN rows: " + Global::int64ToString(nnEval->numRowsProcessed()));
    logger->write("NN batches: " + Global::int64ToString(nnEval->numBatchesProcessed()));
    logger->write("NN avg batch size: " + Global::doubleToString(nnEval->averageProcessedBatchSize()));
  }
}

void SelfplayManager::enqueueDataToWrite(const string& modelName, FinishedGameData* gameData) {
  std::unique_lock<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::enqueueDataToWrite: could not find model. Possible bug - client did not acquire model?");
  assert(foundData->hasDataWriteLoop == true);

  //In case it takes a while to push the game on, drop the lock. We're guaranteed as a precondition that
  //the caller has acquired the model as well, so it won't be cleaned up underneath us.
  lock.unlock();
  foundData->finishedGameQueue.waitPush(gameData);
}

void SelfplayManager::enqueueDataToWrite(NNEvaluator* nnEval, FinishedGameData* gameData) {
  std::unique_lock<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::enqueueDataToWrite: could not find model. Possible bug - client did not acquire model?");

  //In case it takes a while to push the game on, drop the lock. We're guaranteed as a precondition that
  //the caller has acquired the model as well, so it won't be cleaned up underneath us.
  lock.unlock();
  foundData->finishedGameQueue.waitPush(gameData);
}

void SelfplayManager::runDataWriteLoop(ModelData* modelData) {
  Logger::logThreadUncaught("data write loop", logger, [&](){ runDataWriteLoopImpl(modelData); });
}

void SelfplayManager::runDataWriteLoopImpl(ModelData* modelData) {
  if(logger != NULL)
    logger->write("Data write loop starting for neural net: " + modelData->modelName);

  Rand rand;
  while(true) {
    size_t size = modelData->finishedGameQueue.size();
    if(size > maxDataQueueSize / 2 && logger != NULL)
      logger->write(Global::strprintf("WARNING: Struggling to keep up writing data, %d games enqueued out of %d max",size,maxDataQueueSize));

    FinishedGameData* gameData;
    bool suc = modelData->finishedGameQueue.waitPop(gameData);
    if(!suc)
      break;

    assert(gameData != NULL);

    if(rand.nextBool(validationProp))
      modelData->vdataWriter->writeGame(*gameData);
    else
      modelData->tdataWriter->writeGame(*gameData);

    if(modelData->sgfOut != NULL) {
      assert(gameData->startHist.moveHistory.size() <= gameData->endHist.moveHistory.size());
      WriteSgf::writeSgf(*modelData->sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
      (*modelData->sgfOut) << endl;
    }
    delete gameData;
  }

  modelData->tdataWriter->flushIfNonempty();
  if(modelData->vdataWriter != NULL)
    modelData->vdataWriter->flushIfNonempty();
  if(modelData->sgfOut != NULL)
    modelData->sgfOut->close();

  if(logger != NULL)
    logger->write("Data write loop finishing for neural net: " + modelData->modelName);

  assert(modelData->acquireCount == 0);

  string name = modelData->modelName;

  //Lock the manager and do nothing with the lock (except run an assert).
  //The lock is technically necessary for thread-safety - we don't want to delete this modelData until we are
  //absolutely sure that the manager is done removing it from its own tracking in modelDatas, so we lock
  //the manager to make sure that we block until this is the case. While we're at it, we go ahead and assert it too.
  {
    std::lock_guard<std::mutex> lock(managerMutex);
    for(size_t i = 0; i<modelDatas.size(); i++) {
      (void)i;
      assert(modelDatas[i] != modelData);
    }
  }

  //Do logging and cleanup while unlocked, so that our freeing and stopping of this neural net doesn't
  //block anyone else
  if(logger != NULL) {
    logger->write("Final cleanup of net: " + modelData->nnEval->getModelFileName());
    logger->write("Final NN rows: " + Global::int64ToString(modelData->nnEval->numRowsProcessed()));
    logger->write("Final NN batches: " + Global::int64ToString(modelData->nnEval->numBatchesProcessed()));
    logger->write("Final NN avg batch size: " + Global::doubleToString(modelData->nnEval->averageProcessedBatchSize()));
  }

  delete modelData;

  if(logger != NULL) {
    logger->write("Data write loop cleaned up and terminating for " + name);
  }

  //Check back in and notify that we're done once done cleaning up.
  std::unique_lock<std::mutex> lock(managerMutex);
  numDataWriteLoopsActive--;
  assert(numDataWriteLoopsActive >= 0);
  if(numDataWriteLoopsActive == 0) {
    assert(modelDatas.size() == 0);
    dataWriteLoopsAreDone.notify_all();
  }
  lock.unlock();
}

void SelfplayManager::withDataWriters(
  NNEvaluator* nnEval,
  std::function<void(TrainingDataWriter* tdataWriter, TrainingDataWriter* vdataWriter, std::ofstream* sgfOut)> f
) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->nnEval == nnEval) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData == NULL)
    throw StringError("SelfplayManager::withDataWriters: could not find model. Possible bug - client did not acquire model?");
  assert(foundData->hasDataWriteLoop == false);

  f(foundData->tdataWriter, foundData->vdataWriter, foundData->sgfOut);
}
