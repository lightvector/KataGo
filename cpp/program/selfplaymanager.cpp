#include "../program/selfplaymanager.h"
#include "../program/setup.h"

using namespace std;

SelfplayManager::ModelData::ModelData(
  ConfigParser& cfg, const string& name, NNEvaluator* neval, int maxDQueueSize,
  TrainingDataWriter* tdWriter, TrainingDataWriter* vdWriter, ofstream* sOut, double vProp
):
  modelName(name),
  nnEval(neval),
  matchPairer(NULL),
  validationProp(vProp),
  maxDataQueueSize(maxDQueueSize),
  finishedGameQueue(maxDQueueSize),
  numGameThreads(0),
  isDraining(false),
  noGameThreadsLeftVar(),
  tdataWriter(tdWriter),
  vdataWriter(vdWriter),
  sgfOut(sOut),
  rand()
{
   SearchParams baseParams = Setup::loadSingleParams(cfg);

   //Initialize object for randomly pairing bots. Actually since this is only selfplay, this only
   //ever gives is the trivial self-pairing, but we use it also for keeping the game count and some logging.
   bool forSelfPlay = true;
   bool forGateKeeper = false;
   matchPairer = new MatchPairer(cfg, 1, {modelName}, {nnEval}, {baseParams}, forSelfPlay, forGateKeeper);
}

SelfplayManager::ModelData::~ModelData() {
  delete matchPairer;
  delete nnEval;
  delete tdataWriter;
  if(vdataWriter != NULL)
    delete vdataWriter;
  if(sgfOut != NULL)
    delete sgfOut;
}

SelfplayManager::SelfplayManager(
  Logger* lg
):
  logger(lg),
  managerMutex(),
  modelDatas(),
  numDataWriteLoopsActive(0),
  dataWriteLoopsAreDone()
{
}

SelfplayManager::~SelfplayManager() {
  std::unique_lock<std::mutex> lock(managerMutex);
  //Mark everything as draining so our data writing loops quit, once every model's users release
  for(size_t i = 0; i<modelDatas.size(); i++) {
    //If a client tries to delete this while something is still acquired, there's something wrong.
    assert(modelDatas[i]->numGameThreads == 0);
    //Go ahead and mark everything to be cleaned up though, otherwise if things look fine
    scheduleCleanupModelWhenFreeAlreadyLocked(modelDatas[i]);
  }
  while(numDataWriteLoopsActive > 0) {
    dataWriteLoopsAreDone.wait(lock);
  }
  assert(modelDatas.size() == 0);
}

static void dataWriteLoop(SelfplayManager* manager, SelfplayManager::ModelData* modelData) {
  manager->runDataWriteLoop(modelData);
}

void SelfplayManager::loadModelAndStartDataWriting(
  const string& modelName,
  NNEvaluator* nnEval,
  TrainingDataWriter* tdataWriter,
  TrainingDataWriter* vdataWriter,
  ofstream* sgfOut,
  ConfigParser& cfg,
  int maxDataQueueSize,
  double validationProp
) {
  std::unique_lock<std::mutex> lock(managerMutex);
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      throw StringError("SelfplayManager::loadModelAndStartDataWriting: Duplicate model name: " + modelName);
    }
  }

  ModelData* newModel = new ModelData(cfg,modelName,nnEval,maxDataQueueSize,tdataWriter,vdataWriter,sgfOut,validationProp);
  modelDatas.push_back(newModel);
  numDataWriteLoopsActive++;
  std::thread newThread(dataWriteLoop,this,newModel);
  newThread.detach();
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

NNEvaluator* SelfplayManager::acquireModelAlreadyLocked(ModelData* foundData) {
  foundData->numGameThreads += 1;
  return foundData->nnEval;
}
void SelfplayManager::releaseAlreadyLocked(ModelData* foundData) {
  foundData->numGameThreads -= 1;
  if(foundData->numGameThreads <= 0) {
    foundData->noGameThreadsLeftVar.notify_all();
    if(foundData->isDraining)
      foundData->finishedGameQueue.setReadOnly();
  }
}
void SelfplayManager::scheduleCleanupModelWhenFreeAlreadyLocked(ModelData* foundData) {
  foundData->isDraining = true;
  if(foundData->numGameThreads <= 0) {
    foundData->finishedGameQueue.setReadOnly();
  }
}

NNEvaluator* SelfplayManager::acquireModel(const string& modelName) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName && !modelDatas[i]->isDraining) {
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
  if(foundData->isDraining)
    return NULL;
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
  if(foundData != NULL)
    releaseAlreadyLocked(foundData);
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
  if(foundData != NULL)
    releaseAlreadyLocked(foundData);
}

bool SelfplayManager::countOneGameStarted(NNEvaluator* nnEval, MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW) {
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

  lock.unlock();

  //TODO logger might be null, can't always dereference
  return foundData->matchPairer->getMatchup(botSpecB, botSpecW, *logger);
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

void SelfplayManager::scheduleCleanupModelWhenFree(const string& modelName) {
  std::lock_guard<std::mutex> lock(managerMutex);
  ModelData* foundData = NULL;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i]->modelName == modelName) {
      foundData = modelDatas[i];
      break;
    }
  }
  if(foundData != NULL)
    scheduleCleanupModelWhenFreeAlreadyLocked(foundData);
}

void SelfplayManager::runDataWriteLoop(ModelData* modelData) {
  if(logger != NULL)
    logger->write("Data write loop starting for neural net: " + modelData->modelName);

   while(true) {
    size_t size = modelData->finishedGameQueue.size();
    if(size > modelData->maxDataQueueSize / 2 && logger != NULL)
      logger->write(Global::strprintf("WARNING: Struggling to keep up writing data, %d games enqueued out of %d max",size,modelData->maxDataQueueSize));

    FinishedGameData* gameData;
    bool suc = modelData->finishedGameQueue.waitPop(gameData);
    if(!suc)
      break;

    assert(gameData != NULL);

    if(modelData->rand.nextBool(modelData->validationProp))
      modelData->vdataWriter->writeGame(*gameData);
    else
      modelData->tdataWriter->writeGame(*gameData);

    if(modelData->sgfOut != NULL) {
      assert(gameData->startHist.moveHistory.size() <= gameData->endHist.moveHistory.size());
      WriteSgf::writeSgf(*modelData->sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false);
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

  std::unique_lock<std::mutex> lock(managerMutex);

  //Make sure all threads are completely done with it
  while(modelData->numGameThreads > 0)
    modelData->noGameThreadsLeftVar.wait(lock);

  //Find where our modelData is and remove it
  string name = modelData->modelName;
  bool found = false;
  for(size_t i = 0; i<modelDatas.size(); i++) {
    if(modelDatas[i] == modelData) {
      modelDatas.erase(modelDatas.begin()+i);
      found = true;
      break;
    }
  }
  assert(found);
  (void)found; //Avoid warning when asserts are disabled
  lock.unlock();

  //Do logging and cleanup while unlocked, so that our freeing and stopping of this neural net doesn't
  //block anyone else
  if(logger != NULL) {
    logger->write("Final cleanup of net: " + modelData->nnEval->getModelFileName());
    logger->write("Final NN rows: " + Global::int64ToString(modelData->nnEval->numRowsProcessed()));
    logger->write("Final NN batches: " + Global::int64ToString(modelData->nnEval->numBatchesProcessed()));
    logger->write("Final NN avg batch size: " + Global::doubleToString(modelData->nnEval->averageProcessedBatchSize()));
  }

  assert(modelData->numGameThreads == 0);
  assert(modelData->isDraining);
  delete modelData;

  if(logger != NULL) {
    logger->write("Data write loop cleaned up and terminating for " + name);
  }

  //Check back in and notify that we're done once done cleaning up.
  lock.lock();
  numDataWriteLoopsActive--;
  assert(numDataWriteLoopsActive >= 0);
  if(numDataWriteLoopsActive == 0) {
    assert(modelDatas.size() == 0);
    dataWriteLoopsAreDone.notify_all();
  }
  lock.unlock();

}
