#ifndef PROGRAM_SELFPLAYMANAGER_H_
#define PROGRAM_SELFPLAYMANAGER_H_

#include "core/threadsafequeue.h"
#include "dataio/sgf.h"
#include "dataio/trainingwrite.h"
#include "neuralnet/nneval.h"

class SelfplayManager {
 public:
  SelfplayManager(
    double validationProp,
    int maxDataQueueSize,
    Logger* logger,
    int64_t logGamesEvery,
    bool autoCleanupAllButLatestIfUnused
  );
  ~SelfplayManager();

  SelfplayManager(const SelfplayManager& other);
  SelfplayManager& operator=(const SelfplayManager& other);
  SelfplayManager(SelfplayManager&& other);
  SelfplayManager& operator=(SelfplayManager&& other);

  //All below functions are internally synchronized and thread-safe.

  //SelfplayManager takes responsibility for deleting the data writers and closing and deleting sgfOut.
  void loadModelAndStartDataWriting(
    NNEvaluator* nnEval,
    TrainingDataWriter* tdataWriter,
    TrainingDataWriter* vdataWriter,
    std::ofstream* sgfOut
  );

  //For all of the below, model names are simply from nnEval->getModelName().

  //Models that aren't cleaned up yet are in the order from earliest to latest
  std::vector<std::string> modelNames() const;
  std::string getLatestModelName() const;

  //Returns NULL if acquire failed (such as if that model was scheduled to be cleaned up or already cleaned up,).
  //Must call release when done, and cease using the NNEvaluator after that.
  NNEvaluator* acquireModel(const std::string& modelName);
  NNEvaluator* acquireLatest();
  //Release a model either by name or by the nnEval object that was returned.
  void release(const std::string& modelName);
  void release(NNEvaluator* nnEval);

  //====================================================================================
  //These should only be called by a thread that has currently acquired the model.

  //Increment a counter and maybe log some stats
  void countOneGameStarted(NNEvaluator* nnEval);

  //SelfplayManager takes responsibility for deleting the gameData once written.
  void enqueueDataToWrite(const std::string& modelName, FinishedGameData* gameData);
  void enqueueDataToWrite(NNEvaluator* nnEval, FinishedGameData* gameData);

  //====================================================================================

  //For internal use
  struct ModelData {
    std::string modelName;
    NNEvaluator* nnEval;
    int64_t gameStartedCount;

    ThreadSafeQueue<FinishedGameData*> finishedGameQueue;
    int acquireCount;

    TrainingDataWriter* tdataWriter;
    TrainingDataWriter* vdataWriter;
    std::ofstream* sgfOut;

    ModelData(
      const std::string& name, NNEvaluator* neval, int maxDataQueueSize,
      TrainingDataWriter* tdWriter, TrainingDataWriter* vdWriter, std::ofstream* sOut
    );
    ~ModelData();
  };

 private:
  const double validationProp;
  const int maxDataQueueSize;
  Logger* logger;
  const int64_t logGamesEvery;
  const bool autoCleanupAllButLatestIfUnused;

  mutable std::mutex managerMutex;
  std::vector<ModelData*> modelDatas;
  int numDataWriteLoopsActive;
  std::condition_variable dataWriteLoopsAreDone;

  NNEvaluator* acquireModelAlreadyLocked(SelfplayManager::ModelData* foundData);
  void releaseAlreadyLocked(SelfplayManager::ModelData* foundData);
  void maybeAutoCleanupAlreadyLocked();

 public:
  //For internal use
  void runDataWriteLoop(ModelData* modelData);

};

#endif //PROGRAM_SELFPLAYMANAGER_H_
