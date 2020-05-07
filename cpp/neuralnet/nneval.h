#ifndef NEURALNET_NNEVAL_H_
#define NEURALNET_NNEVAL_H_

#include <memory>

#include "../core/global.h"
#include "../core/commontypes.h"
#include "../core/logger.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../search/mutexpool.h"

class NNEvaluator;

class NNCacheTable {
  struct Entry {
    std::shared_ptr<NNOutput> ptr;
    Entry();
    ~Entry();
  };

  Entry* entries;
  MutexPool* mutexPool;
  uint64_t tableSize;
  uint64_t tableMask;
  uint32_t mutexPoolMask;

 public:
  NNCacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);
  ~NNCacheTable();

  NNCacheTable(const NNCacheTable& other) = delete;
  NNCacheTable& operator=(const NNCacheTable& other) = delete;

  //These are thread-safe. For get, ret will be set to nullptr upon a failure to find.
  bool get(Hash128 nnHash, std::shared_ptr<NNOutput>& ret);
  void set(const std::shared_ptr<NNOutput>& p);
  void clear();
};

//Each thread should allocate and re-use one of these
struct NNResultBuf {
  std::condition_variable clientWaitingForResult;
  std::mutex resultMutex;
  bool hasResult;
  bool includeOwnerMap;
  int boardXSizeForServer;
  int boardYSizeForServer;
  int rowSpatialSize;
  int rowGlobalSize;
  float* rowSpatial;
  float* rowGlobal;
  std::shared_ptr<NNOutput> result;
  bool errorLogLockout; //error flag to restrict log to 1 error to prevent spam

  NNResultBuf();
  ~NNResultBuf();
  NNResultBuf(const NNResultBuf& other) = delete;
  NNResultBuf& operator=(const NNResultBuf& other) = delete;
};

//Each server thread should allocate and re-use one of these
struct NNServerBuf {
  InputBuffers* inputBuffers;
  NNResultBuf** resultBufs;

  NNServerBuf(const NNEvaluator& nneval, const LoadedModel* model);
  ~NNServerBuf();
  NNServerBuf(const NNServerBuf& other) = delete;
  NNServerBuf& operator=(const NNServerBuf& other) = delete;
};

class NNEvaluator {
 public:
  NNEvaluator(
    const std::string& modelName,
    const std::string& modelFileName,
    Logger* logger,
    int maxBatchSize,
    int maxConcurrentEvals,
    int nnXLen,
    int nnYLen,
    bool requireExactNNLen,
    bool inputsUseNHWC,
    int nnCacheSizePowerOfTwo,
    int nnMutexPoolSizePowerofTwo,
    bool debugSkipNeuralNet,
    std::string openCLTunerFile,
    bool openCLReTunePerBoardSize,
    enabled_t useFP16Mode,
    enabled_t useNHWCMode,
    int numThreads,
    const std::vector<int>& gpuIdxByServerThread,
    const std::string& randSeed,
    bool doRandomize,
    int defaultSymmetry
  );
  ~NNEvaluator();

  NNEvaluator(const NNEvaluator& other) = delete;
  NNEvaluator& operator=(const NNEvaluator& other) = delete;

  std::string getModelName() const;
  std::string getModelFileName() const;
  std::string getInternalModelName() const;
  bool isNeuralNetLess() const;
  int getMaxBatchSize() const;
  int getNumGpus() const;
  int getNNXLen() const;
  int getNNYLen() const;
  enabled_t getUsingFP16Mode() const;
  enabled_t getUsingNHWCMode() const;

  //Return the "nearest" supported ruleset to desiredRules by this model.
  //Fills supported with true if desiredRules itself was exactly supported, false if some modifications had to be made.
  Rules getSupportedRules(const Rules& desiredRules, bool& supported);

  //Clear all entires cached in the table
  void clearCache();

  //Queue a position for the next neural net batch evaluation and wait for it. Upon evaluation, result
  //will be supplied in NNResultBuf& buf, the shared_ptr there can grabbed via std::move if desired.
  //logStream is for some error logging, can be NULL.
  //This function is threadsafe.
  void evaluate(
    Board& board,
    const BoardHistory& history,
    Player nextPlayer,
    const MiscNNInputParams& nnInputParams,
    NNResultBuf& buf,
    bool skipCache,
    bool includeOwnerMap
  );

  //Actually spawn threads to handle evaluations.
  //If doRandomize, uses randSeed as a seed, further randomized per-thread
  //If not doRandomize, uses defaultSymmetry for all nn evaluations.
  //This function itself is not threadsafe.
  void spawnServerThreads();

  //Kill spawned server threads and join and free them. This function is not threadsafe, and along with spawnServerThreads
  //should have calls to it and spawnServerThreads singlethreaded.
  void killServerThreads();

  //These are thread-safe. Setting them in the middle of operation might only affect future
  //neural net evals, rather than any in-flight.
  bool getDoRandomize() const;
  int getDefaultSymmetry() const;
  void setDoRandomize(bool b);
  void setDefaultSymmetry(int s);

  //Some stats
  uint64_t numRowsProcessed() const;
  uint64_t numBatchesProcessed() const;
  double averageProcessedBatchSize() const;

  void clearStats();

 private:
  const std::string modelName;
  const std::string modelFileName;
  const int nnXLen;
  const int nnYLen;
  const bool requireExactNNLen;
  const int policySize;
  const bool inputsUseNHWC;
  const enabled_t usingFP16Mode;
  const enabled_t usingNHWCMode;
  const int numThreads;
  const std::vector<int> gpuIdxByServerThread;
  const std::string randSeed;
  const bool debugSkipNeuralNet;

  ComputeContext* computeContext;
  LoadedModel* loadedModel;
  NNCacheTable* nnCacheTable;
  Logger* logger;

  int modelVersion;
  int inputsVersion;

  int numServerThreadsEverSpawned;
  std::vector<std::thread*> serverThreads;

  //These are basically constant
  int maxNumRows;
  int numResultBufss;
  int numResultBufssMask;

  //Counters for statistics
  std::atomic<uint64_t> m_numRowsProcessed;
  std::atomic<uint64_t> m_numBatchesProcessed;

  std::condition_variable serverWaitingForBatchStart;
  mutable std::mutex bufferMutex;

  //Everything under here is protected under bufferMutex--------------------------------------------

  bool isKilled; //Flag used for killing server threads

  //Randomization settings for symmetries
  bool currentDoRandomize;
  int currentDefaultSymmetry;

  //An array of NNResultBuf** of length numResultBufss, each NNResultBuf** is an array of NNResultBuf* of length maxNumRows.
  //If a full resultBufs array fills up, client threads can move on to fill up more without waiting. Implemented basically
  //as a circular buffer.
  NNResultBuf*** m_resultBufss;
  int m_currentResultBufsLen; //Number of rows used in in the latest (not yet full) resultBufss.
  int m_currentResultBufsIdx; //Index of the current resultBufs being filled.
  int m_oldestResultBufsIdx; //Index of the oldest resultBufs that still needs to be processed by a server thread

 public:
  //Helper, for internal use only
  void serve(NNServerBuf& buf, Rand& rand,int gpuIdxForThisThread);
};

#endif  // NEURALNET_NNEVAL_H_
