#ifndef NNEVAL_H
#define NNEVAL_H

#include <memory>

#include "../core/global.h"
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
    shared_ptr<NNOutput> ptr;
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
  bool get(Hash128 nnHash, shared_ptr<NNOutput>& ret);
  void set(const shared_ptr<NNOutput>& p);
  void clear();
};

//Each thread should allocate and re-use one of these
struct NNResultBuf {
  condition_variable clientWaitingForResult;
  mutex resultMutex;
  bool hasResult;
  bool includeOwnerMap;
  int boardXSizeForServer;
  int boardYSizeForServer;
  int rowBinSize;
  int rowGlobalSize;
  float* rowBin;
  float* rowGlobal;
  shared_ptr<NNOutput> result;
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
    const string& modelName,
    const string& modelFileName,
    int modelFileIdx,
    int maxBatchSize,
    int maxConcurrentEvals,
    int posLen,
    bool requireExactPosLen,
    bool inputsUseNHWC,
    int nnCacheSizePowerOfTwo,
    int nnMutexPoolSizePowerofTwo,
    bool debugSkipNeuralNet,
    bool alwaysIncludeOwnerMap,
    float nnPolicyTemperature
  );
  ~NNEvaluator();

  string getModelName() const;
  string getModelFileName() const;
  int getMaxBatchSize() const;
  int getPosLen() const;

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
    double drawEquivalentWinsForWhite,
    NNResultBuf& buf,
    Logger* logger,
    bool skipCache,
    bool includeOwnerMap
  );

  //Actually spawn threads and return the results.
  //If doRandomize, uses randSeed as a seed, further randomized per-thread
  //If not doRandomize, uses defaultSymmetry for all nn evaluations.
  //This function itself is not threadsafe.
  void spawnServerThreads(
    int numThreads,
    bool doRandomize,
    string randSeed,
    int defaultSymmetry,
    Logger& logger,
    vector<int> cudaGpuIdxByServerThread,
    bool cudaUseFP16,
    bool cudaUseNHWC
  );

  //Kill spawned server threads and join and free them. This function is not threadsafe, and along with spawnServerThreads
  //should have calls to it and spawnServerThreads singlethreaded.
  void killServerThreads();

  //Some stats
  uint64_t numRowsProcessed() const;
  uint64_t numBatchesProcessed() const;
  double averageProcessedBatchSize() const;

  void clearStats();

 private:
  string modelName;
  string modelFileName;
  int posLen;
  bool requireExactPosLen;
  int policySize;
  bool inputsUseNHWC;

  LoadedModel* loadedModel;
  NNCacheTable* nnCacheTable;

  bool debugSkipNeuralNet;
  bool alwaysIncludeOwnerMap;
  float nnPolicyInvTemperature;

  int modelVersion;
  int inputsVersion;

  vector<thread*> serverThreads;

  condition_variable serverWaitingForBatchStart;
  mutex bufferMutex;
  bool isKilled;

  int maxNumRows;
  int numResultBufss;
  int numResultBufssMask;

  atomic<uint64_t> m_numRowsProcessed;
  atomic<uint64_t> m_numBatchesProcessed;

  //An array of NNResultBuf** of length numResultBufss, each NNResultBuf** is an array of NNResultBuf* of length maxNumRows.
  //If a full resultBufs array fills up, client threads can move on to fill up more without waiting. Implemented basically
  //as a circular buffer.
  NNResultBuf*** m_resultBufss;
  int m_currentResultBufsLen; //Number of rows used in in the latest (not yet full) resultBufss.
  int m_currentResultBufsIdx; //Index of the current resultBufs being filled.
  int m_oldestResultBufsIdx; //Index of the oldest resultBufs that still needs to be processed by a server thread

 public:
  //Helper, for internal use only
  void serve(
    NNServerBuf& buf, Rand& rand, Logger* logger, bool doRandomize, int defaultSymmetry,
    int cudaGpuIdxForThisThread, bool cudaUseFP16, bool cudaUseNHWC
  );
};

#endif
