#ifndef NNEVAL_H
#define NNEVAL_H

#include <memory>

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

using tensorflow::Tensor;
using tensorflow::Session;
using tensorflow::GraphDef;

#include "../core/global.h"
#include "../core/logger.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nninputs.h"

class NNEvaluator;

struct NNOutput {
  Hash128 nnHash; //NNInputs - getHashV0 or getHashV1

  //From the perspective of the player to move at the time of the eval
  float whiteValue;

  //Indexed by pos rather than loc
  //Values in here will be set to negative for illegal moves, including superko
  float policyProbs[NNPos::NN_POLICY_SIZE];

  NNOutput(); //Does NOT initialize values
  NNOutput(const NNOutput& other);

  //Utility --------------------------------------------------------------------
  //The utility of having a particular winner
  static double whiteValueOfWinner(Player winner, double drawValue);
  //The utility of achieving a certain score difference
  static double whiteValueOfScore(double finalWhiteMinusBlackScore, int bSize);
};

class NNCacheTable {
  struct Entry {
    shared_ptr<NNOutput> ptr;
    std::atomic_flag spinLock;
    Entry();
    ~Entry();
  };

  Entry* entries;
  uint64_t tableSize;
  uint64_t tableMask;

 public:
  NNCacheTable(int sizePowerOfTwo);
  ~NNCacheTable();

  NNCacheTable(const NNCacheTable& other) = delete;
  NNCacheTable& operator=(const NNCacheTable& other) = delete;
  NNCacheTable(NNCacheTable&& other) = delete;
  NNCacheTable& operator=(NNCacheTable&& other) = delete;

  //These are thread-safe
  bool get(Hash128 nnHash, shared_ptr<NNOutput>& ret);
  void set(const shared_ptr<NNOutput>& p);
  void clear();
};

//Each thread should allocate and re-use one of these
struct NNResultBuf {
  condition_variable clientWaitingForResult;
  mutex resultMutex;
  bool hasResult;
  shared_ptr<NNOutput> result;
  bool errorLogLockout; //error flag to restrict log to 1 error to prevent spam

  NNResultBuf();
  ~NNResultBuf();
  NNResultBuf(const NNResultBuf& other) = delete;
  NNResultBuf& operator=(const NNResultBuf& other) = delete;
  NNResultBuf(NNResultBuf&& other) = delete;
  NNResultBuf& operator=(NNResultBuf&& other) = delete;
};

//Each server thread should allocate and re-use one of these
struct NNServerBuf {
  Session* session;
  vector<string> outputNames;
  vector<string> targetNames;
  vector<Tensor> outputsBuf;

  float* inputsBuffer;
  bool* symmetriesBuffer;
  vector<pair<string,Tensor>>* inputsList;
  NNResultBuf** resultBufs;

  NNServerBuf(const NNEvaluator& nneval, const string* gpuVisibleDevices, double perProcessGPUMemoryFraction, bool debugSkipNeuralNet);
  ~NNServerBuf();
  NNServerBuf(const NNServerBuf& other) = delete;
  NNServerBuf& operator=(const NNServerBuf& other) = delete;
  NNServerBuf(NNServerBuf&& other) = delete;
  NNServerBuf& operator=(NNServerBuf&& other) = delete;
};

class NNEvaluator {
 public:
  NNEvaluator(
    const string& pbModelFile,
    int maxBatchSize,
    int nnCacheSizePowerOfTwo,
    bool debugSkipNeuralNet
  );
  ~NNEvaluator();

  int getMaxBatchSize() const;

  //Clear all entires cached in the table
  void clearCache();

  //Queue a position for the next neural net batch evaluation and wait for it. Upon evaluation, result
  //will be supplied in NNResultBuf& buf, the shared_ptr there can grabbed via std::move if desired.
  //logStream is for some rror logging, can be NULL.
  //This function is threadsafe.
  void evaluate(Board& board, const BoardHistory& history, Player nextPlayer, NNResultBuf& buf, ostream* logStream);

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
    const vector<string>& gpuVisibleDeviceListByThread,  //can be the empty vector if using tensorflow defaults
    double perProcessGPUMemoryFraction //can be -1 to use tensorflow defaults
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
  string modelFileName;
  GraphDef* graphDef;
  NNCacheTable* nnCacheTable;
  bool debugSkipNeuralNet;

  vector<thread*> serverThreads;

  condition_variable clientWaitingForRow;
  condition_variable serverWaitingForBatchStart;
  condition_variable serverWaitingForBatchFinish;
  mutex bufferMutex;
  bool isKilled;
  bool serverTryingToGrabBatch;

  int maxNumRows;
  int m_numRowsStarted;
  int m_numRowsFinished;

  atomic<uint64_t> m_numRowsProcessed;
  atomic<uint64_t> m_numBatchesProcessed;

  float* m_inputsBuffer;
  bool* m_symmetriesBuffer;
  vector<pair<string,Tensor>>* m_inputsList;
  NNResultBuf** m_resultBufs;

 public:
  //Helper, for internal use only
  void serve(NNServerBuf& buf, Rand& rand, bool doRandomize, int defaultSymmetry);
};

#endif
