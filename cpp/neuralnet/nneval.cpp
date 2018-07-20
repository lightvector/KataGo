
#include "../neuralnet/nneval.h"

using namespace tensorflow;

NNOutput::NNOutput() {}
NNOutput::NNOutput(const NNOutput& other) {
  nnHash = other.nnHash;
  whiteValue = other.whiteValue;
  std::copy(other.policyProbs, other.policyProbs+NNPos::NN_POLICY_SIZE, policyProbs);
}

double NNOutput::whiteValueOfWinner(Player winner) {
  if(winner == P_WHITE)
    return 1.0;
  else if(winner == P_BLACK)
    return -1.0;
  return 0.0;
}

double NNOutput::whiteValueOfScore(double finalWhiteMinusBlackScore, int bSize) {
  return tanh(finalWhiteMinusBlackScore / (bSize*2));
}

//-------------------------------------------------------------------------------------

NNResultBuf::NNResultBuf()
  :clientWaitingForResult(),resultMutex(),hasResult(false),result(nullptr),errorLogLockout(false)
{}

NNResultBuf::~NNResultBuf()
{}

//-------------------------------------------------------------------------------------

static void checkStatus(const Status& status, const char* subLabel) {
  if(!status.ok())
    throw StringError("NN Eval Error: " + string(subLabel) + status.ToString());
}

//-------------------------------------------------------------------------------------

static void initTensorIOBufs(
  int maxNumRows,
  float*& inputsBuffer,
  bool*& symmetriesBuffer,
  vector<pair<string,Tensor>>*& inputsList,
  NNResultBuf**& resultBufs
) {
  Status status;
  //Set up inputs
  TensorShape inputsShape;
  TensorShape symmetriesShape;
  TensorShape isTrainingShape;
  int inputsShapeArr[3] = {maxNumRows,NNPos::MAX_BOARD_AREA,NNInputs::NUM_FEATURES_V1};
  status = TensorShapeUtils::MakeShape(inputsShapeArr,3,&inputsShape);
  checkStatus(status,"making inputs shape");
  int symmetriesShapeArr[1] = {NNInputs::NUM_SYMMETRY_BOOLS};
  status = TensorShapeUtils::MakeShape(symmetriesShapeArr,1,&symmetriesShape);
  checkStatus(status,"making symmetries shape");
  int isTrainingShapeArr[0] = {};
  status = TensorShapeUtils::MakeShape(isTrainingShapeArr,0,&isTrainingShape);
  checkStatus(status,"making isTraining shape");

  Tensor inputs(DT_FLOAT,inputsShape);
  Tensor symmetries(DT_BOOL,symmetriesShape);
  Tensor isTraining(DT_BOOL,isTrainingShape);

  assert(inputs.IsAligned());
  assert(symmetries.IsAligned());

  inputsBuffer = inputs.flat<float>().data();
  symmetriesBuffer = symmetries.flat<bool>().data();
  auto isTrainingMap = isTraining.tensor<bool, 0>();
  isTrainingMap(0) = false;

  inputsList = new vector<pair<string,Tensor>>();
  *inputsList = {
    {"inputs",inputs},
    {"symmetries",symmetries},
    {"is_training",isTraining},
  };

  resultBufs = new NNResultBuf*[maxNumRows];
  for(int i = 0; i < maxNumRows; i++)
    resultBufs[i] = NULL;
}

static void freeTensorInputBufs(
  float*& inputsBuffer,
  bool*& symmetriesBuffer,
  vector<pair<string,Tensor>>*& inputsList,
  NNResultBuf**& resultBufs
) {
  //Clear these out - these are direct pointers into the inputs and symmetries tensor
  //and are invalid once inputList is cleared and those are freed
  inputsBuffer = NULL;
  symmetriesBuffer = NULL;

  //Explictly clean up tensors - their destructors should get called.
  if(inputsList != NULL)
    inputsList->clear();

  delete inputsList;
  inputsList = NULL;

  //Pointers inside here don't need to be deleted, they simply point to the clients waiting for results
  delete[] resultBufs;
  resultBufs = NULL;
}


NNServerBuf::NNServerBuf(const NNEvaluator& nnEval, const string* gpuVisibleDevices, double perProcessGPUMemoryFraction, bool debugSkipNeuralNet)
  :session(NULL),
   outputNames(),
   targetNames(),
   outputsBuf(),
   inputsBuffer(NULL),
   symmetriesBuffer(NULL),
   inputsList(NULL),
   resultBufs(NULL)
{
  Status status;

  //Create tensorflow session
  if(!debugSkipNeuralNet) {
    SessionOptions sessionOptions = SessionOptions();
    if(gpuVisibleDevices != NULL)
      sessionOptions.config.mutable_gpu_options()->set_visible_device_list(*gpuVisibleDevices);
    if(perProcessGPUMemoryFraction >= 0.0)
      sessionOptions.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(perProcessGPUMemoryFraction);
    status = NewSession(sessionOptions, &session);
    checkStatus(status,"creating session");
  }

  outputNames = {
    string("policy_output"),
    string("value_output")
  };
  targetNames = {};

  initTensorIOBufs(nnEval.getMaxBatchSize(), inputsBuffer, symmetriesBuffer, inputsList, resultBufs);
}

NNServerBuf::~NNServerBuf() {
  //Explictly clean up tensors - their destructors should get called.
  outputsBuf.clear();
  freeTensorInputBufs(inputsBuffer, symmetriesBuffer, inputsList, resultBufs);

  if(session != NULL)
    session->Close();
  session = NULL;
}


//-------------------------------------------------------------------------------------

NNEvaluator::NNEvaluator(
  const string& pbModelFile,
  int maxBatchSize,
  int nnCacheSizePowerOfTwo,
  bool skipNeuralNet
)
  :modelFileName(pbModelFile),
   nnCacheTable(NULL),
   debugSkipNeuralNet(skipNeuralNet),
   serverThreads(),
   clientWaitingForRow(),serverWaitingForBatchStart(),serverWaitingForBatchFinish(),
   bufferMutex(),
   isKilled(false),
   serverTryingToGrabBatch(false),
   maxNumRows(maxBatchSize),
   m_numRowsStarted(0),
   m_numRowsFinished(0),
   m_numRowsProcessed(0),
   m_numBatchesProcessed(0),
   m_inputsBuffer(NULL),
   m_symmetriesBuffer(NULL),
   m_inputsList(NULL)
{
  Status status;
  graphDef = new GraphDef();

  if(nnCacheSizePowerOfTwo >= 0)
    nnCacheTable = new NNCacheTable(nnCacheSizePowerOfTwo);

  //Read graph from file
  status = ReadBinaryProto(Env::Default(), pbModelFile, graphDef);
  checkStatus(status,"reading graph");

  initTensorIOBufs(maxNumRows, m_inputsBuffer, m_symmetriesBuffer, m_inputsList, m_resultBufs);
}

NNEvaluator::~NNEvaluator()
{
  killServerThreads();
  assert(!serverTryingToGrabBatch);
  freeTensorInputBufs(m_inputsBuffer, m_symmetriesBuffer, m_inputsList, m_resultBufs);
  delete nnCacheTable;
  delete graphDef;
}

int NNEvaluator::getMaxBatchSize() const {
  return maxNumRows;
}

uint64_t NNEvaluator::numRowsProcessed() const {
  return m_numRowsProcessed.load(std::memory_order_relaxed);
}
uint64_t NNEvaluator::numBatchesProcessed() const {
  return m_numBatchesProcessed.load(std::memory_order_relaxed);
}
double NNEvaluator::averageProcessedBatchSize() const {
  return (double)numRowsProcessed() / (double)numBatchesProcessed();
}

void NNEvaluator::clearStats() {
  m_numRowsProcessed.store(0);
  m_numBatchesProcessed.store(0);
}

void NNEvaluator::clearCache() {
  if(nnCacheTable != NULL)
    nnCacheTable->clear();
}

static void serveEvals(
  int threadIdx, bool doRandomize, string randSeed, int defaultSymmetry, Logger* logger, NNEvaluator* nnEval,
  const string* gpuVisibleDevices, double perProcessGPUMemoryFraction, bool debugSkipNeuralNet
) {
  NNServerBuf* buf = new NNServerBuf(*nnEval, gpuVisibleDevices, perProcessGPUMemoryFraction, debugSkipNeuralNet);
  Rand rand(randSeed + ":NNEvalServerThread:" + Global::intToString(threadIdx));
  ostream* logStream = logger->createOStream();
  try {
    nnEval->serve(*buf,rand,doRandomize,defaultSymmetry);
  }
  catch(const exception& e) {
    (*logStream) << "ERROR: NNEval Server Thread " << threadIdx << " failed: " << e.what() << endl;
  }
  catch(const string& e) {
    (*logStream) << "ERROR: NNEval Server Thread " << threadIdx << " failed: " << e << endl;
  }
  catch(...) {
    (*logStream) << "ERROR: NNEval Server Thread " << threadIdx << " failed with unexpected throw" << endl;
  }
  delete logStream;
  delete buf;
}

void NNEvaluator::spawnServerThreads(
  int numThreads,
  bool doRandomize,
  string randSeed,
  int defaultSymmetry,
  Logger& logger,
  const vector<string>& gpuVisibleDeviceListByThread,
  double perProcessGPUMemoryFraction
) {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");

  if(gpuVisibleDeviceListByThread.size() > 0 && gpuVisibleDeviceListByThread.size() != numThreads)
    throw StringError("NNEvaluator::spawnServerThreads gpuVisibleDeviceListByThread is not the same size as the number of threads!");

  for(int i = 0; i<numThreads; i++) {
    const string* gpuVisibleDevices = NULL;
    if(gpuVisibleDeviceListByThread.size() > 0)
      gpuVisibleDevices = &(gpuVisibleDeviceListByThread[i]);
    std::thread* thread = new std::thread(
      &serveEvals,i,doRandomize,randSeed,defaultSymmetry,&logger,this,
      gpuVisibleDevices,perProcessGPUMemoryFraction,debugSkipNeuralNet
    );
    serverThreads.push_back(thread);
  }
}

void NNEvaluator::killServerThreads() {
  unique_lock<std::mutex> lock(bufferMutex);
  isKilled = true;
  lock.unlock();
  serverWaitingForBatchStart.notify_all();
  serverWaitingForBatchFinish.notify_all();

  for(size_t i = 0; i<serverThreads.size(); i++)
    serverThreads[i]->join();
  for(size_t i = 0; i<serverThreads.size(); i++)
    delete serverThreads[i];
  serverThreads.clear();

  //Can unset now that threads are dead
  isKilled = false;
}

void NNEvaluator::serve(NNServerBuf& buf, Rand& rand, bool doRandomize, int defaultSymmetry) {
  Status status;
  //Add graph to session
  status = buf.session->Create(*graphDef);
  checkStatus(status,"adding graph to session");

  vector<pair<string,Tensor>> slicedInputsList;

  unique_lock<std::mutex> lock(bufferMutex,std::defer_lock);
  while(true) {
    lock.lock();
    while((m_numRowsStarted <= 0 || serverTryingToGrabBatch) && !isKilled)
      serverWaitingForBatchStart.wait(lock);

    if(isKilled)
      break;

    serverTryingToGrabBatch = true;
    while(m_numRowsFinished < m_numRowsStarted && !isKilled)
      serverWaitingForBatchFinish.wait(lock);

    if(isKilled)
      break;

    //It should only be possible for one thread to make it through to here
    assert(serverTryingToGrabBatch);
    assert(m_numRowsFinished > 0);

    int numRows = m_numRowsFinished;
    std::swap(m_inputsBuffer, buf.inputsBuffer);
    std::swap(m_symmetriesBuffer, buf.symmetriesBuffer);
    std::swap(m_inputsList,buf.inputsList);
    std::swap(m_resultBufs,buf.resultBufs);

    m_numRowsStarted = 0;
    m_numRowsFinished = 0;
    serverTryingToGrabBatch = false;
    clientWaitingForRow.notify_all();
    lock.unlock();

    slicedInputsList = *buf.inputsList;
    slicedInputsList[0].second = (*buf.inputsList)[0].second.Slice(0,numRows);

    int symmetry = defaultSymmetry;
    if(doRandomize)
      symmetry = rand.nextUInt(NNInputs::NUM_SYMMETRY_COMBINATIONS);
    buf.symmetriesBuffer[0] = (symmetry & 0x1) != 0;
    buf.symmetriesBuffer[1] = (symmetry & 0x2) != 0;
    buf.symmetriesBuffer[2] = (symmetry & 0x4) != 0;

    if(debugSkipNeuralNet) {
      for(int row = 0; row < numRows; row++) {
        assert(buf.resultBufs[row] != NULL);
        NNResultBuf* resultBuf = buf.resultBufs[row];
        buf.resultBufs[row] = NULL;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        assert(resultBuf->hasResult == false);
        resultBuf->result = std::make_shared<NNOutput>();
        float* policyProbs = resultBuf->result->policyProbs;
        for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++)
          policyProbs[i] = rand.nextGaussian();
        resultBuf->result->whiteValue = rand.nextGaussian() * 0.1;
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
      continue;
    }

    status = buf.session->Run(slicedInputsList, buf.outputNames, buf.targetNames, &(buf.outputsBuf));
    checkStatus(status,"running inference");

    assert(buf.outputsBuf.size() == 2);
    assert(buf.outputsBuf[0].dims() == 2);
    assert(buf.outputsBuf[1].dims() == 1);
    assert(buf.outputsBuf[0].dim_size(0) == numRows);
    assert(buf.outputsBuf[0].dim_size(1) == NNPos::NN_POLICY_SIZE);
    assert(buf.outputsBuf[1].dim_size(0) == numRows);

    assert(buf.outputsBuf[0].IsAligned());
    assert(buf.outputsBuf[1].IsAligned());

    float* policyData = buf.outputsBuf[0].flat<float>().data();
    float* valueData = buf.outputsBuf[1].flat<float>().data();

    for(int row = 0; row < numRows; row++) {
      assert(buf.resultBufs[row] != NULL);
      NNResultBuf* resultBuf = buf.resultBufs[row];
      buf.resultBufs[row] = NULL;

      unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
      assert(resultBuf->hasResult == false);
      resultBuf->result = std::make_shared<NNOutput>();
      float* policyProbs = resultBuf->result->policyProbs;

      //These are not actually correct, the client does the postprocessing to turn them into
      //probabilities and white value
      //Also we don't fill in the nnHash here either
      std::copy(
        policyData + row * NNPos::NN_POLICY_SIZE,
        policyData + (row+1) * NNPos::NN_POLICY_SIZE,
        policyProbs
      );
      resultBuf->result->whiteValue = valueData[row];
      resultBuf->hasResult = true;
      resultBuf->clientWaitingForResult.notify_all();
      resultLock.unlock();
    }
    buf.outputsBuf.clear();

    m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
    m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
    continue;
  }

}

void NNEvaluator::evaluate(Board& board, const BoardHistory& history, Player nextPlayer, NNResultBuf& buf, ostream* logStream) {
  assert(!isKilled);
  buf.hasResult = false;

  Hash128 nnHash = NNInputs::getHashV1(board, history, nextPlayer);
  if(nnCacheTable != NULL && nnCacheTable->get(nnHash,buf.result)) {
    buf.hasResult = true;
    return;
  }

  unique_lock<std::mutex> lock(bufferMutex);
  while(m_numRowsStarted >= maxNumRows || serverTryingToGrabBatch)
    clientWaitingForRow.wait(lock);

  int rowIdx = m_numRowsStarted;
  m_numRowsStarted += 1;
  float* rowInput = m_inputsBuffer + rowIdx * NNInputs::ROW_SIZE_V1;

  if(m_numRowsStarted == 1)
    serverWaitingForBatchStart.notify_one();
  lock.unlock();

  std::fill(rowInput,rowInput+NNInputs::ROW_SIZE_V1,0.0f);
  NNInputs::fillRowV1(board, history, nextPlayer, rowInput);

  lock.lock();
  m_resultBufs[rowIdx] = &buf;
  m_numRowsFinished += 1;
  if(m_numRowsFinished >= m_numRowsStarted)
    serverWaitingForBatchFinish.notify_all();
  lock.unlock();

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
  resultLock.unlock();

  //Perform postprocessing on the result - turn the nn output into probabilities
  float* policy = buf.result->policyProbs;

  assert(board.x_size == board.y_size);
  int bSize = board.x_size;
  int offset = NNPos::getOffset(bSize);

  float maxPolicy = -1e25f;
  bool isLegal[NNPos::NN_POLICY_SIZE];
  int legalCount = 0;
  for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
    Loc loc = NNPos::posToLoc(i,bSize,offset);
    isLegal[i] = history.isLegal(board,loc,nextPlayer);

    float policyValue;
    if(isLegal[i]) {
      legalCount += 1;
      policyValue = policy[i];
    }
    else {
      policyValue = -1e30f;
      policy[i] = policyValue;
    }

    if(policyValue > maxPolicy)
      maxPolicy = policyValue;
  }

  assert(legalCount > 0);

  float policySum = 0.0f;
  for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
    policy[i] = exp(policy[i] - maxPolicy);
    policySum += policy[i];
  }

  //Somehow all legal moves rounded to 0 probability
  if(policySum <= 0.0) {
    if(!buf.errorLogLockout && logStream != NULL) {
      buf.errorLogLockout = true;
      (*logStream) << "Warning: all legal moves rounded to 0 probability for " << modelFileName << " in position " << board << endl;
    }
    float uniform = 1.0f / legalCount;
    for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
      policy[i] = isLegal[i] ? uniform : -1.0f;
    }
  }
  else {
    for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++)
      policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;
  }

  //Fix up the value as well
  if(nextPlayer == P_WHITE)
    buf.result->whiteValue = tanh(buf.result->whiteValue);
  else
    buf.result->whiteValue = -tanh(buf.result->whiteValue);

  //And record the nnHash in the result and put it into the table
  buf.result->nnHash = nnHash;
  if(nnCacheTable != NULL)
    nnCacheTable->set(buf.result);

}



NNCacheTable::Entry::Entry()
  :ptr(nullptr),spinLock(ATOMIC_FLAG_INIT)
{}
NNCacheTable::Entry::~Entry()
{}

NNCacheTable::NNCacheTable(int sizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  tableSize = ((uint64_t)1) << sizePowerOfTwo;
  tableMask = tableSize-1;
  entries = new Entry[tableSize];
}
NNCacheTable::~NNCacheTable() {
  delete[] entries;
}

bool NNCacheTable::get(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  uint64_t idx = nnHash.hash0 & tableMask;
  Entry& entry = entries[idx];
  while(entry.spinLock.test_and_set(std::memory_order_acquire));
  bool found = false;
  if(entry.ptr != nullptr && entry.ptr->nnHash == nnHash) {
    ret = entry.ptr;
    found = true;
  }
  entry.spinLock.clear(std::memory_order_release);
  return found;
}

void NNCacheTable::set(const shared_ptr<NNOutput>& p) {
  uint64_t idx = p->nnHash.hash0 & tableMask;
  Entry& entry = entries[idx];
  while(entry.spinLock.test_and_set(std::memory_order_acquire));
  entry.ptr = p;
  entry.spinLock.clear(std::memory_order_release);
}

void NNCacheTable::clear() {
  for(size_t idx = 0; idx<tableSize; idx++) {
    Entry& entry = entries[idx];
    while(entry.spinLock.test_and_set(std::memory_order_acquire));
    entry.ptr = nullptr;
    entry.spinLock.clear(std::memory_order_release);
  }
}

