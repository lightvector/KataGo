
#include "../neuralnet/nneval.h"

//-------------------------------------------------------------------------------------

NNResultBuf::NNResultBuf()
  :clientWaitingForResult(),resultMutex(),hasResult(false),includeOwnerMap(false),
   boardXSizeForServer(0),boardYSizeForServer(0),rowBinSize(0),rowGlobalSize(0),rowBin(NULL),rowGlobal(NULL),
   result(nullptr),errorLogLockout(false)
{}

NNResultBuf::~NNResultBuf()
{
  if(rowBin != NULL)
    delete[] rowBin;
  if(rowGlobal != NULL)
    delete[] rowGlobal;
}

//-------------------------------------------------------------------------------------

NNServerBuf::NNServerBuf(const NNEvaluator& nnEval, const LoadedModel* model)
  :inputBuffers(NULL),
   resultBufs(NULL)
{
  int maxNumRows = nnEval.getMaxBatchSize();
  if(model != NULL)
    inputBuffers = NeuralNet::createInputBuffers(model,maxNumRows,nnEval.getPosLen());
  resultBufs = new NNResultBuf*[maxNumRows];
  for(int i = 0; i < maxNumRows; i++)
    resultBufs[i] = NULL;
}

NNServerBuf::~NNServerBuf() {
  if(inputBuffers != NULL)
    NeuralNet::freeInputBuffers(inputBuffers);
  inputBuffers = NULL;
  //Pointers inside here don't need to be deleted, they simply point to the clients waiting for results
  delete[] resultBufs;
  resultBufs = NULL;
}

//-------------------------------------------------------------------------------------

NNEvaluator::NNEvaluator(
  const string& mName,
  const string& mFileName,
  int modelFileIdx,
  int maxBatchSize,
  int maxConcurrentEvals,
  int pLen,
  bool rExactPosLen,
  bool iUseNHWC,
  int nnCacheSizePowerOfTwo,
  int nnMutexPoolSizePowerofTwo,
  bool skipNeuralNet,
  bool alwaysOwnerMap,
  float nnPolicyTemp
)
  :modelName(mName),
   modelFileName(mFileName),
   posLen(pLen),
   requireExactPosLen(rExactPosLen),
   policySize(NNPos::getPolicySize(pLen)),
   inputsUseNHWC(iUseNHWC),
   loadedModel(NULL),
   nnCacheTable(NULL),
   debugSkipNeuralNet(skipNeuralNet),
   alwaysIncludeOwnerMap(alwaysOwnerMap),
   nnPolicyInvTemperature(1.0/nnPolicyTemp),
   serverThreads(),
   serverWaitingForBatchStart(),
   bufferMutex(),
   isKilled(false),
   maxNumRows(maxBatchSize),
   numResultBufss(),
   numResultBufssMask(),
   m_numRowsProcessed(0),
   m_numBatchesProcessed(0),
   m_resultBufss(NULL),
   m_currentResultBufsLen(0),
   m_currentResultBufsIdx(0),
   m_oldestResultBufsIdx(0)
{
  if(posLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(maxConcurrentEvals <= 0)
    throw StringError("maxConcurrentEvals is negative: " + Global::intToString(maxConcurrentEvals));
  if(maxBatchSize <= 0)
    throw StringError("maxBatchSize is negative: " + Global::intToString(maxBatchSize));

  //Add three, just to give a bit of extra headroom, and make it a power of two
  numResultBufss = maxConcurrentEvals / maxBatchSize + 3;
  {
    int x = 1;
    while(x < numResultBufss) x *= 2;
    numResultBufss = x;
  }
  numResultBufssMask = numResultBufss-1;

  if(nnCacheSizePowerOfTwo >= 0)
    nnCacheTable = new NNCacheTable(nnCacheSizePowerOfTwo,nnMutexPoolSizePowerofTwo);

  if(!debugSkipNeuralNet) {
    loadedModel = NeuralNet::loadModelFile(modelFileName, modelFileIdx);
    modelVersion = NeuralNet::getModelVersion(loadedModel);
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  }
  else {
    modelVersion = NNModelVersion::defaultModelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  }

  m_resultBufss = new NNResultBuf**[numResultBufss];
  for(int i = 0; i<numResultBufss; i++) {
    m_resultBufss[i] = new NNResultBuf*[maxBatchSize];
    for(int j = 0; j < maxBatchSize; j++)
      m_resultBufss[i][j] = NULL;
  }
}

NNEvaluator::~NNEvaluator()
{
  killServerThreads();

  for(int i = 0; i<numResultBufss; i++) {
    NNResultBuf** resultBufs = m_resultBufss[i];
    //Pointers inside here don't need to be deleted, they simply point to the clients waiting for results
    delete[] resultBufs;
    m_resultBufss[i] = NULL;
  }
  delete[] m_resultBufss;
  m_resultBufss = NULL;

  if(loadedModel != NULL)
    NeuralNet::freeLoadedModel(loadedModel);
  loadedModel = NULL;

  delete nnCacheTable;
}

string NNEvaluator::getModelName() const {
  return modelName;
}
string NNEvaluator::getModelFileName() const {
  return modelFileName;
}
int NNEvaluator::getMaxBatchSize() const {
  return maxNumRows;
}
int NNEvaluator::getPosLen() const {
  return posLen;
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
  int threadIdx, bool doRandomize, string randSeed, int defaultSymmetry, Logger* logger,
  NNEvaluator* nnEval, const LoadedModel* loadedModel,
  int cudaGpuIdxForThisThread,
  bool cudaUseFP16,
  bool cudaUseNHWC
) {
  NNServerBuf* buf = new NNServerBuf(*nnEval,loadedModel);
  Rand rand(randSeed + ":NNEvalServerThread:" + Global::intToString(threadIdx));

  //Used to have a try catch around this but actually we're in big trouble if this raises an exception
  //and causes possibly the only nnEval thread to die, so actually go ahead and let the exception escape to
  //toplevel for easier debugging
  nnEval->serve(*buf,rand,logger,doRandomize,defaultSymmetry,cudaGpuIdxForThisThread,cudaUseFP16,cudaUseNHWC);
  delete buf;
}

void NNEvaluator::spawnServerThreads(
  int numThreads,
  bool doRandomize,
  string randSeed,
  int defaultSymmetry,
  Logger& logger,
  vector<int> cudaGpuIdxByServerThread,
  bool cudaUseFP16,
  bool cudaUseNHWC
) {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");
  if(cudaGpuIdxByServerThread.size() != numThreads)
    throw StringError("cudaGpuIdxByServerThread.size() != numThreads");

  for(int i = 0; i<numThreads; i++) {
    int cudaGpuIdxForThisThread = cudaGpuIdxByServerThread[i];
    std::thread* thread = new std::thread(
      &serveEvals,i,doRandomize,randSeed,defaultSymmetry,&logger,this,loadedModel,cudaGpuIdxForThisThread,cudaUseFP16,cudaUseNHWC
    );
    serverThreads.push_back(thread);
  }
}

void NNEvaluator::killServerThreads() {
  unique_lock<std::mutex> lock(bufferMutex);
  isKilled = true;
  lock.unlock();
  serverWaitingForBatchStart.notify_all();

  for(size_t i = 0; i<serverThreads.size(); i++)
    serverThreads[i]->join();
  for(size_t i = 0; i<serverThreads.size(); i++)
    delete serverThreads[i];
  serverThreads.clear();

  //Can unset now that threads are dead
  isKilled = false;
}

void NNEvaluator::serve(
  NNServerBuf& buf, Rand& rand, Logger* logger, bool doRandomize, int defaultSymmetry,
  int cudaGpuIdxForThisThread, bool cudaUseFP16, bool cudaUseNHWC
) {

  LocalGpuHandle* gpuHandle = NULL;
  if(loadedModel != NULL)
    gpuHandle = NeuralNet::createLocalGpuHandle(loadedModel, logger, maxNumRows, posLen, requireExactPosLen, inputsUseNHWC, cudaGpuIdxForThisThread, cudaUseFP16, cudaUseNHWC);

  vector<NNOutput*> outputBuf;

  unique_lock<std::mutex> lock(bufferMutex,std::defer_lock);
  while(true) {
    lock.lock();
    while(m_currentResultBufsLen <= 0 && m_currentResultBufsIdx == m_oldestResultBufsIdx && !isKilled)
      serverWaitingForBatchStart.wait(lock);

    if(isKilled)
      break;

    std::swap(m_resultBufss[m_oldestResultBufsIdx],buf.resultBufs);

    int numRows;
    //We grabbed everything in the latest buffer, so clients should move on to an entirely new buffer
    if(m_currentResultBufsIdx == m_oldestResultBufsIdx) {
      m_oldestResultBufsIdx = (m_oldestResultBufsIdx + 1) & numResultBufssMask;
      m_currentResultBufsIdx = m_oldestResultBufsIdx;
      numRows = m_currentResultBufsLen;
      m_currentResultBufsLen = 0;
    }
    //We grabbed a buffer that clients have already entirely moved onward from.
    else {
      m_oldestResultBufsIdx = (m_oldestResultBufsIdx + 1) & numResultBufssMask;
      numRows = maxNumRows;
    }

    lock.unlock();

    if(debugSkipNeuralNet) {
      for(int row = 0; row < numRows; row++) {
        assert(buf.resultBufs[row] != NULL);
        NNResultBuf* resultBuf = buf.resultBufs[row];
        buf.resultBufs[row] = NULL;

        int boardXSize = resultBuf->boardXSizeForServer;
        int boardYSize = resultBuf->boardYSizeForServer;
        
        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        assert(resultBuf->hasResult == false);
        resultBuf->result = std::make_shared<NNOutput>();

        float* policyProbs = resultBuf->result->policyProbs;
        for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
          policyProbs[i] = 0;
        
        //At this point, these aren't probabilities, since this is before the postprocessing
        //that happens for each result. These just need to be unnormalized log probabilities.
        //Illegal move filtering happens later.
        for(int y = 0; y<boardYSize; y++) {
          for(int x = 0; x<boardXSize; x++) {
            int pos = NNPos::xyToPos(x,y,posLen);
            policyProbs[pos] = rand.nextGaussian();
          }
        }
        policyProbs[NNPos::locToPos(Board::PASS_LOC,boardXSize,posLen)] = rand.nextGaussian();

        resultBuf->result->posLen = posLen;
        if(resultBuf->includeOwnerMap) {
          float* whiteOwnerMap = new float[posLen*posLen];
          for(int i = 0; i<posLen*posLen; i++)
            whiteOwnerMap[i] = 0.0;
          for(int y = 0; y<boardYSize; y++) {
            for(int x = 0; x<boardXSize; x++) {
              int pos = NNPos::xyToPos(x,y,posLen);
              whiteOwnerMap[pos] = rand.nextGaussian() * 0.20;
            }
          }
          resultBuf->result->whiteOwnerMap = whiteOwnerMap;
        }
        else {
          resultBuf->result->whiteOwnerMap = NULL;
        }

        //These aren't really probabilities. Win/Loss/NoResult will get softmaxed later
        //(or in the case of model version 2, it will only just pay attention to the value of whiteWinProb and tanh it)
        double whiteWinProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteLossProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMean = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMeanSq = 0.0 + rand.nextGaussian() * 0.20;
        double whiteNoResultProb = 0.0 + rand.nextGaussian() * 0.20;
        resultBuf->result->whiteWinProb = whiteWinProb;
        resultBuf->result->whiteLossProb = whiteLossProb;
        resultBuf->result->whiteNoResultProb = whiteNoResultProb;
        resultBuf->result->whiteScoreMean = whiteScoreMean;
        resultBuf->result->whiteScoreMeanSq = whiteScoreMeanSq;
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
      continue;
    }

    int symmetry = defaultSymmetry;
    if(doRandomize)
      symmetry = rand.nextUInt(NNInputs::NUM_SYMMETRY_COMBINATIONS);
    bool* symmetriesBuffer = NeuralNet::getSymmetriesInplace(buf.inputBuffers);
    symmetriesBuffer[0] = (symmetry & 0x1) != 0;
    symmetriesBuffer[1] = (symmetry & 0x2) != 0;
    symmetriesBuffer[2] = (symmetry & 0x4) != 0;

    outputBuf.clear();
    for(int row = 0; row<numRows; row++) {
      NNOutput* emptyOutput = new NNOutput();
      assert(buf.resultBufs[row] != NULL);
      emptyOutput->posLen = posLen;
      if(buf.resultBufs[row]->includeOwnerMap)
        emptyOutput->whiteOwnerMap = new float[posLen*posLen];
      else
        emptyOutput->whiteOwnerMap = NULL;
      outputBuf.push_back(emptyOutput);
    }

    int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
    int rowBinLen = numSpatialFeatures * posLen * posLen;
    int rowGlobalLen = numGlobalFeatures;
    assert(rowBinLen == NeuralNet::getRowLen(buf.inputBuffers));
    assert(rowGlobalLen == NeuralNet::getRowGlobalLen(buf.inputBuffers));

    for(int row = 0; row<numRows; row++) {
      float* rowInput = NeuralNet::getRowInplace(buf.inputBuffers,row);
      float* rowGlobalInput = NeuralNet::getRowGlobalInplace(buf.inputBuffers,row);

      const float* rowBin = buf.resultBufs[row]->rowBin;
      const float* rowGlobal = buf.resultBufs[row]->rowGlobal;
      std::copy(rowBin,rowBin+rowBinLen,rowInput);
      std::copy(rowGlobal,rowGlobal+rowGlobalLen,rowGlobalInput);
    }

    NeuralNet::getOutput(gpuHandle, buf.inputBuffers, numRows, outputBuf);
    assert(outputBuf.size() == numRows);

    m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
    m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);

    for(int row = 0; row < numRows; row++) {
      assert(buf.resultBufs[row] != NULL);
      NNResultBuf* resultBuf = buf.resultBufs[row];
      buf.resultBufs[row] = NULL;

      unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
      assert(resultBuf->hasResult == false);
      resultBuf->result = std::shared_ptr<NNOutput>(outputBuf[row]);
      resultBuf->hasResult = true;
      resultBuf->clientWaitingForResult.notify_all();
      resultLock.unlock();
    }

    continue;
  }

  NeuralNet::freeLocalGpuHandle(gpuHandle);
}

void NNEvaluator::evaluate(
  Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  double drawEquivalentWinsForWhite,
  NNResultBuf& buf,
  Logger* logger,
  bool skipCache,
  bool includeOwnerMap
) {
  assert(!isKilled);
  buf.hasResult = false;

  if(board.x_size > posLen || board.y_size > posLen)
    throw StringError("NNEvaluator was configured with posLen = " + Global::intToString(posLen) +
                      " but was asked to evaluate board with larger x or y size");
  if(requireExactPosLen) {
    if(board.x_size != posLen || board.y_size != posLen)
      throw StringError("NNEvaluator was configured with posLen = " + Global::intToString(posLen) +
                        " and requireExactPosLen, but was asked to evaluate board with different x or y size");
  }

  Hash128 nnHash;
  if(inputsVersion == 1)
    nnHash = NNInputs::getHashV1(board, history, nextPlayer);
  else if(inputsVersion == 2)
    nnHash = NNInputs::getHashV2(board, history, nextPlayer);
  else if(inputsVersion == 3)
    nnHash = NNInputs::getHashV3(board, history, nextPlayer, drawEquivalentWinsForWhite);
  else if(inputsVersion == 4)
    nnHash = NNInputs::getHashV4(board, history, nextPlayer, drawEquivalentWinsForWhite);
  else if(inputsVersion == 5)
    nnHash = NNInputs::getHashV5(board, history, nextPlayer, drawEquivalentWinsForWhite);
  else
    assert(false);

  includeOwnerMap |= alwaysIncludeOwnerMap;
  
  bool hadResultWithoutOwnerMap = false;
  shared_ptr<NNOutput> resultWithoutOwnerMap;
  if(nnCacheTable != NULL && !skipCache && nnCacheTable->get(nnHash,buf.result)) {
    if(!(includeOwnerMap && buf.result->whiteOwnerMap == NULL))
    {
      buf.hasResult = true;
      return;
    }
    else {
      hadResultWithoutOwnerMap = true;
      resultWithoutOwnerMap = std::move(buf.result);
      buf.result = nullptr;
    }
  }
  buf.includeOwnerMap = includeOwnerMap;

  buf.boardXSizeForServer = board.x_size;
  buf.boardYSizeForServer = board.y_size;
  
  if(!debugSkipNeuralNet) {
    int rowBinLen = NNModelVersion::getNumSpatialFeatures(modelVersion) * posLen * posLen;
    if(buf.rowBin == NULL) {
      buf.rowBin = new float[rowBinLen];
      buf.rowBinSize = rowBinLen;
    }
    else {
      if(buf.rowBinSize != rowBinLen)
        throw StringError("Cannot reuse an nnResultBuf with a different posLen or model version");
    }

    if(inputsVersion == 1)
      NNInputs::fillRowV1(board, history, nextPlayer, posLen, inputsUseNHWC, buf.rowBin);
    else if(inputsVersion == 2)
      NNInputs::fillRowV2(board, history, nextPlayer, posLen, inputsUseNHWC, buf.rowBin);
    else if(inputsVersion == 3) {
      int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(modelVersion);
      if(buf.rowGlobal == NULL) {
        buf.rowGlobal = new float[rowGlobalLen];
        buf.rowGlobalSize = rowGlobalLen;
      }
      else {
        if(buf.rowGlobalSize != rowGlobalLen)
          throw StringError("Cannot reuse an nnResultBuf with a different posLen or model version");
      }
      NNInputs::fillRowV3(board, history, nextPlayer, drawEquivalentWinsForWhite, posLen, inputsUseNHWC, buf.rowBin, buf.rowGlobal);
    }
    else if(inputsVersion == 4) {
      int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(modelVersion);
      if(buf.rowGlobal == NULL) {
        buf.rowGlobal = new float[rowGlobalLen];
        buf.rowGlobalSize = rowGlobalLen;
      }
      else {
        if(buf.rowGlobalSize != rowGlobalLen)
          throw StringError("Cannot reuse an nnResultBuf with a different posLen or model version");
      }
      NNInputs::fillRowV4(board, history, nextPlayer, drawEquivalentWinsForWhite, posLen, inputsUseNHWC, buf.rowBin, buf.rowGlobal);
    }
    else if(inputsVersion == 5) {
      int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(modelVersion);
      if(buf.rowGlobal == NULL) {
        buf.rowGlobal = new float[rowGlobalLen];
        buf.rowGlobalSize = rowGlobalLen;
      }
      else {
        if(buf.rowGlobalSize != rowGlobalLen)
          throw StringError("Cannot reuse an nnResultBuf with a different posLen or model version");
      }
      NNInputs::fillRowV5(board, history, nextPlayer, drawEquivalentWinsForWhite, posLen, inputsUseNHWC, buf.rowBin, buf.rowGlobal);
    }
    else
      assert(false);
  }

  unique_lock<std::mutex> lock(bufferMutex);

  m_resultBufss[m_currentResultBufsIdx][m_currentResultBufsLen] = &buf;
  m_currentResultBufsLen += 1;
  if(m_currentResultBufsLen == 1 && m_currentResultBufsIdx == m_oldestResultBufsIdx)
    serverWaitingForBatchStart.notify_one();

  bool overlooped = false;
  if(m_currentResultBufsLen >= maxNumRows) {
    m_currentResultBufsLen = 0;
    m_currentResultBufsIdx = (m_currentResultBufsIdx + 1) & numResultBufssMask;
    overlooped = m_currentResultBufsIdx == m_oldestResultBufsIdx;
  }
  lock.unlock();

  //This should only fire if we have more than maxConcurrentEvals evaluating, such that they wrap the
  //circular buffer.
  assert(!overlooped);

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
  resultLock.unlock();

  //Perform postprocessing on the result - turn the nn output into probabilities
  //As a hack though, if the only thing we were missing was the ownermap, just grab the old policy and values
  //and use those. This avoids recomputing in a randomly different orientation when we just need the ownermap
  //and causing policy weights to be different, which would reduce performance of successive searches in a game
  //by making the successive searches distribute their playouts less coherently and using the cache more poorly.
  if(hadResultWithoutOwnerMap) {
    buf.result->whiteWinProb = resultWithoutOwnerMap->whiteWinProb;
    buf.result->whiteLossProb = resultWithoutOwnerMap->whiteLossProb;
    buf.result->whiteNoResultProb = resultWithoutOwnerMap->whiteNoResultProb;
    buf.result->whiteScoreMean = resultWithoutOwnerMap->whiteScoreMean;
    buf.result->whiteScoreMeanSq = resultWithoutOwnerMap->whiteScoreMeanSq;
    std::copy(resultWithoutOwnerMap->policyProbs, resultWithoutOwnerMap->policyProbs + NNPos::MAX_NN_POLICY_SIZE, buf.result->policyProbs);
    buf.result->posLen = resultWithoutOwnerMap->posLen;
    assert(buf.result->whiteOwnerMap != NULL);
  }
  else {
    float* policy = buf.result->policyProbs;

    int xSize = board.x_size;
    int ySize = board.y_size;

    float maxPolicy = -1e25f;
    bool isLegal[NNPos::MAX_NN_POLICY_SIZE];
    int legalCount = 0;
    for(int i = 0; i<policySize; i++) {
      Loc loc = NNPos::posToLoc(i,xSize,ySize,posLen);
      isLegal[i] = history.isLegal(board,loc,nextPlayer);

      float policyValue;
      if(isLegal[i]) {
        legalCount += 1;
        policyValue = policy[i] * nnPolicyInvTemperature;
      }
      else
        policyValue = -1e30f;

      policy[i] = policyValue;
      if(policyValue > maxPolicy)
        maxPolicy = policyValue;
    }

    assert(legalCount > 0);

    float policySum = 0.0f;
    for(int i = 0; i<policySize; i++) {
      policy[i] = exp(policy[i] - maxPolicy);
      policySum += policy[i];
    }

    if(isnan(policySum)) {
      cout << "Got nan for policy sum" << endl;
      history.printDebugInfo(cout,board);
      throw StringError("Got nan for policy sum");
    }

    //Somehow all legal moves rounded to 0 probability
    if(policySum <= 0.0) {
      if(!buf.errorLogLockout && logger != NULL) {
        buf.errorLogLockout = true;
        logger->write("Warning: all legal moves rounded to 0 probability for " + string(modelFileName));
      }
      float uniform = 1.0f / legalCount;
      for(int i = 0; i<policySize; i++) {
        policy[i] = isLegal[i] ? uniform : -1.0f;
      }
    }
    //Normal case
    else {
      for(int i = 0; i<policySize; i++)
        policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;
    }

    //Fill everything out-of-bounds too, for robustness.
    for(int i = policySize; i<NNPos::MAX_NN_POLICY_SIZE; i++)
      policy[i] = -1.0f;

    //Fix up the value as well. Note that the neural net gives us back the value from the perspective
    //of the player so we need to negate that to make it the white value.
    //For model version 2 and less, we only have single value output that returns tanh, stuffed
    //ad-hocly into the whiteWinProb field.

    if(modelVersion <= 2) {
      double winProb = 0.5 * tanh(buf.result->whiteWinProb) + 0.5;
      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = winProb;
        buf.result->whiteLossProb = 1.0 - winProb;
        buf.result->whiteNoResultProb = 0.0;
        buf.result->whiteScoreMean = 0.0;
        buf.result->whiteScoreMeanSq = 0.0;
      }
      else {
        buf.result->whiteWinProb = 1.0 - winProb;
        buf.result->whiteLossProb = winProb;
        buf.result->whiteNoResultProb = 0.0;
        buf.result->whiteScoreMean = 0.0;
        buf.result->whiteScoreMeanSq = 0.0;
      }
    }
    else if(modelVersion == 3) {
      const double twoOverPi = 0.63661977236758134308;

      double winProb;
      double lossProb;
      double noResultProb;
      //Version 3 neural nets just pack the pre-arctanned scoreValue into the whiteScoreMean field
      double scoreValue = atan(buf.result->whiteScoreMean) * twoOverPi;
      {
        double winLogits = buf.result->whiteWinProb;
        double lossLogits = buf.result->whiteLossProb;
        double noResultLogits = buf.result->whiteNoResultProb;

        //Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        if(isnan(probSum) || isnan(scoreValue)) {
          cout << "Got nan for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits << " " << scoreValue << endl;
          throw StringError("Got nan for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = winProb;
        buf.result->whiteLossProb = lossProb;
        buf.result->whiteNoResultProb = noResultProb;
        buf.result->whiteScoreMean = ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board);
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
      }
      else {
        buf.result->whiteWinProb = lossProb;
        buf.result->whiteLossProb = winProb;
        buf.result->whiteNoResultProb = noResultProb;
        buf.result->whiteScoreMean = -ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board);
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
      }

    }
    else if(modelVersion == 4 || modelVersion == 5 || modelVersion == 6) {
      double winProb;
      double lossProb;
      double noResultProb;
      double scoreMean;
      double scoreMeanSq;
      {
        double winLogits = buf.result->whiteWinProb;
        double lossLogits = buf.result->whiteLossProb;
        double noResultLogits = buf.result->whiteNoResultProb;
        double scoreMeanPreScaled = buf.result->whiteScoreMean;
        double scoreStdevPreSoftplus = buf.result->whiteScoreMeanSq;

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultLogits -= 100000.0;

        //Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultProb = 0.0;

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        scoreMean = scoreMeanPreScaled * 20.0;

        double scoreStdev;
        //Avoid blowup
        if(scoreStdevPreSoftplus > 40.0)
          scoreStdev = scoreStdevPreSoftplus;
        else
          scoreStdev = log(1.0 + exp(scoreStdevPreSoftplus)) * 20.0;

        scoreMeanSq = scoreMean * scoreMean + scoreStdev * scoreStdev;

        //scoreMean and scoreMeanSq are still conditional on having a result, we need to make them unconditional now
        //noResult counts as 0 score for scorevalue purposes.
        scoreMean = scoreMean * (1.0-noResultProb);
        scoreMeanSq = scoreMeanSq * (1.0-noResultProb);

        if(isnan(probSum) || isnan(scoreMean) || isnan(scoreMeanSq)) {
          cout << "Got nan for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits << " " << scoreMean << " " << scoreMeanSq << endl;
          throw StringError("Got nan for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = winProb;
        buf.result->whiteLossProb = lossProb;
        buf.result->whiteNoResultProb = noResultProb;
        buf.result->whiteScoreMean = scoreMean;
        buf.result->whiteScoreMeanSq = scoreMeanSq;
      }
      else {
        buf.result->whiteWinProb = lossProb;
        buf.result->whiteLossProb = winProb;
        buf.result->whiteNoResultProb = noResultProb;
        buf.result->whiteScoreMean = -scoreMean;
        buf.result->whiteScoreMeanSq = scoreMeanSq;
      }

    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }

  //Postprocess ownermap
  if(buf.result->whiteOwnerMap != NULL) {
    if(modelVersion <= 2) {
      //No postprocessing needed, cudabackend fills with zeros, which is exactly fine.
    }
    else if(modelVersion == 3 || modelVersion == 4 || modelVersion == 5 || modelVersion == 6) {
      for(int pos = 0; pos<posLen*posLen; pos++) {
        int y = pos / posLen;
        int x = pos % posLen;
        if(y >= board.y_size || x >= board.x_size)
          buf.result->whiteOwnerMap[pos] = 0.0f;
        else {
          //Similarly as mentioned above, the result we get back from the net is actually not from white's perspective,
          //but from the player to move, so we need to flip it to make it white at the same time as we tanh it.
          if(nextPlayer == P_WHITE)
            buf.result->whiteOwnerMap[pos] = tanh(buf.result->whiteOwnerMap[pos]);
          else
            buf.result->whiteOwnerMap[pos] = -tanh(buf.result->whiteOwnerMap[pos]);
        }
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }


  //And record the nnHash in the result and put it into the table
  buf.result->nnHash = nnHash;
  if(nnCacheTable != NULL)
    nnCacheTable->set(buf.result);

}

//Uncomment this to lower the effective hash size down to one where we get true collisions
//#define SIMULATE_TRUE_HASH_COLLISIONS

NNCacheTable::Entry::Entry()
  :ptr(nullptr)
{}
NNCacheTable::Entry::~Entry()
{}

NNCacheTable::NNCacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  sizePowerOfTwo = sizePowerOfTwo > 12 ? 12 : sizePowerOfTwo;
#endif

  tableSize = ((uint64_t)1) << sizePowerOfTwo;
  tableMask = tableSize-1;
  entries = new Entry[tableSize];
  uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  mutexPoolMask = mutexPoolSize-1;
  mutexPool = new MutexPool(mutexPoolSize);
}
NNCacheTable::~NNCacheTable() {
  delete[] entries;
  delete mutexPool;
}

bool NNCacheTable::get(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  //Free ret BEFORE locking, to avoid any expensive operations while locked.
  if(ret != nullptr)
    ret.reset();

  uint64_t idx = nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);

  bool found = false;
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  if(entry.ptr != nullptr && ((entry.ptr->nnHash.hash0 ^ nnHash.hash0) & 0xFFF) == 0) {
    ret = entry.ptr;
    found = true;
  }
#else
  if(entry.ptr != nullptr && entry.ptr->nnHash == nnHash) {
    ret = entry.ptr;
    found = true;
  }
#endif
  return found;
}

void NNCacheTable::set(const shared_ptr<NNOutput>& p) {
  //Immediately copy p right now, before locking, to avoid any expensive operations while locked.
  shared_ptr<NNOutput> buf(p);

  uint64_t idx = p->nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    //Perform a swap, to avoid any expensive free under the mutex.
    entry.ptr.swap(buf);
  }

  //No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
}

void NNCacheTable::clear() {
  shared_ptr<NNOutput> buf;
  for(size_t idx = 0; idx<tableSize; idx++) {
    Entry& entry = entries[idx];
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
    std::mutex& mutex = mutexPool->getMutex(mutexIdx);
    {
      std::lock_guard<std::mutex> lock(mutex);
      entry.ptr.swap(buf);
    }
    buf.reset();
  }
}

