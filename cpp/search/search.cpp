
#include <inttypes.h>
#include <algorithm>
#include "../search/search.h"
#include "../core/fancymath.h"
#include "../core/timer.h"
#include "../search/distributiontable.h"

NodeStats::NodeStats()
  :visits(0),winValueSum(0.0),noResultValueSum(0.0),scoreMeanSum(0.0),scoreMeanSqSum(0.0),valueSumWeight(0.0)
{}
NodeStats::~NodeStats()
{}

NodeStats::NodeStats(const NodeStats& other)
  :visits(other.visits),
   winValueSum(other.winValueSum),
   noResultValueSum(other.noResultValueSum),
   scoreMeanSum(other.scoreMeanSum),
   scoreMeanSqSum(other.scoreMeanSqSum),
   valueSumWeight(other.valueSumWeight)
{}
NodeStats& NodeStats::operator=(const NodeStats& other) {
  visits = other.visits;
  winValueSum = other.winValueSum;
  noResultValueSum = other.noResultValueSum;
  scoreMeanSum = other.scoreMeanSum;
  scoreMeanSqSum = other.scoreMeanSqSum;
  valueSumWeight = other.valueSumWeight;
  return *this;
}

double NodeStats::getResultUtilitySum(const SearchParams& searchParams) const {
  return (
    (2.0*winValueSum - valueSumWeight + noResultValueSum) * searchParams.winLossUtilityFactor +
    noResultValueSum * searchParams.noResultUtilityForWhite
  );
}

static double getResultUtilityFromNN(const NNOutput& nnOutput, const SearchParams& searchParams) {
  return (
    (nnOutput.whiteWinProb - nnOutput.whiteLossProb) * searchParams.winLossUtilityFactor +
    nnOutput.whiteNoResultProb * searchParams.noResultUtilityForWhite
  );
}

static double getScoreStdev(double scoreMean, double scoreMeanSq) {
  double variance = scoreMeanSq - scoreMean * scoreMean;
  if(variance <= 0.0)
    return 0.0;
  return sqrt(variance);
}

//-----------------------------------------------------------------------------------------

SearchNode::SearchNode(Search& search, SearchThread& thread, Loc moveLoc)
  :lockIdx(),statsLock(ATOMIC_FLAG_INIT),nextPla(thread.pla),prevMoveLoc(moveLoc),
   nnOutput(),
   children(NULL),numChildren(0),childrenCapacity(0),
   stats(),virtualLosses(0)
{
  lockIdx = thread.rand.nextUInt(search.mutexPool->getNumMutexes());
}
SearchNode::~SearchNode() {
  if(children != NULL) {
    for(int i = 0; i<numChildren; i++)
      delete children[i];
  }
  delete[] children;
}

SearchNode::SearchNode(SearchNode&& other) noexcept
:lockIdx(other.lockIdx),statsLock(),
  nextPla(other.nextPla),prevMoveLoc(other.prevMoveLoc),
  nnOutput(std::move(other.nnOutput)),
  stats(other.stats),virtualLosses(other.virtualLosses)
{
  children = other.children;
  other.children = NULL;
  numChildren = other.numChildren;
  childrenCapacity = other.childrenCapacity;
}
SearchNode& SearchNode::operator=(SearchNode&& other) noexcept {
  lockIdx = other.lockIdx;
  nextPla = other.nextPla;
  prevMoveLoc = other.prevMoveLoc;
  nnOutput = std::move(other.nnOutput);
  children = other.children;
  other.children = NULL;
  numChildren = other.numChildren;
  childrenCapacity = other.childrenCapacity;
  stats = other.stats;
  virtualLosses = other.virtualLosses;
  return *this;
}

//-----------------------------------------------------------------------------------------

static string makeSeed(const Search& search, int threadIdx) {
  stringstream ss;
  ss << search.randSeed;
  ss << "$searchThread$";
  ss << threadIdx;
  ss << "$";
  ss << search.rootBoard.pos_hash;
  ss << "$";
  ss << search.rootHistory.moveHistory.size();
  ss << "$";
  ss << search.numSearchesBegun;
  return ss.str();
}

SearchThread::SearchThread(int tIdx, const Search& search, Logger* lg)
  :threadIdx(tIdx),
   pla(search.rootPla),board(search.rootBoard),
   history(search.rootHistory),
   rand(makeSeed(search,tIdx)),
   nnResultBuf(),
   logStream(NULL),
   logger(lg),
   valueChildWeightsBuf(),
   winValuesBuf(),
   noResultValuesBuf(),
   scoreMeansBuf(),
   scoreMeanSqsBuf(),
   utilityBuf(),
   visitsBuf()
{
  if(logger != NULL)
    logStream = logger->createOStream();

  winValuesBuf.resize(NNPos::MAX_NN_POLICY_SIZE);
  noResultValuesBuf.resize(NNPos::MAX_NN_POLICY_SIZE);
  scoreMeansBuf.resize(NNPos::MAX_NN_POLICY_SIZE);
  scoreMeanSqsBuf.resize(NNPos::MAX_NN_POLICY_SIZE);
  utilityBuf.resize(NNPos::MAX_NN_POLICY_SIZE);
  visitsBuf.resize(NNPos::MAX_NN_POLICY_SIZE);

}
SearchThread::~SearchThread() {
  if(logStream != NULL)
    delete logStream;
  logStream = NULL;
  logger = NULL;
}

//-----------------------------------------------------------------------------------------

static const double VALUE_WEIGHT_DEGREES_OF_FREEDOM = 3.0;

Search::Search(SearchParams params, NNEvaluator* nnEval, const string& rSeed)
  :rootPla(P_BLACK),rootBoard(),rootHistory(),rootPassLegal(true),
   rootSafeArea(NULL),
   recentScoreCenter(0.0),
   searchParams(params),numSearchesBegun(0),randSeed(rSeed),
   nnEvaluator(nnEval),
   nonSearchRand(rSeed + string("$nonSearchRand"))
{
  posLen = nnEval->getPosLen();
  assert(posLen > 0 && posLen <= NNPos::MAX_BOARD_LEN);
  policySize = NNPos::getPolicySize(posLen);
  rootKoHashTable = new KoHashTable();

  rootSafeArea = new Color[Board::MAX_ARR_SIZE];

  valueWeightDistribution = new DistributionTable(
    [](double z) { return FancyMath::tdistpdf(z,VALUE_WEIGHT_DEGREES_OF_FREEDOM); },
    [](double z) { return FancyMath::tdistcdf(z,VALUE_WEIGHT_DEGREES_OF_FREEDOM); },
    -50.0,
    50.0,
    2000
  );

  rootNode = NULL;
  mutexPool = new MutexPool(params.mutexPoolSize);

  rootHistory.clear(rootBoard,rootPla,Rules(),0);
  rootKoHashTable->recompute(rootHistory);
}

Search::~Search() {
  delete[] rootSafeArea;
  delete rootKoHashTable;
  delete valueWeightDistribution;
  delete rootNode;
  delete mutexPool;
}

const Board& Search::getRootBoard() const {
  return rootBoard;
}
const BoardHistory& Search::getRootHist() const {
  return rootHistory;
}
Player Search::getRootPla() const {
  return rootPla;
}

void Search::setPosition(Player pla, const Board& board, const BoardHistory& history) {
  clearSearch();
  rootPla = pla;
  rootBoard = board;
  rootHistory = history;
  rootKoHashTable->recompute(rootHistory);
}

void Search::setPlayerAndClearHistory(Player pla) {
  clearSearch();
  rootPla = pla;
  rootBoard.clearSimpleKoLoc();
  Rules rules = rootHistory.rules;
  rootHistory.clear(rootBoard,rootPla,rules,rootHistory.encorePhase);
  rootKoHashTable->recompute(rootHistory);
}

void Search::setRulesAndClearHistory(Rules rules, int encorePhase) {
  clearSearch();
  rootBoard.clearSimpleKoLoc();
  rootHistory.clear(rootBoard,rootPla,rules,encorePhase);
  rootKoHashTable->recompute(rootHistory);
}

void Search::setKomi(float newKomi) {
  if(rootHistory.rules.komi != newKomi) {
    clearSearch();
    rootHistory.setKomi(newKomi);
  }
}

void Search::setRootPassLegal(bool b) {
  clearSearch();
  rootPassLegal = b;
}

void Search::setParams(SearchParams params) {
  clearSearch();
  searchParams = params;
}

void Search::setNNEval(NNEvaluator* nnEval) {
  clearSearch();
  nnEvaluator = nnEval;
  posLen = nnEval->getPosLen();
  assert(posLen > 0 && posLen <= NNPos::MAX_BOARD_LEN);
  policySize = NNPos::getPolicySize(posLen);
}

void Search::clearSearch() {
  delete rootNode;
  rootNode = NULL;
}

bool Search::isLegal(Loc moveLoc, Player movePla) const {
  //If we somehow have the same player making multiple moves in a row (possible in GTP or an sgf file),
  //clear the ko loc - the simple ko loc of a player should not prohibit the opponent playing there!
  if(movePla != rootPla) {
    Board copy = rootBoard;
    copy.clearSimpleKoLoc();
    return copy.isLegal(moveLoc,movePla,rootHistory.rules.multiStoneSuicideLegal);
  }
  else {
    //Don't require that the move is legal for the history, merely the board, so that
    //we're robust to GTP or an sgf file saying that a move was made that violates superko or things like that.
    //In the encore, we also need to ignore the simple ko loc, since the board itself will report a move as illegal
    //when actually it is a legal pass-for-ko.
    if(rootHistory.encorePhase >= 1)
      return rootBoard.isLegalIgnoringKo(moveLoc,rootPla,rootHistory.rules.multiStoneSuicideLegal);
    else
      return rootBoard.isLegal(moveLoc,rootPla,rootHistory.rules.multiStoneSuicideLegal);
  }
}

bool Search::makeMove(Loc moveLoc, Player movePla) {
  if(!isLegal(moveLoc,movePla))
    return false;

  if(movePla != rootPla)
    setPlayerAndClearHistory(movePla);

  if(rootNode != NULL) {
    bool foundChild = false;
    for(int i = 0; i<rootNode->numChildren; i++) {
      SearchNode* child = rootNode->children[i];
      if(child->prevMoveLoc == moveLoc) {
        //Grab out the node to prevent its deletion along with the root
        SearchNode* node = new SearchNode(std::move(*child));
        //Delete the root and replace it with the child
        delete rootNode;
        rootNode = node;
        rootNode->prevMoveLoc = Board::NULL_LOC;
        foundChild = true;
        break;
      }
    }
    if(!foundChild) {
      clearSearch();
    }
  }
  rootHistory.makeBoardMoveAssumeLegal(rootBoard,moveLoc,rootPla,rootKoHashTable);
  rootPla = getOpp(rootPla);
  rootKoHashTable->recompute(rootHistory);
  return true;
}

static const double POLICY_ILLEGAL_SELECTION_VALUE = -1e50;

bool Search::getPlaySelectionValues(
  vector<Loc>& locs, vector<double>& playSelectionValues, int64_t& unreducedNumVisitsBuf, double scaleMaxToAtLeast
) const {
  if(rootNode == NULL) {
    locs.clear();
    playSelectionValues.clear();
    unreducedNumVisitsBuf = 0;
    return false;
  }
  return getPlaySelectionValues(*rootNode, locs, playSelectionValues, unreducedNumVisitsBuf, scaleMaxToAtLeast);
}

bool Search::getPlaySelectionValues(
  const SearchNode& node,
  vector<Loc>& locs, vector<double>& playSelectionValues, int64_t& unreducedNumVisitsBuf, double scaleMaxToAtLeast
) const {
  locs.clear();
  playSelectionValues.clear();
  unreducedNumVisitsBuf = 0;

  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex);

  int numChildren = node.numChildren;

  int64_t totalChildVisits = 0;
  for(int i = 0; i<numChildren; i++) {
    SearchNode* child = node.children[i];
    Loc moveLoc = child->prevMoveLoc;
    int movePos = getPos(moveLoc);
    float nnPolicyProb = node.nnOutput->policyProbs[movePos];
    
    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t childVisits = child->stats.visits;
    child->statsLock.clear(std::memory_order_release);
    
    double selectionValue = getPlaySelectionValue(nnPolicyProb,childVisits,node.nextPla);
    assert(selectionValue >= 0.0);    

    locs.push_back(moveLoc);
    playSelectionValues.push_back(selectionValue);
    totalChildVisits += childVisits;
  }
  //Go ahead and immediately record the raw number of visits, before doing any sort of reductions
  unreducedNumVisitsBuf = 1 + totalChildVisits;

  //Possibly reduce visits on children that we spend too many visits on in retrospect
  if(&node == rootNode && searchParams.rootDesiredPerChildVisitsCoeff > 0 && numChildren > 0) {
    double bestValue = -1e30;
    int bestIdx = 0;
    for(int i = 0; i<numChildren; i++) {
      double value = playSelectionValues[i];
      if(value > bestValue) {
        bestValue = value;
        bestIdx = i;
      }
    }

    const SearchNode* bestChild = node.children[bestIdx];
    double fpuValue = -10.0; //dummy, not actually used since these childs all should actually have visits
    bool isRootDuringSearch = false;
    double bestChildExploreSelectionValue = getExploreSelectionValue(node,bestChild,totalChildVisits,fpuValue,isRootDuringSearch);

    for(int i = 0; i<numChildren; i++) {
      if(i != bestIdx)
        playSelectionValues[i] = getReducedPlaySelectionValue(node, node.children[i], totalChildVisits, bestChildExploreSelectionValue);
    }
  }
  
  shared_ptr<NNOutput> nnOutput = node.nnOutput;
  lock.unlock();

  //If we have no children, then use the policy net directly. Only for the root, though, if calling this on any subtree
  //then just require that we have children, for implementation simplicity (since it requires that we have a board and a boardhistory too)
  //(and we also use isAllowedRootMove)
  if(numChildren == 0) {
    if(nnOutput == nullptr || &node != rootNode)
      return false;
    for(int movePos = 0; movePos<policySize; movePos++) {
      Loc moveLoc = NNPos::posToLoc(movePos,rootBoard.x_size,rootBoard.y_size,posLen);
      double policyProb = nnOutput->policyProbs[movePos];
      if(!rootHistory.isLegal(rootBoard,moveLoc,rootPla) || policyProb < 0 || !isAllowedRootMove(moveLoc))
        continue;
      locs.push_back(moveLoc);
      playSelectionValues.push_back(policyProb);
      numChildren++;
    }
  }

  //Might happen absurdly rarely if we both have no children and don't properly have an nnOutput
  //but have a hash collision or something so we "found" an nnOutput anyways.
  if(numChildren == 0)
    return false;

  double maxValue = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] > maxValue)
      maxValue = playSelectionValues[i];
  }

  if(maxValue <= 1e-50)
    return false;

  //Sanity check - if somehow we had more than this, something must have overflowed or gone wrong
  assert(maxValue < 1e16);

  double amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxValue/64.0);
  double amountToPrune = std::min(searchParams.chosenMovePrune, maxValue/64.0);
  double newMaxValue = maxValue - amountToSubtract;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] < amountToPrune)
      playSelectionValues[i] = 0.0;
    else {
      playSelectionValues[i] -= amountToSubtract;
      if(playSelectionValues[i] <= 0.0)
        playSelectionValues[i] = 0.0;
    }
  }

  assert(newMaxValue > 0.0);

  if(newMaxValue < scaleMaxToAtLeast) {
    for(int i = 0; i<numChildren; i++) {
      playSelectionValues[i] *= scaleMaxToAtLeast / newMaxValue;
    }
  }

  return true;
}

bool Search::getRootValues(
  double& winValue, double& lossValue, double& noResultValue, double& staticScoreValue, double& dynamicScoreValue, double& expectedScore
) const {
  assert(rootNode != NULL);
  return getNodeValues(*rootNode,winValue,lossValue,noResultValue,staticScoreValue,dynamicScoreValue,expectedScore);
}

bool Search::getNodeValues(
  const SearchNode& node,
  double& winValue, double& lossValue, double& noResultValue, double& staticScoreValue, double& dynamicScoreValue, double& expectedScore
) const {
  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex);
  shared_ptr<NNOutput> nnOutput = node.nnOutput;
  lock.unlock();
  if(nnOutput == nullptr)
    return false;

  while(node.statsLock.test_and_set(std::memory_order_acquire));
  double winValueSum = node.stats.winValueSum;
  double noResultValueSum = node.stats.noResultValueSum;
  double scoreMeanSum = node.stats.scoreMeanSum;
  double scoreMeanSqSum = node.stats.scoreMeanSqSum;
  double valueSumWeight = node.stats.valueSumWeight;
  node.statsLock.clear(std::memory_order_release);

  assert(valueSumWeight > 0.0);

  winValue = winValueSum / valueSumWeight;
  lossValue = (valueSumWeight - winValueSum - noResultValueSum) / valueSumWeight;
  noResultValue = noResultValueSum / valueSumWeight;
  double scoreMean = scoreMeanSum / valueSumWeight;
  double scoreMeanSq = scoreMeanSqSum / valueSumWeight;
  double scoreStdev = getScoreStdev(scoreMean,scoreMeanSq);
  staticScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0,rootBoard);
  dynamicScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,recentScoreCenter,1.0,rootBoard);
  expectedScore = scoreMean;

  //Perform a little normalization - due to tiny floating point errors, winValue and lossValue could be outside [0,1].
  //(particularly lossValue, as it was produced by subtractions from valueSumWeight that could have lost precision).
  if(winValue < 0.0) winValue = 0.0;
  if(lossValue < 0.0) lossValue = 0.0;
  if(noResultValue < 0.0) noResultValue = 0.0;
  double sum = winValue + lossValue + noResultValue;
  assert(sum > 0.9 && sum < 1.1); //If it's wrong by more than this, we have a bigger bug somewhere
  winValue /= sum;
  lossValue /= sum;
  noResultValue /= sum;
  
  return true;
}

double Search::getUtility(double resultUtilitySum, double scoreMeanSum, double scoreMeanSqSum, double valueSumWeight) const {
  double resultUtility = resultUtilitySum / valueSumWeight;
  double scoreMean = scoreMeanSum / valueSumWeight;
  double scoreMeanSq = scoreMeanSqSum / valueSumWeight;
  double scoreStdev = getScoreStdev(scoreMean, scoreMeanSq);
  double staticScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0,rootBoard);
  double dynamicScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,recentScoreCenter,1.0,rootBoard);
  return resultUtility + staticScoreValue * searchParams.staticScoreUtilityFactor + dynamicScoreValue * searchParams.dynamicScoreUtilityFactor;
}

double Search::getUtilityFromNN(const NNOutput& nnOutput) const {
  double resultUtility = getResultUtilityFromNN(nnOutput, searchParams);
  return getUtility(resultUtility, nnOutput.whiteScoreMean, nnOutput.whiteScoreMeanSq, 1.0);
}

double Search::getRootUtility() const {
  assert(rootNode != NULL);
  const SearchNode& node = *rootNode;
  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex);
  shared_ptr<NNOutput> nnOutput = node.nnOutput;
  lock.unlock();
  if(nnOutput == nullptr)
    return false;

  while(node.statsLock.test_and_set(std::memory_order_acquire));
  double resultUtilitySum = node.stats.getResultUtilitySum(searchParams);
  double scoreMeanSum = node.stats.scoreMeanSum;
  double scoreMeanSqSum = node.stats.scoreMeanSqSum;
  double valueSumWeight = node.stats.valueSumWeight;
  node.statsLock.clear(std::memory_order_release);

  assert(valueSumWeight > 0.0);
  return getUtility(resultUtilitySum, scoreMeanSum, scoreMeanSqSum, valueSumWeight);
}

uint32_t Search::chooseIndexWithTemperature(Rand& rand, const double* relativeProbs, int numRelativeProbs, double temperature) {
  assert(numRelativeProbs > 0);
  assert(numRelativeProbs < 1024); //We're just doing this on the stack
  double processedRelProbs[numRelativeProbs];

  double maxValue = 0.0;
  for(int i = 0; i<numRelativeProbs; i++) {
    if(relativeProbs[i] > maxValue)
      maxValue = relativeProbs[i];
  }
  assert(maxValue > 0.0);
  
  //Temperature so close to 0 that we just calculate the max directly
  if(temperature <= 1.0e-4) {
    double bestProb = relativeProbs[0];
    int bestIdx = 0;
    for(int i = 1; i<numRelativeProbs; i++) {
      if(relativeProbs[i] > bestProb) {
        bestProb = relativeProbs[i];
        bestIdx = i;
      }
    }
    return bestIdx;
  }
  //Actual temperature
  else {
    double logMaxValue = log(maxValue);
    double sum = 0.0;
    for(int i = 0; i<numRelativeProbs; i++) {
      //Numerically stable way to raise to power and normalize
      processedRelProbs[i] = exp((log(relativeProbs[i]) - logMaxValue) / temperature);
      sum += processedRelProbs[i];
    }
    assert(sum > 0.0);
    uint32_t idxChosen = rand.nextUInt(processedRelProbs,numRelativeProbs);
    return idxChosen;
  }
}

Loc Search::getChosenMoveLoc() {
  if(rootNode == NULL)
    return Board::NULL_LOC;

  vector<Loc> locs;
  vector<double> playSelectionValues;
  int64_t unreducedNumVisitsBuf;
  bool suc = getPlaySelectionValues(locs,playSelectionValues,unreducedNumVisitsBuf,0.0);
  if(!suc)
    return Board::NULL_LOC;

  assert(locs.size() == playSelectionValues.size());

  double rawHalflives = rootHistory.moveHistory.size() / searchParams.chosenMoveTemperatureHalflife;
  double halflives = rawHalflives * 19.0 / sqrt(rootBoard.x_size*rootBoard.y_size);
  double temperature = searchParams.chosenMoveTemperature +
    (searchParams.chosenMoveTemperatureEarly - searchParams.chosenMoveTemperature) *
    pow(0.5, halflives);

  uint32_t idxChosen = chooseIndexWithTemperature(nonSearchRand, playSelectionValues.data(), playSelectionValues.size(), temperature);
  return locs[idxChosen];
}

Loc Search::runWholeSearchAndGetMove(Player movePla, Logger& logger, vector<double>* recordUtilities) {
  runWholeSearch(movePla,logger,recordUtilities);
  return getChosenMoveLoc();
}

void Search::runWholeSearch(Player movePla, Logger& logger, vector<double>* recordUtilities) {
  if(movePla != rootPla)
    setPlayerAndClearHistory(movePla);
  std::atomic<bool> shouldStopNow(false);
  runWholeSearch(logger,shouldStopNow,recordUtilities);
}

void Search::runWholeSearch(Logger& logger, std::atomic<bool>& shouldStopNow, vector<double>* recordUtilities) {

  ClockTimer timer;
  atomic<int64_t> numPlayoutsShared(0);

  if(!std::atomic_is_lock_free(&numPlayoutsShared))
    logger.write("Warning: int64_t atomic numPlayoutsShared is not lock free");
  if(!std::atomic_is_lock_free(&shouldStopNow))
    logger.write("Warning: bool atomic shouldStopNow is not lock free");

  beginSearch(logger);
  int64_t numNonPlayoutVisits = numRootVisits();

  auto searchLoop = [this,&timer,&numPlayoutsShared,numNonPlayoutVisits,&logger,&shouldStopNow,&recordUtilities](int threadIdx) {
    SearchThread* stbuf = new SearchThread(threadIdx,*this,&logger);

    int64_t maxVisits = searchParams.maxVisits;
    int64_t maxPlayouts = searchParams.maxPlayouts;
    double_t maxTime = searchParams.maxTime;

    //Possibly reduce computation time, for human friendliness
    {
      double searchFactor = 1.0;
      if(rootHistory.moveHistory.size() >= 1 && rootHistory.moveHistory[rootHistory.moveHistory.size()-1].loc == Board::PASS_LOC) {
        if(rootHistory.moveHistory.size() >= 3 && rootHistory.moveHistory[rootHistory.moveHistory.size()-3].loc == Board::PASS_LOC)
          searchFactor = searchParams.searchFactorAfterTwoPass;
        else
          searchFactor = searchParams.searchFactorAfterOnePass;
      }
      if(searchFactor != 1.0) {
        maxVisits = (int64_t)ceil(maxVisits * searchFactor);
        maxPlayouts = (int64_t)ceil(maxPlayouts * searchFactor);
        maxTime = maxTime * searchFactor;
      }
    }
    
    int64_t numPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
    try {
      while(true) {
        bool shouldStop =
          (numPlayouts >= 1 && maxTime < 1.0e12 && timer.getSeconds() >= maxTime) ||
          (numPlayouts >= maxPlayouts) ||
          (numPlayouts + numNonPlayoutVisits >= maxVisits);

        if(shouldStop || shouldStopNow.load(std::memory_order_relaxed)) {
          shouldStopNow.store(true,std::memory_order_relaxed);
          break;
        }

        runSinglePlayout(*stbuf);

        numPlayouts = numPlayoutsShared.fetch_add((int64_t)1, std::memory_order_relaxed);
        numPlayouts += 1;

        if(searchParams.numThreads == 1 && recordUtilities != NULL) {
          if(numPlayouts <= recordUtilities->size()) {
            assert(numPlayouts >= 1);
            (*recordUtilities)[numPlayouts-1] = getRootUtility();
          }
        }

      }
    }
    catch(const exception& e) {
      logger.write(string("ERROR: Search thread failed: ") + e.what());
      delete stbuf;
      throw;
    }
    catch(const string& e) {
      logger.write("ERROR: Search thread failed: " + e);
      delete stbuf;
      throw;
    }
    catch(...) {
      logger.write("ERROR: Search thread failed with unexpected throw");
      delete stbuf;
      throw;
    }

    delete stbuf;
  };

  if(searchParams.numThreads <= 1)
    searchLoop(0);
  else {
    std::thread* threads = new std::thread[searchParams.numThreads-1];
    for(int i = 0; i<searchParams.numThreads-1; i++)
      threads[i] = std::thread(searchLoop,i+1);
    searchLoop(0);
    for(int i = 0; i<searchParams.numThreads-1; i++)
      threads[i].join();
    delete[] threads;
  }
}


void Search::beginSearch(Logger& logger) {
  if(rootBoard.x_size > posLen || rootBoard.y_size > posLen)
    throw StringError("Search got from NNEval posLen = " + Global::intToString(posLen) + " but was asked to search board with larger x or y size");
  rootBoard.checkConsistency();

  numSearchesBegun++;
  computeRootValues(logger);

  //Sanity-check a few things
  if(!rootPassLegal && searchParams.rootPruneUselessSuicides)
    throw StringError("Both rootPassLegal=false and searchParams.rootPruneUselessSuicides=true are specified, this could leave the bot without legal moves!");

  if(rootNode == NULL) {
    SearchThread dummyThread(-1, *this, NULL);
    rootNode = new SearchNode(*this, dummyThread, Board::NULL_LOC);
  }
  else {
    //If the root node has any existing children, then prune things down if there are moves that should not be allowed at the root.
    SearchNode& node = *rootNode;
    int numChildren = node.numChildren;
    if(node.children != NULL && numChildren > 0) {
      assert(node.nnOutput != NULL);

      //Perform the filtering
      int numGoodChildren = 0;
      for(int i = 0; i<numChildren; i++) {
        SearchNode* child = node.children[i];
        node.children[i] = NULL;
        if(isAllowedRootMove(child->prevMoveLoc))
          node.children[numGoodChildren++] = child;
        else {
          delete child;
        }
      }
      bool anyFiltered = numChildren != numGoodChildren;
      node.numChildren = numGoodChildren;
      numChildren = numGoodChildren;

      if(anyFiltered) {
        //Fix up the number of visits of the root node after doing this filtering
        int64_t newNumVisits = 0;
        for(int i = 0; i<numChildren; i++) {
          const SearchNode* child = node.children[i];
          while(child->statsLock.test_and_set(std::memory_order_acquire));
          int64_t childVisits = child->stats.visits;
          child->statsLock.clear(std::memory_order_release);
          newNumVisits += childVisits;
        }
        //For the node's own visit itself
        newNumVisits += 1;

        //Set the visits in place
        while(node.statsLock.test_and_set(std::memory_order_acquire));
        node.stats.visits = newNumVisits;
        node.statsLock.clear(std::memory_order_release);

        //Update all other stats
        SearchThread dummyThread(-1, *this, NULL);
        recomputeNodeStats(node, dummyThread, 0, 0, true);
      }
    }
  }
}

void Search::computeRootValues(Logger& logger) {
  //rootSafeArea is strictly pass-alive groups and strictly safe territory.
  bool nonPassAliveStones = false;
  bool safeBigTerritories = false;
  bool unsafeBigTerritories = false;
  bool isMultiStoneSuicideLegal = rootHistory.rules.multiStoneSuicideLegal;
  rootBoard.calculateArea(
    rootSafeArea,
    nonPassAliveStones,
    safeBigTerritories,
    unsafeBigTerritories,
    isMultiStoneSuicideLegal
  );

  //Grab a neural net evaluation for the current position and use that as the center
  Board board = rootBoard;
  const BoardHistory& hist = rootHistory;
  NNResultBuf nnResultBuf;
  bool skipCache = false;
  bool includeOwnerMap = true;
  nnEvaluator->evaluate(
    board, hist, rootPla,
    searchParams.drawEquivalentWinsForWhite,
    nnResultBuf, &logger, skipCache, includeOwnerMap
  );
  double expectedScore = nnResultBuf.result->whiteScoreMean;
  recentScoreCenter = expectedScore;
}

int64_t Search::numRootVisits() {
  if(rootNode == NULL)
    return 0;
  while(rootNode->statsLock.test_and_set(std::memory_order_acquire));
  int64_t n = rootNode->stats.visits;
  rootNode->statsLock.clear(std::memory_order_release);
  return n;
}

//Assumes node is locked
void Search::maybeAddPolicyNoise(SearchThread& thread, SearchNode& node, bool isRoot) const {
  if(!isRoot)
    return;
  if(!searchParams.rootNoiseEnabled && searchParams.rootPolicyTemperature == 1.0)
    return;

  //Copy nnOutput as we're about to modify its policy to add noise or temperature
  shared_ptr<NNOutput> newNNOutput = std::make_shared<NNOutput>(*(node.nnOutput));
  //Replace the old pointer
  node.nnOutput = newNNOutput;

  if(searchParams.rootPolicyTemperature != 1.0) {
    double maxValue = 0.0;
    for(int i = 0; i<policySize; i++) {
      double prob = node.nnOutput->policyProbs[i];
      if(prob > maxValue)
        maxValue = prob;
    }
    assert(maxValue > 0.0);

    double logMaxValue = log(maxValue);
    double invTemp = 1.0 / searchParams.rootPolicyTemperature;
    double sum = 0.0;

    for(int i = 0; i<policySize; i++) {
      if(node.nnOutput->policyProbs[i] > 0) {
        //Numerically stable way to raise to power and normalize
        double p = exp((log((double)node.nnOutput->policyProbs[i]) - logMaxValue) * invTemp);
        node.nnOutput->policyProbs[i] = p;
        sum += p;
      }
    }
    assert(sum > 0.0);
    for(int i = 0; i<policySize; i++) {
      if(node.nnOutput->policyProbs[i] >= 0) {
        node.nnOutput->policyProbs[i] = (double)node.nnOutput->policyProbs[i] / sum;
      }
    }
  }

  if(searchParams.rootNoiseEnabled) {
    int legalCount = 0;
    for(int i = 0; i<policySize; i++) {
      if(node.nnOutput->policyProbs[i] >= 0)
        legalCount += 1;
    }

    if(legalCount <= 0)
      throw StringError("maybeAddPolicyNoise: No move with nonnegative policy value - can't even pass?");

    //Generate gamma draw on each move
    double alpha = searchParams.rootDirichletNoiseTotalConcentration / legalCount;
    double rSum = 0.0;
    double r[policySize];
    for(int i = 0; i<policySize; i++) {
      if(node.nnOutput->policyProbs[i] >= 0) {
        r[i] = thread.rand.nextGamma(alpha);
        rSum += r[i];
      }
      else
        r[i] = 0.0;
    }

    //Normalized gamma draws -> dirichlet noise
    for(int i = 0; i<policySize; i++)
      r[i] /= rSum;

    //At this point, r[i] contains a dirichlet distribution draw, so add it into the nnOutput.
    for(int i = 0; i<policySize; i++) {
      if(node.nnOutput->policyProbs[i] >= 0) {
        double weight = searchParams.rootDirichletNoiseWeight;
        node.nnOutput->policyProbs[i] = r[i] * weight + node.nnOutput->policyProbs[i] * (1.0-weight);
      }
    }
  }

}

bool Search::isAllowedRootMove(Loc moveLoc) const {
  assert(moveLoc == Board::PASS_LOC || rootBoard.isOnBoard(moveLoc));

  //For use on some online go servers, we want to be able to support a cleanup mode, where we force
  //the capture of stones that our training ruleset would consider simply dead by virtue of them
  //being pass-dead, so we add an option to forbid passing at the root.
  if(!rootPassLegal && moveLoc == Board::PASS_LOC)
    return false;
  //A bad situation that can happen that unnecessarily prolongs training games is where one player
  //repeatedly passes and the other side repeatedly fills the opponent's space and/or suicides over and over.
  //To mitigate some of this and save computation, we make it so that at the root, if the last four moves by the opponent
  //were passes, we will never play a move in either player's pass-alive area. In theory this could prune
  //a good move in situations like https://senseis.xmp.net/?1EyeFlaw, but this should be extraordinarly rare,
  if(searchParams.rootPruneUselessSuicides &&
     rootHistory.moveHistory.size() > 0 &&
     moveLoc != Board::PASS_LOC
  ) {
    int lastIdx = rootHistory.moveHistory.size()-1;
    Player opp = getOpp(rootPla);
    if(lastIdx >= 6 &&
       rootHistory.moveHistory[lastIdx-0].loc == Board::PASS_LOC &&
       rootHistory.moveHistory[lastIdx-2].loc == Board::PASS_LOC &&
       rootHistory.moveHistory[lastIdx-4].loc == Board::PASS_LOC &&
       rootHistory.moveHistory[lastIdx-6].loc == Board::PASS_LOC &&
       rootHistory.moveHistory[lastIdx-0].pla == opp &&
       rootHistory.moveHistory[lastIdx-2].pla == opp &&
       rootHistory.moveHistory[lastIdx-4].pla == opp &&
       rootHistory.moveHistory[lastIdx-6].pla == opp &&
       (rootSafeArea[moveLoc] == opp || rootSafeArea[moveLoc] == rootPla))
      return false;
  }
  return true;
}

void Search::getValueChildWeights(
  int numChildren,
  //Unlike everywhere else where values are from white's perspective, values here are from one's own perspective
  const vector<double>& childSelfValuesBuf,
  const vector<int64_t>& childVisitsBuf,
  vector<double>& resultBuf
) const {
  resultBuf.clear();
  if(numChildren <= 0)
    return;
  if(numChildren == 1) {
    resultBuf.push_back(1.0);
    return;
  }

  double stdevs[numChildren];
  for(int i = 0; i<numChildren; i++) {
    int64_t numVisits = childVisitsBuf[i];
    assert(numVisits > 0);
    double precision = 1.5 * sqrt((double)numVisits);

    //Ensure some minimum variance for stability regardless of how we change the above formula
    static const double minVariance = 0.00000001;
    stdevs[i] = sqrt(minVariance + 1.0 / precision);
  }

  double simpleValueSum = 0.0;
  int64_t numChildVisits = 0;
  for(int i = 0; i<numChildren; i++) {
    simpleValueSum += childSelfValuesBuf[i] * childVisitsBuf[i];
    numChildVisits += childVisitsBuf[i];
  }

  double simpleValue = simpleValueSum / numChildVisits;

  double weight[numChildren];
  for(int i = 0; i<numChildren; i++) {
    double z = (childSelfValuesBuf[i] - simpleValue) / stdevs[i];
    //Also just for numeric sanity, make sure everything has some tiny minimum value.
    weight[i] = valueWeightDistribution->getCdf(z) + 0.0001;
  }

  //Post-process and normalize, to make sure we exactly have a probability distribution and sum exactly to 1.
  double totalWeight = 0.0;
  for(int i = 0; i<numChildren; i++) {
    double p = weight[i];
    totalWeight += p;
    resultBuf.push_back(p);
  }

  assert(totalWeight > 0);
  for(int i = 0; i<numChildren; i++) {
    resultBuf[i] /= totalWeight;
  }

}

double Search::getPlaySelectionValue(
  double nnPolicyProb, int64_t childVisits, Player pla
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  (void)(pla);
  return (double)childVisits;
}

double Search::getExploreSelectionValue(
  double nnPolicyProb, int64_t totalChildVisits, int64_t childVisits,
  double childUtility, Player pla
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  double exploreComponent =
    searchParams.cpuctExploration
    * nnPolicyProb
    * sqrt((double)totalChildVisits + 0.01) //TODO this is weird when totalChildVisits == 0, first exploration
    / (1.0 + childVisits);

  //At the last moment, adjust value to be from the player's perspective, so that players prefer values in their favor
  //rather than in white's favor
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;
  return exploreComponent + valueComponent;
}

double Search::getEndingWhiteScoreBonus(const SearchNode& parent, const SearchNode* child) const {
  if(&parent != rootNode || child->prevMoveLoc == Board::NULL_LOC)
    return 0.0;
  if(parent.nnOutput == nullptr || parent.nnOutput->whiteOwnerMap == NULL)
    return 0.0;

  bool isAreaIsh = rootHistory.rules.scoringRule == Rules::SCORING_AREA
    || (rootHistory.rules.scoringRule == Rules::SCORING_TERRITORY && rootHistory.encorePhase >= 2);
  assert(parent.nnOutput->posLen == posLen);
  float* whiteOwnerMap = parent.nnOutput->whiteOwnerMap;
  Loc moveLoc = child->prevMoveLoc;

  //Extra points from the perspective of the root player
  double extraRootPoints = 0.0;
  if(isAreaIsh) {
    //Areaish scoring - in an effort to keep the game short and slightly discourage pointless territory filling at the end
    //discourage any move that, except in case of ko, is either:
    // * On a spot that the opponent almost surely owns
    // * On a spot that the player almost surely owns and it is not adjacent to opponent stones and is not a connection of non-pass-alive groups.
    //These conditions should still make it so that "cleanup" and dame-filling moves are not discouraged.
    if(moveLoc != Board::PASS_LOC && rootBoard.ko_loc == Board::NULL_LOC) {
      int pos = NNPos::locToPos(moveLoc,rootBoard.x_size,posLen);
      double plaOwnership = rootPla == P_WHITE ? whiteOwnerMap[pos] : -whiteOwnerMap[pos];
      if(plaOwnership <= -0.95)
        extraRootPoints -= searchParams.rootEndingBonusPoints * ((-0.95 - plaOwnership) / 0.05);
      else if(plaOwnership >= 0.95) {
        if(!rootBoard.isAdjacentToPla(moveLoc,getOpp(rootPla)) &&
           !rootBoard.isNonPassAliveSelfConnection(moveLoc,rootPla,rootSafeArea)) {
          extraRootPoints -= searchParams.rootEndingBonusPoints * ((plaOwnership - 0.95) / 0.05);
        }
      }
    }
  }
  else {
    //Territorish scoring - slightly encourage dame-filling by discouraging passing, so that the player will try to do everything
    //non-point-losing first, like filling dame.
    //Human japanese rules often "want" you to fill the dame so this is a cosmetic adjustment to encourage the neural
    //net to learn to do so in the main phase rather than waiting until the encore.
    //But cosmetically, it's also not great if we just encourage useless threat moves in the opponent's territory to prolong the game.
    //So also discourage those moves except in cases of ko. Also similar to area scoring just to be symmetrical, discourage moves on spots
    //that the player almost surely owns that are not adjacent to opponent stones and are not a connection of non-pass-alive groups.
    if(moveLoc == Board::PASS_LOC)
      extraRootPoints -= searchParams.rootEndingBonusPoints * (2.0/3.0);
    else if(rootBoard.ko_loc == Board::NULL_LOC) {
      int pos = NNPos::locToPos(moveLoc,rootBoard.x_size,posLen);
      double plaOwnership = rootPla == P_WHITE ? whiteOwnerMap[pos] : -whiteOwnerMap[pos];
      if(plaOwnership <= -0.95)
        extraRootPoints -= searchParams.rootEndingBonusPoints * ((-0.95 - plaOwnership) / 0.05);
      else if(plaOwnership >= 0.95) {
        if(!rootBoard.isAdjacentToPla(moveLoc,getOpp(rootPla)) &&
           !rootBoard.isNonPassAliveSelfConnection(moveLoc,rootPla,rootSafeArea)) {
          extraRootPoints -= searchParams.rootEndingBonusPoints * ((plaOwnership - 0.95) / 0.05);
        }
      }
    }
  }

  if(rootPla == P_WHITE)
    return extraRootPoints;
  else
    return -extraRootPoints;
}

int Search::getPos(Loc moveLoc) const {
  return NNPos::locToPos(moveLoc,rootBoard.x_size,posLen);
}

double Search::getPlaySelectionValue(const SearchNode& parent, const SearchNode* child) const {
  Loc moveLoc = child->prevMoveLoc;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parent.nnOutput->policyProbs[movePos];

  while(child->statsLock.test_and_set(std::memory_order_acquire));
  int64_t childVisits = child->stats.visits;
  child->statsLock.clear(std::memory_order_release);

  return getPlaySelectionValue(nnPolicyProb,childVisits,parent.nextPla);
}
double Search::getExploreSelectionValue(const SearchNode& parent, const SearchNode* child, int64_t totalChildVisits, double fpuValue, bool isRootDuringSearch) const {
  Loc moveLoc = child->prevMoveLoc;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parent.nnOutput->policyProbs[movePos];

  while(child->statsLock.test_and_set(std::memory_order_acquire));
  int64_t childVisits = child->stats.visits;
  double childResultUtilitySum = child->stats.getResultUtilitySum(searchParams);
  double scoreMeanSum = child->stats.scoreMeanSum;
  double scoreMeanSqSum = child->stats.scoreMeanSqSum;
  double valueSumWeight = child->stats.valueSumWeight;
  int32_t childVirtualLosses = child->virtualLosses;
  child->statsLock.clear(std::memory_order_release);

  //It's possible that childVisits is actually 0 here with multithreading because we're visiting this node while a child has
  //been expanded but its thread not yet finished its first visit
  double childUtility;
  if(childVisits <= 0)
    childUtility = fpuValue;
  else {
    assert(valueSumWeight > 0.0);
    //Tiny adjustment for passing
    double endingScoreBonus = getEndingWhiteScoreBonus(parent,child);
    double oldScoreMeanSum = scoreMeanSum;
    scoreMeanSum += endingScoreBonus * valueSumWeight;
    scoreMeanSqSum = scoreMeanSqSum + (oldScoreMeanSum + scoreMeanSum) * endingScoreBonus;
    childUtility = getUtility(childResultUtilitySum, scoreMeanSum, scoreMeanSqSum, valueSumWeight);
  }

  //When multithreading, totalChildVisits could be out of sync with childVisits, so if they provably are, then fix that up
  if(totalChildVisits < childVisits)
    totalChildVisits = childVisits;

  //Virtual losses to direct threads down different paths
  if(childVirtualLosses > 0) {
    //totalChildVisits += childVirtualLosses; //Should get better thread dispersal without this
    childVisits += childVirtualLosses;
    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;
    double virtualLossUtility = (parent.nextPla == P_WHITE ? -utilityRadius : utilityRadius);
    double virtualLossVisitFrac = (double)childVirtualLosses / childVisits;
    childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossVisitFrac;
  }

  //Hack to get the root to funnel more visits down child branches
  if(isRootDuringSearch && searchParams.rootDesiredPerChildVisitsCoeff > 0.0) {
    if(childVisits < sqrt(nnPolicyProb * totalChildVisits * searchParams.rootDesiredPerChildVisitsCoeff)) {
      return 1e20;
    }
  }
  
  return getExploreSelectionValue(nnPolicyProb,totalChildVisits,childVisits,childUtility,parent.nextPla);
}
double Search::getNewExploreSelectionValue(const SearchNode& parent, int movePos, int64_t totalChildVisits, double fpuValue) const {
  float nnPolicyProb = parent.nnOutput->policyProbs[movePos];
  int64_t childVisits = 0;
  double childUtility = fpuValue;
  return getExploreSelectionValue(nnPolicyProb,totalChildVisits,childVisits,childUtility,parent.nextPla);
}

double Search::getReducedPlaySelectionValue(const SearchNode& parent, const SearchNode* child, int64_t totalChildVisits, double bestChildExploreSelectionValue) const {
  assert(&parent == rootNode);
  Loc moveLoc = child->prevMoveLoc;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parent.nnOutput->policyProbs[movePos];

  while(child->statsLock.test_and_set(std::memory_order_acquire));
  int64_t childVisits = child->stats.visits;
  double childResultUtilitySum = child->stats.getResultUtilitySum(searchParams);
  double scoreMeanSum = child->stats.scoreMeanSum;
  double scoreMeanSqSum = child->stats.scoreMeanSqSum;
  double valueSumWeight = child->stats.valueSumWeight;
  child->statsLock.clear(std::memory_order_release);

  //getReducedPlaySelectionValue only happens after the search, so there should be no multithreading shenanigans that give us a 0-visit child.
  assert(childVisits > 0);
  assert(valueSumWeight > 0.0);

  //Tiny adjustment for passing
  double endingScoreBonus = getEndingWhiteScoreBonus(parent,child);
  double oldScoreMeanSum = scoreMeanSum;
  scoreMeanSum += endingScoreBonus * valueSumWeight;
  scoreMeanSqSum = scoreMeanSqSum + (oldScoreMeanSum + scoreMeanSum) * endingScoreBonus;
  double childUtility = getUtility(childResultUtilitySum, scoreMeanSum, scoreMeanSqSum, valueSumWeight);
  
  int64_t desiredVisits = (int64_t)ceil(sqrt(nnPolicyProb * totalChildVisits * searchParams.rootDesiredPerChildVisitsCoeff));
  for(int i = 0; i<desiredVisits; i++) {
    if(childVisits <= 0)
      break;
    double exploreSelectionValue = getExploreSelectionValue(nnPolicyProb,totalChildVisits,childVisits-1,childUtility,parent.nextPla);
    if(exploreSelectionValue < bestChildExploreSelectionValue) {
      childVisits -= 1;
      continue;
    }
    else
      break;
  }

  return getPlaySelectionValue(nnPolicyProb,childVisits,parent.nextPla);
}


//Assumes node is locked
void Search::selectBestChildToDescend(
  const SearchThread& thread, const SearchNode& node, int& bestChildIdx, Loc& bestChildMoveLoc,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot) const
{
  assert(thread.pla == node.nextPla);

  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;

  int numChildren = node.numChildren;

  double policyProbMassVisited = 0.0;
  int64_t totalChildVisits = 0;
  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = node.children[i];
    Loc moveLoc = child->prevMoveLoc;
    int movePos = getPos(moveLoc);
    float nnPolicyProb = node.nnOutput->policyProbs[movePos];
    policyProbMassVisited += nnPolicyProb;

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t childVisits = child->stats.visits;
    child->statsLock.clear(std::memory_order_release);

    totalChildVisits += childVisits;
  }
  //Probability mass should not sum to more than 1, giving a generous allowance
  //for floating point error.
  assert(policyProbMassVisited <= 1.0001);

  //First play urgency
  double parentUtility;
  if(searchParams.fpuUseParentAverage) {
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    int64_t parentVisits = node.stats.visits;
    double resultUtilitySum = node.stats.getResultUtilitySum(searchParams);
    double scoreMeanSum = node.stats.scoreMeanSum;
    double scoreMeanSqSum = node.stats.scoreMeanSqSum;
    double valueSumWeight = node.stats.valueSumWeight;
    node.statsLock.clear(std::memory_order_release);

    assert(parentVisits > 0);
    assert(valueSumWeight > 0.0);
    parentUtility = getUtility(resultUtilitySum, scoreMeanSum, scoreMeanSqSum, valueSumWeight);
  }
  else {
    parentUtility = getUtilityFromNN(*node.nnOutput);
  }

  double fpuValue;
  {
    double fpuReductionMax = isRoot ? searchParams.rootFpuReductionMax : searchParams.fpuReductionMax;
    double fpuLossProp = isRoot ? searchParams.rootFpuLossProp : searchParams.fpuLossProp;
    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;

    double reduction = fpuReductionMax * sqrt(policyProbMassVisited);
    fpuValue = thread.pla == P_WHITE ? parentUtility - reduction : parentUtility + reduction;
    double lossValue = thread.pla == P_WHITE ? -utilityRadius : utilityRadius;
    fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
  }

  std::fill(posesWithChildBuf,posesWithChildBuf+NNPos::MAX_NN_POLICY_SIZE,false);

  //Try all existing children
  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = node.children[i];
    Loc moveLoc = child->prevMoveLoc;
    bool isRootDuringSearch = isRoot;
    double selectionValue = getExploreSelectionValue(node,child,totalChildVisits,fpuValue,isRootDuringSearch);
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = i;
      bestChildMoveLoc = moveLoc;
    }

    posesWithChildBuf[getPos(moveLoc)] = true;
  }

  //Try all new children
  for(int movePos = 0; movePos<policySize; movePos++) {
    bool alreadyTried = posesWithChildBuf[movePos];
    if(alreadyTried)
      continue;

    Loc moveLoc = NNPos::posToLoc(movePos,thread.board.x_size,thread.board.y_size,posLen);
    if(moveLoc == Board::NULL_LOC)
      continue;

    //Special logic for the root
    if(isRoot) {
      assert(thread.board.pos_hash == rootBoard.pos_hash);
      assert(thread.pla == rootPla);
      if(!isAllowedRootMove(moveLoc))
        continue;
    }

    double selectionValue = getNewExploreSelectionValue(node,movePos,totalChildVisits,fpuValue);
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = numChildren;
      bestChildMoveLoc = moveLoc;
    }
  }

}
void Search::updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, int32_t virtualLossesToSubtract, bool isRoot) {
  recomputeNodeStats(node,thread,1,virtualLossesToSubtract,isRoot);
}

//Recompute all the stats of this node based on its children, except its visits and virtual losses, which are not child-dependent and
//are updated in the manner specified.
void Search::recomputeNodeStats(SearchNode& node, SearchThread& thread, int numVisitsToAdd, int32_t virtualLossesToSubtract, bool isRoot) {
  //Find all children and compute weighting of the children based on their values
  vector<double>& valueChildWeights = thread.valueChildWeightsBuf;
  vector<double>& winValues = thread.winValuesBuf;
  vector<double>& noResultValues = thread.noResultValuesBuf;
  vector<double>& scoreMeans = thread.scoreMeansBuf;
  vector<double>& scoreMeanSqs = thread.scoreMeanSqsBuf;
  vector<double>& selfUtilities = thread.utilityBuf;
  vector<int64_t>& visits = thread.visitsBuf;

  int64_t totalChildVisits = 0;
  int64_t maxChildVisits = 0;

  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex);

  int numChildren = node.numChildren;
  int numGoodChildren = 0;
  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = node.children[i];

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t childVisits = child->stats.visits;
    double winValueSum = child->stats.winValueSum;
    double noResultValueSum = child->stats.noResultValueSum;
    double scoreMeanSum = child->stats.scoreMeanSum;
    double scoreMeanSqSum = child->stats.scoreMeanSqSum;
    double valueSumWeight = child->stats.valueSumWeight;
    double childResultUtilitySum = child->stats.getResultUtilitySum(searchParams);
    child->statsLock.clear(std::memory_order_release);

    if(childVisits <= 0)
      continue;
    assert(valueSumWeight > 0.0);

    double childUtility = getUtility(childResultUtilitySum, scoreMeanSum, scoreMeanSqSum, valueSumWeight);

    winValues[numGoodChildren] = winValueSum / valueSumWeight;
    noResultValues[numGoodChildren] = noResultValueSum / valueSumWeight;
    scoreMeans[numGoodChildren] = scoreMeanSum / valueSumWeight;
    scoreMeanSqs[numGoodChildren] = scoreMeanSqSum / valueSumWeight;
    selfUtilities[numGoodChildren] = node.nextPla == P_WHITE ? childUtility : -childUtility;
    visits[numGoodChildren] = childVisits;
    totalChildVisits += childVisits;

    if(childVisits > maxChildVisits)
      maxChildVisits = childVisits;
    numGoodChildren++;
  }
  lock.unlock();

  if(searchParams.valueWeightExponent > 0)
    getValueChildWeights(numGoodChildren,selfUtilities,visits,valueChildWeights);

  //In the case we're enabling noise at the root node, also apply the slight subtraction
  //of visits from the root node's children so as to downweight the effect of the few dozen visits
  //we send towards children that are so bad that we never try them even once again.

  //One slightly surprising behavior is that this slight subtraction won't happen in the case where
  //we have just promoted a child to the root due to preservation of the tree across moves
  //but we haven't sent any playouts through the root yet. But having rootNoiseEnabled without
  //clearing the tree every search is a bit weird anyways.
  double amountToSubtract = 0.0;
  double amountToPrune = 0.0;
  if(isRoot && searchParams.rootNoiseEnabled) {
    amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxChildVisits/64.0);
    amountToPrune = std::min(searchParams.chosenMovePrune, maxChildVisits/64.0);
  }

  double winValueSum = 0.0;
  double noResultValueSum = 0.0;
  double scoreMeanSum = 0.0;
  double scoreMeanSqSum = 0.0;
  double valueSumWeight = 0.0;
  for(int i = 0; i<numGoodChildren; i++) {
    if(visits[i] < amountToPrune)
      continue;
    double weight = (double)visits[i] - amountToSubtract;
    if(weight < 0.0)
      continue;

    if(searchParams.visitsExponent != 1.0)
      weight = pow(weight, searchParams.visitsExponent);
    if(searchParams.valueWeightExponent > 0)
      weight *= pow(valueChildWeights[i], searchParams.valueWeightExponent);

    winValueSum += weight * winValues[i];
    noResultValueSum += weight * noResultValues[i];
    scoreMeanSum += weight * scoreMeans[i];
    scoreMeanSqSum += weight * scoreMeanSqs[i];
    valueSumWeight += weight;
  }

  //Also add in the direct evaluation of this node.
  {
    //Since we've scaled all the child weights in some arbitrary way, adjust and make sure
    //that the direct evaluation of the node still has precisely 1/N weight.
    //Do some things to carefully avoid divide by 0.
    double weight;
    if(searchParams.scaleParentWeight) {
      weight = (totalChildVisits > 0) ? valueSumWeight / totalChildVisits : valueSumWeight;
      if(weight < 0.001) //Just in case
        weight = 0.001;
    }
    else {
      weight = 1.0;
    }

    double winProb = (double)node.nnOutput->whiteWinProb;
    double noResultProb = (double)node.nnOutput->whiteNoResultProb;
    double scoreMean = (double)node.nnOutput->whiteScoreMean;
    double scoreMeanSq = (double)node.nnOutput->whiteScoreMeanSq;

    winValueSum += winProb * weight;
    noResultValueSum += noResultProb * weight;
    scoreMeanSum += scoreMean * weight;
    scoreMeanSqSum += scoreMeanSq * weight;
    valueSumWeight += weight;
  }

  while(node.statsLock.test_and_set(std::memory_order_acquire));
  node.stats.visits += numVisitsToAdd;
  //It's possible that these values are a bit wrong if there's a race and two threads each try to update this
  //each of them only having some of the latest updates for all the children. We just accept this and let the
  //error persist, it will get fixed the next time a visit comes through here and the values will at least
  //be consistent with each other within this node, since statsLock at least ensures these three are set atomically.
  node.stats.winValueSum = winValueSum;
  node.stats.noResultValueSum = noResultValueSum;
  node.stats.scoreMeanSum = scoreMeanSum;
  node.stats.scoreMeanSqSum = scoreMeanSqSum;
  node.stats.valueSumWeight = valueSumWeight;
  node.virtualLosses -= virtualLossesToSubtract;
  node.statsLock.clear(std::memory_order_release);
}

void Search::runSinglePlayout(SearchThread& thread) {
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE];
  playoutDescend(thread,*rootNode,posesWithChildBuf,true,0);

  //Restore thread state back to the root state
  thread.pla = rootPla;
  thread.board = rootBoard;
  thread.history = rootHistory;
}

void Search::setTerminalValue(SearchNode& node, double winValue, double noResultValue, double scoreMean, double scoreMeanSq, int32_t virtualLossesToSubtract) {
  while(node.statsLock.test_and_set(std::memory_order_acquire));
  node.stats.visits += 1;
  node.stats.winValueSum = winValue;
  node.stats.noResultValueSum = noResultValue;
  node.stats.scoreMeanSum = scoreMean;
  node.stats.scoreMeanSqSum = scoreMeanSq;
  node.stats.valueSumWeight = 1.0;
  node.virtualLosses -= virtualLossesToSubtract;
  node.statsLock.clear(std::memory_order_release);
}

void Search::initNodeNNOutput(
  SearchThread& thread, SearchNode& node,
  bool isRoot, bool skipCache, int32_t virtualLossesToSubtract, bool isReInit
) {
  bool includeOwnerMap = isRoot;
  nnEvaluator->evaluate(
    thread.board, thread.history, thread.pla,
    searchParams.drawEquivalentWinsForWhite,
    thread.nnResultBuf, thread.logger, skipCache, includeOwnerMap
  );

  node.nnOutput = std::move(thread.nnResultBuf.result);
  maybeAddPolicyNoise(thread,node,isRoot);

  //If this is a re-initialization of the nnOutput, we don't want to add any visits or anything.
  //Also don't bother updating any of the stats. Technically we should do so because winValueSum
  //and such will have changed potentially due to a new orientation of the neural net eval
  //slightly affecting the evals, but this is annoying to recompute from scratch, and on the next
  //visit updateStatsAfterPlayout should fix it all up anyways.
  if(isReInit)
    return;

  //Values in the search are from the perspective of white positive always
  double winProb = (double)node.nnOutput->whiteWinProb;
  double noResultProb = (double)node.nnOutput->whiteNoResultProb;
  double scoreMean = (double)node.nnOutput->whiteScoreMean;
  double scoreMeanSq = (double)node.nnOutput->whiteScoreMeanSq;

  while(node.statsLock.test_and_set(std::memory_order_acquire));
  node.stats.visits += 1;
  node.stats.winValueSum += winProb;
  node.stats.noResultValueSum += noResultProb;
  node.stats.scoreMeanSum += scoreMean;
  node.stats.scoreMeanSqSum += scoreMeanSq;
  node.stats.valueSumWeight += 1.0;
  node.virtualLosses -= virtualLossesToSubtract;
  node.statsLock.clear(std::memory_order_release);
}

void Search::playoutDescend(
  SearchThread& thread, SearchNode& node,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot, int32_t virtualLossesToSubtract
) {
  //Hit terminal node, finish
  //In the case where we're forcing the search to make another move at the root, don't terminate, actually run search for a move more.
  if(!isRoot && thread.history.isGameFinished) {
    if(thread.history.isNoResult) {
      double winValue = 0.0;
      double noResultValue = 1.0;
      double scoreMean = 0.0;
      double scoreMeanSq = 0.0;
      setTerminalValue(node, winValue, noResultValue, scoreMean, scoreMeanSq, virtualLossesToSubtract);
      return;
    }
    else {
      double winValue = ScoreValue::whiteWinsOfWinner(thread.history.winner, searchParams.drawEquivalentWinsForWhite);
      double noResultValue = 0.0;
      double scoreMean = ScoreValue::whiteScoreDrawAdjust(thread.history.finalWhiteMinusBlackScore,searchParams.drawEquivalentWinsForWhite,thread.history);
      double scoreMeanSq = ScoreValue::whiteScoreMeanSqOfScoreGridded(thread.history.finalWhiteMinusBlackScore,searchParams.drawEquivalentWinsForWhite,thread.history);
      setTerminalValue(node, winValue, noResultValue, scoreMean, scoreMeanSq, virtualLossesToSubtract);
      return;
    }
  }

  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex);

  //Hit leaf node, finish
  if(node.nnOutput == nullptr) {
    initNodeNNOutput(thread,node,isRoot,false,virtualLossesToSubtract,false);
    return;
  }
  //For the root node, make sure we have a whiteOwnerMap
  if(isRoot && node.nnOutput->whiteOwnerMap == NULL) {
    bool isReInit = true;
    initNodeNNOutput(thread,node,isRoot,false,0,isReInit);
    assert(node.nnOutput->whiteOwnerMap != NULL);
    //As isReInit is true, we don't return, just keep going, since we didn't count this as a true visit in the node stats
  }

  //Not leaf node, so recurse

  //Find the best child to descend down
  int bestChildIdx;
  Loc bestChildMoveLoc;
  selectBestChildToDescend(thread,node,bestChildIdx,bestChildMoveLoc,posesWithChildBuf,isRoot);

  //The absurdly rare case that the move chosen is not legal
  //(this should only happen either on a bug or where the nnHash doesn't have full legality information or when there's an actual hash collision).
  //Regenerate the neural net call and continue
  if(!thread.history.isLegal(thread.board,bestChildMoveLoc,thread.pla)) {
    bool isReInit = true;
    initNodeNNOutput(thread,node,isRoot,true,0,isReInit);

    if(thread.logStream != NULL)
      (*thread.logStream) << "WARNING: Chosen move not legal so regenerated nn output, nnhash=" << node.nnOutput->nnHash << endl;

    //As isReInit is true, we don't return, just keep going, since we didn't count this as a true visit in the node stats
    selectBestChildToDescend(thread,node,bestChildIdx,bestChildMoveLoc,posesWithChildBuf,isRoot);
    //We should absolutely be legal this time
    assert(thread.history.isLegal(thread.board,bestChildMoveLoc,thread.pla));
  }

  if(bestChildIdx < -1) {
    lock.unlock();
    throw StringError("Search error: No move with sane selection value - can't even pass?");
  }

  //Reallocate the children array to increase capacity if necessary
  if(bestChildIdx >= node.childrenCapacity) {
    int newCapacity = node.childrenCapacity + (node.childrenCapacity / 4) + 1;
    assert(newCapacity < 0x3FFF);
    SearchNode** newArr = new SearchNode*[newCapacity];
    for(int i = 0; i<node.numChildren; i++) {
      newArr[i] = node.children[i];
      node.children[i] = NULL;
    }
    SearchNode** oldArr = node.children;
    node.children = newArr;
    node.childrenCapacity = (uint16_t)newCapacity;
    delete[] oldArr;
  }

  Loc moveLoc = bestChildMoveLoc;

  //Allocate a new child node if necessary
  SearchNode* child;
  if(bestChildIdx == node.numChildren) {
    assert(thread.history.isLegal(thread.board,moveLoc,thread.pla));
    thread.history.makeBoardMoveAssumeLegal(thread.board,moveLoc,thread.pla,rootKoHashTable);
    thread.pla = getOpp(thread.pla);

    node.numChildren++;
    child = new SearchNode(*this,thread,moveLoc);
    node.children[bestChildIdx] = child;

    while(node.statsLock.test_and_set(std::memory_order_acquire));
    child->virtualLosses += searchParams.numVirtualLossesPerThread;
    node.statsLock.clear(std::memory_order_release);

    lock.unlock();
  }
  else {
    child = node.children[bestChildIdx];

    while(node.statsLock.test_and_set(std::memory_order_acquire));
    child->virtualLosses += searchParams.numVirtualLossesPerThread;
    node.statsLock.clear(std::memory_order_release);

    //Unlock before making moves if the child already exists since we don't depend on it at this point
    lock.unlock();

    assert(thread.history.isLegal(thread.board,moveLoc,thread.pla));
    thread.history.makeBoardMoveAssumeLegal(thread.board,moveLoc,thread.pla,rootKoHashTable);
    thread.pla = getOpp(thread.pla);
  }

  //Recurse!
  playoutDescend(thread,*child,posesWithChildBuf,false,searchParams.numVirtualLossesPerThread);

  //Update this node stats
  updateStatsAfterPlayout(node,thread,virtualLossesToSubtract,isRoot);
}


void Search::printRootOwnershipMap(ostream& out) {
  if(rootNode->nnOutput == nullptr)
    return;
  NNOutput& nnOutput = *(rootNode->nnOutput);
  if(nnOutput.whiteOwnerMap == NULL)
    return;

  for(int y = 0; y<rootBoard.y_size; y++) {
    for(int x = 0; x<rootBoard.x_size; x++) {
      int pos = NNPos::xyToPos(x,y,nnOutput.posLen);
      out << Global::strprintf("%6.1f ", nnOutput.whiteOwnerMap[pos]*100);
    }
    out << endl;
  }
  out << endl;
}

void Search::printRootPolicyMap(ostream& out) {
  if(rootNode->nnOutput == nullptr)
    return;
  NNOutput& nnOutput = *(rootNode->nnOutput);

  for(int y = 0; y<rootBoard.y_size; y++) {
    for(int x = 0; x<rootBoard.x_size; x++) {
      int pos = NNPos::xyToPos(x,y,nnOutput.posLen);
      out << Global::strprintf("%6.1f ", nnOutput.policyProbs[pos]*100);
    }
    out << endl;
  }
  out << endl;
}

void Search::printRootEndingScoreValueBonus(ostream& out) {
  if(rootNode->nnOutput == nullptr)
    return;
  NNOutput& nnOutput = *(rootNode->nnOutput);
  if(nnOutput.whiteOwnerMap == NULL)
    return;

  for(int i = 0; i<rootNode->numChildren; i++) {
    const SearchNode* child = rootNode->children[i];

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t childVisits = child->stats.visits;
    double childResultUtilitySum = child->stats.getResultUtilitySum(searchParams);
    double scoreMeanSum = child->stats.scoreMeanSum;
    double scoreMeanSqSum = child->stats.scoreMeanSqSum;
    double valueSumWeight = child->stats.valueSumWeight;
    child->statsLock.clear(std::memory_order_release);

    double utilityNoBonus = getUtility(childResultUtilitySum, scoreMeanSum, scoreMeanSqSum, valueSumWeight);

    double endingScoreBonus = getEndingWhiteScoreBonus(*rootNode,child);
    double oldScoreMeanSum = scoreMeanSum;
    scoreMeanSum += endingScoreBonus * valueSumWeight;
    scoreMeanSqSum = scoreMeanSqSum + (oldScoreMeanSum + scoreMeanSum) * endingScoreBonus;
    double utilityWithBonus = getUtility(childResultUtilitySum, scoreMeanSum, scoreMeanSqSum, valueSumWeight);

    out << Location::toString(child->prevMoveLoc,rootBoard) << " " << Global::strprintf(
      "visits %d utilityNoBonus %.2fc utilityWithBonus %.2fc endingScoreBonus %.2f",
      childVisits, utilityNoBonus*100, utilityWithBonus*100, endingScoreBonus
    );
    out << endl;
  }
}

void Search::printPV(ostream& out, const SearchNode* n, int maxDepth) {
  if(n == NULL)
    return;
  for(int depth = 0; depth < maxDepth; depth++) {
    const SearchNode& node = *n;
    std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
    unique_lock<std::mutex> lock(mutex);

    double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
    int bestChildIdx = -1;
    Loc bestChildMoveLoc = Board::NULL_LOC;

    for(int i = 0; i<node.numChildren; i++) {
      SearchNode* child = node.children[i];
      Loc moveLoc = child->prevMoveLoc;
      double selectionValue = getPlaySelectionValue(node,child);
      if(selectionValue > maxSelectionValue) {
        maxSelectionValue = selectionValue;
        bestChildIdx = i;
        bestChildMoveLoc = moveLoc;
      }
    }
    if(bestChildIdx < 0 || bestChildMoveLoc == Board::NULL_LOC)
      return;
    n = node.children[bestChildIdx];
    lock.unlock();

    if(depth > 0)
      out << " ";
    out << Location::toString(bestChildMoveLoc,rootBoard);
  }
}

void Search::printTree(ostream& out, const SearchNode* node, PrintTreeOptions options) {
  string prefix;
  printTreeHelper(out, node, options, prefix, 0, 0, NAN, NAN);
}

void Search::printTreeHelper(
  ostream& out, const SearchNode* n, const PrintTreeOptions& options,
  string& prefix, int64_t origVisits, int depth, double policyProb, double valueWeight
) {
  if(n == NULL)
    return;
  const SearchNode& node = *n;
  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex,std::defer_lock);

  while(node.statsLock.test_and_set(std::memory_order_acquire));
  int64_t visits = node.stats.visits;
  double resultUtilitySum = node.stats.getResultUtilitySum(searchParams);
  double scoreMeanSum = node.stats.scoreMeanSum;
  double scoreMeanSqSum = node.stats.scoreMeanSqSum;
  double valueSumWeight = node.stats.valueSumWeight;
  node.statsLock.clear(std::memory_order_release);

  double utility = getUtility(resultUtilitySum, scoreMeanSum, scoreMeanSqSum, valueSumWeight);

  if(depth == 0)
    origVisits = visits;

  //Output for this node
  {
    out << prefix;
    char buf[64];

    out << ": ";

    if(visits > 0) {
      sprintf(buf,"T %6.2fc ",(utility * 100.0));
      out << buf;
      sprintf(buf,"W %6.2fc ",(resultUtilitySum / valueSumWeight * 100.0));
      out << buf;
      sprintf(buf,"S %6.2fc (%+5.1f) ",
              (utility - resultUtilitySum / valueSumWeight) * 100.0,
              scoreMeanSum / valueSumWeight
      );
      out << buf;
    }

    bool hasNNValue = false;
    double nnResultValue;
    double nnTotalValue;
    lock.lock();
    if(node.nnOutput != nullptr) {
      nnResultValue = getResultUtilityFromNN(*node.nnOutput,searchParams);
      nnTotalValue = getUtilityFromNN(*node.nnOutput);
      hasNNValue = true;
    }
    lock.unlock();

    if(hasNNValue) {
      sprintf(buf,"VW %6.2fc VS %6.2fc ", nnResultValue * 100.0, (nnTotalValue - nnResultValue) * 100.0);
      out << buf;
    }
    else {
      sprintf(buf,"VW ---.--c VS ---.--c ");
      out << buf;
    }

    if(!isnan(policyProb)) {
      sprintf(buf,"P %5.2f%% ", policyProb * 100.0);
      out << buf;
    }
    if(!isnan(valueWeight)) {
      sprintf(buf,"VW %5.2f%% ", valueWeight * 100.0);
      out << buf;
    }

    sprintf(buf,"N %7" PRIu64 "  --  ", visits);
    out << buf;

    printPV(out, &node, 7);
    out << endl;
  }

  if(depth >= options.branch_.size()) {
    if(depth >= options.maxDepth_ + options.branch_.size())
      return;
    if(visits < options.minVisitsToExpand_)
      return;
    if((double)visits < origVisits * options.minVisitsPropToExpand_)
      return;
  }
  if(depth == options.branch_.size()) {
    out << "---" << playerToString(node.nextPla) << "(" << (node.nextPla == P_WHITE ? "^" : "v") << ")---" << endl;
  }

  lock.lock();

  int numChildren = node.numChildren;

  //Find all children and compute weighting of the children based on their values
  vector<double> valueChildWeights;
  {
    int numGoodChildren = 0;
    vector<double> goodValueChildWeights;
    vector<double> origMoveIdx;
    vector<double> selfUtilityBuf;
    vector<int64_t> visitsBuf;
    for(int i = 0; i<numChildren; i++) {
      const SearchNode* child = node.children[i];

      while(child->statsLock.test_and_set(std::memory_order_acquire));
      int64_t childVisits = child->stats.visits;
      double childResultUtilitySum = child->stats.getResultUtilitySum(searchParams);
      double childScoreMeanSum = child->stats.scoreMeanSum;
      double childScoreMeanSqSum = child->stats.scoreMeanSqSum;
      double childValueSumWeight = child->stats.valueSumWeight;
      child->statsLock.clear(std::memory_order_release);

      if(childVisits <= 0)
        continue;
      assert(childValueSumWeight > 0.0);

      double childUtility = getUtility(childResultUtilitySum, childScoreMeanSum, childScoreMeanSqSum, childValueSumWeight);

      numGoodChildren++;
      selfUtilityBuf.push_back(node.nextPla == P_WHITE ? childUtility : -childUtility);
      visitsBuf.push_back(childVisits);
      origMoveIdx.push_back(i);
    }

    getValueChildWeights(numGoodChildren,selfUtilityBuf,visitsBuf,goodValueChildWeights);
    for(int i = 0; i<numChildren; i++)
      valueChildWeights.push_back(0.0);
    for(int i = 0; i<numGoodChildren; i++)
      valueChildWeights[origMoveIdx[i]] = goodValueChildWeights[i];
  }

  //Find all children and record their play values
  vector<tuple<const SearchNode*,double,double,double>> valuedChildren;

  valuedChildren.reserve(numChildren);
  assert(node.nnOutput != nullptr);

  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = node.children[i];
    Loc moveLoc = child->prevMoveLoc;
    int movePos = getPos(moveLoc);
    double childPolicyProb = node.nnOutput->policyProbs[movePos];
    double selectionValue = getPlaySelectionValue(node,child);
    valuedChildren.push_back(std::make_tuple(child,childPolicyProb,selectionValue,valueChildWeights[i]));
  }

  lock.unlock();

  //Sort in order that we would want to play them
  auto compByValue = [](const tuple<const SearchNode*,double,double,double>& a, const tuple<const SearchNode*,double,double,double>& b) {
    return (std::get<2>(a)) > (std::get<2>(b));
  };
  std::stable_sort(valuedChildren.begin(),valuedChildren.end(),compByValue);

  //Apply filtering conditions, but include children that don't match the filtering condition
  //but where there are children afterward that do, in case we ever use something more complex
  //than plain visits as a filter criterion. Do this by finding the last child that we want as the threshold.
  int lastIdxWithEnoughVisits = numChildren-1;
  while(true) {
    if(lastIdxWithEnoughVisits <= 0)
      break;
    const SearchNode* child = std::get<0>(valuedChildren[lastIdxWithEnoughVisits]);

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t childVisits = child->stats.visits;
    child->statsLock.clear(std::memory_order_release);

    bool hasEnoughVisits = childVisits >= options.minVisitsToShow_
      && (double)childVisits >= origVisits * options.minVisitsPropToShow_;
    if(hasEnoughVisits)
      break;
    lastIdxWithEnoughVisits--;
  }

  int numChildrenToRecurseOn = numChildren;
  if(options.maxChildrenToShow_ < numChildrenToRecurseOn)
    numChildrenToRecurseOn = options.maxChildrenToShow_;
  if(lastIdxWithEnoughVisits+1 < numChildrenToRecurseOn)
    numChildrenToRecurseOn = lastIdxWithEnoughVisits+1;


  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = std::get<0>(valuedChildren[i]);
    double childPolicyProb =  std::get<1>(valuedChildren[i]);
    double childModelProb = std::get<3>(valuedChildren[i]);

    Loc moveLoc = child->prevMoveLoc;

    if((depth >= options.branch_.size() && i < numChildrenToRecurseOn) ||
       (depth < options.branch_.size() && moveLoc == options.branch_[depth]))
    {
      size_t oldLen = prefix.length();
      string locStr = Location::toString(moveLoc,rootBoard);
      if(locStr == "pass")
        prefix += "pss";
      else
        prefix += locStr;
      prefix += " ";
      while(prefix.length() < oldLen+4)
        prefix += " ";
      printTreeHelper(out,child,options,prefix,origVisits,depth+1,childPolicyProb,childModelProb);
      prefix.erase(oldLen);
    }
  }
}



