
//-------------------------------------------------------------------------------------
//This file contains the main core logic of the search.
//-------------------------------------------------------------------------------------

#include "../search/search.h"

#include <algorithm>
#include <numeric>

#include "../core/fancymath.h"
#include "../core/timer.h"
#include "../search/distributiontable.h"

using namespace std;

ReportedSearchValues::ReportedSearchValues()
{}
ReportedSearchValues::~ReportedSearchValues()
{}

NodeStatsAtomic::NodeStatsAtomic()
  :visits(0),
   winLossValueAvg(0.0),
   noResultValueAvg(0.0),
   scoreMeanAvg(0.0),
   scoreMeanSqAvg(0.0),
   leadAvg(0.0),
   utilityAvg(0.0),
   utilitySqAvg(0.0),
   weightSum(0.0),
   weightSqSum(0.0)
{}
NodeStatsAtomic::~NodeStatsAtomic()
{}

NodeStats::NodeStats()
  :visits(0),
   winLossValueAvg(0.0),
   noResultValueAvg(0.0),
   scoreMeanAvg(0.0),
   scoreMeanSqAvg(0.0),
   leadAvg(0.0),
   utilityAvg(0.0),
   utilitySqAvg(0.0),
   weightSum(0.0),
   weightSqSum(0.0)
{}
NodeStats::NodeStats(const NodeStatsAtomic& other)
  :visits(other.visits.load(std::memory_order_acquire)),
   winLossValueAvg(other.winLossValueAvg.load(std::memory_order_acquire)),
   noResultValueAvg(other.noResultValueAvg.load(std::memory_order_acquire)),
   scoreMeanAvg(other.scoreMeanAvg.load(std::memory_order_acquire)),
   scoreMeanSqAvg(other.scoreMeanSqAvg.load(std::memory_order_acquire)),
   leadAvg(other.leadAvg.load(std::memory_order_acquire)),
   utilityAvg(other.utilityAvg.load(std::memory_order_acquire)),
   utilitySqAvg(other.utilitySqAvg.load(std::memory_order_acquire)),
   weightSum(other.weightSum.load(std::memory_order_acquire)),
   weightSqSum(other.weightSqSum.load(std::memory_order_acquire))
{}
NodeStats::~NodeStats()
{}

MoreNodeStats::MoreNodeStats()
  :stats(),
   selfUtility(0.0),
   weightAdjusted(0.0),
   prevMoveLoc(Board::NULL_LOC)
{}
MoreNodeStats::~MoreNodeStats()
{}

double Search::getResultUtility(double winLossValue, double noResultValue) const {
  return (
    winLossValue * searchParams.winLossUtilityFactor +
    noResultValue * searchParams.noResultUtilityForWhite
  );
}

double Search::getResultUtilityFromNN(const NNOutput& nnOutput) const {
  return (
    (nnOutput.whiteWinProb - nnOutput.whiteLossProb) * searchParams.winLossUtilityFactor +
    nnOutput.whiteNoResultProb * searchParams.noResultUtilityForWhite
  );
}

double Search::getScoreStdev(double scoreMean, double scoreMeanSq) {
  double variance = scoreMeanSq - scoreMean * scoreMean;
  if(variance <= 0.0)
    return 0.0;
  return sqrt(variance);
}

//-----------------------------------------------------------------------------------------

SearchChildPointer::SearchChildPointer():
  data(NULL)
{}

SearchNode* SearchChildPointer::getIfAllocated() {
  return data.load(std::memory_order_acquire);
}

const SearchNode* SearchChildPointer::getIfAllocated() const {
  return data.load(std::memory_order_acquire);
}

SearchNode* SearchChildPointer::getIfAllocatedRelaxed() {
  return data.load(std::memory_order_relaxed);
}

void SearchChildPointer::store(SearchNode* node) {
  data.store(node, std::memory_order_release);
}

void SearchChildPointer::storeRelaxed(SearchNode* node) {
  data.store(node, std::memory_order_relaxed);
}

bool SearchChildPointer::storeIfNull(SearchNode* node) {
  SearchNode* expected = NULL;
  return data.compare_exchange_strong(expected, node, std::memory_order_acq_rel);
}

//-----------------------------------------------------------------------------------------

//Makes a search node resulting from prevPla playing prevLoc
SearchNode::SearchNode(Player prevPla, Loc prevLoc, SearchNode* p)
  :nextPla(getOpp(prevPla)),
   prevMoveLoc(prevLoc),
   parent(p),
   state(SearchNode::STATE_UNEVALUATED),
   nnOutput(),
   nodeAge(0),
   children0(NULL),
   children1(NULL),
   children2(NULL),
   stats(),
   virtualLosses(0),
   lastSubtreeValueBiasDeltaSum(0.0),lastSubtreeValueBiasWeight(0.0),
   subtreeValueBiasTableEntry(),
   dirtyCounter(0)
{
}

SearchChildPointer* SearchNode::getChildren(int& childrenCapacity) {
  return getChildren(state.load(std::memory_order_acquire),childrenCapacity);
}
const SearchChildPointer* SearchNode::getChildren(int& childrenCapacity) const {
  return getChildren(state.load(std::memory_order_acquire),childrenCapacity);
}

int SearchNode::iterateAndCountChildren() const {
  int numChildren = 0;

  int childrenCapacity;
  const SearchChildPointer* children = getChildren(childrenCapacity);
  for(int i = 0; i<childrenCapacity; i++) {
    if(children[i].getIfAllocated() == NULL)
      break;
    numChildren++;
  }
  return numChildren;
}

//Precondition: Assumes that we have actually checked the children array that stateValue suggests that
//we should use, and every slot, and that every slot in it is full up to numChildrenFullPlusOne-1, and
//that we have found a new legal child to add.
//Postcondition:
//Returns true: node state, stateValue, children arrays are all updated if needed so that they are large enough.
//Returns false: failure since another thread is handling it.
//Thread-safe.
bool SearchNode::maybeExpandChildrenCapacityForNewChild(int& stateValue, int numChildrenFullPlusOne) {
  int capacity = getChildrenCapacity(stateValue);
  if(capacity < numChildrenFullPlusOne) {
    assert(capacity == numChildrenFullPlusOne-1);
    return tryExpandingChildrenCapacityAssumeFull(stateValue);
  }
  return true;
}

int SearchNode::getChildrenCapacity(int stateValue) const {
  if(stateValue >= SearchNode::STATE_EXPANDED2)
    return SearchNode::CHILDREN2SIZE;
  if(stateValue >= SearchNode::STATE_EXPANDED1)
    return SearchNode::CHILDREN1SIZE;
  if(stateValue >= SearchNode::STATE_EXPANDED0)
    return SearchNode::CHILDREN0SIZE;
  return 0;
}

void SearchNode::initializeChildren() {
  assert(children0 == NULL);
  children0 = new SearchChildPointer[SearchNode::CHILDREN0SIZE];
}

//Precondition: Assumes that we have actually checked the childen array that stateValue suggests that
//we should use, and that every slot in it is full.
bool SearchNode::tryExpandingChildrenCapacityAssumeFull(int& stateValue) {
  if(stateValue < SearchNode::STATE_EXPANDED1) {
    if(stateValue == SearchNode::STATE_GROWING1)
      return false;
    assert(stateValue == SearchNode::STATE_EXPANDED0);
    bool suc = state.compare_exchange_strong(stateValue,SearchNode::STATE_GROWING1,std::memory_order_acq_rel);
    if(!suc) return false;
    stateValue = SearchNode::STATE_GROWING1;

    SearchChildPointer* children = new SearchChildPointer[SearchNode::CHILDREN1SIZE];
    SearchChildPointer* oldChildren = children0;
    for(int i = 0; i<SearchNode::CHILDREN0SIZE; i++) {
      //Loading relaxed is fine since by precondition, we've already observed that all of these
      //are non-null, so loading again it must be still true and we don't need any other synchronization.
      SearchNode* child = oldChildren[i].getIfAllocatedRelaxed();
      //Assert the precondition for calling this function in the first place
      assert(child != NULL);
      //Storing relaxed is fine since the array is not visible to other threads yet. The entire array will
      //be released shortly and that will ensure consumers see these childs, with an acquire on the whole array.
      children[i].storeRelaxed(child);
    }
    assert(children1 == NULL);
    children1 = children;
    state.store(SearchNode::STATE_EXPANDED1,std::memory_order_release);
    stateValue = SearchNode::STATE_EXPANDED1;
  }
  else if(stateValue < SearchNode::STATE_EXPANDED2) {
    if(stateValue == SearchNode::STATE_GROWING2)
      return false;
    assert(stateValue == SearchNode::STATE_EXPANDED1);
    bool suc = state.compare_exchange_strong(stateValue,SearchNode::STATE_GROWING2,std::memory_order_acq_rel);
    if(!suc) return false;
    stateValue = SearchNode::STATE_GROWING2;

    SearchChildPointer* children = new SearchChildPointer[SearchNode::CHILDREN2SIZE];
    SearchChildPointer* oldChildren = children1;
    for(int i = 0; i<SearchNode::CHILDREN1SIZE; i++) {
      //Loading relaxed is fine since by precondition, we've already observed that all of these
      //are non-null, so loading again it must be still true and we don't need any other synchronization.
      SearchNode* child = oldChildren[i].getIfAllocatedRelaxed();
      //Assert the precondition for calling this function in the first place
      assert(child != NULL);
      //Storing relaxed is fine since the array is not visible to other threads yet. The entire array will
      //be released shortly and that will ensure consumers see these childs, with an acquire on the whole array.
      children[i].storeRelaxed(child);
    }
    assert(children2 == NULL);
    children2 = children;
    state.store(SearchNode::STATE_EXPANDED2,std::memory_order_release);
    stateValue = SearchNode::STATE_EXPANDED2;
  }
  else {
    ASSERT_UNREACHABLE;
  }
  return true;
}

const SearchChildPointer* SearchNode::getChildren(int stateValue, int& childrenCapacity) const {
  if(stateValue >= SearchNode::STATE_EXPANDED2) {
    childrenCapacity = SearchNode::CHILDREN2SIZE;
    return children2;
  }
  if(stateValue >= SearchNode::STATE_EXPANDED1) {
    childrenCapacity = SearchNode::CHILDREN1SIZE;
    return children1;
  }
  if(stateValue >= SearchNode::STATE_EXPANDED0) {
    childrenCapacity = SearchNode::CHILDREN0SIZE;
    return children0;
  }
  childrenCapacity = 0;
  return NULL;
}
SearchChildPointer* SearchNode::getChildren(int stateValue, int& childrenCapacity) {
  if(stateValue >= SearchNode::STATE_EXPANDED2) {
    childrenCapacity = SearchNode::CHILDREN2SIZE;
    return children2;
  }
  if(stateValue >= SearchNode::STATE_EXPANDED1) {
    childrenCapacity = SearchNode::CHILDREN1SIZE;
    return children1;
  }
  if(stateValue >= SearchNode::STATE_EXPANDED0) {
    childrenCapacity = SearchNode::CHILDREN0SIZE;
    return children0;
  }
  childrenCapacity = 0;
  return NULL;
}

NNOutput* SearchNode::getNNOutput() {
  std::shared_ptr<NNOutput>* nn = nnOutput.load(std::memory_order_acquire);
  if(nn == NULL)
    return NULL;
  return nn->get();
}

const NNOutput* SearchNode::getNNOutput() const {
  const std::shared_ptr<NNOutput>* nn = nnOutput.load(std::memory_order_acquire);
  if(nn == NULL)
    return NULL;
  return nn->get();
}

bool SearchNode::storeNNOutput(std::shared_ptr<NNOutput>* newNNOutput, SearchThread& thread) {
  std::shared_ptr<NNOutput>* toCleanUp = nnOutput.exchange(newNNOutput, std::memory_order_acq_rel);
  if(toCleanUp != NULL) {
    thread.oldNNOutputsToCleanUp.push_back(toCleanUp);
    return false;
  }
  return true;
}

bool SearchNode::storeNNOutputIfNull(std::shared_ptr<NNOutput>* newNNOutput) {
  std::shared_ptr<NNOutput>* expected = NULL;
  return nnOutput.compare_exchange_strong(expected, newNNOutput, std::memory_order_acq_rel);
}

SearchNode::~SearchNode() {
  int childrenCapacity;
  SearchChildPointer* children = getChildren(state.load(),childrenCapacity);
  int i = 0;
  for(; i<childrenCapacity; i++) {
    SearchNode* child = children[i].getIfAllocated();
    if(child != NULL)
      delete child;
    else
      break;
  }
  for(; i<childrenCapacity; i++) {
    SearchNode* child = children[i].getIfAllocated();
    (void)child;
    assert(child == NULL);
  }

  if(children2 != NULL)
    delete[] children2;
  if(children1 != NULL)
    delete[] children1;
  if(children0 != NULL)
    delete[] children0;

  if(nnOutput != NULL)
    delete nnOutput;
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

SearchThread::SearchThread(int tIdx, const Search& search)
  :threadIdx(tIdx),
   pla(search.rootPla),board(search.rootBoard),
   history(search.rootHistory),
   rand(makeSeed(search,tIdx)),
   nnResultBuf(),
   statsBuf(),
   upperBoundVisitsLeft(1e30),
   oldNNOutputsToCleanUp()
{
  statsBuf.resize(NNPos::MAX_NN_POLICY_SIZE);

  //Reserving even this many is almost certainly overkill but should guarantee that we never have hit allocation here.
  oldNNOutputsToCleanUp.reserve(8);
}
SearchThread::~SearchThread() {
  for(size_t i = 0; i<oldNNOutputsToCleanUp.size(); i++)
    delete oldNNOutputsToCleanUp[i];
  oldNNOutputsToCleanUp.resize(0);
}

//-----------------------------------------------------------------------------------------

static const double VALUE_WEIGHT_DEGREES_OF_FREEDOM = 3.0;

Search::Search(SearchParams params, NNEvaluator* nnEval, Logger* lg, const string& rSeed)
  :rootPla(P_BLACK),
   rootBoard(),rootHistory(),rootHintLoc(Board::NULL_LOC),
   avoidMoveUntilByLocBlack(),avoidMoveUntilByLocWhite(),
   rootSafeArea(NULL),
   recentScoreCenter(0.0),
   mirroringPla(C_EMPTY),
   mirrorAdvantage(0.0),
   mirrorCenterIsSymmetric(false),
   alwaysIncludeOwnerMap(false),
   searchParams(params),numSearchesBegun(0),searchNodeAge(0),
   plaThatSearchIsFor(C_EMPTY),plaThatSearchIsForLastSearch(C_EMPTY),
   lastSearchNumPlayouts(0),
   effectiveSearchTimeCarriedOver(0.0),
   randSeed(rSeed),
   rootKoHashTable(NULL),
   valueWeightDistribution(NULL),
   normToTApproxZ(0.0),
   normToTApproxTable(),
   rootNode(NULL),
   mutexPool(NULL),
   nnEvaluator(nnEval),
   nnXLen(),
   nnYLen(),
   policySize(),
   nonSearchRand(rSeed + string("$nonSearchRand")),
   subtreeValueBiasTable(NULL),
   logger(lg),
   numThreadsSpawned(0),
   threads(NULL),
   threadTasks(NULL),
   threadTasksRemaining(NULL),
   oldNNOutputsToCleanUpMutex(),
   oldNNOutputsToCleanUp()
{
  assert(logger != NULL);
  nnXLen = nnEval->getNNXLen();
  nnYLen = nnEval->getNNYLen();
  assert(nnXLen > 0 && nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen > 0 && nnYLen <= NNPos::MAX_BOARD_LEN);
  policySize = NNPos::getPolicySize(nnXLen,nnYLen);
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
  clearOldNNOutputs();
  delete[] rootSafeArea;
  delete rootKoHashTable;
  delete valueWeightDistribution;
  delete rootNode;
  delete mutexPool;
  delete subtreeValueBiasTable;
  killThreads();
}

static void threadTaskLoop(Search* search, int threadIdx) {
  while(true) {
    std::function<void(int)>* task;
    bool suc = search->threadTasks[threadIdx-1].waitPop(task);
    if(!suc)
      return;

    try {
      (*task)(threadIdx);
      //Don't delete task, the convention is tasks are owned by the joining thread
    }
    catch(const exception& e) {
      search->logger->write(string("ERROR: Search thread failed: ") + e.what());
      search->threadTasksRemaining->add(-1);
      throw;
    }
    catch(const string& e) {
      search->logger->write("ERROR: Search thread failed: " + e);
      search->threadTasksRemaining->add(-1);
      throw;
    }
    catch(...) {
      search->logger->write("ERROR: Search thread failed with unexpected throw");
      search->threadTasksRemaining->add(-1);
      throw;
    }
    search->threadTasksRemaining->add(-1);
  }
}

void Search::spawnThreadsIfNeeded() {
  int desiredNumThreads = searchParams.numThreads-1;
  if(numThreadsSpawned >= desiredNumThreads)
    return;
  killThreads();
  threadTasks = new ThreadSafeQueue<std::function<void(int)>*>[desiredNumThreads];
  threadTasksRemaining = new ThreadSafeCounter();
  threads = new std::thread[desiredNumThreads];
  for(int i = 0; i<desiredNumThreads; i++)
    threads[i] = std::thread(threadTaskLoop,this,i+1);
  numThreadsSpawned = desiredNumThreads;
}

void Search::killThreads() {
  if(numThreadsSpawned <= 0)
    return;
  for(int i = 0; i<numThreadsSpawned; i++)
    threadTasks[i].close();
  for(int i = 0; i<numThreadsSpawned; i++)
    threads[i].join();
  delete[] threadTasks;
  delete threadTasksRemaining;
  delete[] threads;
  threadTasks = NULL;
  threadTasksRemaining = NULL;
  threads = NULL;
  numThreadsSpawned = 0;
}

void Search::respawnThreads() {
  killThreads();
  spawnThreadsIfNeeded();
}

void Search::performTaskWithThreads(std::function<void(int)>* task) {
  spawnThreadsIfNeeded();
  int numAdditionalThreadsToUse = searchParams.numThreads-1;
  if(numAdditionalThreadsToUse == 0) {
    (*task)(0);
  }
  else {
    assert(numAdditionalThreadsToUse <= numThreadsSpawned);
    threadTasksRemaining->add(numAdditionalThreadsToUse);
    for(int i = 0; i<numAdditionalThreadsToUse; i++)
      threadTasks[i].forcePush(task);
    (*task)(0);
    threadTasksRemaining->waitUntilZero();
  }
}

void Search::clearOldNNOutputs() {
  for(size_t i = 0; i<oldNNOutputsToCleanUp.size(); i++)
    delete oldNNOutputsToCleanUp[i];
  oldNNOutputsToCleanUp.resize(0);
}
void Search::transferOldNNOutputs(SearchThread& thread) {
  std::lock_guard<std::mutex> lock(oldNNOutputsToCleanUpMutex);
  for(size_t i = 0; i<thread.oldNNOutputsToCleanUp.size(); i++)
    oldNNOutputsToCleanUp.push_back(thread.oldNNOutputsToCleanUp[i]);
  thread.oldNNOutputsToCleanUp.resize(0);
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

Player Search::getPlayoutDoublingAdvantagePla() const {
  return searchParams.playoutDoublingAdvantagePla == C_EMPTY ? plaThatSearchIsFor : searchParams.playoutDoublingAdvantagePla;
}

void Search::setPosition(Player pla, const Board& board, const BoardHistory& history) {
  clearSearch();
  rootPla = pla;
  plaThatSearchIsFor = C_EMPTY;
  rootBoard = board;
  rootHistory = history;
  rootKoHashTable->recompute(rootHistory);
  avoidMoveUntilByLocBlack.clear();
  avoidMoveUntilByLocWhite.clear();
}

void Search::setPlayerAndClearHistory(Player pla) {
  clearSearch();
  rootPla = pla;
  plaThatSearchIsFor = C_EMPTY;
  rootBoard.clearSimpleKoLoc();
  Rules rules = rootHistory.rules;
  //Preserve this value even when we get multiple moves in a row by some player
  bool assumeMultipleStartingBlackMovesAreHandicap = rootHistory.assumeMultipleStartingBlackMovesAreHandicap;
  rootHistory.clear(rootBoard,rootPla,rules,rootHistory.encorePhase);
  rootHistory.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);

  rootKoHashTable->recompute(rootHistory);
  avoidMoveUntilByLocBlack.clear();
  avoidMoveUntilByLocWhite.clear();
}

void Search::setKomiIfNew(float newKomi) {
  if(rootHistory.rules.komi != newKomi) {
    clearSearch();
    rootHistory.setKomi(newKomi);
  }
}

void Search::setAvoidMoveUntilByLoc(const std::vector<int>& bVec, const std::vector<int>& wVec) {
  if(avoidMoveUntilByLocBlack == bVec && avoidMoveUntilByLocWhite == wVec)
    return;
  clearSearch();
  avoidMoveUntilByLocBlack = bVec;
  avoidMoveUntilByLocWhite = wVec;
}

void Search::setRootHintLoc(Loc loc) {
  //When we positively change the hint loc, we clear the search to make absolutely sure
  //that the hintloc takes effect, and that all nnevals (including the root noise that adds the hintloc) has a chance to happen
  if(loc != Board::NULL_LOC && rootHintLoc != loc)
    clearSearch();
  rootHintLoc = loc;
}

void Search::setAlwaysIncludeOwnerMap(bool b) {
  if(!alwaysIncludeOwnerMap && b)
    clearSearch();
  alwaysIncludeOwnerMap = b;
}

void Search::setParams(SearchParams params) {
  clearSearch();
  searchParams = params;
}

void Search::setParamsNoClearing(SearchParams params) {
  searchParams = params;
}

void Search::setNNEval(NNEvaluator* nnEval) {
  clearSearch();
  nnEvaluator = nnEval;
  nnXLen = nnEval->getNNXLen();
  nnYLen = nnEval->getNNYLen();
  assert(nnXLen > 0 && nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen > 0 && nnYLen <= NNPos::MAX_BOARD_LEN);
  policySize = NNPos::getPolicySize(nnXLen,nnYLen);
}

void Search::clearSearch() {
  effectiveSearchTimeCarriedOver = 0.0;
  delete rootNode;
  rootNode = NULL;
  clearOldNNOutputs();
  searchNodeAge = 0;
}

bool Search::isLegalTolerant(Loc moveLoc, Player movePla) const {
  //Tolerate sgf files or GTP reporting suicide moves, even if somehow the rules are set to disallow them.
  bool multiStoneSuicideLegal = true;

  //If we somehow have the same player making multiple moves in a row (possible in GTP or an sgf file),
  //clear the ko loc - the simple ko loc of a player should not prohibit the opponent playing there!
  if(movePla != rootPla) {
    Board copy = rootBoard;
    copy.clearSimpleKoLoc();
    return copy.isLegal(moveLoc,movePla,multiStoneSuicideLegal);
  }
  else {
    return rootHistory.isLegalTolerant(rootBoard,moveLoc,movePla);
  }
}

bool Search::isLegalStrict(Loc moveLoc, Player movePla) const {
  return movePla == rootPla && rootHistory.isLegal(rootBoard,moveLoc,movePla);
}

bool Search::makeMove(Loc moveLoc, Player movePla) {
  return makeMove(moveLoc,movePla,false);
}

bool Search::makeMove(Loc moveLoc, Player movePla, bool preventEncore) {
  if(!isLegalTolerant(moveLoc,movePla))
    return false;

  if(movePla != rootPla)
    setPlayerAndClearHistory(movePla);

  if(rootNode != NULL) {
    bool foundChild = false;
    int foundChildIdx = -1;

    int childrenCapacity;
    SearchChildPointer* children = rootNode->getChildren(childrenCapacity);
    int numChildren = 0;
    for(int i = 0; i<childrenCapacity; i++) {
      SearchNode* child = children[i].getIfAllocated();
      if(child == NULL)
        break;
      numChildren++;
      if(!foundChild && child->prevMoveLoc == moveLoc) {
        foundChild = true;
        foundChildIdx = i;
      }
    }

    //Just in case, make sure the child has an nnOutput, otherwise no point keeping it.
    //This is a safeguard against any oddity involving node preservation into states that
    //were considered terminal.
    if(foundChild) {
      SearchNode* child = children[foundChildIdx].getIfAllocated();
      assert(child != NULL);
      NNOutput* nnOutput = child->getNNOutput();
      if(nnOutput == NULL)
        foundChild = false;
    }

    if(foundChild) {
      SearchNode* child = children[foundChildIdx].getIfAllocated();
      assert(child != NULL);

      //Account for time carried over
      {
        int64_t rootVisits = rootNode->stats.visits.load(std::memory_order_acquire);
        int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
        effectiveSearchTimeCarriedOver = effectiveSearchTimeCarriedOver * (double)childVisits / (double)rootVisits * searchParams.treeReuseCarryOverTimeFactor;
      }

      child->parent = NULL;

      //Eliminate child entry in the array to prevent its deletion along with the root
      children[foundChildIdx].store(NULL);
      //But do a bit of hackery to ensure that children remain contiguously allocated - swap the last one into place.
      children[foundChildIdx].store(children[numChildren-1].getIfAllocated());
      children[numChildren-1].store(NULL);
      //Delete the root and replace it with the child
      recursivelyRemoveSubtreeValueBiasBeforeDeleteSynchronous(rootNode);
      delete rootNode;
      rootNode = child;
    }
    else {
      clearSearch();
    }
  }

  //If the white handicap bonus changes due to the move, we will also need to recompute everything since this is
  //basically like a change to the komi.
  float oldWhiteHandicapBonusScore = rootHistory.whiteHandicapBonusScore;

  rootHistory.makeBoardMoveAssumeLegal(rootBoard,moveLoc,rootPla,rootKoHashTable,preventEncore);
  rootPla = getOpp(rootPla);
  rootKoHashTable->recompute(rootHistory);
  avoidMoveUntilByLocBlack.clear();
  avoidMoveUntilByLocWhite.clear();

  if(rootHistory.whiteHandicapBonusScore != oldWhiteHandicapBonusScore)
    clearSearch();

  //In the case that we are conservativePass and a pass would end the game, need to clear the search.
  //This is because deeper in the tree, such a node would have been explored as ending the game, but now that
  //it's a root pass, it needs to be treated as if it no longer ends the game.
  //In the case that we're preventing encore, and the phase would have ended, we also need to clear the search
  //since the search was conducted on the assumption that we're going into encore now.
  if((searchParams.conservativePass && rootHistory.passWouldEndGame(rootBoard,rootPla)) ||
     (preventEncore && rootHistory.passWouldEndPhase(rootBoard,rootPla)))
    clearSearch();

  return true;
}


double Search::getScoreUtility(double scoreMeanAvg, double scoreMeanSqAvg) const {
  double scoreMean = scoreMeanAvg;
  double scoreMeanSq = scoreMeanSqAvg;
  double scoreStdev = getScoreStdev(scoreMean, scoreMeanSq);
  double staticScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0,rootBoard);
  double dynamicScoreValue = ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,recentScoreCenter,searchParams.dynamicScoreCenterScale,rootBoard);
  return staticScoreValue * searchParams.staticScoreUtilityFactor + dynamicScoreValue * searchParams.dynamicScoreUtilityFactor;
}

double Search::getScoreUtilityDiff(double scoreMeanAvg, double scoreMeanSqAvg, double delta) const {
  double scoreMean = scoreMeanAvg;
  double scoreMeanSq = scoreMeanSqAvg;
  double scoreStdev = getScoreStdev(scoreMean, scoreMeanSq);
  double staticScoreValueDiff =
    ScoreValue::expectedWhiteScoreValue(scoreMean + delta,scoreStdev,0.0,2.0,rootBoard)
    -ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,0.0,2.0,rootBoard);
  double dynamicScoreValueDiff =
    ScoreValue::expectedWhiteScoreValue(scoreMean + delta,scoreStdev,recentScoreCenter,searchParams.dynamicScoreCenterScale,rootBoard)
    -ScoreValue::expectedWhiteScoreValue(scoreMean,scoreStdev,recentScoreCenter,searchParams.dynamicScoreCenterScale,rootBoard);
  return staticScoreValueDiff * searchParams.staticScoreUtilityFactor + dynamicScoreValueDiff * searchParams.dynamicScoreUtilityFactor;
}

//Ignores scoreMeanSq's effect on the utility, since that's complicated
double Search::getApproxScoreUtilityDerivative(double scoreMean) const {
  double staticScoreValueDerivative = ScoreValue::whiteDScoreValueDScoreSmoothNoDrawAdjust(scoreMean,0.0,2.0,rootBoard);
  double dynamicScoreValueDerivative = ScoreValue::whiteDScoreValueDScoreSmoothNoDrawAdjust(scoreMean,recentScoreCenter,searchParams.dynamicScoreCenterScale,rootBoard);
  return staticScoreValueDerivative * searchParams.staticScoreUtilityFactor + dynamicScoreValueDerivative * searchParams.dynamicScoreUtilityFactor;
}


double Search::getUtilityFromNN(const NNOutput& nnOutput) const {
  double resultUtility = getResultUtilityFromNN(nnOutput);
  return resultUtility + getScoreUtility(nnOutput.whiteScoreMean, nnOutput.whiteScoreMeanSq);
}

uint32_t Search::chooseIndexWithTemperature(Rand& rand, const double* relativeProbs, int numRelativeProbs, double temperature) {
  assert(numRelativeProbs > 0);
  assert(numRelativeProbs <= Board::MAX_ARR_SIZE); //We're just doing this on the stack
  double processedRelProbs[Board::MAX_ARR_SIZE];

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
      processedRelProbs[i] = relativeProbs[i] <= 0.0 ? 0.0 : exp((log(relativeProbs[i]) - logMaxValue) / temperature);
      sum += processedRelProbs[i];
    }
    assert(sum > 0.0);
    uint32_t idxChosen = rand.nextUInt(processedRelProbs,numRelativeProbs);
    return idxChosen;
  }
}

double Search::interpolateEarly(double halflife, double earlyValue, double value) const {
  double rawHalflives = (rootHistory.initialTurnNumber + rootHistory.moveHistory.size()) / halflife;
  double halflives = rawHalflives * 19.0 / sqrt(rootBoard.x_size*rootBoard.y_size);
  return value + (earlyValue - value) * pow(0.5, halflives);
}

Loc Search::runWholeSearchAndGetMove(Player movePla) {
  return runWholeSearchAndGetMove(movePla,false);
}

Loc Search::runWholeSearchAndGetMove(Player movePla, bool pondering) {
  runWholeSearch(movePla,pondering);
  return getChosenMoveLoc();
}

void Search::runWholeSearch(Player movePla) {
  runWholeSearch(movePla,false);
}

void Search::runWholeSearch(Player movePla, bool pondering) {
  if(movePla != rootPla)
    setPlayerAndClearHistory(movePla);
  std::atomic<bool> shouldStopNow(false);
  runWholeSearch(shouldStopNow,pondering);
}

void Search::runWholeSearch(std::atomic<bool>& shouldStopNow) {
  runWholeSearch(shouldStopNow, false);
}

void Search::runWholeSearch(std::atomic<bool>& shouldStopNow, bool pondering) {
  std::function<void()>* searchBegun = NULL;
  runWholeSearch(shouldStopNow,searchBegun,pondering,TimeControls(),1.0);
}

double Search::numVisitsNeededToBeNonFutile(double maxVisitsMoveVisits) {
  double requiredVisits = searchParams.futileVisitsThreshold * maxVisitsMoveVisits;
  //In the case where we're playing high temperature, also require that we can't get to more than a 1:100 odds of playing the move.
  double chosenMoveTemperature = interpolateEarly(
    searchParams.chosenMoveTemperatureHalflife, searchParams.chosenMoveTemperatureEarly, searchParams.chosenMoveTemperature
  );
  if(chosenMoveTemperature < 1e-3)
    return requiredVisits;
  double requiredVisitsDueToTemp = maxVisitsMoveVisits * pow(0.01, chosenMoveTemperature);
  return std::min(requiredVisits, requiredVisitsDueToTemp);
}

double Search::computeUpperBoundVisitsLeftDueToTime(
  int64_t rootVisits, double timeUsed, double plannedTimeLimit
) {
  if(rootVisits <= 1)
    return 1e30;
  double timeThoughtSoFar = effectiveSearchTimeCarriedOver + timeUsed;
  double timeLeftPlanned = plannedTimeLimit - timeUsed;
  //Require at least a tenth of a second of search to begin to trust an estimate of visits/time.
  if(timeThoughtSoFar < 0.1)
    return 1e30;

  double proportionOfTimeThoughtLeft = timeLeftPlanned / timeThoughtSoFar;
  return ceil(proportionOfTimeThoughtLeft * rootVisits + searchParams.numThreads-1);
}

double Search::recomputeSearchTimeLimit(
  const TimeControls& tc, double timeUsed, double searchFactor, int64_t rootVisits
) {
  double tcMin;
  double tcRec;
  double tcMax;
  tc.getTime(rootBoard,rootHistory,searchParams.lagBuffer,tcMin,tcRec,tcMax);

  tcRec *= searchParams.overallocateTimeFactor;

  if(searchParams.midgameTimeFactor != 1.0) {
    double boardAreaScale = rootBoard.x_size * rootBoard.y_size / 361.0;
    int64_t presumedTurnNumber = rootHistory.initialTurnNumber + rootHistory.moveHistory.size();
    if(presumedTurnNumber < 0) presumedTurnNumber = 0;

    double midGameWeight;
    if(presumedTurnNumber < searchParams.midgameTurnPeakTime * boardAreaScale)
      midGameWeight = (double)presumedTurnNumber / (searchParams.midgameTurnPeakTime * boardAreaScale);
    else
      midGameWeight = exp(
        -(presumedTurnNumber - searchParams.midgameTurnPeakTime * boardAreaScale) /
        (searchParams.endgameTurnTimeDecay * boardAreaScale)
      );
    if(midGameWeight < 0)
      midGameWeight = 0;
    if(midGameWeight > 1)
      midGameWeight = 1;

    tcRec *= 1.0 + midGameWeight * (searchParams.midgameTimeFactor - 1.0);
  }

  if(searchParams.obviousMovesTimeFactor < 1.0) {
    double surprise = 0.0;
    double searchEntropy = 0.0;
    double policyEntropy = 0.0;
    bool suc = getPolicySurpriseAndEntropy(surprise, searchEntropy, policyEntropy);
    if(suc) {
      //If the original policy was confident and the surprise is low, then this is probably an "obvious" move.
      double obviousnessByEntropy = exp(-policyEntropy/searchParams.obviousMovesPolicyEntropyTolerance);
      double obviousnessBySurprise = exp(-surprise/searchParams.obviousMovesPolicySurpriseTolerance);
      double obviousnessWeight = std::min(obviousnessByEntropy, obviousnessBySurprise);
      tcRec *= 1.0 + obviousnessWeight * (searchParams.obviousMovesTimeFactor - 1.0);
    }
  }

  if(tcRec > 1e-20) {
    double remainingTimeNeeded = tcRec - effectiveSearchTimeCarriedOver;
    double remainingTimeNeededFactor = remainingTimeNeeded/tcRec;
    //TODO this is a bit conservative relative to old behavior, it might be of slightly detrimental value, needs testing.
    //Apply softplus so that we still do a tiny bit of search even in the presence of variable search time instead of instamoving,
    //there are some benefits from root-level search due to broader root exploration and the cost is small, also we may be over
    //counting the ponder benefit if search is faster on this node than on the previous turn.
    tcRec = tcRec * std::min(1.0, log(1.0+exp(remainingTimeNeededFactor * 6.0)) / 6.0);
  }

  //Make sure we're not wasting time
  tcRec = tc.roundUpTimeLimitIfNeeded(searchParams.lagBuffer,timeUsed,tcRec);
  if(tcRec > tcMax) tcRec = tcMax;

  //After rounding up time, check if with our planned rounded time, anything is futile to search
  if(searchParams.futileVisitsThreshold > 0) {
    double upperBoundVisitsLeftDueToTime = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, tcRec);
    if(upperBoundVisitsLeftDueToTime < searchParams.futileVisitsThreshold * rootVisits) {
      vector<Loc> locs;
      vector<double> playSelectionValues;
      vector<double> visitCounts;
      bool suc = getPlaySelectionValues(locs, playSelectionValues, &visitCounts, 1.0);
      if(suc && playSelectionValues.size() > 0) {
        //This may fail to hold if we have no actual visits and play selections are being pulled from stuff like raw policy
        if(playSelectionValues.size() == visitCounts.size()) {
          int numMoves = (int)playSelectionValues.size();
          int maxVisitsIdx = 0;
          int bestMoveIdx = 0;
          for(int i = 1; i<numMoves; i++) {
            if(playSelectionValues[i] > playSelectionValues[bestMoveIdx])
              bestMoveIdx = i;
            if(visitCounts[i] > visitCounts[maxVisitsIdx])
              maxVisitsIdx = i;
          }
          if(maxVisitsIdx == bestMoveIdx) {
            double requiredVisits = numVisitsNeededToBeNonFutile(visitCounts[maxVisitsIdx]);
            bool foundPossibleAlternativeMove = false;
            for(int i = 0; i<numMoves; i++) {
              if(i == bestMoveIdx)
                continue;
              if(visitCounts[i] + upperBoundVisitsLeftDueToTime >= requiredVisits) {
                foundPossibleAlternativeMove = true;
                break;
              }
            }
            if(!foundPossibleAlternativeMove) {
              //We should stop search now - set our desired thinking to very slightly smaller than what we used.
              tcRec = timeUsed * (1.0 - (1e-10));
            }
          }
        }
      }
    }
  }

  //Make sure we're not wasting time, even after considering that we might want to stop early
  tcRec = tc.roundUpTimeLimitIfNeeded(searchParams.lagBuffer,timeUsed,tcRec);
  if(tcRec > tcMax) tcRec = tcMax;

  //Apply caps and search factor
  //Since searchFactor is mainly used for friendliness (like, play faster after many passes)
  //we allow it to violate the min time.
  if(tcRec < tcMin) tcRec = tcMin;
  tcRec *= searchFactor;
  if(tcRec > tcMax) tcRec = tcMax;

  return tcRec;
}

void Search::runWholeSearch(
  std::atomic<bool>& shouldStopNow,
  std::function<void()>* searchBegun,
  bool pondering,
  const TimeControls& tc,
  double searchFactor
) {

  ClockTimer timer;
  atomic<int64_t> numPlayoutsShared(0);

  if(!std::atomic_is_lock_free(&numPlayoutsShared))
    logger->write("Warning: int64_t atomic numPlayoutsShared is not lock free");
  if(!std::atomic_is_lock_free(&shouldStopNow))
    logger->write("Warning: bool atomic shouldStopNow is not lock free");

  //Do this first, just in case this causes us to clear things and have 0 effective time carried over
  beginSearch(pondering);
  if(searchBegun != NULL)
    (*searchBegun)();
  const int64_t numNonPlayoutVisits = getRootVisits();

  //Compute caps on search
  int64_t maxVisits = pondering ? searchParams.maxVisitsPondering : searchParams.maxVisits;
  int64_t maxPlayouts = pondering ? searchParams.maxPlayoutsPondering : searchParams.maxPlayouts;
  double_t maxTime = pondering ? searchParams.maxTimePondering : searchParams.maxTime;

  {
    //Possibly reduce computation time, for human friendliness
    if(rootHistory.moveHistory.size() >= 1 && rootHistory.moveHistory[rootHistory.moveHistory.size()-1].loc == Board::PASS_LOC) {
      if(rootHistory.moveHistory.size() >= 3 && rootHistory.moveHistory[rootHistory.moveHistory.size()-3].loc == Board::PASS_LOC)
        searchFactor *= searchParams.searchFactorAfterTwoPass;
      else
        searchFactor *= searchParams.searchFactorAfterOnePass;
    }

    if(searchFactor != 1.0) {
      double cap = (double)((int64_t)1L << 62);
      maxVisits = (int64_t)ceil(std::min(cap, maxVisits * searchFactor));
      maxPlayouts = (int64_t)ceil(std::min(cap, maxPlayouts * searchFactor));
      maxTime = maxTime * searchFactor;
    }
  }

  //Apply time controls. These two don't particularly need to be synchronized with each other so its fine to have two separate atomics.
  std::atomic<double> tcMaxTime(1e30);
  std::atomic<double> upperBoundVisitsLeftDueToTime(1e30);
  const bool hasMaxTime = maxTime < 1.0e12;
  const bool hasTc = !pondering && !tc.isEffectivelyUnlimitedTime();
  if(!pondering && (hasTc || hasMaxTime)) {
    int64_t rootVisits = numPlayoutsShared.load(std::memory_order_relaxed) + numNonPlayoutVisits;
    double timeUsed = timer.getSeconds();
    double tcLimit = 1e30;
    if(hasTc) {
      tcLimit = recomputeSearchTimeLimit(tc, timeUsed, searchFactor, rootVisits);
      tcMaxTime.store(tcLimit, std::memory_order_release);
    }
    double upperBoundVisits = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, std::min(tcLimit,maxTime));
    upperBoundVisitsLeftDueToTime.store(upperBoundVisits, std::memory_order_release);
  }

  std::function<void(int)> searchLoop = [
    this,&timer,&numPlayoutsShared,numNonPlayoutVisits,&tcMaxTime,&upperBoundVisitsLeftDueToTime,&tc,
    &hasMaxTime,&hasTc,
    &shouldStopNow,maxVisits,maxPlayouts,maxTime,pondering,searchFactor
  ](int threadIdx) {
    SearchThread* stbuf = new SearchThread(threadIdx,*this);

    int64_t numPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
    try {
      double lastTimeUsedRecomputingTcLimit = 0.0;
      while(true) {
        double timeUsed = 0.0;
        if(hasTc || hasMaxTime)
          timeUsed = timer.getSeconds();

        double tcMaxTimeLimit = 0.0;
        if(hasTc)
          tcMaxTimeLimit = tcMaxTime.load(std::memory_order_acquire);

        bool shouldStop =
          (numPlayouts >= maxPlayouts) ||
          (numPlayouts + numNonPlayoutVisits >= maxVisits);

        if(hasMaxTime && numPlayouts >= 2 && timeUsed >= maxTime)
          shouldStop = true;
        if(hasTc && numPlayouts >= 2 && timeUsed >= tcMaxTimeLimit)
          shouldStop = true;

        if(shouldStop || shouldStopNow.load(std::memory_order_relaxed)) {
          shouldStopNow.store(true,std::memory_order_relaxed);
          break;
        }

        //Thread 0 alone is responsible for recomputing time limits every once in a while
        //Cap of 10 times per second.
        if(!pondering && (hasTc || hasMaxTime) && threadIdx == 0 && timeUsed >= lastTimeUsedRecomputingTcLimit + 0.1) {
          int64_t rootVisits = numPlayouts + numNonPlayoutVisits;
          double tcLimit = 1e30;
          if(hasTc) {
            tcLimit = recomputeSearchTimeLimit(tc, timeUsed, searchFactor, rootVisits);
            tcMaxTime.store(tcLimit, std::memory_order_release);
          }
          double upperBoundVisits = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, std::min(tcLimit,maxTime));
          upperBoundVisitsLeftDueToTime.store(upperBoundVisits, std::memory_order_release);
        }

        double upperBoundVisitsLeft = 1e30;
        if(hasTc)
          upperBoundVisitsLeft = upperBoundVisitsLeftDueToTime.load(std::memory_order_acquire);
        upperBoundVisitsLeft = std::min(upperBoundVisitsLeft, (double)maxPlayouts - numPlayouts);
        upperBoundVisitsLeft = std::min(upperBoundVisitsLeft, (double)maxVisits - numPlayouts - numNonPlayoutVisits);

        bool finishedPlayout = runSinglePlayout(*stbuf, upperBoundVisitsLeft);
        if(finishedPlayout) {
          numPlayouts = numPlayoutsShared.fetch_add((int64_t)1, std::memory_order_relaxed);
          numPlayouts += 1;
        }
        else {
          //In the case that we didn't finish a playout, give other threads a chance to run before we try again
          //so that it's more likely we become unstuck.
          std::this_thread::yield();
        }
      }
    }
    catch(...) {
      transferOldNNOutputs(*stbuf);
      delete stbuf;
      throw;
    }

    transferOldNNOutputs(*stbuf);
    delete stbuf;
  };

  double actualSearchStartTime = timer.getSeconds();
  performTaskWithThreads(&searchLoop);

  //Relaxed load is fine since numPlayoutsShared should be synchronized already due to the joins
  lastSearchNumPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
  effectiveSearchTimeCarriedOver += timer.getSeconds() - actualSearchStartTime;
}

//If we're being asked to search from a position where the game is over, this is fine. Just keep going, the boardhistory
//should reasonably tolerate just continuing. We do NOT want to clear history because we could inadvertently make a move
//that an external ruleset COULD think violated superko.
void Search::beginSearch(bool pondering) {
  if(rootBoard.x_size > nnXLen || rootBoard.y_size > nnYLen)
    throw StringError("Search got from NNEval nnXLen = " + Global::intToString(nnXLen) +
                      " nnYLen = " + Global::intToString(nnYLen) + " but was asked to search board with larger x or y size");

  rootBoard.checkConsistency();

  numSearchesBegun++;

  //Avoid any issues in principle from rolling over
  if(searchNodeAge > 0x3FFFFFFF)
    clearSearch();

  if(!pondering)
    plaThatSearchIsFor = rootPla;
  //If we begin the game with a ponder, then assume that "we" are the opposing side until we see otherwise.
  if(plaThatSearchIsFor == C_EMPTY)
    plaThatSearchIsFor = getOpp(rootPla);

  //In the case we are doing playoutDoublingAdvantage without a specific player (so, doing the root player)
  //and the player that the search is for changes, we need to clear the tree since we need new evals for the new way around
  if(plaThatSearchIsForLastSearch != plaThatSearchIsFor &&
     searchParams.playoutDoublingAdvantage != 0 &&
     searchParams.playoutDoublingAdvantagePla == C_EMPTY)
    clearSearch();
  plaThatSearchIsForLastSearch = plaThatSearchIsFor;
  //cout << "BEGINSEARCH " << PlayerIO::playerToString(rootPla) << " " << PlayerIO::playerToString(plaThatSearchIsFor) << endl;

  clearOldNNOutputs();
  computeRootValues();
  maybeRecomputeNormToTApproxTable();

  //Prepare value bias table if we need it
  if(searchParams.subtreeValueBiasFactor != 0 && subtreeValueBiasTable == NULL)
    subtreeValueBiasTable = new SubtreeValueBiasTable(searchParams.subtreeValueBiasTableNumShards);

  SearchThread dummyThread(-1, *this);

  if(rootNode == NULL) {
    Loc prevMoveLoc = rootHistory.moveHistory.size() <= 0 ? Board::NULL_LOC : rootHistory.moveHistory[rootHistory.moveHistory.size()-1].loc;
    rootNode = new SearchNode(getOpp(rootPla), prevMoveLoc, NULL);
  }
  else {
    //If the root node has any existing children, then prune things down if there are moves that should not be allowed at the root.
    SearchNode& node = *rootNode;
    int childrenCapacity;
    SearchChildPointer* children = node.getChildren(childrenCapacity);
    if(childrenCapacity > 0 && children != NULL) {

      //This filtering, by deleting children, doesn't conform to the normal invariants that hold during search.
      //However nothing else should be running at this time and the search hasn't actually started yet, so this is okay.
      int numGoodChildren = 0;
      bool anyFiltered = false;
      {
        int i = 0;
        for(; i<childrenCapacity; i++) {
          SearchNode* child = children[i].getIfAllocated();
          if(child == NULL)
            break;
          //Remove the child from its current spot
          children[i].store(NULL);
          //Maybe add it back
          if(isAllowedRootMove(child->prevMoveLoc)) {
            children[numGoodChildren].store(child);
            numGoodChildren++;
          }
          else {
            anyFiltered = true;
            recursivelyRemoveSubtreeValueBiasBeforeDeleteSynchronous(child);
            delete child;
          }
        }
        for(; i<childrenCapacity; i++) {
          SearchNode* child = children[i].getIfAllocated();
          (void)child;
          assert(child == NULL);
        }
      }

      if(anyFiltered) {
        //Fix up the number of visits of the root node after doing this filtering
        int64_t newNumVisits = 0;
        for(int i = 0; i<childrenCapacity; i++) {
          const SearchNode* child = children[i].getIfAllocated();
          if(child == NULL)
            break;
          int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
          newNumVisits += childVisits;
        }

        //Just for cleanliness after filtering - delete the smaller children arrays.
        //They should never be accessed in the upcoming search because all threads
        //spawned will of course be synchronized with any writes we make here,
        //including the current state of the node, so if we've moved on to a
        //higher-capacity array the lower ones will never be accessed.
        if(children == node.children2) {
          delete[] node.children1;
          node.children1 = NULL;
          delete[] node.children0;
          node.children0 = NULL;
        }
        else if(children == node.children1) {
          delete[] node.children0;
          node.children0 = NULL;
        }
        else {
          assert(children == node.children0);
        }

        //For the node's own visit itself
        newNumVisits += 1;

        //Set the visits in place
        while(node.statsLock.test_and_set(std::memory_order_acquire));
        node.stats.visits.store(newNumVisits,std::memory_order_release);
        node.statsLock.clear(std::memory_order_release);

        //Update all other stats
        recomputeNodeStats(node, dummyThread, 0, true);
      }
    }

    //Recursively update all stats in the tree if we have dynamic score values
    //And also to clear out lastResponseBiasDeltaSum and lastResponseBiasWeight
    if(searchParams.dynamicScoreUtilityFactor != 0 || searchParams.subtreeValueBiasFactor != 0) {
      recursivelyRecomputeStats(node,dummyThread,true);
    }
  }

  //Clear unused stuff in value bias table since we may have pruned rootNode stuff
  if(searchParams.subtreeValueBiasFactor != 0 && subtreeValueBiasTable != NULL)
    subtreeValueBiasTable->clearUnusedSynchronous();

  //Mark all nodes old for the purposes of updating old nnoutputs
  searchNodeAge++;
}

//Recursively walk over part of the tree that we are about to delete and remove its contribution to the value bias in the table
//Assumes we aren't doing any multithreadingy stuff, so doesn't bother with locks.
void Search::recursivelyRemoveSubtreeValueBiasBeforeDeleteSynchronous(SearchNode* node) {
  if(node == NULL || searchParams.subtreeValueBiasFactor == 0)
    return;

  int childrenCapacity;
  SearchChildPointer* children = node->getChildren(childrenCapacity);
  for(int i = 0; i<childrenCapacity; i++) {
    SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    recursivelyRemoveSubtreeValueBiasBeforeDeleteSynchronous(child);
  }

  if(node->subtreeValueBiasTableEntry != nullptr) {
    node->subtreeValueBiasTableEntry->deltaUtilitySum -= node->lastSubtreeValueBiasDeltaSum * searchParams.subtreeValueBiasFreeProp;
    node->subtreeValueBiasTableEntry->weightSum -= node->lastSubtreeValueBiasWeight * searchParams.subtreeValueBiasFreeProp;
  }
}


//This function should NOT ever be called concurrently with any other threads modifying the search tree.
//However, it does thread-safely modify things itself, so can safely in theory run concurrently with things
//like ownership computation or analysis that simply read the tree.
void Search::recursivelyRecomputeStats(SearchNode& node, SearchThread& thread, bool isRoot) {
  //First, recompute all children. This function never runs concurrently with anything else
  //but if it did, it might of course only recompute some subset of the children if more were added concurrently.
  bool foundAnyChildren = false;
  int childrenCapacity;
  SearchChildPointer* children = node.getChildren(childrenCapacity);
  int i = 0;
  for(; i<childrenCapacity; i++) {
    SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    recursivelyRecomputeStats(*child,thread,false);
    foundAnyChildren = true;
  }
  for(; i<childrenCapacity; i++) {
    SearchNode* child = children[i].getIfAllocated();
    (void)child;
    assert(child == NULL);
  }

  //If this node has children, it MUST also have an nnOutput.
  if(foundAnyChildren) {
    NNOutput* nnOutput = node.getNNOutput();
    (void)nnOutput; //avoid warning when we have no asserts
    assert(nnOutput != NULL);
  }

  //Also, something is wrong if we have virtual losses at this point
  int32_t numVirtualLosses = node.virtualLosses.load(std::memory_order_acquire);
  (void)numVirtualLosses;
  assert(numVirtualLosses == 0);

  //If the node has no children, then just update its utility directly
  //Again, this would be a little wrong if this function were running concurrently with anything else in the
  //case that new children were added in the meantime. Although maybe it would be okay.
  if(!foundAnyChildren) {
    int64_t numVisits = node.stats.visits.load(std::memory_order_acquire);
    double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
    double winLossValueAvg = node.stats.winLossValueAvg.load(std::memory_order_acquire);
    double noResultValueAvg = node.stats.noResultValueAvg.load(std::memory_order_acquire);
    double scoreMeanAvg = node.stats.scoreMeanAvg.load(std::memory_order_acquire);
    double scoreMeanSqAvg = node.stats.scoreMeanSqAvg.load(std::memory_order_acquire);

    //It's possible that this node has 0 weight in the case where it's the root node
    //and has 0 visits because we began a search and then stopped it before any playouts happened.
    //In that case, there's not much to recompute.
    if(weightSum <= 0.0) {
      assert(numVisits == 0);
      assert(isRoot);
    }
    else {
      double resultUtility = getResultUtility(winLossValueAvg, noResultValueAvg);
      double scoreUtility = getScoreUtility(scoreMeanAvg, scoreMeanSqAvg);
      double newUtilityAvg = resultUtility + scoreUtility;
      double newUtilitySqAvg = newUtilityAvg * newUtilityAvg;

      while(node.statsLock.test_and_set(std::memory_order_acquire));
      node.stats.utilityAvg.store(newUtilityAvg,std::memory_order_release);
      node.stats.utilitySqAvg.store(newUtilitySqAvg,std::memory_order_release);
      node.statsLock.clear(std::memory_order_release);
    }
  }
  else {
    //Otherwise recompute it using the usual method
    recomputeNodeStats(node, thread, 0, isRoot);
  }
}

void Search::computeRootNNEvaluation(NNResultBuf& nnResultBuf, bool includeOwnerMap) {
  Board board = rootBoard;
  const BoardHistory& hist = rootHistory;
  Player pla = rootPla;
  bool skipCache = false;
  // bool isRoot = true;
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
  nnInputParams.conservativePass = searchParams.conservativePass;
  nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
  nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == pla;
  if(searchParams.playoutDoublingAdvantage != 0) {
    Player playoutDoublingAdvantagePla = getPlayoutDoublingAdvantagePla();
    nnInputParams.playoutDoublingAdvantage = (
      getOpp(pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
    );
  }
  nnEvaluator->evaluate(
    board, hist, pla,
    nnInputParams,
    nnResultBuf, skipCache, includeOwnerMap
  );
}

void Search::computeRootValues() {
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

  //Figure out how to set recentScoreCenter
  {
    bool foundExpectedScoreFromTree = false;
    double expectedScore = 0.0;
    if(rootNode != NULL) {
      const SearchNode& node = *rootNode;
      int64_t numVisits = node.stats.visits.load(std::memory_order_acquire);
      double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
      double scoreMeanAvg = node.stats.scoreMeanAvg.load(std::memory_order_acquire);
      if(numVisits > 0 && weightSum > 0) {
        foundExpectedScoreFromTree = true;
        expectedScore = scoreMeanAvg;
      }
    }

    //Grab a neural net evaluation for the current position and use that as the center
    if(!foundExpectedScoreFromTree) {
      NNResultBuf nnResultBuf;
      bool includeOwnerMap = true;
      computeRootNNEvaluation(nnResultBuf,includeOwnerMap);
      expectedScore = nnResultBuf.result->whiteScoreMean;
    }

    recentScoreCenter = expectedScore * (1.0 - searchParams.dynamicScoreCenterZeroWeight);
    double cap =  sqrt(rootBoard.x_size * rootBoard.y_size) * searchParams.dynamicScoreCenterScale;
    if(recentScoreCenter > expectedScore + cap)
      recentScoreCenter = expectedScore + cap;
    if(recentScoreCenter < expectedScore - cap)
      recentScoreCenter = expectedScore - cap;
  }

  Player opponentWasMirroringPla = mirroringPla;
  mirroringPla = C_EMPTY;
  mirrorAdvantage = 0.0;
  mirrorCenterIsSymmetric = false;
  if(searchParams.antiMirror) {
    const Board& board = rootBoard;
    const BoardHistory& hist = rootHistory;
    int mirrorCount = 0;
    int totalCount = 0;
    double mirrorEwms = 0;
    double totalEwms = 0;
    bool lastWasMirror = false;
    for(int i = 1; i<hist.moveHistory.size(); i++) {
      if(hist.moveHistory[i].pla != rootPla) {
        lastWasMirror = false;
        if(hist.moveHistory[i].loc == Location::getMirrorLoc(hist.moveHistory[i-1].loc,board.x_size,board.y_size)) {
          mirrorCount += 1;
          mirrorEwms += 1;
          lastWasMirror = true;
        }
        totalCount += 1;
        totalEwms += 1;
        mirrorEwms *= 0.75;
        totalEwms *= 0.75;
      }
    }
    //If at most of the moves in the game are mirror moves, and many of the recent moves were mirrors, and the last move
    //was a mirror, then the opponent is mirroring.
    if(mirrorCount >= 7.0 + 0.5 * totalCount && mirrorEwms >= 0.45 * totalEwms && lastWasMirror) {
      mirroringPla = getOpp(rootPla);

      double blackExtraPoints = 0.0;
      int numHandicapStones = hist.computeNumHandicapStones();
      if(hist.rules.scoringRule == Rules::SCORING_AREA) {
        if(numHandicapStones > 0)
          blackExtraPoints += numHandicapStones-1;
        bool blackGetsLastMove = (board.x_size % 2 == 1 && board.y_size % 2 == 1) == (numHandicapStones == 0 || numHandicapStones % 2 == 1);
        if(blackGetsLastMove)
          blackExtraPoints += 1;
      }
      if(numHandicapStones > 0 && hist.rules.whiteHandicapBonusRule == Rules::WHB_N)
        blackExtraPoints -= numHandicapStones;
      if(numHandicapStones > 0 && hist.rules.whiteHandicapBonusRule == Rules::WHB_N_MINUS_ONE)
        blackExtraPoints -= numHandicapStones-1;
      mirrorAdvantage = mirroringPla == P_BLACK ? blackExtraPoints - hist.rules.komi : hist.rules.komi - blackExtraPoints;
    }

    if(board.x_size >= 7 && board.y_size >= 7) {
      mirrorCenterIsSymmetric = true;
      int halfX = board.x_size / 2;
      int halfY = board.y_size / 2;
      for(int dy = -3; dy <= 3; dy++) {
        for(int dx = -3; dx <= 3; dx++) {
          Loc loc = Location::getLoc(halfX+dx,halfY+dy,board.x_size);
          Loc mirrorLoc = Location::getMirrorLoc(loc,board.x_size,board.y_size);
          if(loc == mirrorLoc)
            continue;
          Color c = board.colors[mirrorLoc] != C_EMPTY ? getOpp(board.colors[mirrorLoc]) : C_EMPTY;
          if(board.colors[loc] != c)
            mirrorCenterIsSymmetric = false;
        }
      }
    }
  }
  //Clear search if opponent mirror status changed, so that our tree adjusts appropriately
  if(opponentWasMirroringPla != mirroringPla)
    clearSearch();
}

int64_t Search::getRootVisits() const {
  if(rootNode == NULL)
    return 0;
  int64_t n = rootNode->stats.visits.load(std::memory_order_acquire);
  return n;
}

struct PolicySortEntry {
  float policy;
  int pos;
  PolicySortEntry() {}
  PolicySortEntry(float x, int y): policy(x), pos(y) {}
  bool operator<(const PolicySortEntry& other) const {
    return policy > other.policy || (policy == other.policy && pos < other.pos);
  }
  bool operator==(const PolicySortEntry& other) const {
    return policy == other.policy && pos == other.pos;
  }
};

//Finds the top n moves, or fewer if there are fewer than that many total legal moves.
//Returns the number of legal moves found
int Search::findTopNPolicy(const SearchNode* node, int n, PolicySortEntry* sortedPolicyBuf) const {
  const std::shared_ptr<NNOutput>* nnOutput = node->nnOutput.load(std::memory_order_release);
  if(nnOutput == NULL)
    return 0;
  const float* policyProbs = (*nnOutput)->policyProbs;

  int numLegalMovesFound = 0;
  for(int pos = 0; pos<policySize; pos++) {
    if(policyProbs[pos] >= 0.0f) {
      sortedPolicyBuf[numLegalMovesFound++] = PolicySortEntry(policyProbs[pos],pos);
    }
  }
  int numMovesToReturn = std::min(n,numLegalMovesFound);
  std::partial_sort(sortedPolicyBuf,sortedPolicyBuf+numMovesToReturn,sortedPolicyBuf+numLegalMovesFound);
  return numMovesToReturn;
}

void Search::computeDirichletAlphaDistribution(int policySize, const float* policyProbs, double* alphaDistr) {
  int legalCount = 0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0)
      legalCount += 1;
  }

  if(legalCount <= 0)
    throw StringError("computeDirichletAlphaDistribution: No move with nonnegative policy value - can't even pass?");

  //We're going to generate a gamma draw on each move with alphas that sum up to searchParams.rootDirichletNoiseTotalConcentration.
  //Half of the alpha weight are uniform.
  //The other half are shaped based on the log of the existing policy.
  double logPolicySum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = log(std::min(0.01, (double)policyProbs[i]) + 1e-20);
      logPolicySum += alphaDistr[i];
    }
  }
  double logPolicyMean = logPolicySum / legalCount;
  double alphaPropSum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = std::max(0.0, alphaDistr[i] - logPolicyMean);
      alphaPropSum += alphaDistr[i];
    }
  }
  double uniformProb = 1.0 / legalCount;
  if(alphaPropSum <= 0.0) {
    for(int i = 0; i<policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = uniformProb;
    }
  }
  else {
    for(int i = 0; i<policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = 0.5 * (alphaDistr[i] / alphaPropSum + uniformProb);
    }
  }
}

void Search::addDirichletNoise(const SearchParams& searchParams, Rand& rand, int policySize, float* policyProbs) {
  double r[NNPos::MAX_NN_POLICY_SIZE];
  Search::computeDirichletAlphaDistribution(policySize, policyProbs, r);

  //r now contains the proportions with which we would like to split the alpha
  //The total of the alphas is searchParams.rootDirichletNoiseTotalConcentration
  //Generate gamma draw on each move
  double rSum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      r[i] = rand.nextGamma(r[i] * searchParams.rootDirichletNoiseTotalConcentration);
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
    if(policyProbs[i] >= 0) {
      double weight = searchParams.rootDirichletNoiseWeight;
      policyProbs[i] = (float)(r[i] * weight + policyProbs[i] * (1.0-weight));
    }
  }
}


shared_ptr<NNOutput>* Search::maybeAddPolicyNoiseAndTemp(SearchThread& thread, bool isRoot, NNOutput* oldNNOutput) const {
  if(!isRoot)
    return NULL;
  if(!searchParams.rootNoiseEnabled && searchParams.rootPolicyTemperature == 1.0 && searchParams.rootPolicyTemperatureEarly == 1.0 && rootHintLoc == Board::NULL_LOC)
    return NULL;
  if(oldNNOutput == NULL)
    return NULL;
  if(oldNNOutput->noisedPolicyProbs != NULL)
    return NULL;

  //Copy nnOutput as we're about to modify its policy to add noise or temperature
  shared_ptr<NNOutput>* newNNOutputSharedPtr = new shared_ptr<NNOutput>(new NNOutput(*oldNNOutput));
  NNOutput* newNNOutput = newNNOutputSharedPtr->get();

  float* noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
  newNNOutput->noisedPolicyProbs = noisedPolicyProbs;
  std::copy(newNNOutput->policyProbs, newNNOutput->policyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);

  if(searchParams.rootPolicyTemperature != 1.0 || searchParams.rootPolicyTemperatureEarly != 1.0) {
    double rootPolicyTemperature = interpolateEarly(
      searchParams.chosenMoveTemperatureHalflife, searchParams.rootPolicyTemperatureEarly, searchParams.rootPolicyTemperature
    );

    double maxValue = 0.0;
    for(int i = 0; i<policySize; i++) {
      double prob = noisedPolicyProbs[i];
      if(prob > maxValue)
        maxValue = prob;
    }
    assert(maxValue > 0.0);

    double logMaxValue = log(maxValue);
    double invTemp = 1.0 / rootPolicyTemperature;
    double sum = 0.0;

    for(int i = 0; i<policySize; i++) {
      if(noisedPolicyProbs[i] > 0) {
        //Numerically stable way to raise to power and normalize
        float p = (float)exp((log((double)noisedPolicyProbs[i]) - logMaxValue) * invTemp);
        noisedPolicyProbs[i] = p;
        sum += p;
      }
    }
    assert(sum > 0.0);
    for(int i = 0; i<policySize; i++) {
      if(noisedPolicyProbs[i] >= 0) {
        noisedPolicyProbs[i] = (float)(noisedPolicyProbs[i] / sum);
      }
    }
  }

  if(searchParams.rootNoiseEnabled) {
    addDirichletNoise(searchParams, thread.rand, policySize, noisedPolicyProbs);
  }

  //Move a small amount of policy to the hint move, around the same level that noising it would achieve
  if(rootHintLoc != Board::NULL_LOC) {
    const float propToMove = 0.02f;
    int pos = getPos(rootHintLoc);
    if(noisedPolicyProbs[pos] >= 0) {
      double amountToMove = 0.0;
      for(int i = 0; i<policySize; i++) {
        if(noisedPolicyProbs[i] >= 0) {
          amountToMove += noisedPolicyProbs[i] * propToMove;
          noisedPolicyProbs[i] *= (1.0f-propToMove);
        }
      }
      noisedPolicyProbs[pos] += (float)amountToMove;
    }
  }

  return newNNOutputSharedPtr;
}

bool Search::isAllowedRootMove(Loc moveLoc) const {
  assert(moveLoc == Board::PASS_LOC || rootBoard.isOnBoard(moveLoc));

  //A bad situation that can happen that unnecessarily prolongs training games is where one player
  //repeatedly passes and the other side repeatedly fills the opponent's space and/or suicides over and over.
  //To mitigate some of this and save computation, we make it so that at the root, if the last four moves by the opponent
  //were passes, we will never play a move in either player's pass-alive area. In theory this could prune
  //a good move in situations like https://senseis.xmp.net/?1EyeFlaw, but this should be extraordinarly rare,
  if(searchParams.rootPruneUselessMoves &&
     rootHistory.moveHistory.size() > 0 &&
     moveLoc != Board::PASS_LOC
  ) {
    size_t lastIdx = rootHistory.moveHistory.size()-1;
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

void Search::downweightBadChildrenAndNormalizeWeight(
  int numChildren,
  double currentTotalWeight,
  double desiredTotalWeight,
  double amountToSubtract,
  double amountToPrune,
  vector<MoreNodeStats>& statsBuf
) const {
  if(numChildren <= 0 || currentTotalWeight <= 0.0)
    return;

  if(searchParams.valueWeightExponent == 0 || mirroringPla != C_EMPTY) {
    for(int i = 0; i<numChildren; i++) {
      if(statsBuf[i].weightAdjusted < amountToPrune) {
        currentTotalWeight -= statsBuf[i].weightAdjusted;
        statsBuf[i].weightAdjusted = 0.0;
        continue;
      }
      double newWeight = statsBuf[i].weightAdjusted - amountToSubtract;
      if(newWeight <= 0) {
        currentTotalWeight -= statsBuf[i].weightAdjusted;
        statsBuf[i].weightAdjusted = 0.0;
      }
      else {
        currentTotalWeight -= amountToSubtract;
        statsBuf[i].weightAdjusted = newWeight;
      }
    }

    if(currentTotalWeight != desiredTotalWeight) {
      double factor = desiredTotalWeight / currentTotalWeight;
      for(int i = 0; i<numChildren; i++)
        statsBuf[i].weightAdjusted *= factor;
    }
    return;
  }

  assert(numChildren <= NNPos::MAX_NN_POLICY_SIZE);
  double stdevs[NNPos::MAX_NN_POLICY_SIZE];
  double simpleValueSum = 0.0;
  for(int i = 0; i<numChildren; i++) {
    int64_t numVisits = statsBuf[i].stats.visits;
    assert(numVisits >= 0);
    if(numVisits == 0)
      continue;

    double weight = statsBuf[i].weightAdjusted;
    double precision = 1.5 * sqrt(weight);

    //Ensure some minimum variance for stability regardless of how we change the above formula
    static const double minVariance = 0.00000001;
    stdevs[i] = sqrt(minVariance + 1.0 / precision);
    simpleValueSum += statsBuf[i].selfUtility * weight;
  }

  double simpleValue = simpleValueSum / currentTotalWeight;

  double totalNewUnnormWeight = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(statsBuf[i].stats.visits == 0)
      continue;

    if(statsBuf[i].weightAdjusted < amountToPrune) {
      currentTotalWeight -= statsBuf[i].weightAdjusted;
      statsBuf[i].weightAdjusted = 0.0;
      continue;
    }
    double newWeight = statsBuf[i].weightAdjusted - amountToSubtract;
    if(newWeight <= 0) {
      currentTotalWeight -= statsBuf[i].weightAdjusted;
      statsBuf[i].weightAdjusted = 0.0;
    }
    else {
      currentTotalWeight -= amountToSubtract;
      statsBuf[i].weightAdjusted = newWeight;
    }

    double z = (statsBuf[i].selfUtility - simpleValue) / stdevs[i];
    //Also just for numeric sanity, make sure everything has some tiny minimum value.
    double p = valueWeightDistribution->getCdf(z) + 0.0001;
    statsBuf[i].weightAdjusted *= pow(p, searchParams.valueWeightExponent);
    totalNewUnnormWeight += statsBuf[i].weightAdjusted;
  }

  //Post-process and normalize to sum to the desired weight
  assert(totalNewUnnormWeight > 0.0);
  double factor = desiredTotalWeight / totalNewUnnormWeight;
  for(int i = 0; i<numChildren; i++)
    statsBuf[i].weightAdjusted *= factor;
}

static double cpuctExploration(double totalChildWeight, const SearchParams& searchParams) {
  return searchParams.cpuctExploration +
    searchParams.cpuctExplorationLog * log((totalChildWeight + searchParams.cpuctExplorationBase) / searchParams.cpuctExplorationBase);
}

//Tiny constant to add to numerator of puct formula to make it positive
//even when visis = 0.
static constexpr double TOTALCHILDWEIGHT_PUCT_OFFSET = 0.01;

double Search::getExploreSelectionValue(
  double nnPolicyProb, double totalChildWeight, double childWeight,
  double childUtility, double parentUtilityStdevFactor, Player pla
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  double exploreComponent =
    cpuctExploration(totalChildWeight,searchParams)
    * parentUtilityStdevFactor
    * nnPolicyProb
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET)
    / (1.0 + childWeight);

  //At the last moment, adjust value to be from the player's perspective, so that players prefer values in their favor
  //rather than in white's favor
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;
  return exploreComponent + valueComponent;
}

//Return the childWeight that would make Search::getExploreSelectionValue return the given explore selection value.
//Or return 0, if it would be less than 0.
double Search::getExploreSelectionValueInverse(
  double exploreSelectionValue, double nnPolicyProb, double totalChildWeight,
  double childUtility, double parentUtilityStdevFactor, Player pla
) const {
  if(nnPolicyProb < 0)
    return 0;
  double valueComponent = pla == P_WHITE ? childUtility : -childUtility;

  double exploreComponent = exploreSelectionValue - valueComponent;
  double exploreComponentScaling =
    cpuctExploration(totalChildWeight,searchParams)
    * parentUtilityStdevFactor
    * nnPolicyProb
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET);

  //Guard against float weirdness
  if(exploreComponent <= 0)
    return 1e100;

  double childWeight = exploreComponentScaling / exploreComponent - 1;
  if(childWeight < 0)
    childWeight = 0;
  return childWeight;
}


double Search::getEndingWhiteScoreBonus(const SearchNode& parent, const SearchNode* child) const {
  if(&parent != rootNode || child->prevMoveLoc == Board::NULL_LOC)
    return 0.0;

  const NNOutput* nnOutput = parent.getNNOutput();
  if(nnOutput == NULL || nnOutput->whiteOwnerMap == NULL)
    return 0.0;

  bool isAreaIsh = rootHistory.rules.scoringRule == Rules::SCORING_AREA
    || (rootHistory.rules.scoringRule == Rules::SCORING_TERRITORY && rootHistory.encorePhase >= 2);
  assert(nnOutput->nnXLen == nnXLen);
  assert(nnOutput->nnYLen == nnYLen);
  float* whiteOwnerMap = nnOutput->whiteOwnerMap;
  Loc moveLoc = child->prevMoveLoc;

  const double extreme = 0.95;
  const double tail = 0.05;

  //Extra points from the perspective of the root player
  double extraRootPoints = 0.0;
  if(isAreaIsh) {
    //Areaish scoring - in an effort to keep the game short and slightly discourage pointless territory filling at the end
    //discourage any move that, except in case of ko, is either:
    // * On a spot that the opponent almost surely owns
    // * On a spot that the player almost surely owns and it is not adjacent to opponent stones and is not a connection of non-pass-alive groups.
    //These conditions should still make it so that "cleanup" and dame-filling moves are not discouraged.
    // * When playing button go, very slightly discourage passing - so that if there are an even number of dame, filling a dame is still favored over passing.
    if(moveLoc != Board::PASS_LOC && rootBoard.ko_loc == Board::NULL_LOC) {
      int pos = NNPos::locToPos(moveLoc,rootBoard.x_size,nnXLen,nnYLen);
      double plaOwnership = rootPla == P_WHITE ? whiteOwnerMap[pos] : -whiteOwnerMap[pos];
      if(plaOwnership <= -extreme)
        extraRootPoints -= searchParams.rootEndingBonusPoints * ((-extreme - plaOwnership) / tail);
      else if(plaOwnership >= extreme) {
        if(!rootBoard.isAdjacentToPla(moveLoc,getOpp(rootPla)) &&
           !rootBoard.isNonPassAliveSelfConnection(moveLoc,rootPla,rootSafeArea)) {
          extraRootPoints -= searchParams.rootEndingBonusPoints * ((plaOwnership - extreme) / tail);
        }
      }
    }
    if(moveLoc == Board::PASS_LOC && rootHistory.hasButton) {
      extraRootPoints -= searchParams.rootEndingBonusPoints * 0.5;
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
      int pos = NNPos::locToPos(moveLoc,rootBoard.x_size,nnXLen,nnYLen);
      double plaOwnership = rootPla == P_WHITE ? whiteOwnerMap[pos] : -whiteOwnerMap[pos];
      if(plaOwnership <= -extreme)
        extraRootPoints -= searchParams.rootEndingBonusPoints * ((-extreme - plaOwnership) / tail);
      else if(plaOwnership >= extreme) {
        if(!rootBoard.isAdjacentToPla(moveLoc,getOpp(rootPla)) &&
           !rootBoard.isNonPassAliveSelfConnection(moveLoc,rootPla,rootSafeArea)) {
          extraRootPoints -= searchParams.rootEndingBonusPoints * ((plaOwnership - extreme) / tail);
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
  return NNPos::locToPos(moveLoc,rootBoard.x_size,nnXLen,nnYLen);
}

static void maybeApplyWideRootNoise(
  double& childUtility,
  float& nnPolicyProb,
  const SearchParams& searchParams,
  SearchThread* thread,
  const SearchNode& parent
) {
  //For very large wideRootNoise, go ahead and also smooth out the policy
  nnPolicyProb = (float)pow(nnPolicyProb, 1.0 / (4.0*searchParams.wideRootNoise + 1.0));
  if(thread->rand.nextBool(0.5)) {
    double bonus = searchParams.wideRootNoise * abs(thread->rand.nextGaussian());
    if(parent.nextPla == P_WHITE)
      childUtility += bonus;
    else
      childUtility -= bonus;
  }
}

static double square(double x) {
  return x * x;
}

static void maybeApplyAntiMirrorPolicy(
  float& nnPolicyProb,
  Loc moveLoc,
  const float* policyProbs,
  Player movePla,
  const SearchThread* thread,
  const Search* search
) {
  int xSize = thread->board.x_size;
  int ySize = thread->board.y_size;
  //Put significant prior probability on the opponent continuing to mirror, at least for the next few turns.
  if(movePla == getOpp(search->rootPla) && thread->history.moveHistory.size() > 0) {
    Loc prevLoc = thread->history.moveHistory[thread->history.moveHistory.size()-1].loc;
    if(prevLoc == Board::PASS_LOC)
      return;
    Loc mirrorLoc = Location::getMirrorLoc(prevLoc,xSize,ySize);
    if(policyProbs[search->getPos(mirrorLoc)] < 0)
      mirrorLoc = Board::PASS_LOC;
    if(moveLoc == mirrorLoc) {
      float weight = (float)(0.5 / (1.0 + sqrt(thread->history.moveHistory.size() - search->rootHistory.moveHistory.size())));
      nnPolicyProb = nnPolicyProb + (1.0f - nnPolicyProb) * weight;
    }
  }
  //Put a small prior on playing the center or attaching to center, bonusing moves that are relatively more likely.
  else if(movePla == search->rootPla && moveLoc != Board::PASS_LOC) {
    if(Location::isCentral(moveLoc,xSize,ySize)) {
      float weight = (float)(1.0/square(1.0-log10(nnPolicyProb+1e-30)));
      nnPolicyProb = nnPolicyProb + (1.0f - nnPolicyProb) * weight;
    }
    else {
      Loc centerLoc = Location::getCenterLoc(xSize,ySize);
      if(centerLoc != Board::NULL_LOC) {
        if(search->rootBoard.colors[centerLoc] == getOpp(movePla)) {
          if(thread->board.isAdjacentToChain(moveLoc,centerLoc) || Location::euclideanDistanceSquared(moveLoc,centerLoc,xSize) <= 2) {
            float weight = (float)(1.0/square(1.0-log10(nnPolicyProb+1e-30)));
            nnPolicyProb = nnPolicyProb + (1.0f - nnPolicyProb) * weight;
          }
        }
      }
    }
  }
}

//Force the search to dump playouts down a mirror move, so as to encourage moves that cause mirror moves
//to have bad values, and also tolerate us playing certain countering moves even if their values are a bit worse.
static void maybeApplyAntiMirrorForcedExplore(
  double& childUtility,
  Loc moveLoc,
  const float* policyProbs,
  double thisChildWeight,
  double totalChildWeight,
  Player movePla,
  SearchThread* thread,
  const Search* search,
  const SearchNode& parent
) {
  Player mirroringPla = search->mirroringPla;
  assert(mirroringPla == getOpp(search->rootPla));

  int xSize = thread->board.x_size;
  int ySize = thread->board.y_size;
  Loc centerLoc = Location::getCenterLoc(xSize,ySize);
  //The difficult case is when the opponent has occupied tengen, and ALSO the komi favors them.
  //In such a case, we're going to have a hard time.
  //Technically there are other configurations (like if the opponent makes a diamond around tengen)
  //but we're not going to worry about breaking that.
  bool isDifficult = centerLoc != Board::NULL_LOC && thread->board.colors[centerLoc] == search->mirroringPla && search->mirrorAdvantage >= 0.0;
  bool isSemiDifficult = !isDifficult && search->mirrorAdvantage >= 6.5;

  //Force mirroring pla to dump playouts down mirror moves
  if(movePla == mirroringPla && thread->history.moveHistory.size() > 0) {
    Loc prevLoc = thread->history.moveHistory[thread->history.moveHistory.size()-1].loc;
    if(prevLoc == Board::PASS_LOC)
      return;
    Loc mirrorLoc = Location::getMirrorLoc(prevLoc,xSize,ySize);
    if(policyProbs[search->getPos(mirrorLoc)] < 0)
      mirrorLoc = Board::PASS_LOC;
    if(moveLoc == mirrorLoc) {
      //Check that the player has also been mirroring since the start of search
      for(size_t i = search->rootHistory.moveHistory.size()+1; i < thread->history.moveHistory.size(); i += 2) {
        if(thread->history.moveHistory[i].loc != Location::getMirrorLoc(thread->history.moveHistory[i-1].loc,xSize,ySize))
          return;
      }

      double bonus = 0.02;
      if(isDifficult) {
        if(mirrorLoc != Board::PASS_LOC && search->mirrorCenterIsSymmetric) {
          double factor = 0.75 + 0.5 * sqrt(Location::euclideanDistanceSquared(centerLoc,mirrorLoc,xSize));
          if(thisChildWeight * factor < totalChildWeight && mirrorLoc != Board::PASS_LOC) {
            bonus = 1.0;
          }
        }
        if(thisChildWeight * 5 < totalChildWeight)
          bonus = 1.0;
      }
      else if(isSemiDifficult && search->mirrorAdvantage >= 8.5) {
        if(thisChildWeight * 5 < totalChildWeight)
          bonus = 1.0;
      }
      else if(isSemiDifficult) {
        if(thisChildWeight * 8 < totalChildWeight)
          bonus = 1.0;
      }
      else {
        if(thisChildWeight * 20 < totalChildWeight)
          bonus = 0.2;
      }
      bonus *= (float)(2.0 / (1.0 + sqrt(thread->history.moveHistory.size() - search->rootHistory.moveHistory.size())));
      childUtility += (parent.nextPla == P_WHITE ? bonus : -bonus);
    }
  }
  //Encourage us to find refuting moves, even if they look a little bad, in the difficult case
  else if(movePla == search->rootPla && moveLoc != Board::PASS_LOC) {
    if(isDifficult && thread->board.isAdjacentToChain(moveLoc,centerLoc))
      childUtility += (parent.nextPla == P_WHITE ? 0.10 : -0.10);
  }
}


double Search::getExploreSelectionValue(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  double totalChildWeight, double fpuValue,
  double parentUtility, double parentWeightPerVisit, double parentUtilityStdevFactor,
  bool isDuringSearch, double maxChildWeight, SearchThread* thread
) const {
  (void)parentUtility;
  Loc moveLoc = child->prevMoveLoc;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parentPolicyProbs[movePos];

  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double childWeight = child->stats.weightSum.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  int32_t childVirtualLosses = child->virtualLosses.load(std::memory_order_acquire);

  //It's possible that childVisits is actually 0 here with multithreading because we're visiting this node while a child has
  //been expanded but its thread not yet finished its first visit.
  //It's also possible that we observe childWeight <= 0 even though childVisits >= due to multithreading, the two could
  //be out of sync briefly since they are separate atomics.
  double childUtility;
  if(childVisits <= 0 || childWeight <= 0.0)
    childUtility = fpuValue;
  else {
    childUtility = utilityAvg;

    //Tiny adjustment for passing
    double endingScoreBonus = getEndingWhiteScoreBonus(parent,child);
    if(endingScoreBonus != 0)
      childUtility += getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);
  }

  //When multithreading, totalChildWeight could be out of sync with childWeight, so if they provably are, then fix that up
  if(totalChildWeight < childWeight)
    totalChildWeight = childWeight;

  //Virtual losses to direct threads down different paths
  if(childVirtualLosses > 0) {
    double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;
    childWeight += virtualLossWeight;

    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;
    double virtualLossUtility = (parent.nextPla == P_WHITE ? -utilityRadius : utilityRadius);
    double virtualLossWeightFrac = (double)virtualLossWeight / childWeight;
    childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossWeightFrac;
  }

  if(isDuringSearch && (&parent == rootNode)) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small, assuming
    //its average weight per visit is maintained.
    if(searchParams.futileVisitsThreshold > 0) {
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = (childVisits + 1.0) / (childWeight + parentWeightPerVisit);
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(childVisits + thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    //Hack to get the root to funnel more visits down child branches
    if(searchParams.rootDesiredPerChildVisitsCoeff > 0.0) {
      if(childWeight < sqrt(nnPolicyProb * totalChildWeight * searchParams.rootDesiredPerChildVisitsCoeff)) {
        return 1e20;
      }
    }
    //Hack for hintloc - must search this move almost as often as the most searched move
    if(rootHintLoc != Board::NULL_LOC && moveLoc == rootHintLoc) {
      double averageWeightPerVisit = (childWeight + parentWeightPerVisit) / (childVisits + 1.0);
      int childrenCapacity;
      const SearchChildPointer* children = parent.getChildren(childrenCapacity);
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchNode* c = children[i].getIfAllocated();
        if(c == NULL)
          break;
        double cWeight = c->stats.weightSum.load(std::memory_order_acquire);
        if(childWeight + averageWeightPerVisit < cWeight * 0.8)
          return 1e20;
      }
    }

    if(searchParams.wideRootNoise > 0.0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }
  if(isDuringSearch && searchParams.antiMirror && mirroringPla != C_EMPTY) {
    maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, parentPolicyProbs, parent.nextPla, thread, this);
    maybeApplyAntiMirrorForcedExplore(childUtility, moveLoc, parentPolicyProbs, childWeight, totalChildWeight, parent.nextPla, thread, this, parent);
  }

  return getExploreSelectionValue(nnPolicyProb,totalChildWeight,childWeight,childUtility,parentUtilityStdevFactor,parent.nextPla);
}

double Search::getNewExploreSelectionValue(
  const SearchNode& parent, float nnPolicyProb,
  double totalChildWeight, double fpuValue,
  double parentWeightPerVisit, double parentUtilityStdevFactor,
  double maxChildWeight, SearchThread* thread
) const {
  double childWeight = 0;
  double childUtility = fpuValue;
  if(&parent == rootNode) {
    //Futile visits pruning - skip this move if the amount of time we have left to search is too small
    if(searchParams.futileVisitsThreshold > 0) {
      //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
      double averageVisitsPerWeight = 1.0 / parentWeightPerVisit;
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
    if(searchParams.wideRootNoise > 0.0) {
      maybeApplyWideRootNoise(childUtility, nnPolicyProb, searchParams, thread, parent);
    }
  }
  return getExploreSelectionValue(nnPolicyProb,totalChildWeight,childWeight,childUtility,parentUtilityStdevFactor,parent.nextPla);
}

double Search::getReducedPlaySelectionWeight(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  double totalChildWeight, double parentUtilityStdevFactor, double bestChildExploreSelectionValue
) const {
  assert(&parent == rootNode);
  Loc moveLoc = child->prevMoveLoc;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parentPolicyProbs[movePos];

  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double childWeight = child->stats.weightSum.load(std::memory_order_acquire);
  double scoreMeanAvg = child->stats.scoreMeanAvg.load(std::memory_order_acquire);
  double scoreMeanSqAvg = child->stats.scoreMeanSqAvg.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);

  //Child visits may be 0 if this function is called in a multithreaded context, such as during live analysis
  //Child weight may also be 0 if it's out of sync.
  if(childVisits <= 0 || childWeight <= 0.0)
    return 0;

  //Tiny adjustment for passing
  double endingScoreBonus = getEndingWhiteScoreBonus(parent,child);
  double childUtility = utilityAvg;
  if(endingScoreBonus != 0)
    childUtility += getScoreUtilityDiff(scoreMeanAvg, scoreMeanSqAvg, endingScoreBonus);

  double childWeightWeRetrospectivelyWanted = getExploreSelectionValueInverse(
    bestChildExploreSelectionValue, nnPolicyProb, totalChildWeight, childUtility, parentUtilityStdevFactor, parent.nextPla
  );
  if(childWeight > childWeightWeRetrospectivelyWanted)
    return childWeightWeRetrospectivelyWanted;
  return childWeight;
}

double Search::getFpuValueForChildrenAssumeVisited(
  const SearchNode& node, Player pla, bool isRoot, double policyProbMassVisited,
  double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
) const {
  int64_t visits = node.stats.visits.load(std::memory_order_acquire);
  double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
  double utilityAvg = node.stats.utilityAvg.load(std::memory_order_acquire);
  double utilitySqAvg = node.stats.utilitySqAvg.load(std::memory_order_acquire);

  assert(visits > 0);
  assert(weightSum > 0.0);
  parentWeightPerVisit = weightSum / visits;
  parentUtility = utilityAvg;
  double variancePrior = searchParams.cpuctUtilityStdevPrior * searchParams.cpuctUtilityStdevPrior;
  double variancePriorWeight = searchParams.cpuctUtilityStdevPriorWeight;
  double parentUtilityStdev;
  if(visits <= 0 || weightSum <= 1)
    parentUtilityStdev = searchParams.cpuctUtilityStdevPrior;
  else {
    double utilitySq = parentUtility * parentUtility;
    //Make sure we're robust to numerical precision issues or threading desync of these values, so we don't observe negative variance
    if(utilitySqAvg < utilitySq)
      utilitySqAvg = utilitySq;
    parentUtilityStdev = sqrt(
      std::max(
        0.0,
        ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * weightSum)
        / (variancePriorWeight + weightSum - 1.0)
        - utilitySq
      )
    );
  }
  parentUtilityStdevFactor = 1.0 + searchParams.cpuctUtilityStdevScale * (parentUtilityStdev / searchParams.cpuctUtilityStdevPrior - 1.0);

  if(searchParams.fpuParentWeight > 0.0) {
    parentUtility = searchParams.fpuParentWeight * getUtilityFromNN(*(node.getNNOutput())) + (1.0 - searchParams.fpuParentWeight) * parentUtility;
  }

  double fpuValue;
  {
    double fpuReductionMax = isRoot ? searchParams.rootFpuReductionMax : searchParams.fpuReductionMax;
    double fpuLossProp = isRoot ? searchParams.rootFpuLossProp : searchParams.fpuLossProp;
    double utilityRadius = searchParams.winLossUtilityFactor + searchParams.staticScoreUtilityFactor + searchParams.dynamicScoreUtilityFactor;

    double reduction = fpuReductionMax * sqrt(policyProbMassVisited);
    fpuValue = pla == P_WHITE ? parentUtility - reduction : parentUtility + reduction;
    double lossValue = pla == P_WHITE ? -utilityRadius : utilityRadius;
    fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
  }

  return fpuValue;
}


void Search::selectBestChildToDescend(
  SearchThread& thread, const SearchNode& node, int nodeState,
  int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot) const
{
  assert(thread.pla == node.nextPla);

  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;

  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);

  double policyProbMassVisited = 0.0;
  double maxChildWeight = 0.0;
  double totalChildWeight = 0.0;
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = child->prevMoveLoc;
    int movePos = getPos(moveLoc);
    float nnPolicyProb = policyProbs[movePos];
    policyProbMassVisited += nnPolicyProb;

    double childWeight = child->stats.weightSum.load(std::memory_order_acquire);

    totalChildWeight += childWeight;
    if(childWeight > maxChildWeight)
      maxChildWeight = childWeight;
  }
  //Probability mass should not sum to more than 1, giving a generous allowance
  //for floating point error.
  assert(policyProbMassVisited <= 1.0001);

  //First play urgency
  double parentUtility;
  double parentWeightPerVisit;
  double parentUtilityStdevFactor;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    node, thread.pla, isRoot, policyProbMassVisited,
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  std::fill(posesWithChildBuf,posesWithChildBuf+NNPos::MAX_NN_POLICY_SIZE,false);

  //Try all existing children
  //Also count how many children we actually find
  numChildrenFound = 0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    numChildrenFound++;

    Loc moveLoc = child->prevMoveLoc;
    bool isDuringSearch = true;
    double selectionValue = getExploreSelectionValue(
      node,policyProbs,child,totalChildWeight,fpuValue,
      parentUtility,parentWeightPerVisit,parentUtilityStdevFactor,
      isDuringSearch,maxChildWeight,&thread
    );
    if(selectionValue > maxSelectionValue) {
      // if(child->state.load(std::memory_order_seq_cst) == SearchNode::STATE_EVALUATING) {
      //   selectionValue -= EVALUATING_SELECTION_VALUE_PENALTY;
      //   if(isRoot && child->prevMoveLoc == Location::ofString("K4",thread.board)) {
      //     out << "ouch" << "\n";
      //   }
      // }
      if(selectionValue > maxSelectionValue) {
        maxSelectionValue = selectionValue;
        bestChildIdx = i;
        bestChildMoveLoc = moveLoc;
      }
    }

    posesWithChildBuf[getPos(moveLoc)] = true;
  }

  const std::vector<int>& avoidMoveUntilByLoc = thread.pla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;

  //Try the new child with the best policy value
  Loc bestNewMoveLoc = Board::NULL_LOC;
  float bestNewNNPolicyProb = -1.0f;
  for(int movePos = 0; movePos<policySize; movePos++) {
    bool alreadyTried = posesWithChildBuf[movePos];
    if(alreadyTried)
      continue;

    Loc moveLoc = NNPos::posToLoc(movePos,thread.board.x_size,thread.board.y_size,nnXLen,nnYLen);
    if(moveLoc == Board::NULL_LOC)
      continue;

    //Special logic for the root
    if(isRoot) {
      assert(thread.board.pos_hash == rootBoard.pos_hash);
      assert(thread.pla == rootPla);
      if(!isAllowedRootMove(moveLoc))
        continue;
    }
    if(avoidMoveUntilByLoc.size() > 0) {
      assert(avoidMoveUntilByLoc.size() >= Board::MAX_ARR_SIZE);
      int untilDepth = avoidMoveUntilByLoc[moveLoc];
      if(thread.history.moveHistory.size() - rootHistory.moveHistory.size() < untilDepth)
        continue;
    }

    float nnPolicyProb = policyProbs[movePos];
    if(searchParams.antiMirror && mirroringPla != C_EMPTY) {
      maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, policyProbs, node.nextPla, &thread, this);
    }

    if(nnPolicyProb > bestNewNNPolicyProb) {
      bestNewNNPolicyProb = nnPolicyProb;
      bestNewMoveLoc = moveLoc;
    }
  }
  if(bestNewMoveLoc != Board::NULL_LOC) {
    double selectionValue = getNewExploreSelectionValue(
      node,bestNewNNPolicyProb,totalChildWeight,fpuValue,
      parentWeightPerVisit,parentUtilityStdevFactor,
      maxChildWeight,&thread
    );
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = numChildrenFound;
      bestChildMoveLoc = bestNewMoveLoc;
    }
  }
}

double Search::pruneNoiseWeight(vector<MoreNodeStats>& statsBuf, int numChildren, double totalChildWeight, const double* policyProbsBuf) const {
  if(numChildren <= 1 || totalChildWeight <= 0.00001)
    return totalChildWeight;

  // Children are normally sorted in policy order in KataGo.
  // But this is not guaranteed, because at the root, we might recompute the nnoutput, or when finding the best new child, we have hacks like antiMirror policy
  // and other adjustments. For simplicity, we just consider children in sorted order anyways for this pruning, since it will be close.

  // For any child, if its own utility is lower than the weighted average utility of the children before it, it's downweighted if it exceeds much more than a
  // raw-policy share of the weight.
  double utilitySumSoFar = 0;
  double weightSumSoFar = 0;
  //double rawPolicyUtilitySumSoFar = 0;
  double rawPolicySumSoFar = 0;
  for(int i = 0; i<numChildren; i++) {
    double utility = statsBuf[i].selfUtility;
    double oldWeight = statsBuf[i].weightAdjusted;
    double rawPolicy = policyProbsBuf[i];

    double newWeight = oldWeight;
    if(weightSumSoFar > 0 && rawPolicySumSoFar > 0) {
      double avgUtilitySoFar = utilitySumSoFar / weightSumSoFar;
      double utilityGap = avgUtilitySoFar - utility;
      if(utilityGap > 0) {
        double weightShareFromRawPolicy = weightSumSoFar * rawPolicy / rawPolicySumSoFar;
        //If the child is more than double its proper share of the weight
        double lenientWeightShareFromRawPolicy = 2.0 * weightShareFromRawPolicy;
        if(oldWeight > lenientWeightShareFromRawPolicy) {
          double excessWeight = oldWeight - lenientWeightShareFromRawPolicy;
          double weightToSubtract = excessWeight * (1.0 - exp(-utilityGap / searchParams.noisePruneUtilityScale));
          if(weightToSubtract > searchParams.noisePruningCap)
            weightToSubtract = searchParams.noisePruningCap;

          newWeight = oldWeight - weightToSubtract;
          statsBuf[i].weightAdjusted = newWeight;
        }
      }
    }
    utilitySumSoFar += utility * newWeight;
    weightSumSoFar += newWeight;
    //rawPolicyUtilitySumSoFar += utility * rawPolicy;
    rawPolicySumSoFar += rawPolicy;
  }
  return weightSumSoFar;
}


void Search::updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, bool isRoot) {
  //The thread that grabs a 0 from this peforms the recomputation of stats.
  int32_t oldDirtyCounter = node.dirtyCounter.fetch_add(1,std::memory_order_acq_rel);
  assert(oldDirtyCounter >= 0);
  //If we atomically grab a nonzero, then we know another thread must already be doing the work, so we can skip the update ourselves.
  if(oldDirtyCounter > 0)
    return;
  int32_t numVisitsCompleted = 1;
  while(true) {
    //Perform update
    recomputeNodeStats(node,thread,numVisitsCompleted,isRoot);
    //Now attempt to undo the counter
    oldDirtyCounter = node.dirtyCounter.fetch_add(-numVisitsCompleted,std::memory_order_acq_rel);
    int32_t newDirtyCounter = oldDirtyCounter - numVisitsCompleted;
    //If no other threads incremented it in the meantime, so our decrement hits zero, we're done.
    if(newDirtyCounter <= 0) {
      assert(newDirtyCounter == 0);
      break;
    }
    //Otherwise, more threads incremented this more in the meantime. So we need to loop again and add their visits, recomputing again.
    numVisitsCompleted = newDirtyCounter;
    continue;
  }
}

//Recompute all the stats of this node based on its children, except its visits and virtual losses, which are not child-dependent and
//are updated in the manner specified.
//Assumes this node has an nnOutput
void Search::recomputeNodeStats(SearchNode& node, SearchThread& thread, int numVisitsToAdd, bool isRoot) {
  //Find all children and compute weighting of the children based on their values
  vector<MoreNodeStats>& statsBuf = thread.statsBuf;
  int numGoodChildren = 0;

  int childrenCapacity;
  const SearchChildPointer* children = node.getChildren(childrenCapacity);
  double origTotalChildWeight = 0.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    MoreNodeStats& stats = statsBuf[numGoodChildren];

    stats.stats = NodeStats(child->stats);

    if(stats.stats.visits <= 0 || stats.stats.weightSum <= 0.0)
      continue;

    double childUtility = stats.stats.utilityAvg;
    stats.selfUtility = node.nextPla == P_WHITE ? childUtility : -childUtility;
    stats.weightAdjusted = stats.stats.weightSum;
    stats.prevMoveLoc = child->prevMoveLoc;

    origTotalChildWeight += stats.weightAdjusted;
    numGoodChildren++;
  }

  double currentTotalChildWeight = origTotalChildWeight;
  double desiredTotalChildWeight = origTotalChildWeight;

  if(searchParams.useNoisePruning && numGoodChildren > 0) {
    double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE];
    {
      const NNOutput* nnOutput = node.getNNOutput();
      assert(nnOutput != NULL);
      const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
      for(int i = 0; i<numGoodChildren; i++)
        policyProbsBuf[i] = std::max(1e-30, (double)policyProbs[getPos(statsBuf[i].prevMoveLoc)]);
    }
    currentTotalChildWeight = pruneNoiseWeight(statsBuf, numGoodChildren, currentTotalChildWeight, policyProbsBuf);
    desiredTotalChildWeight = currentTotalChildWeight;
  }

  double amountToSubtract = 0.0;
  double amountToPrune = 0.0;
  if(isRoot && searchParams.rootNoiseEnabled && !searchParams.useNoisePruning) {
    double maxChildWeight = 0.0;
    for(int i = 0; i<numGoodChildren; i++) {
      if(statsBuf[i].weightAdjusted > maxChildWeight)
        maxChildWeight = statsBuf[i].weightAdjusted;
    }
    amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxChildWeight/64.0);
    amountToPrune = std::min(searchParams.chosenMovePrune, maxChildWeight/64.0);
  }

  downweightBadChildrenAndNormalizeWeight(
    numGoodChildren, currentTotalChildWeight, desiredTotalChildWeight,
    amountToSubtract, amountToPrune, statsBuf
  );

  double winLossValueSum = 0.0;
  double noResultValueSum = 0.0;
  double scoreMeanSum = 0.0;
  double scoreMeanSqSum = 0.0;
  double leadSum = 0.0;
  double utilitySum = 0.0;
  double utilitySqSum = 0.0;
  double weightSqSum = 0.0;
  double weightSum = desiredTotalChildWeight;
  for(int i = 0; i<numGoodChildren; i++) {
    const NodeStats& stats = statsBuf[i].stats;

    double desiredWeight = statsBuf[i].weightAdjusted;
    double weightScaling = desiredWeight / stats.weightSum;

    winLossValueSum += desiredWeight * stats.winLossValueAvg;
    noResultValueSum += desiredWeight * stats.noResultValueAvg;
    scoreMeanSum += desiredWeight * stats.scoreMeanAvg;
    scoreMeanSqSum += desiredWeight * stats.scoreMeanSqAvg;
    leadSum += desiredWeight * stats.leadAvg;
    utilitySum += desiredWeight * stats.utilityAvg;
    utilitySqSum += desiredWeight * stats.utilitySqAvg;
    weightSqSum += weightScaling * weightScaling * stats.weightSqSum;
  }

  //Also add in the direct evaluation of this node.
  {
    const NNOutput* nnOutput = node.getNNOutput();
    assert(nnOutput != NULL);
    double winProb = (double)nnOutput->whiteWinProb;
    double lossProb = (double)nnOutput->whiteLossProb;
    double noResultProb = (double)nnOutput->whiteNoResultProb;
    double scoreMean = (double)nnOutput->whiteScoreMean;
    double scoreMeanSq = (double)nnOutput->whiteScoreMeanSq;
    double lead = (double)nnOutput->whiteLead;
    double utility =
      getResultUtility(winProb-lossProb, noResultProb)
      + getScoreUtility(scoreMean, scoreMeanSq);

    if(searchParams.subtreeValueBiasFactor != 0 && node.subtreeValueBiasTableEntry != nullptr) {
      SubtreeValueBiasEntry& entry = *(node.subtreeValueBiasTableEntry);

      double newEntryDeltaUtilitySum;
      double newEntryWeightSum;

      if(desiredTotalChildWeight > 1e-10) {
        double utilityChildren = utilitySum / desiredTotalChildWeight;
        double subtreeValueBiasWeight = pow(origTotalChildWeight, searchParams.subtreeValueBiasWeightExponent);
        double subtreeValueBiasDeltaSum = (utilityChildren - utility) * subtreeValueBiasWeight;

        while(entry.entryLock.test_and_set(std::memory_order_acquire));
        entry.deltaUtilitySum += subtreeValueBiasDeltaSum - node.lastSubtreeValueBiasDeltaSum;
        entry.weightSum += subtreeValueBiasWeight - node.lastSubtreeValueBiasWeight;
        newEntryDeltaUtilitySum = entry.deltaUtilitySum;
        newEntryWeightSum = entry.weightSum;
        node.lastSubtreeValueBiasDeltaSum = subtreeValueBiasDeltaSum;
        node.lastSubtreeValueBiasWeight = subtreeValueBiasWeight;
        entry.entryLock.clear(std::memory_order_release);
      }
      else {
        while(entry.entryLock.test_and_set(std::memory_order_acquire));
        newEntryDeltaUtilitySum = entry.deltaUtilitySum;
        newEntryWeightSum = entry.weightSum;
        entry.entryLock.clear(std::memory_order_release);
      }

      //This is the amount of the direct evaluation of this node that we are going to bias towards the table entry
      const double biasFactor = searchParams.subtreeValueBiasFactor;
      if(newEntryWeightSum > 0.001)
        utility += biasFactor * newEntryDeltaUtilitySum / newEntryWeightSum;
      //This is the amount by which we need to scale desiredSelfWeight such that if the table entry were actually equal to
      //the current difference between the direct eval and the children, we would perform a no-op... unless a noop is actually impossible
      //Then we just take what we can get.
      //desiredSelfWeight *= weightSum / (1.0-biasFactor) / std::max(0.001, (weightSum + desiredSelfWeight - desiredSelfWeight / (1.0-biasFactor)));
    }

    double weight = computeWeightFromNNOutput(nnOutput);
    winLossValueSum += (winProb - lossProb) * weight;
    noResultValueSum += noResultProb * weight;
    scoreMeanSum += scoreMean * weight;
    scoreMeanSqSum += scoreMeanSq * weight;
    leadSum += lead * weight;
    utilitySum += utility * weight;
    utilitySqSum += utility * utility * weight;
    weightSqSum += weight * weight;
    weightSum += weight;
  }

  double winLossValueAvg = winLossValueSum / weightSum;
  double noResultValueAvg = noResultValueSum / weightSum;
  double scoreMeanAvg = scoreMeanSum / weightSum;
  double scoreMeanSqAvg = scoreMeanSqSum / weightSum;
  double leadAvg = leadSum / weightSum;
  double utilityAvg = utilitySum / weightSum;
  double utilitySqAvg = utilitySqSum / weightSum;

  //TODO statslock may be unnecessary now with the dirtyCounter mechanism?
  while(node.statsLock.test_and_set(std::memory_order_acquire));
  node.stats.winLossValueAvg.store(winLossValueAvg,std::memory_order_release);
  node.stats.noResultValueAvg.store(noResultValueAvg,std::memory_order_release);
  node.stats.scoreMeanAvg.store(scoreMeanAvg,std::memory_order_release);
  node.stats.scoreMeanSqAvg.store(scoreMeanSqAvg,std::memory_order_release);
  node.stats.leadAvg.store(leadAvg,std::memory_order_release);
  node.stats.utilityAvg.store(utilityAvg,std::memory_order_release);
  node.stats.utilitySqAvg.store(utilitySqAvg,std::memory_order_release);
  node.stats.weightSqSum.store(weightSqSum,std::memory_order_release);
  node.stats.weightSum.store(weightSum,std::memory_order_release);
  node.stats.visits.fetch_add(numVisitsToAdd,std::memory_order_release);
  node.statsLock.clear(std::memory_order_release);
}

bool Search::runSinglePlayout(SearchThread& thread, double upperBoundVisitsLeft) {
  //Store this value, used for futile-visit pruning this thread's root children selections.
  thread.upperBoundVisitsLeft = upperBoundVisitsLeft;

  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE];
  bool finishedPlayout = playoutDescend(thread,*rootNode,posesWithChildBuf,true);

  //Restore thread state back to the root state
  thread.pla = rootPla;
  thread.board = rootBoard;
  thread.history = rootHistory;

  return finishedPlayout;
}

void Search::addLeafValue(
  SearchNode& node,
  double winLossValue,
  double noResultValue,
  double scoreMean,
  double scoreMeanSq,
  double lead,
  double weight,
  bool isTerminal,
  bool assumeNoExistingWeight
) {
  double utility =
    getResultUtility(winLossValue, noResultValue)
    + getScoreUtility(scoreMean, scoreMeanSq);

  if(searchParams.subtreeValueBiasFactor != 0 && !isTerminal && node.subtreeValueBiasTableEntry != nullptr) {
    SubtreeValueBiasEntry& entry = *(node.subtreeValueBiasTableEntry);
    while(entry.entryLock.test_and_set(std::memory_order_acquire));
    double newEntryDeltaUtilitySum = entry.deltaUtilitySum;
    double newEntryWeightSum = entry.weightSum;
    entry.entryLock.clear(std::memory_order_release);
    //This is the amount of the direct evaluation of this node that we are going to bias towards the table entry
    const double biasFactor = searchParams.subtreeValueBiasFactor;
    if(newEntryWeightSum > 0.001)
      utility += biasFactor * newEntryDeltaUtilitySum / newEntryWeightSum;
  }
  double utilitySq = utility * utility;
  double weightSq = weight * weight;

  if(assumeNoExistingWeight) {
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    node.stats.winLossValueAvg.store(winLossValue,std::memory_order_release);
    node.stats.noResultValueAvg.store(noResultValue,std::memory_order_release);
    node.stats.scoreMeanAvg.store(scoreMean,std::memory_order_release);
    node.stats.scoreMeanSqAvg.store(scoreMeanSq,std::memory_order_release);
    node.stats.leadAvg.store(lead,std::memory_order_release);
    node.stats.utilityAvg.store(utility,std::memory_order_release);
    node.stats.utilitySqAvg.store(utilitySq,std::memory_order_release);
    node.stats.weightSqSum.store(weightSq,std::memory_order_release);
    node.stats.weightSum.store(weight,std::memory_order_release);
    int64_t oldVisits = node.stats.visits.fetch_add(1,std::memory_order_release);
    node.statsLock.clear(std::memory_order_release);
    assert(oldVisits == 0);
  }
  else {
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    double oldWeightSum = node.stats.weightSum.load(std::memory_order_relaxed);
    double newWeightSum = oldWeightSum + weight;

    node.stats.winLossValueAvg.store((node.stats.winLossValueAvg.load(std::memory_order_relaxed) * oldWeightSum + winLossValue * weight)/newWeightSum,std::memory_order_release);
    node.stats.noResultValueAvg.store((node.stats.noResultValueAvg.load(std::memory_order_relaxed) * oldWeightSum + noResultValue * weight)/newWeightSum,std::memory_order_release);
    node.stats.scoreMeanAvg.store((node.stats.scoreMeanAvg.load(std::memory_order_relaxed) * oldWeightSum + scoreMean * weight)/newWeightSum,std::memory_order_release);
    node.stats.scoreMeanSqAvg.store((node.stats.scoreMeanSqAvg.load(std::memory_order_relaxed) * oldWeightSum + scoreMeanSq * weight)/newWeightSum,std::memory_order_release);
    node.stats.leadAvg.store((node.stats.leadAvg.load(std::memory_order_relaxed) * oldWeightSum + lead * weight)/newWeightSum,std::memory_order_release);
    node.stats.utilityAvg.store((node.stats.utilityAvg.load(std::memory_order_relaxed) * oldWeightSum + utility * weight)/newWeightSum,std::memory_order_release);
    node.stats.utilitySqAvg.store((node.stats.utilitySqAvg.load(std::memory_order_relaxed) * oldWeightSum + utilitySq * weight)/newWeightSum,std::memory_order_release);
    node.stats.weightSqSum.store(node.stats.weightSqSum.load(std::memory_order_relaxed) + weightSq,std::memory_order_release);
    node.stats.weightSum.store(newWeightSum,std::memory_order_release);
    node.stats.visits.fetch_add(1,std::memory_order_release);
    node.statsLock.clear(std::memory_order_release);
  }
}

//Assumes node already has an nnOutput
void Search::maybeRecomputeExistingNNOutput(
  SearchThread& thread, SearchNode& node, bool isRoot
) {
  //Right now only the root node currently ever needs to recompute, and only if it's old
  if(isRoot && node.nodeAge.load(std::memory_order_acquire) != searchNodeAge) {
    //See if we're the lucky thread that gets to do the update!
    //Threads that pass by here later will NOT wait for us to be done before proceeding with search.
    //We accept this and tolerate that for a few iterations potentially we will be using the OLD policy - without noise,
    //or without root temperature, etc.
    //Or if we have none of those things, then we'll not end up updating anything except the age, which is okay too.
    int oldAge = node.nodeAge.exchange(searchNodeAge,std::memory_order_acq_rel);
    if(oldAge < searchNodeAge) {
      NNOutput* nnOutput = node.getNNOutput();
      assert(nnOutput != NULL);

      //Recompute if we have no ownership map, since we need it for getEndingWhiteScoreBonus
      //If conservative passing, then we may also need to recompute the root policy ignoring the history if a pass ends the game
      //If averaging a bunch of symmetries, then we need to recompute it too
      if(nnOutput->whiteOwnerMap == NULL ||
         (searchParams.conservativePass && thread.history.passWouldEndGame(thread.board,thread.pla)) ||
         searchParams.rootNumSymmetriesToSample > 1
      ) {
        initNodeNNOutput(thread,node,isRoot,false,true);
      }
      //We also need to recompute the root nn if we have root noise or temperature and that's missing.
      else {
        //We don't need to go all the way to the nnEvaluator, we just need to maybe add those transforms
        //to the existing policy.
        shared_ptr<NNOutput>* result = maybeAddPolicyNoiseAndTemp(thread,isRoot,nnOutput);
        if(result != NULL)
          node.storeNNOutput(result,thread);
      }
    }
  }
}

//If isReInit is false, among any threads trying to store, the first one wins
//If isReInit is true, we always replace, even for threads that come later.
//Returns true if a nnOutput was set where there was none before.
bool Search::initNodeNNOutput(
  SearchThread& thread, SearchNode& node,
  bool isRoot, bool skipCache, bool isReInit
) {
  bool includeOwnerMap = isRoot || alwaysIncludeOwnerMap;
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
  nnInputParams.conservativePass = searchParams.conservativePass;
  nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
  nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == thread.pla;
  if(searchParams.playoutDoublingAdvantage != 0) {
    Player playoutDoublingAdvantagePla = getPlayoutDoublingAdvantagePla();
    nnInputParams.playoutDoublingAdvantage = (
      getOpp(thread.pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
    );
  }

  std::shared_ptr<NNOutput>* result;
  if(isRoot && searchParams.rootNumSymmetriesToSample > 1) {
    vector<shared_ptr<NNOutput>> ptrs;
    std::array<int, NNInputs::NUM_SYMMETRY_COMBINATIONS> symmetryIndexes;
    std::iota(symmetryIndexes.begin(), symmetryIndexes.end(), 0);
    for(int i = 0; i<searchParams.rootNumSymmetriesToSample; i++) {
      std::swap(symmetryIndexes[i], symmetryIndexes[thread.rand.nextInt(i,NNInputs::NUM_SYMMETRY_COMBINATIONS-1)]);
      nnInputParams.symmetry = symmetryIndexes[i];
      bool skipCacheThisIteration = true; //Skip cache since there's no guarantee which symmetry is in the cache
      nnEvaluator->evaluate(
        thread.board, thread.history, thread.pla,
        nnInputParams,
        thread.nnResultBuf, skipCacheThisIteration, includeOwnerMap
      );
      ptrs.push_back(std::move(thread.nnResultBuf.result));
    }
    result = new std::shared_ptr<NNOutput>(new NNOutput(ptrs));
  }
  else {
    nnEvaluator->evaluate(
      thread.board, thread.history, thread.pla,
      nnInputParams,
      thread.nnResultBuf, skipCache, includeOwnerMap
    );
    result = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
  }

  assert((*result)->noisedPolicyProbs == NULL);
  std::shared_ptr<NNOutput>* noisedResult = maybeAddPolicyNoiseAndTemp(thread,isRoot,result->get());
  if(noisedResult != NULL) {
    std::shared_ptr<NNOutput>* tmp = result;
    result = noisedResult;
    delete tmp;
  }

  node.nodeAge.store(searchNodeAge,std::memory_order_release);
  //If this is a re-initialization of the nnOutput, we don't want to add any visits or anything.
  //Also don't bother updating any of the stats. Technically we should do so because winLossValueSum
  //and such will have changed potentially due to a new orientation of the neural net eval
  //slightly affecting the evals, but this is annoying to recompute from scratch, and on the next
  //visit updateStatsAfterPlayout should fix it all up anyways.
  if(isReInit) {
    bool wasNullBefore = node.storeNNOutput(result,thread);
    return wasNullBefore;
  }
  else {
    bool suc = node.storeNNOutputIfNull(result);
    if(!suc) {
      delete result;
      return false;
    }
    addCurrentNNOutputAsLeafValue(node,true);
    return true;
  }
}

void Search::addCurrentNNOutputAsLeafValue(SearchNode& node, bool assumeNoExistingWeight) {
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  //Values in the search are from the perspective of white positive always
  double winProb = (double)nnOutput->whiteWinProb;
  double lossProb = (double)nnOutput->whiteLossProb;
  double noResultProb = (double)nnOutput->whiteNoResultProb;
  double scoreMean = (double)nnOutput->whiteScoreMean;
  double scoreMeanSq = (double)nnOutput->whiteScoreMeanSq;
  double lead = (double)nnOutput->whiteLead;
  double weight = computeWeightFromNNOutput(nnOutput);
  addLeafValue(node,winProb-lossProb,noResultProb,scoreMean,scoreMeanSq,lead,weight,false,assumeNoExistingWeight);
}


double Search::computeWeightFromNNOutput(const NNOutput* nnOutput) const {
  if(!searchParams.useUncertainty)
    return 1.0;
  if(!nnEvaluator->supportsShorttermError())
    return 1.0;

  double scoreMean = (double)nnOutput->whiteScoreMean;
  double utilityUncertaintyWL = searchParams.winLossUtilityFactor * nnOutput->shorttermWinlossError;
  double utilityUncertaintyScore = getApproxScoreUtilityDerivative(scoreMean) * nnOutput->shorttermScoreError;
  double utilityUncertainty = utilityUncertaintyWL + utilityUncertaintyScore;

  double poweredUncertainty;
  if(searchParams.uncertaintyExponent == 1.0)
    poweredUncertainty = utilityUncertainty;
  else if(searchParams.uncertaintyExponent == 0.5)
    poweredUncertainty = sqrt(utilityUncertainty);
  else
    poweredUncertainty = pow(utilityUncertainty, searchParams.uncertaintyExponent);

  double baselineUncertainty = searchParams.uncertaintyCoeff / searchParams.uncertaintyMaxWeight;
  double weight = searchParams.uncertaintyCoeff / (poweredUncertainty + baselineUncertainty);
  return weight;
}

bool Search::playoutDescend(
  SearchThread& thread, SearchNode& node,
  bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
  bool isRoot
) {
  //Hit terminal node, finish
  //In the case where we're forcing the search to make another move at the root, don't terminate, actually run search for a move more.
  //In the case where we're conservativePass and the game just ended due to a root pass, actually let it keep going.
  //Note that in the second case with tree reuse we can end up with a weird situation where a terminal node becomes nonterminal due
  //to now being a child of the root! This is okay - subsequent visits to the node will fall through to initNodeNNOutput, and we will
  //have a weird leaf node with 2 visits worth of mixed terminal and nn values, but further visits will even hit recomputeNodeStats
  //which should clean it all it.
  if(!isRoot && thread.history.isGameFinished &&
     !(searchParams.conservativePass &&
       thread.history.moveHistory.size() == rootHistory.moveHistory.size() + 1 &&
       node.prevMoveLoc == Board::PASS_LOC)
  ) {
    if(thread.history.isNoResult) {
      double winLossValue = 0.0;
      double noResultValue = 1.0;
      double scoreMean = 0.0;
      double scoreMeanSq = 0.0;
      double lead = 0.0;
      double weight = (searchParams.useUncertainty && nnEvaluator->supportsShorttermError()) ? searchParams.uncertaintyMaxWeight : 1.0;
      addLeafValue(node, winLossValue, noResultValue, scoreMean, scoreMeanSq, lead, weight, true, false);
      return true;
    }
    else {
      double winLossValue = 2.0 * ScoreValue::whiteWinsOfWinner(thread.history.winner, searchParams.drawEquivalentWinsForWhite) - 1;
      double noResultValue = 0.0;
      double scoreMean = ScoreValue::whiteScoreDrawAdjust(thread.history.finalWhiteMinusBlackScore,searchParams.drawEquivalentWinsForWhite,thread.history);
      double scoreMeanSq = ScoreValue::whiteScoreMeanSqOfScoreGridded(thread.history.finalWhiteMinusBlackScore,searchParams.drawEquivalentWinsForWhite);
      double lead = scoreMean;
      double weight = (searchParams.useUncertainty && nnEvaluator->supportsShorttermError()) ? searchParams.uncertaintyMaxWeight : 1.0;
      addLeafValue(node, winLossValue, noResultValue, scoreMean, scoreMeanSq, lead, weight, true, false);
      return true;
    }
  }

  int nodeState = node.state.load(std::memory_order_acquire);
  if(nodeState == SearchNode::STATE_UNEVALUATED) {
    //Always attempt to set a new nnOutput. That way, if some GPU is slow and malfunctioning, we don't get blocked by it.
    {
      bool suc = initNodeNNOutput(thread,node,isRoot,false,false);
      //Leave the node as unevaluated - only the thread that first actually set the nnOutput into the node
      //gets to update the state, to avoid races where we update the state while the node stats aren't updated yet.
      if(!suc)
        return false;
    }

    bool suc = node.state.compare_exchange_strong(nodeState, SearchNode::STATE_EVALUATING, std::memory_order_seq_cst);
    if(!suc) {
      //Presumably someone else got there first.
      //Just give up on this playout and try again from the start.
      return false;
    }
    else {
      //Perform the nn evaluation and finish!
      node.initializeChildren();
      node.state.store(SearchNode::STATE_EXPANDED0, std::memory_order_seq_cst);
      return true;
    }
  }
  else if(nodeState == SearchNode::STATE_EVALUATING) {
    //Just give up on this playout and try again from the start.
    return false;
  }

  assert(nodeState >= SearchNode::STATE_EXPANDED0);
  maybeRecomputeExistingNNOutput(thread,node,isRoot);

  //Find the best child to descend down
  int numChildrenFound;
  int bestChildIdx;
  Loc bestChildMoveLoc;

  SearchNode* child = NULL;
  while(true) {
    selectBestChildToDescend(thread,node,nodeState,numChildrenFound,bestChildIdx,bestChildMoveLoc,posesWithChildBuf,isRoot);

    //The absurdly rare case that the move chosen is not legal
    //(this should only happen either on a bug or where the nnHash doesn't have full legality information or when there's an actual hash collision).
    //Regenerate the neural net call and continue
    if(bestChildIdx >= 0 && !thread.history.isLegal(thread.board,bestChildMoveLoc,thread.pla)) {
      bool isReInit = true;
      initNodeNNOutput(thread,node,isRoot,true,isReInit);

      {
        NNOutput* nnOutput = node.getNNOutput();
        assert(nnOutput != NULL);
        logger->write("WARNING: Chosen move not legal so regenerated nn output, nnhash=" + nnOutput->nnHash.toString());
      }

      //As isReInit is true, we don't return, just keep going, since we didn't count this as a true visit in the node stats
      nodeState = node.state.load(std::memory_order_acquire);
      selectBestChildToDescend(thread,node,nodeState,numChildrenFound,bestChildIdx,bestChildMoveLoc,posesWithChildBuf,isRoot);

      if(bestChildIdx >= 0) {
        //In THEORY it might still be illegal this time! This would be the case if when we initialized the NN output, we raced
        //against someone reInitializing the output to add dirichlet noise or something, who was doing so based on an older cached
        //nnOutput that still had the illegal move. If so, then just fail this playout and try again.
        if(!thread.history.isLegal(thread.board,bestChildMoveLoc,thread.pla))
          return false;
      }
    }

    if(bestChildIdx <= -1) {
      //TODO add test case
      //This might happen if all moves have been forbidden. The node will just get stuck at 1 visit forever then
      //and we won't do any search.
      addCurrentNNOutputAsLeafValue(node,false);
      return true;
    }

    //Do we think we are searching a new child for the first time?
    if(bestChildIdx >= numChildrenFound) {
      assert(bestChildIdx == numChildrenFound);
      assert(bestChildIdx < NNPos::MAX_NN_POLICY_SIZE);
      bool suc = node.maybeExpandChildrenCapacityForNewChild(nodeState, numChildrenFound+1);
      //Someone else is expanding. Loop again trying to select the best child to explore.
      if(!suc) {
        std::this_thread::yield();
        nodeState = node.state.load(std::memory_order_acquire);
        continue;
      }

      int childrenCapacity;
      SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);
      assert(childrenCapacity > bestChildIdx);
      child = new SearchNode(thread.pla,bestChildMoveLoc,&node);
      child->virtualLosses.fetch_add(1,std::memory_order_release);

      suc = children[bestChildIdx].storeIfNull(child);
      if(!suc) {
        //Someone got there ahead of us. Delete and loop again trying to select the best child to explore.
        delete child;
        child = NULL;
        std::this_thread::yield();
        nodeState = node.state.load(std::memory_order_acquire);
        continue;
      }
    }
    //Searching an existing child
    else {
      int childrenCapacity;
      SearchChildPointer* children = node.getChildren(nodeState,childrenCapacity);
      child = children[bestChildIdx].getIfAllocated();
      assert(child != NULL);

      child->virtualLosses.fetch_add(1,std::memory_order_release);
    }

    break;
  }

  if(searchParams.subtreeValueBiasFactor != 0) {
    if(node.prevMoveLoc != Board::NULL_LOC) {
      assert(subtreeValueBiasTable != NULL);
      child->subtreeValueBiasTableEntry = std::move(subtreeValueBiasTable->get(thread.pla, node.prevMoveLoc, child->prevMoveLoc, thread.board));
    }
  }

  //Make the move!
  thread.history.makeBoardMoveAssumeLegal(thread.board,bestChildMoveLoc,thread.pla,rootKoHashTable);
  thread.pla = getOpp(thread.pla);

  //Recurse!
  bool finishedPlayout = playoutDescend(thread,*child,posesWithChildBuf,false);
  //Update this node stats
  if(finishedPlayout)
    updateStatsAfterPlayout(node,thread,isRoot);
  child->virtualLosses.fetch_add(-1,std::memory_order_release);

  return finishedPlayout;
}
