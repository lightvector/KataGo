
#include <inttypes.h>
#include <algorithm>
#include "../search/search.h"

NodeStats::NodeStats()
  :visits(0),winLossValueSum(0.0),scoreValueSum(0.0),virtualLosses(0)
{}
NodeStats::~NodeStats()
{}

NodeStats::NodeStats(const NodeStats& other)
  :visits(other.visits),winLossValueSum(other.winLossValueSum),scoreValueSum(other.scoreValueSum),virtualLosses(other.virtualLosses)
{}
NodeStats& NodeStats::operator=(const NodeStats& other) {
  visits = other.visits;
  winLossValueSum = other.winLossValueSum;
  scoreValueSum = other.scoreValueSum;
  virtualLosses = other.virtualLosses;
  return *this;
}

double NodeStats::getCombinedValueSum(const SearchParams& searchParams) const {
  return
    winLossValueSum * searchParams.winLossUtilityFactor +
    scoreValueSum * searchParams.scoreUtilityFactor;
}

//-----------------------------------------------------------------------------------------

SearchNode::SearchNode(Search& search, SearchThread& thread, Loc moveLoc)
  :lockIdx(),statsLock(ATOMIC_FLAG_INIT),nextPla(thread.pla),prevMoveLoc(moveLoc),
   nnOutput(),
   children(NULL),numChildren(0),childrenCapacity(0),
   stats()
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
  stats(other.stats)
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
  return *this;
}

//-----------------------------------------------------------------------------------------


SearchThread::SearchThread(int tIdx, const Search& search, Logger* logger)
  :threadIdx(tIdx),
   pla(search.rootPla),board(search.rootBoard),
   history(search.rootHistory),
   rand(search.randSeed + string("$searchThread$") + Global::intToString(threadIdx)),
   nnResultBuf(),
   logStream(NULL)
{
  if(logger != NULL)
    logStream = logger->createOStream();
}
SearchThread::~SearchThread() {
  if(logStream != NULL)
    delete logStream;
  logStream = NULL;
}

//-----------------------------------------------------------------------------------------


Search::Search(SearchParams params, NNEvaluator* nnEval, const string& rSeed)
  :rootPla(P_BLACK),rootBoard(),rootHistory(),rootPassLegal(true),
   searchParams(params),randSeed(rSeed),
   nnEvaluator(nnEval),
   nonSearchRand(rSeed + string("$nonSearchRand"))
{
  rootKoHashTable = new KoHashTable();

  rootNode = NULL;
  mutexPool = new MutexPool(params.mutexPoolSize);

  rootHistory.clear(rootBoard,rootPla,Rules());
  rootKoHashTable->recompute(rootHistory);
}

Search::~Search() {
  delete rootKoHashTable;
  delete rootNode;
  delete mutexPool;
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
  rootHistory.clear(rootBoard,rootPla,rules);
  rootKoHashTable->recompute(rootHistory);
}

void Search::setRulesAndClearHistory(Rules rules) {
  clearSearch();
  rootBoard.clearSimpleKoLoc();
  rootHistory.clear(rootBoard,rootPla,rules);
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

void Search::clearSearch() {
  delete rootNode;
  rootNode = NULL;
}

bool Search::isLegal(Loc moveLoc, Player movePla) const {
  //Don't require that the move is legal for the history, merely the board, so that
  //we're robust to the outside saying that a move was made that violates superko or things like that.
  //Also handle simple ko correctly in case somehow we find out the same player making multiple moves
  //in a row (which is possible in GTP)
  if(movePla != rootPla) {
    Board copy = rootBoard;
    copy.clearSimpleKoLoc();
    return copy.isLegal(moveLoc,movePla,rootHistory.rules.multiStoneSuicideLegal);
  }
  else
    return rootBoard.isLegal(moveLoc,rootPla,rootHistory.rules.multiStoneSuicideLegal);
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

bool Search::getPlaySelectionValues(vector<Loc>& locs, vector<double>& playSelectionValues) {
  locs.clear();
  playSelectionValues.clear();

  if(rootNode == NULL)
    return false;

  const SearchNode& node = *rootNode;
  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex);

  int numChildren = node.numChildren;

  for(int i = 0; i<numChildren; i++) {
    SearchNode* child = node.children[i];
    Loc moveLoc = child->prevMoveLoc;
    double selectionValue = getPlaySelectionValue(node,child);
    assert(selectionValue >= 0.0);
    locs.push_back(moveLoc);
    playSelectionValues.push_back(selectionValue);
  }
  shared_ptr<NNOutput> nnOutput = node.nnOutput;
  lock.unlock();

  //If we have no children, then use the policy net directly
  if(numChildren == 0) {
    if(nnOutput == nullptr)
      return false;
    for(int movePos = 0; movePos<NNPos::NN_POLICY_SIZE; movePos++) {
      assert(rootBoard.x_size == rootBoard.y_size);
      int offset = NNPos::getOffset(rootBoard.x_size);
      Loc moveLoc = NNPos::posToLoc(movePos,rootBoard.x_size,offset);
      double policyProb = nnOutput->policyProbs[movePos];
      if(!rootHistory.isLegal(rootBoard,moveLoc,rootPla) || policyProb <= 0)
        continue;
      locs.push_back(moveLoc);
      playSelectionValues.push_back(policyProb);
      numChildren++;
    }
  }

  //Might happen absurdly rarely if we have a hash collision or something on the nnOutput
  if(numChildren == 0)
    return false;

  double maxValue = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] > maxValue)
      maxValue = playSelectionValues[i];
  }

  if(maxValue <= 1e-50)
    return false;

  double amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxValue/2.0);
  for(int i = 0; i<numChildren; i++) {
    playSelectionValues[i] -= amountToSubtract;
    if(playSelectionValues[i] <= 0.0)
      playSelectionValues[i] = 0.0;
  }

  return true;
}

Loc Search::getChosenMoveLoc() {
  if(rootNode == NULL)
    return Board::NULL_LOC;

  vector<Loc> locs;
  vector<double> playSelectionValues;
  bool suc = getPlaySelectionValues(locs,playSelectionValues);
  if(!suc)
    return Board::NULL_LOC;

  assert(locs.size() == playSelectionValues.size());
  int numChildren = locs.size();

  double maxValue = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] > maxValue)
      maxValue = playSelectionValues[i];
  }
  assert(maxValue > 0.0);

  double temperature = searchParams.chosenMoveTemperature;
  temperature +=
    (searchParams.chosenMoveTemperatureEarly - searchParams.chosenMoveTemperature) *
    pow(0.5,rootHistory.moveHistory.size() / searchParams.chosenMoveTemperatureHalflife);

  //Temperature so close to 0 that we just calculate the max directly
  if(temperature <= 1.0e-4) {
    double bestSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
    Loc bestChildMoveLoc = Board::NULL_LOC;
    for(int i = 0; i<numChildren; i++) {
      if(playSelectionValues[i] > bestSelectionValue) {
        bestSelectionValue = playSelectionValues[i];
        bestChildMoveLoc = locs[i];
      }
    }
    return bestChildMoveLoc;
  }
  //Actual temperature
  else {
    double sum = 0.0;
    for(int i = 0; i<numChildren; i++) {
      //Numerically stable way to raise to power and normalize
      playSelectionValues[i] = exp((log(playSelectionValues[i]) - log(maxValue)) / searchParams.chosenMoveTemperature);
      sum += playSelectionValues[i];
    }
    assert(sum > 0.0);
    uint32_t idxChosen = nonSearchRand.nextUInt(playSelectionValues.data(),playSelectionValues.size());
    return locs[idxChosen];
  }
}

void Search::beginSearch() {
  if(rootNode == NULL) {
    SearchThread dummyThread(-1, *this, NULL);
    rootNode = new SearchNode(*this, dummyThread, Board::NULL_LOC);
  }
}

uint64_t Search::numRootVisits() {
  if(rootNode == NULL)
    return 0;
  while(rootNode->statsLock.test_and_set(std::memory_order_acquire));
  uint64_t n = rootNode->stats.visits;
  rootNode->statsLock.clear(std::memory_order_release);
  return n;
}

//Assumes node is locked
void Search::maybeAddPolicyNoise(SearchThread& thread, SearchNode& node, bool isRoot) const {
  if(!isRoot || !searchParams.rootNoiseEnabled)
    return;
  //Copy nnOutput as we're about to modify its policy to add noise
  shared_ptr<NNOutput> newNNOutput = std::make_shared<NNOutput>(*(node.nnOutput));
  //Replace the old pointer
  node.nnOutput = newNNOutput;

  int legalCount = 0;
  for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
    if(node.nnOutput->policyProbs[i] >= 0)
      legalCount += 1;
  }

  if(legalCount <= 0)
    throw StringError("maybeAddPolicyNoise: No move with nonnegative policy value - can't even pass?");

  //Generate gamma draw on each move
  double alpha = searchParams.rootDirichletNoiseTotalConcentration / legalCount;
  double rSum = 0.0;
  double r[NNPos::NN_POLICY_SIZE];
  for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
    if(node.nnOutput->policyProbs[i] >= 0) {
      r[i] = thread.rand.nextGamma(alpha);
      rSum += r[i];
    }
    else
      r[i] = 0.0;
  }

  //Normalized gamma draws -> dirichlet noise
  for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++)
    r[i] /= rSum;

  //At this point, r[i] contains a dirichlet distribution draw, so add it into the nnOutput.
  for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
    if(node.nnOutput->policyProbs[i] >= 0) {
      double weight = searchParams.rootDirichletNoiseWeight;
      node.nnOutput->policyProbs[i] = r[i] * weight + node.nnOutput->policyProbs[i] * (1.0-weight);
    }
  }
}


double Search::getPlaySelectionValue(
  double nnPolicyProb, uint64_t childVisits,
  double childValueSum, Player pla
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  (void)(childValueSum);
  (void)(pla);
  return (double)childVisits;
}

double Search::getExploreSelectionValue(
  double nnPolicyProb, uint64_t totalChildVisits, uint64_t childVisits,
  double childValueSum, double fpuValue, Player pla
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  double exploreComponent =
    searchParams.cpuctExploration
    * nnPolicyProb
    * sqrt((double)totalChildVisits + 0.01) //TODO this is weird when totalChildVisits == 0, first exploration
    / (1.0 + childVisits);

  double valueComponent;
  if(childVisits > 0)
    valueComponent = childValueSum / childVisits;
  else
    valueComponent = fpuValue;

  //Adjust it to be from the player's perspective, so that players prefer values in their favor
  //rather than in white's favor
  valueComponent = pla == P_WHITE ? valueComponent : -valueComponent;

  return exploreComponent + valueComponent;
}

int Search::getPos(Loc moveLoc) const {
  int offset = NNPos::getOffset(rootBoard.x_size);
  return NNPos::locToPos(moveLoc,rootBoard.x_size,offset);
}

double Search::getPlaySelectionValue(const SearchNode& parent, const SearchNode* child) const {
  Loc moveLoc = child->prevMoveLoc;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parent.nnOutput->policyProbs[movePos];

  while(child->statsLock.test_and_set(std::memory_order_acquire));
  uint64_t childVisits = child->stats.visits;
  double childValueSum = child->stats.getCombinedValueSum(searchParams);
  child->statsLock.clear(std::memory_order_release);

  return getPlaySelectionValue(nnPolicyProb,childVisits,childValueSum,parent.nextPla);
}
double Search::getExploreSelectionValue(const SearchNode& parent, const SearchNode* child, uint64_t totalChildVisits, double fpuValue) const {
  Loc moveLoc = child->prevMoveLoc;
  int movePos = getPos(moveLoc);
  float nnPolicyProb = parent.nnOutput->policyProbs[movePos];

  while(child->statsLock.test_and_set(std::memory_order_acquire));
  uint64_t childVisits = child->stats.visits;
  double childValueSum = child->stats.getCombinedValueSum(searchParams);
  int32_t childVirtualLosses = child->stats.virtualLosses;
  child->statsLock.clear(std::memory_order_release);

  //When multithreading, totalChildVisits could be out of sync with childVisits, so if they provably are, then fix that up
  if(totalChildVisits < childVisits)
    totalChildVisits = childVisits;

  //Virtual losses to direct threads down different paths
  totalChildVisits += childVirtualLosses;
  childVisits += childVirtualLosses;
  childValueSum += (parent.nextPla == P_WHITE ? -childVirtualLosses : childVirtualLosses) *
    (searchParams.winLossUtilityFactor + searchParams.scoreUtilityFactor);

  return getExploreSelectionValue(nnPolicyProb,totalChildVisits,childVisits,childValueSum,fpuValue,parent.nextPla);
}
double Search::getNewExploreSelectionValue(const SearchNode& parent, int movePos, uint64_t totalChildVisits, double fpuValue) const {
  float nnPolicyProb = parent.nnOutput->policyProbs[movePos];
  uint64_t childVisits = 0;
  double childValueSum = 0.0;
  return getExploreSelectionValue(nnPolicyProb,totalChildVisits,childVisits,childValueSum,fpuValue,parent.nextPla);
}


//Assumes node is locked
void Search::selectBestChildToDescend(
  const SearchThread& thread, const SearchNode& node, int& bestChildIdx, Loc& bestChildMoveLoc,
  int posesWithChildBuf[NNPos::NN_POLICY_SIZE],
  bool isRoot) const
{
  assert(thread.pla == node.nextPla);

  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;

  int numChildren = node.numChildren;

  double policyProbMassVisited = 0.0;
  uint64_t totalChildVisits = 0;
  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = node.children[i];
    Loc moveLoc = child->prevMoveLoc;
    int movePos = getPos(moveLoc);
    float nnPolicyProb = node.nnOutput->policyProbs[movePos];
    policyProbMassVisited += nnPolicyProb;

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    uint64_t childVisits = child->stats.visits;
    child->statsLock.clear(std::memory_order_release);

    totalChildVisits += childVisits;
  }
  //Probability mass should not sum to more than 1, giving a generous allowance
  //for floating point error.
  assert(policyProbMassVisited <= 1.0001);

  //First play urgency
  double parentValue;
  if(searchParams.fpuUseParentAverage) {
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    uint64_t parentVisits = node.stats.visits;
    double parentValueSum = node.stats.getCombinedValueSum(searchParams);
    node.statsLock.clear(std::memory_order_release);
    assert(parentVisits > 0);
    parentValue = parentValueSum / parentVisits;
  }
  else
    parentValue = node.nnOutput->whiteValue;

  double fpuValue;
  if(isRoot && searchParams.rootNoiseEnabled)
    fpuValue = parentValue;
  else {
    if(thread.pla == P_WHITE)
      fpuValue = parentValue - searchParams.fpuReductionMax * sqrt(policyProbMassVisited);
    else
      fpuValue = parentValue + searchParams.fpuReductionMax * sqrt(policyProbMassVisited);
  }

  //Try all existing children
  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = node.children[i];
    Loc moveLoc = child->prevMoveLoc;
    double selectionValue = getExploreSelectionValue(node,child,totalChildVisits,fpuValue);
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = i;
      bestChildMoveLoc = moveLoc;
    }

    posesWithChildBuf[i] = getPos(moveLoc);
  }
  std::sort(posesWithChildBuf, posesWithChildBuf + numChildren);

  //Try all new children
  for(int movePos = 0; movePos<NNPos::NN_POLICY_SIZE; movePos++) {
    bool alreadyTried = std::binary_search(posesWithChildBuf, posesWithChildBuf + numChildren, movePos);
    if(alreadyTried)
      continue;
    if(isRoot && !rootPassLegal && NNPos::isPassPos(movePos))
      continue;
    int offset = NNPos::getOffset(thread.board.x_size);
    Loc moveLoc = NNPos::posToLoc(movePos,thread.board.x_size,offset);

    double selectionValue = getNewExploreSelectionValue(node,movePos,totalChildVisits,fpuValue);
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = numChildren;
      bestChildMoveLoc = moveLoc;
    }
  }

}

void Search::runSinglePlayout(SearchThread& thread) {
  double retWinLossValue;
  double retScoreValue;
  int posesWithChildBuf[NNPos::NN_POLICY_SIZE];
  playoutDescend(thread,*rootNode,retWinLossValue,retScoreValue,posesWithChildBuf,true);

  //Update stats coming back up
  while(rootNode->statsLock.test_and_set(std::memory_order_acquire));
  rootNode->stats.visits += 1;
  rootNode->stats.winLossValueSum += retWinLossValue;
  rootNode->stats.scoreValueSum += retScoreValue;
  rootNode->statsLock.clear(std::memory_order_release);

  //Restore thread state back to the root state
  thread.pla = rootPla;
  thread.board = rootBoard;
  thread.history = rootHistory;
}

void Search::initNodeNNOutput(
  SearchThread& thread, SearchNode& node,
  double& retWinLossValue, double& retScoreValue,
  bool isRoot, bool skipCache
) {
  nnEvaluator->evaluate(thread.board, thread.history, thread.pla, thread.nnResultBuf, thread.logStream, skipCache);
  node.nnOutput = std::move(thread.nnResultBuf.result);
  maybeAddPolicyNoise(thread,node,isRoot);

  //TODO update this and other places when we have a score prediction on the net
  //Values in the search are from the perspective of white positive always
  double value = (double)node.nnOutput->whiteValue;
  retWinLossValue = value;
  retScoreValue = 0.0;
}

void Search::playoutDescend(
  SearchThread& thread, SearchNode& node,
  double& retWinLossValue, double& retScoreValue,
  int posesWithChildBuf[NNPos::NN_POLICY_SIZE],
  bool isRoot
) {
  //Hit terminal node, finish
  //In the case where we're forcing the search to make another move at the root, don't terminate, actually run search for a move more.
  if(!isRoot && thread.history.isGameFinished) {
    //TODO what to do here? Is this reasonable? Probably actually want a separate output?
    //weird that this also gets scaled later by winLossUtilityFactor
    if(thread.history.isNoResult) {
      retWinLossValue = searchParams.noResultUtilityForWhite;
      retScoreValue = 0.0;
      return;
    }
    else {
      double winLossValue = NNOutput::whiteValueOfWinner(thread.history.winner, searchParams.drawUtilityForWhite);
      assert(thread.board.x_size == thread.board.y_size);
      double scoreValue = NNOutput::whiteValueOfScore(thread.history.finalWhiteMinusBlackScore, thread.board.x_size);

      retWinLossValue = winLossValue;
      retScoreValue = scoreValue;
      return;
    }
  }

  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex);

  //Hit leaf node, finish
  if(node.nnOutput == nullptr) {
    initNodeNNOutput(thread,node,retWinLossValue,retScoreValue,isRoot,false);
    return;
  }

  //Not leaf node, so recurse

  //Find the best child to descend down
  int bestChildIdx;
  Loc bestChildMoveLoc;
  selectBestChildToDescend(thread,node,bestChildIdx,bestChildMoveLoc,posesWithChildBuf,isRoot);

  //The absurdly rare case that the move chosen is not legal
  //(this should only happen either on a bug or where the nnHash doesn't have full legality information or when there's an actual hash collision).
  //Regenerate the neural net call and continue
  if(!(thread.history.isLegal(thread.board,bestChildMoveLoc,thread.pla))) {
    initNodeNNOutput(thread,node,retWinLossValue,retScoreValue,isRoot,true);
    lock.unlock();
    if(thread.logStream != NULL)
      (*thread.logStream) << "WARNING: Chosen move not legal so regenerated nn output, nnhash=" << node.nnOutput->nnHash << endl;
    return;
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
    child->stats.virtualLosses += searchParams.numVirtualLossesPerThread; //no lock needed since just created
    node.children[bestChildIdx] = child;

    lock.unlock();
  }
  else {
    child = node.children[bestChildIdx];

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    child->stats.virtualLosses += searchParams.numVirtualLossesPerThread;
    child->statsLock.clear(std::memory_order_release);

    //Unlock before making moves if the child already exists since we don't depend on it at this point
    lock.unlock();

    assert(thread.history.isLegal(thread.board,moveLoc,thread.pla));
    thread.history.makeBoardMoveAssumeLegal(thread.board,moveLoc,thread.pla,rootKoHashTable);
    thread.pla = getOpp(thread.pla);
  }

  //Recurse!
  playoutDescend(thread,*child,retWinLossValue,retScoreValue,posesWithChildBuf,false);

  //Update child stats
  while(child->statsLock.test_and_set(std::memory_order_acquire));
  child->stats.visits += 1;
  child->stats.winLossValueSum += retWinLossValue;
  child->stats.scoreValueSum += retScoreValue;
  child->stats.virtualLosses -= searchParams.numVirtualLossesPerThread;
  child->statsLock.clear(std::memory_order_release);
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
  printTreeHelper(out, node, options, prefix, 0, 0, NAN);
}

void Search::printTreeHelper(
  ostream& out, const SearchNode* n, const PrintTreeOptions& options,
  string& prefix, uint64_t origVisits, int depth, double policyProb
) {
  if(n == NULL)
    return;
  const SearchNode& node = *n;
  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex,std::defer_lock);

  while(node.statsLock.test_and_set(std::memory_order_acquire));
  uint64_t visits = node.stats.visits;
  double winLossValueSum = node.stats.winLossValueSum;
  double scoreValueSum = node.stats.scoreValueSum;
  node.statsLock.clear(std::memory_order_release);

  if(depth == 0)
    origVisits = visits;

  //Output for this node
  {
    out << prefix;
    char buf[64];

    out << ": ";

    if(visits > 0) {
      sprintf(buf,"T %6.2fc ",(winLossValueSum * searchParams.winLossUtilityFactor + scoreValueSum * searchParams.scoreUtilityFactor) / visits * 100.0);
      out << buf;
      sprintf(buf,"W %6.2fc ",(winLossValueSum * searchParams.winLossUtilityFactor) / visits * 100.0);
      out << buf;
      sprintf(buf,"S %6.2fc ",(scoreValueSum * searchParams.scoreUtilityFactor) / visits * 100.0);
      out << buf;
    }

    bool hasNNValue = false;
    double nnValue;
    lock.lock();
    if(node.nnOutput != nullptr) {
      nnValue = node.nnOutput->whiteValue;
      hasNNValue = true;
    }
    lock.unlock();

    if(hasNNValue) {
      sprintf(buf,"V %6.2fc ", nnValue * 100.0);
      out << buf;
    }
    else {
      sprintf(buf,"V --.--c ");
      out << buf;
    }

    if(!isnan(policyProb)) {
      sprintf(buf,"P %5.2f%% ", policyProb * 100.0);
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
  if(depth == options.branch_.size())
    out << "----" << endl;

  lock.lock();

  //Find all children and record their play values
  vector<tuple<const SearchNode*,double,double>> valuedChildren;
  int numChildren = node.numChildren;
  valuedChildren.reserve(numChildren);
  assert(node.nnOutput != nullptr);

  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = node.children[i];
    Loc moveLoc = child->prevMoveLoc;
    int movePos = getPos(moveLoc);
    double childPolicyProb = node.nnOutput->policyProbs[movePos];
    double selectionValue = getPlaySelectionValue(node,child);
    valuedChildren.push_back(std::make_tuple(child,childPolicyProb,selectionValue));
  }

  lock.unlock();

  //Sort in order that we would want to play them
  auto compByValue = [](const tuple<const SearchNode*,double,double>& a, const tuple<const SearchNode*,double,double>& b) {
    return (std::get<2>(a)) > (std::get<2>(b));
  };
  std::sort(valuedChildren.begin(),valuedChildren.end(),compByValue);

  //Apply filtering conditions, but include children that don't match the filtering condition
  //but where there are children afterward that do, in case we ever use something more complex
  //than plain visits as a filter criterion. Do this by finding the last child that we want as the threshold.
  int lastIdxWithEnoughVisits = numChildren-1;
  while(true) {
    if(lastIdxWithEnoughVisits <= 0)
      break;
    const SearchNode* child = std::get<0>(valuedChildren[lastIdxWithEnoughVisits]);

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    uint64_t childVisits = child->stats.visits;
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

    Loc moveLoc = child->prevMoveLoc;

    if((depth >= options.branch_.size() && i < numChildrenToRecurseOn) ||
       (depth < options.branch_.size() && moveLoc == options.branch_[depth]))
    {
      size_t oldLen = prefix.length();
      prefix += Location::toString(moveLoc,rootBoard);
      prefix += " ";
      if(prefix.length() < oldLen+4)
        prefix += " ";
      printTreeHelper(out,child,options,prefix,origVisits,depth+1,childPolicyProb);
      prefix.erase(oldLen);
    }
  }
}



