
#include <inttypes.h>
#include <algorithm>
#include "../search/search.h"

NodeStats::NodeStats()
  :visits(0),winLossValueSum(0.0),scoreValueSum(0.0)
{}
NodeStats::~NodeStats()
{}

NodeStats::NodeStats(const NodeStats& other)
  :visits(other.visits),winLossValueSum(other.winLossValueSum),scoreValueSum(other.scoreValueSum)
{}
NodeStats& NodeStats::operator=(const NodeStats& other) {
  visits = other.visits;
  winLossValueSum = other.winLossValueSum;
  scoreValueSum = other.scoreValueSum;
  return *this;
}

double NodeStats::getCombinedValueSum(const SearchParams& searchParams) const {
  return
    winLossValueSum * searchParams.winLossUtilityFactor +
    scoreValueSum * searchParams.scoreUtilityFactor;
}

//-----------------------------------------------------------------------------------------

SearchNode::SearchNode(Search& search, SearchThread& thread, Loc moveLoc)
  :lockIdx(),statsLock(),nextPla(thread.pla),prevMoveLoc(moveLoc),
   nnOutput(),
   children(NULL),numChildren(0),childrenCapacity(0),
   stats()
{
  lockIdx = thread.rand.nextUInt(search.mutexPool->getNumMutexes());
}
SearchNode::~SearchNode() {
  for(int i = 0; i<numChildren; i++)
    delete children[i];
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
   rand(search.randSeed + string("$$") + Global::intToString(threadIdx)),
   nnResultBuf(),
   logout(NULL)
{
  if(logger != NULL)
    logout = logger->createOStream();
}
SearchThread::~SearchThread() {
  if(logout != NULL)
    delete logout;
  logout = NULL;
}

//-----------------------------------------------------------------------------------------


Search::Search(Rules rules, SearchParams params, uint32_t mutexPoolSize)
  :rootPla(P_BLACK),rootBoard(),rootHistory(),rootPassLegal(true),searchParams(params)
{
  rootKoHashTable = new KoHashTable();

  rootNode = NULL;
  mutexPool = new MutexPool(mutexPoolSize);

  rootHistory.clear(rootBoard,rootPla,rules);
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

void Search::setRootPassLegal(bool b) {
  clearSearch();
  rootPassLegal = b;
}

void Search::setParams(SearchParams params) {
  clearSearch();
  searchParams = params;
}

void Search::setLog(ostream* o) {
  logout = o;
}

void Search::clearSearch() {
  delete rootNode;
  rootNode = NULL;
}

bool Search::makeMove(Loc moveLoc) {
  //Don't require that the move is legal for the history, merely the board, so that
  //we're robust to the outside saying that a move was made that violates superko or things like that.
  if(!rootBoard.isLegal(moveLoc,rootPla,rootHistory.rules.multiStoneSuicideLegal))
    return false;

  if(rootNode != NULL) {
    for(int i = 0; i<rootNode->numChildren; i++) {
      SearchNode* child = rootNode->children[i];
      if(child->prevMoveLoc == moveLoc) {
        //Grab out the node to prevent its deletion along with the root
        SearchNode* node = new SearchNode(std::move(*child));
        //Delete the root and replace it with the child
        delete rootNode;
        rootNode = node;
        rootNode->prevMoveLoc = Board::NULL_LOC;
        break;
      }
    }
  }
  rootHistory.makeBoardMoveAssumeLegal(rootBoard,moveLoc,rootPla,rootKoHashTable);
  rootPla = getOpp(rootPla);
  rootKoHashTable->recompute(rootHistory);
  return true;
}

void Search::beginSearch(const string& seed, NNEvaluator* nnEval) {
  randSeed = seed;
  nnEvaluator = nnEval;

  if(rootNode == NULL) {
    SearchThread dummyThread(-1, *this, NULL);
    rootNode = new SearchNode(*this, dummyThread, Board::NULL_LOC);
  }
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

static const double POLICY_ILLEGAL_SELECTION_VALUE = -1e50;

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
  child->statsLock.clear(std::memory_order_release);

  //When multithreading, totalChildVisits could be out of sync with childVisits, so if they provably are, then fix that up
  if(totalChildVisits < childVisits)
    totalChildVisits = childVisits;

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
  bool isRoot, bool checkLegalityOfNewMoves) const
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
  double parentValue = node.nnOutput->whiteValue;
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
    if(checkLegalityOfNewMoves && !thread.history.isLegal(thread.board,moveLoc,thread.pla))
      continue;

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
  //Restore thread state back to the root state
  thread.pla = rootPla;
  thread.board = rootBoard;
  thread.history = rootHistory;
}

void Search::playoutDescend(
  SearchThread& thread, SearchNode& node,
  double& retWinLossValue, double& retScoreValue,
  int posesWithChildBuf[NNPos::NN_POLICY_SIZE],
  bool isRoot
) {
  //Hit terminal node, finish
  //In the case where we're forcing the search to make another move at the root, don't terminate, actually run search for a move more.
  if(!isRoot && thread.history.isGameOver()) {
    //TODO what to do here? Is this reasonable? Probably actually want a separate output?
    //weird that this also gets scaled later by winLossUtilityFactor
    if(thread.history.isNoResult) {

      while(node.statsLock.test_and_set(std::memory_order_acquire));
      node.stats.visits += 1;
      node.stats.winLossValueSum += searchParams.noResultUtilityForWhite;
      node.stats.scoreValueSum += 0.0;
      node.statsLock.clear(std::memory_order_release);

      retWinLossValue = searchParams.noResultUtilityForWhite;
      retScoreValue = 0.0;
      return;
    }
    else {
      double winLossValue = NNOutput::whiteValueOfWinner(thread.history.winner);
      assert(thread.board.x_size == thread.board.y_size);
      double scoreValue = NNOutput::whiteValueOfScore(thread.history.finalWhiteMinusBlackScore, thread.board.x_size);

      while(node.statsLock.test_and_set(std::memory_order_acquire));
      node.stats.visits += 1;
      node.stats.winLossValueSum += winLossValue;
      node.stats.scoreValueSum += scoreValue;
      node.statsLock.clear(std::memory_order_release);

      retWinLossValue = winLossValue;
      retScoreValue = scoreValue;
      return;
    }
  }

  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex);

  //Hit leaf node, finish
  if(node.nnOutput == nullptr) {
    nnEvaluator->evaluate(thread.board, thread.history, thread.pla, thread.nnResultBuf, thread.logout);
    node.nnOutput = std::move(thread.nnResultBuf.result);
    maybeAddPolicyNoise(thread,node,isRoot);
    lock.unlock();

    //Values in the search are from the perspective of white positive always
    double value = (double)node.nnOutput->whiteValue;

    //Update node stats
    while(node.statsLock.test_and_set(std::memory_order_acquire));
    node.stats.visits += 1;
    //TODO update this when we have a score prediction on the net
    node.stats.winLossValueSum += value;
    node.stats.scoreValueSum += 0.0;
    node.statsLock.clear(std::memory_order_release);

    retWinLossValue = value;
    retScoreValue = 0.0;
    return;
  }

  //Not leaf node, so recurse

  //Find the best child to descend down
  int bestChildIdx;
  Loc bestChildMoveLoc;
  selectBestChildToDescend(thread,node,bestChildIdx,bestChildMoveLoc,posesWithChildBuf,isRoot,false);

  //In the absurdly rare case that the move chosen is not legal, try again with legality checking
  //(this should only happen either on a bug or where the nnHash doesn't have full legality information or when there's an actual hash collision).
  //We can do this by reducing the hash size and forcing collisions in the nnEval cache
  //TODO test this code branch and remove this assert
  if(!(thread.history.isLegal(thread.board,bestChildMoveLoc,thread.pla))) {
    assert(false); //In testing, actually fail if this ever happens
    selectBestChildToDescend(thread,node,bestChildIdx,bestChildMoveLoc,posesWithChildBuf,isRoot,true);
  }

  if(bestChildIdx < -1) {
    lock.unlock();
    throw StringError("Search error: No move with sane selection value - can't even pass?");
  }

  //TODO virtual losses

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
    lock.unlock();
  }
  else {
    child = node.children[bestChildIdx];

    //Unlock first if the child already exists since we don't depend on it at this point
    lock.unlock();

    assert(thread.history.isLegal(thread.board,moveLoc,thread.pla));
    thread.history.makeBoardMoveAssumeLegal(thread.board,moveLoc,thread.pla,rootKoHashTable);
    thread.pla = getOpp(thread.pla);
  }

  //Recurse!
  playoutDescend(thread,*child,retWinLossValue,retScoreValue,posesWithChildBuf,false);

  //Update stats coming back up
  while(node.statsLock.test_and_set(std::memory_order_acquire));
  node.stats.visits += 1;
  node.stats.winLossValueSum += retWinLossValue;
  node.stats.scoreValueSum += retScoreValue;
  node.statsLock.clear(std::memory_order_release);
}

void Search::printPV(ostream& out, const SearchNode* n, int maxDepth) {
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
      sprintf(buf,"T %6.2fc ",(winLossValueSum + scoreValueSum) / visits * 100.0);
      out << buf;
      sprintf(buf,"W %6.2fc ",winLossValueSum / visits * 100.0);
      out << buf;
      sprintf(buf,"S %6.2fc ",scoreValueSum / visits * 100.0);
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
    if(depth >= options.maxDepth_)
      return;
    if(visits < options.minVisitsToExpand_)
      return;
    if((double)visits < origVisits * options.minVisitsPropToExpand_)
      return;
  }

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
      printTreeHelper(out,child,options,prefix,origVisits,depth+1,childPolicyProb);
      prefix.erase(oldLen);
    }
  }
}



