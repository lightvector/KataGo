
#include <algorithm>
#include "../search/search.h"

SearchChildren::SearchChildren()
  :children(NULL),numChildren(0),childrenCapacity(0)
{}
SearchChildren::~SearchChildren() {
  for(int i = 0; i<numChildren; i++)
    delete children[i];
  delete[] children;
}

SearchChildren::SearchChildren(SearchChildren&& other) noexcept {
  children = other.children;
  other.children = NULL;
  numChildren = other.numChildren;
  childrenCapacity = other.childrenCapacity;
}
SearchChildren& SearchChildren::operator=(SearchChildren&& other) noexcept {
  children = other.children;
  other.children = NULL;
  numChildren = other.numChildren;
  childrenCapacity = other.childrenCapacity;
  return *this;
}

//-----------------------------------------------------------------------------------------

SearchNode::SearchNode(Search& search, SearchThread& thread)
  :lockIdx(),nnOutput(),children(),visits(0),
   winLossValueSum(0),scoreValueSum(0),
   childVisits(0)
{
  lockIdx = thread.rand.nextUInt(search.mutexPool->getNumMutexes());
}
SearchNode::~SearchNode() {
}

SearchNode::SearchNode(SearchNode&& other) noexcept
 :lockIdx(other.lockIdx),
  nnOutput(std::move(other.nnOutput)),children(std::move(other.children)),
  visits(other.visits),
  winLossValueSum(other.winLossValueSum),scoreValueSum(other.scoreValueSum),
  childVisits(other.childVisits)
{}
SearchNode& SearchNode::operator=(SearchNode&& other) noexcept {
  lockIdx = other.lockIdx;
  nnOutput = std::move(other.nnOutput);
  children = std::move(other.children);
  visits = other.visits;
  winLossValueSum = other.winLossValueSum;
  scoreValueSum = other.scoreValueSum;
  childVisits = other.childVisits;
  return *this;
}

//-----------------------------------------------------------------------------------------

SearchChild::SearchChild(Search& search, SearchThread& thread, Loc mvLoc)
  :moveLoc(mvLoc),node(search,thread)
{}
SearchChild::~SearchChild() {
}

//-----------------------------------------------------------------------------------------


SearchThread::SearchThread(int tIdx, const Search& search)
  :threadIdx(tIdx),
   pla(search.rootPla),board(search.rootBoard),
   history(search.rootHistory),
   rand(search.randSeed + string("$$") + Global::intToString(threadIdx))
{}
SearchThread::~SearchThread() {

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
    for(int i = 0; i<rootNode->children.numChildren; i++) {
      SearchChild* child = rootNode->children.children[i];
      if(child->moveLoc == moveLoc) {
        //Grab out the node to prevent its deletion along with the root
        SearchNode* node = new SearchNode(std::move(child->node));
        //Delete the root and replace it with the child
        delete rootNode;
        rootNode = node;
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
    SearchThread dummyThread(-1, *this);
    rootNode = new SearchNode(*this, dummyThread);
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

//Assumes node is locked
double Search::getCombinedValueSum(const SearchNode& node) const {
  return
    node.winLossValueSum * searchParams.winLossUtilityFactor +
    node.scoreValueSum * searchParams.scoreUtilityFactor;
}

static const double POLICY_ILLEGAL_SELECTION_VALUE = -1e50;
  
double Search::getPolicySelectionValue(
  double nnPolicyProb, uint64_t totalChildVisits, uint64_t childVisits,
  double childValueSum, double fpuValue
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE; 
    
  double exploreComponent = 
    searchParams.cpuctExploration
    * nnPolicyProb
    * sqrt((double)totalChildVisits)
    / (1.0 + childVisits);

  double valueComponent;
  if(childVisits > 0)
    valueComponent = childValueSum / childVisits;
  else
    valueComponent = fpuValue;

  return exploreComponent + valueComponent;
}

//Assumes node is locked
void Search::selectBestChildToDescend(
  const SearchThread& thread, const SearchNode& node, int& bestChildIdx, Loc& bestChildMoveLoc,
  int posesWithChildBuf[NNPos::NN_POLICY_SIZE],
  bool isRoot, bool checkLegalityOfNewMoves) const
{
  assert(node.visits > 0);
  const SearchChildren& children = node.children;
  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveLoc = Board::NULL_LOC;

  int numChildren = children.numChildren;

  double policyProbMassVisited = 0.0;
  for(int i = 0; i<numChildren; i++) {
    const SearchChild* child = children.children[i];
    Loc moveLoc = child->moveLoc;
    int offset = NNPos::getOffset(thread.board.x_size);
    int movePos = NNPos::locToPos(moveLoc,thread.board.x_size,offset);

    float nnPolicyProb = node.nnOutput->policyProbs[movePos];
    policyProbMassVisited += nnPolicyProb;
  }
  //Probability mass should not sum to more than 1, giving a generous allowance
  //for floating point error.
  assert(policyProbMassVisited <= 1.0001);

  //First play urgency
  double fpuValue;
  if(isRoot && searchParams.rootNoiseEnabled)
    fpuValue = 0.0;
  else {
    //double parentValue = getCombinedValueSum(node) / node.visits;
    double parentValue = node.nnOutput->value / node.visits;
    fpuValue = parentValue - searchParams.fpuReductionMax * sqrt(policyProbMassVisited);
  }

  //Try all existing children
  for(int i = 0; i<numChildren; i++) {
    const SearchChild* child = children.children[i];
    Loc moveLoc = child->moveLoc;
    int offset = NNPos::getOffset(thread.board.x_size);
    int movePos = NNPos::locToPos(moveLoc,thread.board.x_size,offset);

    float nnPolicyProb = node.nnOutput->policyProbs[movePos];
    uint64_t totalChildVisits = node.childVisits;
    uint64_t childVisits = child->node.visits;
    double childValueSum = getCombinedValueSum(child->node);

    double selectionValue = getPolicySelectionValue(nnPolicyProb,totalChildVisits,childVisits,childValueSum,fpuValue);
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = i;
      bestChildMoveLoc = moveLoc;
    }

    posesWithChildBuf[i] = movePos;
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

    float nnPolicyProb = node.nnOutput->policyProbs[movePos];
    uint64_t totalChildVisits = node.childVisits;
    uint64_t childVisits = 0;
    double childValueSum = 0.0;

    double selectionValue = getPolicySelectionValue(nnPolicyProb,totalChildVisits,childVisits,childValueSum,fpuValue);
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
}

void Search::playoutDescend(
  SearchThread& thread, SearchNode& node,
  double& retWinLossValue, double& retScoreValue,
  int posesWithChildBuf[NNPos::NN_POLICY_SIZE],
  bool isRoot
) {
  std::mutex& mutex = mutexPool->getMutex(node.lockIdx);
  unique_lock<std::mutex> lock(mutex);

  //Hit leaf node, finish
  if(node.nnOutput == nullptr) {
    int symmetry = thread.rand.nextInt(0,NNEvaluator::NUM_SYMMETRIES-1);
    node.nnOutput = nnEvaluator->evaluate(thread.board, thread.history, thread.pla, symmetry);
    maybeAddPolicyNoise(thread,node,isRoot);
    lock.unlock();

    //Values in the search are from the perspective of white positive always
    double value = (double)node.nnOutput->value;
    value = (thread.pla == C_WHITE ? value : -value);

    node.visits += 1;
    //TODO update this when we have a score prediction on the net
    node.winLossValueSum += value;
    node.scoreValueSum += 0.0;
    retWinLossValue = value;
    retScoreValue = 0.0;
    return;
  }
  //Hit terminal node, finish
  //In the case where we're forcing the search to make another move at the root, don't terminate, actually run search for a move more.
  if(!isRoot && thread.history.isGameOver()) {
    node.visits += 1;
    //TODO what to do here? Is this reasonable? Probably actually want a separate output?
    //weird that this also gets scaled later by winLossUtilityFactor
    if(thread.history.isNoResult) {
      node.winLossValueSum += searchParams.noResultUtilityForWhite;
      node.scoreValueSum += 0.0;
      retWinLossValue = searchParams.noResultUtilityForWhite;
      retScoreValue = 0.0;
      return;
    }
    else {
      double winLossValue = NNOutput::whiteValueOfWinner(thread.history.winner);
      assert(thread.board.x_size == thread.board.y_size);
      double scoreValue = NNOutput::whiteValueOfScore(thread.history.finalWhiteMinusBlackScore, thread.board.x_size);

      node.winLossValueSum += winLossValue;
      node.scoreValueSum += scoreValue;
      retWinLossValue = winLossValue;
      retScoreValue = scoreValue;
      return;
    }
  }

  //Not leaf node, so recurse

  //Find the best child to descend down
  SearchChildren& children = node.children;
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
  if(bestChildIdx >= children.childrenCapacity) {
    int newCapacity = children.childrenCapacity + (children.childrenCapacity / 4) + 1;
    assert(newCapacity < 0x3FFF);
    SearchChild** newArr = new SearchChild*[newCapacity];
    for(int i = 0; i<children.numChildren; i++) {
      newArr[i] = children.children[i];
      children.children[i] = NULL;
    }
    delete[] children.children;
    children.children = newArr;
    children.childrenCapacity = (uint16_t)newCapacity;
  }

  Loc moveLoc = bestChildMoveLoc;
  
  //Allocate a new child node if necessary
  SearchChild* child;
  if(bestChildIdx == children.numChildren) {
    assert(thread.history.isLegal(thread.board,moveLoc,thread.pla));
    thread.history.makeBoardMoveAssumeLegal(thread.board,moveLoc,thread.pla,rootKoHashTable);
    thread.pla = getOpp(thread.pla);

    children.numChildren++;
    child = new SearchChild(*this,thread,moveLoc);
    children.children[bestChildIdx] = child;
    lock.unlock();
  }
  else {
    child = children.children[bestChildIdx];

    //Unlock first if the child already exists since we don't depend on it at this point
    lock.unlock();

    assert(thread.history.isLegal(thread.board,moveLoc,thread.pla));
    thread.board.playMoveAssumeLegal(moveLoc,thread.pla);
    thread.history.makeBoardMoveAssumeLegal(thread.board,moveLoc,thread.pla,rootKoHashTable);
    thread.pla = getOpp(thread.pla);
  }

  //Recurse!
  playoutDescend(thread,child->node,retWinLossValue,retScoreValue,posesWithChildBuf,false);

  //Update stats coming back up
  lock.lock();

  node.visits += 1;
  node.childVisits += 1;
  node.winLossValueSum += retWinLossValue;
  node.scoreValueSum += retScoreValue;

  lock.unlock();

  //Restore thread state back to the root state
  thread.pla = rootPla;
  thread.board = rootBoard;
  thread.history = rootHistory;
}





