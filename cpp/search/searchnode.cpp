#include "../search/searchnode.h"

#include "../search/search.h"
#include "../core/test.h"

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
NodeStatsAtomic::NodeStatsAtomic(const NodeStatsAtomic& other)
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


//----------------------------------------------------------------------------------------


MoreNodeStats::MoreNodeStats()
  :stats(),
   selfUtility(0.0),
   weightAdjusted(0.0),
   prevMoveLoc(Board::NULL_LOC)
{}
MoreNodeStats::~MoreNodeStats()
{}


//----------------------------------------------------------------------------------------


SearchChildPointer::SearchChildPointer():
  data(NULL),
  edgeVisits(0),
  moveLoc(Board::NULL_LOC)
{}

void SearchChildPointer::storeAll(const SearchChildPointer& other) {
  SearchNode* d = other.data.load(std::memory_order_acquire);
  int64_t e = other.edgeVisits.load(std::memory_order_acquire);
  Loc m = other.moveLoc.load(std::memory_order_acquire);
  moveLoc.store(m,std::memory_order_release);
  edgeVisits.store(e,std::memory_order_release);
  data.store(d,std::memory_order_release);
}

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

int64_t SearchChildPointer::getEdgeVisits() const {
  return edgeVisits.load(std::memory_order_acquire);
}
int64_t SearchChildPointer::getEdgeVisitsRelaxed() const {
  return edgeVisits.load(std::memory_order_relaxed);
}
void SearchChildPointer::setEdgeVisits(int64_t x) {
  edgeVisits.store(x, std::memory_order_release);
}
void SearchChildPointer::setEdgeVisitsRelaxed(int64_t x) {
  edgeVisits.store(x, std::memory_order_relaxed);
}
void SearchChildPointer::addEdgeVisits(int64_t delta) {
  edgeVisits.fetch_add(delta, std::memory_order_acq_rel);
}
bool SearchChildPointer::compexweakEdgeVisits(int64_t& expected, int64_t desired) {
  return edgeVisits.compare_exchange_weak(expected, desired, std::memory_order_acq_rel);
}


Loc SearchChildPointer::getMoveLoc() const {
  return moveLoc.load(std::memory_order_acquire);
}
Loc SearchChildPointer::getMoveLocRelaxed() const {
  return moveLoc.load(std::memory_order_relaxed);
}
void SearchChildPointer::setMoveLoc(Loc loc) {
  moveLoc.store(loc, std::memory_order_release);
}
void SearchChildPointer::setMoveLocRelaxed(Loc loc) {
  moveLoc.store(loc, std::memory_order_relaxed);
}


//-----------------------------------------------------------------------------------------


//Makes a search node resulting from prevPla playing prevLoc
SearchNode::SearchNode(Player pla, bool fnt, uint32_t mIdx)
  :nextPla(pla),
   forceNonTerminal(fnt),
   patternBonusHash(),
   mutexIdx(mIdx),
   state(SearchNode::STATE_UNEVALUATED),
   nnOutput(),
   nodeAge(0),
   children0(NULL),
   children1(NULL),
   children2(NULL),
   stats(),
   virtualLosses(0),
   lastSubtreeValueBiasDeltaSum(0.0),
   lastSubtreeValueBiasWeight(0.0),
   subtreeValueBiasTableEntry(),
   dirtyCounter(0)
{
}

SearchNode::SearchNode(const SearchNode& other, bool fnt, bool copySubtreeValueBias)
  :nextPla(other.nextPla),
   forceNonTerminal(fnt),
   patternBonusHash(other.patternBonusHash),
   mutexIdx(other.mutexIdx),
   state(other.state.load(std::memory_order_acquire)),
   nnOutput(new std::shared_ptr<NNOutput>(*(other.nnOutput.load(std::memory_order_acquire)))),
   nodeAge(other.nodeAge.load(std::memory_order_acquire)),
   children0(NULL),
   children1(NULL),
   children2(NULL),
   stats(other.stats),
   virtualLosses(other.virtualLosses.load(std::memory_order_acquire)),
   lastSubtreeValueBiasDeltaSum(0.0),
   lastSubtreeValueBiasWeight(0.0),
   subtreeValueBiasTableEntry(),
   dirtyCounter(other.dirtyCounter.load(std::memory_order_acquire))
{
  if(other.children0 != NULL) {
    children0 = new SearchChildPointer[SearchChildrenSizes::SIZE0OVERFLOW];
    for(int i = 0; i<SearchChildrenSizes::SIZE0OVERFLOW; i++)
      children0[i].storeAll(other.children0[i]);
  }
  if(other.children1 != NULL) {
    children1 = new SearchChildPointer[SearchChildrenSizes::SIZE1OVERFLOW];
    for(int i = 0; i<SearchChildrenSizes::SIZE1OVERFLOW; i++)
      children1[i].storeAll(other.children1[i]);
  }
  if(other.children2 != NULL) {
    children2 = new SearchChildPointer[SearchChildrenSizes::SIZE2OVERFLOW];
    for(int i = 0; i<SearchChildrenSizes::SIZE2OVERFLOW; i++)
      children2[i].storeAll(other.children2[i]);
  }
  if(copySubtreeValueBias) {
    //Currently NOT implemented. If we ever want this, think very carefully about copying subtree value bias since
    //if we later delete this node we risk double-counting removal of the subtree value bias!
    assert(false);
    //lastSubtreeValueBiasDeltaSum = other.lastSubtreeValueBiasDeltaSum;
    //lastSubtreeValueBiasWeight = other.lastSubtreeValueBiasWeight;
    //subtreeValueBiasTableEntry = other.subtreeValueBiasTableEntry;
  }
}

SearchNodeChildrenReference SearchNode::getChildren() {
  return SearchNodeChildrenReference(state.load(std::memory_order_acquire),this);
}
ConstSearchNodeChildrenReference SearchNode::getChildren() const {
  return ConstSearchNodeChildrenReference(state.load(std::memory_order_acquire),this);
}
SearchNodeChildrenReference SearchNode::getChildren(SearchNodeState stateValue) {
  return SearchNodeChildrenReference(stateValue,this);
}
ConstSearchNodeChildrenReference SearchNode::getChildren(SearchNodeState stateValue) const {
  return ConstSearchNodeChildrenReference(stateValue,this);
}

static int getChildrenCapacity(SearchNodeState stateValue) {
  if(stateValue < SearchNode::STATE_EXPANDED0)
    return 0;
  else if(stateValue < SearchNode::STATE_EXPANDED1)
    return SearchChildrenSizes::SIZE0TOTAL;
  else if(stateValue < SearchNode::STATE_EXPANDED2)
    return SearchChildrenSizes::SIZE1TOTAL;
  else
    return SearchChildrenSizes::SIZE2TOTAL;
}

int SearchNodeChildrenReference::getCapacity() const {
  return getChildrenCapacity(snapshottedState);
}
int ConstSearchNodeChildrenReference::getCapacity() const {
  return getChildrenCapacity(snapshottedState);
}

int SearchNodeChildrenReference::iterateAndCountChildren() const {
  return ConstSearchNodeChildrenReference(*this).iterateAndCountChildren();
}

int ConstSearchNodeChildrenReference::iterateAndCountChildren() const {
  SearchChildPointer* arr;
  int arrCapacity;
  int offset;

  if(snapshottedState < SearchNode::STATE_EXPANDED0) {
    return 0;
  }
  else if(snapshottedState < SearchNode::STATE_EXPANDED1) {
    arr = node->children0;
    arrCapacity = SearchChildrenSizes::SIZE0OVERFLOW;
    offset = 0;
  }
  else if(snapshottedState < SearchNode::STATE_EXPANDED2) {
    arr = node->children1;
    arrCapacity = SearchChildrenSizes::SIZE1OVERFLOW;
    offset = SearchChildrenSizes::SIZE0TOTAL;
  }
  else {
    arr = node->children2;
    arrCapacity = SearchChildrenSizes::SIZE2OVERFLOW;
    offset = SearchChildrenSizes::SIZE1TOTAL;
  }

  for(int i = 0; i<arrCapacity; i++) {
    if(arr[i].getIfAllocated() == NULL)
      return offset + i;
  }
  return offset+arrCapacity;
}

//Precondition: Assumes that we have actually checked the children array that stateValue suggests that
//we should use, and that every slot in it is full up to numChildrenFullPlusOne-1, and
//that we have found a new legal child to add.
//Postcondition:
//Returns true: node state, stateValue, children arrays are all updated if needed so that they are large enough.
//Returns false: failure since another thread is handling it.
//Thread-safe.
bool SearchNode::maybeExpandChildrenCapacityForNewChild(SearchNodeState& stateValue, int numChildrenFullPlusOne) {
  int capacity = getChildrenCapacity(stateValue);
  if(capacity < numChildrenFullPlusOne) {
    assert(capacity == numChildrenFullPlusOne-1);
    return tryExpandingChildrenCapacityAssumeFull(stateValue);
  }
  return true;
}

void SearchNode::initializeChildren() {
  assert(children0 == NULL);
  children0 = new SearchChildPointer[SearchChildrenSizes::SIZE0OVERFLOW];
}

//Precondition: Assumes that we have actually checked the childen array that stateValue suggests that
//we should use, and that every slot in it is full.
bool SearchNode::tryExpandingChildrenCapacityAssumeFull(SearchNodeState& stateValue) {
  if(stateValue < SearchNode::STATE_EXPANDED1) {
    if(stateValue == SearchNode::STATE_GROWING1)
      return false;
    assert(stateValue == SearchNode::STATE_EXPANDED0);
    bool suc = state.compare_exchange_strong(stateValue,SearchNode::STATE_GROWING1,std::memory_order_acq_rel);
    if(!suc) return false;
    stateValue = SearchNode::STATE_GROWING1;

    SearchChildPointer* children = new SearchChildPointer[SearchChildrenSizes::SIZE1OVERFLOW];
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

    SearchChildPointer* children = new SearchChildPointer[SearchChildrenSizes::SIZE2OVERFLOW];
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

//If we pruned some of the child nodes, collapse down the arrays and the node state to fit.
//This preserves the invariant that the level of expansion is only the minimum needed to hold those child nodes.
//Not thread-safe.
void SearchNode::collapseChildrenCapacity(int numGoodChildren) {
  int stateValue = state.load(std::memory_order_acquire);
  if(numGoodChildren <= SearchChildrenSizes::SIZE1TOTAL && stateValue > SearchNode::STATE_EXPANDED1) {
    assert(stateValue == SearchNode::STATE_EXPANDED2);
    assert(children2 != NULL);
    for(int i = 0; i<SearchChildrenSizes::SIZE2OVERFLOW; i++) {
      testAssert(children2[i].getIfAllocatedRelaxed() == NULL);
    }
    delete[] children2;
    children2 = NULL;
    stateValue = SearchNode::STATE_EXPANDED1;
    state.store(stateValue,std::memory_order_release);
  }
  if(numGoodChildren <= SearchChildrenSizes::SIZE0TOTAL && stateValue > SearchNode::STATE_EXPANDED0) {
    assert(stateValue == SearchNode::STATE_EXPANDED1);
    assert(children1 != NULL);
    for(int i = 0; i<SearchChildrenSizes::SIZE1OVERFLOW; i++) {
      testAssert(children1[i].getIfAllocatedRelaxed() == NULL);
    }
    delete[] children1;
    children1 = NULL;
    stateValue = SearchNode::STATE_EXPANDED0;
    state.store(stateValue,std::memory_order_release);
  }
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
  // Do NOT recursively delete children
  // The children may have other references (e.g. graph search).
  if(children2 != NULL)
    delete[] children2;
  if(children1 != NULL)
    delete[] children1;
  if(children0 != NULL)
    delete[] children0;
  if(nnOutput != NULL)
    delete nnOutput;
}
