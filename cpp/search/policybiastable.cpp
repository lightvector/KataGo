#include "../search/policybiastable.h"

#include "../core/rand.h"
#include "../neuralnet/nninputs.h"
#include "../search/localpattern.h"
#include "../search/search.h"

static std::mutex initMutex;
static std::atomic<bool> isInited(false);
static LocalPatternHasher patternHasher;
static Hash128 ZOBRIST_PREV_MOVE_LOCS[Board::MAX_ARR_SIZE];
static Hash128 ZOBRIST_MOVE_LOCS[Board::MAX_ARR_SIZE];
static Hash128 ZOBRIST_KO_BAN[Board::MAX_ARR_SIZE];

static void initIfNeeded() {
  if(isInited)
    return;
  std::lock_guard<std::mutex> lock(initMutex);
  if(isInited)
    return;
  Rand rand("PolicyBiasTable ZOBRIST STUFF");
  patternHasher.init(5,5,2,rand);

  for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    ZOBRIST_PREV_MOVE_LOCS[i] = Hash128(h0,h1);
  }

  for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    ZOBRIST_MOVE_LOCS[i] = Hash128(h0,h1);
  }

  rand.init("Reseed ValueBiasTable zobrist so that zobrists don't change when Board::MAX_ARR_SIZE changes");
  for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    ZOBRIST_KO_BAN[i] = Hash128(h0,h1);
  }
  isInited = true;
}



PolicyBiasHandle::PolicyBiasHandle()
  : lastSum(0.0),
    lastWeight(0.0),
    lastPos(-1),
    entries(),
    table(nullptr)
{
}
PolicyBiasHandle::~PolicyBiasHandle() {
  clear();
}

void PolicyBiasHandle::clear() {
  if(table != nullptr) {
    if(table->freePropEnabled) {
      revertUpdates(table->search->searchParams.subtreeValueBiasFreeProp);
      entries.clear();
      table = nullptr;
    }
  }
}

float PolicyBiasHandle::getUpdatedPolicyProb(float nnPolicyProb, int movePos, double policyBiasFactor, bool policyBiasDiscountSelf) const {
  if(entries.size() > 0 && entries[movePos] != nullptr) {
    assert(entries.size() > movePos);
    double policyProbLogSurprise = entries[movePos]->average.load(std::memory_order_acquire);
    double nnPolicyProbComplement = 1.0 - nnPolicyProb;
    if(policyProbLogSurprise > 0 && nnPolicyProbComplement > 1e-10) {
      double desiredLogDelta = policyBiasFactor * policyProbLogSurprise;
      if(policyBiasDiscountSelf && movePos == lastPos) {
        double entryWeightSum = entries[movePos]->weightSum.load(std::memory_order_acquire);
        if(entryWeightSum <= 0.0)
          desiredLogDelta = 0.0;
        else
          desiredLogDelta *= std::max(0.0, entryWeightSum - lastWeight) / entryWeightSum;
      }

      double odds = nnPolicyProb / nnPolicyProbComplement;
      odds *= exp(desiredLogDelta);
      return (float)(odds / (1.0 + odds));
    }
  }
  return nnPolicyProb;
}


void PolicyBiasHandle::revertUpdates(double freeProp) {
  if(table != nullptr) {
    if(lastPos != -1) {
      double sumToSubtract = lastSum * freeProp;
      double weightToSubtract = lastWeight * freeProp;

      PolicyBiasEntry& entry_ = *(entries[lastPos]);
      while(entry_.entryLock.test_and_set(std::memory_order_acquire));
      double average = entry_.average.load(std::memory_order_acquire);
      double oldWeight = entry_.weightSum.load(std::memory_order_acquire);

      double newSum = average * oldWeight;
      double newWeight = oldWeight;
      newSum -= sumToSubtract;
      newWeight -= weightToSubtract;
      if(newWeight < 0.001)
        newSum = 0.0;

      entry_.average.store(newSum / newWeight, std::memory_order_release);
      entry_.weightSum.store(newWeight, std::memory_order_release);

      lastSum = 0.0;
      lastWeight = 0.0;
      lastPos = -1;

      entry_.entryLock.clear(std::memory_order_release);
    }
  }
}

void PolicyBiasHandle::updateValue(double newSumThisNode, double newWeightThisNode, int pos) {
  if(lastPos == pos) {
    PolicyBiasEntry& entry_ = *(entries[pos]);
    while(entry_.entryLock.test_and_set(std::memory_order_acquire));
    double average = entry_.average.load(std::memory_order_acquire);
    double oldWeight = entry_.weightSum.load(std::memory_order_acquire);

    double newSum = average * oldWeight;
    double newWeight = oldWeight;

    {
      double sumToSubtract = lastSum;
      double weightToSubtract = lastWeight;
      newSum -= sumToSubtract;
      newWeight -= weightToSubtract;
      if(newWeight < 0.001)
        newSum = 0.0;
    }

    newSum += newSumThisNode;
    newWeight += newWeightThisNode;

    entry_.average.store(newSum / newWeight, std::memory_order_release);
    entry_.weightSum.store(newWeight, std::memory_order_release);

    lastSum = newSumThisNode;
    lastWeight = newWeightThisNode;
    lastPos = pos;

    entry_.entryLock.clear(std::memory_order_release);
    return;
  }

  revertUpdates(1.0);

  PolicyBiasEntry& entry_ = *(entries[pos]);
  while(entry_.entryLock.test_and_set(std::memory_order_acquire));
  double average = entry_.average.load(std::memory_order_acquire);
  double oldWeight = entry_.weightSum.load(std::memory_order_acquire);

  double newSum = average * oldWeight;
  double newWeight = oldWeight;
  newSum += newSumThisNode;
  newWeight += newWeightThisNode;

  entry_.average.store(newSum / newWeight, std::memory_order_release);
  entry_.weightSum.store(newWeight, std::memory_order_release);

  lastSum = newSumThisNode;
  lastWeight = newWeightThisNode;
  lastPos = pos;

  entry_.entryLock.clear(std::memory_order_release);
}


PolicyBiasTable::PolicyBiasTable(const Search* search_)
  : search(search_), freePropEnabled(true)
{
  initIfNeeded();
  uint32_t numShards = Board::MAX_ARR_SIZE;
  mutexPool = new MutexPool(numShards);
  expectedNNXLen = -1;
  expectedNNYLen = -1;
  entries.resize(numShards);
}
PolicyBiasTable::~PolicyBiasTable() {
  delete mutexPool;
}


void PolicyBiasTable::setFreePropEnabled() {
  freePropEnabled = true;
}
void PolicyBiasTable::setFreePropDisabled() {
  freePropEnabled = false;
}

void PolicyBiasTable::clearUnusedSynchronous() {
  for(size_t i = 0; i<entries.size(); i++) {
    std::map<Hash128,std::shared_ptr<PolicyBiasEntry>>& submap = entries[i];
    for(auto iter = submap.begin(); iter != submap.end(); /* no incr */) {
      // Anything in this map NOT used by anyone else - clear
      if(iter->second.use_count() <= 1) {
        iter = submap.erase(iter);
      }
      else {
        ++iter;
      }
    }
  }
}

void PolicyBiasTable::setNNLenAndAssertEmptySynchronous(int nnXLen, int nnYLen) {
  bool anythingLeft = false;
  for(size_t i = 0; i<entries.size(); i++) {
    std::map<Hash128,std::shared_ptr<PolicyBiasEntry>>& submap = entries[i];
    if(submap.size() > 0)
      anythingLeft = true;
  }
  if(anythingLeft)
    throw StringError("PolicyBiasTable::setNNLenAndAssertEmptySynchronous called when not empty");

  expectedNNXLen = nnXLen;
  expectedNNYLen = nnYLen;
}

void PolicyBiasTable::get(
  PolicyBiasHandle& buf,
  Player pla,
  Loc prevMoveLoc,
  int nnXLen,
  int nnYLen,
  const Board& board,
  const BoardHistory& hist
) {
  if(nnXLen != expectedNNXLen || nnYLen != expectedNNYLen) {
    throw StringError(
      "Bug: PolicyBiasTable was run with unexpected nnXLen or nnYLen " +
      Global::intToString(nnXLen) + " " +
      Global::intToString(nnYLen) + " " +
      Global::intToString(expectedNNXLen) + " " +
      Global::intToString(expectedNNYLen)
    );
  }

  buf.lastSum = 0.0;
  buf.lastWeight = 0.0;
  buf.lastPos = -1;
  buf.table = this;
  buf.entries = std::vector<std::shared_ptr<PolicyBiasEntry>>(nnXLen * nnYLen + 1, nullptr);
  if(prevMoveLoc == Board::NULL_LOC) {
    return;
  }

  Hash128 commonHash = ZOBRIST_PREV_MOVE_LOCS[prevMoveLoc];
  if(board.ko_loc != Board::NULL_LOC) {
    commonHash ^= ZOBRIST_KO_BAN[board.ko_loc];
  }

  uint32_t subMapIdx = (uint32_t)prevMoveLoc;
  std::mutex& mutex = mutexPool->getMutex(subMapIdx);
  std::lock_guard<std::mutex> lock(mutex);

  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(hist.isLegal(board,loc,pla)) {
        Hash128 hash = commonHash ^ ZOBRIST_MOVE_LOCS[loc] ^ patternHasher.getHash(board,loc,pla);
        std::shared_ptr<PolicyBiasEntry>& slot = entries[subMapIdx][hash];
        if(slot == nullptr)
          slot = std::make_shared<PolicyBiasEntry>();
        buf.entries[NNPos::xyToPos(x,y,nnXLen)] = slot;
      }
    }
  }
}
