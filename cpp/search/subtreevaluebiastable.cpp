#include "../search/subtreevaluebiastable.h"

#include "../core/rand.h"
#include "../search/localpattern.h"
#include "../search/search.h"

static std::mutex initMutex;
static std::atomic<bool> isInited(false);
static LocalPatternHasher patternHasher;
static Hash128 ZOBRIST_MOVE_LOCS[Board::MAX_ARR_SIZE][2];
static Hash128 ZOBRIST_KO_BAN[Board::MAX_ARR_SIZE];

static void initIfNeeded() {
  if(isInited)
    return;
  std::lock_guard<std::mutex> lock(initMutex);
  if(isInited)
    return;
  Rand rand("ValueBiasTable ZOBRIST STUFF");
  patternHasher.init(5,5,rand);

  for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
    for(int j = 0; j<2; j++) {
      uint64_t h0 = rand.nextUInt64();
      uint64_t h1 = rand.nextUInt64();
      ZOBRIST_MOVE_LOCS[i][j] = Hash128(h0,h1);
    }
  }

  rand.init("Reseed ValueBiasTable zobrist so that zobrists don't change when Board::MAX_ARR_SIZE changes");
  for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    ZOBRIST_KO_BAN[i] = Hash128(h0,h1);
  }
  isInited = true;
}


SubtreeValueBiasHandle::SubtreeValueBiasHandle()
  : lastSubtreeValueBiasDeltaSum(0.0),
    lastSubtreeValueBiasWeight(0.0),
    entry(nullptr),
    table(nullptr)
{
}
SubtreeValueBiasHandle::~SubtreeValueBiasHandle() {
  clear();
}

void SubtreeValueBiasHandle::clear() {
  if(entry != nullptr) {
    assert(table != nullptr);
    if(table->freePropEnabled) {
      double freeProp = table->search->searchParams.subtreeValueBiasFreeProp;
      double deltaUtilitySumToSubtract = lastSubtreeValueBiasDeltaSum * freeProp;
      double weightSumToSubtract = lastSubtreeValueBiasWeight * freeProp;

      SubtreeValueBiasEntry& entry_ = *entry;
      while(entry_.entryLock.test_and_set(std::memory_order_acquire));
      entry_.deltaUtilitySum -= deltaUtilitySumToSubtract;
      entry_.weightSum -= weightSumToSubtract;
      entry_.entryLock.clear(std::memory_order_release);
      entry = nullptr;
      table = nullptr;
    }
  }
}

double SubtreeValueBiasHandle::getValue() const {
  double newEntryDeltaUtilitySum;
  double newEntryWeightSum;

  SubtreeValueBiasEntry& entry_ = *entry;
  while(entry_.entryLock.test_and_set(std::memory_order_acquire));
  newEntryDeltaUtilitySum = entry_.deltaUtilitySum;
  newEntryWeightSum = entry_.weightSum;
  entry_.entryLock.clear(std::memory_order_release);

  if(newEntryWeightSum > 0.001)
    return newEntryDeltaUtilitySum / newEntryWeightSum;
  return 0.0;
}

double SubtreeValueBiasHandle::updateValue(double newDeltaSumThisNode, double newWeightThisNode) {
  double newEntryDeltaUtilitySum;
  double newEntryWeightSum;

  SubtreeValueBiasEntry& entry_ = *entry;
  while(entry_.entryLock.test_and_set(std::memory_order_acquire));
  entry_.deltaUtilitySum += newDeltaSumThisNode - lastSubtreeValueBiasDeltaSum;
  entry_.weightSum += newWeightThisNode - lastSubtreeValueBiasWeight;
  newEntryDeltaUtilitySum = entry_.deltaUtilitySum;
  newEntryWeightSum = entry_.weightSum;
  lastSubtreeValueBiasDeltaSum = newDeltaSumThisNode;
  lastSubtreeValueBiasWeight = newWeightThisNode;
  entry_.entryLock.clear(std::memory_order_release);

  if(newEntryWeightSum > 0.001)
    return newEntryDeltaUtilitySum / newEntryWeightSum;
  return 0.0;
}

SubtreeValueBiasTable::SubtreeValueBiasTable(int32_t numShards, const Search* _search)
  :search(_search), freePropEnabled(true)
{
  initIfNeeded();
  mutexPool = new MutexPool(numShards);
  entries.resize(numShards);
}
SubtreeValueBiasTable::~SubtreeValueBiasTable() {
  delete mutexPool;
}

void SubtreeValueBiasTable::setFreePropEnabled() {
  freePropEnabled = true;
}
void SubtreeValueBiasTable::setFreePropDisabled() {
  freePropEnabled = false;
}

void SubtreeValueBiasTable::clearUnusedSynchronous() {
  for(size_t i = 0; i<entries.size(); i++) {
    std::map<Hash128,std::shared_ptr<SubtreeValueBiasEntry>>& submap = entries[i];
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

void SubtreeValueBiasTable::get(
  SubtreeValueBiasHandle& buf,
  Player pla,
  Loc parentPrevMoveLoc,
  Loc prevMoveLoc,
  const Board& prevBoard
) {
  assert(buf.entry == nullptr);

  Hash128 hash = ZOBRIST_MOVE_LOCS[parentPrevMoveLoc][0] ^ ZOBRIST_MOVE_LOCS[prevMoveLoc][1];

  hash ^= patternHasher.getHash(prevBoard,prevMoveLoc,pla);
  if(prevBoard.ko_loc != Board::NULL_LOC) {
    hash ^= ZOBRIST_KO_BAN[prevBoard.ko_loc];
  }

  uint32_t subMapIdx = (uint32_t)(hash.hash0 % entries.size());

  std::mutex& mutex = mutexPool->getMutex(subMapIdx);
  std::lock_guard<std::mutex> lock(mutex);
  std::shared_ptr<SubtreeValueBiasEntry>& slot = entries[subMapIdx][hash];
  if(slot == nullptr)
    slot = std::make_shared<SubtreeValueBiasEntry>();

  buf.lastSubtreeValueBiasDeltaSum = 0.0;
  buf.lastSubtreeValueBiasWeight = 0.0;
  buf.entry = slot;
  buf.table = this;
}
