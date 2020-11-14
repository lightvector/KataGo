#include "../core/rand.h"
#include "../game/board.h"
#include "../search/subtreevaluebiastable.h"

static std::mutex initMutex;
static std::atomic<bool> isInited(false);
static Hash128 ZOBRIST_LOCAL_PATTERN[NUM_BOARD_COLORS][5][5];
static Hash128 ZOBRIST_ATARI[5][5];
static Hash128 ZOBRIST_PLA[NUM_BOARD_COLORS];
static Hash128 ZOBRIST_MOVE_LOCS[Board::MAX_ARR_SIZE][2];

static void initIfNeeded() {
  if(isInited)
    return;
  std::lock_guard<std::mutex> lock(initMutex);
  if(isInited)
    return;
  Rand rand("ValueBiasTable ZOBRIST STUFF");
  for(int i = 0; i<NUM_BOARD_COLORS; i++) {
    for(int dy = 0; dy<5; dy++) {
      for(int dx = 0; dx<5; dx++) {
        uint64_t h0 = rand.nextUInt64();
        uint64_t h1 = rand.nextUInt64();
        ZOBRIST_LOCAL_PATTERN[i][dy][dx] = Hash128(h0,h1);
      }
    }
  }
  for(int i = 0; i<NUM_BOARD_COLORS; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    ZOBRIST_PLA[i] = Hash128(h0,h1);
  }
  for(int dy = 0; dy<5; dy++) {
    for(int dx = 0; dx<5; dx++) {
      uint64_t h0 = rand.nextUInt64();
      uint64_t h1 = rand.nextUInt64();
      ZOBRIST_ATARI[dy][dx] = Hash128(h0,h1);
    }
  }
  for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
    for(int j = 0; j<2; j++) {
      uint64_t h0 = rand.nextUInt64();
      uint64_t h1 = rand.nextUInt64();
      ZOBRIST_MOVE_LOCS[i][j] = Hash128(h0,h1);
    }
  }
  isInited = true;
}

SubtreeValueBiasTable::SubtreeValueBiasTable(int32_t numShards) {
  initIfNeeded();
  mutexPool = new MutexPool(numShards);
  entries.resize(numShards);
}
SubtreeValueBiasTable::~SubtreeValueBiasTable() {
  delete mutexPool;
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

std::shared_ptr<SubtreeValueBiasEntry> SubtreeValueBiasTable::get(Player pla, Loc parentPrevMoveLoc, Loc prevMoveLoc, const Board& board) {
  Hash128 hash = ZOBRIST_MOVE_LOCS[parentPrevMoveLoc][0] ^ ZOBRIST_MOVE_LOCS[prevMoveLoc][1] ^ ZOBRIST_PLA[pla];
  if(prevMoveLoc != Board::PASS_LOC && prevMoveLoc != Board::NULL_LOC) {
    const int dxi = board.adj_offsets[2];
    const int dyi = board.adj_offsets[3];
    assert(dxi == 1);
    assert(dyi == board.x_size+1);

    int x = Location::getX(prevMoveLoc,board.x_size);
    int y = Location::getY(prevMoveLoc,board.x_size);
    int dxMin = -2, dxMax = 2, dyMin = -2, dyMax = 2;
    if(x < 2) { dxMin = -x; } else if(x >= board.x_size-2) { dxMax = board.x_size-1-x; }
    if(y < 2) { dyMin = -y; } else if(y >= board.y_size-2) { dyMax = board.y_size-1-y; }
    for(int dy = dyMin; dy <= dyMax; dy++) {
      for(int dx = dxMin; dx <= dxMax; dx++) {
        Loc loc = prevMoveLoc + dx * dxi + dy * dyi;
        hash ^= ZOBRIST_LOCAL_PATTERN[board.colors[loc]][dy+2][dx+2];
        if((board.colors[loc] == P_BLACK || board.colors[loc] == P_WHITE) && board.getNumLiberties(loc) == 1)
          hash ^= ZOBRIST_ATARI[dy+2][dx+2];
      }
    }
  }

  auto subMapIdx = hash.hash0 % entries.size();

  std::mutex& mutex = mutexPool->getMutex(subMapIdx);
  std::lock_guard<std::mutex> lock(mutex);
  std::shared_ptr<SubtreeValueBiasEntry>& slot = entries[subMapIdx][hash];
  if(slot == nullptr)
    slot = std::make_shared<SubtreeValueBiasEntry>();
  return slot;
}
