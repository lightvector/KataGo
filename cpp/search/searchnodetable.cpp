#include "../search/searchnodetable.h"

#include "../core/rand.h"
#include "../search/localpattern.h"

// static std::mutex initMutex;
// static std::atomic<bool> isInited(false);
// static LocalPatternHasher patternHasher;
// static Hash128 ZOBRIST_MOVE_LOCS[Board::MAX_ARR_SIZE][2];
// static Hash128 ZOBRIST_KO_BAN[Board::MAX_ARR_SIZE];

// static void initIfNeeded() {
//   if(isInited)
//     return;
//   std::lock_guard<std::mutex> lock(initMutex);
//   if(isInited)
//     return;
//   Rand rand("ValueBiasTable ZOBRIST STUFF");
//   patternHasher.init(5,5,rand);

//   for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
//     for(int j = 0; j<2; j++) {
//       uint64_t h0 = rand.nextUInt64();
//       uint64_t h1 = rand.nextUInt64();
//       ZOBRIST_MOVE_LOCS[i][j] = Hash128(h0,h1);
//     }
//   }

//   rand.init("Reseed ValueBiasTable zobrist so that zobrists don't change when Board::MAX_ARR_SIZE changes");
//   for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
//     uint64_t h0 = rand.nextUInt64();
//     uint64_t h1 = rand.nextUInt64();
//     ZOBRIST_KO_BAN[i] = Hash128(h0,h1);
//   }
//   isInited = true;
// }

SearchNodeTable::SearchNodeTable(int numShardsPowerOfTwo) {
  numShards = (uint32_t)1 << numShardsPowerOfTwo;
  mutexPool = new MutexPool(numShards);
  entries.resize(numShards);
}
SearchNodeTable::~SearchNodeTable() {
  delete mutexPool;
}

uint32_t SearchNodeTable::getIndex(uint64_t hash) const {
  uint32_t mutexPoolMask = numShards-1; //Always a power of two
  return (uint32_t)(hash & mutexPoolMask);
}

