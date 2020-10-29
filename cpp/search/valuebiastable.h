#ifndef SEARCH_VALUEBIASTABLE_H
#define SEARCH_VALUEBIASTABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../search/mutexpool.h"

struct ValueBiasEntry {
  double deltaUtilitySum = 0.0;
  double weightSum = 0.0;
  mutable std::atomic_flag entryLock = ATOMIC_FLAG_INIT;
};

struct ValueBiasTable {
  std::vector<std::map<Hash128,std::shared_ptr<ValueBiasEntry>>> entries;
  MutexPool* mutexPool;

  ValueBiasTable(int32_t numShards);
  ~ValueBiasTable();

  // ASSUMES there is no concurrent multithreading of this table or any of its entries,
  // and that all past mutations on this table or any of its entries are now visible to this thread.
  void clearUnusedSynchronous();

  std::shared_ptr<ValueBiasEntry> get(Player pla, Loc parentPrevMoveLoc, Loc prevMoveLoc, const Board& board);
};

#endif


