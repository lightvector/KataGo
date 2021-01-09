#ifndef SEARCH_SUBTREEVALUEBIASTABLE_H
#define SEARCH_SUBTREEVALUEBIASTABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../search/mutexpool.h"

struct SubtreeValueBiasEntry {
  double deltaUtilitySum = 0.0;
  double weightSum = 0.0;
  mutable std::atomic_flag entryLock = ATOMIC_FLAG_INIT;
};

struct SubtreeValueBiasTable {
  std::vector<std::map<Hash128,std::shared_ptr<SubtreeValueBiasEntry>>> entries;
  MutexPool* mutexPool;

  SubtreeValueBiasTable(int32_t numShards);
  ~SubtreeValueBiasTable();

  // ASSUMES there is no concurrent multithreading of this table or any of its entries,
  // and that all past mutations on this table or any of its entries are now visible to this thread.
  void clearUnusedSynchronous();

  std::shared_ptr<SubtreeValueBiasEntry> get(Player pla, Loc parentPrevMoveLoc, Loc prevMoveLoc, const Board& board);
};

#endif


