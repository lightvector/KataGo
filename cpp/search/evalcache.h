#ifndef SEARCH_EVALCACHE_H_
#define SEARCH_EVALCACHE_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../game/board.h"
#include "../search/mutexpool.h"

struct FirstExploreEval {
  float avgWinLoss;
  float avgScoreMean;
  float cacheWeight;
  FirstExploreEval();
  FirstExploreEval(float avgWinLoss, float avgScoreMean, float cacheWeight);
};

struct SearchNode;

struct EvalCacheEntry {
  float avgWinLoss;
  float avgNoResult;
  float avgScoreMean;
  float avgLead;
  float cacheWeight;
  std::map<Loc,FirstExploreEval> firstExploreEvals;

  EvalCacheEntry();
};

struct EvalCacheTable {
  std::vector<std::map<Hash128,EvalCacheEntry*>> entries;
  MutexPool* mutexPool;

  EvalCacheTable(int32_t numShards);
  ~EvalCacheTable();

  EvalCacheEntry* find(Hash128 graphHash);
  void update(Hash128 graphHash, const SearchNode* node, int64_t evalCacheMinVisits, bool isRootNode);
};



#endif // SEARCH_EVALCACHE_H_
