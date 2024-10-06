#include "../search/evalcache.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------

FirstExploreEval::FirstExploreEval()
  :avgWinLoss(0.0f),
   avgScoreMean(0.0f),
   cacheWeight(0.0f)
{}
FirstExploreEval::FirstExploreEval(float avgWL, float avgSM, float w)
  :avgWinLoss(avgWL),
   avgScoreMean(avgSM),
   cacheWeight(w)
{}


EvalCacheEntry::EvalCacheEntry()
  :avgWinLoss(0.0f),
   avgNoResult(0.0f),
   avgScoreMean(0.0f),
   avgLead(0.0f),
   cacheWeight(0.0f),
   firstExploreEvals()
{}

EvalCacheTable::EvalCacheTable(int32_t numShards) {
  mutexPool = new MutexPool(numShards);
  entries.resize(numShards);
}
EvalCacheTable::~EvalCacheTable() {
  for(const std::map<Hash128,EvalCacheEntry*>& map: entries) {
    for(const auto& pair: map) {
      delete pair.second;
    }
  }
  delete mutexPool;
}

EvalCacheEntry* EvalCacheTable::find(Hash128 graphHash) {
  uint32_t subMapIdx = (uint32_t)(graphHash.hash0 % entries.size());
  auto iter = entries[subMapIdx].find(graphHash);
  if(iter == entries[subMapIdx].end())
    return NULL;
  return iter->second;
}

void EvalCacheTable::update(Hash128 graphHash, const SearchNode* node, int64_t evalCacheMinVisits, bool isRootNode) {
  uint32_t subMapIdx = (uint32_t)(graphHash.hash0 % entries.size());
  std::mutex& mutex = mutexPool->getMutex(subMapIdx);
  std::lock_guard<std::mutex> lock(mutex);

  EvalCacheEntry*& entry = entries[subMapIdx][graphHash];
  if(entry == NULL)
    entry = new EvalCacheEntry();

  ConstSearchNodeChildrenReference children = node->getChildren();
  int childrenCapacity = children.getCapacity();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;

    int64_t childNumVisits = child->stats.visits.load(std::memory_order_acquire);
    float childWinLossAvg = (float)(child->stats.winLossValueAvg.load(std::memory_order_acquire));
    float childScoreMeanAvg = (float)(child->stats.scoreMeanAvg.load(std::memory_order_acquire));
    Loc moveLoc = children[i].getMoveLocRelaxed();
    if(childNumVisits >= evalCacheMinVisits) {
      float childCacheWeight = (float)childNumVisits;
      FirstExploreEval& eval = entry->firstExploreEvals[moveLoc];
      if(childCacheWeight >= eval.cacheWeight) {
        eval = FirstExploreEval(childWinLossAvg,childScoreMeanAvg,childCacheWeight);
      }
    }
  }

  float newCacheWeight = (float)(node->stats.visits.load(std::memory_order_acquire));
  if(newCacheWeight < entry->cacheWeight * 0.75f)
    return;

  //Should we record this node's aggregate evals in the cache?
  //Or should we only leave it at recording initial first-play evals for exploring children?
  bool shouldRecordEvals = true;
  if(isRootNode) {
    //On the root node, due to it handling passing differently than other nodes, we do NOT record it if passing
    //is near the highest utility move or is at least a third of the visits.
    int64_t totalEdgeVisits = 0;
    int64_t passEdgeVisits = 0;
    double maxSelfUtility = -1e50;
    double passSelfUtility = -1e50;
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      if(child == NULL)
        break;
      int64_t edgeVisits = childPointer.getEdgeVisits();
      double childUtility = child->stats.utilityAvg.load(std::memory_order_acquire);
      double selfUtility = node->nextPla == P_WHITE ? childUtility : -childUtility;
      totalEdgeVisits += edgeVisits;
      maxSelfUtility = std::max(maxSelfUtility,selfUtility);
      if(childPointer.getMoveLocRelaxed() == Board::PASS_LOC) {
        passEdgeVisits += edgeVisits;
        passSelfUtility = selfUtility;
      }
    }
    if(passEdgeVisits * 3 >= totalEdgeVisits || passSelfUtility + 0.01 >= maxSelfUtility) {
      shouldRecordEvals = false;
    }
  }

  if(shouldRecordEvals) {
    entry->cacheWeight = newCacheWeight;
    entry->avgWinLoss = (float)(node->stats.winLossValueAvg.load(std::memory_order_acquire));
    entry->avgNoResult = (float)(node->stats.noResultValueAvg.load(std::memory_order_acquire));
    entry->avgScoreMean = (float)(node->stats.scoreMeanAvg.load(std::memory_order_acquire));
    entry->avgLead = (float)(node->stats.leadAvg.load(std::memory_order_acquire));
  }
}
