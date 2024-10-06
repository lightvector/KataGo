#ifndef SEARCH_SEARCHNODE_H_
#define SEARCH_SEARCHNODE_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nneval.h"
#include "../search/subtreevaluebiastable.h"

typedef int SearchNodeState; // See SearchNode::STATE_*

struct SearchNode;
struct SearchThread;

struct NodeStatsAtomic {
  std::atomic<int64_t> visits;
  std::atomic<double> winLossValueAvg;
  std::atomic<double> noResultValueAvg;
  std::atomic<double> scoreMeanAvg;
  std::atomic<double> scoreMeanSqAvg;
  std::atomic<double> leadAvg;
  std::atomic<double> utilityAvg;
  std::atomic<double> utilitySqAvg;
  std::atomic<double> weightSum;
  std::atomic<double> weightSqSum;

  NodeStatsAtomic();
  explicit NodeStatsAtomic(const NodeStatsAtomic& other);
  ~NodeStatsAtomic();

  NodeStatsAtomic& operator=(const NodeStatsAtomic&) = delete;
  NodeStatsAtomic(NodeStatsAtomic&& other) = delete;
  NodeStatsAtomic& operator=(NodeStatsAtomic&& other) = delete;

  double getChildWeight(int64_t edgeVisits) const;
  double getChildWeight(int64_t edgeVisits, int64_t childVisits) const;
  double getChildWeightSq(int64_t edgeVisits) const;
  double getChildWeightSq(int64_t edgeVisits, int64_t childVisits) const;
};

struct NodeStats {
  int64_t visits;
  double winLossValueAvg;
  double noResultValueAvg;
  double scoreMeanAvg;
  double scoreMeanSqAvg;
  double leadAvg;
  double utilityAvg;
  double utilitySqAvg;
  double weightSum;
  double weightSqSum;

  NodeStats();
  explicit NodeStats(const NodeStatsAtomic& other);
  ~NodeStats();

  NodeStats(const NodeStats&) = default;
  NodeStats& operator=(const NodeStats&) = default;
  NodeStats(NodeStats&& other) = default;
  NodeStats& operator=(NodeStats&& other) = default;

  inline static double childWeight(int64_t edgeVisits, int64_t childVisits, double rawChildWeight) {
    return rawChildWeight * ((double)edgeVisits / (double)std::max(childVisits,(int64_t)1));
  }
  inline static double childWeightSq(int64_t edgeVisits, int64_t childVisits, double rawChildWeightSq) {
    return rawChildWeightSq * ((double)edgeVisits / (double)std::max(childVisits,(int64_t)1));
  }
  double getChildWeight(int64_t edgeVisits) {
    return childWeight(edgeVisits, visits, weightSum);
  }
};

inline double NodeStatsAtomic::getChildWeight(int64_t edgeVisits) const {
  return NodeStats::childWeight(edgeVisits, visits.load(std::memory_order_acquire), weightSum.load(std::memory_order_acquire));
}
inline double NodeStatsAtomic::getChildWeight(int64_t edgeVisits, int64_t childVisits) const {
  return NodeStats::childWeight(edgeVisits, childVisits, weightSum.load(std::memory_order_acquire));
}
inline double NodeStatsAtomic::getChildWeightSq(int64_t edgeVisits) const {
  return NodeStats::childWeightSq(edgeVisits, visits.load(std::memory_order_acquire), weightSqSum.load(std::memory_order_acquire));
}
inline double NodeStatsAtomic::getChildWeightSq(int64_t edgeVisits, int64_t childVisits) const {
  return NodeStats::childWeightSq(edgeVisits, childVisits, weightSqSum.load(std::memory_order_acquire));
}


struct MoreNodeStats {
  NodeStats stats;
  double selfUtility;
  double weightAdjusted;
  Loc prevMoveLoc;

  MoreNodeStats();
  ~MoreNodeStats();

  MoreNodeStats(const MoreNodeStats&) = default;
  MoreNodeStats& operator=(const MoreNodeStats&) = default;
  MoreNodeStats(MoreNodeStats&& other) = default;
  MoreNodeStats& operator=(MoreNodeStats&& other) = default;
};


struct SearchChildPointer {
private:
  std::atomic<SearchNode*> data;
  std::atomic<int64_t> edgeVisits;
  std::atomic<Loc> moveLoc; // Generally this will be always guarded under release semantics of data or of the array itself.
public:
  SearchChildPointer();

  SearchChildPointer(const SearchChildPointer&) = delete;
  SearchChildPointer& operator=(const SearchChildPointer&) = delete;
  SearchChildPointer(SearchChildPointer&& other) = delete;
  SearchChildPointer& operator=(SearchChildPointer&& other) = delete;

  void storeAll(const SearchChildPointer& other);

  SearchNode* getIfAllocated();
  const SearchNode* getIfAllocated() const;
  SearchNode* getIfAllocatedRelaxed();
  void store(SearchNode* node);
  void storeRelaxed(SearchNode* node);
  bool storeIfNull(SearchNode* node);

  int64_t getEdgeVisits() const;
  int64_t getEdgeVisitsRelaxed() const;
  void setEdgeVisits(int64_t x);
  void setEdgeVisitsRelaxed(int64_t x);
  void addEdgeVisits(int64_t delta);
  bool compexweakEdgeVisits(int64_t& expected, int64_t desired);

  Loc getMoveLoc() const;
  Loc getMoveLocRelaxed() const;
  void setMoveLoc(Loc loc);
  void setMoveLocRelaxed(Loc loc);
};

namespace SearchChildrenSizes {
  constexpr int SIZE0TOTAL = 8;
  constexpr int SIZE1TOTAL = 64;
  constexpr int SIZE2TOTAL = NNPos::MAX_NN_POLICY_SIZE;
  constexpr int SIZE0OVERFLOW = SIZE0TOTAL;
  constexpr int SIZE1OVERFLOW = SIZE1TOTAL - SIZE0TOTAL;
  constexpr int SIZE2OVERFLOW = SIZE2TOTAL - SIZE1TOTAL;
}

// Abstracts children{0,1,2} in SearchNode as a single concatenated array
struct SearchNodeChildrenReference {
  SearchNodeState snapshottedState;
  SearchNode* node;

  inline SearchNodeChildrenReference() {}
  inline SearchNodeChildrenReference(SearchNodeState snapshottedState_, SearchNode* node_)
    : snapshottedState(snapshottedState_), node(node_) {}
  SearchChildPointer& operator[](int i);
  int getCapacity() const;
  int iterateAndCountChildren() const;
};
struct ConstSearchNodeChildrenReference {
  int capacity;
  SearchNodeState snapshottedState;
  const SearchNode* node;

  inline ConstSearchNodeChildrenReference() {}
  inline ConstSearchNodeChildrenReference(const SearchNodeChildrenReference& other)
    : snapshottedState(other.snapshottedState), node(other.node) {}
  inline ConstSearchNodeChildrenReference(SearchNodeState snapshottedState_, const SearchNode* node_)
    : snapshottedState(snapshottedState_), node(node_) {}
  const SearchChildPointer& operator[](int i);
  int getCapacity() const;
  int iterateAndCountChildren() const;
};

struct SearchNode {
  //Locks------------------------------------------------------------------------------
  mutable std::atomic_flag statsLock = ATOMIC_FLAG_INIT;

  //Constant during search--------------------------------------------------------------
  const Player nextPla;
  const bool forceNonTerminal;
  Hash128 patternBonusHash;
  const uint32_t mutexIdx; // For lookup into mutex pool

  //Mutable---------------------------------------------------------------------------
  //During search, only ever transitions forward.
  std::atomic<SearchNodeState> state;
  static constexpr SearchNodeState STATE_UNEVALUATED = 0;
  static constexpr SearchNodeState STATE_EVALUATING = 1;
  static constexpr SearchNodeState STATE_EXPANDED0 = 2;
  static constexpr SearchNodeState STATE_GROWING1 = 3;
  static constexpr SearchNodeState STATE_EXPANDED1 = 4;
  static constexpr SearchNodeState STATE_GROWING2 = 5;
  static constexpr SearchNodeState STATE_EXPANDED2 = 6;

  //During search, will only ever transition from NULL -> non-NULL.
  //Guaranteed to be non-NULL once state >= STATE_EXPANDED0.
  //After this is non-NULL, might rarely change mid-search, but it is guaranteed that old values remain
  //valid to access for the duration of the search and will not be deallocated.
  std::atomic<std::shared_ptr<NNOutput>*> nnOutput;
  std::atomic<std::shared_ptr<NNOutput>*> humanOutput;

  //Used to coordinate various multithreaded updates.
  //During search, for updating nnOutput when it needs recomputation at the root if it wasn't updated yet.
  //During various other events - for coordinating recursive updates of the tree or subtree value bias cleanup
  std::atomic<uint32_t> nodeAge;

  //During search, each will only ever transition from NULL -> non-NULL.
  //We get progressive resizing of children array simply overflowing on to successive later arrays.
  //Mutex pool guards insertion of children at a node. Reading of children is always fine.
  SearchChildPointer* children0; //Guaranteed to be non-NULL once state >= STATE_EXPANDED0
  SearchChildPointer* children1; //Guaranteed to be non-NULL once state >= STATE_EXPANDED1
  SearchChildPointer* children2; //Guaranteed to be non-NULL once state >= STATE_EXPANDED2

  //Lightweight mutable---------------------------------------------------------------
  //Protected under statsLock for writing
  NodeStatsAtomic stats;
  std::atomic<int32_t> virtualLosses;

  //Protected under the entryLock in subtreeValueBiasTableEntry
  //Used only if subtreeValueBiasTableEntry is not nullptr.
  //During search, subtreeValueBiasTableEntry itself is set upon creation of the node and remains constant
  //thereafter, making it safe to access without synchronization.
  double lastSubtreeValueBiasDeltaSum;
  double lastSubtreeValueBiasWeight;
  std::shared_ptr<SubtreeValueBiasEntry> subtreeValueBiasTableEntry;

  //Only valid if useGraphSearch is true.
  //Graph hash of this node except with an additional factor hashed in if this node would normally be a terminal node
  //and we are forcing it not to be due to being the root, or a child of the root after passing (if conservativePass)
  //or after any pass in the tree that friendlyPassOk marks as should not be ending the game.
  //Note that this is NOT a unique key for evaluations due to special behavior of the root where it affects how a
  //child handles passes after a pass move, this property is not encoded in the hash!
  //On the root, this might not be up to date outside of the search regarding forceNonTerminal.
  Hash128 graphHashMaybeForceNonTerminal;

  std::atomic<int32_t> dirtyCounter;

  //--------------------------------------------------------------------------------
  SearchNode(Player prevPla, bool forceNonTerminal, uint32_t mutexIdx, Hash128 graphHashMaybeForceNonTerminal);
  SearchNode(const SearchNode&, bool forceNonTerminal, bool copySubtreeValueBias);
  ~SearchNode();

  SearchNode& operator=(const SearchNode&) = delete;
  SearchNode(SearchNode&& other) = delete;
  SearchNode& operator=(SearchNode&& other) = delete;

  //The array returned by these is guaranteed not to be deallocated during the lifetime of a search or even
  //any time up until a new operation is peformed (such as starting a new search, or making a move, or setting params).
  SearchNodeChildrenReference getChildren();
  ConstSearchNodeChildrenReference getChildren() const;
  SearchNodeChildrenReference getChildren(SearchNodeState state);
  ConstSearchNodeChildrenReference getChildren(SearchNodeState state) const;

  //The NNOutput returned by these is guaranteed not to be deallocated during the lifetime of a search or even
  //any time up until a new operation is peformed (such as starting a new search, or making a move, or setting params).
  NNOutput* getNNOutput();
  const NNOutput* getNNOutput() const;
  NNOutput* getHumanOutput();
  const NNOutput* getHumanOutput() const;

  //Always replaces the current nnoutput, and stores the existing one in the thread for later deletion.
  //Returns true if there was NOT already an nnOutput
  bool storeNNOutput(std::shared_ptr<NNOutput>* newNNOutput, SearchThread& thread);
  bool storeHumanOutput(std::shared_ptr<NNOutput>* newHumanOutput, SearchThread& thread);
  //Only stores if there isn't an nnOutput already. Returns true if it was stored.
  bool storeNNOutputIfNull(std::shared_ptr<NNOutput>* newNNOutput);
  bool storeHumanOutputIfNull(std::shared_ptr<NNOutput>* newHumanOutput);

  //Used within search to update state and allocate children arrays
  void initializeChildren();
  bool maybeExpandChildrenCapacityForNewChild(SearchNodeState& stateValue, int numChildrenFullPlusOne);
  void collapseChildrenCapacity(int numGoodChildren);

private:
  bool tryExpandingChildrenCapacityAssumeFull(SearchNodeState& stateValue);
};

inline SearchChildPointer& SearchNodeChildrenReference::operator[](int i) {
  if(i < SearchChildrenSizes::SIZE0TOTAL) return node->children0[i];
  if(i < SearchChildrenSizes::SIZE1TOTAL) return node->children1[i-SearchChildrenSizes::SIZE0TOTAL];
  return node->children2[i-SearchChildrenSizes::SIZE1TOTAL];
}
inline const SearchChildPointer& ConstSearchNodeChildrenReference::operator[](int i) {
  if(i < SearchChildrenSizes::SIZE0TOTAL) return node->children0[i];
  if(i < SearchChildrenSizes::SIZE1TOTAL) return node->children1[i-SearchChildrenSizes::SIZE0TOTAL];
  return node->children2[i-SearchChildrenSizes::SIZE1TOTAL];
}



#endif
