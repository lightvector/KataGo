#ifndef SEARCH_SEARCH_H_
#define SEARCH_SEARCH_H_

#include <memory>

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/logger.h"
#include "../core/multithread.h"
#include "../core/threadsafequeue.h"
#include "../core/threadsafecounter.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
#include "../neuralnet/nneval.h"
#include "../search/analysisdata.h"
#include "../search/mutexpool.h"
#include "../search/patternbonustable.h"
#include "../search/subtreevaluebiastable.h"
#include "../search/searchparams.h"
#include "../search/searchprint.h"
#include "../search/timecontrols.h"

#include "../external/nlohmann_json/json.hpp"

struct SearchNode;
struct SearchThread;
struct Search;
struct DistributionTable;
struct PolicySortEntry;

struct ReportedSearchValues {
  double winValue;
  double lossValue;
  double noResultValue;
  double staticScoreValue;
  double dynamicScoreValue;
  double expectedScore;
  double expectedScoreStdev;
  double lead;
  double winLossValue;
  double utility;
  int64_t visits;

  ReportedSearchValues();
  ReportedSearchValues(const Search& search, double winLossValueAvg, double noResultValueAvg, double scoreMeanAvg,
                       double scoreMeanSqAvg, double leadAvg, double utilityAvg, int64_t totalVisits);
  ~ReportedSearchValues();
};

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
  ~NodeStatsAtomic();

  NodeStatsAtomic(const NodeStatsAtomic&) = delete;
  NodeStatsAtomic& operator=(const NodeStatsAtomic&) = delete;
  NodeStatsAtomic(NodeStatsAtomic&& other) = delete;
  NodeStatsAtomic& operator=(NodeStatsAtomic&& other) = delete;
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
  NodeStats(const NodeStatsAtomic& other);
  ~NodeStats();

  NodeStats(const NodeStats&) = default;
  NodeStats& operator=(const NodeStats&) = default;
  NodeStats(NodeStats&& other) = default;
  NodeStats& operator=(NodeStats&& other) = default;
};

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
public:
  SearchChildPointer();
  SearchNode* getIfAllocated();
  const SearchNode* getIfAllocated() const;
  SearchNode* getIfAllocatedRelaxed();
  void store(SearchNode* node);
  void storeRelaxed(SearchNode* node);
  bool storeIfNull(SearchNode* node);
};

struct SearchNode {
  //Locks------------------------------------------------------------------------------
  mutable std::atomic_flag statsLock = ATOMIC_FLAG_INIT;

  //Constant during search--------------------------------------------------------------
  Player nextPla;
  Loc prevMoveLoc;
  SearchNode* parent;
  Hash128 patternBonusHash;

  //Mutable---------------------------------------------------------------------------
  //During search, only ever transitions forward.
  std::atomic<int> state;
  static constexpr int STATE_UNEVALUATED = 0;
  static constexpr int STATE_EVALUATING = 1;
  static constexpr int STATE_EXPANDED0 = 2;
  static constexpr int STATE_GROWING1 = 3;
  static constexpr int STATE_EXPANDED1 = 4;
  static constexpr int STATE_GROWING2 = 5;
  static constexpr int STATE_EXPANDED2 = 6;

  //During search, will only ever transition from NULL -> non-NULL.
  //Guaranteed to be non-NULL once state >= STATE_EXPANDED0.
  //After this is non-NULL, might rarely change mid-search, but it is guaranteed that old values remain
  //valid to access for the duration of the search and will not be deallocated.
  std::atomic<std::shared_ptr<NNOutput>*> nnOutput;

  //Used to coordinate various multithreaded updates.
  //During search, for updating nnOutput when it needs recomputation at the root if it wasn't updated yet.
  //During various other events - for coordinating recursive updates of the tree or subtree value bias cleanup
  std::atomic<uint32_t> nodeAge;
  std::atomic<uint32_t> nodeAge2;

  //During search, each will only ever transition from NULL -> non-NULL.
  //We get progressive resizing of children array simply by moving on to a later array.
  SearchChildPointer* children0; //Guaranteed to be non-NULL once state >= STATE_EXPANDED0
  SearchChildPointer* children1; //Guaranteed to be non-NULL once state >= STATE_EXPANDED1
  SearchChildPointer* children2; //Guaranteed to be non-NULL once state >= STATE_EXPANDED2

  static constexpr int CHILDREN0SIZE = 8;
  static constexpr int CHILDREN1SIZE = 64;
  static constexpr int CHILDREN2SIZE = NNPos::MAX_NN_POLICY_SIZE;

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

  std::atomic<int32_t> dirtyCounter;

  //--------------------------------------------------------------------------------
  SearchNode(Player prevPla, Loc prevMoveLoc, SearchNode* parent);
  ~SearchNode();

  SearchNode(const SearchNode&) = delete;
  SearchNode& operator=(const SearchNode&) = delete;
  SearchNode(SearchNode&& other) = delete;
  SearchNode& operator=(SearchNode&& other) = delete;

  //The array returned by these is guaranteed not to be deallocated during the lifetime of a search or even
  //any time up until a new operation is peformed (such as starting a new search, or making a move, or setting params).
  SearchChildPointer* getChildren(int& childrenCapacity);
  const SearchChildPointer* getChildren(int& childrenCapacity) const;
  SearchChildPointer* getChildren(int state, int& childrenCapacity);
  const SearchChildPointer* getChildren(int state, int& childrenCapacity) const;

  int iterateAndCountChildren() const;
  static int iterateAndCountChildrenInArray(const SearchChildPointer* children, int childrenCapacity);

  //The NNOutput returned by these is guaranteed not to be deallocated during the lifetime of a search or even
  //any time up until a new operation is peformed (such as starting a new search, or making a move, or setting params).
  NNOutput* getNNOutput();
  const NNOutput* getNNOutput() const;

  //Always replaces the current nnoutput, and stores the existing one in the thread for later deletion.
  //Returns true if there was NOT already an nnOutput
  bool storeNNOutput(std::shared_ptr<NNOutput>* newNNOutput, SearchThread& thread);
  //Only stores if there isn't an nnOutput already. Returns true if it was stored.
  bool storeNNOutputIfNull(std::shared_ptr<NNOutput>* newNNOutput);

  //Used within search to update state and allocate children arrays
  void initializeChildren();
  bool maybeExpandChildrenCapacityForNewChild(int& stateValue, int numChildrenFullPlusOne);

private:
  int getChildrenCapacity(int stateValue) const;
  bool tryExpandingChildrenCapacityAssumeFull(int& stateValue);
};

//Per-thread state
struct SearchThread {
  int threadIdx;

  Player pla;
  Board board;
  BoardHistory history;

  Rand rand;

  NNResultBuf nnResultBuf;

  std::vector<MoreNodeStats> statsBuf;

  double upperBoundVisitsLeft;

  //Occasionally we may need to swap out an NNOutput from a node mid-search.
  //However, to prevent access-after-delete races, the thread that swaps one out stores
  //it here instead of deleting it, so that pointers and accesses to it remain valid.
  std::vector<std::shared_ptr<NNOutput>*> oldNNOutputsToCleanUp;

  SearchThread(int threadIdx, const Search& search);
  ~SearchThread();

  SearchThread(const SearchThread&) = delete;
  SearchThread& operator=(const SearchThread&) = delete;
};

struct Search {
  //Constant during search------------------------------------------------
  Player rootPla;
  Board rootBoard;
  BoardHistory rootHistory;
  Loc rootHintLoc;

  //External user-specified moves that are illegal or that should be nontrivially searched, and the number of turns for which they should
  //be excluded. Empty if not active, else of length MAX_ARR_SIZE and nonzero anywhere a move should be banned, for the number of ply
  //of depth that it should be banned.
  std::vector<int> avoidMoveUntilByLocBlack;
  std::vector<int> avoidMoveUntilByLocWhite;

  //Precomputed values at the root
  //If the board is symmetric, mask all the equivalent copies at root but leave one.
  bool isSymDupLoc[Board::MAX_ARR_SIZE];
  Color* rootSafeArea;
  //Used to center for dynamic scorevalue
  double recentScoreCenter;

  //If the opponent is mirroring, then the color of that opponent, for countering mirroring
  Player mirroringPla;
  double mirrorAdvantage; //Number of points the opponent wins by if mirror holds indefinitely.
  bool mirrorCenterIsSymmetric;

  bool alwaysIncludeOwnerMap;

  SearchParams searchParams;
  int64_t numSearchesBegun;
  uint32_t searchNodeAge;
  Player plaThatSearchIsFor;
  Player plaThatSearchIsForLastSearch;
  int64_t lastSearchNumPlayouts;
  double effectiveSearchTimeCarriedOver; //Effective search time carried over from previous moves due to ponder/tree reuse

  std::string randSeed;

  //Contains all koHashes of positions/situations up to and including the root
  KoHashTable* rootKoHashTable;

  //Precomputed distribution for downweighting child values based on their values
  DistributionTable* valueWeightDistribution;

  //Precomputed Fancymath::normToTApprox values, for a fixed Z
  double normToTApproxZ;
  std::vector<double> normToTApproxTable;

  //Mutable---------------------------------------------------------------
  SearchNode* rootNode;

  //Services--------------------------------------------------------------
  MutexPool* mutexPool;
  NNEvaluator* nnEvaluator; //externally owned
  int nnXLen;
  int nnYLen;
  int policySize;
  Rand nonSearchRand; //only for use not in search, since rand isn't threadsafe

  SubtreeValueBiasTable* subtreeValueBiasTable;

  //Pattern bonuses are currently only looked up for shapes completed by the player who the search is for.
  //Implicitly these utility adjustments "assume" the opponent likes the negative of our adjustments.
  PatternBonusTable* patternBonusTable;
  std::unique_ptr<PatternBonusTable> externalPatternBonusTable;

  Logger* logger;

  //Thread pool
  int numThreadsSpawned;
  std::thread* threads;
  ThreadSafeQueue<std::function<void(int)>*>* threadTasks;
  ThreadSafeCounter* threadTasksRemaining;

  //Occasionally we may need to swap out an NNOutput from a node mid-search.
  //However, to prevent access-after-delete races, this vector collects them after a thread exits, and is cleaned up
  //very lazily only when a new search begins or the search is cleared.
  std::mutex oldNNOutputsToCleanUpMutex;
  std::vector<std::shared_ptr<NNOutput>*> oldNNOutputsToCleanUp;

  //Note - randSeed controls a few things in the search, but a lot of the randomness actually comes from
  //random symmetries of the neural net evaluations, see nneval.h
  Search(SearchParams params, NNEvaluator* nnEval, Logger* logger, const std::string& randSeed);
  ~Search();

  Search(const Search&) = delete;
  Search& operator=(const Search&) = delete;
  Search(Search&&) = delete;
  Search& operator=(Search&&) = delete;

  //TOP-LEVEL OUTSIDE-OF-SEARCH CONTROL -----------------------------------------------------------
  //Functions for setting the board position or other parameters, clearing, and running search.
  //None of these top-level functions are thread-safe. They should only ever be called sequentially.

  const Board& getRootBoard() const;
  const BoardHistory& getRootHist() const;
  Player getRootPla() const;
  Player getPlayoutDoublingAdvantagePla() const;

  //Clear all results of search and sets a new position or something else
  void setPosition(Player pla, const Board& board, const BoardHistory& history);

  void setPlayerAndClearHistory(Player pla);
  void setPlayerIfNew(Player pla);
  void setKomiIfNew(float newKomi); //Does not clear history, does clear search unless komi is equal.
  void setRootHintLoc(Loc hintLoc);
  void setAvoidMoveUntilByLoc(const std::vector<int>& bVec, const std::vector<int>& wVec);
  void setAlwaysIncludeOwnerMap(bool b);
  void setParams(SearchParams params);
  void setParamsNoClearing(SearchParams params); //Does not clear search
  void setExternalPatternBonusTable(std::unique_ptr<PatternBonusTable>&& table);
  void setCopyOfExternalPatternBonusTable(const std::unique_ptr<PatternBonusTable>& table);
  void setNNEval(NNEvaluator* nnEval);

  //If the number of threads is reduced, this can free up some excess threads in the thread pool.
  //Calling this is never necessary, it may just reduce some resource use.
  void respawnThreads();

  //Just directly clear search without changing anything
  void clearSearch();

  //Updates position and preserves the relevant subtree of search
  //If the move is not legal for the specified player, returns false and does nothing, else returns true
  //In the case where the player was not the expected one moving next, also clears history.
  bool makeMove(Loc moveLoc, Player movePla);
  bool makeMove(Loc moveLoc, Player movePla, bool preventEncore);

  //isLegalTolerant also specially handles players moving multiple times in a row.
  bool isLegalTolerant(Loc moveLoc, Player movePla) const;
  bool isLegalStrict(Loc moveLoc, Player movePla) const;

  //Run an entire search from start to finish
  Loc runWholeSearchAndGetMove(Player movePla);
  void runWholeSearch(Player movePla);
  void runWholeSearch(std::atomic<bool>& shouldStopNow);

  //Pondering indicates that we are searching "for" the last player that we did a non-ponder search for, and should use ponder search limits.
  Loc runWholeSearchAndGetMove(Player movePla, bool pondering);
  void runWholeSearch(Player movePla, bool pondering);
  void runWholeSearch(std::atomic<bool>& shouldStopNow, bool pondering);

  void runWholeSearch(
    std::atomic<bool>& shouldStopNow,
    std::function<void()>* searchBegun, //If not null, will be called once search has begun and tree inspection is safe
    bool pondering,
    const TimeControls& tc,
    double searchFactor
  );

  //SEARCH RESULTS AND TREE INSPECTION -------------------------------------------------------------
  //All of these functions are safe to call in multithreadedly WHILE the search is ongoing, to print out
  //intermediate states of the search, so long as the search has initialized itself and actually begun.
  //In particular, they are allowed to run concurrently with runWholeSearch, so long as searchBegun has
  //been called-back, continuing up until the next call to any other top-level control function above or
  //the next runWholeSearch call.
  //They are NOT safe to call in parallel with any of the other top level-functions besides the search.

  //Choose a move at the root of the tree, with randomization, if possible.
  //Might return Board::NULL_LOC if there is no root.
  Loc getChosenMoveLoc();
  //Get the vector of values (e.g. modified visit counts) used to select a move.
  //Does take into account chosenMoveSubtract but does NOT apply temperature.
  //If somehow the max value is less than scaleMaxToAtLeast, scale it to at least that value.
  bool getPlaySelectionValues(
    std::vector<Loc>& locs, std::vector<double>& playSelectionValues, double scaleMaxToAtLeast
  ) const;
  bool getPlaySelectionValues(
    std::vector<Loc>& locs, std::vector<double>& playSelectionValues, std::vector<double>* retVisitCounts, double scaleMaxToAtLeast
  ) const;
  //Same, but works on a node within the search, not just the root
  bool getPlaySelectionValues(
    const SearchNode& node,
    std::vector<Loc>& locs, std::vector<double>& playSelectionValues, std::vector<double>* retVisitCounts, double scaleMaxToAtLeast,
    bool allowDirectPolicyMoves
  ) const;

  //Useful utility function exposed for outside use
  static uint32_t chooseIndexWithTemperature(Rand& rand, const double* relativeProbs, int numRelativeProbs, double temperature);
  static void computeDirichletAlphaDistribution(int policySize, const float* policyProbs, double* alphaDistr);
  static void addDirichletNoise(const SearchParams& searchParams, Rand& rand, int policySize, float* policyProbs);

  //Get the values recorded for the root node, if possible.
  bool getRootValues(ReportedSearchValues& values) const;
  //Same, same, but throws an exception if no values could be obtained
  ReportedSearchValues getRootValuesRequireSuccess() const;
  //Same, but works on a node within the search, not just the root
  bool getNodeValues(const SearchNode& node, ReportedSearchValues& values) const;
  bool getPrunedRootValues(ReportedSearchValues& values) const;

  //Same, but based only on the single raw neural net evaluation.
  bool getRootRawNNValues(ReportedSearchValues& values) const;
  ReportedSearchValues getRootRawNNValuesRequireSuccess() const;
  bool getNodeRawNNValues(const SearchNode& node, ReportedSearchValues& values) const;

  //Get the number of visits recorded for the root node
  int64_t getRootVisits() const;
  //Get the root node's policy prediction
  bool getPolicy(float policyProbs[NNPos::MAX_NN_POLICY_SIZE]) const;
  //Get the surprisingness (kl-divergence) of the search result given the policy prior, as well as the entropy of each.
  //Returns false if could not be computed.
  bool getPolicySurpriseAndEntropy(double& surpriseRet, double& searchEntropyRet, double& policyEntropyRet) const;
  double getPolicySurprise() const;

  void printPV(std::ostream& out, const SearchNode* node, int maxDepth) const;
  void printPVForMove(std::ostream& out, const SearchNode* node, Loc move, int maxDepth) const;
  void printTree(std::ostream& out, const SearchNode* node, PrintTreeOptions options, Player perspective) const;
  void printRootPolicyMap(std::ostream& out) const;
  void printRootOwnershipMap(std::ostream& out, Player perspective) const;
  void printRootEndingScoreValueBonus(std::ostream& out) const;

  //Get detailed analysis data, designed for lz-analyze and kata-analyze commands.
  void getAnalysisData(std::vector<AnalysisData>& buf, int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth) const;
  void getAnalysisData(const SearchNode& node, std::vector<AnalysisData>& buf, int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth) const;

  //Append the PV from node n onward (not including node n's move)
  void appendPV(std::vector<Loc>& buf, std::vector<int64_t>& visitsBuf, std::vector<Loc>& scratchLocs, std::vector<double>& scratchValues, const SearchNode* n, int maxDepth) const;
  //Append the PV from node n for specified move, assuming move is a child move of node n
  void appendPVForMove(std::vector<Loc>& buf, std::vector<int64_t>& visitsBuf, std::vector<Loc>& scratchLocs, std::vector<double>& scratchValues, const SearchNode* n, Loc move, int maxDepth) const;

  //Get the ownership map averaged throughout the search tree.
  //Must have ownership present on all neural net evals.
  //Safe to call DURING search, but NOT necessarily safe to call multithreadedly when updating the root position
  //or changing parameters or clearing search.
  //If node is not providied, defaults to using the root node.
  std::vector<double> getAverageTreeOwnership(double minWeight, const SearchNode* node = NULL) const;

  //Get ownership map as json
  nlohmann::json getJsonOwnershipMap(const Player pla, const Player perspective, const Board& board, const SearchNode* node, double ownershipMinWeight) const;
  //Fill json with analysis engine format information about search results
  bool getAnalysisJson(
    const Player perspective, const Board& board, const BoardHistory& hist,
    int analysisPVLen, double ownershipMinWeight, bool preventEncore, bool includePolicy,
    bool includeOwnership, bool includeMovesOwnership, bool includePVVisits,
    nlohmann::json& ret
  ) const;

  //Expert manual playout-by-playout interface------------------------------------------------
  void beginSearch(bool pondering);
  bool runSinglePlayout(SearchThread& thread, double upperBoundVisitsLeft);

  std::vector<SearchNode*> enumerateTreePostOrder();

  //Helpers-----------------------------------------------------------------------
  int getPos(Loc moveLoc) const;
  static double getScoreStdev(double scoreMeanAvg, double scoreMeanSqAvg);

private:
  static constexpr double POLICY_ILLEGAL_SELECTION_VALUE = -1e50;
  static constexpr double FUTILE_VISITS_PRUNE_VALUE = -1e40;
  static constexpr double EVALUATING_SELECTION_VALUE_PENALTY = 1e20;

  double getResultUtility(double winlossValue, double noResultValue) const;
  double getResultUtilityFromNN(const NNOutput& nnOutput) const;
  double interpolateEarly(double halflife, double earlyValue, double value) const;

  void spawnThreadsIfNeeded();
  void killThreads();
  int numAdditionalThreadsToUseForTasks() const;
  void performTaskWithThreads(std::function<void(int)>* task);

  void clearOldNNOutputs();
  void transferOldNNOutputs(SearchThread& thread);
  int findTopNPolicy(const SearchNode* node, int n, PolicySortEntry* sortedPolicyBuf) const;

  std::shared_ptr<NNOutput>* maybeAddPolicyNoiseAndTemp(SearchThread& thread, bool isRoot, NNOutput* oldNNOutput) const;
  bool isAllowedRootMove(Loc moveLoc) const;

  void computeRootNNEvaluation(NNResultBuf& nnResultBuf, bool includeOwnerMap);

  void computeRootValues();

  double numVisitsNeededToBeNonFutile(double maxVisitsMoveVisits);
  double computeUpperBoundVisitsLeftDueToTime(
    int64_t rootVisits, double timeUsed, double plannedTimeLimit
  );
  double recomputeSearchTimeLimit(const TimeControls& tc, double timeUsed, double searchFactor, int64_t rootVisits);

  double getScoreUtility(double scoreMeanAvg, double scoreMeanSqAvg) const;
  double getScoreUtilityDiff(double scoreMeanAvg, double scoreMeanSqAvg, double delta) const;
  double getApproxScoreUtilityDerivative(double scoreMean) const;
  double getUtilityFromNN(const NNOutput& nnOutput) const;
  double computeWeightFromNNOutput(const NNOutput* nnOutput) const;

  double getPatternBonus(Hash128 patternBonusHash, Player prevMovePla) const;

  //Parent must be locked
  double getEndingWhiteScoreBonus(const SearchNode& parent, const SearchNode* child) const;

  void downweightBadChildrenAndNormalizeWeight(
    int numChildren,
    double currentTotalWeight,
    double desiredTotalWeight,
    double amountToSubtract,
    double amountToPrune,
    std::vector<MoreNodeStats>& statsBuf
  ) const;

  //Parent must be locked
  void getSelfUtilityLCBAndRadius(const SearchNode& parent, const SearchNode* child, double& lcbBuf, double& radiusBuf) const;

  double getExploreSelectionValue(
    double nnPolicyProb, double totalChildWeight, double childWeight,
    double childUtility, double parentUtilityStdevFactor, Player pla
  ) const;
  double getExploreSelectionValueInverse(
    double exploreSelectionValue, double nnPolicyProb, double totalChildWeight,
    double childUtility, double parentUtilityStdevFactor, Player pla
  ) const;
  double getPassingScoreValueBonus(const SearchNode& parent, const SearchNode* child, double scoreValue) const;

  bool getPlaySelectionValues(
    const SearchNode& node,
    std::vector<Loc>& locs, std::vector<double>& playSelectionValues, std::vector<double>* retVisitCounts, double scaleMaxToAtLeast,
    bool allowDirectPolicyMoves, bool alwaysComputeLcb, bool neverUseLcb,
    double lcbBuf[NNPos::MAX_NN_POLICY_SIZE], double radiusBuf[NNPos::MAX_NN_POLICY_SIZE]
  ) const;

  //Parent must be locked
  double getExploreSelectionValue(
    const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
    double totalChildWeight, double fpuValue,
    double parentUtility, double parentWeightPerVisit, double parentUtilityStdevFactor,
    bool isDuringSearch, double maxChildWeight, SearchThread* thread
  ) const;
  double getNewExploreSelectionValue(
    const SearchNode& parent, float nnPolicyProb,
    double totalChildWeight, double fpuValue,
    double parentWeightPerVisit, double parentUtilityStdevFactor,
    double maxChildWeight, SearchThread* thread
  ) const;

  //Parent must be locked
  double getReducedPlaySelectionWeight(
    const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
    double totalChildWeight, double parentUtilityStdevFactor, double bestChildExploreSelectionValue
  ) const;

  double getFpuValueForChildrenAssumeVisited(
    const SearchNode& node, Player pla, bool isRoot, double policyProbMassVisited,
    double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
  ) const;

  double pruneNoiseWeight(std::vector<MoreNodeStats>& statsBuf, int numChildren, double totalChildWeight, const double* policyProbsBuf) const;

  void updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, bool isRoot);
  void recomputeNodeStats(SearchNode& node, SearchThread& thread, int32_t numVisitsToAdd, bool isRoot);
  void recursivelyRecomputeStats(SearchNode& node);

  void recursivelyDelete(SearchNode* node);
  void recursivelyRemoveSubtreeValueBiasAndDelete(const std::vector<SearchNode*>& nodes);
  void applyRecursivelyPostOrderMulithreaded(const std::vector<SearchNode*>& nodes, std::function<void(SearchNode*,int,bool)>* f);
  void applyRecursivelyPostOrderSinglethreadedHelper(SearchNode* node, int threadIdx, std::function<void(SearchNode*,int,bool)>* f);
  int applyRecursivelyPostOrderMulithreadedHelper(SearchNode* node, int threadIdx, PCG32* rand, std::function<void(SearchNode*,int,bool)>* f);

  void maybeRecomputeNormToTApproxTable();
  double getNormToTApproxForLCB(int64_t numVisits) const;

  void selectBestChildToDescend(
    SearchThread& thread, const SearchNode& node, int nodeState,
    int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc,
    bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
    bool isRoot
  ) const;

  void addLeafValue(
    SearchNode& node,
    double winLossValue,
    double noResultValue,
    double scoreMean,
    double scoreMeanSq,
    double lead,
    double weight,
    bool isTerminal,
    bool assumeNoExistingWeight
  );
  void addCurrentNNOutputAsLeafValue(SearchNode& node, bool assumeNoExistingWeight);

  void maybeRecomputeExistingNNOutput(
    SearchThread& thread, SearchNode& node, bool isRoot
  );
  bool initNodeNNOutput(
    SearchThread& thread, SearchNode& node,
    bool isRoot, bool skipCache, bool isReInit
  );

  bool playoutDescend(
    SearchThread& thread, SearchNode& node,
    bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
    bool isRoot
  );

  bool shouldSuppressPass(const SearchNode* n) const;

  AnalysisData getAnalysisDataOfSingleChild(
    const SearchNode* child, std::vector<Loc>& scratchLocs, std::vector<double>& scratchValues,
    Loc move, double policyProb, double fpuValue, double parentUtility, double parentWinLossValue,
    double parentScoreMean, double parentScoreStdev, double parentLead, int maxPVDepth
  ) const;

  void printPV(std::ostream& out, const std::vector<Loc>& buf) const;

  void printTreeHelper(
    std::ostream& out, const SearchNode* node, const PrintTreeOptions& options,
    std::string& prefix, int64_t origVisits, int depth, const AnalysisData& data, Player perspective
  ) const;

  double getAverageTreeOwnershipHelper(std::vector<double>& accum, double minWeight, double desiredWeight, const SearchNode* node) const;

};

#endif  // SEARCH_SEARCH_H_
