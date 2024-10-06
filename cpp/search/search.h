#ifndef SEARCH_SEARCH_H_
#define SEARCH_SEARCH_H_

#include <memory>
#include <unordered_set>

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
#include "../search/evalcache.h"
#include "../search/mutexpool.h"
#include "../search/reportedsearchvalues.h"
#include "../search/searchparams.h"
#include "../search/searchprint.h"
#include "../search/timecontrols.h"

#include "../external/nlohmann_json/json.hpp"

typedef int SearchNodeState; // See SearchNode::STATE_*

struct SearchNode;
struct SearchThread;
struct Search;
struct DistributionTable;
struct PatternBonusTable;
struct PolicySortEntry;
struct MoreNodeStats;
struct ReportedSearchValues;
struct SearchChildPointer;
struct SubtreeValueBiasTable;
struct SearchNodeTable;
struct SearchNodeChildrenReference;
struct ConstSearchNodeChildrenReference;

//Per-thread state
struct SearchThread {
  int threadIdx;

  Player pla;
  Board board;
  BoardHistory history;
  Hash128 graphHash;
  //The path we trace down the graph as we do a playout
  std::unordered_set<SearchNode*> graphPath;

  //Tracks whether this thread did something that "should" be counted as a playout
  //for the purpose of playout limits
  bool shouldCountPlayout;

  Rand rand;

  NNResultBuf nnResultBuf;
  std::vector<MoreNodeStats> statsBuf;

  double upperBoundVisitsLeft;

  //Occasionally we may need to swap out an NNOutput from a node mid-search.
  //However, to prevent access-after-delete races, the thread that swaps one out stores
  //it here instead of deleting it, so that pointers and accesses to it remain valid.
  std::vector<std::shared_ptr<NNOutput>*> oldNNOutputsToCleanUp;

  //Just controls some debug output
  std::set<Hash128> illegalMoveHashes;

  SearchThread(int threadIdx, const Search& search);
  ~SearchThread();

  SearchThread(const SearchThread&) = delete;
  SearchThread& operator=(const SearchThread&) = delete;
};

struct Search {
  //================================================================================================================
  // Constant/immutable during search
  //================================================================================================================

  Player rootPla;
  Board rootBoard;
  BoardHistory rootHistory;
  Hash128 rootGraphHash;
  Loc rootHintLoc;

  //External user-specified moves that are illegal or that should be nontrivially searched, and the number of turns for which they should
  //be excluded. Empty if not active, else of length MAX_ARR_SIZE and nonzero anywhere a move should be banned, for the number of ply
  //of depth that it should be banned.
  std::vector<int> avoidMoveUntilByLocBlack;
  std::vector<int> avoidMoveUntilByLocWhite;
  bool avoidMoveUntilRescaleRoot; // When avoiding moves at the root, rescale the root policy to sum to 1.

  //If rootSymmetryPruning==true and the board is symmetric, mask all the equivalent copies of each move except one.
  bool rootSymDupLoc[Board::MAX_ARR_SIZE];
  //If rootSymmetryPruning==true, symmetries under which the root board and history are invariant, including some heuristics for ko and encore-related state.
  std::vector<int> rootSymmetries;
  std::vector<int> rootPruneOnlySymmetries;

  //Strictly pass-alive areas in the root board position
  Color* rootSafeArea;
  //Used to center for dynamic scorevalue
  double recentScoreCenter;

  //If the opponent is mirroring, then the color of that opponent, for countering mirroring
  Player mirroringPla;
  double mirrorAdvantage; //Number of points the opponent wins by if mirror holds indefinitely.
  double mirrorCenterSymmetryError;

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

  //Pattern bonuses are currently only looked up for shapes completed by the player who the search is for.
  //Implicitly these utility adjustments "assume" the opponent likes the negative of our adjustments.
  PatternBonusTable* patternBonusTable;
  std::unique_ptr<PatternBonusTable> externalPatternBonusTable;

  EvalCacheTable* evalCache;

  Rand nonSearchRand; //only for use not in search, since rand isn't threadsafe

  //================================================================================================================
  // Externally owned values
  //================================================================================================================

  Logger* logger;
  NNEvaluator* nnEvaluator;
  NNEvaluator* humanEvaluator;
  int nnXLen;
  int nnYLen;
  int policySize;

  //================================================================================================================
  // Mutated during search
  //================================================================================================================

  SearchNode* rootNode;
  SearchNodeTable* nodeTable;
  MutexPool* mutexPool;
  SubtreeValueBiasTable* subtreeValueBiasTable;

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

  //================================================================================================================
  // Constructors and Destructors
  // search.cpp
  //================================================================================================================

  //Note - randSeed controls a few things in the search, but a lot of the randomness actually comes from
  //random symmetries of the neural net evaluations, see nneval.h
  Search(
    SearchParams params,
    NNEvaluator* nnEval,
    Logger* logger,
    const std::string& randSeed
  );
  Search(
    SearchParams params,
    NNEvaluator* nnEval,
    NNEvaluator* humanEval,
    Logger* logger,
    const std::string& randSeed
  );
  ~Search();

  Search(const Search&) = delete;
  Search& operator=(const Search&) = delete;
  Search(Search&&) = delete;
  Search& operator=(Search&&) = delete;

  //================================================================================================================
  // TOP-LEVEL OUTSIDE-OF-SEARCH CONTROL METHODS
  // search.cpp
  //
  // Functions for setting the board position or other parameters, clearing, and running search.
  // None of these top-level functions are thread-safe. They should only ever be called sequentially.
  //================================================================================================================

  const Board& getRootBoard() const;
  const BoardHistory& getRootHist() const;
  Player getRootPla() const;
  Player getPlayoutDoublingAdvantagePla() const;

  //Get the NNPos corresponding to a loc, convenience method
  int getPos(Loc moveLoc) const;

  //Clear all results of search and sets a new position or something else
  void setPosition(Player pla, const Board& board, const BoardHistory& history);

  void setPlayerAndClearHistory(Player pla);
  void setPlayerIfNew(Player pla);
  void setKomiIfNew(float newKomi); //Does not clear history, does clear search unless komi is equal.
  void setRootHintLoc(Loc hintLoc);
  void setAvoidMoveUntilByLoc(const std::vector<int>& bVec, const std::vector<int>& wVec);
  void setAvoidMoveUntilRescaleRoot(bool b);
  void setAlwaysIncludeOwnerMap(bool b);
  void setRootSymmetryPruningOnly(const std::vector<int>& rootPruneOnlySymmetries);
  void setParams(SearchParams params);
  void setParamsNoClearing(SearchParams params); //Does not clear search
  void setExternalPatternBonusTable(std::unique_ptr<PatternBonusTable>&& table);
  void setCopyOfExternalPatternBonusTable(const std::unique_ptr<PatternBonusTable>& table);
  void setNNEval(NNEvaluator* nnEval);

  //If the number of threads is reduced, this can free up some excess threads in the thread pool.
  //Calling this is never necessary, it may just reduce some resource use.
  //searchmultithreadhelpers.cpp
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

  //Without performing a whole search, recompute the root nn output for any root-level parameters.
  void maybeRecomputeRootNNOutput();

  //Expert manual playout-by-playout interface
  void beginSearch(bool pondering);
  bool runSinglePlayout(SearchThread& thread, double upperBoundVisitsLeft);

  //================================================================================================================
  // SEARCH RESULTS AND TREE INSPECTION METHODS
  // searchresults.cpp
  //
  // Functions for analyzing the results of search or getting back scores and analysis.
  //
  // All of these functions are safe to call in multithreadedly WHILE the search is ongoing, to print out
  // intermediate states of the search, so long as the search has initialized itself and actually begun.
  // In particular, they are allowed to run concurrently with runWholeSearch, so long as searchBegun has
  // been called-back, continuing up until the next call to any other top-level control function above or
  // the next runWholeSearch call.
  // They are NOT safe to call in parallel with any of the other top level-functions besides the search.
  //================================================================================================================

  //Choose a move at the root of the tree, with randomization, if possible.
  //Might return Board::NULL_LOC if there is no root, or no legal moves that aren't forcibly pruned, etc.
  Loc getChosenMoveLoc();
  //Get the vector of values (e.g. modified visit counts) used to select a move.
  //Does take into account chosenMoveSubtract but does NOT apply temperature.
  //If somehow the max value is less than scaleMaxToAtLeast, scale it to at least that value.
  //Always returns false in the case where no actual legal moves are found or there is no nnOutput or no root node.
  //If returning true, the is at least one loc and playSelectionValue.
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

  //Get the values recorded for the root node, if possible.
  bool getRootValues(ReportedSearchValues& values) const;
  //Same, same, but throws an exception if no values could be obtained
  ReportedSearchValues getRootValuesRequireSuccess() const;
  //Same, but works on a node within the search, not just the root
  bool getNodeValues(const SearchNode* node, ReportedSearchValues& values) const;
  bool getPrunedRootValues(ReportedSearchValues& values) const;
  bool getPrunedNodeValues(const SearchNode* node, ReportedSearchValues& values) const;

  const SearchNode* getRootNode() const;
  const SearchNode* getChildForMove(const SearchNode* node, Loc moveLoc) const;

  //Same, but based only on the single raw neural net evaluation.
  bool getRootRawNNValues(ReportedSearchValues& values) const;
  ReportedSearchValues getRootRawNNValuesRequireSuccess() const;
  bool getNodeRawNNValues(const SearchNode& node, ReportedSearchValues& values) const;

  //Get the number of visits recorded for the root node
  int64_t getRootVisits() const;
  //Get the root node's policy prediction
  bool getPolicy(float policyProbs[NNPos::MAX_NN_POLICY_SIZE]) const;
  bool getPolicy(const SearchNode* node, float policyProbs[NNPos::MAX_NN_POLICY_SIZE]) const;
  //Get the surprisingness (kl-divergence) of the search result given the policy prior, as well as the entropy of each.
  //Returns false if could not be computed.
  bool getPolicySurpriseAndEntropy(double& surpriseRet, double& searchEntropyRet, double& policyEntropyRet) const;
  bool getPolicySurpriseAndEntropy(double& surpriseRet, double& searchEntropyRet, double& policyEntropyRet, const SearchNode* node) const;
  double getPolicySurprise() const;

  void printPV(std::ostream& out, const SearchNode* node, int maxDepth) const;
  void printPVForMove(std::ostream& out, const SearchNode* node, Loc move, int maxDepth) const;
  void printTree(std::ostream& out, const SearchNode* node, PrintTreeOptions options, Player perspective) const;
  void printRootPolicyMap(std::ostream& out) const;
  void printRootOwnershipMap(std::ostream& out, Player perspective) const;
  void printRootEndingScoreValueBonus(std::ostream& out) const;

  //Get detailed analysis data, designed for lz-analyze and kata-analyze commands.
  void getAnalysisData(
    std::vector<AnalysisData>& buf, int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth, bool duplicateForSymmetries
  ) const;
  void getAnalysisData(
    const SearchNode& node, std::vector<AnalysisData>& buf, int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth, bool duplicateForSymmetries
  ) const;

  //Append the PV from node n onward (not including the move if any that reached node n)
  void appendPV(
    std::vector<Loc>& buf,
    std::vector<int64_t>& visitsBuf,
    std::vector<int64_t>& edgeVisitsBuf,
    std::vector<Loc>& scratchLocs,
    std::vector<double>& scratchValues,
    const SearchNode* n,
    int maxDepth
  ) const;
  //Append the PV from node n for specified move, assuming move is a child move of node n
  void appendPVForMove(
    std::vector<Loc>& buf,
    std::vector<int64_t>& visitsBuf,
    std::vector<int64_t>& edgeVisitsBuf,
    std::vector<Loc>& scratchLocs,
    std::vector<double>& scratchValues,
    const SearchNode* n,
    Loc move,
    int maxDepth
  ) const;

  //Get the ownership map averaged throughout the search tree.
  //Must have ownership present on all neural net evals.
  //Safe to call DURING search, but NOT necessarily safe to call multithreadedly when updating the root position
  //or changing parameters or clearing search.
  //If node is not provided, defaults to using the root node.
  std::vector<double> getAverageTreeOwnership(const SearchNode* node = NULL) const;
  std::pair<std::vector<double>,std::vector<double>> getAverageAndStandardDeviationTreeOwnership(const SearchNode* node = NULL) const;

  //Same, but applies symmetry and perspective
  std::vector<double> getAverageTreeOwnership(
    const Player perspective,
    const SearchNode* node,
    int symmetry
 ) const;
  std::pair<std::vector<double>,std::vector<double>> getAverageAndStandardDeviationTreeOwnership(
    const Player perspective,
    const SearchNode* node,
    int symmetry
  ) const;


  std::pair<double,double> getShallowAverageShorttermWLAndScoreError(const SearchNode* node = NULL) const;
  bool getSharpScore(const SearchNode* node, double& ret) const;

  //Fill json with analysis engine format information about search results
  bool getAnalysisJson(
    const Player perspective,
    int analysisPVLen, bool preventEncore, bool includePolicy,
    bool includeOwnership, bool includeOwnershipStdev, bool includeMovesOwnership, bool includeMovesOwnershipStdev, bool includePVVisits,
    nlohmann::json& ret
  ) const;


  //================================================================================================================
  // HELPER FUNCTIONS FOR THE SEARCH
  //================================================================================================================

private:
  static constexpr double POLICY_ILLEGAL_SELECTION_VALUE = -1e50;
  static constexpr double FUTILE_VISITS_PRUNE_VALUE = -1e40;
  static constexpr double EVALUATING_SELECTION_VALUE_PENALTY = 1e20;

  //----------------------------------------------------------------------------------------
  // Dirichlet noise and temperature
  // searchhelpers.cpp
  //----------------------------------------------------------------------------------------
public:
  static uint32_t chooseIndexWithTemperature(
    Rand& rand, const double* relativeProbs, int numRelativeProbs, double temperature, double onlyBelowProb, double* processedRelProbsBuf
  );
  static void computeDirichletAlphaDistribution(int policySize, const float* policyProbs, double* alphaDistr);
  static void addDirichletNoise(const SearchParams& searchParams, Rand& rand, int policySize, float* policyProbs);
private:
  std::shared_ptr<NNOutput>* maybeAddPolicyNoiseAndTemp(SearchThread& thread, bool isRoot, NNOutput* oldNNOutput) const;

  //----------------------------------------------------------------------------------------
  // Computing basic utility and scores
  // searchhelpers.cpp
  //----------------------------------------------------------------------------------------
  double getResultUtility(double winlossValue, double noResultValue) const;
  double getResultUtilityFromNN(const NNOutput& nnOutput) const;
  double getScoreUtility(double scoreMeanAvg, double scoreMeanSqAvg) const;
  double getScoreUtilityDiff(double scoreMeanAvg, double scoreMeanSqAvg, double delta) const;
  double getApproxScoreUtilityDerivative(double scoreMean) const;
  double getUtilityFromNN(const NNOutput& nnOutput) const;

  //----------------------------------------------------------------------------------------
  // Miscellaneous search biasing helpers, root move selection, etc.
  // searchhelpers.cpp
  //----------------------------------------------------------------------------------------
  bool isAllowedRootMove(Loc moveLoc) const;
  double getPatternBonus(Hash128 patternBonusHash, Player prevMovePla) const;
  double getEndingWhiteScoreBonus(const SearchNode& parent, Loc moveLoc) const;
  bool shouldSuppressPass(const SearchNode* n) const;

  double interpolateEarly(double halflife, double earlyValue, double value) const;

  // LCB helpers
  void getSelfUtilityLCBAndRadius(const SearchNode& parent, const SearchNode* child, int64_t edgeVisits, Loc moveLoc, double& lcbBuf, double& radiusBuf) const;
  void getSelfUtilityLCBAndRadiusZeroVisits(double& lcbBuf, double& radiusBuf) const;

  //----------------------------------------------------------------------------------------
  // Mirror handling logic
  // searchmirror.cpp
  //----------------------------------------------------------------------------------------
  void updateMirroring();
  bool isMirroringSinceSearchStart(const BoardHistory& threadHistory, int skipRecent) const;
  void maybeApplyAntiMirrorPolicy(
    float& nnPolicyProb,
    const Loc moveLoc,
    const float* policyProbs,
    const Player movePla,
    const SearchThread* thread
  ) const;
  void maybeApplyAntiMirrorForcedExplore(
    double& childUtility,
    const double parentUtility,
    const Loc moveLoc,
    const float* policyProbs,
    const double thisChildWeight,
    const double totalChildWeight,
    const Player movePla,
    const SearchThread* thread,
    const SearchNode& parent
  ) const;
  void hackNNOutputForMirror(std::shared_ptr<NNOutput>& result) const;

  //----------------------------------------------------------------------------------------
  // Recursive graph-walking and thread pooling
  // searchmultithreadhelpers.cpp
  //----------------------------------------------------------------------------------------
  int numAdditionalThreadsToUseForTasks() const;
  void spawnThreadsIfNeeded();
  void killThreads();
  void performTaskWithThreads(std::function<void(int)>* task, int capThreads);

  void applyRecursivelyPostOrderMulithreaded(const std::vector<SearchNode*>& nodes, std::function<void(SearchNode*,int)>* f);
  void applyRecursivelyPostOrderMulithreadedHelper(
    SearchNode* node, int threadIdx, PCG32* rand, std::unordered_set<SearchNode*>& nodeBuf, std::vector<int>& randBuf, std::function<void(SearchNode*,int)>* f
  );
  void applyRecursivelyAnyOrderMulithreaded(const std::vector<SearchNode*>& nodes, std::function<void(SearchNode*,int)>* f);
  void applyRecursivelyAnyOrderMulithreadedHelper(
    SearchNode* node, int threadIdx, PCG32* rand, std::unordered_set<SearchNode*>& nodeBuf, std::vector<int>& randBuf, std::function<void(SearchNode*,int)>* f
  );

public:
  std::vector<SearchNode*> enumerateTreePostOrder();
private:

  //----------------------------------------------------------------------------------------
  // Time management
  // searchtimehelpers.cpp
  //----------------------------------------------------------------------------------------
  double numVisitsNeededToBeNonFutile(double maxVisitsMoveVisits);
  double computeUpperBoundVisitsLeftDueToTime(
    int64_t rootVisits, double timeUsed, double plannedTimeLimit
  );
  double recomputeSearchTimeLimit(const TimeControls& tc, double timeUsed, double searchFactor, int64_t rootVisits);

  //----------------------------------------------------------------------------------------
  // Neural net queries
  // searchnnhelpers.cpp
  //----------------------------------------------------------------------------------------
  void computeRootNNEvaluation(NNResultBuf& nnResultBuf, bool includeOwnerMap);
  bool initNodeNNOutput(
    SearchThread& thread, SearchNode& node,
    bool isRoot, bool skipCache, bool isReInit
  );
  // Returns true if any recomputation happened
  bool maybeRecomputeExistingNNOutput(
    SearchThread& thread, SearchNode& node, bool isRoot
  );

  bool needsHumanOutputAtRoot() const;
  bool needsHumanOutputInTree() const;

  //----------------------------------------------------------------------------------------
  // Move selection during search
  // searchexplorehelpers.cpp
  //----------------------------------------------------------------------------------------
  double getExploreScaling(
    double totalChildWeight, double parentUtilityStdevFactor
  ) const;
  double getExploreScalingHuman(
    double totalChildWeight
  ) const;
  double getExploreSelectionValue(
    double exploreScaling,
    double nnPolicyProb,
    double childWeight,
    double childUtility,
    Player pla
  ) const;
  double getExploreSelectionValueInverse(
    double exploreScaling,
    double exploreSelectionValue,
    double nnPolicyProb,
    double childUtility,
    Player pla
  ) const;
  double getExploreSelectionValueOfChild(
    const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
    Loc moveLoc,
    double exploreScaling,
    double totalChildWeight, int64_t childEdgeVisits, double fpuValue,
    double parentUtility, double parentWeightPerVisit,
    bool isDuringSearch, bool antiMirror, double maxChildWeight,
    bool countEdgeVisit,
    SearchThread* thread
  ) const;
  double getNewExploreSelectionValue(
    const SearchNode& parent,
    double exploreScaling,
    float nnPolicyProb,
    double fpuValue,
    double parentWeightPerVisit,
    double maxChildWeight,
    bool countEdgeVisit,
    SearchThread* thread
  ) const;
  double getReducedPlaySelectionWeight(
    const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
    Loc moveLoc,
    double exploreScaling,
    int64_t childEdgeVisits,
    double bestChildExploreSelectionValue
  ) const;

  double getFpuValueForChildrenAssumeVisited(
    const SearchNode& node, Player pla, bool isRoot, double policyProbMassVisited,
    double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
  ) const;

  void selectBestChildToDescend(
    SearchThread& thread, const SearchNode& node, SearchNodeState nodeState,
    int& numChildrenFound, int& bestChildIdx, Loc& bestChildMoveLoc, bool& countEdgeVisit,
    bool isRoot
  ) const;

  //----------------------------------------------------------------------------------------
  // Update of node values during search
  // searchupdatehelpers.cpp
  //----------------------------------------------------------------------------------------

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

  double computeWeightFromNNOutput(const NNOutput* nnOutput) const;

  void updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, bool isRoot);
  void recomputeNodeStats(SearchNode& node, SearchThread& thread, int32_t numVisitsToAdd, bool isRoot);

  void adjustEvalsFromCacheHelper(
    EvalCacheEntry* evalCacheEntry,
    int64_t thisNodeVisits,
    double& winLossValueAvg,
    double& noResultValueAvg,
    double& scoreMeanAvg,
    double& scoreMeanSqAvg,
    double& leadAvg,
    double* utilityAvg
  );

  void downweightBadChildrenAndNormalizeWeight(
    int numChildren,
    double currentTotalWeight,
    double desiredTotalWeight,
    double amountToSubtract,
    double amountToPrune,
    std::vector<MoreNodeStats>& statsBuf
  ) const;

  double pruneNoiseWeight(std::vector<MoreNodeStats>& statsBuf, int numChildren, double totalChildWeight, const double* policyProbsBuf) const;

  //----------------------------------------------------------------------------------------
  // Allocation, search clearing and garbage collection
  // search.cpp
  //----------------------------------------------------------------------------------------
  uint32_t createMutexIdxForNode(SearchThread& thread) const;
  SearchNode* allocateOrFindNode(SearchThread& thread, Player nextPla, Loc bestChildMoveLoc, bool forceNonTerminal, Hash128 graphHash);
  void clearOldNNOutputs();
  void transferOldNNOutputs(SearchThread& thread);
  void removeSubtreeValueBias(SearchNode* node);
  void deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded(bool old);
  void deleteAllTableNodesMulithreaded();

  //----------------------------------------------------------------------------------------
  // Initialization and core search logic
  // search.cpp
  //----------------------------------------------------------------------------------------
  void computeRootValues(); // Helper for begin search
  void recursivelyRecomputeStats(SearchNode& node); // Helper for search initialization
  void recursivelyRecordEvalCache(SearchNode& n);

  bool playoutDescend(
    SearchThread& thread, SearchNode& node,
    bool isRoot
  );

  bool maybeCatchUpEdgeVisits(
    SearchThread& thread,
    SearchNode& node,
    SearchNode* child,
    const SearchNodeState& nodeState,
    const int bestChildIdx
  );

  //----------------------------------------------------------------------------------------
  // Private helpers for search results and analysis and top level move selection
  // searchresults.cpp
  //----------------------------------------------------------------------------------------
  bool getPlaySelectionValues(
    const SearchNode& node,
    std::vector<Loc>& locs, std::vector<double>& playSelectionValues, std::vector<double>* retVisitCounts, double scaleMaxToAtLeast,
    bool allowDirectPolicyMoves, bool alwaysComputeLcb, bool neverUseLcb,
    double lcbBuf[NNPos::MAX_NN_POLICY_SIZE], double radiusBuf[NNPos::MAX_NN_POLICY_SIZE]
  ) const;

  AnalysisData getAnalysisDataOfSingleChild(
    const SearchNode* child, int64_t edgeVisits, std::vector<Loc>& scratchLocs, std::vector<double>& scratchValues,
    Loc move, double policyProb, double fpuValue, double parentUtility, double parentWinLossValue,
    double parentScoreMean, double parentScoreStdev, double parentLead, int maxPVDepth
  ) const;

  void printPV(std::ostream& out, const std::vector<Loc>& buf) const;

  void printTreeHelper(
    std::ostream& out, const SearchNode* node, const PrintTreeOptions& options,
    std::string& prefix, int64_t origVisits, int depth, const AnalysisData& data, Player perspective
  ) const;

  bool getSharpScoreHelper(
    const SearchNode* node,
    std::unordered_set<const SearchNode*>& graphPath,
    double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE],
    double minProp,
    double desiredProp,
    double& ret
  ) const;
  void getShallowAverageShorttermWLAndScoreErrorHelper(
    const SearchNode* node,
    std::unordered_set<const SearchNode*>& graphPath,
    double policyProbsBuf[NNPos::MAX_NN_POLICY_SIZE],
    double minProp,
    double desiredProp,
    double& wlError,
    double& scoreError
  ) const;

  template<typename Func>
  bool traverseTreeForOwnership(
    double minProp,
    double pruneProp,
    double desiredProp,
    const SearchNode* node,
    std::unordered_set<const SearchNode*>& graphPath,
    Func& averaging
  ) const;
  template<typename Func>
  double traverseTreeForOwnershipChildren(
    double minProp,
    double pruneProp,
    double desiredProp,
    double thisNodeWeight,
    ConstSearchNodeChildrenReference children,
    double* childWeightBuf,
    int childrenCapacity,
    std::unordered_set<const SearchNode*>& graphPath,
    Func& averaging
  ) const;


};

#endif  // SEARCH_SEARCH_H_
