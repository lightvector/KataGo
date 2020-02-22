#ifndef SEARCH_SEARCH_H_
#define SEARCH_SEARCH_H_

#include <memory>

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/logger.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
#include "../neuralnet/nneval.h"
#include "../search/analysisdata.h"
#include "../search/mutexpool.h"
#include "../search/searchparams.h"
#include "../search/searchprint.h"
#include "../search/timecontrols.h"

struct SearchNode;
struct SearchThread;
struct Search;
struct DistributionTable;

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

  ReportedSearchValues();
  ~ReportedSearchValues();
};

struct NodeStats {
  int64_t visits;
  double winValueSum;
  double noResultValueSum;
  double scoreMeanSum;
  double scoreMeanSqSum;
  double leadSum;
  double utilitySum;
  double utilitySqSum;
  double weightSum;
  double weightSqSum;

  NodeStats();
  ~NodeStats();

  NodeStats(const NodeStats& other);
  NodeStats& operator=(const NodeStats& other);

  double getResultUtilitySum(const SearchParams& searchParams) const;
};

struct SearchNode {
  //Locks------------------------------------------------------------------------------
  uint32_t lockIdx;
  mutable std::atomic_flag statsLock = ATOMIC_FLAG_INIT;

  //Constant during search--------------------------------------------------------------
  Player nextPla;
  Loc prevMoveLoc;

  //Mutable---------------------------------------------------------------------------
  //All of these values are protected under the mutex indicated by lockIdx
  //nnOutput at a given node MAY be mutated during search, but of course will always be done under the lock.
  //The actual NNOutput object itself will NOT be mutated once set here, so having obtained a shared_ptr to
  //it while locked, it's safe to read it while unlocked.
  std::shared_ptr<NNOutput> nnOutput;
  uint32_t nnOutputAge;

  SearchNode** children;
  uint16_t numChildren;
  uint16_t childrenCapacity;

  //Lightweight mutable---------------------------------------------------------------
  //Protected under statsLock
  NodeStats stats;
  //Also protected under statsLock
  int32_t virtualLosses;

  //--------------------------------------------------------------------------------
  SearchNode(Search& search, SearchThread& thread, Loc prevMoveLoc);
  ~SearchNode();

  SearchNode(const SearchNode&) = delete;
  SearchNode& operator=(const SearchNode&) = delete;

  SearchNode(SearchNode&& other) noexcept;
  SearchNode& operator=(SearchNode&& other) noexcept;
};

//Per-thread state
struct SearchThread {
  int threadIdx;

  Player pla;
  Board board;
  BoardHistory history;

  Rand rand;

  NNResultBuf nnResultBuf;
  std::ostream* logStream;
  Logger* logger;

  std::vector<double> weightFactorBuf;
  std::vector<double> weightBuf;
  std::vector<double> weightSqBuf;
  std::vector<double> winValuesBuf;
  std::vector<double> noResultValuesBuf;
  std::vector<double> scoreMeansBuf;
  std::vector<double> scoreMeanSqsBuf;
  std::vector<double> leadsBuf;
  std::vector<double> utilityBuf;
  std::vector<double> utilitySqBuf;
  std::vector<double> selfUtilityBuf;
  std::vector<int64_t> visitsBuf;

  SearchThread(int threadIdx, const Search& search, Logger* logger);
  ~SearchThread();

  SearchThread(const SearchThread&) = delete;
  SearchThread& operator=(const SearchThread&) = delete;
};

struct Search {
  //Constant during search------------------------------------------------
  Player rootPla;
  Board rootBoard;
  BoardHistory rootHistory;
  bool rootPassLegal;

  //Precomputed values at the root
  Color* rootSafeArea;
  //Used to center for dynamic scorevalue
  double recentScoreCenter;

  bool alwaysIncludeOwnerMap;

  SearchParams searchParams;
  int64_t numSearchesBegun;
  uint32_t searchNodeAge;
  Player rootPlaDuringLastSearch;

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

  //Note - randSeed controls a few things in the search, but a lot of the randomness actually comes from
  //random symmetries of the neural net evaluations, see nneval.h
  Search(SearchParams params, NNEvaluator* nnEval, const std::string& randSeed);
  ~Search();

  Search(const Search&) = delete;
  Search& operator=(const Search&) = delete;

  //Outside-of-search functions-------------------------------------------

  const Board& getRootBoard() const;
  const BoardHistory& getRootHist() const;
  Player getRootPla() const;

  //Clear all results of search and sets a new position or something else
  void setPosition(Player pla, const Board& board, const BoardHistory& history);

  void setPlayerAndClearHistory(Player pla);
  void setKomiIfNew(float newKomi); //Does not clear history, does clear search unless komi is equal.
  void setRootPassLegal(bool b);
  void setAlwaysIncludeOwnerMap(bool b);
  void setParams(SearchParams params);
  void setParamsNoClearing(SearchParams params); //Does not clear search
  void setNNEval(NNEvaluator* nnEval);

  //Just directly clear search without changing anything
  void clearSearch();

  //Updates position and preserves the relevant subtree of search
  //If the move is not legal for the specified player, returns false and does nothing, else returns true
  //In the case where the player was not the expected one moving next, also clears history.
  bool makeMove(Loc moveLoc, Player movePla);
  bool makeMove(Loc moveLoc, Player movePla, bool preventEncore);
  bool isLegalTolerant(Loc moveLoc, Player movePla) const;
  bool isLegalStrict(Loc moveLoc, Player movePla) const;

  //Choose a move at the root of the tree, with randomization, if possible.
  //Might return Board::NULL_LOC if there is no root.
  Loc getChosenMoveLoc();
  //Get the vector of values (e.g. modified visit counts) used to select a move.
  //Does take into account chosenMoveSubtract but does NOT apply temperature.
  //If somehow the max value is less than scaleMaxToAtLeast, scale it to at least that value.
  bool getPlaySelectionValues(
    std::vector<Loc>& locs, std::vector<double>& playSelectionValues, double scaleMaxToAtLeast
  ) const;
  //Same, but works on a node within the search, not just the root
  bool getPlaySelectionValues(
    const SearchNode& node,
    std::vector<Loc>& locs, std::vector<double>& playSelectionValues, double scaleMaxToAtLeast,
    bool allowDirectPolicyMoves
  ) const;

  //Useful utility function exposed for outside use
  static uint32_t chooseIndexWithTemperature(Rand& rand, const double* relativeProbs, int numRelativeProbs, double temperature);
  static void addDirichletNoise(const SearchParams& searchParams, Rand& rand, int policySize, float* policyProbs);

  //Get the values recorded for the root node
  bool getRootValues(ReportedSearchValues& values) const;
  ReportedSearchValues getRootValuesAssertSuccess() const;
  //Same, but works on a node within the search, not just the root
  bool getNodeValues(const SearchNode& node, ReportedSearchValues& values) const;

  //Get the combined utility recorded for the root node
  double getRootUtility() const;
  //Get the number of visits recorded for the root node
  int64_t getRootVisits() const;

  //Run an entire search from start to finish
  Loc runWholeSearchAndGetMove(Player movePla, Logger& logger);
  void runWholeSearch(Player movePla, Logger& logger);
  void runWholeSearch(Logger& logger, std::atomic<bool>& shouldStopNow);

  Loc runWholeSearchAndGetMove(Player movePla, Logger& logger, bool pondering);
  void runWholeSearch(Player movePla, Logger& logger, bool pondering);
  void runWholeSearch(Logger& logger, std::atomic<bool>& shouldStopNow, bool pondering);

  void runWholeSearch(Logger& logger, std::atomic<bool>& shouldStopNow, bool pondering, const TimeControls& tc, double searchFactor);

  //Manual playout-by-playout interface------------------------------------------------

  //Call once at the start of each search
  void beginSearch();

  //Within-search functions, threadsafe-------------------------------------------
  void runSinglePlayout(SearchThread& thread);

  //Tree-inspection functions---------------------------------------------------------------
  void printPV(std::ostream& out, const SearchNode* node, int maxDepth) const;
  void printPVForMove(std::ostream& out, const SearchNode* node, Loc move, int maxDepth) const;
  void printTree(std::ostream& out, const SearchNode* node, PrintTreeOptions options, Player perspective) const;
  void printRootPolicyMap(std::ostream& out) const;
  void printRootOwnershipMap(std::ostream& out, Player perspective) const;
  void printRootEndingScoreValueBonus(std::ostream& out) const;

  //Safe to call DURING search, but NOT necessarily safe to call multithreadedly when updating the root position
  //or changing parameters or clearing search.
  void getAnalysisData(std::vector<AnalysisData>& buf, int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth) const;
  void getAnalysisData(const SearchNode& node, std::vector<AnalysisData>& buf, int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth) const;
  //Append the PV from node n onward (not including node n's move)
  void appendPV(std::vector<Loc>& buf, std::vector<Loc>& scratchLocs, std::vector<double>& scratchValues, const SearchNode* n, int maxDepth) const;
  //Append the PV from node n for specified move, assuming move is a child move of node n
  void appendPVForMove(std::vector<Loc>& buf, std::vector<Loc>& scratchLocs, std::vector<double>& scratchValues, const SearchNode* n, Loc move, int maxDepth) const;

  double getPolicySurprise() const;

  //Get the ownership map averaged throughout the search tree.
  //Must have ownership present on all neural net evals.
  //Safe to call DURING search, but NOT necessarily safe to call multithreadedly when updating the root position
  //or changing parameters or clearing search.
  std::vector<double> getAverageTreeOwnership(int64_t minVisits) const;

  int64_t numRootVisits() const;

  //Helpers-----------------------------------------------------------------------
private:
  static constexpr double POLICY_ILLEGAL_SELECTION_VALUE = -1e50;

  double getResultUtility(double winValue, double noResultValue) const;
  double getResultUtilityFromNN(const NNOutput& nnOutput) const;
  static double getScoreStdev(double scoreMean, double scoreMeanSq);
  double interpolateEarly(double halflife, double earlyValue, double value) const;

  void maybeAddPolicyNoiseAndTempAlreadyLocked(SearchThread& thread, SearchNode& node, bool isRoot) const;
  int getPos(Loc moveLoc) const;

  bool isAllowedRootMove(Loc moveLoc) const;

  void computeRootValues();

  double getScoreUtility(double scoreMeanSum, double scoreMeanSqSum, double weightSum) const;
  double getScoreUtilityDiff(double scoreMeanSum, double scoreMeanSqSum, double weightSum, double delta) const;
  double getUtilityFromNN(const NNOutput& nnOutput) const;

  //Parent must be locked
  double getEndingWhiteScoreBonus(const SearchNode& parent, const SearchNode* child) const;

  void getValueChildWeights(
    int numChildren,
    const std::vector<double>& childSelfValuesBuf,
    const std::vector<int64_t>& childVisitsBuf,
    std::vector<double>& resultBuf
  ) const;

  //Parent must be locked
  void getSelfUtilityLCBAndRadius(const SearchNode& parent, const SearchNode* child, double& lcbBuf, double& radiusBuf) const;

  float adjustExplorePolicyProb(
    const SearchThread& thread, const SearchNode& parent, Loc moveLoc, float nnPolicyProb,
    double parentUtility, double totalChildVisits, double childVisits, double& childUtility
  ) const;

  double getExploreSelectionValue(
    double nnPolicyProb, int64_t totalChildVisits, int64_t childVisits,
    double childUtility, Player pla
  ) const;
  double getExploreSelectionValueInverse(
    double exploreSelectionValue, double nnPolicyProb, int64_t totalChildVisits,
    double childUtility, Player pla
  ) const;
  double getPassingScoreValueBonus(const SearchNode& parent, const SearchNode* child, double scoreValue) const;

  bool getPlaySelectionValuesAlreadyLocked(
    const SearchNode& node,
    std::vector<Loc>& locs, std::vector<double>& playSelectionValues, double scaleMaxToAtLeast,
    bool allowDirectPolicyMoves, bool alwaysComputeLcb,
    double lcbBuf[NNPos::MAX_NN_POLICY_SIZE], double radiusBuf[NNPos::MAX_NN_POLICY_SIZE]
  ) const;

  //Parent must be locked
  double getExploreSelectionValue(
    const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
    int64_t totalChildVisits, double fpuValue, double parentUtility,
    bool isDuringSearch, const SearchThread* thread
  ) const;
  double getNewExploreSelectionValue(const SearchNode& parent, float nnPolicyProb, int64_t totalChildVisits, double fpuValue) const;

  //Parent must be locked
  int64_t getReducedPlaySelectionVisits(
    const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
    int64_t totalChildVisits, double bestChildExploreSelectionValue
  ) const;

  double getFpuValueForChildrenAssumeVisited(const SearchNode& node, Player pla, bool isRoot, double policyProbMassVisited, double& parentUtility) const;

  void updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, int32_t virtualLossesToSubtract, bool isRoot);
  void recomputeNodeStats(SearchNode& node, SearchThread& thread, int numVisitsToAdd, int32_t virtualLossesToSubtract, bool isRoot);
  void recursivelyRecomputeStats(SearchNode& node, SearchThread& thread, bool isRoot);

  void maybeRecomputeNormToTApproxTable();
  double getNormToTApproxForLCB(int64_t numVisits) const;

  void selectBestChildToDescend(
    const SearchThread& thread, const SearchNode& node, int& bestChildIdx, Loc& bestChildMoveLoc,
    bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
    bool isRoot
  ) const;

  void addLeafValue(SearchNode& node, double winValue, double noResultValue, double scoreMean, double scoreMeanSq, double lead, int32_t virtualLossesToSubtract);

  void maybeRecomputeExistingNNOutput(
    SearchThread& thread, SearchNode& node, bool isRoot
  );
  void initNodeNNOutput(
    SearchThread& thread, SearchNode& node,
    bool isRoot, bool skipCache, int32_t virtualLossesToSubtract, bool isReInit
  );

  void playoutDescend(
    SearchThread& thread, SearchNode& node,
    bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
    bool isRoot, int32_t virtualLossesToSubtract
  );

  bool shouldSuppressPassAlreadyLocked(const SearchNode* n) const;

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

  double getAverageTreeOwnershipHelper(std::vector<double>& accum, int64_t minVisits, double desiredWeight, const SearchNode* node) const;

};

#endif  // SEARCH_SEARCH_H_
