#ifndef SEARCH_H
#define SEARCH_H

#include <memory>
#include "../core/global.h"
#include "../core/hash.h"
#include "../core/logger.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
#include "../neuralnet/nneval.h"
#include "../search/mutexpool.h"
#include "../search/searchparams.h"
#include "../search/searchprint.h"

struct SearchNode;
struct SearchThread;
struct Search;
struct DistributionTable;

struct NodeStats {
  uint64_t visits;
  double winValueSum;
  double noResultValueSum;
  double scoreMeanSum;
  double scoreMeanSqSum;
  double valueSumWeight;

  NodeStats();
  ~NodeStats();

  NodeStats(const NodeStats& other);
  NodeStats& operator=(const NodeStats& other);

  double getResultUtilitySum(const SearchParams& searchParams) const;
};

struct SearchNode {
  //Locks------------------------------------------------------------------------------
  uint32_t lockIdx;
  mutable std::atomic_flag statsLock;

  //Constant during search--------------------------------------------------------------
  Player nextPla;
  Loc prevMoveLoc;

  //Mutable---------------------------------------------------------------------------
  //All of these values are protected under the mutex indicated by lockIdx
  shared_ptr<NNOutput> nnOutput; //Once set, constant thereafter

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
  ostream* logStream;
  Logger* logger;

  vector<double> valueChildWeightsBuf;
  vector<double> winValuesBuf;
  vector<double> noResultValuesBuf;
  vector<double> scoreMeansBuf;
  vector<double> scoreMeanSqsBuf;
  vector<double> utilityBuf;
  vector<uint64_t> visitsBuf;

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

  SearchParams searchParams;
  int64_t numSearchesBegun;

  string randSeed;

  //Contains all koHashes of positions/situations up to and including the root
  KoHashTable* rootKoHashTable;

  //Precomputed distribution for downweighting child values based on their values
  DistributionTable* valueWeightDistribution;

  //Mutable---------------------------------------------------------------
  SearchNode* rootNode;

  //Services--------------------------------------------------------------
  MutexPool* mutexPool;
  NNEvaluator* nnEvaluator; //externally owned
  int posLen;
  int policySize;
  Rand nonSearchRand; //only for use not in search, since rand isn't threadsafe

  //Note - randSeed controls a few things in the search, but a lot of the randomness actually comes from
  //random symmetries of the neural net evaluations, see nneval.h
  Search(SearchParams params, NNEvaluator* nnEval, const string& randSeed);
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
  void setRulesAndClearHistory(Rules rules, int encorePhase);
  void setKomi(float newKomi); //Does not clear history, does clear search unless komi is equal.
  void setRootPassLegal(bool b);
  void setParams(SearchParams params);
  void setNNEval(NNEvaluator* nnEval);

  //Just directly clear search without changing anything
  void clearSearch();

  //Updates position and preserves the relevant subtree of search
  //If the move is not legal for the specified player, returns false and does nothing, else returns true
  //In the case where the player was not the expected one moving next, also clears history.
  bool makeMove(Loc moveLoc, Player movePla);
  bool isLegal(Loc moveLoc, Player movePla) const;

  //Choose a move at the root of the tree, with randomization, if possible.
  //Might return Board::NULL_LOC if there is no root.
  Loc getChosenMoveLoc();
  //Get the vector of values (e.g. modified visit counts) used to select a move.
  //Does take into account chosenMoveSubtract but does NOT apply temperature.
  //If somehow the max value is less than scaleMaxToAtLeast, scale it to at least that value.
  bool getPlaySelectionValues(
    vector<Loc>& locs, vector<double>& playSelectionValues, double scaleMaxToAtLeast
  ) const;
  //Same, but works on a node within the search, not just the root
  bool getPlaySelectionValues(
    const SearchNode& node,
    vector<Loc>& locs, vector<double>& playSelectionValues, double scaleMaxToAtLeast
  ) const;


  //Get the values recorded for the root node
  bool getRootValues(
    double& winValue, double& lossValue, double& noResultValue, double& staticScoreValue, double& dynamicScoreValue, double& expectedScore
  ) const;
  //Same, but works on a node within the search, not just the root
  bool getNodeValues(
    const SearchNode& node,
    double& winValue, double& lossValue, double& noResultValue, double& staticScoreValue, double& dynamicScoreValue, double& expectedScore
  ) const;

  //Get the combined utility recorded for the root node
  double getRootUtility() const;

  //Run an entire search from start to finish
  //If recordUtilities is provided, and we're doing a singlethreaded search, will fill recordUtilities
  //with the root utility as of the end of each playout performed, up to the length of recordUtilities.
  Loc runWholeSearchAndGetMove(Player movePla, Logger& logger, vector<double>* recordUtilities);
  void runWholeSearch(Player movePla, Logger& logger, vector<double>* recordUtilities);
  void runWholeSearch(Logger& logger, std::atomic<bool>& shouldStopNow, vector<double>* recordUtilities);

  //Manual playout-by-playout interface------------------------------------------------

  //Call once at the start of each search
  void beginSearch(Logger& logger);

  //Within-search functions, threadsafe-------------------------------------------
  void runSinglePlayout(SearchThread& thread);

  //Tree-inspection functions---------------------------------------------------------------
  void printPV(ostream& out, const SearchNode* node, int maxDepth);
  void printTree(ostream& out, const SearchNode* node, PrintTreeOptions options);
  void printRootPolicyMap(ostream& out);
  void printRootOwnershipMap(ostream& out);
  void printRootEndingScoreValueBonus(ostream& out);

  uint64_t numRootVisits();

  //Helpers-----------------------------------------------------------------------
private:
  void maybeAddPolicyNoise(SearchThread& thread, SearchNode& node, bool isRoot) const;
  int getPos(Loc moveLoc) const;

  bool isAllowedRootMove(Loc moveLoc) const;

  void computeRootValues(Logger& logger);

  double getUtility(double resultUtilitySum, double scoreMeanSum, double scoreMeanSqSum, double valueSumWeight) const;
  double getUtilityFromNN(const NNOutput& nnOutput) const;

  double getEndingWhiteScoreBonus(const SearchNode& parent, const SearchNode* child) const;

  void getValueChildWeights(
    int numChildren,
    const vector<double>& childSelfValuesBuf,
    const vector<uint64_t>& childVisitsBuf,
    vector<double>& resultBuf
  ) const;

  double getPlaySelectionValue(
    double nnPolicyProb, uint64_t childVisits, Player pla
  ) const;
  double getExploreSelectionValue(
    double nnPolicyProb, uint64_t totalChildVisits, uint64_t childVisits,
    double childUtility, Player pla
  ) const;
  double getPassingScoreValueBonus(const SearchNode& parent, const SearchNode* child, double scoreValue) const;

  double getPlaySelectionValue(const SearchNode& parent, const SearchNode* child) const;
  double getExploreSelectionValue(const SearchNode& parent, const SearchNode* child, uint64_t totalChildVisits, double fpuValue) const;
  double getNewExploreSelectionValue(const SearchNode& parent, int movePos, uint64_t totalChildVisits, double fpuValue) const;

  void updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, int32_t virtualLossesToSubtract, bool isRoot);
  void recomputeNodeStats(SearchNode& node, SearchThread& thread, int numVisitsToAdd, int32_t virtualLossesToSubtract, bool isRoot);

  void selectBestChildToDescend(
    const SearchThread& thread, const SearchNode& node, int& bestChildIdx, Loc& bestChildMoveLoc,
    bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
    bool isRoot
  ) const;

  void setTerminalValue(SearchNode& node, double winValue, double noResultValue, double scoreMean, double scoreMeanSq, int32_t virtualLossesToSubtract);

  void initNodeNNOutput(
    SearchThread& thread, SearchNode& node,
    bool isRoot, bool skipCache, int32_t virtualLossesToSubtract, bool isReInit
  );

  void playoutDescend(
    SearchThread& thread, SearchNode& node,
    bool posesWithChildBuf[NNPos::MAX_NN_POLICY_SIZE],
    bool isRoot, int32_t virtualLossesToSubtract
  );

  void printTreeHelper(
    ostream& out, const SearchNode* node, const PrintTreeOptions& options,
    string& prefix, uint64_t origVisits, int depth, double policyProb, double valueWeight
  );

};

#endif
