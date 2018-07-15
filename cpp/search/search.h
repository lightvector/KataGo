#ifndef SEARCH_H
#define SEARCH_H

#include <memory>
#include <atomic>
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

struct NodeStats {
  uint64_t visits;
  double winLossValueSum;
  double scoreValueSum;

  NodeStats();
  ~NodeStats();

  NodeStats(const NodeStats& other);
  NodeStats& operator=(const NodeStats& other);

  double getCombinedValueSum(const SearchParams& searchParams) const;
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
  ostream* logout;

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

  SearchParams searchParams;

  string randSeed;

  //Contains all koHashes of positions/situations up to and including the root
  KoHashTable* rootKoHashTable;

  //Mutable---------------------------------------------------------------
  SearchNode* rootNode;

  //Services--------------------------------------------------------------
  MutexPool* mutexPool;
  NNEvaluator* nnEvaluator; //externally owned
  ostream* logout;

  Search(Rules rules, SearchParams params, uint32_t mutexPoolSize);
  ~Search();

  Search(const Search&) = delete;
  Search& operator=(const Search&) = delete;

  //Outside-of-search functions-------------------------------------------

  //Clear all results of search and sets a new position or something else
  void setPosition(Player pla, const Board& board, const BoardHistory& history);
  void setPlayerAndClearHistory(Player pla);
  void setRulesAndClearHistory(Rules rules);
  void setRootPassLegal(bool b);
  void setParams(SearchParams params);
  void setLog(ostream* logout);

  //Just directly clear search without changing anything
  void clearSearch();

  //Updates position and preserves the relevant subtree of search
  //If the move is not legal for the current player, returns false and does nothing, else returns true
  bool makeMove(Loc moveLoc);

  //Call once at the start of each search
  void beginSearch(const string& randSeed, NNEvaluator* nnEval);

  //Within-search functions, threadsafe-------------------------------------------
  void runSinglePlayout(SearchThread& thread);

  //Debug functions---------------------------------------------------------------
  void printPV(ostream& out, const SearchNode* node, int maxDepth);
  void printTree(ostream& out, const SearchNode* node, PrintTreeOptions options);

  //Helpers-----------------------------------------------------------------------
private:
  void maybeAddPolicyNoise(SearchThread& thread, SearchNode& node, bool isRoot) const;
  int getPos(Loc moveLoc) const;

  double getPlaySelectionValue(
    double nnPolicyProb, uint64_t childVisits,
    double childValueSum, Player pla
  ) const;
  double getExploreSelectionValue(
    double nnPolicyProb, uint64_t totalChildVisits, uint64_t childVisits,
    double childValueSum, double fpuValue, Player pla
  ) const;
  double getPlaySelectionValue(const SearchNode& parent, const SearchNode* child) const;
  double getExploreSelectionValue(const SearchNode& parent, const SearchNode* child, uint64_t totalChildVisits, double fpuValue) const;
  double getNewExploreSelectionValue(const SearchNode& parent, int movePos, uint64_t totalChildVisits, double fpuValue) const;

  void selectBestChildToDescend(
    const SearchThread& thread, const SearchNode& node, int& bestChildIdx, Loc& bestChildMoveLoc,
    int posesWithChildBuf[NNPos::NN_POLICY_SIZE],
    bool isRoot, bool checkLegalityOfNewMoves
  ) const;

  void playoutDescend(
    SearchThread& thread, SearchNode& node,
    double& retWinLossValue, double& retScoreValue,
    int posesWithChildBuf[NNPos::NN_POLICY_SIZE],
    bool isRoot
  );

  void printTreeHelper(
    ostream& out, const SearchNode* node, const PrintTreeOptions& options,
    string& prefix, uint64_t origVisits, int depth, double policyProb
  );

};

#endif
