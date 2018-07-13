#ifndef SEARCH_H
#define SEARCH_H

#include <memory>
#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
#include "../neuralnet/nneval.h"
#include "../search/mutexpool.h"
#include "../search/searchparams.h"
#include "../search/searchprint.h"

struct SearchNode;
struct SearchChild;
struct SearchChildren;
struct SearchThread;
struct Search;

struct SearchChildren {
  //Constant--------------------------------------------------------------

  //Mutable---------------------------------------------------------------
  SearchChild** children;
  uint16_t numChildren;
  uint16_t childrenCapacity;

  SearchChildren();
  ~SearchChildren();

  SearchChildren(const SearchChildren&) = delete;
  SearchChildren& operator=(const SearchChildren&) = delete;

  SearchChildren(SearchChildren&& other) noexcept;
  SearchChildren& operator=(SearchChildren&&) noexcept;
};

struct SearchNode {
  //Constant--------------------------------------------------------------
  uint32_t lockIdx;

  //Mutable---------------------------------------------------------------
  shared_ptr<NNOutput> nnOutput;
  SearchChildren children;

  uint64_t visits;
  double winLossValueSum;
  double scoreValueSum;

  uint64_t childVisits;

  //----------------------------------------------------------------------
  SearchNode(Search& search, SearchThread& thread);
  ~SearchNode();

  SearchNode(const SearchNode&) = delete;
  SearchNode& operator=(const SearchNode&) = delete;

  SearchNode(SearchNode&& other) noexcept;
  SearchNode& operator=(SearchNode&& other) noexcept;
};

struct SearchChild {
  //Constant--------------------------------------------------------------
  Loc moveLoc;
  SearchNode node;

  //Mutable---------------------------------------------------------------


  //----------------------------------------------------------------------
  SearchChild(Search& search, SearchThread& thread, Loc moveLoc);
  ~SearchChild();

  SearchChild(const SearchChild&) = delete;
  SearchChild& operator=(const SearchChild&) = delete;
};

//Per-thread state
struct SearchThread {
  int threadIdx;

  Player pla;
  Board board;
  BoardHistory history;

  Rand rand;

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

  double getCombinedValueSum(const SearchNode& node) const;
  double getPlaySelectionValue(
    double nnPolicyProb, uint64_t totalChildVisits, uint64_t childVisits,
    double childValueSum
  ) const;
  double getExploreSelectionValue(
    double nnPolicyProb, uint64_t totalChildVisits, uint64_t childVisits,
    double childValueSum, double fpuValue
  ) const;
  double getPlaySelectionValue(const SearchNode& parent, const SearchChild* child) const;
  double getExploreSelectionValue(const SearchNode& parent, const SearchChild* child, double fpuValue) const;
  double getNewExploreSelectionValue(const SearchNode& parent, int movePos, double fpuValue) const;

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
