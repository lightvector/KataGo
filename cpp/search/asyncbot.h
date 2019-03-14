#ifndef ASYNCBOT_H
#define ASYNCBOT_H

#include "../search/search.h"

class AsyncBot {
 public:
  AsyncBot(SearchParams params, NNEvaluator* nnEval, Logger* logger, const string& randSeed);
  ~AsyncBot();

  AsyncBot(const AsyncBot& other) = delete;
  AsyncBot& operator=(const AsyncBot& other) = delete;

  //Unless otherwise specified, functions in this class are NOT threadsafe, although they may spawn off asynchronous events.
  //Usage of this API should be single-threaded!

  const Board& getRootBoard() const;
  const BoardHistory& getRootHist() const;
  Player getRootPla() const;
  SearchParams getParams() const;

  Search* getSearch();

  //Setup, same as in search.h
  //Calling any of these will stop any ongoing search, waiting for a full stop.
  void setPosition(Player pla, const Board& board, const BoardHistory& history);
  void setRulesAndClearHistory(Rules rules, int encorePhase);
  void setKomiIfNew(float newKomi);
  void setRootPassLegal(bool b);
  void setParams(SearchParams params);
  void setPlayerIfNew(Player movePla);
  void clearSearch();

  //Updates position and preserves the relevant subtree of search
  //Will stop any ongoing search, waiting for a full stop.
  //If the move is not legal for the current player, returns false and does nothing, else returns true
  bool makeMove(Loc moveLoc, Player movePla);
  bool isLegal(Loc moveLoc, Player movePla) const;

  //Begin searching and produce a move.
  //Will stop any ongoing search, waiting for a full stop.
  //Asynchronously calls the provided function upon success, passing back the move and provided searchId.
  //The provided callback is expected to terminate quickly and should NOT call back into this API.
  void genMove(Player movePla, int searchId, const TimeControls& tc, std::function<void(Loc,int)> onMove);
  void genMove(Player movePla, int searchId, const TimeControls& tc, double searchFactor, std::function<void(Loc,int)> onMove);

  //Same as genMove, but waits directly for the move and returns it here.
  Loc genMoveSynchronous(Player movePla, const TimeControls& tc);
  Loc genMoveSynchronous(Player movePla, const TimeControls& tc, double searchFactor);

  //Begin pondering, returning immediately. Future genMoves may be faster if this is called.
  //Will not stop any ongoing searches.
  void ponder();
  void ponder(double searchFactor);

  //Terminate any existing searches, and then begin pondering while periodically calling the specified callback
  void analyze(Player movePla, double searchFactor, double callbackPeriod, std::function<void(Search* search)> callback);

  
  //Signal an ongoing genMove or ponder to stop as soon as possible, and wait for the stop to happen.
  //Safe to call even if nothing is running.
  void stopAndWait();


 private:
  Search* search;
  Logger* logger;

  mutex controlMutex;
  condition_variable threadWaitingToSearch;
  condition_variable userWaitingForStop;
  thread searchThread;

  bool isRunning;
  bool isPondering;
  bool isKilled;
  atomic<bool> shouldStopNow;
  int queuedSearchId;
  std::function<void(Loc,int)> queuedOnMove;
  TimeControls timeControls;
  double searchFactor;
  double analyzeCallbackPeriod;
  std::function<void(Search* search)> analyzeCallback;

  void stopAndWaitAlreadyLocked(unique_lock<std::mutex>& lock);
  void waitForSearchToEnd();
  void waitForSearchToEndAlreadyLocked(unique_lock<std::mutex>& lock);

 public:
  //Only for internal use
  void internalSearchThreadLoop();
};


#endif
