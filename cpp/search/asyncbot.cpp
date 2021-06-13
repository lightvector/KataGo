#include "../search/asyncbot.h"

#include "../core/timer.h"

using namespace std;

static void searchThreadLoop(AsyncBot* asyncBot, Logger* logger) {
  try {
    asyncBot->internalSearchThreadLoop();
  }
  catch(const exception& e) {
    logger->write(string("ERROR: Async bot thread failed: ") + e.what());
  }
  catch(const string& e) {
    logger->write("ERROR: Async bot thread failed: " + e);
  }
  catch(...) {
    logger->write("ERROR: Async bot thread failed with unexpected throw");
  }
}

AsyncBot::AsyncBot(SearchParams params, NNEvaluator* nnEval, Logger* l, const string& randSeed)
  :search(NULL),logger(l),
   controlMutex(),threadWaitingToSearch(),userWaitingForStop(),searchThread(),
   isRunning(false),isPondering(false),isKilled(false),shouldStopNow(false),
   queuedSearchId(0),queuedOnMove(),timeControls(),searchFactor(1.0),
   analyzeCallbackPeriod(-1),analyzeCallback(),searchBegunCallback()
{
  search = new Search(params,nnEval,l,randSeed);
  searchThread = std::thread(searchThreadLoop,this,l);
}

AsyncBot::~AsyncBot() {
  stopAndWait();
  assert(!isRunning);
  {
    lock_guard<std::mutex> lock(controlMutex);
    isKilled = true;
  }
  threadWaitingToSearch.notify_all();
  searchThread.join();
  delete search;
}


const Board& AsyncBot::getRootBoard() const {
  return search->rootBoard;
}
const BoardHistory& AsyncBot::getRootHist() const {
  return search->rootHistory;
}
Player AsyncBot::getRootPla() const {
  return search->rootPla;
}
Player AsyncBot::getPlayoutDoublingAdvantagePla() const {
  return search->getPlayoutDoublingAdvantagePla();
}

Search* AsyncBot::getSearchStopAndWait() {
  stopAndWait();
  return search;
}
const Search* AsyncBot::getSearch() const {
  return search;
}
SearchParams AsyncBot::getParams() const {
  return search->searchParams;
}

void AsyncBot::setPosition(Player pla, const Board& board, const BoardHistory& history) {
  stopAndWait();
  search->setPosition(pla,board,history);
}
void AsyncBot::setPlayerAndClearHistory(Player pla) {
  stopAndWait();
  search->setPlayerAndClearHistory(pla);
}
void AsyncBot::setPlayerIfNew(Player pla) {
  stopAndWait();
  search->setPlayerIfNew(pla);
}
void AsyncBot::setKomiIfNew(float newKomi) {
  stopAndWait();
  search->setKomiIfNew(newKomi);
}
void AsyncBot::setAvoidMoveUntilByLoc(const std::vector<int>& bVec, const std::vector<int>& wVec) {
  stopAndWait();
  search->setAvoidMoveUntilByLoc(bVec,wVec);
}
void AsyncBot::setRootHintLoc(Loc loc) {
  stopAndWait();
  search->setRootHintLoc(loc);
}
void AsyncBot::setAlwaysIncludeOwnerMap(bool b) {
  stopAndWait();
  search->setAlwaysIncludeOwnerMap(b);
}
void AsyncBot::setParams(SearchParams params) {
  stopAndWait();
  search->setParams(params);
}
void AsyncBot::setParamsNoClearing(SearchParams params) {
  stopAndWait();
  search->setParamsNoClearing(params);
}
void AsyncBot::setExternalPatternBonusTable(std::unique_ptr<PatternBonusTable>&& table) {
  stopAndWait();
  search->setExternalPatternBonusTable(std::move(table));
}
void AsyncBot::clearSearch() {
  stopAndWait();
  search->clearSearch();
}

bool AsyncBot::makeMove(Loc moveLoc, Player movePla) {
  stopAndWait();
  return search->makeMove(moveLoc,movePla);
}
bool AsyncBot::makeMove(Loc moveLoc, Player movePla, bool preventEncore) {
  stopAndWait();
  return search->makeMove(moveLoc,movePla,preventEncore);
}

bool AsyncBot::isLegalTolerant(Loc moveLoc, Player movePla) const {
  return search->isLegalTolerant(moveLoc,movePla);
}
bool AsyncBot::isLegalStrict(Loc moveLoc, Player movePla) const {
  return search->isLegalStrict(moveLoc,movePla);
}

void AsyncBot::genMoveAsync(Player movePla, int searchId, const TimeControls& tc, const std::function<void(Loc,int)>& onMove) {
  genMoveAsync(movePla,searchId,tc,1.0,onMove,nullptr);
}

void AsyncBot::genMoveAsync(Player movePla, int searchId, const TimeControls& tc, double sf, const std::function<void(Loc,int)>& onMove) {
  genMoveAsync(movePla,searchId,tc,sf,onMove,nullptr);
}

void AsyncBot::genMoveAsync(Player movePla, int searchId, const TimeControls& tc, double sf, const std::function<void(Loc,int)>& onMove, const std::function<void()>& onSearchBegun) {
  unique_lock<std::mutex> lock(controlMutex);
  stopAndWaitAlreadyLocked(lock);
  assert(!isRunning);
  if(isKilled)
    return;

  if(movePla != search->rootPla)
    search->setPlayerAndClearHistory(movePla);

  queuedSearchId = searchId;
  queuedOnMove = onMove;
  isRunning = true;
  isPondering = false;
  shouldStopNow = false;
  timeControls = tc;
  searchFactor = sf;
  analyzeCallbackPeriod = -1;
  analyzeCallback = nullptr;
  searchBegunCallback = onSearchBegun;
  lock.unlock();
  threadWaitingToSearch.notify_all();
}

Loc AsyncBot::genMoveSynchronous(Player movePla, const TimeControls& tc) {
  return genMoveSynchronous(movePla,tc,1.0,nullptr);
}

Loc AsyncBot::genMoveSynchronous(Player movePla, const TimeControls& tc, double sf) {
  return genMoveSynchronous(movePla,tc,sf,nullptr);
}

Loc AsyncBot::genMoveSynchronous(Player movePla, const TimeControls& tc, double sf, const std::function<void()>& onSearchBegun) {
  Loc moveLoc = Board::NULL_LOC;
  std::function<void(Loc,int)> onMove = [&moveLoc](Loc loc, int searchId) {
    assert(searchId == 0);
    (void)searchId; //avoid warning when asserts disabled
    moveLoc = loc;
  };
  genMoveAsync(movePla,0,tc,sf,onMove,onSearchBegun);
  waitForSearchToEnd();
  return moveLoc;
}

void AsyncBot::ponder() {
  ponder(1.0);
}

void AsyncBot::ponder(double sf) {
  unique_lock<std::mutex> lock(controlMutex);
  if(isRunning)
    return;
  if(isKilled)
    return;

  queuedSearchId = 0;
  queuedOnMove = nullptr;
  isRunning = true;
  isPondering = true; //True - we are searching on the opponent's turn "for" the opponent's opponent
  shouldStopNow = false;
  timeControls = TimeControls(); //Blank time controls since opponent's clock is running, not ours, so no cap other than searchFactor
  searchFactor = sf;
  analyzeCallbackPeriod = -1;
  analyzeCallback = nullptr;
  searchBegunCallback = nullptr;
  lock.unlock();
  threadWaitingToSearch.notify_all();
}
void AsyncBot::analyzeAsync(Player movePla, double sf, double callbackPeriod, const std::function<void(const Search* search)>& callback) {
  unique_lock<std::mutex> lock(controlMutex);
  stopAndWaitAlreadyLocked(lock);
  assert(!isRunning);
  if(isKilled)
    return;

  if(movePla != search->rootPla)
    search->setPlayerAndClearHistory(movePla);

  queuedSearchId = 0;
  queuedOnMove = nullptr;
  isRunning = true;
  isPondering = false; //This should indeed be false because we are searching for the current player, not the last player we did a regular search for.
  shouldStopNow = false;
  timeControls = TimeControls(); //Blank time controls since no clock is not running, we don't cap search time other than through searchFactor.
  searchFactor = sf;
  analyzeCallbackPeriod = callbackPeriod;
  analyzeCallback = callback;
  searchBegunCallback = nullptr;
  lock.unlock();
  threadWaitingToSearch.notify_all();
}

void AsyncBot::genMoveAsyncAnalyze(
  Player movePla, int searchId, const TimeControls& tc, double sf, const std::function<void(Loc,int)>& onMove,
  double callbackPeriod, const std::function<void(const Search* search)>& callback
) {
  genMoveAsyncAnalyze(movePla, searchId, tc, sf, onMove, callbackPeriod, callback, nullptr);
}

void AsyncBot::genMoveAsyncAnalyze(
  Player movePla, int searchId, const TimeControls& tc, double sf, const std::function<void(Loc,int)>& onMove,
  double callbackPeriod, const std::function<void(const Search* search)>& callback,
  const std::function<void()>& onSearchBegun
) {
  unique_lock<std::mutex> lock(controlMutex);
  stopAndWaitAlreadyLocked(lock);
  assert(!isRunning);
  if(isKilled)
    return;

  if(movePla != search->rootPla)
    search->setPlayerAndClearHistory(movePla);

  queuedSearchId = searchId;
  queuedOnMove = onMove;
  isRunning = true;
  isPondering = false;
  shouldStopNow = false;
  timeControls = tc;
  searchFactor = sf;
  analyzeCallbackPeriod = callbackPeriod;
  analyzeCallback = callback;
  searchBegunCallback = onSearchBegun;
  lock.unlock();
  threadWaitingToSearch.notify_all();
}

Loc AsyncBot::genMoveSynchronousAnalyze(
  Player movePla, const TimeControls& tc, double sf,
  double callbackPeriod, const std::function<void(const Search* search)>& callback
) {
  return genMoveSynchronousAnalyze(movePla, tc, sf, callbackPeriod, callback, nullptr);
}

Loc AsyncBot::genMoveSynchronousAnalyze(
  Player movePla, const TimeControls& tc, double sf,
  double callbackPeriod, const std::function<void(const Search* search)>& callback,
  const std::function<void()>& onSearchBegun
) {
  Loc moveLoc = Board::NULL_LOC;
  std::function<void(Loc,int)> onMove = [&moveLoc](Loc loc, int searchId) {
    assert(searchId == 0);
    (void)searchId; //avoid warning when asserts disabled
    moveLoc = loc;
  };
  genMoveAsyncAnalyze(movePla,0,tc,sf,onMove,callbackPeriod,callback,onSearchBegun);
  waitForSearchToEnd();
  return moveLoc;
}

void AsyncBot::stopWithoutWait() {
  shouldStopNow.store(true);
}

void AsyncBot::setKilled() {
  lock_guard<std::mutex> lock(controlMutex);
  isKilled = true;
  shouldStopNow.store(true);
  threadWaitingToSearch.notify_all();
}

void AsyncBot::stopAndWait() {
  shouldStopNow.store(true);
  waitForSearchToEnd();
}

void AsyncBot::stopAndWaitAlreadyLocked(unique_lock<std::mutex>& lock) {
  shouldStopNow.store(true);
  waitForSearchToEndAlreadyLocked(lock);
}

void AsyncBot::waitForSearchToEnd() {
  unique_lock<std::mutex> lock(controlMutex);
  while(isRunning)
    userWaitingForStop.wait(lock);
}

void AsyncBot::waitForSearchToEndAlreadyLocked(unique_lock<std::mutex>& lock) {
  while(isRunning)
    userWaitingForStop.wait(lock);
}

void AsyncBot::internalSearchThreadLoop() {
  unique_lock<std::mutex> lock(controlMutex);
  while(true) {
    while(!isRunning && !isKilled)
      threadWaitingToSearch.wait(lock);
    if(isKilled) {
      isRunning = false;
      isPondering = false;
      userWaitingForStop.notify_all();
      break;
    }

    bool pondering = isPondering;
    TimeControls tc = timeControls;
    double callbackPeriod = analyzeCallbackPeriod;
    //Make local copies just in case, to simplify thread reasoning for the member fields
    std::function<void(const Search*)> analyzeCallbackLocal = analyzeCallback;
    std::function<void()> searchBegunCallbackLocal = searchBegunCallback;
    lock.unlock();

    //Make sure we don't feed in absurdly large numbers, this seems to cause wait_for to hang.
    //For a long period, just don't do callbacks.
    if(callbackPeriod >= 10000000)
      callbackPeriod = -1;

    //Kick off analysis callback loop if desired
    condition_variable callbackLoopWaiting;
    atomic<bool> callbackLoopShouldStop(false);
    atomic<bool> isSearchBegun(false);
    std::function<void()> searchBegun = [&isSearchBegun,&searchBegunCallbackLocal]() {
      isSearchBegun.store(true,std::memory_order_release);
      if(searchBegunCallbackLocal)
        searchBegunCallbackLocal();
    };
    auto callbackLoop = [this,callbackPeriod,&analyzeCallbackLocal,&callbackLoopWaiting,&callbackLoopShouldStop,&isSearchBegun]() {
      unique_lock<std::mutex> callbackLock(controlMutex);
      while(true) {
        callbackLoopWaiting.wait_for(
          callbackLock,
          std::chrono::duration<double>(callbackPeriod),
          [&callbackLoopShouldStop](){return callbackLoopShouldStop.load();}
        );
        if(callbackLoopShouldStop.load())
          break;
        if(!isSearchBegun.load(std::memory_order_acquire))
          continue;
        callbackLock.unlock();
        analyzeCallbackLocal(search);
        callbackLock.lock();
      }
      callbackLock.unlock();
    };

    std::thread callbackLoopThread;
    if(callbackPeriod >= 0 && analyzeCallbackLocal) {
      callbackLoopThread = std::thread(callbackLoop);
    }

    search->runWholeSearch(shouldStopNow,&searchBegun,pondering,tc,searchFactor);
    Loc moveLoc = search->getChosenMoveLoc();

    if(callbackPeriod >= 0) {
      lock.lock();
      callbackLoopShouldStop.store(true);
      callbackLoopWaiting.notify_all();
      lock.unlock();
      callbackLoopThread.join();
    }

    lock.lock();
    //Call queuedOnMove under the lock just in case
    if(queuedOnMove)
      queuedOnMove(moveLoc,queuedSearchId);
    isRunning = false;
    isPondering = false;
    userWaitingForStop.notify_all();
  }
}
