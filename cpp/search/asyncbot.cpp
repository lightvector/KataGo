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
   analyzeCallbackPeriod(-1),analyzeCallback()
{
  search = new Search(params,nnEval,randSeed);
  searchThread = std::thread(searchThreadLoop,this,l);
}

AsyncBot::~AsyncBot() {
  stopAndWait();
  assert(!isRunning);
  assert(!isKilled);
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
void AsyncBot::setKomiIfNew(float newKomi) {
  stopAndWait();
  search->setKomiIfNew(newKomi);
}
void AsyncBot::setRootPassLegal(bool b) {
  stopAndWait();
  search->setRootPassLegal(b);
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
void AsyncBot::setPlayerIfNew(Player movePla) {
  stopAndWait();
  if(movePla != search->rootPla)
    search->setPlayerAndClearHistory(movePla);
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

void AsyncBot::genMove(Player movePla, int searchId, const TimeControls& tc, std::function<void(Loc,int)> onMove) {
  genMove(movePla,searchId,tc,1.0,onMove);
}

void AsyncBot::genMove(Player movePla, int searchId, const TimeControls& tc, double sf, std::function<void(Loc,int)> onMove) {
  unique_lock<std::mutex> lock(controlMutex);
  stopAndWaitAlreadyLocked(lock);
  assert(!isRunning);
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
  analyzeCallback = std::function<void(Search*)>();
  lock.unlock();
  threadWaitingToSearch.notify_all();
}

Loc AsyncBot::genMoveSynchronous(Player movePla, const TimeControls& tc) {
  return genMoveSynchronous(movePla,tc,1.0);
}

Loc AsyncBot::genMoveSynchronous(Player movePla, const TimeControls& tc, double sf) {
  Loc moveLoc = Board::NULL_LOC;
  std::function<void(Loc,int)> onMove = [&moveLoc](Loc loc, int searchId) {
    assert(searchId == 0);
    (void)searchId; //avoid warning when asserts disabled
    moveLoc = loc;
  };
  genMove(movePla,0,tc,sf,onMove);
  waitForSearchToEnd();
  return moveLoc;
}

static void ignoreMove(Loc loc, int searchId) {
  (void)loc;
  (void)searchId;
}

void AsyncBot::ponder() {
  ponder(1.0);
}

void AsyncBot::ponder(double sf) {
  unique_lock<std::mutex> lock(controlMutex);
  if(isRunning)
    return;

  queuedSearchId = 0;
  queuedOnMove = std::function<void(Loc,int)>(ignoreMove);
  isRunning = true;
  isPondering = true;
  shouldStopNow = false;
  timeControls = TimeControls();
  searchFactor = sf;
  analyzeCallbackPeriod = -1;
  analyzeCallback = std::function<void(Search*)>();
  lock.unlock();
  threadWaitingToSearch.notify_all();
}

void AsyncBot::analyze(Player movePla, double sf, double callbackPeriod, std::function<void(Search* search)> callback) {
  unique_lock<std::mutex> lock(controlMutex);
  stopAndWaitAlreadyLocked(lock);
  assert(!isRunning);
  if(movePla != search->rootPla)
    search->setPlayerAndClearHistory(movePla);

  queuedSearchId = 0;
  queuedOnMove = std::function<void(Loc,int)>(ignoreMove);
  isRunning = true;
  isPondering = true;
  shouldStopNow = false;
  timeControls = TimeControls();
  searchFactor = sf;
  analyzeCallbackPeriod = callbackPeriod;
  analyzeCallback = callback;
  lock.unlock();
  threadWaitingToSearch.notify_all();
}

void AsyncBot::genMoveAnalyze(
  Player movePla, int searchId, const TimeControls& tc, double sf, std::function<void(Loc,int)> onMove,
  double callbackPeriod, std::function<void(Search* search)> callback
) {
  unique_lock<std::mutex> lock(controlMutex);
  stopAndWaitAlreadyLocked(lock);
  assert(!isRunning);
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
  lock.unlock();
  threadWaitingToSearch.notify_all();
}

Loc AsyncBot::genMoveSynchronousAnalyze(
  Player movePla, const TimeControls& tc, double sf,
  double callbackPeriod, std::function<void(Search* search)> callback
) {
  Loc moveLoc = Board::NULL_LOC;
  std::function<void(Loc,int)> onMove = [&moveLoc](Loc loc, int searchId) {
    assert(searchId == 0);
    (void)searchId; //avoid warning when asserts disabled
    moveLoc = loc;
  };
  genMoveAnalyze(movePla,0,tc,sf,onMove,callbackPeriod,callback);
  waitForSearchToEnd();
  return moveLoc;
}

void AsyncBot::stopWithoutWait() {
  shouldStopNow.store(true);
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
    if(isKilled)
      break;

    bool pondering = isPondering;
    TimeControls tc = timeControls;
    double callbackPeriod = analyzeCallbackPeriod;
    std::function<void(Search*)> callback = analyzeCallback;
    lock.unlock();

    //Make sure we don't feed in absurdly large numbers, this seems to cause wait_for to hang.
    //For a long period, just don't do callbacks.
    if(callbackPeriod >= 10000000)
      callbackPeriod = -1;

    //Kick off analysis callback loop if desired
    condition_variable callbackLoopWaiting;
    atomic<bool> callbackLoopShouldStop(false);
    atomic<bool> searchBegun(false);
    auto callbackLoop = [this,callbackPeriod,&callback,&callbackLoopWaiting,&callbackLoopShouldStop,&searchBegun]() {
      unique_lock<std::mutex> callbackLock(controlMutex);
      while(true) {
        callbackLoopWaiting.wait_for(
          callbackLock,
          std::chrono::duration<double>(callbackPeriod),
          [&callbackLoopShouldStop](){return callbackLoopShouldStop.load();}
        );
        if(callbackLoopShouldStop.load())
          break;
        if(!searchBegun.load(std::memory_order_acquire))
          continue;
        callbackLock.unlock();
        callback(search);
        callbackLock.lock();
      }
      callbackLock.unlock();
    };

    std::thread callbackLoopThread;
    if(callbackPeriod >= 0) {
      callbackLoopThread = std::thread(callbackLoop);
    }

    search->runWholeSearch(*logger,shouldStopNow,searchBegun,pondering,tc,searchFactor);
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
    queuedOnMove(moveLoc,queuedSearchId);
    isRunning = false;
    isPondering = false;
    userWaitingForStop.notify_all();
  }
}
