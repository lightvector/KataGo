
#include "../core/timer.h"
#include "../search/asyncbot.h"

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
   queuedSearchId(0),queuedOnMove(),timeControls()
{
  search = new Search(params,nnEval,randSeed);
  searchThread = std::thread(searchThreadLoop,this,l);
}

AsyncBot::~AsyncBot() {
  stopAndWait();
  assert(!isRunning);
  assert(!isKilled);
  isKilled = true;
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

Search* AsyncBot::getSearch() {
  return search;
}
SearchParams AsyncBot::getParams() const {
  return search->searchParams;
}

void AsyncBot::setPosition(Player pla, const Board& board, const BoardHistory& history) {
  stopAndWait();
  search->setPosition(pla,board,history);
}
void AsyncBot::setRulesAndClearHistory(Rules rules, int encorePhase) {
  stopAndWait();
  search->setRulesAndClearHistory(rules,encorePhase);
}
void AsyncBot::setKomiIfNew(float newKomi) {
  stopAndWait();
  search->setKomiIfNew(newKomi);
}
void AsyncBot::setRootPassLegal(bool b) {
  stopAndWait();
  search->setRootPassLegal(b);
}
void AsyncBot::setParams(SearchParams params) {
  stopAndWait();
  search->setParams(params);
}
void AsyncBot::clearSearch() {
  stopAndWait();
  search->clearSearch();
}

bool AsyncBot::makeMove(Loc moveLoc, Player movePla) {
  stopAndWait();
  return search->makeMove(moveLoc,movePla);
}

bool AsyncBot::isLegal(Loc moveLoc, Player movePla) const {
  return search->isLegal(moveLoc,movePla);
}

void AsyncBot::genMove(Player movePla, int searchId, const TimeControls& tc, std::function<void(Loc,int)> onMove) {
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
  lock.unlock();
  threadWaitingToSearch.notify_all();
}

Loc AsyncBot::genMoveSynchronous(Player movePla, const TimeControls& tc) {
  Loc moveLoc = Board::NULL_LOC;
  std::function<void(Loc,int)> onMove = [&moveLoc](Loc loc, int searchId) {
    assert(searchId == 0);
    moveLoc = loc;
  };
  genMove(movePla,0,tc,onMove);
  waitForSearchToEnd();
  return moveLoc;
}

static void ignoreMove(Loc loc, int searchId) {
  (void)loc;
  (void)searchId;
}

void AsyncBot::ponder() {
  unique_lock<std::mutex> lock(controlMutex);
  if(isRunning)
    return;

  queuedSearchId = 0;
  queuedOnMove = std::function<void(Loc,int)>(ignoreMove);
  isRunning = true;
  isPondering = true;
  shouldStopNow = false;
  timeControls = TimeControls();
  lock.unlock();
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
    if(isKilled)
      break;

    bool pondering = isPondering;
    TimeControls tc = timeControls;
    lock.unlock();

    search->runWholeSearch(*logger,shouldStopNow,NULL,pondering,tc);
    Loc moveLoc = search->getChosenMoveLoc();

    lock.lock();
    //Must call queuedOnMove under the lock since during the pondering->normal search
    //transition it might change out from underneath us
    queuedOnMove(moveLoc,queuedSearchId);
    isRunning = false;
    isPondering = false;
    userWaitingForStop.notify_all();
  }
}


