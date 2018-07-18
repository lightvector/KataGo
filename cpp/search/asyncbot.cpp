
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

AsyncBot::AsyncBot(SearchParams params, NNEvaluator* nnEval, Logger* l)
  :search(NULL),logger(l),searchParams(params),
   controlMutex(),threadWaitingToSearch(),userWaitingForStop(),searchThread(),
   isRunning(false),isPondering(false),isKilled(false),shouldStopNow(false),
   queuedSearchId(0),queuedOnMove()
{
  search = new Search(params,nnEval);
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

void AsyncBot::setPosition(Player pla, const Board& board, const BoardHistory& history) {
  stopAndWait();
  search->setPosition(pla,board,history);
}
void AsyncBot::setRulesAndClearHistory(Rules rules) {
  stopAndWait();
  search->setRulesAndClearHistory(rules);
}
void AsyncBot::setKomi(float newKomi) {
  stopAndWait();
  search->setKomi(newKomi);
}
void AsyncBot::setParams(SearchParams params) {
  stopAndWait();
  searchParams = params;
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

void AsyncBot::genMove(Player movePla, int searchId, std::function<void(Loc,int)> onMove) {
  unique_lock<std::mutex> lock(controlMutex);
  if(isRunning && isPondering && movePla == search->rootPla) {
    queuedSearchId = searchId;
    queuedOnMove = onMove;
  }
  else {
    stopAndWaitAlreadyLocked(lock);
    assert(!isRunning);
    if(movePla != search->rootPla)
      search->setPlayerAndClearHistory(movePla);

    queuedSearchId = searchId;
    queuedOnMove = onMove;
    isRunning = true;
    isPondering = false;
    shouldStopNow = false;
    lock.unlock();
    threadWaitingToSearch.notify_all();
  }
}

Loc AsyncBot::genMoveSynchronous(Player movePla) {
  Loc moveLoc = Board::NULL_LOC;
  std::function<void(Loc,int)> onMove = [&moveLoc](Loc loc, int searchId) {
    assert(searchId == 0);
    moveLoc = loc;
  };
  genMove(movePla,0,onMove);
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

    lock.unlock();


    ClockTimer timer;
    atomic<uint64_t> numPlayoutsShared(0);

    if(!std::atomic_is_lock_free(&numPlayoutsShared))
      logger->write("Warning: uint64_t atomic numPlayoutsShared is not lock free");
    if(!std::atomic_is_lock_free(&shouldStopNow))
      logger->write("Warning: bool atomic shouldStopNow is not lock free");

    search->beginSearch();
    uint64_t numNonPlayoutVisits = search->numRootVisits();

    auto searchLoop = [this,&timer,&numPlayoutsShared,numNonPlayoutVisits](int threadIdx) {
      SearchThread* stbuf = new SearchThread(threadIdx,*search,logger);
      uint64_t numPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
      try {
        while(true) {
          bool shouldStop =
            (numPlayouts >= 1 && searchParams.maxTime < 1.0e12 && timer.getSeconds() >= searchParams.maxTime) ||
            (numPlayouts >= searchParams.maxPlayouts) ||
            (numPlayouts + numNonPlayoutVisits >= searchParams.maxVisits);

          if(shouldStop || shouldStopNow.load(std::memory_order_relaxed)) {
            shouldStopNow.store(true,std::memory_order_relaxed);
            break;
          }

          search->runSinglePlayout(*stbuf);

          numPlayouts = numPlayoutsShared.fetch_add((uint64_t)1, std::memory_order_relaxed);
          numPlayouts += 1;
        }
      }
      catch(const exception& e) {
        logger->write(string("ERROR: Search thread failed: ") + e.what());
      }
      catch(const string& e) {
        logger->write("ERROR: Search thread failed: " + e);
      }
      catch(...) {
        logger->write("ERROR: Search thread failed with unexpected throw");
      }
      delete stbuf;
    };

    if(searchParams.numThreads <= 1)
      searchLoop(0);
    else {
      std::thread* threads = new std::thread[searchParams.numThreads-1];
      for(int i = 0; i<searchParams.numThreads-1; i++)
        threads[i] = std::thread(searchLoop,i+1);
      searchLoop(0);
      for(int i = 0; i<searchParams.numThreads-1; i++)
        threads[i].join();
      delete[] threads;
    }

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


