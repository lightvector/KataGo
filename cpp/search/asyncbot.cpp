
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
   isRunning(false),isKilled(false),shouldStopNow(false),
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
  stopAndWait();
  if(movePla != search->rootPla)
    search->setPlayerAndClearHistory(movePla);

  unique_lock<std::mutex> lock(controlMutex);
  assert(!isRunning);
  queuedSearchId = searchId;
  queuedOnMove = onMove;
  isRunning = true;
  shouldStopNow = false;
  lock.unlock();
  threadWaitingToSearch.notify_all();
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
  shouldStopNow = false;
  lock.unlock();
  threadWaitingToSearch.notify_all();
}

void AsyncBot::stopAndWait() {
  shouldStopNow.store(true);
  waitForSearchToEnd();
}

void AsyncBot::waitForSearchToEnd() {
  unique_lock<std::mutex> lock(controlMutex);
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
    atomic<uint64_t> numPlayouts(0);

    if(!std::atomic_is_lock_free(&numPlayouts))
      logger->write("Warning: uint64_t atomic numPlayouts is not lock free");
    if(!std::atomic_is_lock_free(&shouldStopNow))
      logger->write("Warning: bool atomic shouldStopNow is not lock free");

    search->beginSearch();

    auto searchLoop = [this,&timer,&numPlayouts](int threadIdx) {
      SearchThread* stbuf = new SearchThread(threadIdx,*search,logger);
      try {
        while(!shouldStopNow.load(std::memory_order_relaxed)) {
          search->runSinglePlayout(*stbuf);

          uint64_t oldNumPlayouts = numPlayouts.fetch_add((uint64_t)1, std::memory_order_relaxed);
          uint64_t newNumPlayouts = oldNumPlayouts + 1;

          bool shouldStop =
            (searchParams.maxTime < 1.0e12 && timer.getSeconds() >= searchParams.maxTime) ||
            (newNumPlayouts >= searchParams.maxPlayouts);

          if(shouldStop) {
            shouldStopNow.store(true,std::memory_order_relaxed);
            break;
          }
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
    queuedOnMove(moveLoc,queuedSearchId);

    lock.lock();
    isRunning = false;
    userWaitingForStop.notify_all();
  }
}


