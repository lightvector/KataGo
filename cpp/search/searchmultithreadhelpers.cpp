#include "../search/search.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------

static void threadTaskLoop(Search* search, int threadIdx) {
  while(true) {
    std::function<void(int)>* task;
    bool suc = search->threadTasks[threadIdx-1].waitPop(task);
    if(!suc)
      return;

    try {
      (*task)(threadIdx);
      //Don't delete task, the convention is tasks are owned by the joining thread
    }
    catch(const std::exception& e) {
      search->logger->write(string("ERROR: Search thread failed: ") + e.what());
      search->threadTasksRemaining->add(-1);
      throw;
    }
    catch(const string& e) {
      search->logger->write("ERROR: Search thread failed: " + e);
      search->threadTasksRemaining->add(-1);
      throw;
    }
    catch(...) {
      search->logger->write("ERROR: Search thread failed with unexpected throw");
      search->threadTasksRemaining->add(-1);
      throw;
    }
    search->threadTasksRemaining->add(-1);
  }
}

int Search::numAdditionalThreadsToUseForTasks() const {
  return searchParams.numThreads-1;
}

void Search::spawnThreadsIfNeeded() {
  int desiredNumAdditionalThreads = numAdditionalThreadsToUseForTasks();
  if(numThreadsSpawned >= desiredNumAdditionalThreads)
    return;
  killThreads();
  threadTasks = new ThreadSafeQueue<std::function<void(int)>*>[desiredNumAdditionalThreads];
  threadTasksRemaining = new ThreadSafeCounter();
  threads = new std::thread[desiredNumAdditionalThreads];
  for(int i = 0; i<desiredNumAdditionalThreads; i++)
    threads[i] = std::thread(threadTaskLoop,this,i+1);
  numThreadsSpawned = desiredNumAdditionalThreads;
}

void Search::killThreads() {
  if(numThreadsSpawned <= 0)
    return;
  for(int i = 0; i<numThreadsSpawned; i++)
    threadTasks[i].close();
  for(int i = 0; i<numThreadsSpawned; i++)
    threads[i].join();
  delete[] threadTasks;
  delete threadTasksRemaining;
  delete[] threads;
  threadTasks = NULL;
  threadTasksRemaining = NULL;
  threads = NULL;
  numThreadsSpawned = 0;
}

void Search::respawnThreads() {
  killThreads();
  spawnThreadsIfNeeded();
}

void Search::performTaskWithThreads(std::function<void(int)>* task, int capThreads) {
  spawnThreadsIfNeeded();
  int numAdditionalThreadsToUse = std::min(capThreads-1, numAdditionalThreadsToUseForTasks());
  if(numAdditionalThreadsToUse <= 0) {
    (*task)(0);
  }
  else {
    assert(numAdditionalThreadsToUse <= numThreadsSpawned);
    threadTasksRemaining->add(numAdditionalThreadsToUse);
    for(int i = 0; i<numAdditionalThreadsToUse; i++)
      threadTasks[i].forcePush(task);
    (*task)(0);
    threadTasksRemaining->waitUntilZero();
  }
}


static void maybeAppendShuffledIntRange(int cap, PCG32* rand, std::vector<int>& randBuf) {
  if(rand != NULL) {
    size_t randBufStart = randBuf.size();
    for(int i = 0; i<cap; i++)
      randBuf.push_back(i);
    for(int i = 1; i<cap; i++) {
      int r = (int)(rand->nextUInt() % (uint32_t)(i+1));
      int tmp = randBuf[randBufStart+i];
      randBuf[randBufStart+i] = randBuf[randBufStart+r];
      randBuf[randBufStart+r] = tmp;
    }
  }
}

//Walk over all nodes and their children recursively and call f, children first.
//Assumes that only other instances of this function are running - in particular, the tree
//is not being mutated by something else. It's okay if f mutates nodes, so long as it only mutates
//nodes that will no longer be iterated over (namely, only stuff at the node or within its subtree).
//As a side effect, nodeAge == searchNodeAge will be true only for the nodes walked over.
void Search::applyRecursivelyPostOrderMulithreaded(const vector<SearchNode*>& nodes, std::function<void(SearchNode*,int)>* f) {
  //We invalidate all node ages so we can use them as a marker for what's done.
  searchNodeAge += 1;

  //Simple cheap RNGs so that we can get the different threads into different parts of the tree and not clash.
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  std::vector<PCG32*> rands(numAdditionalThreads+1, NULL);
  for(int threadIdx = 1; threadIdx<numAdditionalThreads+1; threadIdx++)
    rands[threadIdx] = new PCG32(nonSearchRand.nextUInt64());

  int numChildren = (int)nodes.size();
  std::function<void(int)> g = [&](int threadIdx) {
    assert(threadIdx >= 0 && threadIdx < rands.size());
    PCG32* rand = rands[threadIdx];
    std::unordered_set<SearchNode*> nodeBuf;
    std::vector<int> randBuf;

    size_t randBufStart = randBuf.size();
    maybeAppendShuffledIntRange(numChildren, rand, randBuf);
    for(int i = 0; i<numChildren; i++) {
      int childIdx = rand != NULL ? randBuf[randBufStart+i] : i;
      applyRecursivelyPostOrderMulithreadedHelper(nodes[childIdx],threadIdx,rand,nodeBuf,randBuf,f);
    }
    randBuf.resize(randBufStart);
  };
  performTaskWithThreads(&g, 0x3fffFFFF);
  for(int threadIdx = 1; threadIdx<numAdditionalThreads+1; threadIdx++)
    delete rands[threadIdx];
}

void Search::applyRecursivelyPostOrderMulithreadedHelper(
  SearchNode* node, int threadIdx, PCG32* rand, std::unordered_set<SearchNode*>& nodeBuf, std::vector<int>& randBuf, std::function<void(SearchNode*,int)>* f
) {
  //nodeAge == searchNodeAge means that the node is done.
  if(node->nodeAge.load(std::memory_order_acquire) == searchNodeAge)
    return;
  //Cycle! Just consider this node "done" and return.
  if(nodeBuf.find(node) != nodeBuf.end())
    return;

  //Recurse on all children
  SearchNodeChildrenReference children = node->getChildren();
  int numChildren = children.iterateAndCountChildren();

  if(numChildren > 0) {
    size_t randBufStart = randBuf.size();
    maybeAppendShuffledIntRange(numChildren, rand, randBuf);

    nodeBuf.insert(node);
    for(int i = 0; i<numChildren; i++) {
      int childIdx = rand != NULL ? randBuf[randBufStart+i] : i;
      applyRecursivelyPostOrderMulithreadedHelper(children[childIdx].getIfAllocated(), threadIdx, rand, nodeBuf, randBuf, f);
    }
    randBuf.resize(randBufStart);
    nodeBuf.erase(node);
  }

  //Now call postorder function, protected by lock
  std::lock_guard<std::mutex> lock(mutexPool->getMutex(node->mutexIdx));
  //Make sure another node didn't get there first.
  if(node->nodeAge.load(std::memory_order_acquire) == searchNodeAge)
    return;
  if(f != NULL)
    (*f)(node,threadIdx);
  node->nodeAge.store(searchNodeAge,std::memory_order_release);
}


//Walk over all nodes and their children recursively and call f. No order guarantee, but does guarantee that f is called only once per node.
//Assumes that only other instances of this function are running - in particular, the tree
//is not being mutated by something else. It's okay if f mutates nodes.
//As a side effect, nodeAge == searchNodeAge will be true only for the nodes walked over.
void Search::applyRecursivelyAnyOrderMulithreaded(const vector<SearchNode*>& nodes, std::function<void(SearchNode*,int)>* f) {
  //We invalidate all node ages so we can use them as a marker for what's done.
  searchNodeAge += 1;

  //Simple cheap RNGs so that we can get the different threads into different parts of the tree and not clash.
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  std::vector<PCG32*> rands(numAdditionalThreads+1, NULL);
  for(int threadIdx = 1; threadIdx<numAdditionalThreads+1; threadIdx++)
    rands[threadIdx] = new PCG32(nonSearchRand.nextUInt64());

  int numChildren = (int)nodes.size();
  std::function<void(int)> g = [&](int threadIdx) {
    assert(threadIdx >= 0 && threadIdx < rands.size());
    PCG32* rand = rands[threadIdx];
    std::unordered_set<SearchNode*> nodeBuf;
    std::vector<int> randBuf;

    size_t randBufStart = randBuf.size();
    maybeAppendShuffledIntRange(numChildren, rand, randBuf);
    for(int i = 0; i<numChildren; i++) {
      int childIdx = rand != NULL ? randBuf[randBufStart+i] : i;
      applyRecursivelyAnyOrderMulithreadedHelper(nodes[childIdx],threadIdx,rand,nodeBuf,randBuf,f);
    }
    randBuf.resize(randBufStart);
  };
  performTaskWithThreads(&g, 0x3fffFFFF);
  for(int threadIdx = 1; threadIdx<numAdditionalThreads+1; threadIdx++)
    delete rands[threadIdx];
}

void Search::applyRecursivelyAnyOrderMulithreadedHelper(
  SearchNode* node, int threadIdx, PCG32* rand, std::unordered_set<SearchNode*>& nodeBuf, std::vector<int>& randBuf, std::function<void(SearchNode*,int)>* f
) {
  //nodeAge == searchNodeAge means that the node is done.
  if(node->nodeAge.load(std::memory_order_acquire) == searchNodeAge)
    return;
  //Cycle! Just consider this node "done" and return.
  if(nodeBuf.find(node) != nodeBuf.end())
    return;

  //Recurse on all children
  SearchNodeChildrenReference children = node->getChildren();
  int numChildren = children.iterateAndCountChildren();

  if(numChildren > 0) {
    size_t randBufStart = randBuf.size();
    maybeAppendShuffledIntRange(numChildren, rand, randBuf);

    nodeBuf.insert(node);
    for(int i = 0; i<numChildren; i++) {
      int childIdx = rand != NULL ? randBuf[randBufStart+i] : i;
      applyRecursivelyAnyOrderMulithreadedHelper(children[childIdx].getIfAllocated(), threadIdx, rand, nodeBuf, randBuf, f);
    }
    randBuf.resize(randBufStart);
    nodeBuf.erase(node);
  }

  //The thread that is first to update it wins and does the action.
  uint32_t oldAge = node->nodeAge.exchange(searchNodeAge,std::memory_order_acq_rel);
  if(oldAge == searchNodeAge)
    return;
  if(f != NULL)
    (*f)(node,threadIdx);
}

//Mainly for testing
std::vector<SearchNode*> Search::enumerateTreePostOrder() {
  std::atomic<int64_t> sizeCounter(0);
  std::function<void(SearchNode*,int)> f = [&](SearchNode* node, int threadIdx) {
    (void)node;
    (void)threadIdx;
    sizeCounter.fetch_add(1,std::memory_order_relaxed);
  };
  applyRecursivelyPostOrderMulithreaded({rootNode},&f);

  int64_t size = sizeCounter.load(std::memory_order_relaxed);
  std::vector<SearchNode*> nodes(size,NULL);
  std::atomic<int64_t> indexCounter(0);
  std::function<void(SearchNode*,int)> g = [&](SearchNode* node, int threadIdx) {
    (void)threadIdx;
    int64_t index = indexCounter.fetch_add(1,std::memory_order_relaxed);
    assert(index >= 0 && index < size);
    nodes[index] = node;
  };
  applyRecursivelyPostOrderMulithreaded({rootNode},&g);
  assert(indexCounter.load(std::memory_order_relaxed) == size);
  return nodes;
}
