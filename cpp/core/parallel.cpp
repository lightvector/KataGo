#include "../core/parallel.h"

#include "../core/global.h"

void Parallel::iterRange(int numThreads, size_t size, const std::function<void(int,size_t)>& f) {
  std::atomic<size_t> counter(0);
  auto processLoop = [&](int threadIdx) {
    while(true) {
      size_t oldValue = counter.fetch_add(1);
      if(oldValue >= size)
        return;
      f(threadIdx,oldValue);
    }
  };

  // Start threads
  std::vector<std::thread> threads;
  for(int i = 0; i<numThreads; i++)
    threads.push_back(std::thread(processLoop,i));
  for(size_t i = 0; i<threads.size(); i++)
    threads[i].join();
}

void Parallel::iterRange(int numThreads, size_t size, Logger& logger, const std::function<void(int,size_t)>& f) {
  std::atomic<size_t> counter(0);
  auto processLoop = [&](int threadIdx) {
    while(true) {
      size_t oldValue = counter.fetch_add(1);
      if(oldValue >= size)
        return;
      f(threadIdx,oldValue);
    }
  };
  auto processLoopProtected = [&logger,&processLoop](int threadIdx) {
    Logger::logThreadUncaught("parallel iter range loop", &logger, [&](){ processLoop(threadIdx); });
  };

  // Start threads
  std::vector<std::thread> threads;
  for(int i = 0; i<numThreads; i++)
    threads.push_back(std::thread(processLoopProtected,i));
  for(size_t i = 0; i<threads.size(); i++)
    threads[i].join();
}
