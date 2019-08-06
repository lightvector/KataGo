#ifndef SEARCH_MUTEXPOOL_H_
#define SEARCH_MUTEXPOOL_H_

#include "../core/global.h"
#include "../core/multithread.h"

class MutexPool {
  std::mutex* mutexes;
  uint32_t numMutexes;

 public:
  MutexPool(uint32_t n);
  ~MutexPool();

  uint32_t getNumMutexes() const;
  std::mutex& getMutex(uint32_t idx);
};

#endif  // SEARCH_MUTEXPOOL_H_
