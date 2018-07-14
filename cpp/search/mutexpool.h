#ifndef LOCKPOOL_H
#define LOCKPOOL_H

#include "../core/global.h"
#include "../core/multithread.h"

class MutexPool {
  mutex* mutexes;
  uint32_t numMutexes;

 public:
  MutexPool(uint32_t n);
  ~MutexPool();

  uint32_t getNumMutexes() const;
  mutex& getMutex(uint32_t idx);
};

#endif
