#ifndef SEARCH_MUTEXPOOL_H_
#define SEARCH_MUTEXPOOL_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/multithread.h"

class MutexPool {
  std::mutex* mutexes;
  uint32_t numMutexes;

 public:
  MutexPool(uint32_t n);
  ~MutexPool();

  uint32_t getNumMutexes() const;
  std::mutex& getMutex(uint32_t idx);

  // Convenience methods that will automatically mod by numMutexes.
  std::mutex& getMutexWithModulo(uint32_t idx);
  std::mutex& getMutexWithModulo(uint64_t idx);
  std::mutex& getMutexWithModulo(Hash128 hash);
};

#endif  // SEARCH_MUTEXPOOL_H_
