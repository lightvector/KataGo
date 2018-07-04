#include "../search/mutexpool.h"

MutexPool::MutexPool(uint32_t n) {
  numMutexes = n;
  mutexes = new mutex[n];
}

MutexPool::~MutexPool() {
  delete[] mutexes;
}

uint32_t MutexPool::getNumMutexes() const {
  return numMutexes;
}

mutex& MutexPool::getMutex(uint32_t idx) {
  return mutexes[idx];
}
