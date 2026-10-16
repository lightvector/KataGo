#include "../search/mutexpool.h"

using namespace std;

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

std::mutex& MutexPool::getMutexWithModulo(uint32_t idx) {
  return mutexes[idx % numMutexes];
}
std::mutex& MutexPool::getMutexWithModulo(uint64_t idx) {
  return mutexes[idx % numMutexes];
}
std::mutex& MutexPool::getMutexWithModulo(Hash128 hash) {
  return mutexes[hash.hash0 % numMutexes];
}
