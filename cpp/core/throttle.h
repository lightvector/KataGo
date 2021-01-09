/*
 * throttle.h
 * Author: lightvector
 */

#ifndef CORE_THROTTLE_H_
#define CORE_THROTTLE_H_

#include "../core/global.h"
#include "../core/multithread.h"

class Throttle {
  std::mutex mutex;
  std::condition_variable okayForMore;
  int numThreadsActive;
  const int maxThreadsActive;

 public:

  Throttle(const Throttle& other) = delete;
  Throttle& operator=(const Throttle& other) = delete;
  Throttle(Throttle&& other) = delete;
  Throttle& operator=(Throttle&& other) = delete;

  inline Throttle(int maxThreadsAtATime)
    :mutex(),okayForMore(),numThreadsActive(0),maxThreadsActive(maxThreadsAtATime)
  {
    assert(maxThreadsActive > 0);
  }
  inline ~Throttle()
  {}

  inline void lock() {
    std::unique_lock<std::mutex> lock(mutex);
    assert(numThreadsActive >= 0 && numThreadsActive <= maxThreadsActive);
    while(numThreadsActive >= maxThreadsActive)
      okayForMore.wait(lock);
    numThreadsActive++;
    assert(numThreadsActive >= 0 && numThreadsActive <= maxThreadsActive);
  }

  inline void unlock() {
    std::lock_guard<std::mutex> lock(mutex);
    assert(numThreadsActive >= 0 && numThreadsActive <= maxThreadsActive);
    numThreadsActive--;
    assert(numThreadsActive >= 0 && numThreadsActive <= maxThreadsActive);
    okayForMore.notify_one();
  }
};

class ThrottleLockGuard {
  Throttle* throttle;

 public:
  ThrottleLockGuard(const ThrottleLockGuard& other) = delete;
  ThrottleLockGuard& operator=(const ThrottleLockGuard& other) = delete;
  ThrottleLockGuard(ThrottleLockGuard&& other) = delete;
  ThrottleLockGuard& operator=(ThrottleLockGuard&& other) = delete;

  inline ThrottleLockGuard(Throttle& t)
    : throttle(&t)
  {
    throttle->lock();
  }
  inline ~ThrottleLockGuard() {
    throttle->unlock();
  }
};

#endif
