/*
 * prioritymutex.h
 * Author: lightvector
 */

#ifndef CORE_PRIORITYMUTEX_H_
#define CORE_PRIORITYMUTEX_H_

#include "../core/global.h"
#include "../core/multithread.h"

class PriorityMutex {
  std::mutex mutex;
  std::condition_variable lowPriorityOkayToGo;
  std::atomic<int> numHighPriorityThreads;

 public:

  PriorityMutex(const PriorityMutex& other) = delete;
  PriorityMutex& operator=(const PriorityMutex& other) = delete;
  PriorityMutex(PriorityMutex&& other) = delete;
  PriorityMutex& operator=(PriorityMutex&& other) = delete;

  inline PriorityMutex()
    :mutex(),lowPriorityOkayToGo(),numHighPriorityThreads(0)
  {}
  inline ~PriorityMutex()
  {}

  inline void lockHighPriority() {
    numHighPriorityThreads.fetch_add(1);
    mutex.lock();
  }

  inline void unlockHighPriority() {
    int oldValue = numHighPriorityThreads.fetch_add(-1);
    int newValue = oldValue-1;
    assert(newValue >= 0);
    if(newValue <= 0)
      lowPriorityOkayToGo.notify_all();
    mutex.unlock();
  }

  inline void lockLowPriority() {
    std::unique_lock<std::mutex> lock(mutex);
    while(numHighPriorityThreads > 0)
      lowPriorityOkayToGo.wait(lock);
    lock.release(); //release without unlocking
  }

  inline void unlockLowPriority() {
    mutex.unlock();
  }
};

class PriorityLock {
  PriorityMutex* mutex;
  bool isHighPriority;
  bool isLocked;

 public:
  PriorityLock(const PriorityLock& other) = delete;
  PriorityLock& operator=(const PriorityLock& other) = delete;
  PriorityLock(PriorityLock&& other) = delete;
  PriorityLock& operator=(PriorityLock&& other) = delete;

  //Does NOT begin locked!
  inline PriorityLock(PriorityMutex& m)
    : mutex(&m), isHighPriority(false), isLocked(false)
  {}
  inline ~PriorityLock() {
    if(isLocked) {
      if(isHighPriority)
        mutex->unlockHighPriority();
      else
        mutex->unlockLowPriority();
    }
  }

  inline void lockHighPriority() {
    assert(!isLocked);
    isLocked = true;
    isHighPriority = true;
    mutex->lockHighPriority();
  }

  inline void lockLowPriority() {
    assert(!isLocked);
    isLocked = true;
    isHighPriority = false;
    mutex->lockHighPriority();
  }

  inline void unlock() {
    assert(isLocked);
    isLocked = false;
    if(isHighPriority)
      mutex->unlockHighPriority();
    else
      mutex->unlockLowPriority();
  }

};

#endif
