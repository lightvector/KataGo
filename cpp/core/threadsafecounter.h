/*
 * threadsafecounter.h
 * Author: davidwu
 */

#ifndef CORE_THREADSAFECOUNTER_H_
#define CORE_THREADSAFECOUNTER_H_

#include "../core/global.h"
#include "../core/multithread.h"

class ThreadSafeCounter
{
  int64_t value;
  std::mutex mutex;
  std::condition_variable zeroCondVar;

 public:
  inline ThreadSafeCounter()
    :value(0),mutex(),zeroCondVar()
  {}

  ThreadSafeCounter(const ThreadSafeCounter&) = delete;
  ThreadSafeCounter& operator=(const ThreadSafeCounter&) = delete;
  ThreadSafeCounter(ThreadSafeCounter&&) = delete;
  ThreadSafeCounter& operator=(ThreadSafeCounter&&) = delete;

  inline void add(int64_t x)
  {
    std::unique_lock<std::mutex> lock(mutex);
    value += x;
    if(value == 0)
      zeroCondVar.notify_all();
  }

  inline void setZero()
  {
    std::unique_lock<std::mutex> lock(mutex);
    value = 0;
    zeroCondVar.notify_all();
  }

  inline void waitUntilZero()
  {
    std::unique_lock<std::mutex> lock(mutex);
    while(value != 0)
      zeroCondVar.wait(lock);
  }
};

#endif  // CORE_THREADSAFECOUNTER_H_

