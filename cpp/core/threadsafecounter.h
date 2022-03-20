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

class WaitableFlag
{
  std::atomic<bool> value;
  bool finished;
  mutable std::mutex mutex;
  mutable std::condition_variable falseCondVar;
  mutable std::condition_variable trueCondVar;
public:
  inline WaitableFlag()
    :value(false),finished(false),mutex(),falseCondVar(),trueCondVar()
  {}
  WaitableFlag(const WaitableFlag&) = delete;
  WaitableFlag& operator=(const WaitableFlag&) = delete;
  WaitableFlag(WaitableFlag&&) = delete;
  WaitableFlag& operator=(WaitableFlag&&) = delete;

  inline void set(bool b) {
    std::lock_guard<std::mutex> lock(mutex);
    if(finished)
      return;
    value.store(b,std::memory_order_release);
    if(b)
      trueCondVar.notify_all();
    else
      falseCondVar.notify_all();
  }

  inline void setPermanently(bool b) {
    std::lock_guard<std::mutex> lock(mutex);
    if(finished)
      return;
    finished = true;
    value.store(b,std::memory_order_release);
    if(b)
      trueCondVar.notify_all();
    else
      falseCondVar.notify_all();
  }

  inline bool get() const {
    return value.load(std::memory_order_acquire);
  }

  inline void waitUntilFalse() const {
    bool b = get();
    if(!b)
      return;
    std::unique_lock<std::mutex> lock(mutex);
    while(b) {
      falseCondVar.wait(lock);
      b = get();
    }
    return;
  }

  inline void waitUntilTrue() const {
    bool b = get();
    if(b)
      return;
    std::unique_lock<std::mutex> lock(mutex);
    while(!b) {
      trueCondVar.wait(lock);
      b = get();
    }
    return;
  }

};



#endif  // CORE_THREADSAFECOUNTER_H_

