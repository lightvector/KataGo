/*
 * threadsafequeue.h
 * Author: davidwu
 */

#ifndef CORE_THREADSAFEQUEUE_H_
#define CORE_THREADSAFEQUEUE_H_

#include "../core/global.h"
#include "../core/multithread.h"

template<typename T>
class ThreadSafeQueue
{
  std::vector<T> elts;
  size_t headIdx;
  size_t maxSize;
  std::mutex mutex;
  std::condition_variable notEmptyCondVar;
  std::condition_variable notFullCondVar;

 public:
  inline ThreadSafeQueue()
    :elts(),headIdx(0),maxSize(0x7FFFFFFF),mutex(),notEmptyCondVar(),notFullCondVar()
  {}
  inline ThreadSafeQueue(size_t maxSz)
    :elts(),headIdx(0),maxSize(maxSz),mutex(),notEmptyCondVar(),notFullCondVar()
  {}
  inline ~ThreadSafeQueue()
  {}

  inline size_t size()
  {
    std::lock_guard<std::mutex> lock(mutex);
    return sizeUnsynchronized();
  }

  inline void waitPush(T elt)
  {
    std::unique_lock<std::mutex> lock(mutex);
    while(sizeUnsynchronized() >= maxSize)
      notFullCondVar.wait(lock);

    elts.push_back(elt);
    if(sizeUnsynchronized() == 1)
      notEmptyCondVar.notify_all();
  }

  //Will not block, but can exceed maxSize
  inline void forcePush(T elt)
  {
    std::unique_lock<std::mutex> lock(mutex);
    elts.push_back(elt);
    if(sizeUnsynchronized() == 1)
      notEmptyCondVar.notify_all();
  }

  
  inline bool tryPop(T& buf)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if(sizeUnsynchronized() <= 0)
      return false;
    if(sizeUnsynchronized() == maxSize)
      notFullCondVar.notify_all();
    buf = popUnsynchronized();
    return true;
  }

  inline T waitPop()
  {
    std::unique_lock<std::mutex> lock(mutex);
    while(sizeUnsynchronized() <= 0)
      notEmptyCondVar.wait(lock);
    if(sizeUnsynchronized() == maxSize)
      notFullCondVar.notify_all();
    return popUnsynchronized();
  }

 private:
  inline size_t sizeUnsynchronized()
  {
    assert(elts.size() >= headIdx);
    return elts.size() - headIdx;
  }

  inline T popUnsynchronized()
  {
    T elt = elts[headIdx];
    headIdx++;
    size_t eltsSize = elts.size();
    if(headIdx > eltsSize / 2)
    {
      assert(eltsSize >= headIdx);
      size_t len = eltsSize - headIdx;
      for(size_t i = 0; i<len; i++)
        elts[i] = elts[i+headIdx];
      elts.resize(len);
      headIdx = 0;
    }
    return elt;
  }

};

#endif  // CORE_THREADSAFEQUEUE_H_
