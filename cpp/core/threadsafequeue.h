/*
 * threadsafequeue.h
 * Author: davidwu
 */

#ifndef CORE_THREADSAFEQUEUE_H_
#define CORE_THREADSAFEQUEUE_H_

#include "../core/global.h"
#include "../core/multithread.h"


template<typename T>
class ThreadSafeContainer
{
  size_t maxSize;
  bool closed;
  bool readOnly;
  std::mutex mutex;
  std::condition_variable notEmptyCondVar;
  std::condition_variable notFullCondVar;

  // abstract methods to be implemented in derived clases
  virtual void pushUnsynchronized(T elt) = 0;
  virtual T popUnsynchronized() = 0;
  virtual void clearUnsynchronized() = 0;
  virtual size_t sizeUnsynchronized() = 0;

 public:
  inline ThreadSafeContainer()
    :maxSize(0x7FFFFFFF),closed(false),readOnly(false),mutex(),notEmptyCondVar(),notFullCondVar()
  {}
  inline ThreadSafeContainer(size_t maxSz)
    :maxSize(maxSz),closed(false),readOnly(false),mutex(),notEmptyCondVar(),notFullCondVar()
  {}
  inline ~ThreadSafeContainer()
  {}

  ThreadSafeContainer(const ThreadSafeContainer&) = delete;
  ThreadSafeContainer& operator=(const ThreadSafeContainer&) = delete;
  ThreadSafeContainer(ThreadSafeContainer&&) = delete;
  ThreadSafeContainer& operator=(ThreadSafeContainer&&) = delete;

  // thread-safe wrappers
  inline size_t size()
  {
    std::lock_guard<std::mutex> lock(mutex);
    return sizeUnsynchronized();
  }

  inline bool isClosed()
  {
    std::lock_guard<std::mutex> lock(mutex);
    return closed;
  }

  inline bool isReadOnly()
  {
    std::lock_guard<std::mutex> lock(mutex);
    return readOnly;
  }

  //Close the queue. Any elements still in the queue will be dropped and never read.
  //All blocked threads will unblock and any further reads or writes will not occur (with the respective functions returning false).
  inline void close() {
    std::lock_guard<std::mutex> lock(mutex);
    closed = true;
    clearUnsynchronized();
    notFullCondVar.notify_all();
    notEmptyCondVar.notify_all();
  }

  //Set the queue to be read only.
  //All blocked threads will unblock and any further writes will not occur (with the respective functions returning false).
  //All further reads will never block, but will continue to pop remaining elements in the queue until none are left.
  inline void setReadOnly() {
    std::lock_guard<std::mutex> lock(mutex);
    readOnly = true;
    notFullCondVar.notify_all();
    notEmptyCondVar.notify_all();
  }

  //Wait until the queue is not full or is closed or is readonly, and then push an element into the queue.
  //Returns true if the push was successful, false if the queue was closed or readonly.
  inline bool waitPush(T elt)
  {
    std::unique_lock<std::mutex> lock(mutex);
    while(!closed && !readOnly && sizeUnsynchronized() >= maxSize)
      notFullCondVar.wait(lock);
    if(closed || readOnly)
      return false;
    pushUnsynchronized(elt);
    if(sizeUnsynchronized() == 1)
      notEmptyCondVar.notify_all();
    return true;
  }

  //Push an element without blocking, but cavn exceed maxSize of the queue.
  //Returns true if the push was successful, false if the queue was closed or readonly.
  inline bool forcePush(T elt)
  {
    std::unique_lock<std::mutex> lock(mutex);
    if(closed || readOnly)
      return false;
    pushUnsynchronized(elt);
    if(sizeUnsynchronized() == 1)
      notEmptyCondVar.notify_all();
    return true;
  }

  //Attempt to pop an element into buf without blocking.
  //Returns true if successful, returns false if the queue is closed or empty.
  inline bool tryPop(T& buf)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if(closed)
      return false;
    if(sizeUnsynchronized() <= 0)
      return false;
    if(sizeUnsynchronized() == maxSize)
      notFullCondVar.notify_all();
    buf = popUnsynchronized();
    return true;
  }

  //Wait until the queue is not empty or is closed or is readonly, and then pop an element into buf.
  //Returns true if successful, returns false if the queue is closed, or empty and readonly.
  inline bool waitPop(T& buf)
  {
    std::unique_lock<std::mutex> lock(mutex);
    while(!closed && !readOnly && sizeUnsynchronized() <= 0)
      notEmptyCondVar.wait(lock);
    if(closed)
      return false;
    if(sizeUnsynchronized() <= 0)
      return false;
    if(sizeUnsynchronized() == maxSize)
      notFullCondVar.notify_all();
    buf = popUnsynchronized();
    return true;
  }
};


template<typename T>
class ThreadSafeQueue: public ThreadSafeContainer<T>
{
  size_t headIdx;
  std::vector<T> elts;

 public:
  inline ThreadSafeQueue():
    ThreadSafeContainer<T>(), headIdx(0), elts()
  {}
  inline ThreadSafeQueue(size_t maxSz):
    ThreadSafeContainer<T>(maxSz), headIdx(0), elts()
  {}

  inline void pushUnsynchronized(T elt) {
    elts.push_back(elt);
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

  inline void clearUnsynchronized() {
     elts.clear();
   }

  inline size_t sizeUnsynchronized()
  {
    assert(elts.size() >= headIdx);
    return elts.size() - headIdx;
  }

};


template<typename KT,typename VT>
class ThreadSafePriorityQueue: public ThreadSafeContainer<std::pair<KT,VT> >
{
  size_t headIdx;
  typedef std::pair<KT,VT> T;
  std::priority_queue<T> queue;

 public:
  inline ThreadSafePriorityQueue():
    ThreadSafeContainer<T>(), headIdx(0), queue()
  {}
  inline ThreadSafePriorityQueue(size_t maxSz):
    ThreadSafeContainer<T>(maxSz), headIdx(0), queue()
  {}

  inline void pushUnsynchronized(T elt) {
    queue.push(elt);
  }
  inline T popUnsynchronized()
  {
    T item = queue.top();
    queue.pop();
    return item;
  }

  inline void clearUnsynchronized() {
     while(!queue.empty())
       queue.pop();
   }

  inline size_t sizeUnsynchronized()
  {
    return queue.size();
  }

};





#endif  // CORE_THREADSAFEQUEUE_H_
