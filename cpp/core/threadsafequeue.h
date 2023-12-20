/*
 * threadsafequeue.h
 * Author: davidwu
 */

#ifndef CORE_THREADSAFEQUEUE_H_
#define CORE_THREADSAFEQUEUE_H_

#include "../core/global.h"
#include "../core/multithread.h"

#include <queue>

template<typename T>
class ThreadSafeContainer
{
  size_t maxSize;
  bool closed;
  bool readOnly;
  std::mutex mutex;
  std::condition_variable notEmptyCondVar;
  std::condition_variable notFullCondVar;

  // Abstract methods to be implemented in derived classes
  virtual void pushUnsynchronized(T elt) = 0;
  virtual T popUnsynchronized() = 0;
  virtual void clearUnsynchronized() = 0;
  virtual size_t sizeUnsynchronized() = 0;
  virtual bool empty() = 0;

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

  // Thread-safe wrappers
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

  // Close the queue. Any elements still in the queue will be dropped and never read.
  // All blocked threads will unblock and any further reads or writes will not occur (with the respective functions returning false).
  inline void close() {
    std::lock_guard<std::mutex> lock(mutex);
    closed = true;
    clearUnsynchronized();
    notFullCondVar.notify_all();
    notEmptyCondVar.notify_all();
  }

  // Set the queue to be read only.
  // All blocked threads will unblock and any further writes will not occur (with the respective functions returning false).
  // All further reads will never block, but will continue to pop remaining elements in the queue until none are left.
  inline void setReadOnly() {
    std::lock_guard<std::mutex> lock(mutex);
    readOnly = true;
    notFullCondVar.notify_all();
    notEmptyCondVar.notify_all();
  }

  // Make the queue writable again.
  inline void unsetReadOnly() {
    std::lock_guard<std::mutex> lock(mutex);
    readOnly = false;
  }

  // Wait until the queue is not full or is closed or is readonly, and then push an element into the queue.
  // Returns true if the push was successful, false if the queue was closed or readonly.
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

  // Push an element without blocking, but can exceed maxSize of the queue.
  // Returns true if the push was successful, false if the queue was closed or readonly.
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

  // Attempt to pop an element into buf without blocking.
  // Returns true if successful, returns false if the queue is closed or empty.
  inline bool tryPop(T& buf)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if(closed)
      return false;
    size_t size = sizeUnsynchronized();
    if(size <= 0)
      return false;
    if(size == maxSize)
      notFullCondVar.notify_all();
    buf = popUnsynchronized();
    return true;
  }

  // Wait until the queue is not empty or is closed or is readonly, and then pop an element into buf.
  // Returns true if successful, returns false if the queue is closed, or empty and readonly.
  inline bool waitPop(T& buf)
  {
    std::unique_lock<std::mutex> lock(mutex);
    while(!closed && !readOnly && sizeUnsynchronized() <= 0)
      notEmptyCondVar.wait(lock);
    if(closed)
      return false;
    size_t size = sizeUnsynchronized();
    if(size <= 0)
      return false;
    if(size == maxSize)
      notFullCondVar.notify_all();
    buf = popUnsynchronized();
    return true;
  }

  // Wait until the queue is not empty or is closed or is readonly, and then pop and append up N elements to buf
  // or else as many as possible without further waiting.
  // Returns true if successful, returns false if no elements were popped (queue is closed, or empty and readonly).
  inline bool waitPopUpToN(std::vector<T>& buf, size_t n)
  {
    std::unique_lock<std::mutex> lock(mutex);
    while(!closed && !readOnly && sizeUnsynchronized() <= 0)
      notEmptyCondVar.wait(lock);
    if(closed)
      return false;
    size_t size = sizeUnsynchronized();
    if(size <= 0)
      return false;
    size_t numToPop = std::min(size,n);
    for(size_t i = 0; i<numToPop; i++)
      buf.push_back(popUnsynchronized());
    if(size >= maxSize && size < maxSize + n)
      notFullCondVar.notify_all();
    return true;
  }

};


template<typename T>
class ThreadSafeQueue final : public ThreadSafeContainer<T>
{
  size_t headIdx;
  std::vector<T> eltsDequeue;
  std::vector<T> eltsEnqueue;

 public:
  inline ThreadSafeQueue():
    ThreadSafeContainer<T>(), headIdx(0), eltsDequeue(), eltsEnqueue()
  {}
  inline ThreadSafeQueue(size_t maxSz):
    ThreadSafeContainer<T>(maxSz), headIdx(0), eltsDequeue(), eltsEnqueue()
  {}

  inline void reserve(size_t sz) {
    eltsDequeue.reserve(sz);
    eltsEnqueue.reserve(sz);
  }

  inline void pushUnsynchronized(T elt) override {
    eltsEnqueue.push_back(elt);
  }
  inline T popUnsynchronized() override {
    if(headIdx >= eltsDequeue.size()) {
      assert(eltsEnqueue.size() > 0);
      eltsDequeue.resize(0);
      eltsDequeue.swap(eltsEnqueue);
      headIdx = 0;
    }
    return eltsDequeue[headIdx++];
  }

  inline void clearUnsynchronized() override {
    eltsDequeue.clear();
    eltsEnqueue.clear();
  }

  inline size_t sizeUnsynchronized() override {
    assert(eltsDequeue.size() >= headIdx);
    return eltsDequeue.size() - headIdx + eltsEnqueue.size();
  }

  inline bool empty() override {
    return eltsEnqueue.empty() && eltsDequeue.size() <= headIdx;
  }

};

//Will return the elements with HIGHEST KT first, according to the comparison on KT.
template<typename KT,typename VT>
class ThreadSafePriorityQueue final : public ThreadSafeContainer<std::pair<KT,VT> >
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

  inline void pushUnsynchronized(T elt) override {
    queue.push(elt);
  }
  inline T popUnsynchronized() override {
    T item = queue.top();
    queue.pop();
    return item;
  }

  inline void clearUnsynchronized() override {
    while(!queue.empty())
      queue.pop();
  }

  inline size_t sizeUnsynchronized() override {
    return queue.size();
  }

  inline bool empty() override {
    return queue.empty();
  }
};



#endif  // CORE_THREADSAFEQUEUE_H_
