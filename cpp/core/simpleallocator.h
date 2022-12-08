#ifndef CORE_SIMPLEALLOCATOR_H
#define CORE_SIMPLEALLOCATOR_H

#include "../core/global.h"

template<typename T>
struct SizedBuf;

template<typename T>
class SimpleAllocator {
  std::function<T(size_t)> allocateFunc;
  std::function<void(T)> releaseFunc;

  std::map<size_t,std::vector<T>> buffers;

public:
  SimpleAllocator(std::function<T(size_t)> allocateFunc_, std::function<void(T)> releaseFunc_)
    :allocateFunc(allocateFunc_),releaseFunc(releaseFunc_),buffers()
  {
  }
  ~SimpleAllocator() {
    for(auto& iter: buffers) {
      for(T& buf: iter.second) {
        releaseFunc(buf);
      }
    }
  }

  SimpleAllocator() = delete;
  SimpleAllocator(const SimpleAllocator&) = delete;
  SimpleAllocator& operator=(const SimpleAllocator&) = delete;

  friend class SizedBuf<T>;
};


template<typename T>
struct SizedBuf {
  size_t size;
  T buf;
  SimpleAllocator<T>* allocator;

  SizedBuf(SimpleAllocator<T>* alloc, size_t s)
    : size(s),buf(),allocator(alloc)
  {
    const std::vector<T> vec = allocator->buffers[size];
    if(vec.size() <= 0)
      buf = allocator->allocateFunc(size);
    else {
      std::vector<T>& buffers = allocator->buffers[size];
      buf = buffers.back();
      buffers.pop_back();
    }
  }
  ~SizedBuf() {
    allocator->buffers[size].push_back(buf);
  }

  SizedBuf() = delete;
  SizedBuf(const SizedBuf&) = delete;
  SizedBuf& operator=(const SizedBuf&) = delete;

  friend class SimpleAllocator<T>;
};


#endif // CORE_SIMPLEALLOCATOR_H
