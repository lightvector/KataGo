#ifndef CORE_MULTITHREAD_H_
#define CORE_MULTITHREAD_H_

#include "../core/global.h"

//Enable multithreading
#define MULTITHREADING

#ifdef MULTITHREADING
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#define IS_MULTITHREADING_ENABLED true
#else
#define IS_MULTITHREADING_ENABLED false

namespace std {

class mutex
{ public:
  inline void lock() {};
  inline void unlock() {};
};
class thread
{ public:
  inline thread() {};
  inline void join() {};
};
template <class T>
class unique_lock
{ public:
  unique_lock(T t) {(void)t;};
  inline void lock() {};
  inline void unlock() {};
};
template <class T>
class lock_guard
{ public:
  lock_guard(T t) {(void)t;};
};
class condition_variable
{ public:
  inline void notify_all() {};
  template <class T>
  inline void wait(unique_lock<T> lock) {(void)lock;};
};

enum memory_order
{
    memory_order_relaxed,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst,
    memory_order_consume,
};

template <class T>
class atomic
{
  T t;
  public:
  inline atomic() : t() {}
  inline atomic(const atomic<T>& other) : t(other) {}
  inline T& operator=(const T& other) {return (t = other);}
  inline T& load(memory_order m) {(void)m; return t;}
  inline void store(const T& other, memory_order m) {(void)m; t = other;}
};

}
#endif

#endif  // CORE_MULTITHREAD_H_
