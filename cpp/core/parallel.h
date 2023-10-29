
#ifndef CORE_PARALLEL_H_
#define CORE_PARALLEL_H_

#include "../core/global.h"
#include "../core/multithread.h"

namespace Parallel {
  
  void iterRange(int numThreads, size_t size, const std::function<void(int, size_t)>& f);

}

#endif // CORE_PARALLEL_H_
