#ifndef GTP_CONFIG_H_
#define GTP_CONFIG_H_

#include "../core/global.h"
#include "../game/boardhistory.h"

namespace GTPConfig {
  std::string makeConfig(
    const Rules& rules,
    int64_t maxVisits,
    int64_t maxPlayouts,
    double maxTime,
    double maxPonderTime,
    std::vector<int> deviceIdxs,
    int nnCacheSizePowerOfTwo,
    int nnMutexPoolSizePowerOfTwo,
    int numSearchThreads
  );
}

#endif //GTP_CONFIG_H_
