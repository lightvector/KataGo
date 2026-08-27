#ifndef GTP_CONFIG_H_
#define GTP_CONFIG_H_

#include "../core/global.h"
#include "../game/boardhistory.h"

namespace GTPConfig {
  // deviceIdxs lists the devices to use, one server thread per entry by default;
  // serverThreadsPerDevice > 1 assigns that many server threads to each entry (or to the
  // backend's default device if deviceIdxs is empty). nnMaxBatchSize <= 0 leaves the config's
  // batch size at the backend default.
  std::string makeConfig(
    const Rules& rules,
    int64_t maxVisits,
    int64_t maxPlayouts,
    double maxTime,
    double maxPonderTime,
    const std::vector<int>& deviceIdxs,
    int serverThreadsPerDevice,
    int nnMaxBatchSize,
    int nnCacheSizePowerOfTwo,
    int nnMutexPoolSizePowerOfTwo,
    int numSearchThreads
  );
}

#endif //GTP_CONFIG_H_
