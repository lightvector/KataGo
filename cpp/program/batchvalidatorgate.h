#ifndef PROGRAM_BATCHVALIDATORGATE_H_
#define PROGRAM_BATCHVALIDATORGATE_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace BatchValidatorGate {

struct ModelRequest {
  std::string key;
  std::string name;
  std::function<bool()> isLoadedFn;
  std::function<void()> validateAndLoadFn;
};

struct GateConfig {
  bool enabled = true;
  int targetGames = 32;
  int initBatchCap = 32;
  int initialIdleTimeoutSec = 10;
  int maxPendingWaitSec = 45;
};

inline std::atomic<bool> g_pauseSearch{false};
inline std::atomic<bool> g_shutdownRequested{false};
inline std::mutex g_gateMutex;
inline std::condition_variable g_searchCv;
inline std::condition_variable g_gateCv;

inline std::unordered_map<std::string, std::function<void()>> g_pendingValidators;
inline int g_waitingTasks = 0;
inline int g_activeGames = 0;
inline bool g_isValidating = false;
inline bool g_initialBatchDone = false;
inline std::chrono::steady_clock::time_point g_firstPendingTs;
inline std::chrono::steady_clock::time_point g_lastTaskArrivalTs;

inline void notifyGameStarted() {
  std::lock_guard<std::mutex> lk(g_gateMutex);
  g_activeGames++;
}

inline void notifyGameFinished() {
  std::lock_guard<std::mutex> lk(g_gateMutex);
  if(g_activeGames > 0)
    g_activeGames--;
  g_gateCv.notify_all();
}

inline void requestShutdown() {
  g_shutdownRequested.store(true, std::memory_order_release);
  g_pauseSearch.store(false, std::memory_order_release);
  g_searchCv.notify_all();
  g_gateCv.notify_all();
}

// Lock-free fast path during normal search; blocks only while batch model validation is running.
inline void waitIfSearchPaused() {
  if(!g_pauseSearch.load(std::memory_order_relaxed))
    return;
  if(g_shutdownRequested.load(std::memory_order_acquire))
    return;
  std::unique_lock<std::mutex> lk(g_gateMutex);
  g_searchCv.wait(lk, []() {
    return !g_pauseSearch.load(std::memory_order_acquire) || g_shutdownRequested.load(std::memory_order_acquire);
  });
}

inline void gateBeforeLoadModels(
  const GateConfig& cfg,
  const std::vector<ModelRequest>& requests,
  const std::function<void(const std::string&)>& logFn
) {
  if(!cfg.enabled)
    return;
  if(g_shutdownRequested.load(std::memory_order_acquire))
    return;

  const int targetGames = std::max(1, cfg.targetGames);
  const int initBatchGames = std::max(1, std::min(targetGames, cfg.initBatchCap));
  const int waitNetsWm = (targetGames >= 8) ? std::max(2, targetGames / 8) : ((targetGames >= 2) ? 2 : 1);
  const int idleSlotsWm = (targetGames >= 4) ? std::max(2, targetGames / 4) : 1;

  std::unique_lock<std::mutex> lk(g_gateMutex);
  auto now = std::chrono::steady_clock::now();
  g_lastTaskArrivalTs = now;
  g_waitingTasks++;

  struct WaitingGuard {
    int& waiting;
    explicit WaitingGuard(int& w) : waiting(w) {}
    ~WaitingGuard() {
      if(waiting > 0)
        waiting--;
    }
  } waitingGuard(g_waitingTasks);

  bool hasNewNet = false;
  for(const auto& req : requests) {
    if(!req.name.empty() && !req.isLoadedFn()) {
      if(g_pendingValidators.empty())
        g_firstPendingTs = now;
      const std::string& mapKey = req.key.empty() ? req.name : req.key;
      if(g_pendingValidators.find(mapKey) == g_pendingValidators.end())
        g_pendingValidators.emplace(mapKey, req.validateAndLoadFn);
      hasNewNet = true;
    }
  }

  // Fast path: all required models are already resident in cache and cold-start batch has completed.
  if(!hasNewNet && (g_initialBatchDone || g_activeGames > 0))
    return;

  while(!g_shutdownRequested.load(std::memory_order_acquire)) {
    bool allReady = true;
    for(const auto& req : requests) {
      if(!req.name.empty() && !req.isLoadedFn()) {
        allReady = false;
        break;
      }
    }
    if(allReady && (g_initialBatchDone || g_activeGames > 0))
      break;

    if(!g_isValidating) {
      auto cur = std::chrono::steady_clock::now();
      int idleSlots = std::max(0, targetGames - g_activeGames);
      long long pendingSec = g_pendingValidators.empty()
        ? 0
        : std::chrono::duration_cast<std::chrono::seconds>(cur - g_firstPendingTs).count();
      long long idleArrivalSec = std::chrono::duration_cast<std::chrono::seconds>(cur - g_lastTaskArrivalTs).count();

      bool shouldTrigger = false;
      if(!g_initialBatchDone && g_activeGames == 0) {
        if(g_waitingTasks >= initBatchGames || (g_waitingTasks >= 1 && idleArrivalSec >= cfg.initialIdleTimeoutSec)) {
          g_initialBatchDone = true;
          shouldTrigger = true;
        }
      }
      else if(!g_pendingValidators.empty()) {
        if(static_cast<int>(g_pendingValidators.size()) >= waitNetsWm
           || idleSlots >= idleSlotsWm
           || pendingSec >= cfg.maxPendingWaitSec) {
          shouldTrigger = true;
        }
      }

      if(shouldTrigger) {
        if(g_pendingValidators.empty()) {
          g_initialBatchDone = true;
          g_gateCv.notify_all();
          break;
        }

        g_isValidating = true;
        std::vector<std::pair<std::string, std::function<void()>>> batch;
        batch.reserve(g_pendingValidators.size());
        for(auto& kv : g_pendingValidators)
          batch.emplace_back(kv.first, std::move(kv.second));
        g_pendingValidators.clear();

        g_pauseSearch.store(true, std::memory_order_release);
        int snapActive = g_activeGames;
        int snapWaiting = g_waitingTasks;
        lk.unlock();

        struct ValidationGuard {
          std::unique_lock<std::mutex>& lock;
          bool& validatingFlag;
          ValidationGuard(std::unique_lock<std::mutex>& l, bool& flag) : lock(l), validatingFlag(flag) {}
          ~ValidationGuard() {
            if(!lock.owns_lock())
              lock.lock();
            validatingFlag = false;
            g_pauseSearch.store(false, std::memory_order_release);
            g_searchCv.notify_all();
            g_gateCv.notify_all();
          }
        } valGuard(lk, g_isValidating);

        auto tStart = std::chrono::steady_clock::now();
        logFn(
          "BatchValidatorGate: pausing MCTS search to batch-validate " + std::to_string(batch.size()) +
          " net(s) (active=" + std::to_string(snapActive) + "/" + std::to_string(targetGames) +
          ", idle=" + std::to_string(std::max(0, targetGames - snapActive)) +
          ", waitingTasks=" + std::to_string(snapWaiting) + ")"
        );

        for(const auto& item : batch) {
          if(g_shutdownRequested.load(std::memory_order_acquire))
            break;
          item.second();
        }

        auto tEnd = std::chrono::steady_clock::now();
        long long elapsedSec = std::chrono::duration_cast<std::chrono::seconds>(tEnd - tStart).count();
        logFn(
          "BatchValidatorGate: finished validating " + std::to_string(batch.size()) +
          " net(s) in " + std::to_string(elapsedSec) + "s, resuming search"
        );
        continue;
      }
    }
    g_gateCv.wait_for(lk, std::chrono::seconds(2));
  }
}

} // namespace BatchValidatorGate

#endif // PROGRAM_BATCHVALIDATORGATE_H_
