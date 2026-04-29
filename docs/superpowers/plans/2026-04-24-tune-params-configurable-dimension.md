# tune-params Configurable Tuning Dimension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the parameter tuned by `tune-params` user-selectable from the cfg file, with an allowlist of `cpuctExploration`, `fpuReductionMax`, `rootFpuReductionMax`. QRSOptimizer continues to tune one dimension per run.

**Architecture:** Extend the existing one-entry compile-time `tuneDims[]` table at `cpp/command/tuneparams.cpp:44` to three entries. Add a `TuneParams::resolveDimension(ConfigParser&)` lookup function exposed in a new header `cpp/command/tuneparams.h`, with module-local `TuneParams::runTests()` wired into `cpp/command/runtests.cpp` (matching the existing `QRSTune::runTests()` pattern). At the start of `MainCmds::tuneparams()`, build a 1-element `vector<TuneDimension> activeDims` from the cfg's `tuneDimension` value; the rest of the function iterates `activeDims` instead of `tuneDims` and switches stack-array VLAs to `std::vector` so `nDims` can be runtime-derived.

**Tech Stack:** C++ (KataGo cpp/), CMake build, existing `testAssert` macro from `cpp/tests/testcommon.h`.

**Spec:** `docs/superpowers/specs/2026-04-24-tune-params-configurable-dimension-design.md`

---

## Task 0: Build Setup (one-time)

If `cpp/build/` already exists and `cd cpp/build && make katago` works, skip this task.

**Files:**
- No source edits.

- [ ] **Step 1: Create build directory**

```bash
mkdir -p cpp/build
```

- [ ] **Step 2: Configure CMake (macOS arm64 with Metal)**

The user's auto-memory notes that this codebase needs the `eigen@3` keg, not the default `eigen` 5.x. The `Eigen3_DIR` flag points at the keg's CMake config. From repo root:

```bash
cd cpp/build && \
/opt/homebrew/bin/cmake .. \
  -DUSE_BACKEND=METAL \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path) \
  -DEigen3_DIR=/opt/homebrew/opt/eigen@3/share/eigen3/cmake
```

Expected: configures without errors and ends with "Build files have been written to ...". If a different backend is preferred (e.g. EIGEN-only for portable testing), pass `-DUSE_BACKEND=EIGEN` instead.

- [ ] **Step 3: Verify a baseline build works**

```bash
cd cpp/build && make -j8 katago
```

Expected: builds the `katago` binary in `cpp/build/`. If it fails on the unmodified branch, fix the local environment before continuing — do not start Task 1 against a broken baseline.

- [ ] **Step 4: Verify runtests on baseline**

```bash
./cpp/build/katago runtests
```

Expected: exits 0 with the existing `QRSTune` and other tests passing. This is the regression baseline for Task 1.

---

## Task 1: Add Allowlist, Resolver, Tests, and Wire activeDims

This task does the entire structural change in one commit. The intermediate states are not meaningful: extending `tuneDims[]` to 3 entries while leaving `MainCmds::tuneparams()` indexed at `[0]` would either break the QRSTuner dimension count (if `nDims = sizeof(...)/sizeof(...)` is left alone) or require a throwaway `nDims = 1` placeholder. The TDD ordering inside the task still has the test added before the resolver implementation passes.

**Files:**
- Create: `cpp/command/tuneparams.h`
- Modify: `cpp/command/tuneparams.cpp` (currently 442 lines; this task touches the file-scope table at lines 44-47, function bodies of `computeParamCIs` 60-76 and `printRegressionCurves` 80-141, and the entire `MainCmds::tuneparams()` function at 143-441)
- Modify: `cpp/command/runtests.cpp` (add include and one call site)

- [ ] **Step 1: Create the public header**

`cpp/command/tuneparams.h`:

```cpp
#ifndef COMMAND_TUNEPARAMS_H_
#define COMMAND_TUNEPARAMS_H_

#include "../core/config_parser.h"

namespace TuneParams {
  // Returns the index of the selected dimension within tuneDims[].
  // Throws StringError if cfg's tuneDimension value is not in the allowlist.
  // Propagates ConfigParser's exception if tuneDimension is missing.
  int resolveDimension(ConfigParser& cfg);

  // Self-tests for the dim-resolver. Wired into MainCmds::runtests.
  void runTests();
}

#endif  // COMMAND_TUNEPARAMS_H_
```

- [ ] **Step 2: Add the include and the test wiring in runtests.cpp**

Open `cpp/command/runtests.cpp`. Insert this `#include` near the existing `#include "../qrstune/QRSOptimizer.h"` at line 7:

```cpp
#include "../command/tuneparams.h"
```

Insert this call adjacent to the existing `QRSTune::runTests();` call at line 39:

```cpp
TuneParams::runTests();
```

- [ ] **Step 3: Extend the tuneDims[] table to 3 entries**

In `cpp/command/tuneparams.cpp`, replace the existing one-entry table at lines 44-47:

```cpp
static const TuneDimension tuneDims[] = {
  {"cpuctExploration",          "Cpuct",  0.5,  1.5,  "cpuctExplorationMin",          "cpuctExplorationMax",          &SearchParams::cpuctExploration},
};
static const int nDims = sizeof(tuneDims) / sizeof(tuneDims[0]);
```

with:

```cpp
static const TuneDimension tuneDims[] = {
  {"cpuctExploration",     "Cpuct",   0.5,  1.5,
   "cpuctExplorationMin",     "cpuctExplorationMax",     &SearchParams::cpuctExploration},
  {"fpuReductionMax",      "Fpu",     0.0,  0.5,
   "fpuReductionMaxMin",      "fpuReductionMaxMax",      &SearchParams::fpuReductionMax},
  {"rootFpuReductionMax",  "RootFpu", 0.0,  0.4,
   "rootFpuReductionMaxMin",  "rootFpuReductionMaxMax",  &SearchParams::rootFpuReductionMax},
};
static const int nAllowedDims = (int)(sizeof(tuneDims) / sizeof(tuneDims[0]));
```

Note: the file-scope `static const int nDims` is removed entirely. It will be reintroduced as a *function-local* `const int nDims = (int)activeDims.size();` inside `MainCmds::tuneparams()` (Step 7) and passed as a parameter to the helpers (Steps 5 and 6).

- [ ] **Step 4: Add the include for the new header at the top of tuneparams.cpp**

In `cpp/command/tuneparams.cpp`, add this line near the existing `#include "../qrstune/QRSOptimizer.h"` at line 17:

```cpp
#include "../command/tuneparams.h"
```

- [ ] **Step 5: Refactor computeParamCIs to take nDims as a parameter**

In `cpp/command/tuneparams.cpp`, replace the function at lines 60-76:

```cpp
static bool computeParamCIs(const QRSTune::QRSTuner& tuner,
                             const vector<double>& vBest,
                             const double* mins, const double* maxs,
                             double* ciLo, double* ciHi, bool* clamped) {
  double se[nDims];
  bool hasCIs = tuner.model().computeOptimumSE(
    tuner.buffer().xs(), se, clamped);
  if(!hasCIs) return false;
  for(int d = 0; d < nDims; d++) {
    double radius = (maxs[d] - mins[d]) * 0.5;
    double seReal = se[d] * radius;
    double bestReal = qrsDimToReal(d, vBest[d], mins, maxs);
    ciLo[d] = bestReal - Z_95 * seReal;
    ciHi[d] = bestReal + Z_95 * seReal;
  }
  return true;
}
```

with:

```cpp
static bool computeParamCIs(int nDims,
                             const QRSTune::QRSTuner& tuner,
                             const vector<double>& vBest,
                             const double* mins, const double* maxs,
                             double* ciLo, double* ciHi, bool* clamped) {
  vector<double> se(nDims);
  bool hasCIs = tuner.model().computeOptimumSE(
    tuner.buffer().xs(), se.data(), clamped);
  if(!hasCIs) return false;
  for(int d = 0; d < nDims; d++) {
    double radius = (maxs[d] - mins[d]) * 0.5;
    double seReal = se[d] * radius;
    double bestReal = qrsDimToReal(d, vBest[d], mins, maxs);
    ciLo[d] = bestReal - Z_95 * seReal;
    ciHi[d] = bestReal + Z_95 * seReal;
  }
  return true;
}
```

- [ ] **Step 6: Refactor printRegressionCurves to take activeDims as a parameter**

In `cpp/command/tuneparams.cpp`, replace the function signature and the dim-name access. The function starts at line 80. The signature change:

```cpp
static void printRegressionCurves(const vector<TuneDimension>& activeDims,
                                   const QRSTune::QRSTuner& tuner,
                                   const vector<double>& vBest,
                                   const double* mins, const double* maxs,
                                   Logger& logger) {
  const int plotW = 60;
  const int plotH = 20;
  const int nDims = (int)activeDims.size();
  double bestWinRate = tuner.model().predict(vBest.data());
```

(the `const int nDims = (int)activeDims.size();` line replaces the prior reliance on the file-scope `nDims`).

Inside the function, the only reference to `tuneDims[dim].name` is at line 108:

```cpp
      "[Dim " + Global::intToString(dim) + "] " + tuneDims[dim].name +
```

Change to:

```cpp
      "[Dim " + Global::intToString(dim) + "] " + activeDims[dim].name +
```

No other lines inside `printRegressionCurves` need changing — the loop bound `for(int dim = 0; dim < nDims; dim++)` at line 88 picks up the new local `nDims`.

- [ ] **Step 7: Replace MainCmds::tuneparams() body to use activeDims**

The function `MainCmds::tuneparams()` starts at line 143. Two clusters of changes:

**Cluster A — Setup (replaces the existing range-resolution block at lines 186-206):**

Find this block:

```cpp
  //Search ranges (configurable; defaults preserve prior behaviour)
  double qrsMins[nDims], qrsMaxs[nDims];
  for(int d = 0; d < nDims; d++) {
    qrsMins[d] = cfg.contains(tuneDims[d].minKey)
                    ? cfg.getDouble(tuneDims[d].minKey, -1e9, 1e9)
                    : tuneDims[d].defaultMin;
    qrsMaxs[d] = cfg.contains(tuneDims[d].maxKey)
                    ? cfg.getDouble(tuneDims[d].maxKey, -1e9, 1e9)
                    : tuneDims[d].defaultMax;
    if(qrsMins[d] >= qrsMaxs[d])
      throw StringError(
        string("tune-params: ") + tuneDims[d].minKey + " must be < " + tuneDims[d].maxKey);
  }
  {
    string rangeStr;
    for(int d = 0; d < nDims; d++) {
      if(d > 0) rangeStr += ", ";
      rangeStr += string(tuneDims[d].name) + "=[" +
        Global::strprintf("%.4f", qrsMins[d]) + "," + Global::strprintf("%.4f", qrsMaxs[d]) + "]";
    }
    logger.write("QRS ranges: " + rangeStr);
  }
```

Replace with:

```cpp
  //Resolve which dimension to tune from cfg's tuneDimension key.
  int selectedDimIdx = TuneParams::resolveDimension(cfg);
  vector<TuneDimension> activeDims = { tuneDims[selectedDimIdx] };
  const int nDims = (int)activeDims.size();
  logger.write("Tuning dimension: " + string(activeDims[0].name));

  //Search ranges (configurable; defaults preserve prior behaviour)
  vector<double> qrsMins(nDims), qrsMaxs(nDims);
  for(int d = 0; d < nDims; d++) {
    qrsMins[d] = cfg.contains(activeDims[d].minKey)
                    ? cfg.getDouble(activeDims[d].minKey, -1e9, 1e9)
                    : activeDims[d].defaultMin;
    qrsMaxs[d] = cfg.contains(activeDims[d].maxKey)
                    ? cfg.getDouble(activeDims[d].maxKey, -1e9, 1e9)
                    : activeDims[d].defaultMax;
    if(qrsMins[d] >= qrsMaxs[d])
      throw StringError(
        string("tune-params: ") + activeDims[d].minKey + " must be < " + activeDims[d].maxKey);
  }
  {
    string rangeStr;
    for(int d = 0; d < nDims; d++) {
      if(d > 0) rangeStr += ", ";
      rangeStr += string(activeDims[d].name) + "=[" +
        Global::strprintf("%.4f", qrsMins[d]) + "," + Global::strprintf("%.4f", qrsMaxs[d]) + "]";
    }
    logger.write("QRS ranges: " + rangeStr);
  }
```

**Cluster B — All other `tuneDims[d]` references in MainCmds::tuneparams() become `activeDims[d]`, and stack-array VLAs become vectors. The remaining sites:**

1. Line 268, the per-trial param assignment:
```cpp
    for(int d = 0; d < nDims; d++)
      expParams.*(tuneDims[d].field) = qrsDimToReal(d, sample[d], qrsMins, qrsMaxs);
```
becomes:
```cpp
    for(int d = 0; d < nDims; d++)
      expParams.*(activeDims[d].field) = qrsDimToReal(d, sample[d], qrsMins.data(), qrsMaxs.data());
```

2. Lines 357-367, the progress-report CI block:
```cpp
        string paramStr;
        double ciLo[nDims], ciHi[nDims];
        bool clampedDims[nDims];
        if(computeParamCIs(tuner, vBest, qrsMins, qrsMaxs, ciLo, ciHi, clampedDims)) {
          for(int d = 0; d < nDims; d++) {
            paramStr += Global::strprintf(" %s=[%.4f, %.4f]", tuneDims[d].shortName, ciLo[d], ciHi[d]);
            if(clampedDims[d]) paramStr += "*";
          }
        } else {
          for(int d = 0; d < nDims; d++)
            paramStr += Global::strprintf(" %s=%.4f", tuneDims[d].shortName, qrsDimToReal(d, vBest[d], qrsMins, qrsMaxs));
        }
```
becomes:
```cpp
        string paramStr;
        vector<double> ciLo(nDims), ciHi(nDims);
        std::unique_ptr<bool[]> clampedRaw(new bool[nDims]);
        if(computeParamCIs(nDims, tuner, vBest, qrsMins.data(), qrsMaxs.data(),
                           ciLo.data(), ciHi.data(), clampedRaw.get())) {
          for(int d = 0; d < nDims; d++) {
            paramStr += Global::strprintf(" %s=[%.4f, %.4f]", activeDims[d].shortName, ciLo[d], ciHi[d]);
            if(clampedRaw[d]) paramStr += "*";
          }
        } else {
          for(int d = 0; d < nDims; d++)
            paramStr += Global::strprintf(" %s=%.4f", activeDims[d].shortName,
              qrsDimToReal(d, vBest[d], qrsMins.data(), qrsMaxs.data()));
        }
```

Note: `computeParamCIs`'s last parameter is `bool* clamped`. To avoid relying on the GCC/Clang VLA extension while keeping a contiguous bool array (which `vector<bool>` cannot provide because of its proxy specialization), use `std::unique_ptr<bool[]>` and pass `.get()`. `<memory>` is transitively included via `cpp/core/global.h:22`, so no extra `#include` is needed.

3. Lines 388-402, the final-result CI block:
```cpp
    double ciLo[nDims], ciHi[nDims];
    bool clampedDims[nDims];
    bool hasCIs = computeParamCIs(tuner, vBest, qrsMins, qrsMaxs, ciLo, ciHi, clampedDims);

    for(int d = 0; d < nDims; d++) {
      double bestReal = qrsDimToReal(d, vBest[d], qrsMins, qrsMaxs);
      if(hasCIs) {
        string warn = clampedDims[d] ? "  [boundary - CI may be unreliable]" : "";
        logger.write(Global::strprintf("Best %-25s = %.4f  95%%CI [%.4f, %.4f]%s",
          tuneDims[d].name, bestReal, ciLo[d], ciHi[d], warn.c_str()));
      } else {
        logger.write(Global::strprintf("Best %-25s = %.4f  (CI unavailable)",
          tuneDims[d].name, bestReal));
      }
    }
```
becomes:
```cpp
    vector<double> ciLo(nDims), ciHi(nDims);
    std::unique_ptr<bool[]> clampedRaw(new bool[nDims]);
    bool hasCIs = computeParamCIs(nDims, tuner, vBest, qrsMins.data(), qrsMaxs.data(),
                                   ciLo.data(), ciHi.data(), clampedRaw.get());

    for(int d = 0; d < nDims; d++) {
      double bestReal = qrsDimToReal(d, vBest[d], qrsMins.data(), qrsMaxs.data());
      if(hasCIs) {
        string warn = clampedRaw[d] ? "  [boundary - CI may be unreliable]" : "";
        logger.write(Global::strprintf("Best %-25s = %.4f  95%%CI [%.4f, %.4f]%s",
          activeDims[d].name, bestReal, ciLo[d], ciHi[d], warn.c_str()));
      } else {
        logger.write(Global::strprintf("Best %-25s = %.4f  (CI unavailable)",
          activeDims[d].name, bestReal));
      }
    }
```

(uses `std::unique_ptr<bool[]>` instead of VLA to avoid the compiler-extension; `<memory>` is already transitively included via `cpp/core/global.h:22`).

4. Lines 405-411, raw QRS coordinates output:
```cpp
    string rawStr;
    for(int d = 0; d < nDims; d++) {
      if(d > 0) rawStr += ", ";
      rawStr += Global::doubleToString(vBest[d]);
    }
```
No `tuneDims[d]` reference here; the loop uses `nDims` only — keep as-is (now picking up the function-local `nDims`).

5. Line 414, the call to `printRegressionCurves`:
```cpp
  printRegressionCurves(tuner, vBest, qrsMins, qrsMaxs, logger);
```
becomes:
```cpp
  printRegressionCurves(activeDims, tuner, vBest, qrsMins.data(), qrsMaxs.data(), logger);
```

6. Lines 418-422, the suggested-match command:
```cpp
    string overrides = "botName0=tuned,botName1=default,";
    for(int d = 0; d < nDims; d++)
      overrides += string(tuneDims[d].name) + "0=" +
        Global::strprintf("%.4f", qrsDimToReal(d, vBest[d], qrsMins, qrsMaxs)) + ",";
```
becomes:
```cpp
    string overrides = "botName0=tuned,botName1=default,";
    for(int d = 0; d < nDims; d++)
      overrides += string(activeDims[d].name) + "0=" +
        Global::strprintf("%.4f", qrsDimToReal(d, vBest[d], qrsMins.data(), qrsMaxs.data())) + ",";
```

After all the above, no `tuneDims[d]` references should remain inside `MainCmds::tuneparams()`. The file-scope `tuneDims[]` table is still referenced from `TuneParams::resolveDimension` (Step 8) and `TuneParams::runTests` (Step 9), which are added next.

- [ ] **Step 8: Add TuneParams::resolveDimension implementation in tuneparams.cpp**

Add at the bottom of `cpp/command/tuneparams.cpp`, after `MainCmds::tuneparams()`:

```cpp
int TuneParams::resolveDimension(ConfigParser& cfg) {
  string tuneDimName = cfg.getString("tuneDimension");
  for(int i = 0; i < nAllowedDims; i++) {
    if(tuneDimName == tuneDims[i].name) return i;
  }
  string allowed;
  for(int i = 0; i < nAllowedDims; i++) {
    if(i > 0) allowed += ", ";
    allowed += tuneDims[i].name;
  }
  throw StringError("tune-params: tuneDimension = '" + tuneDimName +
                    "' not recognized; expected one of: " + allowed);
}
```

- [ ] **Step 9: Add TuneParams::runTests implementation in tuneparams.cpp**

Add immediately after `TuneParams::resolveDimension`. Add this `#include` near the top of `tuneparams.cpp` if not already present:

```cpp
#include "../tests/testcommon.h"
```

Then the test body:

```cpp
void TuneParams::runTests() {
  cout << "Running TuneParams tests" << endl;

  // Positive: each allowed name maps to its expected index.
  {
    ConfigParser cfg(std::map<string,string>{{"tuneDimension", "cpuctExploration"}});
    testAssert(TuneParams::resolveDimension(cfg) == 0);
  }
  {
    ConfigParser cfg(std::map<string,string>{{"tuneDimension", "fpuReductionMax"}});
    testAssert(TuneParams::resolveDimension(cfg) == 1);
  }
  {
    ConfigParser cfg(std::map<string,string>{{"tuneDimension", "rootFpuReductionMax"}});
    testAssert(TuneParams::resolveDimension(cfg) == 2);
  }

  // Negative: missing tuneDimension key throws (any exception type — the
  // exact type belongs to ConfigParser and is not this unit's contract).
  {
    ConfigParser cfg(std::map<string,string>{});
    bool threw = false;
    try { (void)TuneParams::resolveDimension(cfg); }
    catch(const std::exception&) { threw = true; }
    testAssert(threw);
  }

  // Negative: unrecognized value throws StringError mentioning the offending
  // name and at least one allowed name.
  {
    ConfigParser cfg(std::map<string,string>{{"tuneDimension", "totallyMadeUpName"}});
    bool threwStringError = false;
    string msg;
    try { (void)TuneParams::resolveDimension(cfg); }
    catch(const StringError& e) { threwStringError = true; msg = e.what(); }
    testAssert(threwStringError);
    testAssert(msg.find("totallyMadeUpName") != string::npos);
    testAssert(msg.find("cpuctExploration") != string::npos);
  }
}
```

The `ConfigParser(const std::map<std::string, std::string>&)` constructor exists at `cpp/core/config_parser.h:26` — gives a clean way to build cfgs in-memory without touching the filesystem.

- [ ] **Step 10: Build**

```bash
cd cpp/build && make -j8 katago 2>&1 | tail -30
```

Expected: clean build to completion. `<memory>` and `<map>` are already pulled in transitively via `cpp/core/global.h:15,22`, so the `std::unique_ptr<bool[]>` and `ConfigParser(std::map<...>)` uses compile without additional includes. The file already has `using namespace std;` at line 24.

- [ ] **Step 11: Run tests**

```bash
./cpp/build/katago runtests 2>&1 | tail -30
```

Expected output includes:
```
Running QRSTune tests
Running TuneParams tests
```
and the run exits 0. If any `testAssert` fires, the run aborts with the assertion location.

- [ ] **Step 12: Commit**

```bash
git add cpp/command/tuneparams.h cpp/command/tuneparams.cpp cpp/command/runtests.cpp
git commit -m "$(cat <<'EOF'
tune-params: add tuneDimension cfg key and 3-dim allowlist

Allowlist: cpuctExploration, fpuReductionMax, rootFpuReductionMax.
Strict validation — missing or unrecognized tuneDimension throws.
Selection lives in TuneParams::resolveDimension; tested via runTests.
QRSOptimizer is unchanged and still tunes one dimension per run.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Update Example Config

**Files:**
- Modify: `cpp/configs/tune_params_example.cfg`

- [ ] **Step 1: Update the file header comment**

Open `cpp/configs/tune_params_example.cfg`. Find the comment at line 9-11:

```
# After all trials, it reports the best-found value for cpuctExploration,
# along with an ASCII regression curve showing the parameter's estimated
# effect on win rate.
```

Replace with:

```
# After all trials, it reports the best-found value for the selected
# tuneDimension, along with an ASCII regression curve showing the
# parameter's estimated effect on win rate.
```

- [ ] **Step 2: Replace the Tuning block**

Find the existing Tuning block at lines 15-25:

```
# Tuning------------------------------------------------------------------------------------

# Total number of tuning trials (games). More trials = better estimates but slower.
# A few hundred trials is a reasonable starting point; 1000+ for higher confidence.
numTrials = 500

# Search range for the PUCT parameter being tuned.
# The optimizer explores within [Min, Max].
# If omitted, defaults are used: cpuctExploration [0.5, 1.5]
# cpuctExplorationMin = 0.5
# cpuctExplorationMax = 1.5
```

Replace with:

```
# Tuning------------------------------------------------------------------------------------

# Total number of tuning trials (games). More trials = better estimates but slower.
# A few hundred trials is a reasonable starting point; 1000+ for higher confidence.
numTrials = 500

# Which PUCT parameter to tune. Required. One of:
#   cpuctExploration       - PUCT exploration constant
#   fpuReductionMax        - FPU reduction at non-root nodes
#   rootFpuReductionMax    - FPU reduction at the root node
tuneDimension = cpuctExploration

# Search range for the selected dimension. Only the pair that matches
# `tuneDimension` is read; the others are ignored. If omitted, the
# defaults below are used:
#   cpuctExploration:    [0.5, 1.5]
#   fpuReductionMax:     [0.0, 0.5]
#   rootFpuReductionMax: [0.0, 0.4]
#
# cpuctExplorationMin     = 0.5    ;  cpuctExplorationMax     = 1.5
# fpuReductionMaxMin      = 0.0    ;  fpuReductionMaxMax      = 0.5
# rootFpuReductionMaxMin  = 0.0    ;  rootFpuReductionMaxMax  = 0.4
```

- [ ] **Step 3: Smoke-run the cfg parses and the command starts**

The cfg references `PATH_TO_MODEL` placeholders that need to resolve. Use a local model file (any kata1 model on disk) via override:

```bash
./cpp/build/katago tune-params \
  -config cpp/configs/tune_params_example.cfg \
  -override-config "numTrials=2,nnModelFile0=/path/to/your/local/model.bin.gz,nnModelFile1=/path/to/your/local/model.bin.gz" \
  -log-file /tmp/tune.log 2>&1 | head -40
```

If a model file isn't readily available, do a parse-only smoke check by running with a deliberately bad model path and confirming the failure happens *after* the cfg has been read and `tuneDimension` resolved:

```bash
./cpp/build/katago tune-params \
  -config cpp/configs/tune_params_example.cfg \
  -override-config "nnModelFile0=/nonexistent.bin.gz,nnModelFile1=/nonexistent.bin.gz" \
  -log-file /tmp/tune.log 2>&1 | head -40
```

Expected: the log should contain `Tuning dimension: cpuctExploration` and `QRS ranges: cpuctExploration=[0.5000,1.5000]` before any model-load failure.

- [ ] **Step 4: Smoke-run with each of the three dims via override**

For each of the new dims, override `tuneDimension` and verify the log line changes:

```bash
./cpp/build/katago tune-params \
  -config cpp/configs/tune_params_example.cfg \
  -override-config "tuneDimension=fpuReductionMax,nnModelFile0=/nonexistent.bin.gz,nnModelFile1=/nonexistent.bin.gz" \
  -log-file /tmp/tune.log 2>&1 | head -40
```

Expected: `Tuning dimension: fpuReductionMax` and `QRS ranges: fpuReductionMax=[0.0000,0.5000]`.

```bash
./cpp/build/katago tune-params \
  -config cpp/configs/tune_params_example.cfg \
  -override-config "tuneDimension=rootFpuReductionMax,nnModelFile0=/nonexistent.bin.gz,nnModelFile1=/nonexistent.bin.gz" \
  -log-file /tmp/tune.log 2>&1 | head -40
```

Expected: `Tuning dimension: rootFpuReductionMax` and `QRS ranges: rootFpuReductionMax=[0.0000,0.4000]`.

And verify the strict-validation negative case: an unrecognized name produces the helpful error message:

```bash
./cpp/build/katago tune-params \
  -config cpp/configs/tune_params_example.cfg \
  -override-config "tuneDimension=fooBar,nnModelFile0=/nonexistent.bin.gz,nnModelFile1=/nonexistent.bin.gz" \
  -log-file /tmp/tune.log 2>&1 | head -10
```

Expected: an error message containing `tuneDimension = 'fooBar' not recognized; expected one of: cpuctExploration, fpuReductionMax, rootFpuReductionMax`.

- [ ] **Step 5: Commit**

```bash
git add cpp/configs/tune_params_example.cfg
git commit -m "$(cat <<'EOF'
tune-params example cfg: document tuneDimension and per-dim ranges

Adds tuneDimension as a required key with the three allowed values
listed inline. The Min/Max overrides for all three dims are shown as
commented-out menu entries; only the pair matching the selected dim
is read.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done

After Task 2's commit:

- `git log --oneline -3` shows two new commits on top of `1611dbdf`.
- `./cpp/build/katago runtests` exits 0 with `Running TuneParams tests` in the output.
- `cpp/configs/tune_params_example.cfg` has `tuneDimension = cpuctExploration` and the per-dim range menu.
- `MainCmds::tuneparams()` no longer references file-scope `nDims` or `tuneDims[d]`; everything goes through `activeDims`.

Push the branch (or open a PR comment) only at the user's explicit request.
