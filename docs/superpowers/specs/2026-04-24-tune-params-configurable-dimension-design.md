# tune-params Configurable Tuning Dimension — Design

**Date:** 2026-04-24
**Subject PR:** lightvector/KataGo PR #1178 (`tune-params` subcommand for QRS-based PUCT tuning)
**Branch:** `claude/katago-puct-tuning-guide-OXzst`
**Goal:** Make the parameter tuned by `tune-params` user-selectable from the cfg file, with an initial allowlist of `cpuctExploration`, `fpuReductionMax`, `rootFpuReductionMax`. QRSOptimizer continues to tune one dimension per run.

---

## Background

The current `tune-params` implementation hard-codes `cpuctExploration` as the only tunable dimension via a one-entry compile-time table at `cpp/command/tuneparams.cpp:44`:

```cpp
static const TuneDimension tuneDims[] = {
  {"cpuctExploration", "Cpuct", 0.5, 1.5,
   "cpuctExplorationMin", "cpuctExplorationMax", &SearchParams::cpuctExploration},
};
static const int nDims = sizeof(tuneDims) / sizeof(tuneDims[0]);
```

`TuneDimension` already carries everything needed to drive the optimizer for an arbitrary `double` field of `SearchParams`: a display name, short label, default `[Min, Max]`, the cfg keys for those bounds, and a pointer-to-member for writing into the experiment bot's parameters. The PR's commit history (`0fadb78e`, `eb8ca5e3`, `7a9f39b9`, `29dba7ab`, `16748a3f`) shows dimensions being added and removed during development — the table-driven design is load-bearing, not incidental.

This design extends the table from one entry to three, adds a cfg selector for which entry is active, and isolates "the active dim for this run" as a separate runtime concept from "the allowlist of supported dims."

## Goals

- Allow users to tune `cpuctExploration`, `fpuReductionMax`, or `rootFpuReductionMax` by changing one line of cfg.
- Reject unset or unrecognized `tuneDimension` values with a clear error listing the allowed names.
- Preserve the existing iteration pattern (`for(int d = 0; d < nDims; d++)`) so a future extension to multi-dim QRS would not require structural changes.
- No change to QRSOptimizer or its tests. The optimizer remains 1D per run.

## Non-Goals

- Tuning multiple dimensions in a single run.
- Reflection-style support for arbitrary `double` fields of `SearchParams` by name.
- Re-introducing dimensions that were deliberately removed during PR development (`cpuctExplorationLog`, `cpuctUtilityStdevPrior`, `cpuctUtilityStdevPriorWeight`, `cpuctUtilityStdevScale`).
- New automated tests beyond the one config-parse/lookup test described under Testing below.

## Architecture

The change touches `cpp/command/tuneparams.cpp`, `cpp/configs/tune_params_example.cfg`, and adds a small new header `cpp/command/tuneparams.h` plus a one-line wiring change in `cpp/command/runtests.cpp`.

The conceptual split:

- `tuneDims[]` is the **allowlist** — a compile-time table of every dimension the command knows how to tune.
- `activeDims` is the **selection for this run** — a `vector<TuneDimension>` of size 1 built at startup from the cfg's `tuneDimension` value.
- `nDims` becomes runtime (`= (int)activeDims.size()`) instead of a compile-time constant.

All loops, output formatting, CI computation, and the suggested-match command iterate over `activeDims`. The optimizer is constructed with `nDims = 1` exactly as before.

## Allowlist

Three entries in `tuneDims[]`:

```cpp
static const TuneDimension tuneDims[] = {
  {"cpuctExploration",     "Cpuct",   0.5,  1.5,
   "cpuctExplorationMin",     "cpuctExplorationMax",     &SearchParams::cpuctExploration},
  {"fpuReductionMax",      "Fpu",     0.0,  0.5,
   "fpuReductionMaxMin",      "fpuReductionMaxMax",      &SearchParams::fpuReductionMax},
  {"rootFpuReductionMax",  "RootFpu", 0.0,  0.4,
   "rootFpuReductionMaxMin",  "rootFpuReductionMaxMax",  &SearchParams::rootFpuReductionMax},
};
```

### Default range justification

| Dim | Canonical value | Default range | Rationale |
|---|---|---|---|
| `cpuctExploration` | ~0.9 | `[0.5, 1.5]` | Unchanged from current PR. |
| `fpuReductionMax` | 0.2 | `[0.0, 0.5]` | Includes 0 (no FPU reduction) as a legitimate optimum. Default 0.2 lands at 40% of the range — good headroom on both sides. Wide enough that the boundary-warning logic (`clampedDims`) self-corrects if the optimum drifts; tight enough that QRS's quadratic regression has signal. |
| `rootFpuReductionMax` | 0.1 | `[0.0, 0.4]` | Same logic. Root tends to want less FPU reduction than interior, so the upper bound is slightly tighter than `fpuReductionMax`. Default 0.1 lands at 25%. |

Both new dims have a lower bound of 0 because "no reduction" is meaningful and may be optimal for some networks. If a tuned optimum lands at exactly 0.0 with `clampedDims` set, the operator can either accept the result or re-run with a different range.

## Cfg Selector

A new required cfg key `tuneDimension` selects which entry of `tuneDims[]` is active. Validation:

- **Missing key** → `cfg.getString("tuneDimension")` throws `ConfigParseError` with the standard "key not found" message. Strict validation: every cfg must explicitly choose. Acceptable because the PR is unmerged, so there is no existing cfg in production to migrate.
- **Unrecognized value** → throw `StringError` with the value the user supplied and a comma-separated list of the allowed names.
- **Recognized value** → `selectedDimIdx` set to the matching index; `activeDims` constructed.

Per-dimension `Min`/`Max` keys (`cpuctExplorationMin/Max`, `fpuReductionMaxMin/Max`, `rootFpuReductionMaxMin/Max`) follow the existing pattern: each dim has its own pair, only the pair matching the selected dim is read, and omitted pairs fall back to the table's default. Switching `tuneDimension` does not require also editing range lines, and the example cfg can document all three dims as a menu of commented-out overrides.

## Code Changes

All in `cpp/command/tuneparams.cpp`.

### Setup phase (before line 209's `Setup::loadParams`)

```cpp
int selectedDimIdx = TuneParams::resolveDimension(cfg);
vector<TuneDimension> activeDims = { tuneDims[selectedDimIdx] };
const int nDims = (int)activeDims.size();
```

The lookup-and-validation logic itself lives in `TuneParams::resolveDimension`:

```cpp
int TuneParams::resolveDimension(ConfigParser& cfg) {
  const int nAllowedDims = (int)(sizeof(tuneDims) / sizeof(tuneDims[0]));
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

The file-scope `static const int nDims = ...` declaration at line 47 is removed; `nDims` becomes a function-local `const int` derived from `activeDims.size()`. The `tuneDims[]` table keeps its `static` qualifier — `TuneParams::resolveDimension` and `TuneParams::runTests` are both defined in the same translation unit (`tuneparams.cpp`), so file-scope linkage is sufficient.

### Iteration changes

Throughout `tuneparams()`, replace every reference to `tuneDims[d]` with `activeDims[d]`. The `for(int d = 0; d < nDims; d++)` loops continue to execute exactly once per iteration (since `nDims == 1`), so the runtime behavior is unchanged.

### VLA → vector

Three sites currently use compile-time-sized stack arrays that depend on the file-scope `nDims`:

| Site | Current | Replacement |
|---|---|---|
| `tuneparams()` setup | `double qrsMins[nDims], qrsMaxs[nDims]` | `vector<double> qrsMins(nDims), qrsMaxs(nDims)` |
| `tuneparams()` progress + final | `double ciLo[nDims], ciHi[nDims]; bool clampedDims[nDims]` | `vector<double> ciLo(nDims), ciHi(nDims); vector<char> clampedDims(nDims)` (using `char` to avoid `vector<bool>`'s proxy semantics) |
| `computeParamCIs` body | `double se[nDims]` | `vector<double> se(nDims)` |

`computeParamCIs` and `printRegressionCurves` currently rely on the file-scope `nDims`. They gain an explicit `int nDims` parameter (and `const vector<TuneDimension>& dims` if they need to format dim names — the body of `printRegressionCurves` already accesses `tuneDims[dim].name` at line 108).

### Suggested-match command (lines 417-431)

The existing loop already iterates `tuneDims[d].name` to construct the override string. After the rename to `activeDims[d].name`, it naturally emits the override for the selected dim only. No further work.

## Example Cfg Changes

`cpp/configs/tune_params_example.cfg`. The "Tuning" block becomes:

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

The header-comment paragraph at the top of the file changes from "After all trials, it reports the best-found value for cpuctExploration" to "for the selected `tuneDimension`."

The "Internal params" block (lines 95-105) — which lists the *fixed* bot0 reference values, including `fpuReductionMax = 0.2` and `rootFpuReductionMax = 0.1` — is unchanged. Those are bot0 defaults, not bot1 sweep ranges.

All other sections (Bots, Match, Rules, Search limits, GPU Settings, GTP-equivalent defaults) are unchanged.

## Error Handling

| Condition | Behavior |
|---|---|
| `tuneDimension` key absent from cfg | `ConfigParser` throws — message matches the standard cfg "missing required key" format. |
| `tuneDimension` value not in allowlist | `StringError` with the supplied value and the comma-joined list of allowed names. |
| Per-dim `Min`/`Max` keys for a non-selected dim present in cfg | Silently ignored (no warning). Matches existing cfg behavior for unread keys. |
| Per-dim `Min` >= `Max` for the selected dim | Existing `StringError` at line 195 fires unchanged. |

## Testing

The PR's existing test pattern is module-local `runTests()` functions wired into `MainCmds::runtests` (e.g. `QRSTune::runTests()` at `cpp/command/runtests.cpp:39`). The tune-params test follows the same pattern.

**File layout.** Add a new header `cpp/command/tuneparams.h` exposing a `TuneParams` namespace:

```cpp
namespace TuneParams {
  // Returns the index of the selected dimension in the allowlist.
  // Throws StringError if `tuneDimension` is unrecognized.
  // Propagates ConfigParseError if `tuneDimension` is missing.
  int resolveDimension(ConfigParser& cfg);

  void runTests();
}
```

`tuneparams.cpp` includes this header and provides the implementations of both `TuneParams::resolveDimension` and `TuneParams::runTests`. The `tuneDims[]` table stays a file-scope `static` symbol in `tuneparams.cpp` — both functions live in the same translation unit, so they can reference it directly without exposing the table publicly.

**Wiring.** Add `#include "../command/tuneparams.h"` to `cpp/command/runtests.cpp` and call `TuneParams::runTests();` adjacent to the existing `QRSTune::runTests();` call.

**Test cases** (implemented in `tuneparams.cpp` inside `TuneParams::runTests()` using the existing `testAssert` macro from `cpp/tests/testcommon.h`):

- **Positive:** For each of `cpuctExploration`, `fpuReductionMax`, `rootFpuReductionMax`, construct an in-memory `ConfigParser` with `tuneDimension = <name>`, call `resolveDimension`, and assert the returned index matches the expected position in `tuneDims[]`.
- **Negative — missing key:** Construct a `ConfigParser` without `tuneDimension`. Assert that calling `resolveDimension` throws (any exception — the precise type belongs to `ConfigParser` and is not this unit's contract).
- **Negative — unrecognized value:** Construct a `ConfigParser` with `tuneDimension = totallyMadeUpName`. Assert that a `StringError` is thrown and its message contains the offending name and at least one allowed name (e.g. `"cpuctExploration"`). Pinning the exact message is fragile; pinning the substrings is enough.

End-to-end testing of the actual tuning loop under each dim is out of scope — too slow for CI and adds no signal beyond what the lookup test gives.

## Out of Scope

- The `Out of Scope` items listed under Non-Goals above.
- Any changes to QRSOptimizer, its tests, or the QRS-Tune mathematical formulation.
- Documentation beyond the example cfg's inline comments. (The PR's `README` updates already describe the command in general terms; they do not need per-dim coverage.)

## Deliverables

1. New `cpp/command/tuneparams.h` exposing the `TuneParams` namespace (`resolveDimension`, `runTests`).
2. Modified `cpp/command/tuneparams.cpp` with the 3-entry `tuneDims[]` allowlist, the `resolveDimension` implementation, the `activeDims` plumbing, and the `runTests()` body.
3. Modified `cpp/configs/tune_params_example.cfg` with the new `tuneDimension` key, menu of allowed names, and per-dim range overrides.
4. Modified `cpp/command/runtests.cpp` to invoke `TuneParams::runTests()`.
5. One commit per logical step (header + resolver, allowlist + activeDims plumbing, example cfg, test wiring) for clean review.
