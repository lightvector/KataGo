#ifndef NEURALNET_GREEDYSEARCH_H_
#define NEURALNET_GREEDYSEARCH_H_

// Pure, header-only greedy coordinate descent over discrete axes. No MLX/Metal
// dependency, so it is unit-tested standalone. Axes are index-based: each axis
// has a fixed number of candidate value-indices [0, size); the caller maps an
// index assignment to a concrete config inside its score callback. Lower score
// is better; the callback returns +inf for an invalid assignment.

#include <cassert>
#include <functional>
#include <limits>
#include <vector>

namespace GreedySearch {

struct Result {
  std::vector<int> indices;  // best value-index per axis
  double score;              // its score
  int evaluated;             // number of scoreFn calls (instrumentation/tests)
};

// axisSizes[a]  = number of candidate values for axis a.
// order         = axis indices, highest-sensitivity first (a permutation of [0,nAxes)).
// seedIndices   = starting index per axis; MUST score finite (it is the always-valid floor).
// scoreFn(idx)  = lower is better; return +inf for invalid assignments.
// maxPasses     = pass cap; descent also stops early on a no-change pass.
inline Result coordinateDescent(
    const std::vector<int>& axisSizes,
    const std::vector<int>& order,
    const std::vector<int>& seedIndices,
    const std::function<double(const std::vector<int>&)>& scoreFn,
    int maxPasses) {
  const size_t nAxes = axisSizes.size();
  assert(seedIndices.size() == nAxes);
  assert(order.size() == nAxes);

#ifndef NDEBUG
  {
    std::vector<char> seen(nAxes, 0);
    for(int a : order) {
      assert(a >= 0 && (size_t)a < nAxes);
      assert(!seen[a]);
      seen[a] = 1;
    }
  }
#endif

  std::vector<int> best = seedIndices;
  double bestScore = scoreFn(best);
  int evaluated = 1;

  for(int pass = 0; pass < maxPasses; pass++) {
    bool changed = false;
    for(int axis : order) {
      const int curVal = best[axis];
      int bestVal = curVal;
      for(int v = 0; v < axisSizes[axis]; v++) {
        if(v == curVal) continue;  // current value's score is already bestScore
        std::vector<int> trial = best;
        trial[axis] = v;
        const double s = scoreFn(trial);
        evaluated++;
        if(s < bestScore) { bestScore = s; bestVal = v; }
      }
      if(bestVal != curVal) { best[axis] = bestVal; changed = true; }
    }
    if(!changed) break;
  }
  return Result{best, bestScore, evaluated};
}

}  // namespace GreedySearch

#endif  // NEURALNET_GREEDYSEARCH_H_
