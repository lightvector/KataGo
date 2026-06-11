// Standalone unit test for the pure greedy coordinate-descent core.
// Build & run (no Xcode needed):
//   clang++ -std=c++20 -I cpp cpp/neuralnet/greedysearch_test.cpp -o /tmp/greedysearch_test && /tmp/greedysearch_test
#include "neuralnet/greedysearch.h"
#include <cassert>
#include <cstdio>
#include <vector>
#include <cmath>

using std::vector;

static int failures = 0;
#define CHECK(cond) do { if(!(cond)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); failures++; } } while(0)

int main() {
  // Axes: 3 axes of sizes 4,4,3. Separable score with a planted optimum at
  // indices (3,0,2): score = |i0-3| + |i1-0| + |i2-2|. Coordinate descent on a
  // separable convex score must reach the exact optimum (score 0).
  {
    vector<int> sizes = {4,4,3};
    vector<int> order = {0,1,2};
    vector<int> seed  = {0,0,0};
    int target0=3, target1=0, target2=2;
    auto score = [&](const vector<int>& idx)->double {
      return std::abs(idx[0]-target0) + std::abs(idx[1]-target1) + std::abs(idx[2]-target2);
    };
    GreedySearch::Result r = GreedySearch::coordinateDescent(sizes, order, seed, score, 3);
    CHECK(r.indices == (vector<int>{3,0,2}));
    CHECK(r.score == 0.0);
    CHECK(r.evaluated >= 1);
  }

  // Invalid combos (score +inf) are never selected and never crash.
  {
    vector<int> sizes = {3,3};
    vector<int> order = {0,1};
    vector<int> seed  = {0,0};
    auto score = [&](const vector<int>& idx)->double {
      if(idx[0]==2 && idx[1]==2) return std::numeric_limits<double>::infinity(); // forbidden
      return (idx[0]==2 ? 0.0 : 1.0) + (idx[1]==2 ? 0.0 : 1.0); // wants (2,2) but it's invalid
    };
    GreedySearch::Result r = GreedySearch::coordinateDescent(sizes, order, seed, score, 3);
    CHECK(!(r.indices[0]==2 && r.indices[1]==2));
    CHECK(std::isfinite(r.score));
  }

  // Deterministic: identical inputs → identical result.
  {
    vector<int> sizes = {4,3,2};
    vector<int> order = {2,0,1};
    vector<int> seed  = {1,1,0};
    auto score = [&](const vector<int>& idx)->double { return (idx[0]-2)*(idx[0]-2) + idx[1] + (1-idx[2]); };
    GreedySearch::Result a = GreedySearch::coordinateDescent(sizes, order, seed, score, 3);
    GreedySearch::Result b = GreedySearch::coordinateDescent(sizes, order, seed, score, 3);
    CHECK(a.indices == b.indices);
    CHECK(a.score == b.score);
  }

  // Constant score → no axis improves → returns the seed; evaluations bounded.
  {
    vector<int> sizes = {4,4,3};
    vector<int> order = {0,1,2};
    vector<int> seed  = {2,3,1};
    auto score = [&](const vector<int>&)->double { return 7.0; };
    GreedySearch::Result r = GreedySearch::coordinateDescent(sizes, order, seed, score, 3);
    CHECK(r.indices == seed);
    // 1 seed eval + one pass of (sizes-1) probes, then a no-change pass stops it.
    CHECK(r.evaluated <= 1 + 3*((4-1)+(4-1)+(3-1)));
  }

  if(failures==0) std::printf("ALL GREEDY TESTS PASSED\n");
  return failures==0 ? 0 : 1;
}
