#ifndef NEURALNET_MLXWINOTUNER_H_
#define NEURALNET_MLXWINOTUNER_H_

#ifdef USE_MLX_BACKEND

#include <array>
#include <map>
#include <string>
#include <vector>
#include "../neuralnet/mlxwinograd.h"

class Logger;
struct ModelDesc;
struct ConvLayerDesc;

struct MLXWinogradTuneParams {
  MLXWinograd::InputTransform    inputTransform;
  MLXWinograd::OutputUntransform outputUntransform;

  // tg0 * tg1 <= 1024, all positive. Input gridOrder stands alone (no global
  // companion; output kernel is Cfast-monomorphic).
  // vw must divide the fast-axis dim of the current model —
  // that check happens at candidate-enumeration time, not here.
  bool isValid() const;

  // VERSION=3 plain-text persistence. Format:
  //   VERSION=3
  //   #inputTransform
  //   tg0=<int> tg1=<int> wpt=<int> vw=<int> gridOrder=<0|1>
  //   #outputUntransform
  //   tg0=<int> tg1=<int> wpt=<int>
  static void save(const std::string& filename, const MLXWinogradTuneParams& params);
  static MLXWinogradTuneParams load(const std::string& filename);
};

namespace MLXWinogradTuner {
  struct ModelInfoForTuning {
    int trunkNumChannels;   // cache file key
    int modelVersion;       // cache file key
    std::vector<std::pair<int,int>> conv3x3InputHistogram;
    std::vector<std::pair<int,int>> conv3x3OutputHistogram;
  };

  // Per-shape rep allocation produced by planShapeRotation. The tuner loops
  // over a vector<ShapePlan> when scoring a candidate: each entry contributes
  // `weight * median(time over `measureReps` reps at this channel count)` to
  // the total score.
  struct ShapePlan {
    int channels;     // C value to time
    int measureReps;  // number of timing reps (does not include warmup)
    double weight;    // normalized score weight, Σ weights == 1.0
  };

  // Pure, deterministic. Given (channel, count) pairs, returns the planned
  // rotation:
  //   1. work_i = count_i * channels_i; sort desc by work; take top-3.
  //   2. drop shapes with work < 3% of the post-top3 total work; renormalize.
  //   3. weight_i = work_i / total_work after renormalization.
  //   4. allocate 19 measureReps proportionally; bump any below 3 up to 3,
  //      taking the deficit from the dominant shape; repair rounding so the
  //      dominant absorbs the +/-1 to make Σ measureReps == 19 exactly.
  // Asserts on empty input.
  std::vector<ShapePlan> planShapeRotationForTesting(
      const std::vector<std::pair<int,int>>& histogram);

  std::string defaultDirectory(bool makeDir, const std::string& homeDataDirOverride);
  std::string defaultFileName(const std::string& gpuName,
                              int nnXLen, int nnYLen,
                              int trunkNumChannels, int modelVersion,
                              bool useFP16);

  // Loads existing tune file if present and valid; otherwise runs the two
  // grid searches, saves the result, and returns it.
  // useFP16: passed to defaultFileName for cache-file naming AND to the
  // search-timing kernels so geometry is measured at the active precision.
  // seedOverride: reserved for API stability; currently ignored by the flat
  // sweep. Production callers pass nullptr.
  MLXWinogradTuneParams loadOrAutoTune(
    std::string tunerFile,
    const std::string& homeDataDirOverride,
    const std::string& gpuName,
    int nnXLen, int nnYLen, int batchSize,
    ModelInfoForTuning modelInfo,
    Logger* logger,
    bool full,
    bool reTune,
    bool useFP16,
    const MLXWinogradTuneParams* seedOverride = nullptr
  );

  // Test-only — exposes the per-model candidate enumeration. Not part of the
  // stable API; production callers should use loadOrAutoTune.
  std::vector<MLXWinograd::InputTransform>
  buildInputCandidatesForTesting(bool full, int C, int Ntiles, MLXWinograd::GridOrder go);
  std::vector<MLXWinograd::OutputUntransform>
  buildOutputCandidatesForTesting(bool full, int outC, int Ntiles);

  // Test-only — exposes the per-stage scoring primitives so tests can compare
  // configs apples-to-apples without depending on the full tuner measurement path.
  double scoreInputTransformForTesting(const MLXWinograd::InputTransform& cfg,
                                       int N, int H, int W,
                                       const ModelInfoForTuning& mi,
                                       bool useFP16);
  double scoreOutputUntransformForTesting(const MLXWinograd::OutputUntransform& cfg,
                                          int N, int H, int W,
                                          const ModelInfoForTuning& mi,
                                          bool useFP16);

  // Per-shape median timing for diagnostic logging. Same rotation as the
  // scoring functions, but reports median per planned shape instead of a
  // single weighted score. One entry per shape in planShapeRotation's
  // output, in the same order (dominant first). Used by the flat-sweep
  // log "shape_ms=" field and the gated per-shape consistency test.
  std::vector<std::pair<int,double>>
  scoreInputTransformPerShapeForTesting(const MLXWinograd::InputTransform& cfg,
                                        int N, int H, int W,
                                        const ModelInfoForTuning& mi,
                                        bool useFP16);
  std::vector<std::pair<int,double>>
  scoreOutputUntransformPerShapeForTesting(const MLXWinograd::OutputUntransform& cfg,
                                           int N, int H, int W,
                                           const ModelInfoForTuning& mi,
                                           bool useFP16);

  // Conv-3x3 shape distribution log: one-line summary of the model's 3x3
  // conv shape mix, computed at model load and printed alongside the tuner
  // log so operators can correlate cached winners with the per-pass shape
  // distribution the cache was tuned for. Pure formatter is exposed for
  // testability; wrapper does the descriptor walk.
  //
  // formatConv3x3DistributionLine: pure function — given pre-computed
  // histograms keyed by channel count, returns the log line. No I/O.
  std::string formatConv3x3DistributionLine(
      int total,
      const std::map<int,int>& inputChannelCounts,
      const std::map<int,int>& outputChannelCounts);

  // formatConv3x3Distribution: delegates to buildConv3x3Histograms, then
  // rebuilds maps and calls formatConv3x3DistributionLine. Single line;
  // safe to log on every model load.
  std::string formatConv3x3Distribution(const ModelDesc& modelDesc);

  // Pure core of the conv-3x3 histogram build: filters to 3x3, returns
  // (channels, count) vectors for inputs and outputs. Decoupled from
  // ModelDesc so it can be tested without synthesizing the
  // copy-deleted/stream-constructed ModelDesc hierarchy.
  //
  // NOTE on the pointer signature: ConvLayerDesc has a deleted copy ctor
  // (desc.h:29), so we cannot collect them by value. The shim collects
  // pointers to descriptors owned by the ModelDesc; the test constructs
  // descriptors in a local vector via emplace_back and passes pointers.
  // All pointers must be non-null and outlive the call.
  std::pair<std::vector<std::pair<int,int>>,
            std::vector<std::pair<int,int>>>
  buildConv3x3HistogramsFromConvsForTesting(
      const std::vector<const ConvLayerDesc*>& convs);

  // ModelDesc shim. Walks modelDesc.iterConvLayers into a pointer vector
  // and delegates to the pure core above. Used by mlxbackend.cpp at model
  // load.
  std::pair<std::vector<std::pair<int,int>>,
            std::vector<std::pair<int,int>>>
  buildConv3x3Histograms(const ModelDesc& modelDesc);
}

#endif // USE_MLX_BACKEND
#endif // NEURALNET_MLXWINOTUNER_H_
