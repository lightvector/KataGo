#ifdef USE_MLX_BACKEND

#include "../neuralnet/mlxwinotuner.h"
#include "../neuralnet/desc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <map>
#include <string>
#include <vector>

#include <sys/sysctl.h>  // sysctlbyname, for detectGpuName()
#include <unistd.h>      // getpid(), for atomic temp-file save

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../core/logger.h"
#include "../core/makedir.h"
#include "../dataio/homedata.h"

#include "mlx/mlx.h"
#include "mlx/fast.h"
#include <chrono>
#include <random>
#include <regex>

using namespace std;

static const int MLX_WINO_TUNER_VERSION = 3;
static const std::string MLX_WINO_TUNEPARAMS_VERSION_LINE =
    "VERSION=" + std::to_string(MLX_WINO_TUNER_VERSION);

// Mirrors OpenCLTuner's readDescKeyValues: parse "KEY=VALUE KEY=VALUE ..." line into a map.
static map<string,int> parseKeyValueLine(const string& fileName, const string& line) {
  map<string,int> kvs;
  vector<string> tokens = Global::split(line);
  for(const string& tok : tokens) {
    size_t eq = tok.find('=');
    if(eq == string::npos)
      throw IOError("MLXWinogradTuneParams: token without '=' in " + fileName + " line: " + line);
    string k = tok.substr(0, eq);
    string v = tok.substr(eq + 1);
    if(k.empty())
      throw IOError("MLXWinogradTuneParams: key-value pair without key in " + fileName + " line: " + line);
    if(v.empty())
      throw IOError("MLXWinogradTuneParams: key-value pair without value for key '" + k + "' in " + fileName + " line: " + line);
    if(kvs.count(k) > 0)
      throw IOError("MLXWinogradTuneParams: duplicate key " + k + " in " + fileName);
    try {
      kvs[k] = Global::stringToInt(v);
    } catch(const StringError&) {
      throw IOError("MLXWinogradTuneParams: could not parse value for key " + k + " in " + fileName);
    }
  }
  return kvs;
}

static int requireKey(const map<string,int>& kvs, const string& key, const string& fileName) {
  auto it = kvs.find(key);
  if(it == kvs.end())
    throw IOError("MLXWinogradTuneParams: missing key " + key + " in " + fileName);
  return it->second;
}

bool MLXWinogradTuneParams::isValid() const {
  if(inputTransform.tg0 <= 0 || inputTransform.tg1 <= 0) return false;
  if(outputUntransform.tg0 <= 0 || outputUntransform.tg1 <= 0) return false;
  if(inputTransform.tg0 * inputTransform.tg1 > 1024) return false;
  if(outputUntransform.tg0 * outputUntransform.tg1 > 1024) return false;
  if(inputTransform.wpt < 1 || outputUntransform.wpt < 1) return false;
  if(inputTransform.vw  < 1) return false;
  // Tfast (GRID_ORDER=1) requires VW=1 in the kernels. Reject any input
  // candidate that violates this — surfaces the constraint earlier than
  // the Metal JIT static_assert. (Output VW is gone; global gridOrder
  // is gone; input gridOrder stands alone.)
  if(inputTransform.gridOrder == MLXWinograd::GridOrder::Tfast
     && inputTransform.vw != 1) return false;
  return true;
}

void MLXWinogradTuneParams::save(const string& filename, const MLXWinogradTuneParams& params) {
  // Write to a per-process-unique temp path, then atomically rename onto the
  // final path. This prevents two katago processes that both cache-miss on the
  // same model and tune concurrently from tearing the shared cache file.
  const string tmpPath = filename + ".tmp." + std::to_string((long)getpid());
  ofstream out;
  FileUtils::open(out, tmpPath);
  out << MLX_WINO_TUNEPARAMS_VERSION_LINE << "\n";
  out << "#inputTransform\n";
  out << "tg0=" << params.inputTransform.tg0
      << " tg1=" << params.inputTransform.tg1
      << " wpt=" << params.inputTransform.wpt
      << " vw="  << params.inputTransform.vw
      << " gridOrder=" << (int)params.inputTransform.gridOrder << "\n";
  out << "#outputUntransform\n";
  out << "tg0=" << params.outputUntransform.tg0
      << " tg1=" << params.outputUntransform.tg1
      << " wpt=" << params.outputUntransform.wpt << "\n";
  out.flush();
  out.close();
  // Atomic publish: only fully-written content ever appears at `filename`.
  FileUtils::rename(tmpPath, filename);
}

MLXWinogradTuneParams MLXWinogradTuneParams::load(const string& filename) {
  vector<string> raw = FileUtils::readFileLines(filename, '\n');
  vector<string> lines;
  for(const string& r : raw) {
    string s = Global::stripComments(r);
    s = Global::trim(s);
    if(!s.empty()) lines.push_back(s);
  }
  if(lines.empty())
    throw IOError("MLXWinogradTuneParams::load: no content in " + filename);
  if(lines[0] != MLX_WINO_TUNEPARAMS_VERSION_LINE)
    throw IOError("MLXWinogradTuneParams::load: expected first line to be "
                  + MLX_WINO_TUNEPARAMS_VERSION_LINE + " in " + filename);
  if(lines.size() != 3)
    throw IOError("MLXWinogradTuneParams::load: expected 3 non-comment lines in " + filename);

  MLXWinogradTuneParams params;
  {
    map<string,int> kvs = parseKeyValueLine(filename, lines[1]);
    params.inputTransform.tg0 = requireKey(kvs, "tg0", filename);
    params.inputTransform.tg1 = requireKey(kvs, "tg1", filename);
    params.inputTransform.wpt = requireKey(kvs, "wpt", filename);
    params.inputTransform.vw  = requireKey(kvs, "vw",  filename);
    params.inputTransform.gridOrder = (MLXWinograd::GridOrder)requireKey(kvs, "gridOrder", filename);
  }
  {
    map<string,int> kvs = parseKeyValueLine(filename, lines[2]);
    params.outputUntransform.tg0 = requireKey(kvs, "tg0", filename);
    params.outputUntransform.tg1 = requireKey(kvs, "tg1", filename);
    params.outputUntransform.wpt = requireKey(kvs, "wpt", filename);
  }
  return params;
}

string MLXWinogradTuner::defaultDirectory(bool makeDir, const string& homeDataDirOverride) {
  string dir = HomeData::getHomeDataDir(makeDir, homeDataDirOverride);
  dir += "/mlxwinotuning";
  if(makeDir) MakeDir::make(dir);
  return dir;
}

string MLXWinogradTuner::defaultFileName(const string& gpuName,
                                         int nnXLen, int nnYLen,
                                         int trunkNumChannels, int modelVersion,
                                         bool useFP16) {
  string clean;
  for(char c : gpuName) {
    if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))
      clean += c;
  }
  const char* dtypeSuffix = useFP16 ? "_fp16" : "_fp32";
  return Global::strprintf("tunemlxwino%d_gpu%s_x%d_y%d_c%d_mv%d%s.txt",
                           MLX_WINO_TUNER_VERSION, clean.c_str(),
                           nnXLen, nnYLen, trunkNumChannels, modelVersion,
                           dtypeSuffix);
}

string MLXWinogradTuner::detectGpuName() {
  // The optimal Winograd launch geometry differs across Apple GPU variants, so
  // the cache key must distinguish them; otherwise a cache tuned on one chip
  // (e.g. M1) would be loaded verbatim on another (e.g. M4 Max). MLX does not
  // reliably export a device name (mlx::core::metal::device_info() is declared
  // but not exported in all libmlx builds), so query the chip brand string
  // directly. On Apple Silicon this returns e.g. "Apple M3 Max";
  // defaultFileName() sanitizes it to [A-Za-z0-9].
  char buf[128];
  size_t len = sizeof(buf);
  if(sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0 && len > 1) {
    buf[sizeof(buf) - 1] = '\0';  // guarantee NUL-termination
    string name(buf);             // stops at the first NUL
    if(!name.empty())
      return name;
  }
  return "AppleSilicon";
}

namespace mx = mlx::core;

namespace {

// One stage-1 (input transform) timed run on a synthetic [N,H,W,C] tensor.
// Mirrors the inner-loop shape of winogradConv2d's stage 1, but issues only
// the input-transform kernel so we can score it in isolation. Returns wall ms.
// Input kernel always writes Std layout (matmulOrient axis is gone).
static double timeOneInputTransform(
    const MLXWinograd::InputTransform& cfg,
    const mx::array& input, int channels,
    bool useFP16, bool doWarmup) {
  int N = input.shape(0);
  int H = input.shape(1);
  int W = input.shape(2);
  int tilesY = (H + 1) / 2;
  int tilesX = (W + 1) / 2;
  int Ntiles = N * tilesY * tilesX;

  const mx::Dtype dtype = useFP16 ? mx::float16 : mx::float32;

  // Kernel name encodes the still-live axes so the Metal JIT cache sees a
  // unique entry per (dtype, wpt, vw, gridOrder) combination.
  std::string kernelName =
      std::string(useFP16 ? "wino_input_transform_f16" : "wino_input_transform_f32")
      + "_w" + std::to_string(cfg.wpt)
      + "_v" + std::to_string(cfg.vw)
      + "_g" + std::to_string((int)cfg.gridOrder)
      + "_tune";

  auto fn = mx::fast::metal_kernel(
      kernelName.c_str(),
      /*input_names=*/{"inp"},
      /*output_names=*/{"outp"},
      /*source=*/MLXWinograd::kWinoInputSource);

  // Output shape: [16, Ntiles, C] (Std only).
  mx::Shape outShape = {16, Ntiles, channels};

  // Grid depends on gridOrder: Cfast → (ceil(C/vw), ceil(Ntiles/wpt), 1),
  //                             Tfast → (Ntiles, ceil(C/wpt), 1).
  int gridX = (cfg.gridOrder == MLXWinograd::GridOrder::Cfast)
      ? ((channels + cfg.vw - 1) / cfg.vw)
      : Ntiles;
  int gridY = (cfg.gridOrder == MLXWinograd::GridOrder::Cfast)
      ? ((Ntiles + cfg.wpt - 1) / cfg.wpt)
      : ((channels + cfg.wpt - 1) / cfg.wpt);

  std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmplArgs = {
    {"T",             dtype},
    {"WPT",           cfg.wpt},
    {"VW",            cfg.vw},
    {"GRID_ORDER",    (int)cfg.gridOrder}
  };

  // Untimed warmup (gated to the first measured rep per shape): hots
  // pipeline-state + lazy-graph caches for THIS config before the timed eval.
  // Caller gates so we don't re-warm on every rep.
  if(doWarmup) {
    auto warmOuts = fn(
        /*inputs=*/{input},
        /*output_shapes=*/{ outShape },
        /*output_dtypes=*/{ dtype },
        /*grid=*/std::make_tuple(gridX, gridY, 1),
        /*threadgroup=*/std::make_tuple(cfg.tg0, cfg.tg1, 1),
        /*template_args=*/tmplArgs,
        /*init_value=*/std::nullopt,
        /*verbose=*/false,
        /*stream=*/mx::StreamOrDevice{});
    mx::eval(warmOuts[0]);
  }

  // Timed pass — build fresh lazy node and eval it.
  auto outs = fn(
      /*inputs=*/{input},
      /*output_shapes=*/{ outShape },
      /*output_dtypes=*/{ dtype },
      /*grid=*/std::make_tuple(gridX, gridY, 1),
      /*threadgroup=*/std::make_tuple(cfg.tg0, cfg.tg1, 1),
      /*template_args=*/tmplArgs,
      /*init_value=*/std::nullopt,
      /*verbose=*/false,
      /*stream=*/mx::StreamOrDevice{});
  auto t0 = std::chrono::steady_clock::now();
  mx::eval(outs[0]);
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Same shape for output untransform: synthetic [16, Ntiles, outC] -> [N,H,W,outC].
// m is always Std-layout ([16, Ntiles, outC]).
static double timeOneOutputUntransform(
    const MLXWinograd::OutputUntransform& cfg,
    const mx::array& m, int N, int H, int W, int outC,
    bool useFP16, bool doWarmup) {
  int tilesY = (H + 1) / 2;
  int tilesX = (W + 1) / 2;
  int Ntiles = N * tilesY * tilesX;

  int nhwc_arr[4] = {N, H, W, outC};
  mx::array nhwcArr(nhwc_arr, {4}, mx::int32);

  const mx::Dtype dtype = useFP16 ? mx::float16 : mx::float32;

  // Kernel name encodes the still-live axes so the Metal JIT cache sees a
  // unique entry per (dtype, wpt) combination. (Output kernel is VW=1
  // monomorphic, Cfast monomorphic, and Std-only.)
  std::string kernelName =
      std::string(useFP16 ? "wino_output_untransform_f16" : "wino_output_untransform_f32")
      + "_w" + std::to_string(cfg.wpt)
      + "_tune";

  auto fn = mx::fast::metal_kernel(
      kernelName.c_str(),
      /*input_names=*/{"m", "nhwc"},
      /*output_names=*/{"outp"},
      /*source=*/MLXWinograd::kWinoOutputSource);

  // Cfast-only grid: (outC, ceil(Ntiles/wpt), 1).
  int gridX = outC;
  int gridY = (Ntiles + cfg.wpt - 1) / cfg.wpt;

  std::vector<std::pair<std::string, mx::fast::TemplateArg>> tmplArgs = {
    {"T",             dtype},
    {"WPT",           cfg.wpt}
  };

  // Untimed warmup (gated to the first measured rep per shape): hots
  // pipeline-state + lazy-graph caches for THIS config before the timed eval.
  // Caller gates so we don't re-warm on every rep.
  if(doWarmup) {
    auto warmOuts = fn(
        /*inputs=*/{m, nhwcArr},
        /*output_shapes=*/{ mx::Shape{N, H, W, outC} },
        /*output_dtypes=*/{ dtype },
        /*grid=*/std::make_tuple(gridX, gridY, 1),
        /*threadgroup=*/std::make_tuple(cfg.tg0, cfg.tg1, 1),
        /*template_args=*/tmplArgs,
        /*init_value=*/std::nullopt,
        /*verbose=*/false,
        /*stream=*/mx::StreamOrDevice{});
    mx::eval(warmOuts[0]);
  }

  // Timed pass — build fresh lazy node and eval it.
  auto outs = fn(
      /*inputs=*/{m, nhwcArr},
      /*output_shapes=*/{ mx::Shape{N, H, W, outC} },
      /*output_dtypes=*/{ dtype },
      /*grid=*/std::make_tuple(gridX, gridY, 1),
      /*threadgroup=*/std::make_tuple(cfg.tg0, cfg.tg1, 1),
      /*template_args=*/tmplArgs,
      /*init_value=*/std::nullopt,
      /*verbose=*/false,
      /*stream=*/mx::StreamOrDevice{});
  auto t0 = std::chrono::steady_clock::now();
  mx::eval(outs[0]);
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Random NHWC input tensor for the input-transform timing harness.
// When useFP16, astype the fp32 source to fp16 so the timed kernel measures
// the active precision.
static mx::array makeRandomInput(int N, int H, int W, int C, uint32_t seed, bool useFP16) {
  std::vector<float> v((size_t)N * H * W * C);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for(auto& x : v) x = dist(rng);
  mx::array arr(v.data(), {N, H, W, C}, mx::float32);
  if(useFP16) return mx::astype(arr, mx::float16);
  return arr;
}

// Random [16, Ntiles, outC] tensor for the output-untransform timing harness.
// When useFP16, astype the fp32 source to fp16 so the timed kernel measures
// the active precision.
static mx::array makeRandomMatmulOut(int Ntiles, int outC, uint32_t seed, bool useFP16) {
  std::vector<float> v((size_t)16 * Ntiles * outC);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for(auto& x : v) x = dist(rng);
  mx::array arr(v.data(), {16, Ntiles, outC}, mx::float32);
  if(useFP16) return mx::astype(arr, mx::float16);
  return arr;
}

// Forward decl: planShapeRotation is defined further down in this anonymous
// namespace alongside its policy constants, but the scoring functions above
// reference it. Pure function; safe to forward-declare.
static std::vector<MLXWinogradTuner::ShapePlan>
planShapeRotation(const std::vector<std::pair<int,int>>& histogram);

// Score one input-transform candidate. Adaptive rotation over the model's
// actual 3x3 conv input-channel distribution: planShapeRotation produces a
// list of (channels, measureReps, weight) entries; per shape we time
// `measureReps` reps and take the median, weighted into the final score by
// `weight`. Each shape warms once on its first measured rep (gated via the
// doWarmup arg to timeOneInputTransform); subsequent reps skip the warmup.
static double scoreInputTransform(const MLXWinograd::InputTransform& cfg,
                                  int N, int H, int W,
                                  const MLXWinogradTuner::ModelInfoForTuning& mi,
                                  bool useFP16) {
  auto plan = planShapeRotation(mi.conv3x3InputHistogram);
  assert(!plan.empty());

  // Pre-build one random input array per planned shape. Each shape warms once
  // on its first measured rep (gated via doWarmup), so no separate warmup pass.
  std::vector<mx::array> inputs;
  inputs.reserve(plan.size());
  uint32_t seed = 0xA1A1A1A1u;
  for(const auto& sp : plan) {
    inputs.push_back(makeRandomInput(N, H, W, sp.channels, seed, useFP16));
    mx::eval(inputs.back());
    seed = seed * 1664525u + 1013904223u;  // distinct seed per shape
  }

  double score = 0.0;
  for(size_t i = 0; i < plan.size(); i++) {
    std::vector<double> samples;
    samples.reserve(plan[i].measureReps);
    for(int r = 0; r < plan[i].measureReps; r++) {
      double ms = timeOneInputTransform(cfg, inputs[i], plan[i].channels, useFP16, /*doWarmup=*/(r == 0));
      samples.push_back(ms);
    }
    // Median (upper of two middles for even sizes; identical to nth_element
    // at index size/2).
    std::nth_element(samples.begin(),
                     samples.begin() + samples.size() / 2,
                     samples.end());
    double median = samples[samples.size() / 2];
    // A non-finite measurement (nan/inf from a failed/pathological kernel run)
    // must NEVER win selection. The tuner minimizes time, so map it to +inf so
    // this candidate loses every comparison rather than mapping to 0 (best).
    if(!std::isfinite(median)) median = std::numeric_limits<double>::infinity();
    score += plan[i].weight * median;
  }
  return score;
}

// Score one output-untransform candidate. Symmetric to scoreInputTransform:
// adaptive rotation over the model's 3x3 conv output-channel distribution.
static double scoreOutputUntransform(const MLXWinograd::OutputUntransform& cfg,
                                     int N, int H, int W,
                                     const MLXWinogradTuner::ModelInfoForTuning& mi,
                                     bool useFP16) {
  int tilesY = (H + 1) / 2;
  int tilesX = (W + 1) / 2;
  int Ntiles = N * tilesY * tilesX;

  auto plan = planShapeRotation(mi.conv3x3OutputHistogram);
  assert(!plan.empty());

  std::vector<mx::array> matmulOuts;
  matmulOuts.reserve(plan.size());
  uint32_t seed = 0xD4D4D4D4u;
  for(const auto& sp : plan) {
    matmulOuts.push_back(makeRandomMatmulOut(Ntiles, sp.channels, seed, useFP16));
    mx::eval(matmulOuts.back());
    seed = seed * 1664525u + 1013904223u;
  }

  double score = 0.0;
  for(size_t i = 0; i < plan.size(); i++) {
    std::vector<double> samples;
    samples.reserve(plan[i].measureReps);
    for(int r = 0; r < plan[i].measureReps; r++) {
      double ms = timeOneOutputUntransform(cfg, matmulOuts[i], N, H, W,
                                           plan[i].channels, useFP16, /*doWarmup=*/(r == 0));
      samples.push_back(ms);
    }
    std::nth_element(samples.begin(),
                     samples.begin() + samples.size() / 2,
                     samples.end());
    double median = samples[samples.size() / 2];
    // A non-finite measurement (nan/inf from a failed/pathological kernel run)
    // must NEVER win selection. The tuner minimizes time, so map it to +inf so
    // this candidate loses every comparison rather than mapping to 0 (best).
    if(!std::isfinite(median)) median = std::numeric_limits<double>::infinity();
    score += plan[i].weight * median;
  }
  return score;
}

// Selection-and-allocation policy for the work-weighted shape rotation.
// Pure function. Inputs: list of (channels, occurrence_count) pairs from the
// model's 3x3 conv distribution. Output: vector<ShapePlan> sorted desc by
// weight, with Σ measureReps == 19 and Σ weight ≈ 1.0.
//
// Selection-rule constants:
static constexpr int    kTotalReps         = 20;
static constexpr int    kWarmupReps        = 1;
static constexpr int    kMeasureReps       = kTotalReps - kWarmupReps;  // 19
static constexpr size_t kMaxShapes         = 3;
static constexpr double kWorkFractionFloor = 0.03;
static constexpr int    kRepFloor          = 3;

static std::vector<MLXWinogradTuner::ShapePlan>
planShapeRotation(const std::vector<std::pair<int,int>>& histogram) {
  // Degenerate case: empty histogram is a model-corruption signal we
  // surface, not silently mask.
  assert(!histogram.empty());

  // Step 1: compute work = count * channels; sort desc by work; take top-K.
  struct Entry { int channels; long long work; };
  std::vector<Entry> entries;
  entries.reserve(histogram.size());
  for(const auto& [c, n] : histogram) {
    if(c <= 0 || n <= 0) continue;
    entries.push_back({c, static_cast<long long>(c) * static_cast<long long>(n)});
  }
  assert(!entries.empty());

  std::sort(entries.begin(), entries.end(),
            [](const Entry& a, const Entry& b) {
              if(a.work != b.work) return a.work > b.work;
              return a.channels > b.channels;  // tie-break: larger C first
            });
  if(entries.size() > kMaxShapes)
    entries.resize(kMaxShapes);

  // Step 2: threshold against post-top-K total work; recompute total.
  long long totalWork = 0;
  for(const auto& e : entries) totalWork += e.work;
  assert(totalWork > 0);
  entries.erase(
      std::remove_if(entries.begin(), entries.end(),
          [totalWork](const Entry& e) {
            return static_cast<double>(e.work) / static_cast<double>(totalWork)
                   < kWorkFractionFloor;
          }),
      entries.end());
  // Dominant survives (it's the largest; if its share < 3% then total<dominant/0.03
  // which is impossible). So entries is non-empty.
  assert(!entries.empty());

  totalWork = 0;
  for(const auto& e : entries) totalWork += e.work;

  // Step 3: normalize weights.
  std::vector<MLXWinogradTuner::ShapePlan> plan;
  plan.reserve(entries.size());
  for(const auto& e : entries) {
    MLXWinogradTuner::ShapePlan sp;
    sp.channels = e.channels;
    sp.weight = static_cast<double>(e.work) / static_cast<double>(totalWork);
    sp.measureReps = 0;  // assigned below
    plan.push_back(sp);
  }

  // Step 4: allocate kMeasureReps with floor.
  if(plan.size() == 1) {
    plan[0].measureReps = kMeasureReps;
    return plan;
  }

  // Tentative round-to-nearest allocation.
  for(auto& sp : plan) {
    sp.measureReps = static_cast<int>(std::lround(sp.weight * kMeasureReps));
  }

  // Floor-bump: any minor shape below kRepFloor gets bumped, deficit out of dominant.
  for(size_t i = 1; i < plan.size(); i++) {
    if(plan[i].measureReps < kRepFloor) {
      int deficit = kRepFloor - plan[i].measureReps;
      plan[i].measureReps += deficit;
      plan[0].measureReps -= deficit;
    }
  }

  // Rounding repair: dominant absorbs +/-1 so Σ == kMeasureReps.
  int sum = 0;
  for(const auto& sp : plan) sum += sp.measureReps;
  plan[0].measureReps += (kMeasureReps - sum);

  // Final invariants. The dominant-underflow assert here will fire only for
  // numShapes > 6 (3*kRepFloor + 1 > kMeasureReps), which is unreachable
  // given kMaxShapes = 3.
  assert(plan[0].measureReps >= kRepFloor);
#ifndef NDEBUG
  int finalSum = 0;
  for(const auto& sp : plan) finalSum += sp.measureReps;
  assert(finalSum == kMeasureReps);
#endif

  return plan;
}

// Per-shape median timing for diagnostic logging. Same rotation/plan as the
// scoring functions; reports one (channels, median_ms) entry per planned
// shape instead of a single weighted score. Used by the flat-sweep log's
// "shape_ms=" field and the gated per-shape consistency test.

static std::vector<std::pair<int,double>>
scoreInputTransformPerShape(const MLXWinograd::InputTransform& cfg,
                            int N, int H, int W,
                            const MLXWinogradTuner::ModelInfoForTuning& mi,
                            bool useFP16) {
  auto plan = planShapeRotation(mi.conv3x3InputHistogram);
  assert(!plan.empty());

  std::vector<mx::array> inputs;
  inputs.reserve(plan.size());
  uint32_t seed = 0xA1A1A1A1u;
  for(const auto& sp : plan) {
    inputs.push_back(makeRandomInput(N, H, W, sp.channels, seed, useFP16));
    mx::eval(inputs.back());
    seed = seed * 1664525u + 1013904223u;
  }

  std::vector<std::pair<int,double>> out;
  out.reserve(plan.size());
  for(size_t i = 0; i < plan.size(); i++) {
    std::vector<double> samples;
    samples.reserve(plan[i].measureReps);
    for(int r = 0; r < plan[i].measureReps; r++) {
      samples.push_back(
          timeOneInputTransform(cfg, inputs[i], plan[i].channels, useFP16, /*doWarmup=*/(r == 0)));
    }
    std::nth_element(samples.begin(),
                     samples.begin() + samples.size() / 2,
                     samples.end());
    double median = samples[samples.size() / 2];
    if(!std::isfinite(median)) median = 0.0;
    out.emplace_back(plan[i].channels, median);
  }
  return out;
}

static std::vector<std::pair<int,double>>
scoreOutputUntransformPerShape(const MLXWinograd::OutputUntransform& cfg,
                               int N, int H, int W,
                               const MLXWinogradTuner::ModelInfoForTuning& mi,
                               bool useFP16) {
  int Ntiles = N * ((H + 1) / 2) * ((W + 1) / 2);

  auto plan = planShapeRotation(mi.conv3x3OutputHistogram);
  assert(!plan.empty());

  std::vector<mx::array> matmulOuts;
  matmulOuts.reserve(plan.size());
  uint32_t seed = 0xD4D4D4D4u;
  for(const auto& sp : plan) {
    matmulOuts.push_back(makeRandomMatmulOut(Ntiles, sp.channels, seed, useFP16));
    mx::eval(matmulOuts.back());
    seed = seed * 1664525u + 1013904223u;
  }

  std::vector<std::pair<int,double>> out;
  out.reserve(plan.size());
  for(size_t i = 0; i < plan.size(); i++) {
    std::vector<double> samples;
    samples.reserve(plan[i].measureReps);
    for(int r = 0; r < plan[i].measureReps; r++) {
      samples.push_back(
          timeOneOutputUntransform(cfg, matmulOuts[i], N, H, W,
                                   plan[i].channels, useFP16, /*doWarmup=*/(r == 0)));
    }
    std::nth_element(samples.begin(),
                     samples.begin() + samples.size() / 2,
                     samples.end());
    double median = samples[samples.size() / 2];
    if(!std::isfinite(median)) median = 0.0;
    out.emplace_back(plan[i].channels, median);
  }
  return out;
}

// Candidate axis value sets, in two breadths that mirror OpenCL's tuner:
//
//   full=false (default; the model-load AUTO-tune): a COARSE grid of a few
//     representative threadgroup / work-per-thread points. Measured on this
//     hardware, the winning configs form a broad plateau — many configs land
//     within ~7% of each other and ~25-40% above the baked default — and
//     geometry moves end-to-end throughput <=1.5%. So a coarse sweep finds the
//     plateau in ~2s instead of the wide grid's ~16s, which otherwise burns
//     that time discriminating between near-equivalent (and run-to-run noisy)
//     configs.
//
//   full=true (a deliberate command tune via `./katago tuner -full`):
//     the wide grid, for operators who want to squeeze the plateau. Both
//     backends pin full=false at model load (openclbackend.cpp /
//     mlxbackend.cpp) and reach the wide grid only through the explicit
//     tuner command.
static const std::vector<int>& inputTg0Values(bool full) {
  static const std::vector<int> vFull   = {1,2,4,8,16,24,32,48,64,96,128,160,192,256,384,512,1024};
  static const std::vector<int> vCoarse = {8,16,32,64,128};
  return full ? vFull : vCoarse;
}
static const std::vector<int>& inputTg1Values(bool full) {
  static const std::vector<int> vFull   = {1,2,4,5,8,10,16,20,25,32,40,50,64,100,128};
  static const std::vector<int> vCoarse = {1,2,4,8,16};
  return full ? vFull : vCoarse;
}
static const std::vector<int>& outputTg0Values(bool full) {
  // Mirror input set — treat tg0 symmetrically.
  static const std::vector<int> vFull   = {1,2,4,8,16,24,32,48,64,96,128,160,192,256,384,512,1024};
  static const std::vector<int> vCoarse = {8,16,32,64,128};
  return full ? vFull : vCoarse;
}
static const std::vector<int>& outputTg1Values(bool full) {
  static const std::vector<int> vFull   = {1,2,4,5,8,10,16,20,25,32,40,50,64,100,128};
  static const std::vector<int> vCoarse = {1,2,4,8,16};
  return full ? vFull : vCoarse;
}

// wptValues() is used by both stages; vwValues() is input-only
// (output kernel is VW=1 monomorphic). wpt narrows under the coarse auto grid
// too — the wpt=8 tail rarely wins for these tiny transform kernels. vw has
// only three values, so coarse == full there.
static const std::vector<int>& wptValues(bool full) {
  static const std::vector<int> vFull   = {1, 2, 4, 8};
  static const std::vector<int> vCoarse = {1, 2, 4};
  return full ? vFull : vCoarse;
}
static const std::vector<int>& vwValues() {
  static const std::vector<int> v = {1, 2, 4};
  return v;
}

// Returns true iff (tg0, tg1, wpt, vw, gridOrder) is structurally valid
// AND vw divides the fast-axis dim of the current stage shape.
static bool isInputCandidateValid(int tg0, int tg1, int wpt, int vw,
                                  MLXWinograd::GridOrder go,
                                  int C, int /*Ntiles*/) {
  if(tg0 <= 0 || tg1 <= 0 || wpt <= 0 || vw <= 0) return false;
  if(tg0 * tg1 > 1024) return false;
  if(go == MLXWinograd::GridOrder::Cfast) {
    if(vw > 1 && (C % vw) != 0) return false;
  } else {
    // Tfast: vw must be 1 (kernel static_assert enforces this).
    if(vw != 1) return false;
  }
  return true;
}
// Output kernel is VW=1 monomorphic — no vw parameter, no
// vw-divisibility check on outC. Output kernel is also Cfast monomorphic
// — no gridOrder parameter.
static bool isOutputCandidateValid(int tg0, int tg1, int wpt,
                                   int /*outC*/, int /*Ntiles*/) {
  if(tg0 <= 0 || tg1 <= 0 || wpt <= 0) return false;
  if(tg0 * tg1 > 1024) return false;
  return true;
}

static std::vector<MLXWinograd::InputTransform>
buildInputCandidates(bool full, int C, int Ntiles, MLXWinograd::GridOrder go) {
  std::vector<MLXWinograd::InputTransform> out;
  for(int tg0 : inputTg0Values(full))
  for(int tg1 : inputTg1Values(full))
  for(int wpt : wptValues(full))
  for(int vw  : vwValues()) {
    if(!isInputCandidateValid(tg0, tg1, wpt, vw, go, C, Ntiles)) continue;
    out.push_back({tg0, tg1, wpt, vw, go});
  }
  return out;
}
static std::vector<MLXWinograd::OutputUntransform>
buildOutputCandidates(bool full, int outC, int Ntiles) {
  std::vector<MLXWinograd::OutputUntransform> out;
  for(int tg0 : outputTg0Values(full))
  for(int tg1 : outputTg1Values(full))
  for(int wpt : wptValues(full)) {
    if(!isOutputCandidateValid(tg0, tg1, wpt, outC, Ntiles)) continue;
    out.push_back({tg0, tg1, wpt});
  }
  return out;
}

// Flat sweep over (tg0, tg1, wpt, vw, gridOrder) for the input transform.
// Returns the best (lowest-time)
// candidate that passes isInputCandidateValid; nullopt if no candidate is
// valid (defensive -- should not happen for a real model).
static std::optional<MLXWinograd::InputTransform>
flatSweepInput(int N, int H, int W,
               const MLXWinogradTuner::ModelInfoForTuning& mi,
               bool useFP16, bool full, Logger* logger) {
  using GO = MLXWinograd::GridOrder;
  // Candidate enumeration's vw-divisibility filter uses C as the most
  // restrictive channel count the kernel will encounter. Use the max of the
  // model's actual 3x3 input channel distribution.
  int C = 0;
  for(const auto& p : mi.conv3x3InputHistogram) C = std::max(C, p.first);
  assert(C > 0);
  const int tilesY = (H + 1) / 2;
  const int tilesX = (W + 1) / 2;
  const int Ntiles = N * tilesY * tilesX;

  // Score the baked default (default-constructed = {tg0=32, tg1=1, wpt=1,
  // vw=1, gridOrder=Cfast}) so the sweep log carries a baseline the operator
  // can compare the winner against. Always adopted-winner; no fallback.
  // The defaults satisfy isInputCandidateValid for any (C, Ntiles) because
  // vw=1 divides every channel count; see mlxwinograd.h for the struct defaults.
  const double baselineMs =
      scoreInputTransform(MLXWinograd::InputTransform{}, N, H, W, mi, useFP16);

  // Seed the floor with the baked default so a sweep in which every candidate
  // throws still yields a valid result instead of aborting model load. The
  // default ({tg0=32,...}, 32 threads) always passes isInputCandidateValid and
  // never exceeds maxTotalThreadsPerThreadgroup, so it scores without throwing.
  std::optional<MLXWinograd::InputTransform> best = MLXWinograd::InputTransform{};
  double bestTime = baselineMs;
  int considered = 0;
  int skipped = 0;

  // The output gridOrder check in isValid() is gone (output kernel is
  // Cfast-monomorphic), so the input gridOrder axis can be searched over
  // both Cfast and Tfast. The global gridOrder field is also gone —
  // input gridOrder stands alone, no cross-stage consistency to enforce.
  for(GO go : {GO::Cfast, GO::Tfast}) {
    auto cands = MLXWinogradTuner::buildInputCandidatesForTesting(full, C, Ntiles, go);
    for(const auto& cand : cands) {
      considered++;
      double t;
      try {
        t = scoreInputTransform(cand, N, H, W, mi, useFP16);
      } catch(const std::exception&) {
        // A candidate whose threadgroup exceeds the pipeline's register-pressure-
        // dependent maxTotalThreadsPerThreadgroup (can be < 1024), or that hits a
        // transient GPU error, throws out of mx::eval. Skip it; the seeded default
        // remains the valid floor.
        skipped++;
        continue;
      }
      if(t < bestTime) { bestTime = t; best = cand; }
    }
  }
  if(logger && skipped > 0)
    logger->write("MLX tuner flatSweepInput skipped=" + std::to_string(skipped)
                  + " candidate(s) that failed to score; kept best valid config");
  if(logger) {
    std::string deltaStr;
    std::string perShapeStr;
    if(best && baselineMs >= 1e-9) {
      double deltaPct = (bestTime - baselineMs) / baselineMs * 100.0;
      // %+.1f always emits a sign; the gated log-format test regex relies on
      // this (matches [-+], not [-+]?). Don't drop the + flag.
      deltaStr = Global::strprintf("%+.1f", deltaPct);

      // Per-shape median timing on the winner — diagnostic only; winner
      // selection above used the weighted score from scoreInputTransform.
      auto perShape = scoreInputTransformPerShape(*best, N, H, W, mi, useFP16);
      perShapeStr = " shape_ms=";
      for(size_t i = 0; i < perShape.size(); i++) {
        if(i > 0) perShapeStr += ",";
        perShapeStr += "c" + std::to_string(perShape[i].first)
                    + ":" + Global::strprintf("%.3f", perShape[i].second);
      }
    } else {
      deltaStr = "nan";
      // best=none branch: omit per-shape fields (matches existing degenerate
      // log shape).
      perShapeStr = "";
    }
    logger->write("MLX tuner flatSweepInput: considered=" + std::to_string(considered)
                  + (best
                     ? " best=tg0=" + std::to_string(best->tg0)
                       + " tg1=" + std::to_string(best->tg1)
                       + " wpt=" + std::to_string(best->wpt)
                       + " vw="  + std::to_string(best->vw)
                       + " gridOrder=" + std::to_string((int)best->gridOrder)
                       + " time_ms=" + Global::strprintf("%.3f", bestTime)
                     : " best=none")
                  + " baseline_ms=" + Global::strprintf("%.3f", baselineMs)
                  + " delta_pct=" + deltaStr
                  + perShapeStr);
  }
  return best;
}

// Flat sweep over (tg0, tg1, wpt) for the output untransform. Output VW
// and gridOrder are not searched: the kernel is monomorphic on VW=1 and
// Cfast.
static std::optional<MLXWinograd::OutputUntransform>
flatSweepOutput(int N, int H, int W,
                const MLXWinogradTuner::ModelInfoForTuning& mi,
                bool useFP16, bool full, Logger* logger) {
  // Output-untransform candidate enumeration doesn't filter on outC
  // (isOutputCandidateValid ignores it — VW=1 monomorphic), but we still
  // pass a representative value. Use the max of the model's actual 3x3
  // output distribution.
  int outC = 0;
  for(const auto& p : mi.conv3x3OutputHistogram) outC = std::max(outC, p.first);
  assert(outC > 0);
  const int Ntiles = N * ((H + 1) / 2) * ((W + 1) / 2);

  // Score the baked default (default-constructed = {tg0=32, tg1=1, wpt=1})
  // so the sweep log carries a baseline the operator can compare the winner
  // against. Symmetric to flatSweepInput.
  const double baselineMs =
      scoreOutputUntransform(MLXWinograd::OutputUntransform{}, N, H, W, mi, useFP16);

  // Seed the floor with the baked default (see flatSweepInput for rationale).
  std::optional<MLXWinograd::OutputUntransform> best = MLXWinograd::OutputUntransform{};
  double bestTime = baselineMs;
  int considered = 0;
  int skipped = 0;

  // Output kernel is VW=1 monomorphic and Cfast monomorphic, so neither
  // VW nor gridOrder is searched here.
  auto cands = MLXWinogradTuner::buildOutputCandidatesForTesting(full, outC, Ntiles);
  for(auto cand : cands) {
    considered++;
    double t;
    try {
      t = scoreOutputUntransform(cand, N, H, W, mi, useFP16);
    } catch(const std::exception&) {
      skipped++;
      continue;
    }
    if(t < bestTime) { bestTime = t; best = cand; }
  }
  if(logger && skipped > 0)
    logger->write("MLX tuner flatSweepOutput skipped=" + std::to_string(skipped)
                  + " candidate(s) that failed to score; kept best valid config");
  if(logger) {
    std::string deltaStr;
    std::string perShapeStr;
    if(best && baselineMs >= 1e-9) {
      double deltaPct = (bestTime - baselineMs) / baselineMs * 100.0;
      // %+.1f always emits a sign; the gated log-format test regex relies on
      // this (matches [-+], not [-+]?). Don't drop the + flag.
      deltaStr = Global::strprintf("%+.1f", deltaPct);

      auto perShape = scoreOutputUntransformPerShape(*best, N, H, W, mi, useFP16);
      perShapeStr = " shape_ms=";
      for(size_t i = 0; i < perShape.size(); i++) {
        if(i > 0) perShapeStr += ",";
        perShapeStr += "c" + std::to_string(perShape[i].first)
                    + ":" + Global::strprintf("%.3f", perShape[i].second);
      }
    } else {
      deltaStr = "nan";
      perShapeStr = "";
    }
    logger->write("MLX tuner flatSweepOutput: considered=" + std::to_string(considered)
                  + (best
                     ? " best=tg0=" + std::to_string(best->tg0)
                       + " tg1=" + std::to_string(best->tg1)
                       + " wpt=" + std::to_string(best->wpt)
                       + " time_ms=" + Global::strprintf("%.3f", bestTime)
                     : " best=none")
                  + " baseline_ms=" + Global::strprintf("%.3f", baselineMs)
                  + " delta_pct=" + deltaStr
                  + perShapeStr);
  }
  return best;
}

} // namespace

MLXWinogradTuneParams MLXWinogradTuner::loadOrAutoTune(
    string tunerFile,
    const string& homeDataDirOverride,
    const string& gpuName,
    int nnXLen, int nnYLen, int batchSize,
    ModelInfoForTuning modelInfo,
    Logger* logger,
    bool full,
    bool reTune,
    bool useFP16) {
  if(tunerFile.empty()) {
    string dir = defaultDirectory(true, homeDataDirOverride);
    tunerFile = dir + "/" + defaultFileName(gpuName, nnXLen, nnYLen,
                                            modelInfo.trunkNumChannels,
                                            modelInfo.modelVersion, useFP16);
  }

  // Cache load path: if the file exists, validates, and reTune is false, use it.
  if(!reTune && !tunerFile.empty() && FileUtils::exists(tunerFile)) {
    try {
      MLXWinogradTuneParams loaded = MLXWinogradTuneParams::load(tunerFile);
      if(loaded.isValid()) {
        if(logger)
          logger->write("Loaded MLX Winograd tuning parameters from " + tunerFile);
        return loaded;
      }
      if(logger)
        logger->write("MLX Winograd cache " + tunerFile + " failed isValid(); re-tuning");
    } catch(const IOError& e) {
      if(logger)
        logger->write(std::string("MLX Winograd cache load failed: ") + e.what() + "; re-tuning");
    }
  }

  // Flat per-stage sweep.
  auto t0 = std::chrono::steady_clock::now();
  auto bestIn  = flatSweepInput (batchSize, nnYLen, nnXLen, modelInfo, useFP16, full, logger);
  auto bestOut = flatSweepOutput(batchSize, nnYLen, nnXLen, modelInfo, useFP16, full, logger);
  auto t1 = std::chrono::steady_clock::now();
  double tuneMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  if(logger)
    logger->write("MLX tuner flat sweep complete in " + Global::strprintf("%.0f", tuneMs) + " ms");

  if(!bestIn || !bestOut)
    throw StringError("MLXWinogradTuner: flat sweep returned no valid candidate");

  MLXWinogradTuneParams result;
  result.inputTransform    = *bestIn;
  result.outputUntransform = *bestOut;
  // Global gridOrder is deleted; input gridOrder stands alone.

  if(!result.isValid())
    throw StringError("MLXWinogradTuner: flat sweep result failed isValid()");

  if(!tunerFile.empty()) {
    MLXWinogradTuneParams::save(tunerFile, result);
    if(logger)
      logger->write("Saved MLX Winograd tuning parameters to " + tunerFile);
  }
  return result;
}

std::vector<MLXWinograd::InputTransform>
MLXWinogradTuner::buildInputCandidatesForTesting(bool full, int C, int Ntiles, MLXWinograd::GridOrder go) {
  return buildInputCandidates(full, C, Ntiles, go);
}
std::vector<MLXWinograd::OutputUntransform>
MLXWinogradTuner::buildOutputCandidatesForTesting(bool full, int outC, int Ntiles) {
  return buildOutputCandidates(full, outC, Ntiles);
}

std::vector<MLXWinogradTuner::ShapePlan>
MLXWinogradTuner::planShapeRotationForTesting(
    const std::vector<std::pair<int,int>>& histogram) {
  return planShapeRotation(histogram);
}

double MLXWinogradTuner::scoreInputTransformForTesting(
    const MLXWinograd::InputTransform& cfg,
    int N, int H, int W,
    const ModelInfoForTuning& mi,
    bool useFP16) {
  return scoreInputTransform(cfg, N, H, W, mi, useFP16);
}

double MLXWinogradTuner::scoreOutputUntransformForTesting(
    const MLXWinograd::OutputUntransform& cfg,
    int N, int H, int W,
    const ModelInfoForTuning& mi,
    bool useFP16) {
  return scoreOutputUntransform(cfg, N, H, W, mi, useFP16);
}

std::vector<std::pair<int,double>>
MLXWinogradTuner::scoreInputTransformPerShapeForTesting(
    const MLXWinograd::InputTransform& cfg,
    int N, int H, int W,
    const ModelInfoForTuning& mi,
    bool useFP16) {
  return scoreInputTransformPerShape(cfg, N, H, W, mi, useFP16);
}

std::vector<std::pair<int,double>>
MLXWinogradTuner::scoreOutputUntransformPerShapeForTesting(
    const MLXWinograd::OutputUntransform& cfg,
    int N, int H, int W,
    const ModelInfoForTuning& mi,
    bool useFP16) {
  return scoreOutputUntransformPerShape(cfg, N, H, W, mi, useFP16);
}

std::string MLXWinogradTuner::formatConv3x3DistributionLine(
    int total,
    const std::map<int,int>& inputChannelCounts,
    const std::map<int,int>& outputChannelCounts) {
  // Build a deterministic ordering: pairs sorted descending by invocation
  // count, ties broken by channel count descending. Truncate each histogram
  // to top-10 with a trailing ",..." guard for pathological models.
  auto serialize = [](const std::map<int,int>& counts) -> std::string {
    if(counts.empty()) return "{}";
    std::vector<std::pair<int,int>> pairs(counts.begin(), counts.end());
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
                if(a.second != b.second) return a.second > b.second;
                return a.first > b.first;
              });
    constexpr size_t kMax = 10;
    bool truncated = pairs.size() > kMax;
    if(truncated) pairs.resize(kMax);

    std::string s;
    for(size_t i = 0; i < pairs.size(); i++) {
      if(i > 0) s += ",";
      s += std::to_string(pairs[i].first) + ":" + std::to_string(pairs[i].second);
    }
    if(truncated) s += ",...";
    return s;
  };

  return "MLX tuner conv3x3 distribution: total=" + std::to_string(total)
       + " input_c="  + serialize(inputChannelCounts)
       + " output_c=" + serialize(outputChannelCounts);
}

// Pure core: filter to 3x3 convs and emit (channels, count) histograms.
// Decoupled from ModelDesc so it's testable without synthesizing the
// copy-deleted ModelDesc hierarchy. Takes pointers because ConvLayerDesc
// has a deleted copy ctor; pointers must be non-null and outlive the call.
static std::pair<std::vector<std::pair<int,int>>,
                 std::vector<std::pair<int,int>>>
buildConv3x3HistogramsFromConvs(const std::vector<const ConvLayerDesc*>& convs) {
  std::map<int,int> inputC, outputC;
  for(const ConvLayerDesc* c : convs) {
    if(c->convXSize == 3 && c->convYSize == 3) {
      inputC[c->inChannels]++;
      outputC[c->outChannels]++;
    }
  }
  std::vector<std::pair<int,int>> inVec(inputC.begin(), inputC.end());
  std::vector<std::pair<int,int>> outVec(outputC.begin(), outputC.end());
  return {std::move(inVec), std::move(outVec)};
}

std::pair<std::vector<std::pair<int,int>>,
          std::vector<std::pair<int,int>>>
MLXWinogradTuner::buildConv3x3HistogramsFromConvsForTesting(
    const std::vector<const ConvLayerDesc*>& convs) {
  return buildConv3x3HistogramsFromConvs(convs);
}

// ModelDesc shim. Walks iterConvLayers, collects pointers to the
// descriptors owned by modelDesc, and delegates to the pure core. Used
// by mlxbackend.cpp at model load. The returned histograms reference no
// memory from modelDesc — only ints — so the descriptor lifetime
// requirement is local to this call.
std::pair<std::vector<std::pair<int,int>>,
          std::vector<std::pair<int,int>>>
MLXWinogradTuner::buildConv3x3Histograms(const ModelDesc& modelDesc) {
  std::vector<const ConvLayerDesc*> convs;
  modelDesc.iterConvLayers([&](const ConvLayerDesc& c) { convs.push_back(&c); });
  return buildConv3x3HistogramsFromConvs(convs);
}

std::string MLXWinogradTuner::formatConv3x3Distribution(const ModelDesc& modelDesc) {
  // Convenience wrapper for callers that want the formatted line directly
  // from a ModelDesc. The histogram is built here and (separately) again by
  // mlxbackend.cpp for the tuner's ModelInfoForTuning — two walks per model
  // load. This is acceptable because model load happens once per process;
  // a single-walk refactor would tangle the mlxbackend call site without
  // measurable savings.
  auto [inVec, outVec] = MLXWinogradTuner::buildConv3x3Histograms(modelDesc);
  std::map<int,int> inMap(inVec.begin(), inVec.end());
  std::map<int,int> outMap(outVec.begin(), outVec.end());
  int total = 0;
  for(const auto& kv : outVec) total += kv.second;  // total = #3x3 convs
  return formatConv3x3DistributionLine(total, inMap, outMap);
}

#endif // USE_MLX_BACKEND
