#ifdef USE_MIGRAPHX_BACKEND

#include <hip/hip_runtime.h>
#include <migraphx/migraphx.hpp>
#include <migraphx/version.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <random>
#include <set>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/sha2.h"
#include "../core/test.h"
#include "../dataio/homedata.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/onnxmodelbuilder.h"

using namespace std;

// AMD ROCm backend for KataGo, built on MIGraphX.
//
// This is the AMD analogue of the TensorRT backend and it deliberately reuses that backend's
// network-construction path: OnnxModelBuilder emits a self-contained ONNX ModelProto (weights
// baked in as initializers) with RAW-head outputs, and MIGraphX parses/compiles it into a GPU
// program. Because the emitted graph is identical to the one TensorRT consumes, the getOutput
// decode below is the same decode the TensorRT backend does, and the two backends agree
// numerically up to precision.
//
// Two MIGraphX specifics drive the design here:
//
// 1. MIGraphX compiles for one static shape, and there is no TensorRT-style optimization profile
//    with a dynamic batch dimension. A single program compiled at maxBatchSize therefore has to
//    zero-pad every short batch, and that padding is expensive: measured on MI325X at 3200 visits,
//    compiling at maxBatchSize instead of near the actual batch size costs 1.68x at 192 threads
//    and 1.53x at 160 (5 interleaved trials per point, sd <= 0.7%). The search's mean batch is
//    ~89 while maxBatchSize is 192, i.e. under half the compute is useful.
//
//    So we compile a small set of BUCKETS and dispatch each eval to the smallest bucket that fits.
//    Three measurements shaped this:
//      - MIGraphX 2.15's dynamic-batch path is unusable: the graph parses with dynamic dims
//        propagated correctly, then the GPU compile aborts in shape.cpp with
//        "lens() called on a dynamic shape". Buckets are the only option available.
//      - Per-slot throughput is FLAT across compiled shapes (4411 vs 4478 inf/s at bs=96 vs 192,
//        a 1.5% difference), so the entire win comes from not computing padding rows and the
//        buckets do not need to be finely spaced.
//      - Batch sizes are not uniformly distributed; MCTS batches cluster near full. Bucket
//        spacing is therefore geometric, which bounds worst-case padding to <2x while keeping
//        the bucket count (and so the compile time and weight memory) small.
//
//    Cost: each compiled program bakes in its own copy of the weights (~100MB FP16 for
//    b18c384nbt). The I/O buffers are NOT duplicated - see the note in ComputeHandle.
//
// 2. Manual device buffers (set_offload_copy(false)). With offload copy MIGraphX would allocate
//    and copy every input and output on each eval; instead we hipMalloc each parameter once and
//    hand MIGraphX raw device pointers, so the steady-state eval does only the H2D copies of the
//    inputs that actually changed and the D2H copies of the outputs.

static void checkHipError(const hipError_t status, const char* opName, const char* file, const char* func, int line) {
  if(status != hipSuccess)
    throw StringError(
      string("HIP Error, for ") + opName + " file " + file + ", func " + func + ", line " + Global::intToString(line) +
      ", error " + hipGetErrorString(status));
}
#define HIP_ERR(opName, x) \
  { checkHipError((x), opName, __FILE__, #x, __LINE__); }

void NeuralNet::globalInitialize() {
  // Nothing to do, MIGraphX and HIP initialize lazily.
}

void NeuralNet::globalCleanup() {
  (void)hipDeviceReset();
}

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  string homeDataDirOverride;
  bool transformerNHWC;  // ONNX emitter: run transformer blocks channel-last
  string dumpDebugModelToDir;
  bool useExhaustiveTune;  // MIGraphX exhaustive_tune: slower compile, faster kernels
  bool useProgramCache;    // persist compiled programs to disk (see createComputeContext)
  bool useBatchBuckets;    // compile a ladder of batch sizes instead of only maxBatchSize
};

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  const LoadedModel* loadedModel,
  ConfigParser& cfg
) {
  (void)gpuIdxs;
  (void)logger;

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->homeDataDirOverride = homeDataDirOverride;
  // Mirrors the TensorRT backend's trtTransformerNHWC, but defaults to FALSE here, unlike
  // TensorRT which defaults it to true.
  //
  // The channel-last trunk produces wrong POLICY output under MIGraphX on transformer models
  // while the value heads stay correct. Measured against the OpenCL backend over KataGo's own
  // runnnonmanyposestest (254 positions), FP32:
  //
  //   model                            NHWC=true          NHWC=false
  //   b7c96h3tfrs-test5-cnorm          policySqErr 136.1  policySqErr 6.0e-10
  //   b7c96h6kv3qk32v16tflrs-fson-bnh  policySqErr 125.4  policySqErr 1.4e-10
  //
  // Every board position on every test position is affected, with the logits collapsing toward
  // a near-flat distribution, so this is a wrong computation rather than a layout permutation.
  // Root cause is still open (it is either MIGraphX's lowering of an op the channel-last path
  // emits, or an emitter assumption that only holds for TensorRT); until that is resolved the
  // safe default is the NCHW trunk, which is correct on every model tested.
  //
  // Convnets never take this path at all: the emitter only goes channel-last when the model
  // actually has transformer blocks.
  context->transformerNHWC =
    (cfg.contains("migraphxTransformerNHWC") ? cfg.getBool("migraphxTransformerNHWC") : false) &&
    NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks();
  context->dumpDebugModelToDir =
    cfg.contains("migraphxDumpDebugModelToDir") ? cfg.getString("migraphxDumpDebugModelToDir") : "";
  // Exhaustive tuning searches more kernel candidates (notably for the trunk convolutions) at
  // compile time. It costs minutes per compile, so it is off unless asked for.
  context->useExhaustiveTune =
    cfg.contains("migraphxExhaustiveTune") ? cfg.getBool("migraphxExhaustiveTune") : false;
  // Persist compiled programs to disk. Bucketing compiles one program per bucket at startup, which
  // costs initialization time rather than throughput, but on a MIGraphX build without rocMLIR every
  // conv and GEMM falls back to MIOpen/rocBLAS JIT and that cost becomes severe. Caching makes it a
  // one-time cost per machine: measured on MI325X, a 38.5s compile reloads in 1.1s.
  //
  // Default OFF because the cache is large - roughly 160MB per bucket, so a full ladder exceeds a
  // gigabyte per (model, board size, precision) combination, and that should not land in a user's
  // home directory unasked.
  context->useProgramCache =
    cfg.contains("migraphxProgramCache") ? cfg.getBool("migraphxProgramCache") : false;
  // Batch bucketing (on by default; see the rationale at the top of this file). Setting this
  // false compiles a single program at maxBatchSize, which is the pre-bucketing behavior — it
  // trades throughput for a shorter startup and one copy of the weights, and gives a way to
  // A/B the feature or fall back if a future MIGraphX regresses on multi-program compiles.
  context->useBatchBuckets =
    cfg.contains("migraphxBatchBuckets") ? cfg.getBool("migraphxBatchBuckets") : true;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
    modelDesc.applyScale8ToReduceActivations();
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

// MIGraphX compilation is not thread-safe against itself in all ROCm versions, and KataGo creates
// one ComputeHandle per server thread, all of which compile the same model at startup. Serialize
// compiles so that N threads do not race inside the compiler.
static mutex compileMutex;

// Bump to invalidate every previously written cache entry, e.g. if the emitted graph or the
// bucketing scheme changes in a way the key below would not otherwise capture.
static constexpr int kProgramCacheSalt = 1;

// Path for one cached compiled program.
//
// The key MUST capture everything that can change the compiled result. Loading a stale entry would
// silently run a program built for different weights and return wrong answers, which is a worse
// outcome than the slow compile this avoids. So it covers:
//   - a hash of the emitted ONNX bytes (weights, board size, layout, scale8, NHWC choice: the
//     emitter bakes all of these in, so the bytes are the authority)
//   - the bucket's batch size, since every bucket is a separately compiled shape
//   - FP16 vs FP32 and exhaustive tuning, both of which change kernel selection
//   - the GPU architecture, since a program built for gfx942 is meaningless on gfx1100
//   - the MIGraphX version, whose serialized format and codegen are not stable across releases
static string programCachePath(
  const string& cacheDir,
  const string& onnxBytes,
  int bucketBatchSize,
  bool willUseFP16,
  bool exhaustiveTune,
  const string& gcnArchName
) {
  char onnxHash[65];
  SHA2::get256(onnxBytes.c_str(), onnxHash);
  string key = Global::strprintf(
    "%s-bs%d-%s-%s-%s-mgx%d%d%d.%s-salt%d",
    string(onnxHash).substr(0, 32).c_str(),
    bucketBatchSize,
    willUseFP16 ? "fp16" : "fp32",
    exhaustiveTune ? "exh" : "std",
    gcnArchName.c_str(),
    MIGRAPHX_VERSION_MAJOR, MIGRAPHX_VERSION_MINOR, MIGRAPHX_VERSION_PATCH,
    MIGRAPHX_VERSION_TWEAK,
    kProgramCacheSalt);
  return cacheDir + "/" + key + ".mxr";
}

// Write to a unique temp path and rename, so neither a crash nor two racing processes can leave a
// half-written .mxr for a later run to load. Mirrors the TensorRT backend's cache write.
// A failure here is never fatal: we already hold the compiled program.
static void saveProgramCache(const migraphx::program& prog, const string& path, Logger* logger) {
  static const uint64_t randBase = std::random_device{}();
  static std::atomic<uint64_t> counter{0};
  string tmpPath = Global::strprintf(
    "%s.tmp_%llx_%llu", path.c_str(),
    (unsigned long long)randBase, (unsigned long long)counter.fetch_add(1));
  try {
    migraphx::file_options fo;
    fo.set_file_format("msgpack");
    migraphx::save(prog, tmpPath.c_str(), fo);
    if(!FileUtils::tryRename(tmpPath, path)) {
      FileUtils::tryRemoveFile(tmpPath);
      if(logger != NULL)
        logger->write("MIGraphX backend: could not rename program cache file, continuing uncached");
    }
  }
  catch(const std::exception& e) {
    FileUtils::tryRemoveFile(tmpPath);
    if(logger != NULL)
      logger->write(string("MIGraphX backend: failed to write program cache (continuing): ") + e.what());
  }
}

struct ComputeHandle {
  ComputeContext* ctx;

  bool usingFP16;
  int maxBatchSize;
  int modelVersion;
  bool hasInputMeta;

  // All work for this handle is ordered on one non-default stream: the input H2D copies, the
  // program itself (via run_async), and the output D2H copies. Using the default stream instead
  // would not order correctly against MIGraphX, which runs on its own internal stream.
  hipStream_t stream;

  // hipGraph capture was prototyped and set aside. This path is GPU-bound, not launch-bound:
  // instrumenting the eval measured 10.294 ms blocked in hipStreamSynchronize against 0.028 ms
  // of host-side output decode per batch (avgRows 60.8), i.e. the host is 0.3% of the time.
  // Collapsing the ~12 driver calls per eval into one graph launch cannot beat that 0.3%.
  //
  // One compiled program per bucket, ascending by batch size. Each carries its own
  // program_parameters because the parameter shapes differ per bucket.
  struct Bucket {
    int batchSize;
    migraphx::program prog;
    migraphx::program_parameters params;
  };
  vector<Bucket> buckets;

  // Device allocations for every program parameter and output, keyed by name. Owned here.
  //
  // Shared across ALL buckets. This is safe and is what keeps bucketing cheap: every buffer is
  // allocated at maxBatchSize, and a bucket compiled for a smaller batch simply uses a prefix of
  // it. MIGraphX is handed a raw device pointer plus the shape it expects, so a bucket of size B
  // reads/writes only the first B rows. Without this, each bucket would duplicate the full I/O
  // working set on top of its weights.
  //
  // The one parameter that is NOT shared is MIGraphX's internal scratch ("main:scratch"), whose
  // size is a property of the compiled program rather than of the batch dimension; each bucket
  // gets its own, keyed by bucket index.
  map<string, void*> buffers;
  map<string, size_t> bufferBytes;
  map<string, size_t> bufferRowElts;
  // Output parameter names, in the order MIGraphX returns them from eval().
  vector<string> outputNames;

  // Smallest bucket that can run batchSize rows. Buckets are ascending, and the last one is
  // always maxBatchSize, so this always finds a home for any batchSize <= maxBatchSize.
  const Bucket& bucketFor(int batchSize) const {
    for(const Bucket& b: buckets) {
      if(b.batchSize >= batchSize)
        return b;
    }
    throw StringError(Global::strprintf(
      "MIGraphX backend: batch size %d exceeds maxBatchSize %d", batchSize, maxBatchSize));
  }

  // Hybrid ladder: geometric (8,16,32,64) below the knee, then linear steps of 32 up to
  // maxBatchSize.
  //
  // Geometric spacing bounds worst-case padding by the RATIO between adjacent buckets, which is
  // the right property for SMALL batches: a batch of 5 padding to 8 wastes a few rows, while
  // uniform spacing would pad it to 32. But a ratio bound is scale-free, and above ~64 a 2x ratio
  // means an absolute gap of 64+ rows, which is where the real waste lives.
  //
  // Two measurements drove this. First, throughput appeared to be governed by fill alone: padded
  // rows/s measured constant at ~2240 across compiled shapes (2194-2266 for bs 64/96/128/192 on
  // b11c768h12nbt3tflrs-fson-silu), so nnEvals/s = paddedRowsPerSec * (avgBatch / bucket).
  //
  // That is too strong, and the counterexample matters for anyone retuning this ladder: the
  // constant was taken on ONE net over shapes >= 64. On b18c384 at 64 threads the per-row rate is
  // 255/254/204 rows/s for shapes 16/24/32 - a ~20% step between 24 and 32 - while fill moves only
  // 0.980 -> 0.971. Fill cannot produce a 20% change from a 0.9% difference, so some compiled
  // shapes really are cheaper per row than others, presumably via convolution algorithm selection.
  // Rung placement is thus an open question, not a solved one: rungs currently sit on round
  // numbers because fit was believed to be all that mattered.
  //
  // Second, a purely geometric ladder halving
  // DOWN from maxBatchSize=192 yields {12,24,48,96,192} - integer division never lands on 128 or
  // 64 - so a batch of ~111 (the measured mean at 192 threads) padded all the way to 192, a fill
  // of 58%. That single gap was this backend's only remaining loss to the ROCm backend.
  //
  // Anchoring the geometric part at fixed powers of two and stepping linearly above the knee
  // keeps the ladder on round shapes and puts a rung near wherever the search's batch actually
  // lands. Cost is bounded: each rung is one more compiled program with its own copy of the
  // weights, so the step is kept coarse (32) rather than tracking the distribution exactly.
  static vector<int> bucketSizesFor(int maxBatchSize, bool useBuckets) {
    vector<int> sizes;
    if(useBuckets) {
      static const int kKnee = 64;   // geometric below, linear above
      static const int kStep = 32;   // linear step; coarse to bound the program count
      for(int b = std::min(maxBatchSize, kKnee); b >= 8; b /= 2)
        sizes.push_back(b);
      for(int b = kKnee + kStep; b < maxBatchSize; b += kStep)
        sizes.push_back(b);
    }
    sizes.push_back(maxBatchSize);
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
    // Never emit a bucket above the cap: getOutput dispatches to the smallest that fits, so the
    // largest rung has to be able to serve maxBatchSize itself.
    while(sizes.size() > 1 && sizes.back() > maxBatchSize)
      sizes.pop_back();
    return sizes;
  }

  ComputeHandle(
    Logger* logger,
    ComputeContext* context,
    const LoadedModel* loadedModel,
    int maxBatchSz,
    bool requireExactNNLen
  ) {
    ctx = context;
    maxBatchSize = maxBatchSz;
    modelVersion = loadedModel->modelDesc.modelVersion;
    hasInputMeta = loadedModel->modelDesc.numInputMetaChannels > 0;

    HIP_ERR("ComputeHandle", hipStreamCreate(&stream));

    const ModelDesc& desc = loadedModel->modelDesc;

    // Emit the same ONNX graph the TensorRT backend builds. Weights are baked in as initializers,
    // so the returned bytes are fully self-contained.
    //
    OnnxModelBuilder::BuildParams buildParams;
    buildParams.nnXLen = ctx->nnXLen;
    buildParams.nnYLen = ctx->nnYLen;
    buildParams.requireExactNNLen = requireExactNNLen;
    buildParams.transformerNHWC = ctx->transformerNHWC;
    // LoadedModel's constructor calls applyScale8ToReduceActivations() unconditionally, so this
    // is always true here. It is only reported to the emitter, never re-applied: the compensation
    // lives in postProcessParams.outputScaleMultiplier, which is recorded already transformed.
    buildParams.scale8Applied = true;
    OnnxModelBuilder::Result onnxResult = OnnxModelBuilder::build(desc, buildParams, logger);
    const string& onnxBytes = onnxResult.serializedModel;

    if(!ctx->dumpDebugModelToDir.empty()) {
      MakeDir::make(ctx->dumpDebugModelToDir);
      string onnxPath = ctx->dumpDebugModelToDir + "/model_" + Global::intToString(ctx->nnXLen) + "x" +
        Global::intToString(ctx->nnYLen) + "_bs" + Global::intToString(maxBatchSize) + ".onnx";
      ofstream dumpOut;
      FileUtils::open(dumpOut, onnxPath, ios::binary);
      dumpOut.write(onnxBytes.data(), (std::streamsize)onnxBytes.size());
      dumpOut.close();
      if(logger != NULL)
        logger->write("MIGraphX backend: dumped emitted ONNX to " + onnxPath);
    }

    // MIGraphX compiles a static shape, so pin the batch dimension of every input. The ONNX
    // emitter declares batch as a dynamic dim; set_input_parameter_shape fixes it. One compile
    // per bucket; see the bucketing rationale at the top of this file.
    const vector<int> bucketSizes = bucketSizesFor(maxBatchSize, ctx->useBatchBuckets);

    // FP16 is decided by config alone, so it is known before any bucket is built and can go into
    // the cache key. The per-bucket code below still sets usingFP16 as it quantizes.
    const bool willUseFP16 =
      (ctx->useFP16Mode == enabled_t::True || ctx->useFP16Mode == enabled_t::Auto);

    // Resolve the cache directory once. Failure to create it disables caching rather than aborting.
    string cacheDir;
    string gcnArchName;
    if(ctx->useProgramCache) {
      try {
        hipDeviceProp_t cacheProp;
        int cacheDev = 0;
        HIP_ERR("ComputeHandle", hipGetDevice(&cacheDev));
        HIP_ERR("ComputeHandle", hipGetDeviceProperties(&cacheProp, cacheDev));
        gcnArchName = string(cacheProp.gcnArchName);
        cacheDir = HomeData::getHomeDataDir(true, ctx->homeDataDirOverride) + "/migraphxcache";
        MakeDir::make(cacheDir);
      }
      catch(const std::exception& e) {
        cacheDir.clear();
        if(logger != NULL)
          logger->write(string("MIGraphX backend: program cache unavailable (continuing): ") + e.what());
      }
    }

    usingFP16 = false;
    {
      lock_guard<mutex> lock(compileMutex);

      for(int bucketBatchSize: bucketSizes) {
        // Try the cache first. Every failure path here falls through to a normal compile: a
        // missing, truncated or foreign entry must cost time, never correctness.
        string cachePath;
        if(!cacheDir.empty()) {
          cachePath = programCachePath(
            cacheDir, onnxBytes, bucketBatchSize, willUseFP16, ctx->useExhaustiveTune, gcnArchName);
          if(FileUtils::exists(cachePath)) {
            try {
              migraphx::program cachedProg = migraphx::load(cachePath.c_str());
              Bucket cachedBucket;
              cachedBucket.batchSize = bucketBatchSize;
              cachedBucket.prog = std::move(cachedProg);
              buckets.push_back(std::move(cachedBucket));
              usingFP16 = willUseFP16;
              if(logger != NULL)
                logger->write(Global::strprintf(
                  "MIGraphX backend: loaded cached program for batch size %d", bucketBatchSize));
              continue;
            }
            catch(const std::exception& e) {
              if(logger != NULL)
                logger->write(string("MIGraphX backend: cached program unusable, recompiling: ") + e.what());
            }
          }
        }

        migraphx::onnx_options onnxOptions;
        const size_t bs = (size_t)bucketBatchSize;
        onnxOptions.set_input_parameter_shape("InputMask", {bs, 1, (size_t)ctx->nnYLen, (size_t)ctx->nnXLen});
        onnxOptions.set_input_parameter_shape(
          "InputSpatial", {bs, (size_t)desc.numInputChannels, (size_t)ctx->nnYLen, (size_t)ctx->nnXLen});
        onnxOptions.set_input_parameter_shape(
          "InputGlobal", {bs, (size_t)desc.numInputGlobalChannels, 1, 1});
        if(hasInputMeta)
          onnxOptions.set_input_parameter_shape("InputMeta", {bs, (size_t)desc.numInputMetaChannels, 1, 1});

        migraphx::program bucketProg = migraphx::parse_onnx_buffer(onnxBytes, onnxOptions);

        if(ctx->useFP16Mode == enabled_t::True || ctx->useFP16Mode == enabled_t::Auto) {
          // quantize_fp16 converts convolutions and dots to FP16 while leaving the reductions that
          // feed RMSNorm and the policy/value heads in FP32, which is the same split the TensorRT
          // backend enforces via per-layer setPrecision. All CDNA parts have fast FP16 so Auto
          // enables it, matching the TensorRT backend's platformHasFastFp16 behavior.
          migraphx::quantize_fp16(bucketProg);
          usingFP16 = true;
        }

        migraphx::compile_options options;
        // Manage device memory ourselves; see the note at the top of this file.
        options.set_offload_copy(false);
        options.set_fast_math(true);
        options.set_exhaustive_tune_flag(ctx->useExhaustiveTune);
        bucketProg.compile(migraphx::target("gpu"), options);

        if(!cachePath.empty())
          saveProgramCache(bucketProg, cachePath, logger);

        Bucket bucket;
        bucket.batchSize = bucketBatchSize;
        bucket.prog = std::move(bucketProg);
        buckets.push_back(std::move(bucket));
      }
    }

    // The largest bucket is maxBatchSize; use it to size the shared I/O buffers and to derive the
    // name/shape metadata the rest of this file relies on.
    migraphx::program& prog = buckets.back().prog;

    // Allocate a device buffer for every program parameter of the LARGEST bucket. This covers the
    // graph inputs, the graph outputs (MIGraphX exposes each output as an "outputName" parameter
    // when offload copy is off), and the internal scratch parameter.
    //
    // Every batch-dimensioned buffer is sized for maxBatchSize and then SHARED by all buckets: a
    // bucket compiled for B rows is handed the same base pointer with its own (smaller) shape, so
    // it touches only the leading B rows. Only scratch is per-bucket, because its size comes from
    // the compiled program rather than from the batch dimension.
    migraphx::program_parameter_shapes paramShapes = prog.get_parameter_shapes();
    // names() hands back pointers into MIGraphX-owned storage; copy them into strings we own.
    vector<string> paramNames;
    for(const char* n: paramShapes.names())
      paramNames.emplace_back(n);
    for(const string& name: paramNames) {
      migraphx::shape s = paramShapes[name.c_str()];
      size_t bytes = s.bytes();
      void* devPtr = nullptr;
      HIP_ERR("ComputeHandle", hipMalloc(&devPtr, bytes));
      HIP_ERR("ComputeHandle", hipMemset(devPtr, 0, bytes));
      buffers[name] = devPtr;
      bufferBytes[name] = bytes;

      // Reject any parameter the compiler decided to hand us non-contiguously.
      //
      // MIGraphX may fold a Transpose into a parameter's STRIDES rather than materialising it.
      // That happens with the transformer NHWC path: the emitter's trailing tonchw Transpose is
      // absorbed, so OutputPolicy keeps shape {N,2,H,W} but gains strides {.., 1, W*C, C} - it is
      // channel-last in memory. Sizes and element counts are unchanged, so nothing below notices,
      // and getOutput would copy raw bytes into a host buffer the caller reads as contiguous
      // NCHW: the policy plane comes back permuted with no error anywhere. Outputs with H=W=1 or
      // a single channel are unaffected, which is why such a bug looks like "only policy is
      // wrong". Fail loudly instead; the fix is to de-permute on the host, not to ignore this.
      {
        vector<size_t> lens = s.lengths();
        vector<size_t> strides = s.strides();
        if(strides.size() == lens.size()) {
          size_t expected = 1;
          bool contiguous = true;
          for(size_t i = lens.size(); i-- > 0;) {
            if(lens[i] != 1 && strides[i] != expected)
              contiguous = false;
            expected *= lens[i];
          }
          if(!contiguous) {
            string got, want;
            size_t e = 1;
            vector<size_t> exp(lens.size(), 1);
            for(size_t i = lens.size(); i-- > 0;) { exp[i] = e; e *= lens[i]; }
            for(size_t i = 0; i < lens.size(); i++) {
              got += (i ? "," : "") + Global::uint64ToString((uint64_t)strides[i]);
              want += (i ? "," : "") + Global::uint64ToString((uint64_t)exp[i]);
            }
            throw StringError(
              "MIGraphX backend: parameter " + name + " is not contiguous (strides {" + got +
              "}, contiguous would be {" + want + "}). MIGraphX folded a layout change into this "
              "buffer instead of materialising it, so a plain byte copy would silently permute the "
              "data. Emit a Reshape on this output so the compiler cannot fold the layout change "
              "into its strides, or de-permute in getOutput using the strides above.");
          }
        }
      }

      // Row elements: elements per batch element. The scratch parameter has no batch dim, so guard.
      vector<size_t> lens = s.lengths();
      size_t rowElts = 1;
      if(lens.size() >= 1 && lens[0] == (size_t)maxBatchSize) {
        for(size_t i = 1; i < lens.size(); i++)
          rowElts *= lens[i];
      } else {
        rowElts = s.elements();
      }
      bufferRowElts[name] = rowElts;
    }

    // Bind each bucket's parameters to those shared buffers, using that bucket's own shapes.
    for(size_t bi = 0; bi < buckets.size(); bi++) {
      Bucket& bucket = buckets[bi];
      migraphx::program_parameter_shapes bucketShapes = bucket.prog.get_parameter_shapes();
      vector<string> bucketNames;
      for(const char* n: bucketShapes.names())
        bucketNames.emplace_back(n);

      // The set of parameters must not vary by bucket - only their batch extent may. If it does,
      // the shared-buffer assumption is void, so fail loudly rather than bind a wrong pointer.
      if(bucketNames.size() != paramNames.size())
        throw StringError(Global::strprintf(
          "MIGraphX backend: bucket %d has %llu parameters but the max bucket has %llu; the "
          "compiled parameter set must not depend on batch size",
          bucket.batchSize, (unsigned long long)bucketNames.size(),
          (unsigned long long)paramNames.size()));

      for(const string& name: bucketNames) {
        migraphx::shape s = bucketShapes[name.c_str()];
        auto it = buffers.find(name);
        if(it == buffers.end())
          throw StringError(
            "MIGraphX backend: bucket " + Global::intToString(bucket.batchSize) +
            " has parameter " + name + " that the max bucket does not");

        void* devPtr = it->second;
        if(s.bytes() > bufferBytes.at(name)) {
          // Scratch can legitimately be larger for a smaller batch (different kernel choices), so
          // give this bucket its own allocation rather than overrunning the shared one.
          HIP_ERR("ComputeHandle", hipMalloc(&devPtr, s.bytes()));
          HIP_ERR("ComputeHandle", hipMemset(devPtr, 0, s.bytes()));
          string ownName = name + "#bucket" + Global::uint64ToString((uint64_t)bi);
          buffers[ownName] = devPtr;
          bufferBytes[ownName] = s.bytes();
        }
        bucket.params.add(name.c_str(), migraphx::argument(s, devPtr));
      }
    }

    // Inputs are addressable by their ONNX names directly.
    for(const char* n: {"InputMask", "InputSpatial", "InputGlobal"})
      aliasName[n] = n;
    if(hasInputMeta)
      aliasName["InputMeta"] = "InputMeta";

    // Outputs are positional. OnnxModelBuilder declares them in this fixed order (see the
    // markOutput calls in onnxmodelbuilder.cpp), so index i corresponds to outputOrder[i].
    static const char* outputOrder[] = {
      "OutputPolicyPass", "OutputPolicy", "OutputValue", "OutputScoreValue", "OutputOwnership"};
    const size_t numOutputs = sizeof(outputOrder) / sizeof(outputOrder[0]);
    for(size_t i = 0; i < numOutputs; i++) {
      string param = "main:#output_" + Global::uint64ToString((uint64_t)i);
      if(buffers.find(param) == buffers.end())
        throw StringError(
          "MIGraphX backend: expected output parameter " + param + " for " + outputOrder[i] +
          " but the compiled program does not have it. MIGraphX's output parameter naming may have "
          "changed; the program has these parameters: " + [&] {
            string all;
            for(const auto& kv: buffers) all += kv.first + " ";
            return all;
          }());
      aliasName[outputOrder[i]] = param;
    }

    // Sanity-check the positional mapping against the shapes the model actually declares, so a
    // reordering in the emitter surfaces here rather than as silently swapped policy/value data.
    auto expectRowElts = [&](const char* name, size_t expected) {
      size_t actual = bufferRowElts.at(aliasName.at(name));
      if(actual != expected)
        throw StringError(Global::strprintf(
          "MIGraphX backend: output %s mapped to %s has %llu elts per row, expected %llu — the "
          "ONNX graph output order does not match this backend's assumed order",
          name, aliasName.at(name).c_str(), (unsigned long long)actual, (unsigned long long)expected));
    };
    const size_t area = (size_t)ctx->nnXLen * ctx->nnYLen;
    expectRowElts("OutputPolicyPass", (size_t)desc.numPolicyChannels);
    expectRowElts("OutputPolicy", (size_t)desc.numPolicyChannels * area);
    expectRowElts("OutputValue", (size_t)desc.numValueChannels);
    expectRowElts("OutputScoreValue", (size_t)desc.numScoreValueChannels);
    expectRowElts("OutputOwnership", (size_t)desc.numOwnershipChannels * area);

    if(logger != NULL) {
      string bucketList;
      for(const Bucket& b: buckets)
        bucketList += (bucketList.empty() ? "" : ",") + Global::intToString(b.batchSize);
      logger->write(
        "MIGraphX backend: compiled model at batch sizes " + bucketList +
        " board " + Global::intToString(ctx->nnXLen) + "x" + Global::intToString(ctx->nnYLen) +
        " FP16 = " + Global::boolToString(usingFP16));
    }
  }

  ~ComputeHandle() {
    // Destructors must not throw, so free errors are swallowed rather than routed through HIP_ERR.
    (void)hipStreamSynchronize(stream);
    for(auto& kv: buffers) {
      (void)hipFree(kv.second);
    }
    (void)hipStreamDestroy(stream);
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  // Inputs keep their ONNX names as parameter names, but MIGraphX does NOT: graph outputs become
  // positional parameters "main:#output_0", "main:#output_1", ... in graph-declaration order.
  // aliasName maps the ONNX tensor name the rest of this file uses onto the actual parameter name.
  map<string, string> aliasName;

  const string& resolveName(const char* name) const {
    auto it = aliasName.find(name);
    if(it != aliasName.end())
      return it->second;
    throw StringError(Global::strprintf("MIGraphX ComputeHandle: unknown tensor name %s", name));
  }

  void* getBuffer(const char* name) const {
    return buffers.at(resolveName(name));
  }

  size_t getBufferBytes(const char* name) const {
    return bufferBytes.at(resolveName(name));
  }

  size_t getBufferRowElts(const char* name) const {
    return bufferRowElts.at(resolveName(name));
  }
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  if(inputsUseNHWC) {
    throw StringError("MIGraphX backend: inputsUseNHWC = false required, other configurations not supported");
  }

  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;
  HIP_ERR("createComputeHandle", hipSetDevice(gpuIdxForThisThread));

  hipDeviceProp_t prop;
  HIP_ERR("createComputeHandle", hipGetDeviceProperties(&prop, gpuIdxForThisThread));

  if(logger != NULL) {
    logger->write(
      "MIGraphX backend thread " + Global::intToString(serverThreadIdx) + ": Found GPU " + string(prop.name) +
      " (" + string(prop.gcnArchName) + ") memory " + Global::uint64ToString(prop.totalGlobalMem));
    logger->write(
      "MIGraphX backend thread " + Global::intToString(serverThreadIdx) + ": Initializing (may take a long time)");
  }

  auto handle = new ComputeHandle(logger, context, loadedModel, maxBatchSize, requireExactNNLen);

  if(logger != NULL) {
    logger->write(
      "MIGraphX backend thread " + Global::intToString(serverThreadIdx) + ": Model version " +
      Global::intToString(loadedModel->modelDesc.modelVersion) +
      " useFP16 = " + Global::boolToString(handle->usingFP16));
    logger->write(
      "MIGraphX backend thread " + Global::intToString(serverThreadIdx) +
      ": Model name: " + loadedModel->modelDesc.name +
      " (" + loadedModel->modelDesc.getShortInfoString() + ")");
  }

  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* gpuHandle) {
  return gpuHandle->usingFP16;
}

bool NeuralNet::setIsWarmup(const ComputeHandle* gpuHandle, bool isWarmup) {
  (void)gpuHandle;
  (void)isWarmup;
  return false;
}

void NeuralNet::printDevices() {
  int numDevices = 0;
  HIP_ERR("printDevices", hipGetDeviceCount(&numDevices));
  for(int i = 0; i < numDevices; i++) {
    hipDeviceProp_t prop;
    HIP_ERR("printDevices", hipGetDeviceProperties(&prop, i));
    cout << "Found GPU device " << i << ": " << prop.name << " (" << prop.gcnArchName << ")" << endl;
  }
}

struct InputBuffers {
  int maxBatchSize;

  size_t singleMaskElts;
  size_t singleMaskBytes;
  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singleInputMetaElts;
  size_t singleInputMetaBytes;
  size_t singlePolicyPassResultElts;
  size_t singlePolicyPassResultBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t inputMaskBufferBytes;
  size_t inputSpatialBufferBytes;
  size_t inputGlobalBufferBytes;
  size_t inputMetaBufferBytes;
  size_t policyPassResultBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  // Host staging buffers. Allocated as pinned memory so the H2D/D2H copies run on the DMA engines
  // rather than through a pageable-memory bounce buffer; at MCTS batch sizes these copies are
  // frequent enough that the difference is measurable.
  float* maskInputs;
  float* spatialInputs;
  float* globalInputs;
  float* metaInputs;
  float* policyPassResults;
  float* policyResults;
  float* valueResults;
  float* scoreValueResults;
  float* ownershipResults;

  // All-ones mask rows used to pad a short batch up to maxBatchSize. See the note in getOutput:
  // an all-zero mask row divides by zero in the graph's masked-mean ops. Sized lazily.
  std::vector<float> paddingMaskOnes;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnXLen, NNPos::MAX_BOARD_LEN));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(
        Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)", nnYLen, NNPos::MAX_BOARD_LEN));

    maxBatchSize = maxBatchSz;
    singleMaskElts = (size_t)nnXLen * nnYLen;
    singleMaskBytes = singleMaskElts * sizeof(float);
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputBytes = singleInputElts * sizeof(float);
    singleInputGlobalElts = m.numInputGlobalChannels;
    singleInputGlobalBytes = singleInputGlobalElts * sizeof(float);
    singleInputMetaElts = m.numInputMetaChannels;
    singleInputMetaBytes = singleInputMetaElts * sizeof(float);
    singlePolicyPassResultElts = (size_t)m.numPolicyChannels;
    singlePolicyPassResultBytes = singlePolicyPassResultElts * sizeof(float);
    singlePolicyResultElts = (size_t)m.numPolicyChannels * nnXLen * nnYLen;
    singlePolicyResultBytes = singlePolicyResultElts * sizeof(float);
    singleValueResultElts = m.numValueChannels;
    singleValueResultBytes = singleValueResultElts * sizeof(float);
    singleScoreValueResultElts = m.numScoreValueChannels;
    singleScoreValueResultBytes = singleScoreValueResultElts * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;
    singleOwnershipResultBytes = singleOwnershipResultElts * sizeof(float);

    testAssert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    testAssert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0) {
      testAssert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);
    }

    inputMaskBufferBytes = maxBatchSize * singleMaskBytes;
    inputSpatialBufferBytes = maxBatchSize * singleInputBytes;
    inputGlobalBufferBytes = maxBatchSize * singleInputGlobalBytes;
    inputMetaBufferBytes = maxBatchSize * singleInputMetaBytes;
    policyPassResultBufferBytes = maxBatchSize * singlePolicyPassResultBytes;
    policyResultBufferBytes = maxBatchSize * singlePolicyResultBytes;
    valueResultBufferBytes = maxBatchSize * singleValueResultBytes;
    scoreValueResultBufferBytes = maxBatchSize * singleScoreValueResultBytes;
    ownershipResultBufferBytes = maxBatchSize * singleOwnershipResultBytes;

    auto allocHost = [](float** ptr, size_t bytes) {
      if(bytes == 0) {
        *ptr = nullptr;
        return;
      }
      HIP_ERR("InputBuffers", hipHostMalloc((void**)ptr, bytes, hipHostMallocDefault));
      memset(*ptr, 0, bytes);
    };
    allocHost(&maskInputs, inputMaskBufferBytes);
    allocHost(&spatialInputs, inputSpatialBufferBytes);
    allocHost(&globalInputs, inputGlobalBufferBytes);
    allocHost(&metaInputs, inputMetaBufferBytes);
    allocHost(&policyPassResults, policyPassResultBufferBytes);
    allocHost(&policyResults, policyResultBufferBytes);
    allocHost(&valueResults, valueResultBufferBytes);
    allocHost(&scoreValueResults, scoreValueResultBufferBytes);
    allocHost(&ownershipResults, ownershipResultBufferBytes);
  }

  ~InputBuffers() {
    for(float* p: {maskInputs, spatialInputs, globalInputs, metaInputs, policyPassResults,
                   policyResults, valueResults, scoreValueResults, ownershipResults}) {
      if(p != nullptr)
        (void)hipHostFree(p);
    }
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);

  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->ctx->nnXLen;
  const int nnYLen = gpuHandle->ctx->nnYLen;
  const int modelVersion = gpuHandle->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = inputBuffers->singleInputMetaElts;
  assert((size_t)numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowMaskInput = &inputBuffers->maskInputs[inputBuffers->singleMaskElts * nIdx];
    float* rowSpatialInput = &inputBuffers->spatialInputs[inputBuffers->singleInputElts * nIdx];
    float* rowGlobalInput = &inputBuffers->globalInputs[inputBuffers->singleInputGlobalElts * nIdx];
    float* rowMetaInput = &inputBuffers->metaInputs[inputBuffers->singleInputMetaElts * nIdx];

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    const bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    if(numMetaFeatures > 0) {
      testAssert(rowMeta != NULL);
      testAssert(hasRowMeta);
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    } else {
      testAssert(!hasRowMeta);
    }
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);
    std::copy(rowSpatialInput, rowSpatialInput + inputBuffers->singleMaskElts, rowMaskInput);
  }

  assert(inputBuffers->singleMaskElts == gpuHandle->getBufferRowElts("InputMask"));
  assert(inputBuffers->singleInputElts == gpuHandle->getBufferRowElts("InputSpatial"));
  assert(inputBuffers->singleInputGlobalElts == gpuHandle->getBufferRowElts("InputGlobal"));
  if(numMetaFeatures > 0)
    assert(inputBuffers->singleInputMetaElts == gpuHandle->getBufferRowElts("InputMeta"));
  assert(inputBuffers->singlePolicyPassResultElts == gpuHandle->getBufferRowElts("OutputPolicyPass"));
  assert(inputBuffers->singlePolicyResultElts == gpuHandle->getBufferRowElts("OutputPolicy"));
  assert(inputBuffers->singleValueResultElts == gpuHandle->getBufferRowElts("OutputValue"));
  assert(inputBuffers->singleScoreValueResultElts == gpuHandle->getBufferRowElts("OutputScoreValue"));
  assert(inputBuffers->singleOwnershipResultElts == gpuHandle->getBufferRowElts("OutputOwnership"));

  const int numPolicyChannels = inputBuffers->singlePolicyPassResultElts;
  assert(inputBuffers->singlePolicyResultElts == (size_t)numPolicyChannels * nnXLen * nnYLen);

  // The selected bucket's program is compiled for exactly its own batch size, so only the first
  // batchSize rows are copied in and read back; the padding rows' outputs are ignored.
  //
  // Padding rows must NOT be left as all-zero. When requireExactNNLen is false the emitted graph
  // takes masked means as Div(ReduceSum(x), maskSum), where maskSum is the per-row count of
  // on-board cells. An all-zero mask row makes that a 0/0 division, so the padding rows produce
  // NaN/Inf rather than harmless garbage.
  //
  // Give every padding row a fully on-board mask (all ones) so maskSum == H*W. The rows then
  // compute finite values from zero spatial input and are discarded.
  //
  // Note: this does NOT fix the transformer policy discrepancy — that was the original hypothesis
  // and it was disproved (the error was bit-identical afterwards, because the single-position test
  // path never pads at all). This guard matters for the MCTS path, where short batches are real.
  // Re-padded on every call rather than cached: a larger batch overwrites this region with real
  // data, so a later smaller batch would otherwise inherit stale rows. With bucketing this is
  // doubly true, since consecutive evals may run different-sized programs over the same buffers.
  // The copy is one contiguous memcpy of (shapeBatchSize-batchSize) mask rows and is negligible
  // next to the forward pass.
  hipStream_t stream = gpuHandle->stream;

  // Dispatch to the smallest compiled bucket that fits, and pad only up to THAT bucket rather
  // than up to maxBatchSize. This is the whole point of bucketing: at 192 threads the search's
  // mean batch is ~89, so a single maxBatchSize program spends over half its compute on padding.
  const ComputeHandle::Bucket& bucket = gpuHandle->bucketFor(batchSize);
  const int shapeBatchSize = bucket.batchSize;

  if(batchSize < shapeBatchSize) {
    const int padRows = shapeBatchSize - batchSize;
    if(inputBuffers->paddingMaskOnes.size() != inputBuffers->singleMaskElts * (size_t)padRows)
      inputBuffers->paddingMaskOnes.assign(inputBuffers->singleMaskElts * (size_t)padRows, 1.0f);
    HIP_ERR(
      "getOutput",
      hipMemcpyAsync(
        (char*)gpuHandle->getBuffer("InputMask") + inputBuffers->singleMaskBytes * batchSize,
        inputBuffers->paddingMaskOnes.data(), inputBuffers->singleMaskBytes * (size_t)padRows,
        hipMemcpyHostToDevice, stream));
  }
  HIP_ERR(
    "getOutput",
    hipMemcpyAsync(
      gpuHandle->getBuffer("InputMask"), inputBuffers->maskInputs,
      inputBuffers->singleMaskBytes * batchSize, hipMemcpyHostToDevice, stream));
  HIP_ERR(
    "getOutput",
    hipMemcpyAsync(
      gpuHandle->getBuffer("InputSpatial"), inputBuffers->spatialInputs,
      inputBuffers->singleInputBytes * batchSize, hipMemcpyHostToDevice, stream));
  HIP_ERR(
    "getOutput",
    hipMemcpyAsync(
      gpuHandle->getBuffer("InputGlobal"), inputBuffers->globalInputs,
      inputBuffers->singleInputGlobalBytes * batchSize, hipMemcpyHostToDevice, stream));
  if(numMetaFeatures > 0) {
    HIP_ERR(
      "getOutput",
      hipMemcpyAsync(
        gpuHandle->getBuffer("InputMeta"), inputBuffers->metaInputs,
        inputBuffers->singleInputMetaBytes * batchSize, hipMemcpyHostToDevice, stream));
  }

  // run_async rather than eval: eval() runs on MIGraphX's own internal stream, which is not
  // ordered against the copies above, so the program could read inputs before they land.
  //
  // const_cast: run_async is non-const in the MIGraphX C++ API, but selecting a bucket is a
  // read-only operation on the handle and the buffers it writes are this handle's own.
  const_cast<ComputeHandle::Bucket&>(bucket).prog.run_async(
    const_cast<ComputeHandle::Bucket&>(bucket).params, stream);

  HIP_ERR(
    "getOutput",
    hipMemcpyAsync(
      inputBuffers->policyPassResults, gpuHandle->getBuffer("OutputPolicyPass"),
      inputBuffers->singlePolicyPassResultBytes * batchSize, hipMemcpyDeviceToHost, stream));
  HIP_ERR(
    "getOutput",
    hipMemcpyAsync(
      inputBuffers->policyResults, gpuHandle->getBuffer("OutputPolicy"),
      inputBuffers->singlePolicyResultBytes * batchSize, hipMemcpyDeviceToHost, stream));
  HIP_ERR(
    "getOutput",
    hipMemcpyAsync(
      inputBuffers->valueResults, gpuHandle->getBuffer("OutputValue"),
      inputBuffers->singleValueResultBytes * batchSize, hipMemcpyDeviceToHost, stream));
  HIP_ERR(
    "getOutput",
    hipMemcpyAsync(
      inputBuffers->scoreValueResults, gpuHandle->getBuffer("OutputScoreValue"),
      inputBuffers->singleScoreValueResultBytes * batchSize, hipMemcpyDeviceToHost, stream));
  HIP_ERR(
    "getOutput",
    hipMemcpyAsync(
      inputBuffers->ownershipResults, gpuHandle->getBuffer("OutputOwnership"),
      inputBuffers->singleOwnershipResultBytes * batchSize, hipMemcpyDeviceToHost, stream));

  // One sync per eval, after all the D2H copies are queued, rather than an implicit sync per copy.
  HIP_ERR("getOutput", hipStreamSynchronize(stream));

  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];

    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = &inputBuffers->policyPassResults[row * inputBuffers->singlePolicyPassResultElts];
    const float* policySrcBuf = &inputBuffers->policyResults[row * inputBuffers->singlePolicyResultElts];
    float* policyProbs = output->policyProbs;

    // These are in logits, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    // Handle version >= 12 policy optimism
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
      // MIGraphX outputs are NCHW, same as TensorRT
      for(int i = 0; i < nnXLen * nnYLen; i++) {
        float p = policySrcBuf[i];
        float pOpt = policySrcBuf[i + nnXLen * nnYLen];
        policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
      }
      SymmetryHelpers::copyOutputsWithSymmetry(
        policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
    } else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
    }

    int numValueChannels = inputBuffers->singleValueResultElts;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    // As above, these are NOT actually from white's perspective, but rather the player to move.
    // As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = &inputBuffers->ownershipResults[row * nnXLen * nnYLen];
      assert(inputBuffers->singleOwnershipResultElts == (size_t)nnXLen * nnYLen);
      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    int numScoreValueChannels = inputBuffers->singleScoreValueResultElts;
    if(modelVersion >= 9) {
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    } else if(modelVersion >= 8) {
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(modelVersion >= 4) {
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(modelVersion >= 3) {
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      // Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the
      // mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}

// These per-layer test entry points exist for the CUDA/Eigen backends which build the net layer by
// layer. This backend hands a whole ONNX graph to MIGraphX and has no per-layer handles, so like
// the TensorRT backend it declines all of them.
bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)outputBuffer;
  return false;
}

// Mask should be in 'NHW' format (no "C" channel).
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

#endif  // USE_MIGRAPHX_BACKEND
