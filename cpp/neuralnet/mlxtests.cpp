#ifdef USE_MLX_BACKEND

/**
 * MLX backend unit tests.
 *
 * Holds the test entry points runMLXWinogradTests() and runMLXWinotunerTests(),
 * extracted from mlxbackend.cpp / mlxwinotuner.cpp so the production backend
 * translation units stay focused on inference code.
 *
 * These run once per process, on the first call to NeuralNet::testEvaluateConv
 * (see the ranMLXAuxTests guard in mlxbackend.cpp). The runnnlayertests
 * subcommand exercises that path.
 *
 * runMLXBatchNormFP16Test() / runMLXConvLayerFP16WinogradTest() remain defined
 * in mlxbackend.cpp because they construct the file-local BatchNormLayer /
 * ConvLayer classes; they are forward-declared here and called by
 * runMLXWinogradTests().
 */

#include "../neuralnet/mlxwinograd.h"
#include "../neuralnet/mlxwinotuner.h"
#include "../neuralnet/desc.h"
#include "../core/global.h"
#include "../core/logger.h"
#include "../core/test.h"

#include <mlx/mlx.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
namespace mx = mlx::core;

// Defined in mlxbackend.cpp — they need the file-local BatchNormLayer /
// ConvLayer classes, so they cannot move here.
void runMLXBatchNormFP16Test();
void runMLXConvLayerFP16WinogradTest();
void runMLXTransformerLayerFP16Test();

void runMLXWinogradTests() {
  cout << "Running MLX Winograd F(2,3) tests" << endl;
  // Naive direct 3x3 "same" conv NHWC, OIHW weights, as independent oracle.
  auto direct = [](const vector<float>& in,int N,int H,int W,int Cin,
                    const vector<float>& w,int Cout){
    vector<float> out((size_t)N*H*W*Cout,0.f);
    for(int n=0;n<N;n++)for(int oy=0;oy<H;oy++)for(int ox=0;ox<W;ox++)
    for(int oc=0;oc<Cout;oc++){ float s=0.f;
      for(int ic=0;ic<Cin;ic++)for(int a=0;a<3;a++)for(int b=0;b<3;b++){
        int iy=oy+a-1,ix=ox+b-1;
        if(iy>=0&&iy<H&&ix>=0&&ix<W)
          s+=in[(((size_t)n*H+iy)*W+ix)*Cin+ic]
             *w[(((size_t)oc*Cin+ic)*3+a)*3+b];
      }
      out[(((size_t)n*H+oy)*W+ox)*Cout+oc]=s;
    }
    return out;
  };
  // Scope `rng`/`dist` to this CPU-oracle loop so they don't shadow same-named
  // locals in the per-test blocks below.
  {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(auto dims : vector<array<int,5>>{{1,5,5,3,4},{2,19,19,8,16},{1,7,13,4,4}}){
      int N=dims[0],H=dims[1],W=dims[2],Cin=dims[3],Cout=dims[4];
      vector<float> in((size_t)N*H*W*Cin); for(auto&x:in)x=dist(rng);
      vector<float> w((size_t)Cout*Cin*9); for(auto&x:w)x=dist(rng);
      auto ref = direct(in,N,H,W,Cin,w,Cout);
      auto got = MLXWinograd::cpuConv2d3x3(in,N,H,W,Cin,w,Cout);
      double maxErr=0.0;
      for(size_t i=0;i<ref.size();i++)
        maxErr=std::max(maxErr,(double)std::fabs(ref[i]-got[i]));
      cout<<"  dims "<<N<<"x"<<H<<"x"<<W<<"x"<<Cin<<"->"<<Cout
          <<" maxErr="<<maxErr<<endl;
      testAssert(maxErr < 1e-4);
    }
  }
  cout << "MLX Winograd F(2,3) CPU reference OK" << endl;

  // GPU Winograd metal_kernel validated against the CPU oracle.
  {
    namespace mxc = mlx::core;
    int N=2,H=19,W=19,Cin=8,Cout=16;
    std::mt19937 grng(777);
    std::uniform_real_distribution<float> gdist(-1.f,1.f);
    vector<float> in((size_t)N*H*W*Cin); for(auto&x:in)x=gdist(grng);
    vector<float> w((size_t)Cout*Cin*9); for(auto&x:w)x=gdist(grng);
    auto refv = MLXWinograd::cpuConv2d3x3(in,N,H,W,Cin,w,Cout);
    mxc::array inArr(in.data(),{N,H,W,Cin},mxc::float32);
    auto Uw = MLXWinograd::makeWinogradWeights(w,Cout,Cin);
    MLXWinograd::InputTransform inCfg;
    MLXWinograd::OutputUntransform outCfg;
    mxc::array o = MLXWinograd::winogradConv2d(inArr,Uw,Cout,inCfg,outCfg);
    mxc::eval(o);
    const float* od = o.data<float>();
    double maxErr=0.0;
    for(size_t i=0;i<refv.size();i++)
      maxErr=std::max(maxErr,(double)std::fabs(refv[i]-od[i]));
    cout<<"  MLX-metal winograd maxErr="<<maxErr<<endl;
    testAssert(maxErr < 2e-3);
    // Also assert against the naive direct-conv oracle. cpuConv2d3x3 is itself a
    // Winograd F(2,3) impl sharing the same B/G/A transform matrices the GPU
    // kernel hard-codes, so a shared sign/transpose error in those matrices would
    // cancel in both and pass. `direct` is an independent O(K^2) conv, so it
    // catches such a shared-transform-matrix bug that cpuConv2d3x3 would mask.
    auto refDirect = direct(in,N,H,W,Cin,w,Cout);
    double maxErrD=0.0;
    for(size_t i=0;i<refDirect.size();i++)
      maxErrD=std::max(maxErrD,(double)std::fabs(refDirect[i]-od[i]));
    cout<<"  MLX-metal winograd vs direct maxErr="<<maxErrD<<endl;
    testAssert(maxErrD < 2e-3);
  }

  // FP16 Winograd: input/weights/output all fp16, compared against fp32 CPU oracle.
  // Tolerance ~5e-2 covers (a) fp16 input quantization, (b) fp16 weight quantization,
  // (c) fp16 transform/store rounding. The matmul itself accumulates in fp32 (MLX
  // steel gemm default), so the dominant error is the storage round-trip.
  {
    namespace mxc = mlx::core;
    int N=2,H=19,W=19,Cin=8,Cout=16;
    std::mt19937 grng(778);
    std::uniform_real_distribution<float> gdist(-1.f,1.f);
    vector<float> in((size_t)N*H*W*Cin); for(auto&x:in)x=gdist(grng);
    vector<float> w((size_t)Cout*Cin*9); for(auto&x:w)x=gdist(grng);
    auto refv = MLXWinograd::cpuConv2d3x3(in,N,H,W,Cin,w,Cout);
    mxc::array inArrF32(in.data(),{N,H,W,Cin},mxc::float32);
    mxc::array inArr = mxc::astype(inArrF32, mxc::float16);
    auto Uw = MLXWinograd::makeWinogradWeights(w,Cout,Cin,/*useFP16=*/true);
    MLXWinograd::InputTransform inCfg;
    MLXWinograd::OutputUntransform outCfg;
    mxc::array o = MLXWinograd::winogradConv2d(inArr,Uw,Cout,inCfg,outCfg,/*useFP16=*/true);
    mxc::eval(o);
    testAssert(o.dtype() == mxc::float16);
    mxc::array oF32 = mxc::astype(o, mxc::float32);
    mxc::eval(oF32);
    const float* od = oF32.data<float>();
    double maxErr=0.0;
    for(size_t i=0;i<refv.size();i++)
      maxErr=std::max(maxErr,(double)std::fabs(refv[i]-od[i]));
    cout<<"  MLX-metal winograd FP16 maxErr="<<maxErr<<endl;
    testAssert(maxErr < 5e-2);
    // Independent oracle (see GPU block above): catch a shared-transform-matrix
    // error that cpuConv2d3x3 would mask, here at the fp16 tolerance.
    auto refDirect = direct(in,N,H,W,Cin,w,Cout);
    double maxErrD=0.0;
    for(size_t i=0;i<refDirect.size();i++)
      maxErrD=std::max(maxErrD,(double)std::fabs(refDirect[i]-od[i]));
    cout<<"  MLX-metal winograd FP16 vs direct maxErr="<<maxErrD<<endl;
    testAssert(maxErrD < 5e-2);
  }

  // FP16 Winograd stage-2 matmul ACCUMULATION guard, at production width.
  //
  // The small-Cin absolute-tol block above only sums ~72 terms per output, so a
  // regression of the stage-2 matmul accumulator (MLX steel gemm AccumType: float ->
  // half) would stay under its 5e-2 bound and slip through. A larger Cin can't reuse
  // an absolute tolerance either: output magnitude grows ~sqrt(9*Cin), so the fp16
  // STORAGE round-trip error grows with it and at Cin=384 the max ABSOLUTE error
  // (~0.05) already coincides with 5e-2.
  //
  // So measure a scale-invariant NORMALIZED error (maxAbsErr / outMagMax) at two
  // widths and compare. With fp32 accumulation the error is storage-bound, hence both
  // small AND roughly FLAT in Cin (measured ~8e-4 at both Cin=8 and Cin=384, even
  // dipping slightly). fp16 accumulation instead adds error that grows with the number
  // of summed terms, so its Cin=384 value would exceed the storage floor and grow far
  // above the Cin=8 (~72-term) value. We assert both: (a) the Cin=384 normalized error
  // stays under a generous storage-floor bound, and (b) it does not grow much vs Cin=8.
  // The ratio check needs no absolute-magnitude tuning, so it is hardware-independent.
  {
    namespace mxc = mlx::core;
    int N=2,H=19,W=19,Cout=64;
    auto normErrAtCin = [&](int Cin, int seed) {
      std::mt19937 grng(seed);
      std::uniform_real_distribution<float> gdist(-1.f,1.f);
      vector<float> in((size_t)N*H*W*Cin); for(auto&x:in)x=gdist(grng);
      vector<float> w((size_t)Cout*Cin*9); for(auto&x:w)x=gdist(grng);
      auto refv = MLXWinograd::cpuConv2d3x3(in,N,H,W,Cin,w,Cout);
      mxc::array inArrF32(in.data(),{N,H,W,Cin},mxc::float32);
      mxc::array inArr = mxc::astype(inArrF32, mxc::float16);
      auto Uw = MLXWinograd::makeWinogradWeights(w,Cout,Cin,/*useFP16=*/true);
      MLXWinograd::InputTransform inCfg;
      MLXWinograd::OutputUntransform outCfg;
      mxc::array o = MLXWinograd::winogradConv2d(inArr,Uw,Cout,inCfg,outCfg,/*useFP16=*/true);
      mxc::eval(o);
      testAssert(o.dtype() == mxc::float16);
      mxc::array oF32 = mxc::astype(o, mxc::float32); mxc::eval(oF32);
      const float* od = oF32.data<float>();
      double maxErr=0.0, outMagMax=0.0;
      for(size_t i=0;i<refv.size();i++){
        maxErr=std::max(maxErr,(double)std::fabs(refv[i]-od[i]));
        outMagMax=std::max(outMagMax,(double)std::fabs(refv[i]));
      }
      return maxErr / (outMagMax + 1e-3);
    };
    double relErrSmall = normErrAtCin(8,   778);   // ~72-term reductions
    double relErrBig   = normErrAtCin(384, 779);   // ~3456-term reductions
    cout<<"  MLX-metal winograd FP16 accum guard: relErr(Cin=8)="<<relErrSmall
        <<" relErr(Cin=384)="<<relErrBig<<" ratio="<<(relErrBig/relErrSmall)<<endl;
    // fp32 accumulation: normalized error storage-bound (~8e-4) and flat in Cin
    // (measured ratio < 1). 1e-2 is ~12x over that floor (robust across chips); the
    // 3x growth bound catches an fp16-accum regression, whose Cin=384 error climbs
    // with the term count while the Cin=8 value barely moves.
    testAssert(relErrBig < 1e-2);
    testAssert(relErrBig < 3.0 * relErrSmall);
  }

  runMLXBatchNormFP16Test();
  runMLXConvLayerFP16WinogradTest();
  runMLXTransformerLayerFP16Test();

  // Smoke test — verify Winograd plumbing.
  // Trivial 4x4x1 input, 1 output channel, all-ones filter.
  {
    namespace mxc = mlx::core;
    std::vector<float> in_data(16, 1.0f);
    mxc::array inp(in_data.data(), {1, 4, 4, 1}, mxc::float32);
    std::vector<float> w_data(9, 1.0f);
    auto Uw = MLXWinograd::makeWinogradWeights(w_data, /*Cout*/1, /*Cin*/1,
                                               /*useFP16*/false);
    MLXWinograd::InputTransform    inCfg;
    MLXWinograd::OutputUntransform outCfg;
    mxc::array out = MLXWinograd::winogradConv2d(inp, Uw, /*Cout*/1, inCfg, outCfg,
                                                 /*useFP16*/false);
    mxc::eval(out);
    testAssert(out.shape().size() == 4);
    testAssert(out.shape(0) == 1);
    testAssert(out.shape(1) == 4);
    testAssert(out.shape(2) == 4);
    testAssert(out.shape(3) == 1);
    // Verify expected values for all-ones input × all-ones 3x3 filter (same padding).
    // Uses cpuConv2d3x3 as the CPU reference to avoid hard-coding per-pixel expected values.
    auto ref = MLXWinograd::cpuConv2d3x3(in_data, /*N*/1, /*H*/4, /*W*/4, /*Cin*/1,
                                          w_data, /*Cout*/1);
    const float* op = out.data<float>();
    double smokeMaxErr = 0.0;
    for(int idx = 0; idx < 16; idx++)
      smokeMaxErr = std::max(smokeMaxErr, (double)std::fabs(op[idx] - ref[idx]));
    testAssert(smokeMaxErr < 1e-4);
    cout << "  MLX Winograd kernel-plumbing smoke test passed (value maxErr=" << smokeMaxErr << ")" << endl;
  }

  // WPT=1, 4, 8 must produce bit-identical output (fp32).
  // Realistic shape: N=2, H=W=19, C=64 -> Ntiles = 2*10*10 = 200.
  {
    using namespace MLXWinograd;
    namespace mx = mlx::core;
    std::vector<float> in_data((size_t)2*19*19*64);
    std::mt19937 rng(0x1234u);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    for(auto& x : in_data) x = fdist(rng);
    mx::array inp(in_data.data(), {2, 19, 19, 64}, mx::float32);

    std::vector<float> w_data((size_t)64*64*9, 1.0f);
    mx::array Uw = makeWinogradWeights(w_data, 64, 64, false);

    auto runWith = [&](int wpt_in, int wpt_out) {
      InputTransform    inCfg;  inCfg.wpt  = wpt_in;
      OutputUntransform outCfg; outCfg.wpt = wpt_out;
      mx::array out = winogradConv2d(inp, Uw, 64, inCfg, outCfg, false);
      mx::eval(out);
      return out;
    };

    // Vary input WPT, output stays at WPT=1.
    mx::array out_w1   = runWith(1, 1);
    mx::array out_w4   = runWith(4, 1);
    mx::array out_w8   = runWith(8, 1);
    // Vary output WPT, input stays at WPT=1.
    mx::array out_ow4  = runWith(1, 4);
    mx::array out_ow8  = runWith(1, 8);

    // Compare bit-for-bit (no FP-ordering change — only thread loop unroll differs).
    const float* p1   = out_w1.data<float>();
    const float* p4   = out_w4.data<float>();
    const float* p8   = out_w8.data<float>();
    const float* po4  = out_ow4.data<float>();
    const float* po8  = out_ow8.data<float>();
    size_t n = (size_t)2 * 19 * 19 * 64;
    for(size_t i = 0; i < n; i++) {
      testAssert(p1[i] == p4[i]);
      testAssert(p1[i] == p8[i]);
      testAssert(p1[i] == po4[i]);
      testAssert(p1[i] == po8[i]);
    }
    cout << "  MLX Winograd WPT bit-for-bit equivalence (1/4/8) passed" << endl;
  }

  // Tail-guard coverage: Ntiles=100 (N=1, H=W=19) is NOT
  // divisible by WPT=8, so the last thread along the slow axis has
  // tileIdx in {96..103}; iterations 100..103 must hit the break.
  {
    using namespace MLXWinograd;
    namespace mx = mlx::core;
    std::vector<float> in_data((size_t)1*19*19*64);
    std::mt19937 rng(0xBEEFu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(auto& x : in_data) x = dist(rng);
    mx::array inp(in_data.data(), {1, 19, 19, 64}, mx::float32);

    std::vector<float> w_data((size_t)64*64*9, 1.0f);
    mx::array Uw = makeWinogradWeights(w_data, 64, 64, false);

    auto runWith = [&](int wpt_in, int wpt_out) {
      InputTransform    inCfg;  inCfg.wpt  = wpt_in;
      OutputUntransform outCfg; outCfg.wpt = wpt_out;
      mx::array out = winogradConv2d(inp, Uw, 64, inCfg, outCfg, false);
      mx::eval(out);
      return out;
    };

    mx::array out_w1   = runWith(1, 1);
    mx::array out_w8in  = runWith(8, 1);  // input WPT=8 with Ntiles%WPT != 0
    mx::array out_w8out = runWith(1, 8);  // output WPT=8 with Ntiles%WPT != 0

    const float* p1   = out_w1.data<float>();
    const float* p8i  = out_w8in.data<float>();
    const float* p8o  = out_w8out.data<float>();
    size_t n = (size_t)1 * 19 * 19 * 64;
    for(size_t i = 0; i < n; i++) {
      testAssert(p1[i] == p8i[i]);
      testAssert(p1[i] == p8o[i]);
    }
    cout << "  MLX Winograd WPT tail-guard coverage (Ntiles=100, WPT=8) passed" << endl;
  }

  // Input VW=1, 2, 4 must produce bit-identical fp16 output (Cfast). C=64
  // is divisible by 4 — VW=4 valid. Output VW is gone (kernel is VW=1
  // monomorphic).
  {
    using namespace MLXWinograd;
    namespace mx = mlx::core;
    std::vector<float> in_data((size_t)2*19*19*64);
    std::mt19937 rng(0x9ABCu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(auto& x : in_data) x = dist(rng);
    mx::array inp = mx::astype(mx::array(in_data.data(), {2, 19, 19, 64}, mx::float32), mx::float16);

    std::vector<float> w_data((size_t)64*64*9, 0.5f);
    mx::array Uw = makeWinogradWeights(w_data, 64, 64, true);

    auto runWith = [&](int vw_in) {
      InputTransform    inCfg;  inCfg.vw  = vw_in;
      OutputUntransform outCfg;
      mx::array out = winogradConv2d(inp, Uw, 64, inCfg, outCfg, true);
      mx::eval(out);
      return out;
    };

    mx::array out_v1   = runWith(1);
    mx::array out_v2in = runWith(2);
    mx::array out_v4in = runWith(4);

    // Cast to fp32 and compare bit-for-bit (no FP-op reordering — only channel
    // sequencing differs across input VW, so equality must hold exactly).
    mx::array out_v1_fp32   = mx::astype(out_v1,   mx::float32);
    mx::array out_v2in_fp32 = mx::astype(out_v2in, mx::float32);
    mx::array out_v4in_fp32 = mx::astype(out_v4in, mx::float32);
    mx::eval(out_v1_fp32, out_v2in_fp32, out_v4in_fp32);
    const float* p1  = out_v1_fp32.data<float>();
    const float* p2i = out_v2in_fp32.data<float>();
    const float* p4i = out_v4in_fp32.data<float>();
    size_t n = (size_t)2 * 19 * 19 * 64;
    for(size_t i = 0; i < n; i++) {
      testAssert(p1[i] == p2i[i]);
      testAssert(p1[i] == p4i[i]);
    }
    cout << "  MLX Winograd input-VW bit-for-bit equivalence (1/2/4 fp16, Cfast) passed" << endl;
  }

  // Input-stage GridOrder::Cfast and GridOrder::Tfast must produce
  // bit-identical fp32 output. They differ only in which thread does which
  // (c, tileIdx) pair; the on-disk layout is unchanged. The output kernel
  // is Cfast-monomorphic, so only the input gridOrder is varied here.
  {
    using namespace MLXWinograd;
    namespace mx = mlx::core;
    std::vector<float> in_data((size_t)2*19*19*64);
    std::mt19937 rng(0xDEADu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(auto& x : in_data) x = dist(rng);
    mx::array inp(in_data.data(), {2, 19, 19, 64}, mx::float32);

    std::vector<float> w_data((size_t)64*64*9);
    for(auto& x : w_data) x = dist(rng);
    mx::array Uw = makeWinogradWeights(w_data, 64, 64, false);

    auto runWith = [&](GridOrder go_in) {
      InputTransform    inC;  inC.gridOrder  = go_in;
      OutputUntransform outC;
      mx::array out = winogradConv2d(inp, Uw, 64, inC, outC, false);
      mx::eval(out);
      return out;
    };

    // Input Cfast (baseline).
    mx::array out_c = runWith(GridOrder::Cfast);
    // Input Tfast — kernel swaps thread mapping, output must match.
    mx::array out_t = runWith(GridOrder::Tfast);

    const float* pc = out_c.data<float>();
    const float* pt = out_t.data<float>();
    size_t n = (size_t)2 * 19 * 19 * 64;
    for(size_t i = 0; i < n; i++) {
      testAssert(pc[i] == pt[i]);
    }
    std::cout << "  MLX Winograd input-stage Cfast vs Tfast bit-for-bit equivalence passed" << std::endl;
  }

  // Tail-guard coverage: input Tfast with C=67 (not
  // divisible by WPT=8). Last thread group has only 3 channels (67 % 8 = 3);
  // the tail-guard `if (c >= C_k) break;` fires for the other 5 iterations.
  // We verify input Tfast still matches input Cfast for this shape.
  {
    using namespace MLXWinograd;
    namespace mx = mlx::core;
    std::vector<float> in_data((size_t)1*19*19*67);
    std::mt19937 rng(0xFEEDu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(auto& x : in_data) x = dist(rng);
    mx::array inp(in_data.data(), {1, 19, 19, 67}, mx::float32);

    std::vector<float> w_data((size_t)67*67*9);
    for(auto& x : w_data) x = dist(rng);
    mx::array Uw = makeWinogradWeights(w_data, 67, 67, false);

    auto runWith = [&](GridOrder go, int wpt) {
      InputTransform    inC;  inC.gridOrder  = go;  inC.wpt  = wpt;
      OutputUntransform outC; outC.wpt = wpt;
      mx::array out = winogradConv2d(inp, Uw, 67, inC, outC, false);
      mx::eval(out);
      return out;
    };

    mx::array out_cfast = runWith(GridOrder::Cfast, 1);
    mx::array out_tfast = runWith(GridOrder::Tfast, 8);  // input Tfast with WPT=8, C=67 not divisible.

    const float* pc = out_cfast.data<float>();
    const float* pt = out_tfast.data<float>();
    size_t n = (size_t)1 * 19 * 19 * 67;
    for(size_t i = 0; i < n; i++) {
      testAssert(pc[i] == pt[i]);
    }
    std::cout << "  MLX Winograd input-stage Tfast tail-guard coverage (C=67, WPT=8) passed" << std::endl;
  }

  {
    // Output kernel is monomorphic on VW=1, GRID_ORDER=Cfast.
    // Run a full conv via winogradConv2d with a deterministic input and weight
    // tensor; assert the output is finite and matches a stable reference
    // checksum (sum of absolute values to 4 decimal places). This catches:
    //   - stale Tfast read paths in the output kernel
    //   - stale VW>1 vector-load paths
    //   - Std-only weight layout not consistent with kernel reads
    namespace mx = mlx::core;
    using namespace MLXWinograd;

    const int N = 1, H = 8, W = 8, Cin = 8, Cout = 8;

    // Deterministic input: i*0.01.
    std::vector<float> inData(N * H * W * Cin);
    for(size_t i = 0; i < inData.size(); i++) inData[i] = (float)i * 0.01f;
    mx::array input(inData.data(), {N, H, W, Cin}, mx::float32);

    // Deterministic 3x3 weights: (oc*Cin*9 + ic*9 + k)*0.001.
    std::vector<float> wData(Cout * Cin * 9);
    for(size_t i = 0; i < wData.size(); i++) wData[i] = (float)i * 0.001f;
    // makeWinogradWeights takes raw [Cout, Cin, 3, 3] flattened and produces
    // the transformed [16, Cin, Cout] tensor (Std-only).
    mx::array U = makeWinogradWeights(wData, Cout, Cin, /*useFP16=*/false);

    // Output config: Std OutputUntransform has tg0/tg1/wpt only.
    InputTransform inCfg{};
    inCfg.tg0 = 32; inCfg.tg1 = 1; inCfg.wpt = 1; inCfg.vw = 1;
    inCfg.gridOrder = GridOrder::Cfast;
    OutputUntransform outCfg{};
    outCfg.tg0 = 16; outCfg.tg1 = 4; outCfg.wpt = 1;

    mx::array out = winogradConv2d(input, U, Cout, inCfg, outCfg);
    mx::eval(out);

    // Output shape must be [N, H, W, Cout].
    testAssert(out.shape(0) == N);
    testAssert(out.shape(1) == H);
    testAssert(out.shape(2) == W);
    testAssert(out.shape(3) == Cout);

    // Pull data; assert all finite.
    std::vector<float> outData(out.size());
    out.eval();
    std::memcpy(outData.data(), out.data<float>(), outData.size() * sizeof(float));
    for(float v : outData) testAssert(std::isfinite(v));

    // Stable checksum: sum of absolute values. This is a regression check —
    // a change in numerics suggests a kernel-template mismatch (e.g., output
    // kernel reads channels via VW>1 path that no longer exists, producing
    // UB-flavored garbage).
    double sumAbs = 0.0;
    for(float v : outData) sumAbs += std::abs(v);
    // Recompute this expected value once after the test is first written —
    // it captures the deterministic conv result for the inputs above. The
    // test passes thereafter as a regression check, not a correctness check.
    // Tolerance: 0.5% to absorb minor reordering noise from MLX graph rewrites.
    constexpr double expectedSumAbs = 22788.156637847424;  // captured 2026-05-21
    testAssert(std::abs(sumAbs - expectedSumAbs) / expectedSumAbs < 0.005);
    std::cout << "  Output-kernel monomorphic smoke test OK" << std::endl;
  }
}

void runMLXWinotunerTests() {
  cout << "Running MLX Winograd tuner tests" << endl;

  {
    // Conv-3x3 distribution formatter — pure-function test. Verifies the
    // log-line format directly without any descriptor walk or GPU work.
    // Order convention: pairs sorted descending by invocation
    // count, ties broken by channel count descending.

    // Case A: two distinct shapes, each appearing once. Tie on count, so
    // tie-break by channel count descending: 64 before 32.
    {
      std::map<int,int> inputC  = {{32, 1}, {64, 1}};
      std::map<int,int> outputC = {{32, 1}, {64, 1}};
      std::string line = MLXWinogradTuner::formatConv3x3DistributionLine(2, inputC, outputC);
      testAssert(line.find("MLX tuner conv3x3 distribution:") != std::string::npos);
      testAssert(line.find("total=2") != std::string::npos);
      testAssert(line.find("input_c=64:1,32:1") != std::string::npos);
      testAssert(line.find("output_c=64:1,32:1") != std::string::npos);
    }

    // Case B: asymmetric counts. 384 appears 36 times, 192 once. Sort by
    // count descending, so 384 first regardless of channel-count order.
    {
      std::map<int,int> inputC  = {{384, 36}, {192, 1}};
      std::map<int,int> outputC = {{384, 37}};
      std::string line = MLXWinogradTuner::formatConv3x3DistributionLine(37, inputC, outputC);
      testAssert(line.find("total=37") != std::string::npos);
      testAssert(line.find("input_c=384:36,192:1") != std::string::npos);
      testAssert(line.find("output_c=384:37") != std::string::npos);
    }

    // Case C: empty model — no 3x3 convs. Error handling: print the
    // line with explicit "{}" markers; don't suppress.
    {
      std::map<int,int> empty;
      std::string line = MLXWinogradTuner::formatConv3x3DistributionLine(0, empty, empty);
      testAssert(line.find("total=0") != std::string::npos);
      testAssert(line.find("input_c={}") != std::string::npos);
      testAssert(line.find("output_c={}") != std::string::npos);
    }
    std::cout << "  conv3x3 distribution formatter OK" << std::endl;
  }

  {
    // planShapeRotation — pure-function tests. Verifies the selection rule
    // (top-3, 3% threshold, 3-rep floor, proportional remainder) directly
    // without any GPU work.

    // Case A: single shape — entire budget on that shape, weight = 1.0.
    {
      auto plan = MLXWinogradTuner::planShapeRotationForTesting({{192, 72}});
      testAssert(plan.size() == 1);
      testAssert(plan[0].channels == 192);
      testAssert(plan[0].measureReps == 19);
      testAssert(std::abs(plan[0].weight - 1.0) < 1e-9);
    }

    // Case B: two shapes both above threshold (b18c384nbt-like, after the
    // 22:1 entry has already been dropped by threshold). Expected:
    // work = 192*72, 128*5 = 13824, 640; weights 0.956, 0.044;
    // round(0.956*19)=18, round(0.044*19)=1; floor bumps 1->3; dominant 18-2=16.
    {
      auto plan = MLXWinogradTuner::planShapeRotationForTesting({{192, 72}, {128, 5}});
      testAssert(plan.size() == 2);
      testAssert(plan[0].channels == 192);
      testAssert(plan[1].channels == 128);
      testAssert(plan[0].measureReps == 16);
      testAssert(plan[1].measureReps == 3);
      testAssert(plan[0].measureReps + plan[1].measureReps == 19);
      testAssert(std::abs(plan[0].weight + plan[1].weight - 1.0) < 1e-9);
      testAssert(plan[0].weight > plan[1].weight);
    }

    // Case C: minor shape below 3% threshold — dropped entirely, dominant
    // absorbs all 19 reps. Histogram: 192:72 (work 13824, 95.5%), 22:1 (work 22, 0.15%).
    {
      auto plan = MLXWinogradTuner::planShapeRotationForTesting({{192, 72}, {22, 1}});
      testAssert(plan.size() == 1);
      testAssert(plan[0].channels == 192);
      testAssert(plan[0].measureReps == 19);
      testAssert(std::abs(plan[0].weight - 1.0) < 1e-9);
    }

    // Case D: four shapes — top-3 cut drops the 4th, then threshold drops
    // one more. Input: 384:60, 192:8, 128:5, 64:5. After top-3: drop 64:5.
    // work remaining = 23040, 1536, 640; total 25216; 128's share = 2.54% < 3%
    // -> drop 128. Final: 384 (93.75%) + 192 (6.25%). reps: round(0.9375*19)=18,
    // round(0.0625*19)=1; floor bumps 1->3; dominant 18-2=16.
    {
      auto plan = MLXWinogradTuner::planShapeRotationForTesting(
          {{384, 60}, {192, 8}, {128, 5}, {64, 5}});
      testAssert(plan.size() == 2);
      testAssert(plan[0].channels == 384);
      testAssert(plan[1].channels == 192);
      testAssert(plan[0].measureReps == 16);
      testAssert(plan[1].measureReps == 3);
    }

    // Case E: three shapes all above threshold. Input: 200:10, 100:10, 50:10.
    // work = 2000, 1000, 500; total 3500; shares 57.1%, 28.6%, 14.3% (all >3%).
    // reps: round(0.571*19)=11, round(0.286*19)=5, round(0.143*19)=3.
    // Sum = 19 exactly (no rounding repair needed). All >= floor of 3.
    {
      auto plan = MLXWinogradTuner::planShapeRotationForTesting(
          {{200, 10}, {100, 10}, {50, 10}});
      testAssert(plan.size() == 3);
      testAssert(plan[0].channels == 200);
      testAssert(plan[1].channels == 100);
      testAssert(plan[2].channels == 50);
      int total = plan[0].measureReps + plan[1].measureReps + plan[2].measureReps;
      testAssert(total == 19);
      testAssert(plan[2].measureReps >= 3);
      testAssert(plan[0].measureReps >= plan[1].measureReps);
      testAssert(plan[1].measureReps >= plan[2].measureReps);
    }

    // Case F: 2 shapes with equal work and complementary 0.5 shares —
    // exercises the rounding-repair branch. Input: 200:1, 100:2 (work
    // 200, 200; tied; tie-break by larger C → plan[0]=C=200). Each
    // share is 0.5; lround(0.5*19) = lround(9.5) = 10 each (lround
    // rounds halves away from zero); pre-repair sum = 20; repair:
    // dominant absorbs delta = 19 - 20 = -1; final (9, 10). Both
    // measureReps stay ≥ kRepFloor=3 so floor-bump is a no-op.
    {
      auto plan = MLXWinogradTuner::planShapeRotationForTesting(
          {{200, 1}, {100, 2}});
      testAssert(plan.size() == 2);
      testAssert(plan[0].channels == 200);
      testAssert(plan[1].channels == 100);
      testAssert(plan[0].measureReps + plan[1].measureReps == 19);
      testAssert(plan[0].measureReps == 9);
      testAssert(plan[1].measureReps == 10);
      testAssert(plan[0].measureReps >= 3);
      testAssert(plan[1].measureReps >= 3);
    }

    std::cout << "  planShapeRotation OK" << std::endl;
  }

  {
    // buildConv3x3HistogramsFromConvs — pure-function test on the conv
    // filter+histogram. Constructs ConvLayerDesc instances directly
    // (ConvLayerDesc is default-constructible but has a deleted copy ctor;
    // see desc.h), so we build the descriptors in a deque (stable addresses,
    // no copies on growth) and pass pointers to the helper. Does not touch
    // ModelDesc.

    auto initConv = [](ConvLayerDesc& c, int kY, int kX, int inC, int outC) {
      c.convYSize  = kY;
      c.convXSize  = kX;
      c.inChannels = inC;
      c.outChannels = outC;
    };

    // Four layers: only the two 3x3 layers should contribute.
    std::deque<ConvLayerDesc> storage;
    std::vector<const ConvLayerDesc*> convs;
    storage.emplace_back(); initConv(storage.back(), 1, 1, 10, 10); convs.push_back(&storage.back());  // 1x1 — filtered
    storage.emplace_back(); initConv(storage.back(), 3, 3, 20, 30); convs.push_back(&storage.back());  // input_c[20]++, output_c[30]++
    storage.emplace_back(); initConv(storage.back(), 3, 3, 30, 30); convs.push_back(&storage.back());  // input_c[30]++, output_c[30]++
    storage.emplace_back(); initConv(storage.back(), 5, 5, 40, 40); convs.push_back(&storage.back());  // 5x5 — filtered

    auto [inHist, outHist] =
        MLXWinogradTuner::buildConv3x3HistogramsFromConvsForTesting(convs);

    // Convert to maps for order-independent comparison.
    std::map<int,int> inMap(inHist.begin(), inHist.end());
    std::map<int,int> outMap(outHist.begin(), outHist.end());

    testAssert(inMap.size() == 2);
    testAssert(inMap[20] == 1);
    testAssert(inMap[30] == 1);
    testAssert(inMap.count(10) == 0);  // 1x1 didn't leak through
    testAssert(inMap.count(40) == 0);  // 5x5 didn't leak through

    testAssert(outMap.size() == 1);
    testAssert(outMap[30] == 2);
    testAssert(outMap.count(10) == 0);
    testAssert(outMap.count(40) == 0);

    // Asymmetric 3x3 (e.g. 3x1) must also be filtered — the kernel is
    // strictly square-3.
    std::deque<ConvLayerDesc> asymStorage;
    std::vector<const ConvLayerDesc*> asym;
    asymStorage.emplace_back(); initConv(asymStorage.back(), 3, 1, 16, 16); asym.push_back(&asymStorage.back());
    asymStorage.emplace_back(); initConv(asymStorage.back(), 1, 3, 16, 16); asym.push_back(&asymStorage.back());
    asymStorage.emplace_back(); initConv(asymStorage.back(), 3, 3, 16, 16); asym.push_back(&asymStorage.back());
    auto [inA, outA] =
        MLXWinogradTuner::buildConv3x3HistogramsFromConvsForTesting(asym);
    testAssert(inA.size() == 1 && inA[0].first == 16 && inA[0].second == 1);
    testAssert(outA.size() == 1 && outA[0].first == 16 && outA[0].second == 1);

    // Empty input → empty histograms (no assert; this is just the pure
    // core. The mlxbackend.cpp call site asserts non-empty after a real
    // model walk; mlxbackend.cpp pre-computes the histogram at model
    // load and stores it on ModelInfoForTuning so the tuner does not
    // re-walk the descriptor).
    std::vector<const ConvLayerDesc*> empty;
    auto [inE, outE] =
        MLXWinogradTuner::buildConv3x3HistogramsFromConvsForTesting(empty);
    testAssert(inE.empty());
    testAssert(outE.empty());

    std::cout << "  buildConv3x3HistogramsFromConvs OK" << std::endl;
  }

  // ---- v3 round-trip: tg0/tg1/wpt/vw/gridOrder (input), tg0/tg1/wpt (output) ----
  {
    // v3 roundtrip: write -> load -> compare all 8 fields. Two
    // cases for input gridOrder: Cfast and Tfast. (Tfast forces vw=1 per
    // isValid invariant.)
    using namespace MLXWinograd;
    for(auto inGo : {GridOrder::Cfast, GridOrder::Tfast}) {
      MLXWinogradTuneParams p;
      p.inputTransform.tg0 = 32;
      p.inputTransform.tg1 = 1;
      p.inputTransform.wpt = 2;
      p.inputTransform.vw  = (inGo == GridOrder::Cfast) ? 2 : 1;
      p.inputTransform.gridOrder = inGo;
      p.outputUntransform.tg0 = 32;
      p.outputUntransform.tg1 = 8;
      p.outputUntransform.wpt = 1;
      testAssert(p.isValid());

      std::string tmpFile = "/tmp/katago_mlx_winotuner_v3_roundtrip_" + std::to_string((int)inGo) + ".txt";
      MLXWinogradTuneParams::save(tmpFile, p);
      MLXWinogradTuneParams q = MLXWinogradTuneParams::load(tmpFile);
      testAssert(q.inputTransform.tg0 == p.inputTransform.tg0);
      testAssert(q.inputTransform.tg1 == p.inputTransform.tg1);
      testAssert(q.inputTransform.wpt == p.inputTransform.wpt);
      testAssert(q.inputTransform.vw  == p.inputTransform.vw);
      testAssert(q.inputTransform.gridOrder == p.inputTransform.gridOrder);
      testAssert(q.outputUntransform.tg0 == p.outputUntransform.tg0);
      testAssert(q.outputUntransform.tg1 == p.outputUntransform.tg1);
      testAssert(q.outputUntransform.wpt == p.outputUntransform.wpt);
      testAssert(q.isValid());
      std::remove(tmpFile.c_str());
    }
    cout << "  v3 roundtrip (Cfast + Tfast) OK" << endl;
  }

  // dtype-aware cache filenames must coexist in the same directory
  // without collision. Verify defaultFileName gains a _fp16/_fp32 suffix.
  {
    std::string nameF32 = MLXWinogradTuner::defaultFileName(
      "AppleSilicon", 19, 19, 384, 13, /*useFP16=*/false);
    std::string nameF16 = MLXWinogradTuner::defaultFileName(
      "AppleSilicon", 19, 19, 384, 13, /*useFP16=*/true);
    testAssert(nameF32 != nameF16);
    testAssert(nameF32.find("_fp32") != std::string::npos);
    testAssert(nameF16.find("_fp16") != std::string::npos);
    testAssert(nameF32.size() >= 4 && nameF32.substr(nameF32.size()-4) == ".txt");
    testAssert(nameF16.size() >= 4 && nameF16.substr(nameF16.size()-4) == ".txt");
    cout << "  defaultFileName dtype suffix OK: "
         << nameF32 << " vs " << nameF16 << endl;
  }

  // detectGpuName() must yield a stable, non-empty, chip-specific cache key so
  // that different Apple chips don't share one Winograd cache file, and must be
  // filesystem-safe once threaded through defaultFileName (no spaces).
  {
    std::string gpu = MLXWinogradTuner::detectGpuName();
    testAssert(!gpu.empty());
    // Deterministic: the backend load path and the `tuner` command must derive
    // the identical key on the same machine.
    testAssert(MLXWinogradTuner::detectGpuName() == gpu);
    std::string name = MLXWinogradTuner::defaultFileName(gpu, 19, 19, 384, 13, /*useFP16=*/true);
    testAssert(name.find(' ') == std::string::npos);
    testAssert(name.find("_gpu") != std::string::npos);
    testAssert(name.size() >= 4 && name.substr(name.size()-4) == ".txt");
    cout << "  detectGpuName OK: \"" << gpu << "\" -> " << name << endl;
  }

  // ---- Corrupt-version rejection ----
  {
    std::string tmp = "/tmp/katago_mlx_winotuner_badversion.txt";
    {
      std::ofstream f(tmp);
      f << "VERSION=999\n#inputTransform\ntg0=32 tg1=1\n#outputUntransform\ntg0=32 tg1=1\n";
    }
    bool threw = false;
    try { (void)MLXWinogradTuneParams::load(tmp); }
    catch(const IOError&) { threw = true; }
    testAssert(threw);
  }

  // ---- v3 isValid invariants ----
  {
    // v3 isValid invariants.
    using namespace MLXWinograd;
    auto basePass = [&]() {
      MLXWinogradTuneParams p;
      p.inputTransform = {32, 1, 1, 2, GridOrder::Cfast};
      p.outputUntransform = {32, 2, 1};
      return p;
    };

    // Baseline passes.
    testAssert(basePass().isValid());

    // tg0 <= 0 fails.
    { auto p = basePass(); p.inputTransform.tg0 = 0;  testAssert(!p.isValid()); }
    { auto p = basePass(); p.outputUntransform.tg0 = -1; testAssert(!p.isValid()); }

    // tg0 * tg1 > 1024 fails.
    { auto p = basePass(); p.inputTransform.tg0 = 64; p.inputTransform.tg1 = 32;
      testAssert(!p.isValid()); }

    // wpt < 1 fails.
    { auto p = basePass(); p.inputTransform.wpt = 0;  testAssert(!p.isValid()); }
    { auto p = basePass(); p.outputUntransform.wpt = 0; testAssert(!p.isValid()); }

    // vw < 1 fails on input.
    { auto p = basePass(); p.inputTransform.vw = 0;   testAssert(!p.isValid()); }

    // Tfast on input forces vw=1.
    { auto p = basePass();
      p.inputTransform.gridOrder = GridOrder::Tfast;
      p.inputTransform.vw = 2;
      testAssert(!p.isValid()); }
    { auto p = basePass();
      p.inputTransform.gridOrder = GridOrder::Tfast;
      p.inputTransform.vw = 1;
      testAssert(p.isValid()); }

    cout << "  v3 isValid invariants OK" << endl;
  }

  // Candidate enumeration with validity filtering.
  {
    using namespace MLXWinograd;
    // Cfast, C=64 (divisible by all vw): full Cartesian product over all axes
    // minus tg0*tg1>1024.
    auto cands = MLXWinogradTuner::buildInputCandidatesForTesting(
        /*full*/true, /*C*/64, /*Ntiles*/200, GridOrder::Cfast);

    // Sanity: returns hundreds of valid configs.
    testAssert(cands.size() > 100);
    testAssert(cands.size() < 5000);   // bounded by validity filter

    // All candidates satisfy tg0*tg1 <= 1024.
    for(const auto& c : cands)
      testAssert(c.tg0 * c.tg1 <= 1024);

    // C=66 with vw>1: should filter out vw=2 (66%2=0 — VW=2 allowed)
    // and vw=4 (66%4=2 != 0 — VW=4 should NOT appear in candidates).
    auto cands_C66 = MLXWinogradTuner::buildInputCandidatesForTesting(
        true, /*C*/66, /*Ntiles*/200, GridOrder::Cfast);
    for(const auto& c : cands_C66) {
      if(c.vw == 4)
        testAssert(false);  // vw=4 candidate should have been filtered out for C=66
    }

    // Tfast: vw must be 1 (kernel static_assert). All Tfast candidates have vw=1.
    auto cands_Tfast = MLXWinogradTuner::buildInputCandidatesForTesting(
        true, 64, 200, GridOrder::Tfast);
    for(const auto& c : cands_Tfast) {
      testAssert(c.vw == 1);
      testAssert(c.gridOrder == GridOrder::Tfast);
    }

    // Output side: same shape of assertions. (gridOrder is not a parameter
    // of buildOutputCandidatesForTesting — output is Cfast-only.)
    auto out_cands = MLXWinogradTuner::buildOutputCandidatesForTesting(
        true, /*outC*/64, /*Ntiles*/200);
    testAssert(out_cands.size() > 100);
    for(const auto& c : out_cands)
      testAssert(c.tg0 * c.tg1 <= 1024);

    std::cout << "  MLX Winograd candidate enumeration validity passed ("
              << cands.size() << " input / " << out_cands.size() << " output candidates C=64)"
              << std::endl;
  }

  // ---- Measurement primitives return finite positive times ----
  // We can't call the static helpers from the test, so we use the public
  // surface: loadOrAutoTune with reTune=true runs the search and we verify
  // that the public schema struct works with valid configs. The measurement
  // primitive itself is exercised by the search-works test below.

  {
    // Gated flat-sweep convergence test.
    // Runs the production flat sweep on a small synthetic problem and asserts
    // that the winner is isValid and that its timing is no worse than the
    // baked default (tg0=32, tg1=1, wpt=1, vw=1, Cfast).
    const char* gate = std::getenv("KATAGO_MLX_WINOTUNER_RUN_SWEEP_TEST");
    if(gate != nullptr && std::string(gate) == "1") {
      MLXWinogradTuner::ModelInfoForTuning mi;
      mi.trunkNumChannels    = 64;
      mi.modelVersion        = 11;
      // Synthetic single-shape histogram for the toy C=64 test model.
      mi.conv3x3InputHistogram  = {{64, 1}};
      mi.conv3x3OutputHistogram = {{64, 1}};

      // loadOrAutoTune rewrites an empty tunerFile to a default cache path,
      // so use an explicit temp path and remove it after to avoid touching
      // the user's cache directory.
      std::string tmpTunerFile = "/tmp/katago_mlx_winotuner_sweep_cache.txt";
      std::remove(tmpTunerFile.c_str());

      MLXWinogradTuneParams tuned = MLXWinogradTuner::loadOrAutoTune(
          /*tunerFile=*/tmpTunerFile,
          /*homeDataDirOverride=*/"",
          /*gpuName=*/"AppleSilicon",
          /*nnXLen=*/19, /*nnYLen=*/19, /*batchSize=*/1,
          mi,
          /*logger=*/nullptr,
          /*full=*/false,
          /*reTune=*/true,
          /*useFP16=*/true);
      testAssert(tuned.isValid());

      // Score the baked default and the tuned winner via scoreInputTransform.
      // tuned.time <= baked.time (within noise).
      MLXWinograd::InputTransform baked{};
      baked.tg0 = 32; baked.tg1 = 1; baked.wpt = 1; baked.vw = 1;
      baked.gridOrder = MLXWinograd::GridOrder::Cfast;
      auto bestOf5 = [&](const MLXWinograd::InputTransform& cfg) -> double {
        double best = std::numeric_limits<double>::infinity();
        for(int rep = 0; rep < 5; rep++) {
          double t = MLXWinogradTuner::scoreInputTransformForTesting(
              cfg, 1, 19, 19, mi, true);
          if(t < best) best = t;
        }
        return best;
      };
      double bakedMs = bestOf5(baked);
      double tunedMs = bestOf5(tuned.inputTransform);
      // The sweep seeds the default as its floor and only replaces it with a
      // strictly-faster candidate, so the winner is <= default in the sweep's
      // OWN measurement by construction. Here we re-measure both (bestOf5) as
      // an independent sanity check that the winner isn't grossly worse. On the
      // coarse auto grid the winner on this toy single-shape problem is often
      // within noise of the default, so the two bestOf5 values are two noisy
      // sub-millisecond samples of near-tied configs (each carries the ~10-15%
      // dispatch/sync-overhead noise floor — the same noise behind the tuner's
      // plateau). A 1.10 bound flips intermittently on both this and the
      // pre-change binary (observed ratios to ~1.06 even on the wide grid); a
      // 1.30 bound absorbs that while still catching a winner that is genuinely
      // much slower than default (which would signal a real sweep/scoring bug).
      testAssert(tunedMs <= bakedMs * 1.30);
      std::cout << "  flat-sweep convergence (gated) OK"
                << " bakedMs=" << bakedMs
                << " tunedMs=" << tunedMs << std::endl;

      std::remove(tmpTunerFile.c_str());
    }
  }

  {
    // Baseline anchor — Test 1: log-format gated check (input stage).
    // Asserts that flatSweepInput's log line carries the new baseline_ms and
    // delta_pct fields with the documented format. Gated because the synthetic
    // sweep takes a few seconds; opt in with the env var below.
    const char* gate = std::getenv("KATAGO_MLX_WINOTUNER_RUN_LOG_FORMAT_TEST");
    if(gate != nullptr && std::string(gate) == "1") {
      MLXWinogradTuner::ModelInfoForTuning mi;
      mi.trunkNumChannels    = 64;
      mi.modelVersion        = 11;
      // Synthetic single-shape histogram for the toy C=64 test model.
      mi.conv3x3InputHistogram  = {{64, 1}};
      mi.conv3x3OutputHistogram = {{64, 1}};

      std::string tmpTunerFile = "/tmp/baseline_anchor_log_format.txt";
      std::remove(tmpTunerFile.c_str());

      std::ostringstream captured;
      Logger logger(nullptr, /*logToStdoutDefault=*/false,
                    /*logToStderrDefault=*/false, /*logTimeDefault=*/false,
                    /*logConfigContents=*/false);
      logger.addOStream(captured);

      (void)MLXWinogradTuner::loadOrAutoTune(
          /*tunerFile=*/tmpTunerFile,
          /*homeDataDirOverride=*/"",
          /*gpuName=*/"AppleSilicon",
          /*nnXLen=*/19, /*nnYLen=*/19, /*batchSize=*/1,
          mi,
          /*logger=*/&logger,
          /*full=*/false,
          /*reTune=*/true,
          /*useFP16=*/true);

      const std::string log = captured.str();
      // Logger::writeLocked prefixes each line with ": " when logTime=false, so
      // `log` reads ": MLX tuner ...". std::regex_search is anchor-free so the
      // ": " prefix is transparent; only the substring match matters here.
      // The regex matches the non-degenerate path only (best != nullopt). The
      // best=none / delta_pct=nan branch is unreachable for the synthetic 19x19
      // C=64 problem this test runs against (hundreds of valid candidates).
      // Updated for shape diagnostic: regex now requires the per-shape
      // median fields appended by flatSweepInput.
      std::regex inputRe(
          R"(MLX tuner flatSweepInput: considered=[0-9]+ best=tg0=[0-9]+ tg1=[0-9]+ wpt=[0-9]+ vw=[0-9]+ gridOrder=[01] time_ms=[0-9]+\.[0-9]+ baseline_ms=[0-9]+\.[0-9]+ delta_pct=[-+][0-9]+\.[0-9]+ shape_ms=c[0-9]+:[0-9]+\.[0-9]+(?:,c[0-9]+:[0-9]+\.[0-9]+)*)");
      testAssert(std::regex_search(log, inputRe));
      std::cout << "  flatSweepInput log-format (gated) OK" << std::endl;

      std::regex outputRe(
          R"(MLX tuner flatSweepOutput: considered=[0-9]+ best=tg0=[0-9]+ tg1=[0-9]+ wpt=[0-9]+ time_ms=[0-9]+\.[0-9]+ baseline_ms=[0-9]+\.[0-9]+ delta_pct=[-+][0-9]+\.[0-9]+ shape_ms=c[0-9]+:[0-9]+\.[0-9]+(?:,c[0-9]+:[0-9]+\.[0-9]+)*)");
      testAssert(std::regex_search(log, outputRe));
      std::cout << "  flatSweepOutput log-format (gated) OK" << std::endl;

      std::remove(tmpTunerFile.c_str());
    }
  }

  {
    // Baseline anchor — Test 2: baseline-consistency gated check.
    // Asserts that the baseline_ms value printed by flatSweepInput is in the
    // same ballpark as an independent re-score of the default-constructed
    // InputTransform — a gross-error sanity check (catches a ~2x units/logic
    // bug in the logged baseline), NOT a precision check.
    //
    // parsedBaseline is a SINGLE 20-rep weighted mean (one call into
    // scoreInputTransform, logged by the sweep). minOf3 is the min of three
    // such weighted means — biased low by min-selection (~5-10%), while
    // parsedBaseline stays a high-variance single sample. Both carry the ~10%
    // per-sample noise floor of these sub-millisecond kernels (steady_clock
    // around one dispatch includes fixed dispatch/sync overhead — the same
    // noise that makes the tuner's own winner a draw from a plateau). The gap
    // is therefore positively skewed: across runs the same-config relErr
    // clusters <0.15 but tails to ~0.3 and occasionally past 0.4. A single
    // sample cannot support a tight bound, so we use the same 0.50 budget as
    // the sibling per-shape check below; it still flags a gross measurement
    // bug. Tight precision belongs in same-config bit-for-bit tests, not here.
    //
    // Reuses the KATAGO_MLX_WINOTUNER_RUN_SWEEP_TEST gate so users who
    // opt into the sweep-convergence cost also get this check. Note
    // this runs an INDEPENDENT loadOrAutoTune sweep — total cost when
    // the gate is set is roughly 2x the cost of a single sweep.
    //
    // Coverage scope: input stage only. flatSweepOutput's baseline_ms
    // is format-checked by Test 1 but not consistency-checked here.
    // The output kernel uses a different scoring function and default
    // struct (OutputUntransform{}); a symmetric check is deferred.
    const char* gate = std::getenv("KATAGO_MLX_WINOTUNER_RUN_SWEEP_TEST");
    if(gate != nullptr && std::string(gate) == "1") {
      MLXWinogradTuner::ModelInfoForTuning mi;
      mi.trunkNumChannels    = 64;
      mi.modelVersion        = 11;
      // Synthetic single-shape histogram for the toy C=64 test model.
      mi.conv3x3InputHistogram  = {{64, 1}};
      mi.conv3x3OutputHistogram = {{64, 1}};

      std::string tmpTunerFile = "/tmp/baseline_anchor_consistency.txt";
      std::remove(tmpTunerFile.c_str());

      std::ostringstream captured;
      Logger logger(nullptr, /*logToStdoutDefault=*/false,
                    /*logToStderrDefault=*/false, /*logTimeDefault=*/false,
                    /*logConfigContents=*/false);
      logger.addOStream(captured);

      (void)MLXWinogradTuner::loadOrAutoTune(
          /*tunerFile=*/tmpTunerFile,
          /*homeDataDirOverride=*/"",
          /*gpuName=*/"AppleSilicon",
          /*nnXLen=*/19, /*nnYLen=*/19, /*batchSize=*/1,
          mi,
          /*logger=*/&logger,
          /*full=*/false,
          /*reTune=*/true,
          /*useFP16=*/true);

      const std::string log = captured.str();
      std::smatch m;
      std::regex baselineRe(R"(flatSweepInput:[^\n]*baseline_ms=([0-9]+\.[0-9]+))");
      testAssert(std::regex_search(log, m, baselineRe));
      const double parsedBaseline = std::stod(m[1].str());

      double minOf3 = std::numeric_limits<double>::infinity();
      for(int rep = 0; rep < 3; rep++) {
        double t = MLXWinogradTuner::scoreInputTransformForTesting(
            MLXWinograd::InputTransform{}, 1, 19, 19, mi, true);
        if(t < minOf3) minOf3 = t;
      }

      const double relErr = std::abs(parsedBaseline - minOf3) / minOf3;
      testAssert(relErr < 0.50);
      std::cout << "  baseline-consistency (gated) OK"
                << " parsed=" << parsedBaseline
                << " minOf3=" << minOf3
                << " relErr=" << relErr << std::endl;

      std::remove(tmpTunerFile.c_str());
    }
  }

  {
    // Per-shape numeric consistency — Test 2 from the shape-diagnostic spec.
    // Asserts the dominant-shape median printed by flatSweepInput
    // (shape_ms=c<C>:<ms>) is in the same ballpark as an independent
    // reference measurement of the default InputTransform{} on that shape.
    //
    // IMPORTANT — cross-config comparison: parsedDominantMs is measured by
    // flatSweepInput on the WINNER configuration (whatever the sweep
    // selected). minOf3 is computed on the DEFAULT InputTransform{} via
    // three independent scoreInputTransformPerShapeForTesting calls. These
    // are not the same config, so the relative-error budget is necessarily
    // loose. The budget covers:
    //   - winner-vs-default speed gap (sweep can find configs 10-40%
    //     faster than default on some shapes/hardware)
    //   - selection bias on the min-of-3 reference (~5-10% low vs single)
    //   - per-call noise floor (~10%)
    // The 50% budget is intentionally conservative; this is a sanity-check
    // that measurement is roughly working, not a tight precision check.
    // Tighter precision checks belong in same-config stability tests.
    //
    // Coverage scope: input stage only. flatSweepOutput's per-shape fields
    // are format-checked by the log-format test (gate
    // KATAGO_MLX_WINOTUNER_RUN_LOG_FORMAT_TEST) but not consistency-
    // checked here — symmetric output check is deferred.
    //
    // Gate is new (KATAGO_MLX_WINOTUNER_RUN_PER_SHAPE_TEST) and separate
    // from the baseline-anchor gate above; this test runs an additional
    // tuner sweep.
    const char* gate = std::getenv("KATAGO_MLX_WINOTUNER_RUN_PER_SHAPE_TEST");
    if(gate != nullptr && std::string(gate) == "1") {
      MLXWinogradTuner::ModelInfoForTuning mi;
      mi.trunkNumChannels    = 64;
      mi.modelVersion        = 11;
      // Synthetic single-shape histogram for the toy C=64 test model.
      mi.conv3x3InputHistogram  = {{64, 1}};
      mi.conv3x3OutputHistogram = {{64, 1}};

      std::string tmpTunerFile = "/tmp/per_shape_consistency.txt";
      std::remove(tmpTunerFile.c_str());

      std::ostringstream captured;
      Logger logger(nullptr, /*logToStdoutDefault=*/false,
                    /*logToStderrDefault=*/false, /*logTimeDefault=*/false,
                    /*logConfigContents=*/false);
      logger.addOStream(captured);

      (void)MLXWinogradTuner::loadOrAutoTune(
          /*tunerFile=*/tmpTunerFile,
          /*homeDataDirOverride=*/"",
          /*gpuName=*/"AppleSilicon",
          /*nnXLen=*/19, /*nnYLen=*/19, /*batchSize=*/1,
          mi,
          /*logger=*/&logger,
          /*full=*/false,
          /*reTune=*/true,
          /*useFP16=*/true);

      const std::string log = captured.str();
      std::smatch m;
      std::regex trunkRe(R"(flatSweepInput:[^\n]*shape_ms=c[0-9]+:([0-9]+\.[0-9]+))");
      testAssert(std::regex_search(log, m, trunkRe));
      const double parsedDominantMs = std::stod(m[1].str());

      // Per-shape consistency: parse the dominant shape's median from
      // the flatSweepInput log line (which used scoreInputTransformPerShape
      // on the winner) and compare against scoreInputTransformPerShapeForTesting
      // on the default InputTransform. Cross-config (winner vs default)
      // so a wide relErr bound (<0.50) is appropriate.
      std::vector<std::pair<int,double>> r1 =
          MLXWinogradTuner::scoreInputTransformPerShapeForTesting(
              MLXWinograd::InputTransform{}, 1, 19, 19, mi, true);
      std::vector<std::pair<int,double>> r2 =
          MLXWinogradTuner::scoreInputTransformPerShapeForTesting(
              MLXWinograd::InputTransform{}, 1, 19, 19, mi, true);
      std::vector<std::pair<int,double>> r3 =
          MLXWinogradTuner::scoreInputTransformPerShapeForTesting(
              MLXWinograd::InputTransform{}, 1, 19, 19, mi, true);
      testAssert(!r1.empty() && !r2.empty() && !r3.empty());
      // Each result has the same shapes in the same order; take the
      // dominant (index 0) per-shape median across the 3 runs.
      double minOf3 = std::min({r1[0].second, r2[0].second, r3[0].second});

      const double relErr = std::abs(parsedDominantMs - minOf3) / minOf3;
      // 50% budget — see comment block above for rationale on the loose
      // bound (cross-config comparison + selection bias + noise).
      testAssert(relErr < 0.50);
      std::cout << "  per-shape dominant consistency (gated) OK"
                << " parsed=" << parsedDominantMs
                << " minOf3=" << minOf3
                << " relErr=" << relErr << std::endl;

      std::remove(tmpTunerFile.c_str());
    }
  }

  {
    // Per-shape scoring smoke test: verify that scoreInputTransformPerShape
    // and scoreOutputUntransformPerShape return finite positive values for
    // each planned shape with a default-constructed
    // InputTransform/OutputUntransform on a tiny shape. Gated under the same
    // env var as the other GPU-touching tests; ungated CI shouldn't pay for
    // GPU work.
    const char* gate = std::getenv("KATAGO_MLX_WINOTUNER_RUN_SWEEP_TEST");
    if(gate != nullptr && std::string(gate) == "1") {
      MLXWinogradTuner::ModelInfoForTuning mi;
      mi.trunkNumChannels    = 64;
      mi.modelVersion        = 11;
      // Synthetic single-shape histogram for the toy C=64 test model.
      mi.conv3x3InputHistogram  = {{64, 1}};
      mi.conv3x3OutputHistogram = {{64, 1}};

      std::vector<std::pair<int,double>> in =
          MLXWinogradTuner::scoreInputTransformPerShapeForTesting(
              MLXWinograd::InputTransform{}, 1, 19, 19, mi, true);
      testAssert(!in.empty());
      for(const auto& [c, v] : in) {
        testAssert(c > 0);
        testAssert(std::isfinite(v));
        testAssert(v > 0.0);
        testAssert(v < 1000.0);  // sanity: <1s per call on Apple Silicon
      }

      std::vector<std::pair<int,double>> out =
          MLXWinogradTuner::scoreOutputUntransformPerShapeForTesting(
              MLXWinograd::OutputUntransform{}, 1, 19, 19, mi, true);
      testAssert(!out.empty());
      for(const auto& [c, v] : out) {
        testAssert(c > 0);
        testAssert(std::isfinite(v));
        testAssert(v > 0.0);
        testAssert(v < 1000.0);
      }
      std::cout << "  per-shape scoring smoke (gated) OK"
                << " in[0]=c" << in[0].first << ":" << in[0].second
                << " out[0]=c" << out[0].first << ":" << out[0].second
                << std::endl;
    }
  }

  cout << "MLX Winograd tuner tests passed" << endl;
}

#endif // USE_MLX_BACKEND
