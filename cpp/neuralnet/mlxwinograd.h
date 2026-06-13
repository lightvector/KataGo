#ifndef NEURALNET_MLXWINOGRAD_H_
#define NEURALNET_MLXWINOGRAD_H_

#ifdef USE_MLX_BACKEND

#include <vector>

namespace MLXWinograd {

enum class GridOrder    : int { Cfast = 0, Tfast = 1 };

// Per-stage launch-geometry configs. Input transform exposes
// (tg0, tg1, wpt, vw, gridOrder); output untransform exposes (tg0, tg1, wpt).
// The output kernel is monomorphic on VW=1, GRID_ORDER=Cfast, and the
// matmul layout is monomorphic on Std for both stages.
struct InputTransform {
  int tg0 = 32;
  int tg1 = 1;
  int wpt = 1;            // tiles per thread; {1, 2, 4, 8}
  int vw  = 1;            // vector width; {1, 2, 4}
  GridOrder gridOrder = GridOrder::Cfast;
};
struct OutputUntransform {
  int tg0 = 32;
  int tg1 = 1;
  int wpt = 1;
};

// F(2,3) 1D transform matrices.
inline constexpr float BT[4][4] = {
  {1.f, 0.f,-1.f, 0.f},
  {0.f, 1.f, 1.f, 0.f},
  {0.f,-1.f, 1.f, 0.f},
  {0.f, 1.f, 0.f,-1.f}
};
inline constexpr float G[4][3] = {
  {1.f, 0.f, 0.f},
  {0.5f,0.5f,0.5f},
  {0.5f,-0.5f,0.5f},
  {0.f, 0.f, 1.f}
};
inline constexpr float AT[2][4] = {
  {1.f, 1.f, 1.f, 0.f},
  {0.f, 1.f,-1.f,-1.f}
};

// Transform one 3x3 filter g -> 4x4 U = G g G^T.
inline void transformWeight(const float g[3][3], float U[4][4]) {
  float Gg[4][3];
  for(int i=0;i<4;i++) for(int j=0;j<3;j++) {
    float s=0.f; for(int k=0;k<3;k++) s += G[i][k]*g[k][j]; Gg[i][j]=s;
  }
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
    float s=0.f; for(int k=0;k<3;k++) s += Gg[i][k]*G[j][k]; U[i][j]=s;
  }
}

// Transform one 4x4 input tile d -> 4x4 V = B^T d B.
inline void transformInput(const float d[4][4], float V[4][4]) {
  float Bd[4][4];
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
    float s=0.f; for(int k=0;k<4;k++) s += BT[i][k]*d[k][j]; Bd[i][j]=s;
  }
  for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
    float s=0.f; for(int k=0;k<4;k++) s += Bd[i][k]*BT[j][k]; V[i][j]=s;
  }
}

// Inverse transform 4x4 M -> 2x2 Y = A^T M A.
inline void transformOutput(const float M[4][4], float Y[2][2]) {
  float AM[2][4];
  for(int i=0;i<2;i++) for(int j=0;j<4;j++) {
    float s=0.f; for(int k=0;k<4;k++) s += AT[i][k]*M[k][j]; AM[i][j]=s;
  }
  for(int i=0;i<2;i++) for(int j=0;j<2;j++) {
    float s=0.f; for(int k=0;k<4;k++) s += AM[i][k]*AT[j][k]; Y[i][j]=s;
  }
}

// Full CPU reference NHWC Winograd F(2,3) "same" conv, stride 1.
// in: [N][H][W][Cin], weights OIHW flattened [Cout][Cin][3][3], out: [N][H][W][Cout].
inline std::vector<float> cpuConv2d3x3(
  const std::vector<float>& in, int N, int H, int W, int Cin,
  const std::vector<float>& wOIHW, int Cout
) {
  std::vector<float> out((size_t)N*H*W*Cout, 0.f);
  // Precompute U per (oc,ic).
  std::vector<float> U((size_t)Cout*Cin*16);
  for(int oc=0;oc<Cout;oc++) for(int ic=0;ic<Cin;ic++) {
    float g[3][3];
    for(int a=0;a<3;a++) for(int b=0;b<3;b++)
      g[a][b]=wOIHW[(((size_t)oc*Cin+ic)*3+a)*3+b];
    float Um[4][4]; transformWeight(g,Um);
    for(int a=0;a<4;a++) for(int b=0;b<4;b++)
      U[(((size_t)oc*Cin+ic)*4+a)*4+b]=Um[a][b];
  }
  // Tile over 2x2 output blocks; pad to "same" (pad=1 each side for 3x3).
  for(int n=0;n<N;n++)
  for(int ty=0; ty<H; ty+=2)
  for(int tx=0; tx<W; tx+=2) {
    for(int oc=0; oc<Cout; oc++) {
      float Macc[4][4] = {};
      for(int ic=0; ic<Cin; ic++) {
        float d[4][4];
        for(int a=0;a<4;a++) for(int b=0;b<4;b++) {
          int iy=ty+a-1, ix=tx+b-1; // pad=1
          d[a][b]=(iy>=0&&iy<H&&ix>=0&&ix<W)
            ? in[(((size_t)n*H+iy)*W+ix)*Cin+ic] : 0.f;
        }
        float V[4][4]; transformInput(d,V);
        for(int a=0;a<4;a++) for(int b=0;b<4;b++)
          Macc[a][b]+=U[(((size_t)oc*Cin+ic)*4+a)*4+b]*V[a][b];
      }
      float Y[2][2]; transformOutput(Macc,Y);
      for(int a=0;a<2;a++) for(int b=0;b<2;b++) {
        int oy=ty+a, ox=tx+b;
        if(oy<H&&ox<W) out[(((size_t)n*H+oy)*W+ox)*Cout+oc]=Y[a][b];
      }
    }
  }
  return out;
}

} // namespace MLXWinograd

#include "mlx/mlx.h"
#include "mlx/fast.h"

namespace MLXWinograd {
namespace mx = mlx::core;

// Output-untransform epilogue fused into the store. None keeps the kernel
// byte-identical to the unfused path. BNAct applies (scale*x+bias) then an
// activation in fp32 on the rounded-to-T conv value; Residual adds a same-shape
// T tensor in T arithmetic. Pointers (not values) so the default is trivially
// None and callers pass arrays they already hold alive across the synchronous
// winogradConv2d call. act: 0=identity, 1=mish, 2=relu (kernel-local vocab).
struct Epilogue {
  enum Mode { None = 0, BNAct = 1, Residual = 2, BiasBNAct = 3 };
  Mode mode = None;
  const mx::array* scale = nullptr;   // BNAct/BiasBNAct: fp32 [Cout]
  const mx::array* bias  = nullptr;   // BNAct/BiasBNAct: fp32 [Cout]
  int act = 0;                        // BNAct/BiasBNAct: kernel-local activation id
  const mx::array* resid = nullptr;   // Residual: T [N,H,W,Cout]
  const mx::array* gbias = nullptr;   // BiasBNAct: broadcast bias [N,outC], T
  static Epilogue none() { return Epilogue{}; }
  static Epilogue bnAct(const mx::array& s, const mx::array& b, int a) {
    Epilogue e; e.mode = BNAct; e.scale = &s; e.bias = &b; e.act = a; return e;
  }
  static Epilogue residual(const mx::array& r) {
    Epilogue e; e.mode = Residual; e.resid = &r; return e;
  }
  static Epilogue biasBNAct(const mx::array& gb, const mx::array& s, const mx::array& b, int a) {
    Epilogue e; e.mode = BiasBNAct; e.gbias = &gb; e.scale = &s; e.bias = &b; e.act = a; return e;
  }
};

// Host-side weight transform: OIHW [Cout][Cin][3][3] -> U array.
// Layout: [16, Cin, Cout] — Cout fast (matmul sees [16,Ntiles,Cin] x [16,Cin,Cout] -> [16,Ntiles,Cout]).
// Output layout: Std only.
inline mx::array makeWinogradWeights(const std::vector<float>& wOIHW,
                                     int Cout, int Cin,
                                     bool useFP16 = false) {
  std::vector<float> U((size_t)16 * Cin * Cout, 0.0f);
  for(int oc = 0; oc < Cout; oc++) {
    for(int ic = 0; ic < Cin; ic++) {
      float g[3][3];
      for(int a = 0; a < 3; a++)
        for(int b = 0; b < 3; b++)
          g[a][b] = wOIHW[(((size_t)oc * Cin + ic) * 3 + a) * 3 + b];
      float Um[4][4]; transformWeight(g, Um);
      for(int a = 0; a < 4; a++) {
        for(int b = 0; b < 4; b++) {
          // [16, Cin, Cout] — Cout fast
          size_t idx = ((size_t)(a * 4 + b) * Cin + ic) * Cout + oc;
          U[idx] = Um[a][b];
        }
      }
    }
  }
  mx::Shape shape = {16, Cin, Cout};
  mx::array arr(U.data(), shape, mx::float32);
  if(useFP16) {
    mx::array casted = mx::astype(arr, mx::float16);
    // Realize on the constructor thread so the resulting array is a
    // materialized constant. Without this, a model cached and shared
    // across threads carries an unevaluated AsType primitive that is
    // stamped with the constructor thread's stream — calling thread's
    // mx::eval then fails with "There is no Stream(gpu, N) in current
    // thread." for the constructor thread's stream index.
    mx::eval(casted);
    return casted;
  }
  return arr;
}

// F(2,3) input transform kernel: NHWC T input -> [16, Ntiles, C] T output.
// The matmul layout is monomorphic on Std ([16, Ntiles, C]).
// Template args (JIT-substituted via MLX template_args):
//   T              — float or half (precision)
//   WPT            — tiles per thread
//   VW             — vector width for packed loads
//   GRID_ORDER     — 0=Cfast (C is fast axis), 1=Tfast (Ntiles fast)
// Grid:
//   Cfast: (ceil(C/VW), ceil(Ntiles/WPT), 1)
//   Tfast: (Ntiles,     ceil(C/WPT),      1)
inline constexpr const char* kWinoInputSource = R"METAL(
    static_assert(WPT >= 1 && VW >= 1, "WPT and VW must be positive");
    // Tfast (GRID_ORDER=1) does not support VW>1.
    static_assert(GRID_ORDER == 0 || VW == 1, "Tfast (GRID_ORDER=1) requires VW=1");

    int N_k      = inp_shape[0];
    int H_k      = inp_shape[1];
    int W_k      = inp_shape[2];
    int C_k      = inp_shape[3];
    int tilesY_k = (H_k + 1) / 2;
    int tilesX_k = (W_k + 1) / 2;
    int Ntiles_k = N_k * tilesY_k * tilesX_k;

    if (GRID_ORDER == 0) {
      // Cfast: grid x = ceil(C/VW), grid y = ceil(Ntiles/WPT).
      // Each thread owns VW channels (inner vc loop) and WPT tiles (outer w loop).
      uint c_group  = thread_position_in_grid.x;
      uint t_group  = thread_position_in_grid.y;

      for (int w = 0; w < WPT; w++) {
        int tileIdx = (int)t_group * WPT + w;
        if (tileIdx >= Ntiles_k) break;

        int rem = tileIdx;
        int n   = rem / (tilesY_k * tilesX_k); rem -= n * tilesY_k * tilesX_k;
        int ty  = rem / tilesX_k;
        int tx  = rem % tilesX_k;

        for (int vc = 0; vc < VW; vc++) {
          int c = (int)c_group * VW + vc;
          if (c >= C_k) break;
          T d[4][4];
          for (int i = 0; i < 4; i++) {
            int iy = 2 * ty - 1 + i;
            for (int j = 0; j < 4; j++) {
              int ix = 2 * tx - 1 + j;
              if (iy < 0 || iy >= H_k || ix < 0 || ix >= W_k) {
                d[i][j] = (T)0.0f;
              } else {
                d[i][j] = inp[((n * H_k + iy) * W_k + ix) * C_k + c];
              }
            }
          }
          // Transform accumulates in fp32; only the stored V rounds to T (fp16-safe).
          float tmp[4][4];
          for (int j = 0; j < 4; j++) {
            float v0 = (float)d[0][j], v1 = (float)d[1][j], v2 = (float)d[2][j], v3 = (float)d[3][j];
            tmp[0][j] = v0 - v2;
            tmp[1][j] = v1 + v2;
            tmp[2][j] = v2 - v1;
            tmp[3][j] = v1 - v3;
          }
          for (int r = 0; r < 4; r++) {
            float u0 = tmp[r][0], u1 = tmp[r][1], u2 = tmp[r][2], u3 = tmp[r][3];
            float V0 = u0 - u2;
            float V1 = u1 + u2;
            float V2 = u2 - u1;
            float V3 = u1 - u3;
            // outp [16, Ntiles, C] — C is the fast axis.
            int base = ((r * 4 + 0) * Ntiles_k + tileIdx) * C_k + c;
            outp[base + 0 * Ntiles_k * C_k] = (T)V0;
            outp[base + 1 * Ntiles_k * C_k] = (T)V1;
            outp[base + 2 * Ntiles_k * C_k] = (T)V2;
            outp[base + 3 * Ntiles_k * C_k] = (T)V3;
          }
        }
      }
    } else {
      // Tfast: grid x = Ntiles, grid y = ceil(C/WPT). VW must be 1 (enforced
      // by the static_assert above).
      uint t_group_ = thread_position_in_grid.x;
      uint c_group_ = thread_position_in_grid.y;
      int tileIdx = (int)t_group_;
      if (tileIdx >= Ntiles_k) return;

      int rem = tileIdx;
      int n   = rem / (tilesY_k * tilesX_k); rem -= n * tilesY_k * tilesX_k;
      int ty  = rem / tilesX_k;
      int tx  = rem % tilesX_k;

      for (int w = 0; w < WPT; w++) {
        int c = (int)c_group_ * WPT + w;
        if (c >= C_k) break;
        T d[4][4];
        for (int i = 0; i < 4; i++) {
          int iy = 2 * ty - 1 + i;
          for (int j = 0; j < 4; j++) {
            int ix = 2 * tx - 1 + j;
            if (iy < 0 || iy >= H_k || ix < 0 || ix >= W_k) {
              d[i][j] = (T)0.0f;
            } else {
              d[i][j] = inp[((n * H_k + iy) * W_k + ix) * C_k + c];
            }
          }
        }
        // Transform accumulates in fp32; only the stored V rounds to T (fp16-safe).
        float tmp[4][4];
        for (int j = 0; j < 4; j++) {
          float v0 = (float)d[0][j], v1 = (float)d[1][j], v2 = (float)d[2][j], v3 = (float)d[3][j];
          tmp[0][j] = v0 - v2;
          tmp[1][j] = v1 + v2;
          tmp[2][j] = v2 - v1;
          tmp[3][j] = v1 - v3;
        }
        for (int r = 0; r < 4; r++) {
          float u0 = tmp[r][0], u1 = tmp[r][1], u2 = tmp[r][2], u3 = tmp[r][3];
          float V0 = u0 - u2;
          float V1 = u1 + u2;
          float V2 = u2 - u1;
          float V3 = u1 - u3;
          // outp [16, Ntiles, C] — C is the fast axis.
          int base = ((r * 4 + 0) * Ntiles_k + tileIdx) * C_k + c;
          outp[base + 0 * Ntiles_k * C_k] = (T)V0;
          outp[base + 1 * Ntiles_k * C_k] = (T)V1;
          outp[base + 2 * Ntiles_k * C_k] = (T)V2;
          outp[base + 3 * Ntiles_k * C_k] = (T)V3;
        }
      }
    }
)METAL";

// F(2,3) output untransform kernel: [16, Ntiles, outC] T input -> NHWC T output.
// Template args (JIT-substituted via MLX template_args):
//   T              — float or half (precision)
//   WPT            — tiles per thread
// Grid: (Cout, ceil(Ntiles/WPT), 1).
// nhwc input array carries the [N,H,W,outC] dims because metal_kernel only
// exposes *_shape for inputs, not outputs.
// The output kernel is monomorphic on VW=1, GRID_ORDER=Cfast, and matmul
// layout=Std. (GRID_ORDER=Cfast was chosen from an empirical sensitivity
// sweep showing <1% delta vs Tfast; the other two are structural.)
inline constexpr const char* kWinoOutputSource = R"METAL(
    static_assert(WPT >= 1, "WPT must be positive");

    // m shape [16, Ntiles, outC] — Ntiles=m_shape[1], outC=m_shape[2].
    int Ntiles_k = m_shape[1];
    int outC_k   = m_shape[2];
    int H_k      = nhwc[1];
    int W_k      = nhwc[2];
    int tilesY_k = (H_k + 1) / 2;
    int tilesX_k = (W_k + 1) / 2;

    // Cfast: grid x = Cout, grid y = ceil(Ntiles/WPT).
    uint oc_group = thread_position_in_grid.x;
    uint t_group  = thread_position_in_grid.y;

    for (int w = 0; w < WPT; w++) {
      int tileIdx = (int)t_group * WPT + w;
      if (tileIdx >= Ntiles_k) break;

      int rem = tileIdx;
      int n   = rem / (tilesY_k * tilesX_k); rem -= n * tilesY_k * tilesX_k;
      int ty  = rem / tilesX_k;
      int tx  = rem % tilesX_k;

      {
        int oc = (int)oc_group;
        if (oc >= outC_k) break;

        T mm[4][4];
        for (int r = 0; r < 4; r++) {
          for (int c2 = 0; c2 < 4; c2++) {
            int p = r * 4 + c2;
            // m shape [16, Ntiles, outC].
            mm[r][c2] = m[(p * Ntiles_k + tileIdx) * outC_k + oc];
          }
        }
        // Untransform accumulates in fp32; only the stored Y rounds to T (fp16-safe).
        float tmp[2][4];
        for (int c2 = 0; c2 < 4; c2++) {
          float v0 = (float)mm[0][c2], v1 = (float)mm[1][c2], v2 = (float)mm[2][c2], v3 = (float)mm[3][c2];
          tmp[0][c2] = v0 + v1 + v2;
          tmp[1][c2] = v1 - v2 - v3;
        }
        for (int a = 0; a < 2; a++) {
          float u0 = tmp[a][0], u1 = tmp[a][1], u2 = tmp[a][2], u3 = tmp[a][3];
          float Y0 = u0 + u1 + u2;
          float Y1 = u1 - u2 - u3;
          int oy0 = 2 * ty + a;
          if (oy0 < H_k) {
            int ox0 = 2 * tx + 0;
            if (ox0 < W_k)
              outp[((n * H_k + oy0) * W_k + ox0) * outC_k + oc] = (T)Y0;
            int ox1 = 2 * tx + 1;
            if (ox1 < W_k)
              outp[((n * H_k + oy0) * W_k + ox1) * outC_k + oc] = (T)Y1;
          }
        }
      }
    }
)METAL";

// Output untransform + fused BN/activation epilogue. Extra inputs: scale,bias
// (fp32 [outC]). Template arg ACT: 0 identity, 1 mish, 2 relu. The epilogue
// consumes the rounded-to-T conv value (float)(T)Y to match the unfused path
// (which stores the conv as T, then a separate BN kernel reads it).
inline constexpr const char* kWinoOutputSourceBNAct = R"METAL(
    static_assert(WPT >= 1, "WPT must be positive");
    int Ntiles_k = m_shape[1];
    int outC_k   = m_shape[2];
    int H_k      = nhwc[1];
    int W_k      = nhwc[2];
    int tilesY_k = (H_k + 1) / 2;
    int tilesX_k = (W_k + 1) / 2;
    uint oc_group = thread_position_in_grid.x;
    uint t_group  = thread_position_in_grid.y;
    for (int w = 0; w < WPT; w++) {
      int tileIdx = (int)t_group * WPT + w;
      if (tileIdx >= Ntiles_k) break;
      int rem = tileIdx;
      int n   = rem / (tilesY_k * tilesX_k); rem -= n * tilesY_k * tilesX_k;
      int ty  = rem / tilesX_k;
      int tx  = rem % tilesX_k;
      {
        int oc = (int)oc_group;
        if (oc >= outC_k) break;
        T mm[4][4];
        for (int r = 0; r < 4; r++)
          for (int c2 = 0; c2 < 4; c2++)
            mm[r][c2] = m[((r*4+c2) * Ntiles_k + tileIdx) * outC_k + oc];
        float tmp[2][4];
        for (int c2 = 0; c2 < 4; c2++) {
          float v0=(float)mm[0][c2], v1=(float)mm[1][c2], v2=(float)mm[2][c2], v3=(float)mm[3][c2];
          tmp[0][c2] = v0 + v1 + v2;
          tmp[1][c2] = v1 - v2 - v3;
        }
        float sc = (float)scale[oc];
        float bi = (float)bias[oc];
        for (int a = 0; a < 2; a++) {
          float u0=tmp[a][0], u1=tmp[a][1], u2=tmp[a][2], u3=tmp[a][3];
          float Y0 = u0 + u1 + u2;
          float Y1 = u1 - u2 - u3;
          // Round to T first (match unfused stored-then-read), then BN+act in fp32.
          float x0 = (float)(T)Y0, x1 = (float)(T)Y1;
          x0 = x0*sc + bi; x1 = x1*sc + bi;
          if (ACT == 1) {            // mish: x * tanh(softplus(x)), softplus = logaddexp(0,x)
            float s0 = metal::max(0.0f,x0) + metal::precise::log(1.0f + metal::precise::exp(-metal::abs(x0)));
            float s1 = metal::max(0.0f,x1) + metal::precise::log(1.0f + metal::precise::exp(-metal::abs(x1)));
            x0 = x0 * metal::precise::tanh(s0);
            x1 = x1 * metal::precise::tanh(s1);
          } else if (ACT == 2) {     // relu
            x0 = metal::max(0.0f,x0); x1 = metal::max(0.0f,x1);
          }
          int oy0 = 2*ty + a;
          if (oy0 < H_k) {
            int ox0 = 2*tx + 0;
            if (ox0 < W_k) outp[((n*H_k+oy0)*W_k+ox0)*outC_k + oc] = (T)x0;
            int ox1 = 2*tx + 1;
            if (ox1 < W_k) outp[((n*H_k+oy0)*W_k+ox1)*outC_k + oc] = (T)x1;
          }
        }
      }
    }
)METAL";

// Output untransform + fused broadcast-bias add then BN/activation. Identical to
// kWinoOutputSourceBNAct except a per-(n,oc) broadcast bias gbias ([N,outC], T) is
// added to the rounded-to-T conv value before BN+act (matches gpool's regularOut +
// bias). Extra inputs: gbias (T [N,outC]), scale,bias (fp32 [outC]). Template arg
// ACT: 0 identity, 1 mish, 2 relu.
inline constexpr const char* kWinoOutputSourceBiasBNAct = R"METAL(
    static_assert(WPT >= 1, "WPT must be positive");
    int Ntiles_k = m_shape[1];
    int outC_k   = m_shape[2];
    int H_k      = nhwc[1];
    int W_k      = nhwc[2];
    int tilesY_k = (H_k + 1) / 2;
    int tilesX_k = (W_k + 1) / 2;
    uint oc_group = thread_position_in_grid.x;
    uint t_group  = thread_position_in_grid.y;
    for (int w = 0; w < WPT; w++) {
      int tileIdx = (int)t_group * WPT + w;
      if (tileIdx >= Ntiles_k) break;
      int rem = tileIdx;
      int n   = rem / (tilesY_k * tilesX_k); rem -= n * tilesY_k * tilesX_k;
      int ty  = rem / tilesX_k;
      int tx  = rem % tilesX_k;
      {
        int oc = (int)oc_group;
        if (oc >= outC_k) break;
        T mm[4][4];
        for (int r = 0; r < 4; r++)
          for (int c2 = 0; c2 < 4; c2++)
            mm[r][c2] = m[((r*4+c2) * Ntiles_k + tileIdx) * outC_k + oc];
        float tmp[2][4];
        for (int c2 = 0; c2 < 4; c2++) {
          float v0=(float)mm[0][c2], v1=(float)mm[1][c2], v2=(float)mm[2][c2], v3=(float)mm[3][c2];
          tmp[0][c2] = v0 + v1 + v2;
          tmp[1][c2] = v1 - v2 - v3;
        }
        float sc = (float)scale[oc];
        float bi = (float)bias[oc];
        for (int a = 0; a < 2; a++) {
          float u0=tmp[a][0], u1=tmp[a][1], u2=tmp[a][2], u3=tmp[a][3];
          float Y0 = u0 + u1 + u2;
          float Y1 = u1 - u2 - u3;
          // Broadcast bias add (T+T, matches gpool's regularOut + bias), then round, then BN+act.
          T gb = gbias[n * outC_k + oc];
          float x0 = (float)((T)Y0 + gb), x1 = (float)((T)Y1 + gb);
          x0 = x0*sc + bi; x1 = x1*sc + bi;
          if (ACT == 1) {            // mish: x * tanh(softplus(x)), softplus = logaddexp(0,x)
            float s0 = metal::max(0.0f,x0) + metal::precise::log(1.0f + metal::precise::exp(-metal::abs(x0)));
            float s1 = metal::max(0.0f,x1) + metal::precise::log(1.0f + metal::precise::exp(-metal::abs(x1)));
            x0 = x0 * metal::precise::tanh(s0);
            x1 = x1 * metal::precise::tanh(s1);
          } else if (ACT == 2) {     // relu
            x0 = metal::max(0.0f,x0); x1 = metal::max(0.0f,x1);
          }
          int oy0 = 2*ty + a;
          if (oy0 < H_k) {
            int ox0 = 2*tx + 0;
            if (ox0 < W_k) outp[((n*H_k+oy0)*W_k+ox0)*outC_k + oc] = (T)x0;
            int ox1 = 2*tx + 1;
            if (ox1 < W_k) outp[((n*H_k+oy0)*W_k+ox1)*outC_k + oc] = (T)x1;
          }
        }
      }
    }
)METAL";

// Output untransform + fused residual add. Extra input: resid (T [N,H,W,outC]).
// Adds in T arithmetic on the rounded conv value -> bit-identical to unfused
// (T)conv + resid.
inline constexpr const char* kWinoOutputSourceResidual = R"METAL(
    static_assert(WPT >= 1, "WPT must be positive");
    int Ntiles_k = m_shape[1];
    int outC_k   = m_shape[2];
    int H_k      = nhwc[1];
    int W_k      = nhwc[2];
    int tilesY_k = (H_k + 1) / 2;
    int tilesX_k = (W_k + 1) / 2;
    uint oc_group = thread_position_in_grid.x;
    uint t_group  = thread_position_in_grid.y;
    for (int w = 0; w < WPT; w++) {
      int tileIdx = (int)t_group * WPT + w;
      if (tileIdx >= Ntiles_k) break;
      int rem = tileIdx;
      int n   = rem / (tilesY_k * tilesX_k); rem -= n * tilesY_k * tilesX_k;
      int ty  = rem / tilesX_k;
      int tx  = rem % tilesX_k;
      {
        int oc = (int)oc_group;
        if (oc >= outC_k) break;
        T mm[4][4];
        for (int r = 0; r < 4; r++)
          for (int c2 = 0; c2 < 4; c2++)
            mm[r][c2] = m[((r*4+c2) * Ntiles_k + tileIdx) * outC_k + oc];
        float tmp[2][4];
        for (int c2 = 0; c2 < 4; c2++) {
          float v0=(float)mm[0][c2], v1=(float)mm[1][c2], v2=(float)mm[2][c2], v3=(float)mm[3][c2];
          tmp[0][c2] = v0 + v1 + v2;
          tmp[1][c2] = v1 - v2 - v3;
        }
        for (int a = 0; a < 2; a++) {
          float u0=tmp[a][0], u1=tmp[a][1], u2=tmp[a][2], u3=tmp[a][3];
          float Y0 = u0 + u1 + u2;
          float Y1 = u1 - u2 - u3;
          int oy0 = 2*ty + a;
          if (oy0 < H_k) {
            int ox0 = 2*tx + 0;
            if (ox0 < W_k) { int idx=((n*H_k+oy0)*W_k+ox0)*outC_k+oc; outp[idx] = (T)Y0 + resid[idx]; }
            int ox1 = 2*tx + 1;
            if (ox1 < W_k) { int idx=((n*H_k+oy0)*W_k+ox1)*outC_k+oc; outp[idx] = (T)Y1 + resid[idx]; }
          }
        }
      }
    }
)METAL";

inline mx::array winogradConv2d(const mx::array& input,
                                const mx::array& Uw,
                                int Cout,
                                const InputTransform& inCfg,
                                const OutputUntransform& outCfg,
                                bool useFP16 = false,
                                const Epilogue& epi = Epilogue::none()) {
  int N = input.shape(0);
  int H = input.shape(1);
  int W = input.shape(2);
  int C = input.shape(3);
  int tilesY = (H + 1) / 2;
  int tilesX = (W + 1) / 2;
  int Ntiles = N * tilesY * tilesX;

  const mx::Dtype dtype = useFP16 ? mx::float16 : mx::float32;

  auto inSuffix = [&](const char* base, int wpt, int vw, GridOrder go) {
    return std::string(base) + "_" + (useFP16 ? "f16" : "f32")
         + "_w" + std::to_string(wpt)
         + "_v" + std::to_string(vw)
         + "_g" + std::to_string((int)go);
  };
  // Output kernel is monomorphic on VW=1, GRID_ORDER=Cfast,
  // and MATMUL_ORIENT=Std.
  auto outSuffix = [&](const char* base, int wpt) {
    return std::string(base) + "_" + (useFP16 ? "f16" : "f32")
         + "_w" + std::to_string(wpt);
  };
  std::string inName  = inSuffix ("wino_input_transform",    inCfg.wpt,  inCfg.vw,  inCfg.gridOrder);

  auto makeInTemplateArgs = [&](int wpt, int vw, GridOrder go) {
    return std::vector<std::pair<std::string, mx::fast::TemplateArg>>{
      {"T",             dtype},
      {"WPT",           wpt},
      {"VW",            vw},
      {"GRID_ORDER",    (int)go}
    };
  };

  // Stage 1: input transform. Output shape: [16, Ntiles, C].
  mx::Shape inOutShape = {16, Ntiles, C};

  // Grid: when gridOrder=Cfast the fast axis is C (grid x=C, y=Ntiles/WPT).
  // When gridOrder=Tfast we swap. WPT>1 reduces the slow-axis dim.
  int gridX_in = (inCfg.gridOrder == GridOrder::Cfast)
    ? ((C + inCfg.vw - 1) / inCfg.vw)
    : Ntiles;
  int gridY_in = (inCfg.gridOrder == GridOrder::Cfast)
    ? ((Ntiles + inCfg.wpt - 1) / inCfg.wpt)
    : ((C      + inCfg.wpt - 1) / inCfg.wpt);

  auto inFn = mx::fast::metal_kernel(
      inName.c_str(),
      /*input_names=*/{"inp"},
      /*output_names=*/{"outp"},
      /*source=*/kWinoInputSource);
  auto inOuts = inFn(
      /*inputs=*/{input},
      /*output_shapes=*/{ inOutShape },
      /*output_dtypes=*/{ dtype },
      /*grid=*/std::make_tuple(gridX_in, gridY_in, 1),
      /*threadgroup=*/std::make_tuple(inCfg.tg0, inCfg.tg1, 1),
      /*template_args=*/makeInTemplateArgs(inCfg.wpt, inCfg.vw, inCfg.gridOrder),
      /*init_value=*/std::nullopt,
      /*verbose=*/false,
      /*stream=*/mx::StreamOrDevice{});
  mx::array t = inOuts[0];

  // Stage 2: matmul. [16,Ntiles,C] @ [16,C,Cout] -> [16,Ntiles,Cout].
  // MLX steel gemm uses AccumType=float (static-asserted in mma.h:772) when
  // T=half, so fp32 accumulation is automatic.
  mx::array m = mx::matmul(t, Uw);

  // Stage 3: output untransform (+ optional fused epilogue) -> [N, H, W, Cout]
  int nhwc_arr[4] = {N, H, W, Cout};
  mx::array nhwcArr(nhwc_arr, {4}, mx::int32);
  int gridX_out = Cout;
  int gridY_out = (Ntiles + outCfg.wpt - 1) / outCfg.wpt;

  std::string outName;
  const char* outSrc;
  std::vector<std::string> outInputNames;
  std::vector<mx::array> outInputs;
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> outTpl = {
    {"T", dtype}, {"WPT", outCfg.wpt}
  };
  if(epi.mode == Epilogue::None) {
    outName = outSuffix("wino_output_untransform", outCfg.wpt);     // unchanged name
    outSrc = kWinoOutputSource;
    outInputNames = {"m", "nhwc"};
    outInputs = {m, nhwcArr};
  } else if(epi.mode == Epilogue::BNAct) {
    outName = outSuffix("wino_output_untransform", outCfg.wpt) + "_bnact_a" + std::to_string(epi.act);
    outSrc = kWinoOutputSourceBNAct;
    outInputNames = {"m", "nhwc", "scale", "bias"};
    outInputs = {m, nhwcArr, *epi.scale, *epi.bias};
    outTpl.push_back({"ACT", epi.act});
  } else if(epi.mode == Epilogue::BiasBNAct) {
    outName = outSuffix("wino_output_untransform", outCfg.wpt) + "_biasbnact_a" + std::to_string(epi.act);
    outSrc = kWinoOutputSourceBiasBNAct;
    outInputNames = {"m", "nhwc", "gbias", "scale", "bias"};
    outInputs = {m, nhwcArr, *epi.gbias, *epi.scale, *epi.bias};
    outTpl.push_back({"ACT", epi.act});
  } else { // Residual
    outName = outSuffix("wino_output_untransform", outCfg.wpt) + "_resid";
    outSrc = kWinoOutputSourceResidual;
    outInputNames = {"m", "nhwc", "resid"};
    outInputs = {m, nhwcArr, *epi.resid};
  }

  auto outFn = mx::fast::metal_kernel(
      outName.c_str(), outInputNames, /*output_names=*/{"outp"}, outSrc);
  auto outOuts = outFn(
      outInputs,
      /*output_shapes=*/{ mx::Shape{N, H, W, Cout} },
      /*output_dtypes=*/{ dtype },
      /*grid=*/std::make_tuple(gridX_out, gridY_out, 1),
      /*threadgroup=*/std::make_tuple(outCfg.tg0, outCfg.tg1, 1),
      /*template_args=*/outTpl,
      /*init_value=*/std::nullopt,
      /*verbose=*/false,
      /*stream=*/mx::StreamOrDevice{});
  return outOuts[0];
}

} // namespace MLXWinograd

#endif // USE_MLX_BACKEND
#endif // NEURALNET_MLXWINOGRAD_H_
