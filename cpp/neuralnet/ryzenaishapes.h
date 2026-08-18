#ifndef NEURALNET_RYZENAI_SHAPES_H_
#define NEURALNET_RYZENAI_SHAPES_H_

// Enumerates every matrix multiply a RyzenAI forward pass would want to run,
// straight out of KataGo's own ModelDesc. This exists because the NPU artifacts
// are shape-specific: an .xclbin bakes K into the AIE core program, so the set
// of shapes a model actually needs decides the set of artifacts that must ship.
// See ../../.claude/skills/ryzenai-npu-backend/SKILL.md section 2.1.
//
// Only the reduction dim K and the output dim N are enumerated. M is not a grid
// dimension: rows of a GEMM are independent, so any M is handled by dispatching
// a fixed-M kernel ceil(M/Mkernel) times over row slices, padding only the tail.
//
// Reads shapes only, never weights, so it is cheap and safe to run at startup.

#include <string>
#include <vector>

struct ModelDesc;

namespace RyzenAIShapes {

// Where a GEMM's row count comes from, which decides how big M gets at runtime.
enum class RowKind {
  Spatial,   // M = batch * nnXLen * nnYLen  (one row per board point)
  Batch,     // M = batch                    (one row per position, e.g. gpool -> bias)
  AttnScore  // M = batch * numHeads * nnXY  (attention's own QK^T / PV products)
};

struct GemmUse {
  std::string path;  // "trunk.block07.ffn.linear1"
  std::string op;    // "matmul" | "conv1x1" | "conv3x3" | "attn.qk" | "attn.pv"
  int convY;         // 1 for a matmul
  int convX;
  int inChannels;
  int outChannels;
  int K;  // convY * convX * inChannels -- the implicit-GEMM reduction dim
  int N;  // outChannels
  RowKind rows;
};

const char* rowKindName(RowKind kind);

// Every GEMM in model order, recursing into nested-bottleneck stacks.
// nnXY = nnXLen*nnYLen. Attention's own QK^T / PV products have a board-dependent
// K or N, so they are only emitted when nnXY > 0.
std::vector<GemmUse> enumerate(const ModelDesc& desc, int nnXY = 0);

// Human-readable dump: per-layer table, deduplicated (K,N) set, and how many
// artifacts a quantized grid would need at several quantum choices together
// with the compute wasted on padding. This is the input to the M3.5 grid decision.
std::string report(const ModelDesc& desc, int nnXLen, int nnYLen);

// Picks a single reduction dim to run every spatial layer at, or 0 to keep
// per-layer choices.
//
// Every distinct K is a separate xclbin and therefore a separate hardware
// context, and alternating contexts measured ~0.46 ms per dispatch -- far more
// than these kernels spend on arithmetic. Collapsing onto one K trades
// zero-padded multiply-accumulates for one context, which wins whenever the
// padding stays modest. It does not win on convolution-heavy models, where the
// largest K (9*inChannels) dwarfs the smallest and the arithmetic is real, so
// the spread is bounded before accepting.
int chooseSingleK(const ModelDesc& desc, int nnXLen, int nnYLen, double maxSpread);

}  // namespace RyzenAIShapes

#endif  // NEURALNET_RYZENAI_SHAPES_H_
