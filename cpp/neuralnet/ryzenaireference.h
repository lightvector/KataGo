/*
 * Pure C++ (C++17, standard library only) CPU reference implementation of the
 * KataGo neural network forward pass, for the RyzenAI (AMD NPU) backend.
 *
 * Purpose:
 *  - Numerical ground truth for elementwise A/B checking of the accelerated
 *    NPU/GPU paths, and a per-operator fallback when an operator is not (yet)
 *    available on device.
 *
 * Layout contract:
 *  - ALL internal activations and ALL input/output buffers below are NHWC:
 *    spatial tensors are [N][nnYLen][nnXLen][C] row-major (C innermost), so a
 *    per-batch-element slab is [H*W][C] and can be fed directly to a GEMM as
 *    the A matrix. This is identical to the runtime layout of
 *    cpp/neuralnet/eigenbackend.cpp (whose Eigen tensors are (C,W,H,N)
 *    column-major, i.e. the same memory order).
 *
 * Semantics:
 *  - Ported 1:1 from cpp/neuralnet/eigenbackend.cpp: mask handling, global
 *    pooling normalization, RoPE/attention, SwiGLU FFN, and the exact
 *    boundary between backend outputs and client post-processing.
 *  - KataGo convention: ALL outputs are raw logits / pre-activation values.
 *    No softmax, no tanh, no policy-optimism interpolation and no symmetry
 *    handling is applied by this code. That post-processing is the caller's
 *    job, exactly as in eigenbackend.cpp's getOutput() (which reads buffers
 *    with precisely the layouts documented below).
 *
 * Supported model structure (all block kinds from desc.h):
 *  - ORDINARY_BLOCK_KIND (0)          plain residual block
 *  - GLOBAL_POOLING_BLOCK_KIND (2)    residual block with global-pooling bias
 *  - NESTED_BOTTLENECK_BLOCK_KIND (3) bottleneck block containing a nested
 *                                     stack of any of the block kinds here
 *                                     (recursion supported to any depth)
 *  - TRANSFORMER_ATTENTION_BLOCK_KIND (4)  multi-head attention with GQA,
 *                                     optional RoPE (learnable rope_freqs or
 *                                     fixed rope_theta), TransformerRMSNorm
 *  - TRANSFORMER_FFN_BLOCK_KIND (5)   SwiGLU FFN (linear1/linearGate/linear2)
 *  - initial conv + initial matmul (global feature projection)
 *  - SGF metadata encoder (when ModelDesc::metaEncoderVersion > 0)
 *  - trunk tip norm+activation, both TRUNK_NORM_KIND_STANDARD (BatchNorm)
 *    and TRUNK_NORM_KIND_RMSNORM (RMSNormLayerDesc, spatial and non-spatial)
 *  - policy head (both modelVersion >= 15 and older pass branches)
 *  - value head (value, scoreValue, ownership)
 *
 * Threading/reentrancy:
 *  - No global mutable state. A Workspace may only be used by one thread at a
 *    time (same rule as a ComputeHandle in the other backends). Different
 *    threads must use different Workspaces; each Workspace is independently
 *    reentrant across calls.
 *  - All scratch memory is allocated once in createWorkspace(); forward()
 *    performs no heap allocation.
 */

#ifndef NEURALNET_RYZENAI_REFERENCE_H_
#define NEURALNET_RYZENAI_REFERENCE_H_

#include <string>

#include "../neuralnet/desc.h"

namespace RyzenAIMatMul { struct Accel; }

namespace RyzenAIRef {

  // Opaque handle holding all scratch buffers and precomputed tables
  // (e.g. RoPE cos/sin) for one (model, maxBatchSize, nnXLen, nnYLen)
  // configuration.
  struct Workspace;

  // Allocates all scratch space needed by forward() for the given model and
  // geometry. Throws StringError on unsupported model features (currently the
  // only unsupported feature is a non-SwiGLU transformer FFN).
  //
  // The ModelDesc is NOT copied: `model` must remain alive and unmodified
  // (in particular, do not call releaseWeights() on it) until freeWorkspace().
  //
  // nnXLen/nnYLen are the (possibly padded) board dimensions the net is
  // evaluated at; boards smaller than this are handled via the mask channel
  // of spatialInput (see forward()).
  Workspace* createWorkspace(
    const ModelDesc& model,
    int maxBatchSize,
    int nnXLen,
    int nnYLen
  );

  void freeWorkspace(Workspace* workspace);

  // Attaches an NPU accelerator for the trunk's dense layers (attention
  // projections and FFN linears -- the layers whose row count is the whole
  // board, which is where nearly all the arithmetic is). Not owned; it must
  // outlive the workspace. nullptr, the default, keeps everything on the CPU.
  //
  // Anything the accelerator declines runs on the CPU path, so attaching one
  // never changes which models work, only how fast they are.
  void setMatMulAccel(Workspace* workspace, RyzenAIMatMul::Accel* accel);

  // Coarse timing of the work still done on the CPU, for deciding what to move
  // onto the NPU next. Enabling it costs a clock read per region, which is
  // negligible next to the regions themselves. Not thread-safe: the counters
  // are process-wide, so enable it only on single-threaded runs.
  void setProfileEnabled(bool enabled);
  std::string profileReport();

  // Runs one batched forward pass. batchSize must satisfy
  // 1 <= batchSize <= workspace's maxBatchSize.
  //
  // ============================ INPUT BUFFERS ============================
  //
  // spatialInput:
  //   [batchSize][nnYLen][nnXLen][ModelDesc::numInputChannels] row-major.
  //   Element (n,y,x,c) at spatialInput[((n*nnYLen + y)*nnXLen + x)*C + c].
  //   Channel 0 MUST be the on-board mask: exactly 1.0f at positions inside
  //   the actual board and 0.0f at padding positions (when the board is
  //   smaller than nnXLen x nnYLen). All global-pooling denominators and all
  //   masked reductions are computed from this channel, matching
  //   eigenbackend.cpp (`*mask = input->chip(0,0)`).
  //
  // globalInput:
  //   [batchSize][ModelDesc::numInputGlobalChannels] row-major.
  //   Element (n,c) at globalInput[n*Cg + c].
  //
  // metaInput:
  //   [batchSize][ModelDesc::numInputMetaChannels] row-major.
  //   Must be non-null iff the model has an SGF metadata encoder
  //   (ModelDesc::metaEncoderVersion > 0); pass nullptr otherwise.
  //
  // ============================ OUTPUT BUFFERS ===========================
  // All are written fully (for all batchSize rows) by forward().
  // All values are raw network outputs (logits / pre-activation); no final
  // softmax/tanh is applied anywhere. Below, PC = ModelDesc::numPolicyChannels,
  // VC = ModelDesc::numValueChannels, SVC = ModelDesc::numScoreValueChannels,
  // OC = ModelDesc::numOwnershipChannels.
  //
  // policy:
  //   [batchSize][nnYLen][nnXLen][PC] row-major (NHWC).
  //   Element (n,y,x,c) at policy[((n*nnYLen + y)*nnXLen + x)*PC + c].
  //   Raw output of the policy head's final conv (p2Conv). Per KataGo
  //   convention this buffer is NOT masked after the final conv (padded
  //   positions contain whatever the conv produces; the caller must ignore
  //   or legalize them, as eigenbackend's getOutput does via the client).
  //   Channel meaning:
  //     c=0: policy logit for playing at (y,x).
  //     c=1: (only if PC == 2 or PC == 4) optimistic-policy logit.
  //     c=2,3: (only if PC == 4, modelVersion >= 16) auxiliary q-value
  //            prediction channels.
  //   The optimism interpolation between channels 0/1 is NOT done here; it is
  //   part of getOutput()-side post-processing.
  //
  // policyPass:
  //   [batchSize][PC] row-major. Element (n,c) at policyPass[n*PC + c].
  //   Raw pass-move logits, same channel semantics as `policy`.
  //
  // value:
  //   [batchSize][VC] row-major (VC == 3). Element (n,c) at value[n*VC + c].
  //   c=0: win logit, c=1: loss logit, c=2: no-result logit, from the
  //   perspective of the player to move. No softmax applied.
  //
  // scoreValue:
  //   [batchSize][SVC] row-major. Element (n,c) at scoreValue[n*SVC + c].
  //   Raw linear outputs (no activation). Channel order:
  //     modelVersion >= 9 (SVC == 6): [scoreMean, scoreMeanSq, lead,
  //       varTimeLeft, shorttermWinlossError, shorttermScoreError]
  //     modelVersion == 8 (SVC == 4): first 4 of the above
  //     4 <= modelVersion <= 7 (SVC == 2): first 2 of the above
  //     modelVersion == 3 (SVC == 1): scoreMean only
  //
  // ownership:
  //   [batchSize][nnYLen][nnXLen][OC] row-major (OC == 1).
  //   Element (n,y,x,c) at ownership[((n*nnYLen + y)*nnXLen + x)*OC + c].
  //   Raw output of vOwnershipConv: pre-tanh ownership prediction from the
  //   perspective of the player to move. NOT masked (raw conv output).
  //
  // The caller-provided output buffers may not alias each other or the input
  // buffers.
  void forward(
    Workspace* workspace,
    int batchSize,
    const float* spatialInput,
    const float* globalInput,
    const float* metaInput,
    float* policy,
    float* policyPass,
    float* value,
    float* scoreValue,
    float* ownership
  );

}  // namespace RyzenAIRef

#endif  // NEURALNET_RYZENAI_REFERENCE_H_
