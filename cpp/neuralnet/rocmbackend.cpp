#ifdef USE_ROCM_BACKEND

#include "../neuralnet/rocmerrorcheck.h"
#include "../neuralnet/rocmincludes.h"

// Optional Composable Kernel FMHA fused attention support, see cpp/external/composable_kernel_fmha
// and KATAGO_ROCM_HAS_CK_FMHA in CMakeLists.txt. Mirrors the CUDA backend's optional cudnn-frontend
// SDPA path, but CK's fmha_fwd() has no expensive one-time "build plan" step to cache - each call
// directly checks traits/shape compatibility and either executes or returns a negative "unsupported"
// sentinel, so unlike the CUDA backend there is no need to tolerate plan-build failures only
// during warmup.
// ck_tile's FMHA kernels were never ported to GCN/Vega (gfx90x) or RDNA1 (gfx101x) - no
// MFMA/WMMA on those architectures. In the ck_tile version this glue targets (TheRock 7.13),
// arch.hpp's get_compiler_target() has no branch for them, so merely including its headers while
// compiling a device pass for one of these archs (as happens in a multi-arch fat binary that
// targets them) hard-fails with "member reference base type 'void' is not a structure or union",
// since get_compiler_target() falls through without a return. (Other ck_tile versions fail
// differently, but none support these archs.) Skip CK entirely for just these archs'
// device-compile passes, where the plain (non-fused) attention kernel still covers them. This
// list must stay in sync with the exclusion list in CMakeLists.txt's CK section.
#if defined(__gfx900__) || defined(__gfx902__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__) \
  || defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1013__)
  #define KATAGO_ROCM_CK_FMHA_ARCH_OK 0
#else
  #define KATAGO_ROCM_CK_FMHA_ARCH_OK 1
#endif

#if KATAGO_ROCM_HAS_CK_FMHA && KATAGO_ROCM_CK_FMHA_ARCH_OK
  #include <cstring>
  #include <utility>
  #include "fmha_fwd.hpp"
#endif

#include "../neuralnet/rocmhelpers.h"
#include "../neuralnet/rocmutils.h"
#include "../neuralnet/rocmcudanames.h"

// Backend selector for the shared implementation file included below. See the comment at the
// top of that file for the full contract.
#define KATAGO_GPU_HIP 1

// Short backend name used in error messages and debug-output labels that are otherwise
// identical between the CUDA and ROCm backends.
#define KATAGO_GPU_BACKEND_NAME "ROCm"

// Element type that hipblasHgemm expects buffers to be cast to.
using cublas_half_t = hipblasHalf;

#include "../neuralnet/cudaandrocmbackend.inc"

#endif  // USE_ROCM_BACKEND
