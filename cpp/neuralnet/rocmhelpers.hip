// ROCm/HIP wrapper for the shared CUDA/ROCm GPU kernels.
// All kernel code lives in cudaandrocmhelpers.inc, which is shared with the CUDA backend
// (cudahelpers.cu). See the comment at the top of that file for the macro contract.

#include "../neuralnet/rocmhelpers.h"

// HIP_SUPPORTS_FP16 is defined (or not) by the build system (see CMakeLists.txt) uniformly for
// all target archs at once - unlike CUDA there is no per-arch preprocessor gating here. Every
// arch KataGo's default arch list targets supports packed FP16. If this is absent, the half
// kernels compile to empty stubs, and the backend detects that via customCudaFp16KernelsCompiled()
// and refuses FP16 at runtime rather than silently computing garbage.
#ifdef HIP_SUPPORTS_FP16
#define KATAGO_GPU_SUPPORTS_FP16
#endif

#define KATAGO_GPU_HIP 1
#define KATAGO_GPU_SINCOSF sincosf

#include "../neuralnet/cudaandrocmhelpers.inc"
