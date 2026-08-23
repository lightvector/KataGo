#ifdef USE_CUDA_BACKEND
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudaincludes.h"

// cuDNN frontend SDPA support. The header is vendored under external/cudnn-frontend and
// requires cuDNN >= 8.5. The SDPA path this backend uses additionally requires cuDNN 8.9.3+,
// hence the version gate below. NO_CUDNN_SDPA (a CMake option) forces the no-SDPA fallback
// configuration for testing it on machines whose cuDNN would otherwise enable SDPA.
// IMPORTANT: cudnn_frontend bundles nlohmann/json 3.11.3 which uses the same include guard
// (INCLUDE_NLOHMANN_JSON_HPP_) as KataGo's older nlohmann/json 3.8.0. Including cudnn_frontend.h
// first ensures the 3.11.3 version wins and that the template signatures cudnn_frontend expects
// are the ones actually available.
#if CUDNN_VERSION >= 8903 && !defined(NO_CUDNN_SDPA)
  #define KATAGO_CUDA_HAS_SDPA 1
  // Note: cudnn_frontend's Execution_plan_list::query_properties() trips GCC's -Wnull-dereference.
  // It's a benign issue in vendored third-party header code, but it can't be silenced with a
  // `#pragma GCC diagnostic ignored` here: GCC emits this one from the -O2 interprocedural-analysis
  // phase with no source location, so it ignores the per-region diagnostic pragma state. Instead it's
  // suppressed file-scoped via -Wno-null-dereference on this source in CMakeLists.txt.
  #include <cudnn_frontend.h>
#else
  #define KATAGO_CUDA_HAS_SDPA 0
#endif

#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/cudautils.h"

#ifdef USE_CUTLASS_FUSED_FFN
#include "../neuralnet/cudafusedffn.h"
#endif

// Backend selector for the shared implementation file included below. See the comment at the
// top of that file for the full contract.
#define KATAGO_GPU_CUDA 1

// Short backend name used in error messages and debug-output labels that are otherwise
// identical between the CUDA and ROCm backends.
#define KATAGO_GPU_BACKEND_NAME "CUDA"

// Element type that cublasHgemm expects buffers to be cast to.
using cublas_half_t = half;

#include "../neuralnet/cudaandrocmbackend.inc"

#endif  // USE_CUDA_BACKEND
