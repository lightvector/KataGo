
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// MODIFIED by David Wu ("lightvector") to remove some unnecessary parts of the interfaces
// for this project's use, such as alpha and beta scaling, and to split part1 into 1a and 1b
// as it was too large.
// MODIFIED from the original by David Wu ("lightvector") to add FP16 storage with FP32 compute as an option.
//
// This file contains two optimized matrix-multiplication kernels:
// - Kernel 0: inspired by the paper by Matsumoto et al. and the tutorial on
//   http://www.cedricnugteren.nl/tutorial.php
// - Kernel 1: inspired by a Qualcomm optimized GPU kernel with 2D register tiling
//   https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
// Both are fully configurable (and tunable!) using many parameters. Both kernels support
// different data-types (SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM) through a pre-processor define.
//
// For kernel 0 matrices are accessed as follows:
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
// For kernel 1, both A and C are transposed w.r.t. the above
//
// Or as an image (assuming column-major)
//       K
//    o-------o
//    |       |
//  N | [B^T] |
//    |       |
//    o-------o
//        K               N
//    o-------o        o-----o
//  M |  [A]  |      M | [C] |
//    |       |        |     |
//    o-------o        o-----o
//
//
// This kernel is separated into multiple files. This is part 1a out of 4.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef GEMMK
  #define GEMMK 0    // Kernel to choose: 0 regular, 1 with 2D register tiling
#endif
#ifndef MWG
  #define MWG 8      // Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWG
  #define NWG 8      // Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWG
  #define KWG 8      // Tile-size in dimension K (e.g. 8, 16)
#endif
#ifndef MDIMC
  #define MDIMC 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMC
  #define NDIMC 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMA
  #define MDIMA 8    // Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
#endif
#ifndef NDIMB
  #define NDIMB 8    // Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
#endif
#ifndef KWI
  #define KWI 1      // Unroll factor of the KWG loop (smaller or equal than KWG)
#endif
#ifndef VWM
  #define VWM 1      // Vector width of matrices A and C
#endif
#ifndef VWN
  #define VWN 1      // Vector width of matrix B
#endif
#ifndef STRM
  #define STRM 0     // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef STRN
  #define STRN 0     // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef SA
  #define SA 0       // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)
#endif
#ifndef SB
  #define SB 0       // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)
#endif
#ifndef KREG
  #define KREG 1     // Amount of register tiling in second dimension, multiple of VWN (kernel 1 only)
#endif

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#ifndef USE_VECTOR_MAD
  #define USE_VECTOR_MAD 0      // Unroll (0) or don't (1) unroll the vector MAD manually
#endif
#ifndef GLOBAL_MEM_FENCE
  #define GLOBAL_MEM_FENCE 0    // Global synchronisation barrier for potential better performance
#endif

#ifndef SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA
  #define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA
  #define SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_INTEL
  #define SUBGROUP_SHUFFLING_INTEL 0
#endif
#ifndef USE_SUBGROUP_SHUFFLING
  #define USE_SUBGROUP_SHUFFLING 0     // Optionally enables subgroup shuffling for Intel GPUs
#endif

// Intel subgroups (https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_subgroups.html)
#if USE_SUBGROUP_SHUFFLING == 1 && SUBGROUP_SHUFFLING_INTEL == 1
  #pragma OPENCL EXTENSION cl_intel_subgroups: enable
  #define SUBGROUP_SIZE 8              // Assumes subgroup size is always 8 on Intel GPUs
#endif

// NVIDIA warps as subgroups using inline PTX (https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
#if USE_SUBGROUP_SHUFFLING == 1
  #if SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
    #define SUBGROUP_SIZE 32            // Assumes subgroup size is always 32 on NVIDIA GPUs
  #endif
#endif

#if NWI != SUBGROUP_SIZE || MDIMC < SUBGROUP_SIZE
  #undef USE_SUBGROUP_SHUFFLING
  #define USE_SUBGROUP_SHUFFLING 0     // Disables subgroups in case the assumptions don't hold
#endif

// =================================================================================================

// Data-widths in dimension M
#if VWM == 1
    typedef real realM;
    typedef realstore realstoreM;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECM(__buf,__x) vload_half((__x),(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECM(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWM == 2
    typedef real2 realM;
    typedef realstore2 realstoreM;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half2((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECM(__buf,__x) vload_half2((__x),(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half2((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half2((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECM(__buf,__x) vload2((__x),(__buf))
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWM == 4
    typedef real4 realM;
    typedef realstore4 realstoreM;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half4((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECM(__buf,__x) vload_half4((__x),(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half4((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half4((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECM(__buf,__x) vload4((__x),(__buf))
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWM == 8
    typedef real8 realM;
    typedef realstore8 realstoreM;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half8((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECM(__buf,__x) vload_half8((__x),(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half8((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half8((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECM(__buf,__x) vload8((__x),(__buf))
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWM == 16
    typedef real16 realM;
    typedef realstore16 realstoreM;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half16((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECM(__buf,__x) vload_half16((__x),(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half16((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half16((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECM(__buf,__x) vload16((__x),(__buf))
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#endif

// Data-widths in dimension N
#if VWN == 1
    typedef real realN;
    typedef realstore realstoreN;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECN(__buf,__x) vload_half((__x),(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECN(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWN == 2
    typedef real2 realN;
    typedef realstore2 realstoreN;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half2((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECN(__buf,__x) vload_half2((__x),(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half2((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half2((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECN(__buf,__x) vload2((__x),(__buf))
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWN == 4
    typedef real4 realN;
    typedef realstore4 realstoreN;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half4((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECN(__buf,__x) vload_half4((__x),(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half4((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half4((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECN(__buf,__x) vload4((__x),(__buf))
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWN == 8
    typedef real8 realN;
    typedef realstore8 realstoreN;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half8((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECN(__buf,__x) vload_half8((__x),(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half8((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half8((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECN(__buf,__x) vload8((__x),(__buf))
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWN == 16
    typedef real16 realN;
    typedef realstore16 realstoreN;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half16((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECN(__buf,__x) vload_half16((__x),(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half16((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half16((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADGLOBALTOVECN(__buf,__x) vload16((__x),(__buf))
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#endif

)"
// End of the C++11 raw string literal

// =================================================================================================
