
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// MODIFIED by David Wu ("lightvector") to remove some unnecessary parts of the interfaces
// for this project's use, such as alpha and beta scaling.
// MODIFIED from the original by David Wu ("lightvector") to add FP16 storage with FP32 compute as an option.
//
// This is part 4 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// The upper-triangular and lower-triangular kernels are only used in special cases
#if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)

// Main entry point of the kernel. This is the upper-triangular version.
__kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
void XgemmUpper(const int kSizeN, const int kSizeK,
                const __global realstoreM* restrict agm,
                const __global realstoreN* restrict bgm,
                __global realstoreM* cgm) {

  // Skip these threads if they do not contain threads contributing to the upper-triangle
  if ((GetGroupID1() + 1)*NWG < GetGroupID0()*MWG) {
    return;
  }

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realstoreM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realstoreN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alm);
  #elif SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, blm);
  #else
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm);
  #endif
}

// Main entry point of the kernel. This is the lower-triangular version.
__kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
void XgemmLower(const int kSizeN, const int kSizeK,
                const __global realstoreM* restrict agm,
                const __global realstoreN* restrict bgm,
                __global realstoreM* cgm) {

  // Skip these threads if they do not contain threads contributing to the lower-triangle
  if (GetGroupID1()*NWG > (GetGroupID0() + 1)*MWG) {
    return;
  }

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realstoreM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realstoreN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alm);
  #elif SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, blm);
  #else
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm);
  #endif
}

// =================================================================================================
// If not using a triangular version, include the regular kernel
#else

// Main entry point of the kernel. This is the regular full version.
__kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
void Xgemm(const int kSizeM, const int kSizeN, const int kSizeK,
           const __global realstoreM* restrict agm,
           const __global realstoreN* restrict bgm,
           __global realstoreM* cgm,
           const int b_offset, const int c_offset) {

  // Adds the offsets (in case of use of a single temporary buffer for A, B, and C)
  bgm = &bgm[b_offset];
  cgm = &cgm[c_offset];

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realstoreM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realstoreN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alm);
  #elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, blm);
  #else
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm);
  #endif
}

#endif

)"
// End of the C++11 raw string literal

// =================================================================================================
