
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
// This file contains the batched version of the non-direct GEMM kernel. See part 1 for information
// about the non-batched version of the kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_GEMMBATCHED)

__kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
void XgemmBatched(const int kSizeM, const int kSizeN, const int kSizeK,
                  const __global realstoreM* restrict agm, const int a_one, const int a_two,
                  const __global realstoreN* restrict bgm, const int b_one, const int b_two,
                  __global realstoreM* cgm, const int c_one, const int c_two) {
  const int batch = get_group_id(2);

  // Sets the offsets
  const int a_offset = batch * a_one * a_two;
  const int b_offset = batch * b_one * b_two;
  const int c_offset = batch * c_one * c_two;
  const __global realstoreM* restrict agm_ = &agm[a_offset / VWM];
  const __global realstoreN* restrict bgm_ = &bgm[b_offset / VWN];
  __global realstoreM* restrict cgm_ = &cgm[c_offset / VWM];

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realstoreM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realstoreN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alm);
  #elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, blm);
  #else
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_);
  #endif
}

#endif
// =================================================================================================
#if defined(ROUTINE_GEMMSTRIDEDBATCHED)

__kernel __attribute__((reqd_work_group_size(MDIMC, NDIMC, 1)))
void XgemmStridedBatched(const int kSizeM, const int kSizeN, const int kSizeK,
                         const __global realstoreM* restrict agm, const int a_one, const int a_two,
                         const __global realstoreN* restrict bgm, const int b_one, const int b_two,
                         __global realstoreM* cgm, const int c_one, const int c_two) {
  const int batch = get_group_id(2);

  // Sets the offsets
  const int a_offset = batch * a_one * a_two;
  const int b_offset = batch * b_one * b_two;
  const int c_offset = batch * c_one * c_two;
  const __global realstoreM* restrict agm_ = &agm[a_offset / VWM];
  const __global realstoreN* restrict bgm_ = &bgm[b_offset / VWN];
  __global realstoreM* restrict cgm_ = &cgm[c_offset / VWM];

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realstoreM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realstoreN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, alm);
  #elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_, blm);
  #else
    XgemmBody(kSizeM, kSizeN, kSizeK, agm_, bgm_, cgm_);
  #endif
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
