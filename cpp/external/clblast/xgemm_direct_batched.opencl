
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Original Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// MODIFIED by David Wu ("lightvector") to remove some unnecessary parts of the interfaces
// for this project's use.
// MODIFIED from the original by David Wu ("lightvector") to add FP16 storage with FP32 compute as an option.
//
// This file contains the batched version of the direct GEMM kernels. See part 1 for information
// about the non-batched version of the kernel.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================
#if defined(ROUTINE_GEMMBATCHED)

// Direct version of the batched GEMM kernel with [A, B] = [non-transposed, non-transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectBatchedNN(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __global realstoreMD* restrict agm, const int a_ld,
                          const __global realstoreND* restrict bgm, const int b_ld,
                          __global realstore* cgm, const int c_ld,
                          const int c_transpose) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = 1;
  const real_arg arg_beta = 0;
  const int a_offset = batch * kSizeM * kSizeK;
  const int b_offset = batch * kSizeN * kSizeK;
  const int c_offset = batch * kSizeM * kSizeN;
  const int a_conjugate = 0;
  const int b_conjugate = 0;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the batched GEMM kernel with [A, B] = [non-transposed, transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectBatchedNT(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __global realstoreMD* restrict agm, const int a_ld,
                          const __global realstoreND* restrict bgm, const int b_ld,
                          __global realstore* cgm, const int c_ld,
                          const int c_transpose) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = 1;
  const real_arg arg_beta = 0;
  const int a_offset = batch * kSizeM * kSizeK;
  const int b_offset = batch * kSizeN * kSizeK;
  const int c_offset = batch * kSizeM * kSizeN;
  const int a_conjugate = 0;
  const int b_conjugate = 0;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the batched GEMM kernel with [A, B] = [transposed, non-transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectBatchedTN(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __global realstoreMD* restrict agm, const int a_ld,
                          const __global realstoreND* restrict bgm, const int b_ld,
                          __global realstore* cgm, const int c_ld,
                          const int c_transpose) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = 1;
  const real_arg arg_beta = 0;
  const int a_offset = batch * kSizeM * kSizeK;
  const int b_offset = batch * kSizeN * kSizeK;
  const int c_offset = batch * kSizeM * kSizeN;
  const int a_conjugate = 0;
  const int b_conjugate = 0;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the batched GEMM kernel with [A, B] = [transposed, transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectBatchedTT(const int kSizeM, const int kSizeN, const int kSizeK,
                          const __global realstoreMD* restrict agm, const int a_ld,
                          const __global realstoreND* restrict bgm, const int b_ld,
                          __global realstore* cgm, const int c_ld,
                          const int c_transpose) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = 1;
  const real_arg arg_beta = 0;
  const int a_offset = batch * kSizeM * kSizeK;
  const int b_offset = batch * kSizeN * kSizeK;
  const int c_offset = batch * kSizeM * kSizeN;
  const int a_conjugate = 0;
  const int b_conjugate = 0;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}

#endif
// =================================================================================================
#if defined(ROUTINE_GEMMSTRIDEDBATCHED)

// Direct version of the strided-batched GEMM kernel with [A, B] = [non-transposed, non-transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectStridedBatchedNN(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const __global realstoreMD* restrict agm, const int a_ld, const int a_stride,
                                 const __global realstoreND* restrict bgm, const int b_ld, const int b_stride,
                                 __global realstore* cgm, const int c_ld, const int c_stride,
                                 const int c_transpose) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = 1;
  const real_arg arg_beta = 0;
  const int a_offset_batch = a_stride * batch;
  const int b_offset_batch = b_stride * batch;
  const int c_offset_batch = c_stride * batch;
  const int a_conjugate = 0;
  const int b_conjugate = 0;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the strided-batched GEMM kernel with [A, B] = [non-transposed, transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectStridedBatchedNT(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const __global realstoreMD* restrict agm, const int a_ld, const int a_stride,
                                 const __global realstoreND* restrict bgm, const int b_ld, const int b_stride,
                                 __global realstore* cgm, const int c_ld, const int c_stride,
                                 const int c_transpose) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = 1;
  const real_arg arg_beta = 0;
  const int a_offset_batch = a_stride * batch;
  const int b_offset_batch = b_stride * batch;
  const int c_offset_batch = c_stride * batch;
  const int a_conjugate = 0;
  const int b_conjugate = 0;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the strided-batched GEMM kernel with [A, B] = [transposed, non-transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectStridedBatchedTN(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const __global realstoreMD* restrict agm, const int a_ld, const int a_stride,
                                 const __global realstoreND* restrict bgm, const int b_ld, const int b_stride,
                                 __global realstore* cgm, const int c_ld, const int c_stride,
                                 const int c_transpose) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = 1;
  const real_arg arg_beta = 0;
  const int a_offset_batch = a_stride * batch;
  const int b_offset_batch = b_stride * batch;
  const int c_offset_batch = c_stride * batch;
  const int a_conjugate = 0;
  const int b_conjugate = 0;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the strided-batched GEMM kernel with [A, B] = [transposed, transposed]
__kernel __attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
void XgemmDirectStridedBatchedTT(const int kSizeM, const int kSizeN, const int kSizeK,
                                 const __global realstoreMD* restrict agm, const int a_ld, const int a_stride,
                                 const __global realstoreND* restrict bgm, const int b_ld, const int b_stride,
                                 __global realstore* cgm, const int c_ld, const int c_stride,
                                 const int c_transpose) {
  const int batch = get_group_id(2);
  const real_arg arg_alpha = 1;
  const real_arg arg_beta = 0;
  const int a_offset_batch = a_stride * batch;
  const int b_offset_batch = b_stride * batch;
  const int c_offset_batch = c_stride * batch;
  const int a_conjugate = 0;
  const int b_conjugate = 0;
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset_batch, a_ld, bgm, b_offset_batch, b_ld, cgm, c_offset_batch, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}

#endif
// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
