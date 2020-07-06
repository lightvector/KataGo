
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// MODIFIED from the original by David Wu ("lightvector") to add FP16 storage with FP32 compute as an option.
//
// This is a generic GEMM kernel that works for all sizes and configurations: it doesn't require any
// pre and and post-processing kernels.
//
// This kernel is seperated into three files. This is part 1 out of 3.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library. Note that all parameters here have a
// suffix 'D' to denote that they are for the 'direct' version of the GEMM kernel.
#ifndef WGD
  #define WGD 8      // Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)
#endif
#ifndef MDIMCD
  #define MDIMCD 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMCD
  #define NDIMCD 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMAD
  #define MDIMAD 8    // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#endif
#ifndef NDIMBD
  #define NDIMBD 8    // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#endif
#ifndef KWID
  #define KWID 1      // Unroll factor of the WGD loop (smaller or equal than WGD)
#endif
#ifndef VWMD
  #define VWMD 1      // Vector width of matrices A and C
#endif
#ifndef VWND
  #define VWND 1      // Vector width of matrix B
#endif
#ifndef PADA
  #define PADA 1      // Local memory padding for matrix A
#endif
#ifndef PADB
  #define PADB 1      // Local memory padding for matrix B
#endif

// Helper parameters based on the above tuning parameters
#define MWID (WGD/MDIMCD)                // Work per work-item (M-dimension)
#define NWID (WGD/NDIMCD)                // Work per work-item (N-dimension)
#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#define MWAD (WGD/MDIMAD)                // Amount of loads-per-thread for matrix A (M-dimension)
#define KWAD (WGD/KDIMAD)                // Amount of loads-per-thread for matrix A (K-dimension)
#define KWBD (WGD/KDIMBD)                // Amount of loads-per-thread for matrix B (K-dimension)
#define NWBD (WGD/NDIMBD)                // Amount of loads-per-thread for matrix B (N-dimension)

// =================================================================================================

#if PRECISION_STORAGE == 16 && PRECISION == 32
    #define LOADGLOBAL(__buf,__x) vloada_half((__x),(const __global half*)(__buf))
    #define LOADLOCAL(__buf,__x) vloada_half((__x),(LOCAL_PTR half*)(__buf))
    #define STOREGLOBAL(__buf,__x,__val) vstorea_half((__val),(__x),(__global half*)(__buf))
    #define SetToZeroStore(a) (a) = 0
#else
    #define LOADGLOBAL(__buf,__x) ((__buf)[(__x)])
    #define LOADLOCAL(__buf,__x) ((__buf)[(__x)])
    #define STOREGLOBAL(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #define SetToZeroStore(a) SetToZero(a)
#endif


// Data-widths in dimension M
#if VWMD == 1
    typedef real realMD;
    typedef realstore realstoreMD;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half((__x),(const __global half*)(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWMD == 2
    typedef real2 realMD;
    typedef realstore2 realstoreMD;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half2((__x),(const __global half*)(__buf))
        #define LOADGLOBALTOVECM(__buf,__x) vload_half2((__x),(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half2((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half2((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWMD == 4
    typedef real4 realMD;
    typedef realstore4 realstoreMD;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half4((__x),(const __global half*)(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half4((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half4((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWMD == 8
    typedef real8 realMD;
    typedef realstore8 realstoreMD;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half8((__x),(const __global half*)(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half8((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half8((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWMD == 16
    typedef real16 realMD;
    typedef realstore16 realstoreMD;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALM(__buf,__x) vloada_half16((__x),(const __global half*)(__buf))
        #define LOADLOCALM(__buf,__x) vloada_half16((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALM(__buf,__x,__val) vstorea_half16((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALM(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALM(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALM(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#endif

// Data-widths in dimension N
#if VWND == 1
    typedef real realND;
    typedef realstore realstoreND;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half((__x),(const __global half*)(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWND == 2
    typedef real2 realND;
    typedef realstore2 realstoreND;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half2((__x),(const __global half*)(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half2((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half2((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWND == 4
    typedef real4 realND;
    typedef realstore4 realstoreND;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half4((__x),(const __global half*)(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half4((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half4((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWND == 8
    typedef real8 realND;
    typedef realstore8 realstoreND;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half8((__x),(const __global half*)(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half8((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half8((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#elif VWND == 16
    typedef real16 realND;
    typedef realstore16 realstoreND;
    #if PRECISION_STORAGE == 16 && PRECISION == 32
        #define LOADGLOBALN(__buf,__x) vloada_half16((__x),(const __global half*)(__buf))
        #define LOADLOCALN(__buf,__x) vloada_half16((__x),(LOCAL_PTR half*)(__buf))
        #define STOREGLOBALN(__buf,__x,__val) vstorea_half16((__val),(__x),(__global half*)(__buf))
    #else
        #define LOADGLOBALN(__buf,__x) ((__buf)[(__x)])
        #define LOADLOCALN(__buf,__x) ((__buf)[(__x)])
        #define STOREGLOBALN(__buf,__x,__val) ((__buf)[(__x)] = (__val))
    #endif
#endif

// =================================================================================================

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix.
INLINE_FUNC real GlobalToPrivateDirectA(const __global realstore* restrict agms, const int _mi,
                                        const int a_ld, const int a_offset, const int idm, const int idk,
                                        const int a_transpose, const int a_conjugate) {
  const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);
  real result = LOADGLOBAL(agms,a_index + a_offset);
  if (a_conjugate) { COMPLEX_CONJUGATE(result); }
  return result;
}

// Same as above, but now for the B input matrix
INLINE_FUNC real GlobalToPrivateDirectB(const __global realstore* restrict bgms, const int _ni,
                                        const int b_ld, const int b_offset, const int idn, const int idk,
                                        const int b_transpose, const int b_conjugate) {
  const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);
  real result = LOADGLOBAL(bgms,b_index + b_offset);
  if (b_conjugate) { COMPLEX_CONJUGATE(result); }
  return result;
}

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix. This is the same as above but now includes a bounds check.
INLINE_FUNC real GlobalToPrivateCheckedA(const __global realstore* restrict agms, const int _mi,
                                         const int a_ld, const int a_offset, const int idm, const int idk,
                                         const int a_transpose, const int a_conjugate,
                                         const int kSizeM) {
  real result;
  if (idm + _mi < kSizeM) {
    const int a_index = (a_transpose) ? (idm + _mi)*a_ld + idk : idk*a_ld + (idm + _mi);
    result = LOADGLOBAL(agms,a_index + a_offset);
    if (a_conjugate) { COMPLEX_CONJUGATE(result); }
  }
  else {
    SetToZero(result);
  }
  return result;
}

// Same as above, but now for the B input matrix
INLINE_FUNC real GlobalToPrivateCheckedB(const __global realstore* restrict bgms, const int _ni,
                                         const int b_ld, const int b_offset, const int idn, const int idk,
                                         const int b_transpose, const int b_conjugate,
                                         const int kSizeN) {
  real result;
  if (idn + _ni < kSizeN) {
    const int b_index = (b_transpose) ? (idn + _ni)*b_ld + idk : idk*b_ld + (idn + _ni);
    result = LOADGLOBAL(bgms,b_index + b_offset);
    if (b_conjugate) { COMPLEX_CONJUGATE(result); }
  }
  else {
    SetToZero(result);
  }
  return result;
}

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
INLINE_FUNC real LocalToPrivateDirectA(LOCAL_PTR realstore* alm, const int _mi, const int kg,
                                       const int a_transpose) {
  const int mg = _mi + get_local_id(0)*MWID;
  const int index = (a_transpose) ? mg*(WGD + PADA) + kg : kg*(WGD + PADA) + mg;
  return LOADLOCAL(alm,index);
}

// Same as above, but now for the B input matrix
INLINE_FUNC real LocalToPrivateDirectB(LOCAL_PTR realstore* blm, const int _ni, const int kg,
                                       const int b_transpose) {
  const int ng = _ni + get_local_id(1)*NWID;
  const int index = (b_transpose) ? ng*(WGD + PADB) + kg : kg*(WGD + PADB) + ng;
  return LOADLOCAL(blm,index);
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResultsDirect(__global realstore* cgm, const real c_value,
                                    const int _mi, const int _ni, const int idm, const int idn,
                                    const real alpha, const real beta,
                                    const int c_ld, const int c_offset, const int c_transpose) {

  // Determines the destination index
  int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);

  // The final multiplication with alpha (in case beta == 0)
  real result;
  if (IsZero(beta)) {
    Multiply(result, alpha, c_value);
  }
  // The final multiplication with alpha and the addition with beta*C
  else {
    real cloaded = LOADGLOBAL(cgm,c_index + c_offset);
    AXPBY(result, alpha, c_value, beta, cloaded);
  }
  STOREGLOBAL(cgm,c_index + c_offset,result);
}

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResultsChecked(__global realstore* cgm, const real c_value,
                                     const int _mi, const int _ni, const int idm, const int idn,
                                     const int kSizeM, const int kSizeN,
                                     const real alpha, const real beta,
                                     const int c_ld, const int c_offset, const int c_transpose) {
  if ((idm + _mi) < kSizeM && (idn + _ni) < kSizeN) {

    // Deter_mines the destination index
    int c_index = (c_transpose) ? (idm + _mi)*c_ld + (idn + _ni) : (idn + _ni)*c_ld + (idm + _mi);

    // The final multiplication with alpha (in case beta == 0)
    real result;
    if (IsZero(beta)) {
      Multiply(result, alpha, c_value);
    }
    // The final multiplication with alpha and the addition with beta*C
    else {
      real cloaded = LOADGLOBAL(cgm,c_index + c_offset);
      AXPBY(result, alpha, c_value, beta, cloaded);
    }
    STOREGLOBAL(cgm,c_index + c_offset,result);
  }
}

// =================================================================================================

// End of the C++11 raw string literal
)"

// =================================================================================================
