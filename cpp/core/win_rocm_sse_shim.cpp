#ifdef USE_ROCM_BACKEND
#ifdef _WIN32

// On Windows ROCm builds, clang compiles every source file with -x hip (see the comment on this
// in CMakeLists.txt), which force-includes clang's own __clang_hip_runtime_wrapper.h before any
// user code. That wrapper's own includes (via <cstdlib>/<cmath>) transitively pull in ucrt's
// <wchar.h>, which includes <intrin.h> - and on this toolchain, that resolves to MSVC's own
// <intrin.h> (chained in via clang's own intrin.h doing "#include_next <intrin.h>"), not clang's
// compatible one. MSVC's <intrin.h> declares SSE2 intrinsics like _mm_loadu_si128 as plain
// bodyless extern functions (relying on cl.exe's special-cased intrinsic recognition, which clang
// doesn't replicate for headers reached this way), rather than the "static __inline__" functions
// with actual bodies that clang's own <emmintrin.h> provides. Any code that ends up calling one of
// these - e.g. ucrt's own SSE2-optimized wmemcmp/wmemchr in <wchar.h>, used transitively by
// std::filesystem/std::wstring paths in fileutils.cpp/makedir.cpp/loadmodel.cpp - then links with
// "undefined symbol: _mm_loadu_si128"-style errors, since no definition exists anywhere in the
// link (HIP's --hip-link mode also passes -nostdlib, so nothing here falls back to a prebuilt
// ucrt.lib implementation either).
//
// Providing real definitions for just the 3 intrinsics ucrt's wmemcmp/wmemchr actually use fixes
// this: matching MSVC's own __m128i type (a union, already visible via the same transitive
// include chain above, avoiding a "conflicting types" redeclaration error) for the external
// signature, but implementing the body using GNU vector-extension types instead of any SSE
// header, which need no header at all and so sidestep the whole conflict.
extern "C" __m128i _mm_loadu_si128(__m128i const* p) {
  __m128i r;
  __builtin_memcpy(&r, p, sizeof(r));
  return r;
}

extern "C" __m128i _mm_cmpeq_epi16(__m128i a, __m128i b) {
  typedef short katago_v8hi __attribute__((__vector_size__(16)));
  katago_v8hi va, vb;
  __builtin_memcpy(&va, &a, sizeof(va));
  __builtin_memcpy(&vb, &b, sizeof(vb));
  katago_v8hi vr = (va == vb);
  __m128i r;
  __builtin_memcpy(&r, &vr, sizeof(r));
  return r;
}

extern "C" int _mm_movemask_epi8(__m128i a) {
  typedef char katago_v16qi __attribute__((__vector_size__(16)));
  katago_v16qi va;
  __builtin_memcpy(&va, &a, sizeof(va));
  return __builtin_ia32_pmovmskb128(va);
}

#endif
#endif
