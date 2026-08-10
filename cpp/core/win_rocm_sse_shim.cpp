#ifdef USE_ROCM_BACKEND
#ifdef _WIN32
// These are x86-64 SSE2 intrinsics, so only compile the shim bodies on x86-64.
#if defined(_M_X64) || defined(__x86_64__)

// On Windows ROCm builds, clang compiles host code against MSVC's ucrt headers, and ucrt's
// <wchar.h> includes <intrin.h> - which on that toolchain can resolve to MSVC's own <intrin.h>
// (chained in via clang's own intrin.h doing "#include_next <intrin.h>"), not clang's compatible
// one. MSVC's <intrin.h> declares SSE2 intrinsics like _mm_loadu_si128 as plain bodyless extern
// functions (relying on cl.exe's special-cased intrinsic recognition, which clang doesn't
// replicate for headers reached this way), rather than the "static __inline__" functions with
// actual bodies that clang's own <emmintrin.h> provides. Any code that ends up calling one of
// these - e.g. ucrt's own SSE2-optimized wmemcmp/wmemchr in <wchar.h>, used transitively by
// std::filesystem/std::wstring paths in fileutils.cpp, makedir.cpp, dataio/files.cpp and
// loadmodel.cpp - then links with "undefined symbol: _mm_loadu_si128"-style errors, since no
// definition exists anywhere in the link.
//
// The clang-resource-dir-first include ordering set up in CMakeLists.txt's
// katago_win_autoselect_msvc_toolset does not remove the need for this shim. Objects compiled
// under that ordering still contain undefined references to these three symbols, which
// "llvm-nm -u" will show. Check that before concluding this file can be dropped.
//
// Providing real definitions for just the 3 intrinsics ucrt's wmemcmp/wmemchr actually use fixes
// this. The <intrin.h> include below makes MSVC's __m128i type and the bodyless declarations
// visible so the extern signatures here match them. The bodies use GNU vector-extension types
// instead of any SSE header functions, sidestepping the missing-definitions problem entirely.
// If a toolchain instead resolves <intrin.h> to clang's own header chain with inline bodies,
// this file fails to compile with redefinition errors - which is the correct loud signal that
// the shim is unnecessary on that toolchain and can be dropped from the build.
#include <intrin.h>

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
#endif
