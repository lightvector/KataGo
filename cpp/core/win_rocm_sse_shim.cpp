#ifdef USE_ROCM_BACKEND
#ifdef _WIN32
// These are x86-64 SSE2 intrinsics, so only compile the shim bodies on x86-64.
#if defined(_M_X64) || defined(__x86_64__)

// On Windows ROCm builds every katago source is compiled with -x hip, since TheRock's hip-config
// adds that as an interface compile flag on hip::device and katago links that target. clang
// therefore force-includes __clang_hip_runtime_wrapper.h ahead of user code, and whether the
// resulting header chain reaches clang's own <emmintrin.h> or MSVC's <intrin.h> differs between
// HIP SDK versions. Both cases occur in practice, so this file handles each.
//
// Where MSVC's chain wins, the SSE2 intrinsics arrive as bodyless extern declarations, since
// cl.exe recognizes them specially and clang does not replicate that for headers reached this way.
// Code calling one of them then fails to link with "undefined symbol: _mm_loadu_si128". That
// happens via ucrt's SSE2-optimized wmemcmp/wmemchr in <wchar.h>, reached transitively from the
// std::filesystem and std::wstring paths in fileutils.cpp, makedir.cpp, dataio/files.cpp and
// loadmodel.cpp. The definitions below supply the three intrinsics ucrt actually uses. Including
// <intrin.h> first makes MSVC's __m128i and its declarations visible so these signatures match,
// while the bodies use GNU vector extensions and so need no SSE header of their own.
//
// Where clang's chain wins, it already defines all three as static __inline__ __host__ __device__
// functions, and defining them again is a compile error rather than a fix. __EMMINTRIN_H is that
// header's include guard and MSVC's chain does not define it, so testing for it selects between
// the two cases directly instead of assuming either. This file is deliberately empty in that case.
//
// Note that the clang-resource-dir-first include ordering set up in CMakeLists.txt's
// katago_win_autoselect_msvc_toolset does not by itself decide which chain wins, and does not
// remove the need for this file. Before concluding it can be dropped, check that no object still
// has undefined references to these three symbols, which "llvm-nm -u" will show.
#include <intrin.h>

#if !defined(__EMMINTRIN_H)

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
#endif
