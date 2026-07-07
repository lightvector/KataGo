cmake_minimum_required(VERSION 3.18.2)

# Helper: produce the default broad list of AMD GPU architectures the ROCm
# backend should target when the user has not passed -DCMAKE_HIP_ARCHITECTURES=...
# Required archs are always included (well-established across many ROCm releases); optional archs
# are probed via an actual compile test and only included if the installed toolchain accepts them -
# this covers newer/rarer archs (RDNA1, RDNA2 sub-variants, RDNA3.5, RDNA4, ...) without hard-failing
# the whole build on an older toolchain that doesn't know about them yet. Keep this list in sync with
# whatever `apt-cache search '^amdrocm-core-sdk<ver>-gfx'` (or the CK codegen's own supported archs,
# see cpp/external/composable_kernel_fmha/README.md) shows as AMD adds new chips.
function(katago_default_hip_archs out_var)
  set(_required_archs
      gfx906 gfx908 gfx90a gfx942 gfx950
      gfx1030 gfx1100 gfx1101)
  set(_optional_archs
      gfx1010 gfx1011 gfx1012
      gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
      gfx1102 gfx1103
      gfx1150 gfx1151 gfx1152 gfx1153
      gfx1200 gfx1201)
  set(_result "${_required_archs}")

  if(CMAKE_HIP_COMPILER)
    set(_probe_dir "${CMAKE_BINARY_DIR}/_katago_hip_arch_probe")
    file(MAKE_DIRECTORY "${_probe_dir}")
    set(_probe_src "${_probe_dir}/probe.cpp")
    if(NOT EXISTS "${_probe_src}")
      file(WRITE "${_probe_src}" "int main(){return 0;}\n")
    endif()
    foreach(_arch IN LISTS _optional_archs)
      execute_process(
        COMMAND "${CMAKE_HIP_COMPILER}" --offload-arch=${_arch} -x hip -c
                "${_probe_src}" -o "${_probe_dir}/probe_${_arch}.o"
        RESULT_VARIABLE _rc
        OUTPUT_QUIET ERROR_QUIET)
      if(_rc EQUAL 0)
        list(APPEND _result ${_arch})
        message(STATUS "HIP arch ${_arch} accepted by compiler — including")
      else()
        message(STATUS "HIP arch ${_arch} not accepted by compiler — skipping")
      endif()
    endforeach()
  else()
    message(STATUS "CMAKE_HIP_COMPILER not yet known; optional archs not probed")
  endif()

  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

# Helper: resolve the actual ROCm install prefix on Linux. Since ROCm ~7.9, AMD's packaging puts
# real per-version installs under /opt/rocm/core-<version>/ (e.g. /opt/rocm/core-7.13), with
# /opt/rocm itself just a distro-managed symlink/alternatives entry that can point anywhere (or get
# silently repointed by installing another package) - relying on it directly is what caused this
# project to intermittently pick up a stale/mismatched ROCm install. Prefer the highest-versioned
# /opt/rocm/core-* directory if any exist; otherwise fall back to the old flat /opt/rocm layout.
function(katago_find_rocm_prefix out_var)
  set(_result "")
  if(EXISTS "/opt/rocm")
    file(GLOB _core_dirs "/opt/rocm/core-*")
    set(_versioned "")
    foreach(_dir ${_core_dirs})
      if(IS_DIRECTORY "${_dir}")
        get_filename_component(_name "${_dir}" NAME)
        string(REGEX REPLACE "^core-" "" _ver "${_name}")
        list(APPEND _versioned "${_ver}|${_dir}")
      endif()
    endforeach()
    if(_versioned)
      list(SORT _versioned COMPARE NATURAL ORDER DESCENDING)
      list(GET _versioned 0 _best)
      string(REGEX REPLACE "^[^|]*\\|" "" _result "${_best}")
    endif()
  endif()
  if(_result STREQUAL "" AND EXISTS "/opt/rocm")
    set(_result "/opt/rocm")
  endif()
  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

# Helper: locate vswhere.exe via the well-known Program Files env vars (Microsoft's own documented,
# stable install location for it: "<Program Files(x86)>/Microsoft Visual Studio/Installer/"), so we
# never hardcode a user- or edition-specific path.
function(katago_win_find_vswhere out_var)
  foreach(_pf_var "ProgramFiles(x86)" "ProgramFiles")
    if(DEFINED ENV{${_pf_var}})
      file(TO_CMAKE_PATH "$ENV{${_pf_var}}" _pf)
      set(_candidate "${_pf}/Microsoft Visual Studio/Installer/vswhere.exe")
      if(EXISTS "${_candidate}")
        set(${out_var} "${_candidate}" PARENT_SCOPE)
        return()
      endif()
    endif()
  endforeach()
  set(${out_var} "" PARENT_SCOPE)
endfunction()

# Helper: fallback Windows Kits (SDK) install root candidates, used only when the registry lookup
# ("Installed Roots" key) doesn't resolve. Built from the ProgramFiles env vars rather than a
# hardcoded drive letter, so this still works on a system with Windows installed to a non-C: drive.
function(katago_win_winsdk_fallback_dirs out_var)
  set(_dirs)
  foreach(_pf_var "ProgramFiles(x86)" "ProgramFiles")
    if(DEFINED ENV{${_pf_var}})
      file(TO_CMAKE_PATH "$ENV{${_pf_var}}" _pf)
      list(APPEND _dirs "${_pf}/Windows Kits/10")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _dirs)
  set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()

# Helper: apply the environment vcvarsall.bat would set for a given MSVC toolset version into the
# CURRENT cmake process's environment (equivalent to the user running vcvarsall.bat before invoking
# cmake, but automatic). This is enough for every check that happens during THIS SAME configure run
# (probing below, and the actual project()-triggered HIP compiler ABI test), since execute_process
# children inherit it - but it does NOT persist into a later, separate `ninja` invocation (a sibling
# process, not a child of cmake.exe), which is why katago_win_autoselect_msvc_toolset (below) also
# bakes the resulting INCLUDE/LIB paths into cached compiler/linker flags for that to keep working.
function(katago_win_apply_vcvarsall vcvarsall_bat vcvars_ver)
  # Generate a tiny .bat wrapper instead of trying to pass the whole "call ... && set" pipeline as
  # one execute_process COMMAND argument - CMake's own argument quoting mangles a string that
  # complex (embedded quotes plus '&&'), even though the identical command works fine typed
  # directly into a shell.
  set(_bat_file "${CMAKE_BINARY_DIR}/_katago_vcvars_dump.bat")
  set(_out_file "${CMAKE_BINARY_DIR}/_katago_vcvars_env.txt")
  file(WRITE "${_bat_file}" "@echo off\r\ncall \"${vcvarsall_bat}\" x64 -vcvars_ver=${vcvars_ver} >NUL 2>NUL\r\nset\r\n")
  execute_process(
    COMMAND cmd /c "${_bat_file}"
    OUTPUT_FILE "${_out_file}"
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0 OR NOT EXISTS "${_out_file}")
    return()
  endif()
  # Read line-by-line via file(STRINGS) rather than splitting the raw output ourselves - INCLUDE/
  # LIB/PATH are themselves semicolon-separated, which would corrupt a naive string(REPLACE "\n" ";"
  # ...) split; file(STRINGS) keeps each real line (with its embedded semicolons) as one list entry.
  file(STRINGS "${_out_file}" _env_lines)
  foreach(_line IN LISTS _env_lines)
    if(_line MATCHES "^([A-Za-z_][A-Za-z0-9_().]*)=(.*)$")
      set(ENV{${CMAKE_MATCH_1}} "${CMAKE_MATCH_2}")
    endif()
  endforeach()
endfunction()

# Helper: with the given HIP compiler and whatever environment is currently active, try compiling a
# tiny HIP program that includes <cmath>, to check whether the active MSVC toolset's STL conflicts
# with clang's own CUDA/HIP math forward declarations (see katago_win_autoselect_msvc_toolset below).
function(katago_win_probe_hip_cmath hip_compiler out_var)
  set(_probe_dir "${CMAKE_BINARY_DIR}/_katago_msvc_toolset_probe")
  file(MAKE_DIRECTORY "${_probe_dir}")
  set(_probe_src "${_probe_dir}/probe.cpp")
  if(NOT EXISTS "${_probe_src}")
    file(WRITE "${_probe_src}" "#include <cmath>\nint main(){ return 0; }\n")
  endif()
  # gfx900 (the oldest/most universally-recognized ROCm target ID, not tied to any GPU actually
  # installed on this machine) is used here only to give --offload-arch a valid value so the HIP
  # compile path (and its host-side <cmath> resolution, which is what's actually being tested) runs
  # at all - this probe is unrelated to katago_default_hip_archs()'s real per-arch compatibility
  # probing for the final build's architecture list.
  execute_process(
    COMMAND "${hip_compiler}" --offload-arch=gfx900 -x hip -c "${_probe_src}" -o "${_probe_dir}/probe.o"
    RESULT_VARIABLE _rc
    OUTPUT_QUIET ERROR_QUIET)
  set(${out_var} ${_rc} PARENT_SCOPE)
endfunction()

# Helper: turn a semicolon-separated path list (like the env var INCLUDE or LIB) into a single
# string of "-Iflagprefix<dir>" (or "-Lflagprefix<dir>") tokens, skipping empty entries.
function(katago_win_pathlist_to_flags pathlist flagprefix out_var)
  set(_flags "")
  # CMake's own ';' list-splitting would fight with the semicolons already in the pathlist - split
  # it as a plain string via regex instead of list(...).
  string(REGEX MATCHALL "[^;]+" _dirs "${pathlist}")
  foreach(_dir IN LISTS _dirs)
    if(NOT _dir STREQUAL "")
      set(_flags "${_flags} ${flagprefix}\"${_dir}\"")
    endif()
  endforeach()
  set(${out_var} "${_flags}" PARENT_SCOPE)
endfunction()

# Helper: find an MSVC "v143" toolset (possibly one of several 14.3x/14.4x versions installed
# side-by-side under one or more VS installs) whose STL doesn't conflict with the HIP compiler's
# own CUDA/HIP math forward-declare headers, apply the vcvarsall.bat environment for it (so the
# rest of THIS configure run - the HIP arch probing below, and project()'s own HIP compiler ABI
# test - see it), and also bake the resulting INCLUDE/LIB paths into cached compiler/linker flags
# so a later, separate `ninja` invocation (which does NOT inherit this configure run's environment
# - see katago_win_apply_vcvarsall's comment) keeps working too.
#
# Deliberately pinned to the v143 toolset family (MSVC 14.3x/14.4x) rather than probing every
# installed toolset regardless of version: a newer MSVC STL (seen with a VS "18" preview-channel
# toolset, MSVC 14.51+, i.e. the next "v144" family) declares math functions like isgreater/isless
# in a way that conflicts with clang's forward declarations for CUDA/HIP device overloads ("device
# function cannot overload host device function"), because that clang version predates the STL's
# newly-added declarations. v143 is the toolset actually verified to work with TheRock's clang;
# still probing within the v143 family (oldest first) rather than hardcoding one exact version,
# since multiple v143 point releases can be installed side by side and this needs one that's
# actually present.
function(katago_win_autoselect_msvc_toolset hip_compiler)
  katago_win_find_vswhere(_vswhere)
  if(NOT _vswhere)
    message(FATAL_ERROR "vswhere.exe not found (expected under \"<Program Files(x86)>/Microsoft Visual Studio/Installer/\") - install Visual Studio Build Tools or Community with the \"Desktop development with C++\" workload.")
  endif()
  execute_process(
    COMMAND "${_vswhere}" -all -products * -property installationPath
    OUTPUT_VARIABLE _install_paths_raw
    RESULT_VARIABLE _rc
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _rc EQUAL 0 OR _install_paths_raw STREQUAL "")
    message(FATAL_ERROR "vswhere found no Visual Studio installation - install Visual Studio Build Tools or Community with the \"Desktop development with C++\" workload.")
  endif()
  string(REPLACE "\r\n" "\n" _install_paths_raw "${_install_paths_raw}")
  string(REPLACE "\n" ";" _install_paths "${_install_paths_raw}")

  set(_toolsets)
  foreach(_install_path IN LISTS _install_paths)
    if(_install_path STREQUAL "")
      continue()
    endif()
    file(TO_CMAKE_PATH "${_install_path}" _install_path)
    file(GLOB _tool_dirs "${_install_path}/VC/Tools/MSVC/*")
    foreach(_t IN LISTS _tool_dirs)
      if(IS_DIRECTORY "${_t}" AND EXISTS "${_t}/include/cmath")
        get_filename_component(_ver "${_t}" NAME)
        # v143 toolset versions are 14.3x/14.4x (the next-generation "v144" toolset, seen with
        # VS 18 preview builds, starts at 14.5x and is the one known to conflict - see above).
        if(NOT _ver MATCHES "^14\\.[34][0-9]\\.")
          continue()
        endif()
        set(_vcvarsall "${_install_path}/VC/Auxiliary/Build/vcvarsall.bat")
        if(EXISTS "${_vcvarsall}")
          list(APPEND _toolsets "${_ver}|${_vcvarsall}")
        endif()
      endif()
    endforeach()
  endforeach()

  if(NOT _toolsets)
    message(FATAL_ERROR "No installed MSVC v143 toolset (14.3x/14.4x) found. Open the Visual Studio Installer, choose \"Modify\" on your VS installation, go to \"Individual Components\", and install \"MSVC v143 - VS 2022 C++ x64/x86 build tools\" (a newer toolset alone, e.g. a VS 18 preview's v144, is not compatible with TheRock's bundled clang).")
  endif()
  list(REMOVE_DUPLICATES _toolsets)
  list(SORT _toolsets COMPARE NATURAL ORDER ASCENDING)

  foreach(_entry IN LISTS _toolsets)
    string(REPLACE "|" ";" _entry_list "${_entry}")
    list(GET _entry_list 0 _ver)
    list(GET _entry_list 1 _vcvarsall)
    message(STATUS "Trying MSVC toolset ${_ver} for HIP compiler compatibility...")
    katago_win_apply_vcvarsall("${_vcvarsall}" "${_ver}")
    katago_win_probe_hip_cmath("${hip_compiler}" _probe_rc)
    if(_probe_rc EQUAL 0)
      message(STATUS "MSVC toolset ${_ver} is compatible with the HIP compiler; using it")
      # Bake this toolset's resolved INCLUDE/LIB (now sitting in ENV{} courtesy of
      # katago_win_apply_vcvarsall) into cached flags, so a later separate `ninja` invocation -
      # which won't have this environment - still finds the same headers/libs.
      #
      # Clang's own resource-dir/include (containing its clang-compatible emmintrin.h/immintrin.h
      # etc) must come FIRST, ahead of the MSVC toolset's own include dir: <cwchar> pulls in
      # <intrin.h>, and MSVC's own <intrin.h>/<immintrin.h> declare SSE/AVX intrinsics using
      # __declspec(intrin_type), a cl.exe-only mechanism clang doesn't understand - it just sees an
      # ordinary extern function declaration with no body, so any code that ends up calling one
      # (e.g. the MSVC STL's wmemcmp) links with an "undefined symbol: _mm_loadu_si128"-style error.
      # Explicitly preferring clang's own compatible headers for those specific filenames avoids
      # this, while still falling through to the MSVC/SDK -I entries below for headers clang has no
      # equivalent for (the actual MSVC STL, ucrt, etc). This needs a plain -I (not -isystem/
      # -idirafter): the earlier __clang_hip_runtime_wrapper.h force-included in every HIP
      # compilation uses #include_next to reach the "real" <cmath> starting from wherever clang's
      # own cuda_wrappers/cmath shim was found, and that continuation only walks through the same
      # -I search list, not -isystem/-idirafter entries.
      execute_process(COMMAND "${hip_compiler}" -print-resource-dir
        OUTPUT_VARIABLE _clang_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
      file(TO_CMAKE_PATH "${_clang_resource_dir}" _clang_resource_dir)
      set(_inc_flags "-I\"${_clang_resource_dir}/include\"")
      katago_win_pathlist_to_flags("$ENV{INCLUDE}" "-I" _msvc_inc_flags)
      katago_win_pathlist_to_flags("$ENV{LIB}" "-L" _lib_flags)
      set(_inc_flags "${_inc_flags} ${_msvc_inc_flags}")
      set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${_inc_flags}" CACHE STRING "" FORCE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_inc_flags}" CACHE STRING "" FORCE)
      set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${_inc_flags}"   CACHE STRING "" FORCE)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${_lib_flags}" CACHE STRING "" FORCE)
      return()
    else()
      message(STATUS "MSVC toolset ${_ver} conflicts with the HIP compiler's CUDA/HIP math headers; trying another")
    endif()
  endforeach()
  message(FATAL_ERROR "None of the installed MSVC v143 toolset(s) (14.3x/14.4x) compiled cleanly against TheRock's HIP compiler. This indicates a problem beyond the usual v143-vs-newer-toolset conflict this check is meant to catch - please report this along with the MSVC/TheRock versions involved.")
endfunction()

if(USE_BACKEND STREQUAL "METAL")
  project(katago LANGUAGES CXX Swift)
elseif(USE_BACKEND STREQUAL "ROCM")
  if(WIN32)
    # Normalize HIP_PATH to forward slashes.
    # cmake's HIP detection calls `hipconfig --rocmpath` which returns
    # backslash paths on Windows; it writes the result into the generated
    # CMakeHIPCompiler.cmake without normalizing, causing "Invalid character
    # escape '\T'" errors.  Pre-setting CMAKE_HIP_COMPILER_ROCM_ROOT (and
    # CMAKE_PREFIX_PATH) before project() bypasses that detection entirely.
    if(DEFINED ENV{HIP_PATH})
      file(TO_CMAKE_PATH "$ENV{HIP_PATH}" _hip_path_fwd)
      set(ENV{HIP_PATH} "${_hip_path_fwd}")
      list(APPEND CMAKE_PREFIX_PATH "${_hip_path_fwd}")
      if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_hip_path_fwd}" CACHE PATH "" FORCE)
      endif()
    endif()
    # ---------- C/C++ compiler (clang++ from HIP SDK) ----------
    # hipcc.exe was tried here (to mirror the Linux code path, where CMAKE_CXX_COMPILER=hipcc
    # transparently treats every .cpp as HIP source) but it mis-tokenizes quoted paths containing
    # spaces (e.g. "C:/Program Files (x86)/Windows Kits/..."), breaking the compiler-detection step
    # entirely - not usable as CMAKE_CXX_COMPILER on Windows. Use clang++ directly instead; the
    # rocmbackend.cpp / CK generated sources that need real HIP compilation get it via an explicit
    # LANGUAGE HIP source property instead (see below and the CK FMHA section further down).
    if(NOT CMAKE_CXX_COMPILER)
      if(DEFINED ENV{HIP_PATH})
        if(EXISTS "$ENV{HIP_PATH}/lib/llvm/bin/clang++.exe")
          set(CMAKE_CXX_COMPILER "$ENV{HIP_PATH}/lib/llvm/bin/clang++.exe" CACHE FILEPATH "" FORCE)
          set(CMAKE_C_COMPILER   "$ENV{HIP_PATH}/lib/llvm/bin/clang.exe"   CACHE FILEPATH "" FORCE)
        elseif(EXISTS "$ENV{HIP_PATH}/bin/clang++.exe")
          set(CMAKE_CXX_COMPILER "$ENV{HIP_PATH}/bin/clang++.exe" CACHE FILEPATH "" FORCE)
          set(CMAKE_C_COMPILER   "$ENV{HIP_PATH}/bin/clang.exe"   CACHE FILEPATH "" FORCE)
        endif()
      endif()
    endif()
    # HIP compiler = same binary as C++ compiler
    if(NOT CMAKE_HIP_COMPILER AND CMAKE_CXX_COMPILER)
      set(CMAKE_HIP_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "" FORCE)
    endif()
    # ---------- MSVC toolset auto-selection (avoid newer-STL / clang HIP header conflicts) ----------
    # Automatically finds and applies (via vcvarsall.bat) a locally-installed MSVC toolset that's
    # compatible with this HIP compiler - see katago_win_autoselect_msvc_toolset()'s comment above
    # for why this is needed. Runs once per fresh CMakeCache (cached so a plain re-run of cmake
    # doesn't redo the probing every time).
    if(CMAKE_HIP_COMPILER AND NOT KATAGO_WIN_MSVC_TOOLSET_CHECKED)
      katago_win_autoselect_msvc_toolset("${CMAKE_HIP_COMPILER}")
      set(KATAGO_WIN_MSVC_TOOLSET_CHECKED TRUE CACHE INTERNAL "")
    endif()
    # ---------- HIP architectures (must be set before project() / enable_language(HIP)) ----------
    # Default to the broad set of AMD GPU architectures KataGo supports unless
    # the user explicitly passed -DCMAKE_HIP_ARCHITECTURES=... on the command line.
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
      katago_default_hip_archs(_default_archs)
      set(CMAKE_HIP_ARCHITECTURES "${_default_archs}" CACHE STRING "Default broad set of AMD GPU targets")
      message(STATUS "Pre-project default CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
    else()
      message(STATUS "Pre-project user-specified CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
    endif()
    # ---------- Windows SDK includes (needed by HIP compiler test during project()) ----------
    # The HIP runtime wrapper includes MSVC headers that require Windows SDK ucrt/shared/um.
    # These flags must be set before project() so cmake's HIP compiler test can compile.
    if(NOT KATAGO_WINSDK_ROOT)
      get_filename_component(_pre_winsdk_root
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]"
        ABSOLUTE)
      if(NOT EXISTS "${_pre_winsdk_root}")
        katago_win_winsdk_fallback_dirs(_pre_winsdk_candidates)
        foreach(_p IN LISTS _pre_winsdk_candidates)
          if(EXISTS "${_p}")
            set(_pre_winsdk_root "${_p}")
            break()
          endif()
        endforeach()
      endif()
      if(EXISTS "${_pre_winsdk_root}")
        set(KATAGO_WINSDK_ROOT "${_pre_winsdk_root}" CACHE INTERNAL "")
      endif()
    endif()
    if(KATAGO_WINSDK_ROOT)
      file(GLOB _pre_sdk_ver_dirs "${KATAGO_WINSDK_ROOT}/Include/*/ucrt")
      if(_pre_sdk_ver_dirs)
        list(SORT _pre_sdk_ver_dirs ORDER DESCENDING)
        list(GET _pre_sdk_ver_dirs 0 _pre_ucrt_dir)
        get_filename_component(_pre_sdk_ver_dir "${_pre_ucrt_dir}" DIRECTORY)
        set(_winsdk_cflags
          "-I\"${_pre_sdk_ver_dir}/ucrt\" -I\"${_pre_sdk_ver_dir}/shared\" -I\"${_pre_sdk_ver_dir}/um\"")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${_winsdk_cflags}" CACHE STRING "" FORCE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_winsdk_cflags}" CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${_winsdk_cflags}" CACHE STRING "" FORCE)
        message(STATUS "Pre-project: injected Windows SDK includes from ${_pre_sdk_ver_dir}")
      endif()
    endif()
    # ---------- RC compiler (Windows resource compiler, highest SDK version) ----------
    if(NOT CMAKE_RC_COMPILER)
      # Try registry first
      get_filename_component(_winsdk_root
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]"
        ABSOLUTE)
      if(NOT EXISTS "${_winsdk_root}")
        katago_win_winsdk_fallback_dirs(_winsdk_candidates)
        foreach(_p IN LISTS _winsdk_candidates)
          if(EXISTS "${_p}")
            set(_winsdk_root "${_p}")
            break()
          endif()
        endforeach()
      endif()
      if(EXISTS "${_winsdk_root}")
        file(GLOB _rc_candidates "${_winsdk_root}/bin/*/x64/rc.exe")
        if(_rc_candidates)
          list(SORT _rc_candidates ORDER DESCENDING)
          list(GET _rc_candidates 0 _rc_exe)
          set(CMAKE_RC_COMPILER "${_rc_exe}" CACHE FILEPATH "" FORCE)
        endif()
        # Persist for use in compiler flags section below
        set(KATAGO_WINSDK_ROOT "${_winsdk_root}" CACHE INTERNAL "")
      endif()
    endif()
  else()
    # Linux: resolve the real ROCm install prefix first (see katago_find_rocm_prefix - handles the
    # /opt/rocm/core-<version> layout used since ROCm ~7.9, preferring the highest version found,
    # falling back to the old flat /opt/rocm layout). Respect a user-provided CMAKE_PREFIX_PATH.
    if(NOT DEFINED CMAKE_PREFIX_PATH OR CMAKE_PREFIX_PATH STREQUAL "")
      katago_find_rocm_prefix(KATAGO_ROCM_PREFIX)
      if(KATAGO_ROCM_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH "${KATAGO_ROCM_PREFIX}")
        message(STATUS "Pre-project: resolved ROCm install prefix to ${KATAGO_ROCM_PREFIX}")
      endif()
    endif()

    # Use hipcc. Its location varies across ROCm packaging generations (a classic apt "hipcc"
    # package at /usr/bin, one bundled under the ROCm prefix's bin/, or newer installs that only
    # ship amdclang++ with no hipcc wrapper at all) - probe candidates instead of hardcoding one
    # path, since a missing hardcoded path fails project() below with a confusing compiler-not-
    # found error rather than falling back.
    find_program(KATAGO_HIPCC_EXECUTABLE NAMES hipcc
      HINTS ${CMAKE_PREFIX_PATH}
      PATH_SUFFIXES bin)
    if(NOT KATAGO_HIPCC_EXECUTABLE AND EXISTS "/usr/bin/hipcc")
      set(KATAGO_HIPCC_EXECUTABLE "/usr/bin/hipcc")
    endif()
    # Search for clang++ unconditionally (not just as a hipcc-missing fallback): CMAKE_HIP_COMPILER
    # always needs the clang++ binary directly below, even when hipcc is used for C/CXX, since CMake
    # rejects the hipcc wrapper script for CMAKE_HIP_COMPILER ("...not supported").
    find_program(KATAGO_CLANGXX_EXECUTABLE NAMES clang++
      HINTS ${CMAKE_PREFIX_PATH}
      PATH_SUFFIXES lib/llvm/bin bin)
    if(KATAGO_HIPCC_EXECUTABLE)
      set(CMAKE_C_COMPILER  "${KATAGO_HIPCC_EXECUTABLE}" CACHE FILEPATH "" FORCE)
      set(CMAKE_CXX_COMPILER "${KATAGO_HIPCC_EXECUTABLE}" CACHE FILEPATH "" FORCE)
    elseif(KATAGO_CLANGXX_EXECUTABLE)
      message(STATUS "hipcc not found; falling back to ${KATAGO_CLANGXX_EXECUTABLE}")
      get_filename_component(_clang_dir "${KATAGO_CLANGXX_EXECUTABLE}" DIRECTORY)
      set(CMAKE_CXX_COMPILER "${KATAGO_CLANGXX_EXECUTABLE}" CACHE FILEPATH "" FORCE)
      set(CMAKE_C_COMPILER  "${_clang_dir}/clang" CACHE FILEPATH "" FORCE)
    endif()
    # ---------- HIP architectures (must be set before project()/enable_language(HIP)) ----------
    # project(... HIP) below triggers CMake's own HIP language detection, which auto-populates
    # CMAKE_HIP_ARCHITECTURES with just the native/current GPU's arch if it isn't already set by
    # then. That makes the broad-default-arch logic further down (before enable_language(HIP))
    # a no-op, since by that point CMAKE_HIP_ARCHITECTURES already looks "user-specified". Set the
    # broad default here instead, before project(), mirroring the Windows pre-project block above.
    # Note: CMAKE_HIP_COMPILER must be the clang++ binary directly - CMake rejects the hipcc
    # wrapper script here ("CMAKE_HIP_COMPILER is set to the hipcc wrapper... not supported").
    if(NOT CMAKE_HIP_COMPILER AND KATAGO_CLANGXX_EXECUTABLE)
      set(CMAKE_HIP_COMPILER "${KATAGO_CLANGXX_EXECUTABLE}" CACHE FILEPATH "" FORCE)
    endif()
    if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
      katago_default_hip_archs(_default_archs)
      set(CMAKE_HIP_ARCHITECTURES "${_default_archs}" CACHE STRING "Default broad set of AMD GPU targets")
      message(STATUS "Pre-project default CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
    else()
      message(STATUS "Pre-project user-specified CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
    endif()
  endif()
  project(katago LANGUAGES C CXX HIP)
else()
  project(katago)
endif()

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_CONFIGURATION_TYPES)
  # Default to RelWithDebInfo if no build type specified
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

include_directories(external)
include_directories(SYSTEM external/tclap-1.2.5/include) #SYSTEM suppresses warnings from external library
include_directories(SYSTEM external/filesystem-1.5.8/include) #SYSTEM suppresses a few warnings

#--------------------------- PLATFORM SPECIFIC -------------------------------------------------------------------------

if(APPLE)
  # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
  if(IS_DIRECTORY /usr/local/include)
    include_directories(/usr/local/include)
  endif()
  if(IS_DIRECTORY /usr/local/lib)
    link_directories(/usr/local/lib)
  endif()
endif()

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset   "${Esc}[m")
  set(ColorBold    "${Esc}[1m")
  set(ColorRed     "${Esc}[31m")
  set(ColorBoldRed "${ColorRed}${ColorBold}")
endif()

#--------------------------- CMAKE VARIABLES (partly for Cmake GUI) ----------------------------------------------------

set(BUILD_DISTRIBUTED 0 CACHE BOOL "Build with http support for contributing to distributed training")
set(USE_BACKEND CACHE STRING "Neural net backend")
string(TOUPPER "${USE_BACKEND}" USE_BACKEND)
set_property(CACHE USE_BACKEND PROPERTY STRINGS "" CUDA TENSORRT OPENCL EIGEN METAL ROCM)

set(USE_TCMALLOC 0 CACHE BOOL "Use TCMalloc")
set(NO_GIT_REVISION 0 CACHE BOOL "Disable embedding the git revision into the compiled exe")
set(USE_AVX2 0 CACHE BOOL "Compile with AVX2")
set(USE_BIGGER_BOARDS_EXPENSIVE 0 CACHE BOOL "Allow boards up to size 50. Compiling with this will use more memory and slow down KataGo, even when playing on boards of size 19.")

set(USE_CACHE_TENSORRT_PLAN 0 CACHE BOOL "Use TENSORRT plan cache. May use a lot of disk space. Only applies when USE_BACKEND is TENSORRT.")
mark_as_advanced(USE_CACHE_TENSORRT_PLAN)

# ---------- Auto-fetch missing third-party deps (e.g. zlib on a Windows ROCm/TheRock install that
# doesn't ship a linkable zlib) via a local vcpkg clone in the build tree. Mirrors the same
# KATAGO_AUTO_FETCH_DEPS / vcpkg-in-build-tree pattern used on the Intel_NPU branch, so this stays
# consistent across branches instead of reinventing a different fetch mechanism here.
if(WIN32 OR (UNIX AND NOT APPLE))
  set(_katago_auto_fetch_default ON)
else()
  set(_katago_auto_fetch_default OFF)
endif()
option(KATAGO_AUTO_FETCH_DEPS "Automatically fetch missing dependencies into build/deps (Windows/Linux use vcpkg)." ${_katago_auto_fetch_default})
set(KATAGO_DEPS_DIR "${CMAKE_SOURCE_DIR}/build/deps" CACHE PATH "Directory for auto-fetched third-party dependencies")
if(WIN32)
  set(_katago_vcpkg_triplet_default "x64-windows")
elseif(UNIX AND NOT APPLE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_katago_vcpkg_triplet_default "arm64-linux")
  else()
    set(_katago_vcpkg_triplet_default "x64-linux")
  endif()
else()
  set(_katago_vcpkg_triplet_default "x64-windows")
endif()
set(KATAGO_VCPKG_TRIPLET "${_katago_vcpkg_triplet_default}" CACHE STRING "vcpkg triplet used by KATAGO_AUTO_FETCH_DEPS")
set(KATAGO_VCPKG_ROOT "${KATAGO_DEPS_DIR}/vcpkg" CACHE PATH "Path to local vcpkg clone used by KATAGO_AUTO_FETCH_DEPS")
mark_as_advanced(KATAGO_VCPKG_TRIPLET KATAGO_VCPKG_ROOT)

function(katago_vcpkg_bootstrap_if_needed)
  if(NOT WIN32 AND NOT (UNIX AND NOT APPLE))
    message(FATAL_ERROR "katago_vcpkg_bootstrap_if_needed is only supported on Windows and Linux")
  endif()

  if(NOT KATAGO_AUTO_FETCH_DEPS)
    message(FATAL_ERROR "KATAGO_AUTO_FETCH_DEPS is OFF, cannot auto-fetch missing dependency")
  endif()

  file(MAKE_DIRECTORY "${KATAGO_DEPS_DIR}")

  if(WIN32)
    set(_katago_vcpkg_exe "${KATAGO_VCPKG_ROOT}/vcpkg.exe")
  else()
    set(_katago_vcpkg_exe "${KATAGO_VCPKG_ROOT}/vcpkg")
  endif()

  if(NOT EXISTS "${_katago_vcpkg_exe}")
    if(NOT EXISTS "${KATAGO_VCPKG_ROOT}/.git")
      find_package(Git QUIET)
      if(NOT GIT_FOUND)
        message(FATAL_ERROR "KATAGO_AUTO_FETCH_DEPS requires git to clone vcpkg")
      endif()
      message(STATUS "Auto-fetch deps: cloning vcpkg into ${KATAGO_VCPKG_ROOT}")
      execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --depth=1 https://github.com/microsoft/vcpkg.git "${KATAGO_VCPKG_ROOT}"
        RESULT_VARIABLE _clone_result
        OUTPUT_VARIABLE _clone_out
        ERROR_VARIABLE _clone_err
      )
      if(NOT _clone_result EQUAL 0)
        message(FATAL_ERROR "Failed to clone vcpkg.\n${_clone_out}\n${_clone_err}")
      endif()
    endif()

    message(STATUS "Auto-fetch deps: bootstrapping vcpkg")
    if(WIN32)
      execute_process(
        COMMAND "${KATAGO_VCPKG_ROOT}/bootstrap-vcpkg.bat" -disableMetrics
        WORKING_DIRECTORY "${KATAGO_VCPKG_ROOT}"
        RESULT_VARIABLE _bootstrap_result
      )
    else()
      execute_process(
        COMMAND sh "${KATAGO_VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
        WORKING_DIRECTORY "${KATAGO_VCPKG_ROOT}"
        RESULT_VARIABLE _bootstrap_result
      )
    endif()
    if(NOT _bootstrap_result EQUAL 0)
      message(FATAL_ERROR "Failed to bootstrap vcpkg")
    endif()
  endif()
endfunction()

function(katago_vcpkg_install_if_needed package_name)
  katago_vcpkg_bootstrap_if_needed()

  if(WIN32)
    set(_katago_vcpkg_exe "${KATAGO_VCPKG_ROOT}/vcpkg.exe")
  else()
    set(_katago_vcpkg_exe "${KATAGO_VCPKG_ROOT}/vcpkg")
  endif()
  if(NOT EXISTS "${_katago_vcpkg_exe}")
    message(FATAL_ERROR "vcpkg executable not found after bootstrap: ${_katago_vcpkg_exe}")
  endif()

  set(_spec "${package_name}:${KATAGO_VCPKG_TRIPLET}")
  message(STATUS "Auto-fetch deps: ensuring ${_spec} via vcpkg")
  execute_process(
    COMMAND "${_katago_vcpkg_exe}" install "${_spec}" --disable-metrics
    WORKING_DIRECTORY "${KATAGO_VCPKG_ROOT}"
    RESULT_VARIABLE _install_result
  )
  if(NOT _install_result EQUAL 0)
    message(FATAL_ERROR "Failed to install ${_spec} via vcpkg")
  endif()
endfunction()

#--------------------------- NEURAL NET BACKEND ------------------------------------------------------------------------

message(STATUS "Building 'katago' executable for GTP engine and other tools.")
if(USE_BACKEND STREQUAL "CUDA")
  message(STATUS "-DUSE_BACKEND=CUDA, using CUDA backend.")

  enable_language(CUDA)

  set(CUDA_STANDARD 11)
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/cudabackend.cpp
    neuralnet/cudautils.cpp
    neuralnet/cudahelpers.cu
  )
  # The vendored header-only cudnn_frontend library trips GCC's -Wnull-dereference from its -O2
  # interprocedural analysis (no source location, so an in-file #pragma can't suppress it). It's
  # benign third-party code; silence it just for the one source that includes cudnn_frontend.h.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(neuralnet/cudabackend.cpp PROPERTIES COMPILE_OPTIONS "-Wno-null-dereference")
  endif()
# https://en.wikipedia.org/wiki/CUDA#GPUs_supported
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 88 89 90 100 110 103 120 121)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 90 120)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0)
    set(CMAKE_CUDA_ARCHITECTURES 50 52 53 60 61 62 70 72 75 80 86 87 90)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87 90)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.5)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86 87)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.1)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80 86)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
    set(CMAKE_CUDA_ARCHITECTURES 35 37 50 52 53 60 61 62 70 72 75 80)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2)
    set(CMAKE_CUDA_ARCHITECTURES 30 35 37 50 52 53 60 61 62 70 72 75)
    set(CMAKE_CUDA_FLAGS "-Wno-deprecated-gpu-targets")
  else()
    message(WARNING "CUDA 10.2 or greater is recommended, but attempting to build anyways")
    set(CMAKE_CUDA_ARCHITECTURES 30 37 53 70)
  endif()
elseif(USE_BACKEND STREQUAL "TENSORRT")
  message(STATUS "-DUSE_BACKEND=TENSORRT, using TensorRT backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/trtbackend.cpp
    )
  if(USE_CACHE_TENSORRT_PLAN AND (NOT BUILD_DISTRIBUTED))
    message(STATUS "-DUSE_CACHE_TENSORRT_PLAN is set, using TENSORRT plan cache.")
    add_compile_definitions(CACHE_TENSORRT_PLAN)
  elseif(USE_CACHE_TENSORRT_PLAN AND BUILD_DISTRIBUTED)
    message(FATAL_ERROR "Combining USE_CACHE_TENSORRT_PLAN with BUILD_DISTRIBUTED is not supported - it would consume excessive disk space and might worsen performance every time models are updated. Use only one at a time in a given build of KataGo.")
  endif()
elseif(USE_BACKEND STREQUAL "METAL")
  message(STATUS "-DUSE_BACKEND=METAL, using Metal backend with hybrid MPSGraph + CoreML execution.")
  if(NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
    message(FATAL_ERROR "Bidirectional C++ Interop requires Ninja generator. Have ${CMAKE_GENERATOR}")
  endif()
  if("${CMAKE_Swift_COMPILER_VERSION}" VERSION_LESS 5.9)
    message(FATAL_ERROR "Bidirectional C++ Interop requires Swift 5.9 or greater. Have ${CMAKE_Swift_COMPILER_VERSION}")
  endif()
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    message(FATAL_ERROR "Project requires building with AppleClang. Have ${CMAKE_CXX_COMPILER_ID}")
  endif()
  add_subdirectory(external/katagocoreml)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/macos/cmake/modules")

  if (NOT CMAKE_OSX_SYSROOT)
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE CMAKE_OSX_SYSROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()

  include(InitializeSwift)
  include(AddSwift)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0)
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/metalbackend.cpp
    )
  add_library(KataGoSwift STATIC
    neuralnet/metalbackend.swift
    neuralnet/metallayers.swift)
  _swift_generate_cxx_header(
    KataGoSwift
    "${CMAKE_CURRENT_BINARY_DIR}/include/KataGoSwift/KataGoSwift-swift.h")
  target_include_directories(KataGoSwift PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/include")
  set_target_properties(KataGoSwift PROPERTIES Swift_MODULE_NAME "KataGoSwift")
  target_compile_options(KataGoSwift PUBLIC
    "$<$<COMPILE_LANGUAGE:Swift>:-cxx-interoperability-mode=default>")
elseif(USE_BACKEND STREQUAL "OPENCL")
  message(STATUS "-DUSE_BACKEND=OPENCL, using OpenCL backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/openclbackend.cpp
    neuralnet/openclkernels.cpp
    neuralnet/openclhelpers.cpp
    neuralnet/opencltuner.cpp
    )
elseif(USE_BACKEND STREQUAL "EIGEN")
  message(STATUS "-DUSE_BACKEND=EIGEN, using Eigen CPU backend.")
  if(NOT USE_AVX2)
    message(STATUS "You can also specify USE_AVX2 (-DUSE_AVX2=1 on command line) if you have a modern CPU for better performance.")
  endif()
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/eigenbackend.cpp
    )
# --------------------------- ROCM backend（AMD GPU / HIP  MIOpen） ---------------------------
elseif(USE_BACKEND STREQUAL "ROCM")
  message(STATUS "-DUSE_BACKEND=ROCM, using AMD ROCm backend.")

  if(CMAKE_PREFIX_PATH STREQUAL "" OR NOT DEFINED CMAKE_PREFIX_PATH)
    if(WIN32)
      # Windows: HIP SDK installed via installer or manually
      if(DEFINED ENV{HIP_PATH})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{HIP_PATH}")
        message(STATUS "Auto-detected HIP_PATH=$ENV{HIP_PATH} → CMAKE_PREFIX_PATH")
      elseif(DEFINED ENV{ROCM_PATH})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
        message(STATUS "Auto-detected ROCM_PATH=$ENV{ROCM_PATH} → CMAKE_PREFIX_PATH")
      else()
        message(FATAL_ERROR "ROCM backend on Windows requires HIP_PATH or ROCM_PATH environment variable (or -DCMAKE_PREFIX_PATH=<HIP SDK path>). Please install the HIP SDK for Windows.")
      endif()
    else()
      # Linux: resolve the real ROCm install prefix (same core-<version> vs flat /opt/rocm
      # resolution as the pre-project block above; this block re-runs it standalone in case
      # USE_BACKEND wasn't ROCM early enough for that block to have fired).
      katago_find_rocm_prefix(KATAGO_ROCM_PREFIX)
      if(KATAGO_ROCM_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH "${KATAGO_ROCM_PREFIX}")
        message(STATUS "CMAKE_PREFIX_PATH not given; defaulting to ${KATAGO_ROCM_PREFIX}")
      endif()
    endif()
  endif()

  # Ensure CMAKE_HIP_COMPILER is set BEFORE we run the optional-arch probe and before
  # enable_language(HIP). On Windows the pre-project block already sets it; on Linux it should
  # already be set by the pre-project block too - this is just a safety net. Must be the clang++
  # binary directly, not the hipcc wrapper (CMake rejects hipcc for CMAKE_HIP_COMPILER).
  if(NOT WIN32 AND NOT CMAKE_HIP_COMPILER)
    if(NOT KATAGO_CLANGXX_EXECUTABLE)
      find_program(KATAGO_CLANGXX_EXECUTABLE NAMES clang++ HINTS ${CMAKE_PREFIX_PATH} PATH_SUFFIXES lib/llvm/bin bin)
    endif()
    if(KATAGO_CLANGXX_EXECUTABLE)
      set(CMAKE_HIP_COMPILER "${KATAGO_CLANGXX_EXECUTABLE}" CACHE FILEPATH "" FORCE)
    endif()
  endif()

  # Default GPU architectures must be set BEFORE enable_language(HIP).
  # Users who pass -DCMAKE_HIP_ARCHITECTURES=... on the command line get
  # exactly that list; otherwise we default to the broad set KataGo supports
  # (plus gfx1031/gfx1032 if the compiler accepts them).
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    katago_default_hip_archs(_default_archs)
    set(CMAKE_HIP_ARCHITECTURES "${_default_archs}" CACHE STRING "Default broad set of AMD GPU targets")
    message(STATUS "Default CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
  else()
    message(STATUS "User-specified CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
  endif()

  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)

  # 2) Specify backend source code. rocmhelpers.hip contains GPU kernels, don't forget it
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/rocmbackend.cpp
    neuralnet/rocmutils.cpp
    neuralnet/rocmhelpers.hip
  )

  # Optional: Enable model-size‑based autotuning and other macros
  # add_compile_definitions(HIP_SUPPORTS_FP16)

elseif(USE_BACKEND STREQUAL "")
  message(WARNING "${ColorBoldRed}WARNING: Using dummy neural net backend, intended for non-neural-net testing only, will fail on any code path requiring a neural net. To use neural net, specify -DUSE_BACKEND=CUDA or -DUSE_BACKEND=TENSORRT or -DUSE_BACKEND=OPENCL or -DUSE_BACKEND=EIGEN to compile with the respective backend.${ColorReset}")
  set(NEURALNET_BACKEND_SOURCES neuralnet/dummybackend.cpp)
else()
  message(FATAL_ERROR "Unrecognized backend: " ${USE_BACKEND})
endif()


#--------------------------- TCMALLOC ----------------------------------------------------------------------------------

if(USE_TCMALLOC)
  message(STATUS "-DUSE_TCMALLOC=1 is set, using tcmalloc as the allocator")
  find_library(TCMALLOC_LIB NAMES tcmalloc_minimal HINTS /usr)
  if(NOT TCMALLOC_LIB)
    message(FATAL_ERROR "Could not find tcmalloc")
  endif()
endif()

# set (Gperftools_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")
# find_package(Gperftools REQUIRED)

#--------------------------- GIT ---------------------------------------------------------------------------------------

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  message(STATUS "-DNO_GIT_REVISION=1 is set, avoiding including the Git revision in compiled executable")
  unset(GIT_HEADER_FILE_ALWAYS_UPDATED)
else()
  if(NO_GIT_REVISION AND BUILD_DISTRIBUTED)
    message(STATUS "${ColorRed}NO_GIT_REVISION is set, but BUILD_DISTRIBUTED is also set and distributed requires git revision, so ignoring NO_GIT_REVISION.${ColorReset}")
  elseif(BUILD_DISTRIBUTED)
    message(STATUS "Including Git revision in the compiled executable")
  else()
    message(STATUS "Including Git revision in the compiled executable, specify -DNO_GIT_REVISION=1 to disable")
  endif()
  find_package(Git)
  if(NOT GIT_FOUND)
    set(GIT_EXECUTABLE ${GIT_EXECUTABLE} CACHE FILEPATH "Path to git executable")
    mark_as_advanced(CLEAR GIT_EXECUTABLE)
    if(BUILD_DISTRIBUTED)
      message(SEND_ERROR "${ColorBoldRed}Git executable was not found, specify GIT_EXECUTABLE as the path to the git executable.${ColorReset}")
    else()
      message(SEND_ERROR "${ColorBoldRed}Git executable was not found. Either specify GIT_EXECUTABLE as the path to the git executable, or use NO_GIT_REVISION to disable.${ColorReset}")
    endif()
  endif()
  set(GIT_HEADER_FILE_TEMPLATE_BARE program/gitinfotemplate.h)
  set(GIT_HEADER_FILE_ALWAYS_UPDATED_BARE program/gitinfoupdated.h)
  set(GIT_HEADER_FILE_BARE program/gitinfo.h)
  set(GIT_HEADER_FILE_TEMPLATE ${CMAKE_SOURCE_DIR}/${GIT_HEADER_FILE_TEMPLATE_BARE})
  set(GIT_HEADER_FILE_ALWAYS_UPDATED ${CMAKE_BINARY_DIR}/${GIT_HEADER_FILE_ALWAYS_UPDATED_BARE})
  set(GIT_HEADER_FILE ${CMAKE_BINARY_DIR}/${GIT_HEADER_FILE_BARE})
  # Embeds the current git SHA into the binary on every build, while only
  # triggering recompilation of dependents when the SHA actually changes.
  #
  # How it works:
  #   - ALWAYS_UPDATED is the rule's OUTPUT and is removed at the end of the
  #     commands, so it is always missing at the start of the next build,
  #     which forces this rule to re-fire.
  #   - copy_if_different updates GIT_HEADER_FILE only when the new SHA
  #     differs from the previous one, leaving its mtime untouched otherwise.
  #
  # Both files must be listed as OUTPUTs for Ninja. Ninja's restat (set
  # automatically for custom_commands) re-checks every declared OUTPUT's
  # mtime after the rule runs, and only then propagates rebuilds. Listing
  # GIT_HEADER_FILE is what lets ninja notice an actual SHA change and
  # rebuild dependents such as main.cpp; without it, dependents stay stale
  # across commits in incremental builds. The Make generator is more
  # pessimistic about generated headers and doesn't need this, and adding
  # it causes unnecessary re-linking on every build.
  set(GIT_CUSTOM_COMMAND_OUTPUTS ${GIT_HEADER_FILE_ALWAYS_UPDATED})
  if(CMAKE_GENERATOR MATCHES "Ninja")
    list(APPEND GIT_CUSTOM_COMMAND_OUTPUTS ${GIT_HEADER_FILE})
  endif()
  add_custom_command(
    OUTPUT ${GIT_CUSTOM_COMMAND_OUTPUTS}
    COMMAND ${CMAKE_COMMAND} -E copy ${GIT_HEADER_FILE_TEMPLATE} ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    COMMAND ${GIT_EXECUTABLE} describe --match=DummyTagNotExisting --always --abbrev=40 --dirty >> ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GIT_HEADER_FILE_ALWAYS_UPDATED} ${GIT_HEADER_FILE}
    COMMAND ${CMAKE_COMMAND} -E remove ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
    )
endif()

#--------------------------- KATAGO COMPILING AND LINKING --------------------------------------------------------------

add_executable(katago
  core/global.cpp
  core/base64.cpp
  core/bsearch.cpp
  core/commandloop.cpp
  core/config_parser.cpp
  core/datetime.cpp
  core/elo.cpp
  core/fancymath.cpp
  core/fileutils.cpp
  core/hash.cpp
  core/logger.cpp
  core/mainargs.cpp
  core/makedir.cpp
  core/md5.cpp
  core/multithread.cpp
  core/parallel.cpp
  core/rand.cpp
  core/rand_helpers.cpp
  core/sha2.cpp
  core/test.cpp
  core/threadsafecounter.cpp
  core/threadsafequeue.cpp
  core/threadtest.cpp
  core/timer.cpp
  game/board.cpp
  game/rules.cpp
  game/boardhistory.cpp
  game/graphhash.cpp
  dataio/sgf.cpp
  dataio/numpywrite.cpp
  dataio/poswriter.cpp
  dataio/trainingwrite.cpp
  dataio/loadmodel.cpp
  dataio/homedata.cpp
  dataio/files.cpp
  neuralnet/nninputs.cpp
  neuralnet/sgfmetadata.cpp
  neuralnet/modelversion.cpp
  neuralnet/nneval.cpp
  neuralnet/desc.cpp
  neuralnet/debugprint.cpp
  ${NEURALNET_BACKEND_SOURCES}
  book/book.cpp
  book/bookcssjs.cpp
  search/timecontrols.cpp
  search/searchparams.cpp
  search/mutexpool.cpp
  search/search.cpp
  search/searchnode.cpp
  search/searchresults.cpp
  search/searchhelpers.cpp
  search/searchexplorehelpers.cpp
  search/searchmirror.cpp
  search/searchmultithreadhelpers.cpp
  search/searchnnhelpers.cpp
  search/searchtimehelpers.cpp
  search/searchupdatehelpers.cpp
  search/asyncbot.cpp
  search/distributiontable.cpp
  search/localpattern.cpp
  search/searchnodetable.cpp
  search/subtreevaluebiastable.cpp
  search/evalcache.cpp
  search/patternbonustable.cpp
  search/analysisdata.cpp
  search/reportedsearchvalues.cpp
  program/gtpconfig.cpp
  program/setup.cpp
  program/playutils.cpp
  program/playsettings.cpp
  program/play.cpp
  program/selfplaymanager.cpp
  ${GIT_HEADER_FILE_ALWAYS_UPDATED}
  tests/testboardarea.cpp
  tests/testboardbasic.cpp
  tests/testbook.cpp
  tests/testcommon.cpp
  tests/testconfig.cpp
  tests/testmisc.cpp
  tests/testnnevalcanary.cpp
  tests/testrules.cpp
  tests/testscore.cpp
  tests/testsgf.cpp
  tests/testsymmetries.cpp
  tests/testnninputs.cpp
  tests/testownership.cpp
  tests/testsearchcommon.cpp
  tests/testsearchnonn.cpp
  tests/testsearch.cpp
  tests/testsearchv3.cpp
  tests/testsearchv8.cpp
  tests/testsearchv9.cpp
  tests/testsearchmisc.cpp
  tests/testtime.cpp
  tests/testtrainingwrite.cpp
  tests/testnn.cpp
  tests/tinymodel.cpp
  tests/tinymodeldata.cpp
  distributed/client.cpp
  command/commandline.cpp
  command/analysis.cpp
  command/benchmark.cpp
  command/contribute.cpp
  command/evalsgf.cpp
  command/gatekeeper.cpp
  command/genbook.cpp
  command/gputest.cpp
  command/gtp.cpp
  command/match.cpp
  command/misc.cpp
  command/runtests.cpp
  command/sandbox.cpp
  command/selfplay.cpp
  command/startposes.cpp
  command/tune.cpp
  command/writetrainingdata.cpp
  main.cpp
  )

if(USE_BACKEND STREQUAL "CUDA")
  target_compile_definitions(katago PRIVATE USE_CUDA_BACKEND)
  target_compile_definitions(katago PRIVATE CUDA_TARGET_VERSION=${CMAKE_CUDA_COMPILER_VERSION})
  find_package(CUDAToolkit REQUIRED)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_INCLUDE_DIRS} PATH_SUFFIXES include)
  if((NOT CUDNN_INCLUDE_DIR))
    message(FATAL_ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is. ${ColorReset}")
  endif()
  find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_ROOT_DIR} ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES lib64)
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
  include_directories(SYSTEM external/cudnn-frontend/include) #SYSTEM suppresses warnings inside the vendored header-only cudnn_frontend library
  # cudnn_frontend's experimental OSS engine headers reference NVRTC symbols at link time
  # (we don't actually use those engines, but the references are pulled in by the header-only includes).
  target_link_libraries(katago CUDA::cublas ${CUDNN_LIBRARY} CUDA::nvrtc)
elseif(USE_BACKEND STREQUAL "TENSORRT")
  target_compile_definitions(katago PRIVATE USE_TENSORRT_BACKEND)
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES include)
  if((NOT TENSORRT_INCLUDE_DIR))
    message(FATAL_ERROR "${ColorBoldRed} NvInfer.h was NOT found, specify TENSORRT_INCLUDE_DIR to indicate where it is. ${ColorReset}")
  endif()

  # Hackily extract out the version from the TensorRT header
  # In each case, try the old format and on failure try the new format.
  file(READ "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" tensorrt_version_header)

  set(TENSORRT_VERSION_MAJOR "")
  set(TENSORRT_VERSION_MINOR "")
  set(TENSORRT_VERSION_PATCH "")

  string(REGEX MATCH "#define NV_TENSORRT_MAJOR +([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1})
  else()
    string(REGEX MATCH "#define TRT_MAJOR_ENTERPRISE +([0-9]+)" tensorrt_major_enterprise ${tensorrt_version_header})
    if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
      set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1})
    endif()
  endif()

  string(REGEX MATCH "#define NV_TENSORRT_MINOR +([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1})
  else()
    string(REGEX MATCH "#define TRT_MINOR_ENTERPRISE +([0-9]+)" tensorrt_minor_enterprise ${tensorrt_version_header})
    if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
      set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1})
    endif()
  endif()

  string(REGEX MATCH "#define NV_TENSORRT_PATCH +([0-9]+)" tensorrt_version_macro ${tensorrt_version_header})
  if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
    set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1})
  else()
    string(REGEX MATCH "#define TRT_PATCH_ENTERPRISE +([0-9]+)" tensorrt_patch_enterprise ${tensorrt_version_header})
    if(DEFINED CMAKE_MATCH_1 AND NOT "${CMAKE_MATCH_1}" STREQUAL "")
      set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1})
    endif()
  endif()

  set(TENSORRT_VERSION "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}.${TENSORRT_VERSION_PATCH}")
  message(STATUS "Detected TensorRT version: ${TENSORRT_VERSION}")

  if("${TENSORRT_VERSION_MAJOR}" STREQUAL "" OR "${TENSORRT_VERSION_MINOR}" STREQUAL "" OR "${TENSORRT_VERSION_PATCH}" STREQUAL "")
    message(FATAL_ERROR "Could not determine TensorRT version from header file")
  endif()

  find_library(TENSORRT_LIBRARY NAMES nvinfer nvinfer_${TENSORRT_VERSION_MAJOR} HINTS ${TENSORRT_ROOT_DIR} PATH_SUFFIXES lib)

  # Version 8 is required for serializing the builder timing cache.
  # Version 8.2 is required for eliminating the global logger for Builder and Runtime.
  # Version 8.5 is required for eliminating many deprecated APIs and adopting new features.
  # Version 8.6 is for CUDA 12 support and further reduction in initialization time.
  if(TENSORRT_VERSION VERSION_LESS 8.5)
    message(FATAL_ERROR "TensorRT 8.5 or greater is required but ${TENSORRT_VERSION} was found.")
  endif()
  include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
  target_link_libraries(katago CUDA::cudart_static ${TENSORRT_LIBRARY})

  # The TensorRT backend builds networks by emitting an ONNX ModelProto and handing the
  # serialized bytes to TensorRT's nvonnxparser. We compile the vendored ONNX schema
  # (cpp/external/onnx/onnx.proto) with protoc and link against our own protobuf; this is
  # symbol-isolated from the protobuf that TensorRT statically links into nvonnxparser, so
  # there is no ABI contact between the two (the handoff is serialized bytes).
  find_package(Protobuf REQUIRED)
  message(STATUS "Found Protobuf version: ${Protobuf_VERSION}")
  set(ONNX_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/onnx")
  protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS "${ONNX_PROTO_DIR}/onnx.proto")
  # protoc-generated code is not ours to lint; silence its warnings to keep build output readable.
  set_source_files_properties(${ONNX_PROTO_SRCS} PROPERTIES COMPILE_OPTIONS "-w")
  target_sources(katago PRIVATE ${ONNX_PROTO_SRCS} neuralnet/onnxmodelbuilder.cpp)
  # Generated onnx.pb.h lands in the build dir; let backend code include it.
  target_include_directories(katago SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})

  # nvonnxparser ships alongside nvinfer in the TensorRT lib dir. Derive that dir from the
  # already-resolved TENSORRT_LIBRARY so this works both with TENSORRT_ROOT_DIR and with a
  # directly-specified -DTENSORRT_LIBRARY
  get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
  find_library(TENSORRT_ONNXPARSER_LIBRARY NAMES nvonnxparser nvonnxparser_${TENSORRT_VERSION_MAJOR} HINTS ${TENSORRT_ROOT_DIR} ${TENSORRT_LIBRARY_DIR} PATH_SUFFIXES lib)
  if(NOT TENSORRT_ONNXPARSER_LIBRARY)
    message(FATAL_ERROR "${ColorBoldRed} libnvonnxparser was NOT found in the TensorRT lib dir. ${ColorReset}")
  endif()
  target_link_libraries(katago ${TENSORRT_ONNXPARSER_LIBRARY} ${Protobuf_LIBRARIES})
elseif(USE_BACKEND STREQUAL "METAL")
  target_compile_definitions(katago PRIVATE USE_METAL_BACKEND)
  target_link_libraries(katago KataGoSwift katagocoreml
    ${KATAGOCOREML_DEP_LDFLAGS}
    "-framework MetalPerformanceShaders"
    "-framework MetalPerformanceShadersGraph")
  if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    message(WARNING "Metal backend may not work optimally on Intel. ARM64 architecture is recommended.")
  endif()
elseif(USE_BACKEND STREQUAL "OPENCL")
  target_compile_definitions(katago PRIVATE USE_OPENCL_BACKEND)
  find_package(OpenCL)
  if(NOT OpenCL_FOUND)
    message(WARNING "OpenCL not found, attempting to see if CUDA exists and has OpenCL since sometimes CUDA may provide OpenCL where cmake can't find it.")
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
      message(FATAL_ERROR "OpenCL installation not found")
    else()
      message(WARNING "OpenCL not found, but found CUDA, attempting to use OpenCL via CUDA.")
    endif()
    include_directories(${OpenCL_INCLUDE_DIRS})
    include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(katago CUDA::OpenCL)
  else()
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARY})
    target_link_libraries(katago ${OpenCL_LIBRARY})
  endif()
# --------------------------- ROCM linking stage ---------------------------
elseif(USE_BACKEND STREQUAL "ROCM")
  # Macro: used in source code with #ifdef USE_ROCM_BACKEND
  target_compile_definitions(katago PRIVATE USE_ROCM_BACKEND)
  target_compile_definitions(katago PRIVATE HIP_TARGET_VERSION=${CMAKE_HIP_COMPILER_VERSION})

  string(TOLOWER "${CMAKE_HIP_ARCHITECTURES}" _gfxlist)  # e.g. "90a;942"
  # Every architecture in katago_default_hip_archs()'s full candidate list (CDNA gfx906/908/90a/942/
  # 950, RDNA1 gfx101x, RDNA2 gfx103x, RDNA3 gfx110x, RDNA3.5 gfx115x, RDNA4 gfx120x) supports packed
  # FP16 ops. The regex below matches each of them so user-specified subset lists still work.
  if(_gfxlist MATCHES "(gfx)?(90[06a]|908|94[02]|950|101[012]|103[0-6]|110[0-3]|115[0-3]|120[01])")
    target_compile_definitions(katago PRIVATE HIP_SUPPORTS_FP16)
    message(STATUS "Detected FP16‑capable GFX arch (${CMAKE_HIP_ARCHITECTURES}); defining HIP_SUPPORTS_FP16")
  endif()

  # 3) Find ROCm runtime & libraries. Since ROCm 6.x, CMake config-mode packages are included. If not found, add -DCMAKE_PREFIX_PATH=/opt/rocm (Linux) or HIP SDK path (Windows)
  find_package(hip        QUIET CONFIG)   # Export hip::device / hip::host
  find_package(hipblas    QUIET CONFIG)   # Export roc::hipblas
  find_package(miopen     QUIET CONFIG)   # Export roc::miopen or MIOpen

  # Explicitly (re-)add the resolved HIP include dir via a raw -isystem compile option (not
  # target_include_directories(... SYSTEM ...)): hip::device's own INTERFACE_INCLUDE_DIRECTORIES
  # already carries this same path as a plain -I, and CMake's include-directory de-duplication
  # drops our SYSTEM entry as "already covered" rather than upgrading it - so the plain -I from
  # hip::device wins, which matters here because some systems have a stale hip_runtime.h under
  # /usr/include (e.g. from an old standalone "hipcc"/"libamdhip64-dev" apt package) that's
  # incompatible with a newer compiler's builtins (__AMDGCN_WAVEFRONT_SIZE etc). A plain -I loses to
  # that stale /usr/include; a raw -isystem (bypassing the dedup) wins.
  find_path(KATAGO_HIP_SYSTEM_INCLUDE_DIR hip/hip_runtime.h
    HINTS ${CMAKE_PREFIX_PATH} ENV HIP_PATH ENV ROCM_PATH
    PATH_SUFFIXES include)
  if(KATAGO_HIP_SYSTEM_INCLUDE_DIR)
    target_compile_options(katago PRIVATE "-isystem${KATAGO_HIP_SYSTEM_INCLUDE_DIR}")
  endif()

  # ---------- fallback：HIP Runtime ----------
  if(NOT hip_FOUND)
    if(WIN32)
      # Windows: Search in HIP SDK installation
      find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
                HINTS ${CMAKE_PREFIX_PATH} ENV HIP_PATH ENV ROCM_PATH
                PATH_SUFFIXES include)
      find_library(HIP_RUNTIME_LIB 
                   NAMES amdhip64 amdhip64_6
                   HINTS ${CMAKE_PREFIX_PATH} ENV HIP_PATH ENV ROCM_PATH
                   PATH_SUFFIXES lib bin)
    else()
      # Linux: Search in /opt/rocm
      find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
                HINTS ${CMAKE_PREFIX_PATH} /opt/rocm
                PATH_SUFFIXES include)
      find_library(HIP_RUNTIME_LIB amdhip64
                   HINTS ${CMAKE_PREFIX_PATH} /opt/rocm
                   PATH_SUFFIXES lib lib64)
    endif()
    
    if(NOT HIP_INCLUDE_DIR OR NOT HIP_RUNTIME_LIB)
      if(WIN32)
        message(FATAL_ERROR "HIP headers or runtime NOT found; install HIP SDK for Windows or set CMAKE_PREFIX_PATH to HIP SDK installation path.")
      else()
        message(FATAL_ERROR "HIP headers or runtime NOT found; install ROCm or set CMAKE_PREFIX_PATH.")
      endif()
    endif()
    add_library(hip::device UNKNOWN IMPORTED)
    set_target_properties(hip::device PROPERTIES
      IMPORTED_LOCATION "${HIP_RUNTIME_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}")
    target_include_directories(katago SYSTEM PRIVATE ${HIP_INCLUDE_DIR})
  endif()

  # ---------- fallback：hipBLAS / MIOpen ----------
  foreach(_pkg hipblas miopen)
    if(NOT ${_pkg}_FOUND)
      if(WIN32)
        # Windows naming conventions
        if(_pkg STREQUAL "hipblas")
          set(_lib_names hipblas)
        else()
          set(_lib_names MIOpen)
        endif()
        find_library(${_pkg}_LIB 
                     NAMES ${_lib_names}
                     HINTS ${CMAKE_PREFIX_PATH} ENV HIP_PATH ENV ROCM_PATH
                     PATH_SUFFIXES lib bin)
      else()
        # Linux naming
        find_library(${_pkg}_LIB ${_pkg}
                     HINTS ${CMAKE_PREFIX_PATH} /opt/rocm
                     PATH_SUFFIXES lib lib64)
      endif()
      
      if(${_pkg}_LIB)
        add_library(roc::${_pkg} UNKNOWN IMPORTED)
        set_target_properties(roc::${_pkg} PROPERTIES
          IMPORTED_LOCATION "${${_pkg}_LIB}")
        target_include_directories(katago SYSTEM PRIVATE ${HIP_INCLUDE_DIR})
        message(STATUS "Found ${_pkg} at ${${_pkg}_LIB}")
      else()
        if(WIN32)
          message(FATAL_ERROR "Required ROCm component ${_pkg} not found – install HIP SDK for Windows or set CMAKE_PREFIX_PATH.")
        else()
          message(FATAL_ERROR "Required ROCm component ${_pkg} not found – install it or set CMAKE_PREFIX_PATH.")
        endif()
      endif()
    endif()
  endforeach()

  # 4) Link libraries
  # Note: On Windows, MIOpen might need to be linked as "MIOpen" directly if the target doesn't exist
  if(TARGET MIOpen)
    set(_miopen_target MIOpen)
  elseif(TARGET roc::miopen)
    set(_miopen_target roc::miopen)
  else()
    message(FATAL_ERROR "Neither 'MIOpen' nor 'roc::miopen' imported target exists after find_package + fallback. This is a bug in the ROCM backend detection logic in cpp/CMakeLists.txt.")
  endif()
  
  target_link_libraries(katago
    hip::device          # HIP runtime & kernel offload
    roc::hipblas         # BLAS
    ${_miopen_target}    # DNN primitives
  )

  # See the comment in core/win_rocm_sse_shim.cpp: on Windows, ucrt's own <wchar.h> ends up calling
  # SSE2 intrinsics that resolve to bodyless declarations under this toolchain, causing "undefined
  # symbol" link errors. This file is a no-op everywhere except Windows (guarded internally).
  if(WIN32)
    target_sources(katago PRIVATE core/win_rocm_sse_shim.cpp)
  endif()

  # KATAGO_ROCM_PREFIX may not be set if the user passed an explicit -DCMAKE_PREFIX_PATH (which
  # skips the auto-detect branches that populate it) - fall back to CMAKE_PREFIX_PATH's first entry
  # in that case, so RPATH still gets set below regardless of how the prefix was determined.
  if(NOT KATAGO_ROCM_PREFIX)
    if(CMAKE_PREFIX_PATH)
      list(GET CMAKE_PREFIX_PATH 0 KATAGO_ROCM_PREFIX)
    else()
      katago_find_rocm_prefix(KATAGO_ROCM_PREFIX)
    endif()
  endif()

  # Bake the resolved ROCm lib dir into the binary's RPATH. Without this, the built katago relies
  # on /opt/rocm (a distro-managed symlink that can get silently repointed by installing another
  # ROCm package, or a subsequent /opt/rocm/core-<newer-version> release) resolving to the same
  # install this was actually built and tested against - which is not guaranteed, especially for a
  # binary meant to be copied to another machine. CMAKE_SKIP_BUILD_RPATH/INSTALL_RPATH aren't set by
  # this project, so this also takes effect for the build-tree binary, not just `make install`.
  if(KATAGO_ROCM_PREFIX)
    set_target_properties(katago PROPERTIES
      BUILD_RPATH "${KATAGO_ROCM_PREFIX}/lib"
      INSTALL_RPATH "${KATAGO_ROCM_PREFIX}/lib"
    )
    message(STATUS "Set katago RPATH to ${KATAGO_ROCM_PREFIX}/lib")
  endif()

  # ---------- Optional: Composable Kernel FMHA (fused attention) ----------
  # Mirrors the CUDA backend's optional cudnn-frontend SDPA path: KataGo vendors a small set of
  # glue headers + pre-generated ck_tile FMHA kernel instantiations under
  # external/composable_kernel_fmha, but these only compile against a matching version of CK's own
  # core ck_tile headers (system dependency, not vendored - part of a "composablekernel-dev" /
  # "amdrocm-ck*" package). If a compatible version isn't installed, we just skip this and the
  # backend always uses its own plain (non-fused) attention kernel - this is a pure performance
  # optimization, not required for correctness.
  find_path(KATAGO_CK_TILE_INCLUDE_DIR
    NAMES ck_tile/ops/fmha_fwd.hpp
    HINTS ${CMAKE_PREFIX_PATH} ENV ROCM_PATH
    PATH_SUFFIXES include
  )
  if(KATAGO_CK_TILE_INCLUDE_DIR)
    message(STATUS "Found ck_tile headers at ${KATAGO_CK_TILE_INCLUDE_DIR}; enabling optional CK FMHA fused attention path")
    file(GLOB KATAGO_CK_FMHA_GENERATED_SOURCES external/composable_kernel_fmha/generated/*.cpp)
    # ck_tile's own arch.hpp (get_compiler_target()) has no case at all for gfx906 or RDNA1
    # (gfx1010/1011/1012) - its FMHA kernels were never ported to these (no MFMA/WMMA), so its
    # constexpr dispatch function silently falls through to nothing on a device-compile pass for
    # them, and merely including ck_tile headers then hard-fails ("member reference base type
    # 'void' is not a structure or union"). These generated files exist solely to provide CK kernel
    # instantiations, so simply exclude them from those particular archs' compilation entirely via
    # a dedicated object library with a narrowed HIP_ARCHITECTURES - the plain (non-fused) attention
    # kernel already covers those archs regardless.
    set(KATAGO_CK_FMHA_SUPPORTED_ARCHS ${CMAKE_HIP_ARCHITECTURES})
    list(REMOVE_ITEM KATAGO_CK_FMHA_SUPPORTED_ARCHS gfx906 gfx1010 gfx1011 gfx1012)
    set_source_files_properties(${KATAGO_CK_FMHA_GENERATED_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(katago_ck_fmha_kernels OBJECT ${KATAGO_CK_FMHA_GENERATED_SOURCES})
    set_target_properties(katago_ck_fmha_kernels PROPERTIES HIP_ARCHITECTURES "${KATAGO_CK_FMHA_SUPPORTED_ARCHS}")
    # SYSTEM/-isystem is required, not just for warning suppression: ck_tile's own headers
    # #include <hip/hip_runtime.h>, and on systems with a stale libamdhip64-dev under /usr/include,
    # a plain -I here still loses to /usr/include for angle-bracket resolution. -isystem does not.
    target_include_directories(katago_ck_fmha_kernels SYSTEM PRIVATE ${KATAGO_CK_TILE_INCLUDE_DIR} external/composable_kernel_fmha)
    target_link_libraries(katago katago_ck_fmha_kernels)
    target_include_directories(katago SYSTEM PRIVATE ${KATAGO_CK_TILE_INCLUDE_DIR} external/composable_kernel_fmha)
    target_compile_definitions(katago PRIVATE KATAGO_ROCM_HAS_CK_FMHA=1)
    # rocmbackend.cpp itself (unlike the generated files above) still needs to compile for every
    # arch, since it also holds the non-CK ROCm backend logic - it guards its own CK usage per-arch
    # internally (see KATAGO_ROCM_CK_FMHA_ARCH_OK in rocmbackend.cpp) rather than being excluded here.
    if(WIN32)
      set_source_files_properties(neuralnet/rocmbackend.cpp PROPERTIES LANGUAGE HIP)
    endif()
  else()
    message(STATUS "ck_tile headers not found; ROCm backend will only use its built-in (non-fused) attention kernel")
    target_compile_definitions(katago PRIVATE KATAGO_ROCM_HAS_CK_FMHA=0)
  endif()
elseif(USE_BACKEND STREQUAL "EIGEN")
  target_compile_definitions(katago PRIVATE USE_EIGEN_BACKEND)
  # Allow EIGEN3_INCLUDE_DIRS as a manual override (for users who downloaded
  # Eigen3 and just want to point at the unpacked directory).
  if(EIGEN3_INCLUDE_DIRS)
    message(STATUS "Using user-provided EIGEN3_INCLUDE_DIRS=${EIGEN3_INCLUDE_DIRS}")
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
  else()
    if(NOT (MSVC))
      find_package(Eigen3 REQUIRED)
    else()
      find_package(Eigen3)
      if(NOT Eigen3_FOUND)
        set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS} CACHE PATH "Directory containing 'Eigen' and 'unsupported' subdirs with Eigen headers")
        message(FATAL_ERROR "*** KataGo: Eigen3 was not found. If you've downloaded Eigen3, you can IGNORE the whole above error from CMake and just set 'EIGEN3_INCLUDE_DIRS' to the root eigen3-3.*.* directory after unzipping it. Among other things it should contain 'Eigen' and 'unsupported' subdirs, which is all KataGo needs. Otherwise, you will need to 'build' Eigen3 even though it is header-only and install it via MSVC in order for cmake to find it via FindEigen3.cmake.")
      endif()
    endif()
    # Modern Eigen3Config.cmake (Homebrew, apt, vcpkg, ...) only exports the
    # imported target Eigen3::Eigen with INTERFACE_INCLUDE_DIRECTORIES set; the
    # legacy variable EIGEN3_INCLUDE_DIRS is not always populated. Prefer the
    # target where available, fall back to the variable for older installs.
    if(TARGET Eigen3::Eigen)
      # Plain (non-keyword) form to match the rest of KataGo's CMakeLists,
      # which doesn't use PRIVATE/PUBLIC/INTERFACE on target_link_libraries.
      target_link_libraries(katago Eigen3::Eigen)
      message(STATUS "Found Eigen3 (using Eigen3::Eigen imported target)")
    else()
      include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
      message(STATUS "Found Eigen3 at ${EIGEN3_INCLUDE_DIRS}")
    endif()
  endif()

endif()

if(USE_BIGGER_BOARDS_EXPENSIVE)
  target_compile_definitions(katago PRIVATE COMPILE_MAX_BOARD_LEN=50)
endif()

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  target_compile_definitions(katago PRIVATE NO_GIT_REVISION)
endif()

# On Windows ROCm builds, zlib is bundled inside the HIP SDK (TheRock layout) - try that first.
if(WIN32 AND USE_BACKEND STREQUAL "ROCM" AND DEFINED ENV{HIP_PATH})
  if(NOT ZLIB_INCLUDE_DIR AND EXISTS "$ENV{HIP_PATH}/lib/rocm_sysdeps/include/zlib.h")
    set(ZLIB_INCLUDE_DIR "$ENV{HIP_PATH}/lib/rocm_sysdeps/include" CACHE PATH "" FORCE)
  endif()
  if(NOT ZLIB_LIBRARY)
    foreach(_zlib_name "zlibstatic.lib" "zlib.lib" "zlibstaticd.lib")
      if(EXISTS "$ENV{HIP_PATH}/lib/rocm_sysdeps/lib/${_zlib_name}")
        set(ZLIB_LIBRARY "$ENV{HIP_PATH}/lib/rocm_sysdeps/lib/${_zlib_name}" CACHE FILEPATH "" FORCE)
        break()
      endif()
    endforeach()
  endif()
endif()

# TheRock's Windows package (as of ~7.13) ships zlib.h under rocm_sysdeps/include but no linkable
# .lib anymore (the block above then leaves ZLIB_LIBRARY unset) - auto-fetch zlib via vcpkg instead
# of requiring the user to separately install a package manager (see KATAGO_AUTO_FETCH_DEPS above).
set(KATAGO_ZLIB_IS_VCPKG_DLL FALSE)
if(KATAGO_AUTO_FETCH_DEPS AND (NOT ZLIB_INCLUDE_DIR OR NOT ZLIB_LIBRARY))
  katago_vcpkg_install_if_needed("zlib")
  set(_katago_vcpkg_installed_root "${KATAGO_VCPKG_ROOT}/installed/${KATAGO_VCPKG_TRIPLET}")
  if(NOT ZLIB_INCLUDE_DIR AND EXISTS "${_katago_vcpkg_installed_root}/include/zlib.h")
    set(ZLIB_INCLUDE_DIR "${_katago_vcpkg_installed_root}/include" CACHE PATH "Path to directory with zlib.h and other header files" FORCE)
  endif()
  if(NOT ZLIB_LIBRARY)
    find_library(_katago_zlib_lib NAMES zlib z zlibstatic HINTS "${_katago_vcpkg_installed_root}/lib" NO_DEFAULT_PATH)
    if(_katago_zlib_lib)
      set(ZLIB_LIBRARY "${_katago_zlib_lib}" CACHE FILEPATH "Path to 'libz.so' on Linux or 'libz.lib' on Windows" FORCE)
      # vcpkg's default triplets (e.g. x64-windows) build zlib as a DLL, not a static lib - the
      # .lib found above is just an import lib, so the matching runtime DLL needs to be copied
      # next to katago.exe at build time (see the POST_BUILD step further below).
      if(WIN32)
        set(KATAGO_ZLIB_IS_VCPKG_DLL TRUE)
      endif()
    endif()
  endif()
endif()

find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  target_link_libraries(katago ${ZLIB_LIBRARIES})
else()
  set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIR} CACHE PATH "Path to directory with zlib.h and other header files")
  set(ZLIB_LIBRARY ${ZLIB_LIBRARY} CACHE FILEPATH "Path to 'libz.so' on Linux or 'libz.lib' on Windows")
  mark_as_advanced(CLEAR ZLIB_INCLUDE_DIR ZLIB_LIBRARY)
  message(SEND_ERROR "${ColorBoldRed}zlib was not found, if zlib is actually installed but not being found you can set ZLIB_INCLUDE_DIR to the directory with zlib.h and other headers, and ZLIB_LIBRARY to the compiled library 'libz.so' on Linux or 'libz.lib' on Windows. On the command line, this is -DZLIB_INCLUDE_DIR=... and -DZLIB_LIBRARY=... ${ColorReset}")
endif(ZLIB_FOUND)

if(KATAGO_ZLIB_IS_VCPKG_DLL)
  get_filename_component(_katago_zlib_lib_dir "${ZLIB_LIBRARY}" DIRECTORY)
  get_filename_component(_katago_vcpkg_pkg_root "${_katago_zlib_lib_dir}" DIRECTORY)
  file(GLOB _katago_zlib_dlls "${_katago_vcpkg_pkg_root}/bin/*.dll")
  foreach(_katago_zlib_dll IN LISTS _katago_zlib_dlls)
    add_custom_command(TARGET katago POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_katago_zlib_dll}" "$<TARGET_FILE_DIR:katago>"
      VERBATIM)
  endforeach()
endif()

# On Windows ROCm builds, copy the HIP/MIOpen/rocBLAS runtime DLLs next to katago.exe so it runs
# without the user needing to manually copy anything or add HIP_PATH/bin to PATH themselves.
# amdhip64_*.dll in particular must be copied (not just PATH-resolved): AMD's GPU driver installs
# its own, potentially older/incompatible copy into System32, which would otherwise take priority.
if(WIN32 AND USE_BACKEND STREQUAL "ROCM" AND DEFINED ENV{HIP_PATH})
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" _katago_hip_path)
  file(GLOB _katago_rocm_runtime_dlls
    "${_katago_hip_path}/bin/amdhip64_*.dll"
    "${_katago_hip_path}/bin/hipblas.dll"
    "${_katago_hip_path}/bin/MIOpen.dll"
    "${_katago_hip_path}/bin/rocblas.dll"
    "${_katago_hip_path}/bin/libhipblaslt.dll"
    "${_katago_hip_path}/bin/amdocl64.dll"
    "${_katago_hip_path}/bin/hiprtc*.dll"
    "${_katago_hip_path}/bin/amd_comgr*.dll"
  )
  foreach(_katago_rocm_dll IN LISTS _katago_rocm_runtime_dlls)
    add_custom_command(TARGET katago POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_katago_rocm_dll}" "$<TARGET_FILE_DIR:katago>"
      VERBATIM)
  endforeach()
  # rocBLAS/hipBLASLt look for their kernel library files relative to their own DLL's location.
  foreach(_katago_rocm_lib_subdir IN ITEMS rocblas hipblaslt)
    if(IS_DIRECTORY "${_katago_hip_path}/bin/${_katago_rocm_lib_subdir}/library")
      add_custom_command(TARGET katago POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${_katago_hip_path}/bin/${_katago_rocm_lib_subdir}/library"
          "$<TARGET_FILE_DIR:katago>/${_katago_rocm_lib_subdir}/library"
        VERBATIM)
    endif()
  endforeach()
endif()

find_library(LIBZIP_LIBRARY NAMES zip)
find_path(LIBZIP_INCLUDE_DIR_ZIP NAMES zip.h)
find_path(LIBZIP_INCLUDE_DIR_ZIPCONF NAMES zipconf.h)
if((NOT LIBZIP_LIBRARY) OR (NOT LIBZIP_INCLUDE_DIR_ZIP) OR (NOT LIBZIP_INCLUDE_DIR_ZIPCONF))
  if(BUILD_DISTRIBUTED)
    message(SEND_ERROR "${ColorBoldRed}WARNING: BUILD_DISTRIBUTED was requested but libzip library was NOT found. KataGo needs this for writing training data so libzip is required. On Linux, install through your normal package manager. On Windows, set LIBZIP_INCLUDE_DIR_ZIP to the directory that includes zip.h and other files, and LIBZIP_INCLUDE_DIR_ZIPCONF to the directory that includes zipconf.h and other files, and LIBZIP_LIBRARY to the libzip.lib or zip.lib file. ${ColorReset}")
  endif()
  target_compile_definitions(katago PRIVATE NO_LIBZIP)
  message(WARNING "${ColorBoldRed}WARNING: libzip library was NOT found. KataGo should still work for GTP/matches/analysis if everything else is good, but selfplay for writing training data will not be possible.${ColorReset}")
  set(LIBZIP_INCLUDE_DIR_ZIP ${LIBZIP_INCLUDE_DIR_ZIP} CACHE PATH "Path to directory with zip.h and other header files")
  set(LIBZIP_INCLUDE_DIR_ZIPCONF ${LIBZIP_INCLUDE_DIR_ZIPCONF} CACHE PATH "Path to directory with zipconf.h and other header files")
  set(LIBZIP_LIBRARY ${LIBZIP_LIBRARY} CACHE FILEPATH "Path to 'libzip.so' on Linux or 'libzip.lib' or 'zip.lib' on Windows")
  mark_as_advanced(CLEAR LIBZIP_INCLUDE_DIR_ZIP LIBZIP_INCLUDE_DIR_ZIPCONF LIBZIP_LIBRARY)
else()
  include_directories(${LIBZIP_INCLUDE_DIR_ZIP})
  include_directories(${LIBZIP_INCLUDE_DIR_ZIPCONF})
  target_link_libraries(katago ${LIBZIP_LIBRARY})
endif()

if(USE_TCMALLOC)
  target_link_libraries(katago ${TCMALLOC_LIB})
endif(USE_TCMALLOC)

if(BUILD_DISTRIBUTED)
  message(STATUS "-DBUILD_DISTRIBUTED=1 is set, compiling code and dependencies to contribute to distributed training")
  target_compile_definitions(katago PRIVATE BUILD_DISTRIBUTED)
  find_package(OpenSSL REQUIRED)
  target_link_libraries(katago ${OPENSSL_SSL_LIBRARIES} ${OPENSSL_CRYPTO_LIBRARIES})
  include_directories(${OPENSSL_INCLUDE_DIR})
  include_directories(external/httplib)
endif()

#------------------------------------------------------------------------------------

if (WIN32) # handles both x86 and x64
  message(STATUS "Setting up build for Windows.")
  # Suppress min and max macros on windows
  # Also define a few other things
  target_compile_definitions(katago PRIVATE NOMINMAX)
  target_compile_definitions(katago PRIVATE BYTE_ORDER=1234)
  target_compile_definitions(katago PRIVATE LITTLE_ENDIAN=1234)
  target_compile_definitions(katago PRIVATE BIG_ENDIAN=4321)
  # core/rand.cpp uses winsock for a gethostname
  target_link_libraries(katago ws2_32)
  if (MINGW)
    message(STATUS "Setting up build for mingw.")
    if(USE_BACKEND STREQUAL "CUDA" OR USE_BACKEND STREQUAL "TENSORRT")
      message(SEND_ERROR "CUDA and TENSORRT MinGW backends are not supported by NVIDIA, see https://forums.developer.nvidia.com/t/cuda-with-mingw-how-to-get-cuda-running-under-mingw")
    endif()

    # Make the build independent of MinGW dynamic libraries
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
      # Strip symbols. It significantly reduces binary size (makes it 20x smaller)
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
    endif()

    # It's needed for linking `PathRemoveFileSpecW` and probably other Windows API functions
    target_link_libraries(katago shlwapi)

    # For some reason MINGW doesn't define `HAVE_LONG_LONG` directive on his own unlike MSVC
    # Define it because TCLAP library needs it. Actually it's part of standard and should be supported everywhere
    include(CheckTypeSize)
    check_type_size("long long" HAVE_LONG_LONG)
    if(HAVE_LONG_LONG)
      target_compile_definitions(katago PRIVATE HAVE_LONG_LONG)
    endif()
  endif()
endif()

if(MSVC)
  message(STATUS "Setting up build for MSVC.")
  # Go ahead and suppress some MSVC warnings about sprintf and similar
  # things. They might be useful in some cases, but also are noisy.
  target_compile_definitions(katago PRIVATE _CRT_SECURE_NO_WARNINGS)

  if(USE_AVX2)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /arch:AVX2 -D__FMA__")
    target_compile_definitions(katago PRIVATE USE_AVX2)
  endif()

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:8388608")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  message(STATUS "Setting up build for GNU, Clang or MinGW.")
  if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(arm|aarch32|aarch64)") AND NOT USE_BACKEND STREQUAL "ROCM")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse")
  else()
    # For ARM architecture, as a hack, ensure that char is signed
    message(STATUS "ARM architecture detected: adding -fsigned-char flag")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")
  endif()
  if(USE_AVX2)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    target_compile_definitions(katago PRIVATE USE_AVX2)
  endif()

  find_package (Threads REQUIRED)
  target_link_libraries(katago Threads::Threads)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Wmissing-declarations -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-overflow=1 -Wswitch-default -Wfloat-conversion -Wunused")
  set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -fno-omit-frame-pointer")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O2")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

  if(NOT CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
    message(STATUS "CXX flags from build type: ${CMAKE_CXX_FLAGS_${_bt}}")
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Enabling GNU-specific build options.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wlogical-op -Wnoexcept -Wstrict-null-sentinel")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 6.1)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Walloc-zero -Wduplicated-branches -Wduplicated-cond -Wdangling-else -Wrestrict")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Enabling AppleClang-specific build options.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Wdangling-else -Wno-deprecated-declarations -Wno-nullability-extension -Wno-cast-qual -Wno-c11-extensions")
  else()
    message(STATUS "Enabling Clang-specific build options.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnull-dereference -Wdangling-else")
    if(NOT WIN32)
      # libatomic is a Linux GCC/Clang runtime; not needed (or available) on Windows
      target_link_libraries(katago "atomic")
    endif()
  endif()

  if(USE_TCMALLOC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
  endif()

endif()

target_include_directories(katago PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# On Windows ROCm builds, clang compiles all files with -x hip which pulls in
# MSVC-compatibility headers that require Windows SDK ucrt/shared/um headers.
if(WIN32 AND USE_BACKEND STREQUAL "ROCM")
  # Prefer the KATAGO_WINSDK_ROOT detected earlier in the pre-project() block
  if(NOT KATAGO_WINSDK_ROOT)
    get_filename_component(_winsdk_root2
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]"
      ABSOLUTE)
    if(EXISTS "${_winsdk_root2}")
      set(KATAGO_WINSDK_ROOT "${_winsdk_root2}" CACHE INTERNAL "")
    else()
      katago_win_winsdk_fallback_dirs(_winsdk_candidates2)
      foreach(_p IN LISTS _winsdk_candidates2)
        if(EXISTS "${_p}")
          set(KATAGO_WINSDK_ROOT "${_p}" CACHE INTERNAL "")
          break()
        endif()
      endforeach()
    endif()
  endif()
  if(KATAGO_WINSDK_ROOT)
    # Pick highest version directory
    file(GLOB _sdk_ver_dirs "${KATAGO_WINSDK_ROOT}/Include/*/ucrt")
    if(_sdk_ver_dirs)
      list(SORT _sdk_ver_dirs ORDER DESCENDING)
      list(GET _sdk_ver_dirs 0 _ucrt_dir)
      get_filename_component(_sdk_ver_dir "${_ucrt_dir}" DIRECTORY)
      set(_winsdk_include "${_sdk_ver_dir}")
      message(STATUS "Auto-detected Windows SDK include root: ${_winsdk_include}")
      target_include_directories(katago PRIVATE
        "${_winsdk_include}/ucrt"
        "${_winsdk_include}/shared"
        "${_winsdk_include}/um"
      )
    endif()
  endif()
endif()

