cmake_minimum_required(VERSION 3.18.2)

# Helper: produce the default broad list of AMD GPU architectures the ROCm
# backend should target when the user has not passed -DCMAKE_HIP_ARCHITECTURES=...
# Required archs are always included, being well-established across many ROCm releases. Optional
# archs are probed via an actual compile test and only included if the installed toolchain accepts
# them, which covers newer or rarer archs without hard-failing the whole build on an older
# toolchain that doesn't know about them yet. Keep this list in sync with whatever
# `apt-cache search '^amdrocm-core-sdk<ver>-gfx'` shows as AMD adds new chips, and with the CK
# codegen's own supported archs in cpp/external/composable_kernel_fmha/README.md.
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
    # -nogpulib -nogpuinc: the probe only tests whether the compiler recognizes
    # the arch name (unknown archs fail at the driver stage regardless). Pulling
    # in the HIP headers and device libs would make every probe fail whenever
    # clang can't autodetect the ROCm path, e.g. an SDK outside /opt/rocm,
    # silently shrinking the default arch list to just the required archs.
    foreach(_arch IN LISTS _optional_archs)
      execute_process(
        COMMAND "${CMAKE_HIP_COMPILER}" --offload-arch=${_arch} -nogpulib -nogpuinc -x hip -c
                "${_probe_src}" -o "${_probe_dir}/probe_${_arch}.o"
        RESULT_VARIABLE _rc
        OUTPUT_QUIET ERROR_QUIET)
      if(_rc EQUAL 0)
        list(APPEND _result ${_arch})
        message(STATUS "HIP arch ${_arch} accepted by compiler - including")
      else()
        message(STATUS "HIP arch ${_arch} not accepted by compiler - skipping")
      endif()
    endforeach()
  else()
    message(STATUS "CMAKE_HIP_COMPILER not yet known; optional archs not probed")
  endif()

  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

# Helper: resolve the actual ROCm install prefix on Linux. Since ROCm ~7.9, AMD's TheRock-style
# packaging puts real per-version installs under /opt/rocm/core-<version>/ (e.g.
# /opt/rocm/core-7.13), with /opt/rocm itself just a distro-managed symlink/alternatives entry that
# can point anywhere (or get silently repointed by installing another package) - relying on it
# directly risks intermittently picking up a stale/mismatched ROCm install.
# Prefer the highest-versioned /opt/rocm/core-* directory that is actually a COMPLETE install,
# meaning it contains the hip-lang CMake package. AMD's other packaging generation also creates
# core-<ver> dirs that are partial overlays with just runtime components and no headers or CMake
# packages, and preferring one of those breaks the build. Otherwise fall back to the flat
# /opt/rocm layout.
function(katago_find_rocm_prefix out_var)
  set(_result "")
  if(EXISTS "/opt/rocm")
    file(GLOB _core_dirs "/opt/rocm/core-*")
    set(_best_ver "")
    foreach(_dir ${_core_dirs})
      if(IS_DIRECTORY "${_dir}"
         AND (EXISTS "${_dir}/lib/cmake/hip-lang/hip-lang-config.cmake"
              OR EXISTS "${_dir}/lib64/cmake/hip-lang/hip-lang-config.cmake"))
        get_filename_component(_name "${_dir}" NAME)
        string(REGEX REPLACE "^core-" "" _ver "${_name}")
        # Compare versions directly rather than list(SORT)ing combined "<ver>|<dir>" strings -
        # string sorting mis-orders e.g. "7" vs "7.14" because of the characters after the version.
        if(_best_ver STREQUAL "" OR _ver VERSION_GREATER _best_ver)
          set(_best_ver "${_ver}")
          set(_result "${_dir}")
        endif()
      endif()
    endforeach()
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
  # Walk the dump one line at a time with plain string operations, not by iterating a cmake list
  # from file(STRINGS). Two properties of this output make list handling wrong: INCLUDE, LIB, and
  # PATH are themselves semicolon-separated, and many Windows values end in a backslash
  # (VSINSTALLDIR, WindowsSdkBinPath, ...). A trailing backslash escapes the list separator that
  # follows it, silently gluing the next variable onto this one's value and losing that variable
  # entirely. A corrupted VSINSTALLDIR then breaks -vcvars_ver resolution in every later vcvarsall
  # call, which is exactly what probing more than one toolset in a single configure run does.
  file(READ "${_out_file}" _env_raw)
  string(REPLACE "\r\n" "\n" _env_raw "${_env_raw}")
  while(NOT _env_raw STREQUAL "")
    string(FIND "${_env_raw}" "\n" _nl)
    if(_nl EQUAL -1)
      set(_line "${_env_raw}")
      set(_env_raw "")
    else()
      string(SUBSTRING "${_env_raw}" 0 ${_nl} _line)
      math(EXPR _nl_next "${_nl} + 1")
      string(SUBSTRING "${_env_raw}" ${_nl_next} -1 _env_raw)
    endif()
    if(_line MATCHES "^([A-Za-z_][A-Za-z0-9_().]*)=(.*)$")
      set(ENV{${CMAKE_MATCH_1}} "${CMAKE_MATCH_2}")
    endif()
  endwhile()
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
  # gfx900 just gives --offload-arch a valid value so the HIP compile path runs at all. What is
  # actually under test is the host-side <cmath> resolution, so the arch need not match any
  # installed GPU. This is unrelated to katago_default_hip_archs()'s per-arch probing for the
  # final build's architecture list.
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

# Helper: find an MSVC toolset (possibly one of several versions installed side-by-side under one
# or more VS installs) whose STL doesn't conflict with the HIP compiler's own CUDA/HIP math
# forward-declare headers, apply the vcvarsall.bat environment for it (so the rest of THIS
# configure run - the HIP arch probing below, and project()'s own HIP compiler ABI test - see it),
# and also bake the resulting INCLUDE/LIB paths into cached compiler/linker flags so a later,
# separate `ninja` invocation (which does NOT inherit this configure run's environment - see
# katago_win_apply_vcvarsall's comment) keeps working too.
#
# Every installed toolset is a candidate, probed newest first, and the compile probe rather than
# a hardcoded version range decides which one gets used. The conflict the probe catches: newer
# MSVC STLs (first seen with MSVC 14.51 in a VS 18 preview) declare math functions like
# isgreater/isless in a way that clashes with clang's forward declarations for CUDA/HIP device
# overloads ("device function cannot overload host device function") whenever the clang predates
# those STL declarations. That is a property of the specific STL/clang pair, not of a toolset
# version family, so a newer TheRock clang can start accepting newer toolsets without needing an
# edit here. If the newest toolset fails the probe, successively older ones are tried.
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
        set(_vcvarsall "${_install_path}/VC/Auxiliary/Build/vcvarsall.bat")
        if(EXISTS "${_vcvarsall}")
          list(APPEND _toolsets "${_ver}|${_vcvarsall}")
        endif()
      endif()
    endforeach()
  endforeach()

  if(NOT _toolsets)
    message(FATAL_ERROR "No installed MSVC toolset found under any Visual Studio installation. Open the Visual Studio Installer, choose \"Modify\" on your VS installation, go to \"Individual Components\", and install an \"MSVC ... C++ x64/x86 build tools\" component.")
  endif()
  list(REMOVE_DUPLICATES _toolsets)
  # Newest first: prefer the most recent toolset installed, falling back to older ones only if the
  # probe below rejects it.
  list(SORT _toolsets COMPARE NATURAL ORDER DESCENDING)

  # vcvarsall.bat prepends to these rather than replacing them, so probing a second toolset in
  # the same cmake process would otherwise leave the rejected toolset's directories trailing
  # behind the accepted one's and bake those leftovers into the cached flags below. Snapshot the
  # pre-vcvars values and restore them before each attempt so every probe starts from the same
  # clean environment. Each value is held in its own variable rather than in one cmake list,
  # since the values are themselves semicolon-separated.
  foreach(_env_var PATH INCLUDE LIB LIBPATH)
    set(_orig_env_${_env_var} "$ENV{${_env_var}}")
  endforeach()

  foreach(_entry IN LISTS _toolsets)
    foreach(_env_var PATH INCLUDE LIB LIBPATH)
      set(ENV{${_env_var}} "${_orig_env_${_env_var}}")
    endforeach()
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
      # most of this, while still falling through to the MSVC/SDK -I entries below for headers
      # clang has no equivalent for (the actual MSVC STL, ucrt, etc). This needs a plain -I (not
      # -isystem/-idirafter): the earlier __clang_hip_runtime_wrapper.h force-included in every HIP
      # compilation uses #include_next to reach the "real" <cmath> starting from wherever clang's
      # own cuda_wrappers/cmath shim was found, and that continuation only walks through the same
      # -I search list, not -isystem/-idirafter entries.
      #
      # This ordering is not sufficient on its own, and does not by itself decide which <intrin.h>
      # a given HIP SDK ends up using. On SDKs that resolve it to MSVC's, objects still reference
      # the bodyless declarations, and core/win_rocm_sse_shim.cpp supplies the definitions that let
      # them link. Do not drop that file on the assumption that this ordering replaces it.
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
  message(FATAL_ERROR "None of the installed MSVC toolsets compiled cleanly against TheRock's HIP compiler. A newer MSVC STL can conflict with an older bundled clang (\"device function cannot overload host device function\"); if every installed toolset is affected, install an older one - e.g. \"MSVC v143 - VS 2022 C++ x64/x86 build tools\" from the Visual Studio Installer's \"Individual Components\" tab - or use a newer TheRock distribution.")
endfunction()

# Normalize case here too (it is normalized again later next to the cache definition): the
# pre-project blocks below dispatch on USE_BACKEND before that later normalization runs, and
# e.g. -DUSE_BACKEND=rocm would otherwise silently skip all the ROCm pre-project setup.
string(TOUPPER "${USE_BACKEND}" USE_BACKEND)

if(USE_BACKEND STREQUAL "METAL")
  project(katago LANGUAGES CXX Swift)
elseif(USE_BACKEND STREQUAL "ROCM")
  if(CMAKE_VERSION VERSION_LESS 3.21)
    message(FATAL_ERROR "USE_BACKEND=ROCM requires CMake >= 3.21 for HIP language support (found ${CMAKE_VERSION}).")
  endif()
  if(WIN32)
    # Everything in the Windows ROCm build is located through HIP_PATH: the clang++ compiler
    # selection below, the MSVC toolset probing, the bundled zlib, and the runtime DLL copying.
    # Without it, configure fails later inside project()'s own HIP compiler detection with no
    # useful message, so require it here with a clear one. ROCM_PATH or -DCMAKE_PREFIX_PATH
    # alone is not a substitute.
    if(NOT DEFINED ENV{HIP_PATH})
      message(FATAL_ERROR "USE_BACKEND=ROCM on Windows requires the HIP_PATH environment variable, pointing at an installed AMD HIP SDK / extracted TheRock distribution (see Compiling.md).")
    endif()
    # Normalize HIP_PATH to forward slashes.
    # cmake's HIP detection calls `hipconfig --rocmpath` which returns
    # backslash paths on Windows, and writes the result into the generated
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
    # Unlike on Linux (where CMAKE_CXX_COMPILER=hipcc transparently treats every .cpp as HIP
    # source), hipcc.exe cannot be used as CMAKE_CXX_COMPILER on Windows: it mis-tokenizes quoted
    # paths containing spaces (e.g. "C:/Program Files (x86)/Windows Kits/..."), breaking the
    # compiler-detection step entirely. Use clang++ directly instead. The rocmbackend.cpp / CK
    # generated sources that need real HIP compilation get it via an explicit LANGUAGE HIP source
    # property (see below and the CK FMHA section further down).
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
    # Guarded to run once per fresh CMakeCache: these set(... CACHE ... FORCE) calls append to the
    # already-cached flag values, so re-running them on every cmake invocation would duplicate the
    # flags without bound (and the changed command lines would force full rebuilds each time).
    if(KATAGO_WINSDK_ROOT AND NOT KATAGO_WINSDK_FLAGS_ADDED)
      file(GLOB _pre_sdk_ver_dirs "${KATAGO_WINSDK_ROOT}/Include/*/ucrt")
      if(_pre_sdk_ver_dirs)
        list(SORT _pre_sdk_ver_dirs COMPARE NATURAL ORDER DESCENDING)
        list(GET _pre_sdk_ver_dirs 0 _pre_ucrt_dir)
        get_filename_component(_pre_sdk_ver_dir "${_pre_ucrt_dir}" DIRECTORY)
        set(_winsdk_cflags
          "-I\"${_pre_sdk_ver_dir}/ucrt\" -I\"${_pre_sdk_ver_dir}/shared\" -I\"${_pre_sdk_ver_dir}/um\"")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${_winsdk_cflags}" CACHE STRING "" FORCE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_winsdk_cflags}" CACHE STRING "" FORCE)
        set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${_winsdk_cflags}" CACHE STRING "" FORCE)
        set(KATAGO_WINSDK_FLAGS_ADDED TRUE CACHE INTERNAL "")
        message(STATUS "Pre-project: injected Windows SDK includes from ${_pre_sdk_ver_dir}")
      endif()
    endif()
    # ---------- RC compiler (Windows resource compiler, highest SDK version) ----------
    if(NOT CMAKE_RC_COMPILER)
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
          list(SORT _rc_candidates COMPARE NATURAL ORDER DESCENDING)
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

    # Pin CMake's own HIP-language ROCm root to the same install we resolved above. Without this,
    # CMakeDetermineHIPCompiler derives the root from whatever `hipconfig` is first in PATH - which
    # on a machine with multiple ROCm installs (or a partial runtime-only overlay install) can point
    # somewhere that lacks the hip-lang CMake package, failing project() below with a confusing
    # error even when CMAKE_PREFIX_PATH was set correctly.
    if(NOT DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT AND CMAKE_PREFIX_PATH)
      foreach(_katago_hip_rocm_root ${CMAKE_PREFIX_PATH})
        if(EXISTS "${_katago_hip_rocm_root}/lib/cmake/hip-lang/hip-lang-config.cmake"
           OR EXISTS "${_katago_hip_rocm_root}/lib64/cmake/hip-lang/hip-lang-config.cmake")
          set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_katago_hip_rocm_root}" CACHE PATH "" FORCE)
          message(STATUS "Pre-project: set CMAKE_HIP_COMPILER_ROCM_ROOT=${_katago_hip_rocm_root}")
          break()
        endif()
      endforeach()
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
    # Respect an explicitly user-specified compiler, mirroring the Windows branch above: only
    # auto-select when the user didn't pass -DCMAKE_CXX_COMPILER themselves.
    if(CMAKE_CXX_COMPILER)
      # User's choice stands.
    elseif(KATAGO_HIPCC_EXECUTABLE)
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
set_property(CACHE USE_BACKEND PROPERTY STRINGS "" CUDA TENSORRT OPENCL EIGEN METAL ONNX ROCM)

set(USE_TCMALLOC 0 CACHE BOOL "Use TCMalloc")
set(NO_GIT_REVISION 0 CACHE BOOL "Disable embedding the git revision into the compiled exe")
set(USE_AVX2 0 CACHE BOOL "Compile with AVX2")
set(USE_BIGGER_BOARDS_EXPENSIVE 0 CACHE BOOL "Allow boards up to size 50. Compiling with this will use more memory and slow down KataGo, even when playing on boards of size 19.")

set(USE_CACHE_TENSORRT_PLAN 0 CACHE BOOL "Use TENSORRT plan cache. May use a lot of disk space. Only applies when USE_BACKEND is TENSORRT.")
mark_as_advanced(USE_CACHE_TENSORRT_PLAN)

# CUDA-backend opt-outs of optional compiled-in paths, replicating what a build gets when the
# dependency is unavailable (no vendored CUTLASS / MSVC, cuDNN older than 8.9.3). Mainly for
# testing those fallback configurations on a machine whose toolchain would otherwise enable them.
set(NO_CUTLASS_FUSED_FFN 0 CACHE BOOL "CUDA backend: build without the CUTLASS fused FFN kernel even if CUTLASS is available.")
set(NO_CUDNN_SDPA 0 CACHE BOOL "CUDA backend: build without the cudnn frontend graph SDPA attention path.")
mark_as_advanced(NO_CUTLASS_FUSED_FFN)
mark_as_advanced(NO_CUDNN_SDPA)

# ---------- Auto-fetch missing third-party deps (e.g. zlib on a Windows ROCm/TheRock install that
# doesn't ship a linkable zlib) via a local vcpkg clone in the build tree.
# Defaults ON only on Windows ROCm builds, where the ROCm/TheRock toolchain genuinely may have no
# other way to get a linkable zlib. Everywhere else it defaults OFF: pre-existing build setups
# (system packages on Linux, vcpkg/manual paths on Windows CUDA/OpenCL per Compiling.md) are the
# expected path, and cloning vcpkg from the network as a side effect of a plain configure would be
# a surprising new behavior there. Users can still opt in with -DKATAGO_AUTO_FETCH_DEPS=ON.
if(WIN32 AND USE_BACKEND STREQUAL "ROCM")
  set(_katago_auto_fetch_default ON)
else()
  set(_katago_auto_fetch_default OFF)
endif()
option(KATAGO_AUTO_FETCH_DEPS "Automatically fetch missing dependencies into <build dir>/deps (Windows/Linux use vcpkg)." ${_katago_auto_fetch_default})
# Inside the build directory (NOT the source tree), so configure never writes into a checkout and
# separate build directories can't race on a shared vcpkg clone. Override to a common location if
# you want fetched deps shared across build dirs.
set(KATAGO_DEPS_DIR "${CMAKE_BINARY_DIR}/deps" CACHE PATH "Directory for auto-fetched third-party dependencies")
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

  # cudabackend.cpp, cudautils.cpp, and cudahelpers.cu are thin wrappers around
  # cudaandrocmbackend.inc, cudaandrocmutils.inc, and cudaandrocmhelpers.inc, which are shared
  # with the ROCm backend (tracked via compiler depfiles, so they do not appear here).
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/cudabackend.cpp
    neuralnet/cudautils.cpp
    neuralnet/cudahelpers.cu
  )
  # Optional vendored CUTLASS enables the fused transformer FFN kernel. The fused kernel is
  # built on the DualGemm extension shipped in CUTLASS's example 45, so both the include tree
  # and that example directory must be present. CUTLASS 4.x requires CUDA Toolkit 11.4+ and
  # does not build under MSVC, so those builds skip the kernel rather than break.
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/cutlass/include/cutlass/cutlass.h
     AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/cutlass/examples/45_dual_gemm/device/dual_gemm.h
     AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.4
     AND NOT MSVC
     AND NOT NO_CUTLASS_FUSED_FFN)
    message(STATUS "Found vendored CUTLASS at external/cutlass, enabling the fused FFN kernel.")
    set(KATAGO_USE_CUTLASS_FUSED_FFN 1)
    set(NEURALNET_BACKEND_SOURCES ${NEURALNET_BACKEND_SOURCES} neuralnet/cudafusedffn.cu)
    # CUTLASS requires C++17. CUDA_STANDARD is a target property that is silently ignored when
    # set per-source, and nvcc before CUDA 12 defaults to C++14, so pass -std explicitly (no
    # conflict: CMAKE_CUDA_STANDARD is unset, so CMake adds no -std flag of its own).
    set_source_files_properties(neuralnet/cudafusedffn.cu PROPERTIES
      COMPILE_OPTIONS "-std=c++17;--expt-relaxed-constexpr")
  else()
    message(STATUS "Vendored CUTLASS not found, not usable with this toolchain, or disabled via NO_CUTLASS_FUSED_FFN; building without the fused FFN kernel.")
  endif()
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
elseif(USE_BACKEND STREQUAL "ONNX")
  message(STATUS "-DUSE_BACKEND=ONNX, using ONNX Runtime backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/onnxbackend.cpp
    )
# --------------------------- ROCM backend (AMD GPU / HIP + MIOpen) ---------------------------
elseif(USE_BACKEND STREQUAL "ROCM")
  message(STATUS "-DUSE_BACKEND=ROCM, using AMD ROCm backend.")

  if(CMAKE_PREFIX_PATH STREQUAL "" OR NOT DEFINED CMAKE_PREFIX_PATH)
    # On Windows the pre-project block already required HIP_PATH and appended it to
    # CMAKE_PREFIX_PATH, so only Linux needs handling here.
    if(NOT WIN32)
      # Linux: resolve the real ROCm install prefix, using the same core-<version> vs flat
      # /opt/rocm resolution as the pre-project block above. This block re-runs it standalone in
      # case USE_BACKEND wasn't ROCM early enough for that block to have fired.
      katago_find_rocm_prefix(KATAGO_ROCM_PREFIX)
      if(KATAGO_ROCM_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH "${KATAGO_ROCM_PREFIX}")
        message(STATUS "CMAKE_PREFIX_PATH not given; defaulting to ${KATAGO_ROCM_PREFIX}")
      endif()
    endif()
  endif()

  # Ensure CMAKE_HIP_COMPILER is set BEFORE we run the optional-arch probe and before
  # enable_language(HIP). The pre-project block normally sets it on both Windows and Linux, so
  # this is a safety net for a Linux configure where it did not find clang++ then. Must be the
  # clang++ binary directly, not the hipcc wrapper (CMake rejects hipcc for CMAKE_HIP_COMPILER).
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
  # exactly that list. Otherwise we default to the broad set KataGo supports:
  # katago_default_hip_archs()'s required list plus whichever of its optional
  # (newer/less common) archs this compiler accepts.
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    katago_default_hip_archs(_default_archs)
    set(CMAKE_HIP_ARCHITECTURES "${_default_archs}" CACHE STRING "Default broad set of AMD GPU targets")
    message(STATUS "Default CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
  else()
    message(STATUS "User-specified CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
  endif()

  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)

  # rocmbackend.cpp, rocmutils.cpp, and rocmhelpers.hip are thin wrappers around
  # cudaandrocmbackend.inc, cudaandrocmutils.inc, and cudaandrocmhelpers.inc, which are shared
  # with the CUDA backend (tracked via compiler depfiles, so they do not appear here).
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/rocmbackend.cpp
    neuralnet/rocmutils.cpp
    neuralnet/rocmhelpers.hip
  )

elseif(USE_BACKEND STREQUAL "")
  message(WARNING "${ColorBoldRed}WARNING: Using dummy neural net backend, intended for non-neural-net testing only, will fail on any code path requiring a neural net. To use neural net, specify -DUSE_BACKEND=CUDA or -DUSE_BACKEND=TENSORRT or -DUSE_BACKEND=OPENCL or -DUSE_BACKEND=EIGEN or -DUSE_BACKEND=ONNX or -DUSE_BACKEND=ROCM to compile with the respective backend.${ColorReset}")
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
  tests/testbackendreference.cpp
  tests/backendreferencedata.cpp
  tests/testpassalivesuicide.cpp
  tests/testexcludeterritoryatari.cpp
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
  tests/testonnxmodelfile.cpp
  tests/tinymodel.cpp
  tests/tinymodeldata.cpp
  distributed/client.cpp
  command/commandline.cpp
  command/analysis.cpp
  command/benchmark.cpp
  command/benchmarknn.cpp
  command/contribute.cpp
  command/dumponnx.cpp
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
  if(KATAGO_USE_CUTLASS_FUSED_FFN)
    target_compile_definitions(katago PRIVATE USE_CUTLASS_FUSED_FFN)
    target_include_directories(katago SYSTEM PRIVATE
      external/cutlass/include
      external/cutlass/examples/45_dual_gemm)
  endif()
  if(NO_CUDNN_SDPA)
    message(STATUS "-DNO_CUDNN_SDPA=1 is set, building without the cudnn graph SDPA attention path.")
    target_compile_definitions(katago PRIVATE NO_CUDNN_SDPA)
  endif()
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

  # The ONNX emitter this backend feeds nvonnxparser with, and the protobuf it needs, are set up
  # in the shared block after this backend chain.

  # nvonnxparser ships alongside nvinfer in the TensorRT lib dir. Derive that dir from the
  # already-resolved TENSORRT_LIBRARY so this works both with TENSORRT_ROOT_DIR and with a
  # directly-specified -DTENSORRT_LIBRARY
  get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
  find_library(TENSORRT_ONNXPARSER_LIBRARY NAMES nvonnxparser nvonnxparser_${TENSORRT_VERSION_MAJOR} HINTS ${TENSORRT_ROOT_DIR} ${TENSORRT_LIBRARY_DIR} PATH_SUFFIXES lib)
  if(NOT TENSORRT_ONNXPARSER_LIBRARY)
    message(FATAL_ERROR "${ColorBoldRed} libnvonnxparser was NOT found in the TensorRT lib dir. ${ColorReset}")
  endif()
  target_link_libraries(katago ${TENSORRT_ONNXPARSER_LIBRARY})
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
  target_compile_definitions(katago PRIVATE USE_ROCM_BACKEND)

  string(TOLOWER "${CMAKE_HIP_ARCHITECTURES}" _gfxlist)  # e.g. "gfx90a;gfx942"
  # Every architecture in katago_default_hip_archs()'s full candidate list (Vega20/CDNA gfx906/908/90a/942/
  # 950, RDNA1 gfx101x, RDNA2 gfx103x, RDNA3 gfx110x, RDNA3.5 gfx115x, RDNA4 gfx120x) supports packed
  # FP16 ops. The regex below matches each of them so user-specified subset lists still work.
  # It also matches several Vega-class and RDNA1 architectures outside that default list, namely
  # gfx900, gfx902, gfx909, gfx90c and gfx1013, which are packed-FP16 capable too, so that a user
  # who names one of them explicitly still gets FP16 support.
  if(_gfxlist MATCHES "(gfx)?(90[0269ac]|908|94[0-2]|950|101[0-3]|103[0-6]|110[0-3]|115[0-3]|120[01])")
    target_compile_definitions(katago PRIVATE HIP_SUPPORTS_FP16)
    message(STATUS "Detected FP16-capable GFX arch (${CMAKE_HIP_ARCHITECTURES}); defining HIP_SUPPORTS_FP16")
  endif()

  # Since ROCm 6.x, CMake config-mode packages are included. If not found, add -DCMAKE_PREFIX_PATH=/opt/rocm (Linux) or HIP SDK path (Windows)
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
  # that stale /usr/include, whereas a raw -isystem, bypassing the dedup, wins.
  find_path(KATAGO_HIP_SYSTEM_INCLUDE_DIR hip/hip_runtime.h
    HINTS ${CMAKE_PREFIX_PATH} ENV HIP_PATH ENV ROCM_PATH
    PATH_SUFFIXES include)
  # Never -isystem /usr/include itself: that is exactly the stale-header location this override
  # exists to beat, and /usr/include as -isystem also breaks #include_next chains.
  if(KATAGO_HIP_SYSTEM_INCLUDE_DIR STREQUAL "/usr/include")
    set(KATAGO_HIP_SYSTEM_INCLUDE_DIR "")
  endif()
  if(KATAGO_HIP_SYSTEM_INCLUDE_DIR)
    target_compile_options(katago PRIVATE "-isystem${KATAGO_HIP_SYSTEM_INCLUDE_DIR}")
  endif()

  # ---------- fallback: HIP Runtime ----------
  if(NOT hip_FOUND)
    if(WIN32)
      # Windows: Search in HIP SDK installation
      find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
                HINTS ${CMAKE_PREFIX_PATH} ENV HIP_PATH ENV ROCM_PATH
                PATH_SUFFIXES include)
      find_library(HIP_RUNTIME_LIB
                   NAMES amdhip64 amdhip64_7 amdhip64_6
                   HINTS ${CMAKE_PREFIX_PATH} ENV HIP_PATH ENV ROCM_PATH
                   PATH_SUFFIXES lib bin)
    else()
      # Linux: search the resolved prefix, then the ROCm env vars, then /opt/rocm. The env vars
      # are only a last-resort hint here. A ROCm fully outside /opt/rocm generally still needs
      # CMAKE_PREFIX_PATH, since the compiler and package resolution earlier in the configure
      # does not consult these env vars.
      find_path(HIP_INCLUDE_DIR hip/hip_runtime.h
                HINTS ${CMAKE_PREFIX_PATH} ENV ROCM_PATH ENV HIP_PATH /opt/rocm
                PATH_SUFFIXES include)
      find_library(HIP_RUNTIME_LIB amdhip64
                   HINTS ${CMAKE_PREFIX_PATH} ENV ROCM_PATH ENV HIP_PATH /opt/rocm
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
    # The real hip-config.cmake also propagates __HIP_PLATFORM_AMD__; without it any TU compiled
    # by a non-hipcc host compiler that includes the HIP headers selects no platform and fails.
    set_target_properties(hip::device PROPERTIES
      IMPORTED_LOCATION "${HIP_RUNTIME_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
      INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1")
    target_include_directories(katago SYSTEM PRIVATE ${HIP_INCLUDE_DIR})
  endif()

  # ---------- fallback: hipBLAS / MIOpen ----------
  foreach(_pkg hipblas miopen)
    if(NOT ${_pkg}_FOUND)
      if(WIN32)
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
        # NAMES covers both spellings: ROCm ships libMIOpen.so (capitalized), which a plain
        # lowercase search misses on a case-sensitive filesystem.
        if(_pkg STREQUAL "miopen")
          set(_lib_names MIOpen miopen)
        else()
          set(_lib_names ${_pkg})
        endif()
        find_library(${_pkg}_LIB
                     NAMES ${_lib_names}
                     HINTS ${CMAKE_PREFIX_PATH} ENV ROCM_PATH ENV HIP_PATH /opt/rocm
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
          message(FATAL_ERROR "Required ROCm component ${_pkg} not found - install HIP SDK for Windows or set CMAKE_PREFIX_PATH.")
        else()
          message(FATAL_ERROR "Required ROCm component ${_pkg} not found - install it or set CMAKE_PREFIX_PATH.")
        endif()
      endif()
    endif()
  endforeach()

  # MIOpen's CMake package exports either 'MIOpen' or 'roc::miopen' depending on version, so
  # accept whichever exists.
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
  # symbol" link errors.
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
  #
  # The generic /opt/rocm/lib symlink path is appended as a fallback, after the pinned path, for
  # binaries run on machines that have a ROCm runtime but not the exact build-machine version
  # (ROCm's packages don't register their lib dir with ldconfig, so without this the loader finds
  # nothing). Sonames carry ROCm's major version, so a wrong-major install found there is skipped
  # rather than mis-linked.
  if(KATAGO_ROCM_PREFIX)
    set(_katago_rocm_rpath "${KATAGO_ROCM_PREFIX}/lib")
    if(NOT "${KATAGO_ROCM_PREFIX}" STREQUAL "/opt/rocm")
      list(APPEND _katago_rocm_rpath "/opt/rocm/lib")
    endif()
    set_target_properties(katago PROPERTIES
      BUILD_RPATH "${_katago_rocm_rpath}"
      INSTALL_RPATH "${_katago_rocm_rpath}"
    )
    message(STATUS "Set katago RPATH to ${_katago_rocm_rpath}")
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
    HINTS ${CMAKE_PREFIX_PATH} ENV HIP_PATH ENV ROCM_PATH
    PATH_SUFFIXES include
  )
  # find_path leaves a user-provided cache value untouched, so validate that whatever we ended up
  # with actually contains the ck_tile FMHA header rather than trusting it blindly.
  if(KATAGO_CK_TILE_INCLUDE_DIR AND NOT EXISTS "${KATAGO_CK_TILE_INCLUDE_DIR}/ck_tile/ops/fmha_fwd.hpp")
    message(WARNING "KATAGO_CK_TILE_INCLUDE_DIR=${KATAGO_CK_TILE_INCLUDE_DIR} does not contain ck_tile/ops/fmha_fwd.hpp; ignoring it")
    set(KATAGO_CK_TILE_INCLUDE_DIR "KATAGO_CK_TILE_INCLUDE_DIR-NOTFOUND")
  endif()

  # ck_tile's FMHA kernels were never ported to GCN/Vega (gfx90x) or RDNA1 (gfx101x) - no
  # MFMA/WMMA on those architectures. In the ck_tile version this glue targets (TheRock 7.13),
  # arch.hpp's get_compiler_target() has no branch for them, so its constexpr dispatch function
  # silently falls through to nothing on a device-compile pass for one of them, and merely
  # including ck_tile headers then hard-fails ("member reference base type 'void' is not a
  # structure or union"). These generated files exist solely to provide CK kernel instantiations,
  # so simply exclude them from those particular archs' compilation entirely via a dedicated
  # object library with a narrowed HIP_ARCHITECTURES - the plain (non-fused) attention kernel
  # already covers those archs regardless.
  # Matched on the base arch name so feature-suffixed spellings like "gfx906:xnack-" are also
  # excluded. This list must stay in sync with the corresponding #if in rocmbackend.cpp. It covers
  # more architectures than katago_default_hip_archs() builds by default, so that a user who names
  # an older one explicitly still gets a working build.
  set(_ck_fmha_excluded_archs gfx900 gfx902 gfx906 gfx909 gfx90c gfx1010 gfx1011 gfx1012 gfx1013)
  set(KATAGO_CK_FMHA_SUPPORTED_ARCHS "")
  foreach(_arch ${CMAKE_HIP_ARCHITECTURES})
    string(REGEX REPLACE ":.*$" "" _arch_base "${_arch}")
    if(NOT _arch_base IN_LIST _ck_fmha_excluded_archs)
      list(APPEND KATAGO_CK_FMHA_SUPPORTED_ARCHS "${_arch}")
    endif()
  endforeach()

  # The vendored kernels only compile against an API-compatible version of the installed ck_tile
  # core headers (see external/composable_kernel_fmha/README.md), and ck_tile's API moves between
  # ROCm releases. Probe-compile against the headers we found before committing: if they are
  # incompatible we skip the fused path with a clear message instead of hard-failing the whole
  # build partway through with hundreds of template errors. The probe result is cached per ck_tile
  # directory so a plain re-run of cmake doesn't redo it.
  #
  # The probe compiles a real generated kernel file rather than a translation unit that merely
  # includes the glue headers, because some incompatibilities only appear once ck_tile's kernel
  # templates are instantiated for a target arch. ck_tile 7.14's cast_to_amdgpu_buffer_rsrc_t, for
  # instance, calls std::memcpy from device code, which compiles only where the standard library's
  # memcpy is host+device. It is with libstdc++, which gets that from HIP's headers, but not with
  # MSVC's, whose <cstring> is already included by the time those headers are reached. A probe that
  # only includes headers accepts that and the build then dies in the middle of the kernels.
  set(_katago_ck_fmha_enabled FALSE)
  # Each generated kernel file targets one CK codegen family and produces code only for archs in
  # that family, so probe one file per distinct family among the target archs. Families are matched
  # most-specific-first, so gfx950 and gfx115x do not fall into the broader gfx9 and gfx11 buckets,
  # mirroring the dispatch order in generated/fmha_fwd_api.cpp. Archs matching no family (RDNA2
  # gfx103x) have no CK kernels at all, so a build targeting only those gets nothing from CK.
  set(_ck_probe_families "")
  set(_ck_probe_archs "")
  foreach(_arch ${KATAGO_CK_FMHA_SUPPORTED_ARCHS})
    string(REGEX REPLACE ":.*$" "" _arch_base "${_arch}")
    set(_ck_fam "")
    if(_arch_base MATCHES "^gfx950")
      set(_ck_fam "gfx950")
    elseif(_arch_base MATCHES "^gfx9")
      set(_ck_fam "gfx9")
    elseif(_arch_base MATCHES "^gfx115")
      set(_ck_fam "gfx115")
    elseif(_arch_base MATCHES "^gfx11")
      set(_ck_fam "gfx11")
    elseif(_arch_base MATCHES "^gfx12")
      set(_ck_fam "gfx12")
    endif()
    if(_ck_fam AND NOT _ck_fam IN_LIST _ck_probe_families)
      list(APPEND _ck_probe_families "${_ck_fam}")
      list(APPEND _ck_probe_archs "${_arch}")
    endif()
  endforeach()
  if(KATAGO_CK_TILE_INCLUDE_DIR AND NOT _ck_probe_families)
    message(STATUS "ck_tile headers found, but no targeted GPU arch has CK FMHA kernels; ROCm backend will only use its built-in (non-fused) attention kernel")
  elseif(KATAGO_CK_TILE_INCLUDE_DIR)
    # The cache signature includes the compiler, the probed archs, a content hash of ck_tile's own
    # fmha_fwd.hpp (a good proxy for the installed ck_tile API version), and a tag for the probe
    # scheme itself, so an in-place ROCm or header upgrade, a changed toolchain or arch list, or a
    # future change to what the probe compiles all re-probe instead of trusting a stale verdict.
    file(SHA1 "${KATAGO_CK_TILE_INCLUDE_DIR}/ck_tile/ops/fmha_fwd.hpp" _ck_header_hash)
    set(_ck_probe_sig "kernels-v2|${KATAGO_CK_TILE_INCLUDE_DIR}|${CMAKE_HIP_COMPILER}|${CMAKE_HIP_COMPILER_VERSION}|${CMAKE_BUILD_TYPE}|${_ck_probe_archs}|${_ck_header_hash}")
    if(NOT DEFINED KATAGO_CK_FMHA_COMPILE_OK OR NOT "${KATAGO_CK_FMHA_PROBED_SIG}" STREQUAL "${_ck_probe_sig}")
      set(_ck_probe_dir "${CMAKE_BINARY_DIR}/_katago_ck_fmha_probe")
      file(MAKE_DIRECTORY "${_ck_probe_dir}")
      # Probe with the same HIP flags the real build will use (on Windows these carry the baked
      # MSVC/WinSDK include paths without which any HIP compile fails outside the vcvars env).
      separate_arguments(_ck_probe_extra_flags NATIVE_COMMAND "${CMAKE_HIP_FLAGS}")
      # The per-configuration flags have to be included too, not just the base ones. ck_tile's
      # gfx11 and gfx12 kernels use inline asm with "n" (immediate) operand constraints that only
      # fold to constants once optimized, so those kernels genuinely do not compile at -O0. Probing
      # with the flags the build will really use keeps the verdict honest in both directions: no
      # false negative for a normal optimized build, and a correct one for a debug build that would
      # in fact fail to compile them.
      string(TOUPPER "${CMAKE_BUILD_TYPE}" _ck_probe_build_type)
      if(NOT _ck_probe_build_type)
        # Multi-config generators choose the configuration at build time rather than now, so probe
        # the optimized one.
        set(_ck_probe_build_type "RELEASE")
      endif()
      if(CMAKE_HIP_FLAGS_${_ck_probe_build_type})
        separate_arguments(_ck_probe_config_flags NATIVE_COMMAND "${CMAKE_HIP_FLAGS_${_ck_probe_build_type}}")
        list(APPEND _ck_probe_extra_flags ${_ck_probe_config_flags})
      else()
        list(APPEND _ck_probe_extra_flags "-O3")
      endif()
      if(KATAGO_HIP_SYSTEM_INCLUDE_DIR)
        list(APPEND _ck_probe_extra_flags "-isystem${KATAGO_HIP_SYSTEM_INCLUDE_DIR}")
      endif()
      message(STATUS "Probing whether the vendored CK FMHA kernels compile against ${KATAGO_CK_TILE_INCLUDE_DIR} for ${_ck_probe_archs} (may take a little while)...")
      set(_ck_probe_ok TRUE)
      set(KATAGO_CK_FMHA_FAILED_ARCH "" CACHE INTERNAL "")
      foreach(_ck_fam _arch IN ZIP_LISTS _ck_probe_families _ck_probe_archs)
        # One d32 file stands in for its whole family: the files share their traits and pipeline
        # machinery, so a single instantiation catches API drift and device-code errors without
        # paying to compile all fifty-odd of them. An incompatibility confined to the d64 shapes
        # would still reach the build.
        file(GLOB _ck_fam_srcs "${CMAKE_SOURCE_DIR}/external/composable_kernel_fmha/generated/fmha_fwd_d32_${_ck_fam}_*.cpp")
        list(SORT _ck_fam_srcs)
        if(NOT _ck_fam_srcs)
          continue()
        endif()
        list(GET _ck_fam_srcs 0 _ck_fam_src)
        # A full object compile rather than -fsyntax-only, so that code generation for the arch is
        # covered too and the probe tests exactly what the build will do.
        execute_process(
          COMMAND "${CMAKE_HIP_COMPILER}" -x hip --offload-arch=${_arch} -std=c++17 -c
                  "-I${CMAKE_SOURCE_DIR}/external/composable_kernel_fmha"
                  "-isystem${KATAGO_CK_TILE_INCLUDE_DIR}"
                  ${_ck_probe_extra_flags}
                  "${_ck_fam_src}" -o "${_ck_probe_dir}/probe_${_ck_fam}.o"
          RESULT_VARIABLE _ck_probe_rc
          ERROR_VARIABLE _ck_probe_err
          OUTPUT_QUIET)
        if(NOT _ck_probe_rc EQUAL 0)
          file(WRITE "${_ck_probe_dir}/probe_error.log"
            "Arch ${_arch} (CK kernel family ${_ck_fam})\nSource: ${_ck_fam_src}\n\n${_ck_probe_err}")
          set(KATAGO_CK_FMHA_FAILED_ARCH "${_arch}" CACHE INTERNAL "")
          set(_ck_probe_ok FALSE)
          break()
        endif()
      endforeach()
      if(_ck_probe_ok)
        set(KATAGO_CK_FMHA_COMPILE_OK TRUE CACHE INTERNAL "")
      else()
        set(KATAGO_CK_FMHA_COMPILE_OK FALSE CACHE INTERNAL "")
      endif()
      set(KATAGO_CK_FMHA_PROBED_SIG "${_ck_probe_sig}" CACHE INTERNAL "")
    endif()
    if(KATAGO_CK_FMHA_COMPILE_OK)
      set(_katago_ck_fmha_enabled TRUE)
    else()
      message(STATUS "The vendored CK FMHA kernels failed to compile against the ck_tile headers at ${KATAGO_CK_TILE_INCLUDE_DIR} for arch ${KATAGO_CK_FMHA_FAILED_ARCH} - that ck_tile version may be API-incompatible with the glue, or unable to compile its own kernels for that arch under this toolchain, or the probe hit a toolchain/environment issue (see ${CMAKE_BINARY_DIR}/_katago_ck_fmha_probe/probe_error.log). ROCm backend will only use its built-in (non-fused) attention kernel. For the fused path, install the CK version noted in external/composable_kernel_fmha/README.md or point KATAGO_CK_TILE_INCLUDE_DIR at a compatible copy of the ck_tile headers, and check that it builds for every targeted GPU arch.")
    endif()
  endif()

  if(_katago_ck_fmha_enabled)
    message(STATUS "Found compatible ck_tile headers at ${KATAGO_CK_TILE_INCLUDE_DIR}; enabling optional CK FMHA fused attention path")
    file(GLOB KATAGO_CK_FMHA_GENERATED_SOURCES external/composable_kernel_fmha/generated/*.cpp)
    set_source_files_properties(${KATAGO_CK_FMHA_GENERATED_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(katago_ck_fmha_kernels OBJECT ${KATAGO_CK_FMHA_GENERATED_SOURCES})
    set_target_properties(katago_ck_fmha_kernels PROPERTIES HIP_ARCHITECTURES "${KATAGO_CK_FMHA_SUPPORTED_ARCHS}")
    # SYSTEM/-isystem is required, not just for warning suppression: ck_tile's own headers
    # #include <hip/hip_runtime.h>, and on systems with a stale libamdhip64-dev under /usr/include,
    # a plain -I here still loses to /usr/include for angle-bracket resolution. -isystem does not.
    target_include_directories(katago_ck_fmha_kernels SYSTEM PRIVATE ${KATAGO_CK_TILE_INCLUDE_DIR} external/composable_kernel_fmha)
    # As above (see the KATAGO_HIP_SYSTEM_INCLUDE_DIR block for `katago` itself): the SYSTEM
    # keyword alone isn't enough here either - CMake's include-directory de-duplication drops it as
    # "already covered" against hip::device's own plain -I of the same path, so the stale
    # /usr/include/hip still wins for this target too unless we force a raw -isystem. This is a
    # separate target from `katago`, so it needs its own copy of the same compile option - compile
    # options aren't inherited across targets, only via target_link_libraries propagating usage
    # requirements, which compile options set via target_compile_options(PRIVATE) are not.
    if(KATAGO_HIP_SYSTEM_INCLUDE_DIR)
      target_compile_options(katago_ck_fmha_kernels PRIVATE "-isystem${KATAGO_HIP_SYSTEM_INCLUDE_DIR}")
    endif()
    target_link_libraries(katago katago_ck_fmha_kernels)
    target_include_directories(katago SYSTEM PRIVATE ${KATAGO_CK_TILE_INCLUDE_DIR} external/composable_kernel_fmha)
    target_compile_definitions(katago PRIVATE KATAGO_ROCM_HAS_CK_FMHA=1)
  else()
    if(NOT KATAGO_CK_TILE_INCLUDE_DIR)
      message(STATUS "ck_tile headers not found; ROCm backend will only use its built-in (non-fused) attention kernel")
    endif()
    target_compile_definitions(katago PRIVATE KATAGO_ROCM_HAS_CK_FMHA=0)
  endif()
  # On Windows plain clang++ (not hipcc) is the CXX compiler, so rocmbackend.cpp - which contains
  # the device-code-including shared backend - must get real HIP compilation via a source property,
  # CK or no CK. (The CK generated files get the same property inside the CK branch above.)
  # rocmbackend.cpp still compiles for every arch, since it also holds the non-CK ROCm backend
  # logic - it guards its own CK usage per-arch internally (see KATAGO_ROCM_CK_FMHA_ARCH_OK in
  # rocmbackend.cpp) rather than being arch-excluded here.
  if(WIN32)
    set_source_files_properties(neuralnet/rocmbackend.cpp PROPERTIES LANGUAGE HIP)
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
elseif(USE_BACKEND STREQUAL "ONNX")
  target_compile_definitions(katago PRIVATE USE_ONNX_BACKEND)

  # ONNX Runtime install tree (include/ lib/ bin/). The official prebuilt ORT packages do
  # NOT ship the OpenVINO execution provider, so for Intel GPU acceleration ORT must be
  # built from source with --use_openvino GPU (see Compiling.md).
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime package root (containing include/, lib/, bin/)")
  if(NOT IS_DIRECTORY "${ONNXRUNTIME_ROOT}")
    message(FATAL_ERROR "ONNXRUNTIME_ROOT does not exist: ${ONNXRUNTIME_ROOT}. Set -DONNXRUNTIME_ROOT=<ort install dir>.")
  endif()
  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime")
  if(NOT IS_DIRECTORY "${ONNXRUNTIME_INCLUDE_DIR}")
    # Official prebuilt ORT packages (e.g. onnxruntime-win-x64-*.zip / -linux-x64-*.tgz) lay
    # the headers flat under include/ (include/onnxruntime_cxx_api.h, ...), while an ORT
    # built from source installs them under include/onnxruntime/. Support both layouts.
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
  endif()
  if(NOT IS_DIRECTORY "${ONNXRUNTIME_INCLUDE_DIR}")
    message(FATAL_ERROR "ONNX Runtime include directory not found under ${ONNXRUNTIME_ROOT}. Looked for both include/ and include/onnxruntime/.")
  endif()
  target_include_directories(katago SYSTEM PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
  if(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll" "${ONNXRUNTIME_ROOT}/bin/*.dll")
  else()
    find_library(ONNXRUNTIME_LIB onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/bin" "${ONNXRUNTIME_ROOT}")
  endif()
  if(NOT ONNXRUNTIME_LIB OR ONNXRUNTIME_LIB STREQUAL "ONNXRUNTIME_LIB-NOTFOUND" OR NOT EXISTS "${ONNXRUNTIME_LIB}")
    message(FATAL_ERROR "Could not find onnxruntime library under ${ONNXRUNTIME_ROOT}. Looked for: ${ONNXRUNTIME_LIB}")
  endif()
  target_link_libraries(katago ${ONNXRUNTIME_LIB})

  # The ONNX emitter whose bytes go to Ort::Session, and the protobuf it needs, are set up in the
  # shared block after this backend chain. Protobuf and protoc must be findable by
  # find_package(Protobuf); for an ORT built from source they live under its _deps/protobuf-build.

  # Deploy the ORT runtime DLLs next to katago.exe so the build dir is self-contained.
  # NOTE: OpenVINO's own runtime DLLs are not shipped by ORT and must be copied
  # separately (see Compiling.md).
  if(WIN32 AND ONNXRUNTIME_DLLS)
    foreach(_onnxruntime_dll IN LISTS ONNXRUNTIME_DLLS)
      add_custom_command(TARGET katago POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_onnxruntime_dll}"
          $<TARGET_FILE_DIR:katago>)
    endforeach()
  endif()
endif()

# The TensorRT and ONNX backends both build their network by emitting an ONNX ModelProto with
# OnnxModelBuilder and handing the serialized bytes to nvonnxparser or to Ort::Session, so both
# compile the vendored schema external/onnx/onnx.proto with protoc and link our own protobuf.
# Serialized bytes are not ABI contact, so our protobuf never meets the one TensorRT links into
# nvonnxparser, or whichever one lives inside the ONNX Runtime library.
if(USE_BACKEND STREQUAL "TENSORRT" OR USE_BACKEND STREQUAL "ONNX")
  # Static by default for ONNX only, and deliberately so - do not unify this with the TensorRT
  # case, which would change how Windows TensorRT builds resolve protobuf. A from-source ONNX
  # Runtime bundles a static protobuf, which module-mode FindProtobuf would otherwise take for a
  # shared import lib, injecting PROTOBUF_USE_DLLS into the imported target and breaking the link.
  # Linking ours statically also keeps it out of the global symbol namespace, where it could merge
  # with the protobuf inside ONNX Runtime: both embed onnx.proto, and one descriptor pool holding
  # two registrations of the same file aborts at startup. That second point is a precaution rather
  # than a diagnosed bug, since the shared-protobuf configuration has never been tried against a
  # real ONNX Runtime, so whether it actually clashes is unknown.
  # Override with -DProtobuf_USE_STATIC_LIBS=FALSE. Only module-mode FindProtobuf honors it: a
  # config-mode protobuf, as from a vcpkg toolchain or protobuf 22 and up with its abseil
  # dependencies, ignores the variable entirely.
  if(USE_BACKEND STREQUAL "ONNX" AND NOT DEFINED Protobuf_USE_STATIC_LIBS)
    set(Protobuf_USE_STATIC_LIBS TRUE)
  endif()
  find_package(Protobuf REQUIRED)
  message(STATUS "Found Protobuf version: ${Protobuf_VERSION}")
  set(ONNX_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/onnx")
  protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS "${ONNX_PROTO_DIR}/onnx.proto")
  # protoc-generated code is not ours to lint; silence its warnings to keep build output readable.
  set_source_files_properties(${ONNX_PROTO_SRCS} PROPERTIES COMPILE_OPTIONS "-w")
  target_sources(katago PRIVATE ${ONNX_PROTO_SRCS} neuralnet/onnxmodelbuilder.cpp)
  # Generated onnx.pb.h lands in the build dir; let backend code include it.
  target_include_directories(katago SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})
  # Link the imported target rather than ${Protobuf_LIBRARIES}: when protobuf comes from a CMake
  # package config (e.g. vcpkg), the variable can resolve to the DLL itself rather than the import
  # lib, and it also omits protobuf's own dependencies such as abseil.
  target_link_libraries(katago protobuf::libprotobuf)
endif()

if(USE_BIGGER_BOARDS_EXPENSIVE)
  target_compile_definitions(katago PRIVATE COMPILE_MAX_BOARD_LEN=50)
endif()

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  target_compile_definitions(katago PRIVATE NO_GIT_REVISION)
endif()

# On Windows ROCm builds, zlib is bundled inside the HIP SDK (TheRock layout) - try that first.
# The bundled header is only taken together with the bundled library: mixing zlib.h from one
# distribution with a .lib/.dll from another (e.g. vcpkg's, below) risks a version skew that
# zlib's inflateInit_ header-vs-library check would reject at runtime, at first .gz model load.
if(WIN32 AND USE_BACKEND STREQUAL "ROCM" AND DEFINED ENV{HIP_PATH})
  if(NOT ZLIB_LIBRARY AND NOT ZLIB_INCLUDE_DIR AND EXISTS "$ENV{HIP_PATH}/lib/rocm_sysdeps/include/zlib.h")
    foreach(_zlib_name "zlibstatic.lib" "zlib.lib" "zlibstaticd.lib")
      if(EXISTS "$ENV{HIP_PATH}/lib/rocm_sysdeps/lib/${_zlib_name}")
        set(ZLIB_LIBRARY "$ENV{HIP_PATH}/lib/rocm_sysdeps/lib/${_zlib_name}" CACHE FILEPATH "" FORCE)
        set(ZLIB_INCLUDE_DIR "$ENV{HIP_PATH}/lib/rocm_sysdeps/include" CACHE PATH "" FORCE)
        break()
      endif()
    endforeach()
  endif()
endif()

# TheRock's Windows package (as of ~7.13) ships zlib.h under rocm_sysdeps/include but no linkable
# .lib anymore (the block above then takes neither, keeping header and library paired) - auto-fetch
# zlib via vcpkg instead of requiring the user to separately install a package manager (see
# KATAGO_AUTO_FETCH_DEPS above).
#
# Probe for a normal system zlib first (quiet - the "real", error-emitting find_package(ZLIB) call
# is further below): on a fresh Linux configure, ZLIB_INCLUDE_DIR/ZLIB_LIBRARY are never set by
# anything above this point regardless of whether the system already has zlib1g-dev installed (as
# Compiling.md's Linux instructions require), so without this probe the check below would always
# be true and vcpkg would fire unconditionally on every Linux build - needlessly cloning vcpkg,
# downloading its own CMake, and building zlib from source even when the system package is right
# there. When the probe finds the system zlib, the later find_package call reuses the cached
# result at no extra cost. On Windows this probe typically finds nothing and the vcpkg path
# still fires.
if(KATAGO_AUTO_FETCH_DEPS AND (NOT ZLIB_INCLUDE_DIR OR NOT ZLIB_LIBRARY))
  find_package(ZLIB QUIET)
endif()

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
    endif()
  endif()
endif()

# vcpkg's default triplets (e.g. x64-windows) build zlib as a DLL, not a static lib - the .lib is
# just an import lib, so the matching runtime DLL must be copied next to katago.exe at build time
# (the POST_BUILD step further below). Derived from the cached library path rather than set inside
# the install flow above: on a reconfigure ZLIB_LIBRARY is already cached and that flow is skipped
# entirely, so a flow-scoped flag would silently drop the DLL-copy step from the regenerated build.
set(KATAGO_ZLIB_IS_VCPKG_DLL FALSE)
if(WIN32 AND ZLIB_LIBRARY)
  file(TO_CMAKE_PATH "${ZLIB_LIBRARY}" _katago_zlib_norm)
  file(TO_CMAKE_PATH "${KATAGO_VCPKG_ROOT}" _katago_vcpkg_norm)
  string(FIND "${_katago_zlib_norm}" "${_katago_vcpkg_norm}/" _katago_zlib_in_vcpkg)
  if(_katago_zlib_in_vcpkg EQUAL 0)
    set(KATAGO_ZLIB_IS_VCPKG_DLL TRUE)
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
    # Transitive dependencies that katago does not link against directly. hipblas imports
    # rocsolver, and amdhip64 imports rocm_kpack in TheRock's multi-arch distributions, where
    # rocm_kpack loads the per-architecture device code. Without these, katago.exe fails to start
    # and prints nothing at all unless HIP_PATH/bin is on PATH.
    "${_katago_hip_path}/bin/rocsolver.dll"
    "${_katago_hip_path}/bin/rocm_kpack.dll"
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

  # MSVC toolset 14.40+ (VS 2022 17.10 and later, including VS 2026) made std::mutex's
  # constructor constexpr, so mutexes are zero-initialized at compile time instead of being
  # initialized at runtime via _Mtx_init_in_situ. If the process then loads a copy of
  # msvcp140.dll older than 14.40 (outdated VC++ redistributable, or a stale msvcp140.dll
  # sitting next to the exe or shipped by a GUI frontend, which takes precedence over the
  # system one), the old _Mtx_lock misreads the zero-initialized mutex and crashes on the
  # first lock. This define restores the old runtime initialization, which is compatible
  # with both old and new runtimes. See https://github.com/microsoft/STL/issues/4978
  target_compile_definitions(katago PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)

  if(USE_AVX2)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /arch:AVX2 -D__FMA__")
    target_compile_definitions(katago PRIVATE USE_AVX2)
  endif()

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:8388608")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  message(STATUS "Setting up build for GNU, Clang or MinGW.")
  if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "(arm|aarch32|aarch64)")
    # For ARM architecture, as a hack, ensure that char is signed
    message(STATUS "ARM architecture detected: adding -fsigned-char flag")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")
  else()
    # KataGo wants IEEE-compliant float math.
    # On x86-64 SSE-based IEEE float math is already the ABI default (x87 excess precision cannot occur),
    # but mandate it explicitly anyway.
    if(USE_BACKEND STREQUAL "ROCM")
      if(NOT WIN32)
        # hipcc/hip-clang treats every source as an offload compilation, and the AMDGPU device
        # pass hard-errors on a bare -mfpmath=sse ("unknown FP unit 'sse'"). -Xarch_host applies
        # the flag to the host-side half of the compilation only. Device float math is IEEE
        # arithmetic in its own right, so no equivalent flag exists or is needed there.
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xarch_host -mfpmath=sse")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -Xarch_host -mfpmath=sse")
      endif()
    else()
      set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse")
    endif()
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
      # libatomic is a Linux GCC/Clang runtime, not needed or available on Windows
      target_link_libraries(katago "atomic")
    else()
      # Match the 8 MB stack every MSVC-built KataGo requests in the if(MSVC) branch above.
      # Without this a Windows clang/lld-link build gets the 1 MB default.
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,/STACK:8388608")
    endif()
  endif()

  if(USE_TCMALLOC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
  endif()

endif()

target_include_directories(katago PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# On Windows ROCm builds, clang compiles host code against MSVC-compatibility headers
# that require Windows SDK ucrt/shared/um headers.
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
    file(GLOB _sdk_ver_dirs "${KATAGO_WINSDK_ROOT}/Include/*/ucrt")
    if(_sdk_ver_dirs)
      list(SORT _sdk_ver_dirs COMPARE NATURAL ORDER DESCENDING)
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

