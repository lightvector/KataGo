cmake_minimum_required(VERSION 3.18.2)

#--------------------------- MSVC ENVIRONMENT BOOTSTRAP (Windows, pre-project) -----------------------------------------
#
# Configuring with Ninja normally requires a Developer Command Prompt, because cl.exe reads
# INCLUDE/LIB/PATH that only vcvarsall.bat sets. Finding and applying that automatically lets a
# plain shell do the documented thing:
#
#   cd cpp && mkdir build && cd build
#   cmake .. -G Ninja -DUSE_BACKEND=RYZENAI -DCMAKE_BUILD_TYPE=Release && ninja
#
# Nothing here hardcodes an install path, edition or version. vswhere sits at a location Microsoft
# documents as stable, it reports where Visual Studio actually is, and the newest installed toolset
# wins. Running inside a Developer Command Prompt still works - the bootstrap sees the environment
# is already set up and does nothing.

# Locate vswhere.exe via the Program Files env vars (its documented install location), so we never
# hardcode a drive letter or edition-specific path.
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

# Apply the environment vcvarsall.bat would set into THIS cmake process, which is what every
# compiler check during this configure run needs (execute_process children inherit it).
function(katago_win_apply_vcvarsall vcvarsall_bat vcvars_ver)
  # Generate a tiny .bat wrapper rather than passing the whole "call ... && set" pipeline as one
  # execute_process COMMAND argument - CMake's argument quoting mangles a string that complex
  # (embedded quotes plus '&&'), even though it works typed straight into a shell.
  set(_bat_file "${CMAKE_BINARY_DIR}/_katago_vcvars_dump.bat")
  set(_out_file "${CMAKE_BINARY_DIR}/_katago_vcvars_env.txt")
  file(WRITE "${_bat_file}" "@echo off\r\ncall \"${vcvarsall_bat}\" x64 -vcvars_ver=${vcvars_ver} >NUL 2>NUL\r\nset\r\n")
  execute_process(COMMAND cmd /c "${_bat_file}" OUTPUT_FILE "${_out_file}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0 OR NOT EXISTS "${_out_file}")
    set(${out_var} 0 PARENT_SCOPE)
    return()
  endif()
  # Read line by line with file(STRINGS) rather than splitting the raw output ourselves: INCLUDE,
  # LIB and PATH contain semicolons, which a string(REPLACE "\n" ";") split would corrupt.
  file(STRINGS "${_out_file}" _env_lines)
  foreach(_line IN LISTS _env_lines)
    if(_line MATCHES "^([A-Za-z_][A-Za-z0-9_().]*)=(.*)$")
      set(ENV{${CMAKE_MATCH_1}} "${CMAKE_MATCH_2}")
    endif()
  endforeach()
endfunction()

# Newest MSVC toolset across every Visual Studio install, as "<version>|<vcvarsall.bat>".
function(katago_win_find_newest_msvc out_var)
  katago_win_find_vswhere(_vswhere)
  if(NOT _vswhere)
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()
  execute_process(
    COMMAND "${_vswhere}" -all -products * -property installationPath
    OUTPUT_VARIABLE _raw OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  string(REPLACE "\r\n" "\n" _raw "${_raw}")
  string(REPLACE "\n" ";" _install_paths "${_raw}")

  set(_best_ver "")
  set(_best_bat "")
  foreach(_install_path IN LISTS _install_paths)
    if(_install_path STREQUAL "")
      continue()
    endif()
    file(TO_CMAKE_PATH "${_install_path}" _install_path)
    set(_vcvarsall "${_install_path}/VC/Auxiliary/Build/vcvarsall.bat")
    if(NOT EXISTS "${_vcvarsall}")
      continue()
    endif()
    # Version comes from the toolset directory name; require the C++ headers to actually be there,
    # so a partially-installed toolset is not selected.
    file(GLOB _tool_dirs "${_install_path}/VC/Tools/MSVC/*")
    foreach(_t IN LISTS _tool_dirs)
      if(IS_DIRECTORY "${_t}" AND EXISTS "${_t}/include/cmath")
        get_filename_component(_ver "${_t}" NAME)
        # Compare as versions, not strings: list(SORT) mis-orders e.g. 14.9 against 14.40.
        if(_ver MATCHES "^[0-9]+\\.[0-9]+" AND (_best_ver STREQUAL "" OR _ver VERSION_GREATER _best_ver))
          set(_best_ver "${_ver}")
          set(_best_bat "${_vcvarsall}")
        endif()
      endif()
    endforeach()
  endforeach()
  if(_best_ver STREQUAL "")
    set(${out_var} "" PARENT_SCOPE)
  else()
    set(${out_var} "${_best_ver}|${_best_bat}" PARENT_SCOPE)
  endif()
endfunction()

# Turn a semicolon-separated path list (the INCLUDE or LIB env var) into quoted compiler flags.
function(katago_win_pathlist_to_flags pathlist flagprefix out_var)
  set(_flags "")
  # Split as a plain string via regex: CMake's own ';' list handling would fight the semicolons
  # already inside the path list.
  string(REGEX MATCHALL "[^;]+" _dirs "${pathlist}")
  foreach(_dir IN LISTS _dirs)
    if(NOT _dir STREQUAL "")
      set(_flags "${_flags} ${flagprefix}\"${_dir}\"")
    endif()
  endforeach()
  set(${out_var} "${_flags}" PARENT_SCOPE)
endfunction()

# CMAKE_HOST_WIN32, not WIN32: the latter is a result of platform detection inside project(), so
# it is still undefined this early.
#
# The guard asks whether the environment is usable, not whether a variable happens to exist: a
# stray empty or whitespace VCToolsInstallDir would otherwise make this skip the bootstrap and
# leave the build with no compiler at all.
set(_katago_vctools "$ENV{VCToolsInstallDir}")
set(_katago_include "$ENV{INCLUDE}")
string(STRIP "${_katago_vctools}" _katago_vctools)
string(STRIP "${_katago_include}" _katago_include)
if(CMAKE_HOST_WIN32 AND NOT CMAKE_CXX_COMPILER
   AND (_katago_vctools STREQUAL "" OR _katago_include STREQUAL ""))
  katago_win_find_newest_msvc(_katago_msvc)
  if(_katago_msvc)
    string(REPLACE "|" ";" _katago_msvc_list "${_katago_msvc}")
    list(GET _katago_msvc_list 0 _katago_msvc_ver)
    list(GET _katago_msvc_list 1 _katago_msvc_bat)
    katago_win_apply_vcvarsall("${_katago_msvc_bat}" "${_katago_msvc_ver}")
    message(STATUS "KataGo: using MSVC toolset ${_katago_msvc_ver} (found via vswhere)")

    # The environment applied above lasts only for this configure run. A later, separate `ninja`
    # (or `cmake --build`) is a sibling process that inherits none of it, so cl.exe would be found
    # by its baked-in full path and then fail on <algorithm>. Bake the resolved INCLUDE/LIB into
    # flags below, once project() has created the variables to append them to.
    katago_win_pathlist_to_flags("$ENV{INCLUDE}" "/I" _katago_msvc_incflags)
    katago_win_pathlist_to_flags("$ENV{LIB}" "/LIBPATH:" _katago_msvc_libflags)

    # Visual Studio bundles Ninja, but only on the IDE's private PATH, so a plain shell asking for
    # -G Ninja would otherwise need a separate install.
    if(CMAKE_GENERATOR MATCHES "Ninja" AND NOT CMAKE_MAKE_PROGRAM)
      find_program(_katago_ninja ninja)
      if(NOT _katago_ninja)
        get_filename_component(_vs_root "${_katago_msvc_bat}" DIRECTORY)
        get_filename_component(_vs_root "${_vs_root}/../../.." ABSOLUTE)
        set(_bundled "${_vs_root}/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe")
        if(EXISTS "${_bundled}")
          set(CMAKE_MAKE_PROGRAM "${_bundled}" CACHE FILEPATH "" FORCE)
          message(STATUS "KataGo: using Ninja bundled with Visual Studio: ${_bundled}")
        endif()
      endif()
    endif()
  else()
    message(WARNING "KataGo: no Visual Studio C++ toolchain found via vswhere. Install the "
                    "\"Desktop development with C++\" workload, or configure from a Developer "
                    "Command Prompt.")
  endif()
endif()
#-----------------------------------------------------------------------------------------------------------------------

if(USE_BACKEND STREQUAL "METAL")
  project(katago LANGUAGES CXX Swift)
else()
  project(katago)
endif()

# See the bootstrap block above: these carry the auto-detected toolchain's header and library
# search paths into the generated build files, so building later from a plain shell works.
if(_katago_msvc_incflags)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}${_katago_msvc_incflags}" CACHE STRING "" FORCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}${_katago_msvc_incflags}" CACHE STRING "" FORCE)
  foreach(_lk EXE SHARED MODULE)
    set(CMAKE_${_lk}_LINKER_FLAGS "${CMAKE_${_lk}_LINKER_FLAGS}${_katago_msvc_libflags}"
        CACHE STRING "" FORCE)
  endforeach()
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
set_property(CACHE USE_BACKEND PROPERTY STRINGS "" CUDA TENSORRT OPENCL EIGEN METAL RYZENAI)

set(USE_TCMALLOC 0 CACHE BOOL "Use TCMalloc")
set(NO_GIT_REVISION 0 CACHE BOOL "Disable embedding the git revision into the compiled exe")
set(USE_AVX2 0 CACHE BOOL "Compile with AVX2")
set(USE_BIGGER_BOARDS_EXPENSIVE 0 CACHE BOOL "Allow boards up to size 50. Compiling with this will use more memory and slow down KataGo, even when playing on boards of size 19.")

set(USE_CACHE_TENSORRT_PLAN 0 CACHE BOOL "Use TENSORRT plan cache. May use a lot of disk space. Only applies when USE_BACKEND is TENSORRT.")
mark_as_advanced(USE_CACHE_TENSORRT_PLAN)

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
elseif(USE_BACKEND STREQUAL "RYZENAI")
  message(STATUS "-DUSE_BACKEND=RYZENAI, using AMD Ryzen AI NPU (XDNA/XDNA2) backend.")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/ryzenaibackend.cpp
    neuralnet/ryzenaidevice.cpp
    neuralnet/ryzenaikernel.cpp
    neuralnet/ryzenaireference.cpp
    neuralnet/ryzenaisequence.cpp
    neuralnet/ryzenaishapes.cpp
    neuralnet/ryzenaimatmul.cpp
    )
elseif(USE_BACKEND STREQUAL "")
  message(WARNING "${ColorBoldRed}WARNING: Using dummy neural net backend, intended for non-neural-net testing only, will fail on any code path requiring a neural net. To use neural net, specify -DUSE_BACKEND=CUDA or -DUSE_BACKEND=TENSORRT or -DUSE_BACKEND=OPENCL or -DUSE_BACKEND=EIGEN or -DUSE_BACKEND=RYZENAI to compile with the respective backend.${ColorReset}")
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
  tests/testpassalivesuicide.cpp
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
  # Link the imported target rather than ${Protobuf_LIBRARIES}: when protobuf comes from a
  # CMake package config (e.g. vcpkg), the variable can resolve to the DLL itself rather
  # than the import lib, and it also omits protobuf's own dependencies such as abseil.
  target_link_libraries(katago ${TENSORRT_ONNXPARSER_LIBRARY} protobuf::libprotobuf)
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
elseif(USE_BACKEND STREQUAL "RYZENAI")
  target_compile_definitions(katago PRIVATE USE_RYZENAI_BACKEND)
  message(STATUS "RyzenAI backend: talks to the NPU directly through XRT.")
  message(STATUS "  It does NOT require the AMD Ryzen AI Software SDK, ONNX Runtime, or Python.")
  message(STATUS "  Build time needs the XRT SDK; run time needs only the NPU driver.")

  # ---- XRT SDK discovery ----
  # Nothing here hardcodes a path or a version. Resolution order:
  #   1. -DXRT_ROOT=... given on the command line
  #   2. the XILINX_XRT environment variable (set by XRT's own setup scripts on
  #      Linux, and by the Windows XRT SDK install step)
  #   3. CMake's normal search paths
  set(XRT_ROOT "" CACHE PATH "Root of the XRT SDK. Falls back to the XILINX_XRT environment variable.")
  # Consult both every configure. Seeding the cache entry from the environment
  # instead would stick on whatever the first configure saw, so installing the
  # SDK afterwards would not be picked up without wiping the build directory.
  set(_katago_xrt_hints "")
  if(XRT_ROOT)
    list(APPEND _katago_xrt_hints "${XRT_ROOT}")
  endif()
  if(DEFINED ENV{XILINX_XRT} AND NOT "$ENV{XILINX_XRT}" STREQUAL "")
    list(APPEND _katago_xrt_hints "$ENV{XILINX_XRT}")
  endif()
  find_path(XRT_INCLUDE_DIR
    NAMES xrt/xrt_device.h
    HINTS ${_katago_xrt_hints}
    PATH_SUFFIXES include
    DOC "Directory containing xrt/xrt_device.h"
  )
  find_library(XRT_COREUTIL_LIBRARY
    NAMES xrt_coreutil
    HINTS ${_katago_xrt_hints}
    PATH_SUFFIXES lib lib64
    DOC "The xrt_coreutil library"
  )
  if(NOT XRT_INCLUDE_DIR OR NOT XRT_COREUTIL_LIBRARY)
    message(FATAL_ERROR "${ColorBoldRed}The XRT SDK was not found. Install it and either set the XILINX_XRT environment variable to its root or pass -DXRT_ROOT=<path>. On Windows, download xrt_windows_sdk.zip from the Xilinx/XRT releases and extract its xrt_sdk/xrt directory; on Linux, install the XRT packages (typically /opt/xilinx/xrt).${ColorReset}")
  endif()
  message(STATUS "  XRT include dir: ${XRT_INCLUDE_DIR}")
  message(STATUS "  XRT library:     ${XRT_COREUTIL_LIBRARY}")
  target_include_directories(katago SYSTEM PRIVATE "${XRT_INCLUDE_DIR}")
  target_link_libraries(katago "${XRT_COREUTIL_LIBRARY}")

  # XRT's headers select std::any vs boost::any on __cplusplus. MSVC reports
  # 199711L there unless told otherwise, which sends xrt/detail/any.h looking for
  # a Boost that we do not ship. Set per-target so the compiler defaults CMake
  # supplies (notably /EHsc) are left intact.
  target_compile_options(katago PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>)

  # ---- NPU kernel binaries (xclbin) ----
  # Shipped as build artifacts next to the executable, in a "ryzenai" subdir.
  # The backend resolves them relative to the executable at run time, so a
  # deployed install is just exe + dlls + this directory.
  set(RYZENAI_ARTIFACT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/ryzenai_artifacts")
  add_custom_command(TARGET katago POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:katago>/ryzenai"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${RYZENAI_ARTIFACT_SRC_DIR}" "$<TARGET_FILE_DIR:katago>/ryzenai"
    COMMENT "Copying RyzenAI NPU kernel artifacts next to katago"
  )

  if(WIN32)
    # The XRT runtime DLLs ship with the NPU driver rather than with the SDK, so
    # they are picked up from the Windows driver store. Located by globbing on
    # top of %SystemRoot%, so neither a path nor a driver version is hardcoded.
    # Override with -DXRT_RUNTIME_DLL_DIR=<dir> if the driver lives elsewhere.
    set(XRT_RUNTIME_DLL_DIR "" CACHE PATH "Directory containing xrt_coreutil.dll. Auto-detected from the NPU driver store if empty.")
    if(NOT XRT_RUNTIME_DLL_DIR)
      file(GLOB _katago_xrt_driver_dirs
        "$ENV{SystemRoot}/System32/DriverStore/FileRepository/*/xrt_coreutil.dll")
      if(_katago_xrt_driver_dirs)
        # Several driver revisions can coexist in the store; take the newest.
        list(SORT _katago_xrt_driver_dirs COMPARE FILE_BASENAME ORDER DESCENDING)
        list(GET _katago_xrt_driver_dirs 0 _katago_xrt_dll)
        get_filename_component(XRT_RUNTIME_DLL_DIR "${_katago_xrt_dll}" DIRECTORY)
      endif()
    endif()
    if(XRT_RUNTIME_DLL_DIR)
      message(STATUS "  XRT runtime DLLs: ${XRT_RUNTIME_DLL_DIR}")
      file(GLOB _katago_xrt_runtime_dlls "${XRT_RUNTIME_DLL_DIR}/xrt_*.dll")
      foreach(_katago_xrt_dll IN LISTS _katago_xrt_runtime_dlls)
        add_custom_command(TARGET katago POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_katago_xrt_dll}" "$<TARGET_FILE_DIR:katago>"
        )
      endforeach()
    else()
      message(WARNING "${ColorBoldRed}Could not locate xrt_coreutil.dll in the driver store. katago will need it on PATH at run time, or pass -DXRT_RUNTIME_DLL_DIR=<dir>.${ColorReset}")
    endif()
  endif()
endif()

if(USE_BIGGER_BOARDS_EXPENSIVE)
  target_compile_definitions(katago PRIVATE COMPILE_MAX_BOARD_LEN=50)
endif()

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  target_compile_definitions(katago PRIVATE NO_GIT_REVISION)
endif()

# Windows has no system zlib, so a plain `cmake .. -G Ninja` would fail here even though a usable
# copy is almost always already on the machine - conda ships one, so does vcpkg. Look in the roots
# those tools advertise through the environment before giving up. Explicit -DZLIB_ROOT= still wins,
# and nothing here names a fixed drive, user or version.
if(CMAKE_HOST_WIN32 AND NOT ZLIB_ROOT AND NOT ZLIB_LIBRARY)
  set(_katago_zlib_roots "")
  if(DEFINED ENV{CONDA_PREFIX})
    file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}" _p)
    list(APPEND _katago_zlib_roots "${_p}/Library")
  endif()
  if(DEFINED ENV{VCPKG_ROOT})
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _p)
    list(APPEND _katago_zlib_roots "${_p}/installed/x64-windows")
  endif()
  if(DEFINED ENV{USERPROFILE})
    file(TO_CMAKE_PATH "$ENV{USERPROFILE}" _home)
    # A base conda install is not "activated", so it exports no CONDA_PREFIX; its default location
    # under the user profile is the only thing to go on.
    foreach(_dist miniconda3 anaconda3 miniforge3 mambaforge)
      list(APPEND _katago_zlib_roots "${_home}/${_dist}/Library")
    endforeach()
  endif()
  foreach(_root IN LISTS _katago_zlib_roots)
    if(EXISTS "${_root}/include/zlib.h")
      # Prefer the import library for the DLL over the static one: KataGo links the rest of its
      # dependencies dynamically here, and mixing runtimes is what produces the confusing
      # duplicate-symbol failures at link time.
      foreach(_cand zdll.lib zlib.lib z.lib)
        if(EXISTS "${_root}/lib/${_cand}")
          set(ZLIB_ROOT "${_root}" CACHE PATH "Root of a zlib install" FORCE)
          set(ZLIB_INCLUDE_DIR "${_root}/include" CACHE PATH "" FORCE)
          set(ZLIB_LIBRARY "${_root}/lib/${_cand}" CACHE FILEPATH "" FORCE)
          message(STATUS "KataGo: using zlib from ${_root}")
          break()
        endif()
      endforeach()
    endif()
    if(ZLIB_ROOT)
      break()
    endif()
  endforeach()
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
  if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(arm|aarch32|aarch64)"))
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
    target_link_libraries(katago "atomic")
  endif()

  if(USE_TCMALLOC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
  endif()

endif()

target_include_directories(katago PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

