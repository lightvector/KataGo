cmake_minimum_required(VERSION 3.18.2)
if(USE_BACKEND STREQUAL "METAL")
  project(katago LANGUAGES CXX Swift)
elseif(USE_BACKEND STREQUAL "ROCM")
  if(WIN32)
    # Windows: Use clang++ from HIP SDK (hipcc doesn't work well on Windows)
    # User can override with -DCMAKE_CXX_COMPILER if needed
    if(NOT DEFINED CMAKE_CXX_COMPILER)
      if(DEFINED ENV{HIP_PATH})
        set(CMAKE_CXX_COMPILER "$ENV{HIP_PATH}/bin/clang++.exe" CACHE FILEPATH "" FORCE)
        set(CMAKE_C_COMPILER "$ENV{HIP_PATH}/bin/clang.exe" CACHE FILEPATH "" FORCE)
      endif()
    endif()
  else()
    # Linux: Use hipcc
    set(CMAKE_C_COMPILER  /opt/rocm/bin/hipcc CACHE FILEPATH "" FORCE)
    set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc CACHE FILEPATH "" FORCE)
  endif()
  project(katago LANGUAGES C CXX HIP)
else()
  project(katago)
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(external)
include_directories(SYSTEM external/tclap-1.2.5/include) #SYSTEM suppresses warnings from external library
include_directories(SYSTEM external/filesystem-1.5.8/include) #SYSTEM suppresses a few warnings

#--------------------------- PLATFORM SPECIFIC -------------------------------------------------------------------------

if(APPLE)
  # Fix linking on 10.14+. See https://stackoverflow.com/questions/54068035
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)
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
set_property(CACHE USE_BACKEND PROPERTY STRINGS "" CUDA TENSORRT OPENCL EIGEN ROCM ONNX)

set(USE_TCMALLOC 0 CACHE BOOL "Use TCMalloc")
set(NO_GIT_REVISION 0 CACHE BOOL "Disable embedding the git revision into the compiled exe")
set(USE_AVX2 0 CACHE BOOL "Compile with AVX2")
set(USE_BIGGER_BOARDS_EXPENSIVE 0 CACHE BOOL "Allow boards up to size 50. Compiling with this will use more memory and slow down KataGo, even when playing on boards of size 19.")

set(USE_CACHE_TENSORRT_PLAN 0 CACHE BOOL "Use TENSORRT plan cache. May use a lot of disk space. Only applies when USE_BACKEND is TENSORRT.")
mark_as_advanced(USE_CACHE_TENSORRT_PLAN)

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
  message(STATUS "-DUSE_BACKEND=METAL, using Metal backend.")
  if(NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
    message(FATAL_ERROR "Bidirectional C++ Interop requires Ninja generator. Have ${CMAKE_GENERATOR}")
  endif()
  if("${CMAKE_Swift_COMPILER_VERSION}" VERSION_LESS 5.9)
    message(FATAL_ERROR "Bidirectional C++ Interop requires Swift 5.9 or greater. Have ${CMAKE_Swift_COMPILER_VERSION}")
  endif()
  if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
    message(FATAL_ERROR "Project requires building with AppleClang. Have ${CMAKE_CXX_COMPILER_ID}")
  endif()
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
    neuralnet/metalbackend.swift)
  _swift_generate_cxx_header(
    KataGoSwift
    "${CMAKE_CURRENT_BINARY_DIR}/include/KataGoSwift/KataGoSwift-swift.h"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/neuralnet/metalbackend.swift")
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
    neuralnet/onnxmodelbuilder.cpp
    )
# --------------------------- ROCM backend（AMD GPU / HIP  MIOpen） ---------------------------
elseif(USE_BACKEND STREQUAL "ROCM")
  message(STATUS "-DUSE_BACKEND=ROCM, using AMD ROCm backend.")

  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)

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
        message(WARNING "HIP_PATH or ROCM_PATH environment variable not set. Please install HIP SDK for Windows.")
      endif()
    else()
      # Linux: Standard ROCm installation path
      if(EXISTS "/opt/rocm")
        list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
        message(STATUS "CMAKE_PREFIX_PATH not given; defaulting to /opt/rocm")
      endif()
    endif()
  endif()

  # Users can -DCMAKE_HIP_ARCHITECTURES=gfx90a;gfx942 manually specify GFX architectures
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    # Default compile MI200 / RDNA3 cards, can be simplified as needed
    # set(CMAKE_HIP_ARCHITECTURES gfx950 gfx942 gfx90a gfx908 gfx1100 gfx1101 gfx1151 gfx1201 gfx1030 CACHE STRING "AMD GPU targets")
    add_compile_definitions(-DGPU_TARGETS=gfx950,gfx942,gfx90a,gfx908,gfx1100,gfx1101,gfx1151,gfx1201,gfx1030)
  endif()

  # 2) Specify backend source code. rocmhelpers.hip contains GPU kernels, don't forget it
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/rocmbackend.cpp
    neuralnet/rocmutils.cpp
    neuralnet/rocmhelpers.hip
  )

  # Optional: Enable model-size‑based autotuning and other macros
  # add_compile_definitions(HIP_SUPPORTS_FP16)

elseif(USE_BACKEND STREQUAL "")
  message(WARNING "${ColorBoldRed}WARNING: Using dummy neural net backend, intended for non-neural-net testing only, will fail on any code path requiring a neural net. To use neural net, specify -DUSE_BACKEND=CUDA or -DUSE_BACKEND=TENSORRT or -DUSE_BACKEND=OPENCL or -DUSE_BACKEND=EIGEN or -DUSE_BACKEND=ONNX to compile with the respective backend.${ColorReset}")
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
  add_custom_command(
    OUTPUT ${GIT_HEADER_FILE_ALWAYS_UPDATED}
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
  command/demoplay.cpp
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
  target_link_libraries(katago CUDA::cublas ${CUDNN_LIBRARY})
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
elseif(USE_BACKEND STREQUAL "METAL")
  target_compile_definitions(katago PRIVATE USE_METAL_BACKEND)
  target_link_libraries(katago KataGoSwift)
  if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    message(WARNING "You are currently running cmake on an Intel-based processor. It is known that running KataGo in this configuration may encounter performance issues. It is recommended to switch to a cmake version designed for ARM64 architecture for optimal performance.")
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

  # Option to disable FP16 support
  set(ROCM_DISABLE_FP16 OFF CACHE BOOL "Disable FP16 support for ROCm backend")
  
  string(TOLOWER "${CMAKE_HIP_ARCHITECTURES}" _gfxlist)  # e.g. "90a;942"
  if(_gfxlist MATCHES "803|900|90a|94[0-9]|110[0-9]|120[0-9]|115[0-9]|1030")
    if(ROCM_DISABLE_FP16)
      message(STATUS "Detected FP16‑capable GFX arch (${CMAKE_HIP_ARCHITECTURES}), but ROCM_DISABLE_FP16 is set, NOT defining HIP_SUPPORTS_FP16")
    else()
      target_compile_definitions(katago PRIVATE HIP_SUPPORTS_FP16)
      message(STATUS "Detected FP16‑capable GFX arch (${CMAKE_HIP_ARCHITECTURES}); defining HIP_SUPPORTS_FP16")
    endif()
  endif()

  # 3) Find ROCm runtime & libraries. Since ROCm 6.x, CMake config-mode packages are included. If not found, add -DCMAKE_PREFIX_PATH=/opt/rocm (Linux) or HIP SDK path (Windows)
  find_package(hip        QUIET CONFIG)   # Export hip::device / hip::host
  find_package(hipblas    QUIET CONFIG)   # Export roc::hipblas
  find_package(miopen     QUIET CONFIG)   # Export roc::miopen or MIOpen
  
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
    set(_miopen_target roc::miopen)
  endif()
  
  target_link_libraries(katago
    hip::device          # HIP runtime & kernel offload
    roc::hipblas         # BLAS
    ${_miopen_target}    # DNN primitives
  )
elseif(USE_BACKEND STREQUAL "ONNX")
  target_compile_definitions(katago PRIVATE USE_ONNX_BACKEND)

  if(WIN32)
    set(_onnx_default_root "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime-win-x64-openvino")
  elseif(UNIX AND NOT APPLE)
    set(_onnx_default_root "${CMAKE_CURRENT_SOURCE_DIR}/external/onnxruntime-linux-x64-openvino")
  else()
    set(_onnx_default_root "")
  endif()
  set(ONNXRUNTIME_ROOT "${_onnx_default_root}" CACHE PATH "Path to ONNX Runtime package root")

  if(NOT IS_DIRECTORY "${ONNXRUNTIME_ROOT}")
    message(FATAL_ERROR "ONNXRUNTIME_ROOT does not exist: ${ONNXRUNTIME_ROOT}")
  endif()

  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
  if(NOT IS_DIRECTORY "${ONNXRUNTIME_INCLUDE_DIR}")
    message(FATAL_ERROR "ONNX Runtime include directory not found: ${ONNXRUNTIME_INCLUDE_DIR}")
  endif()
  target_include_directories(katago SYSTEM PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")

  if(WIN32)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    file(GLOB ONNXRUNTIME_DLLS "${ONNXRUNTIME_ROOT}/lib/*.dll")
  else()
    find_library(ONNXRUNTIME_LIB onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}")
  endif()
  if(NOT EXISTS "${ONNXRUNTIME_LIB}" AND NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could not find onnxruntime library under ${ONNXRUNTIME_ROOT}")
  endif()
  target_link_libraries(katago ${ONNXRUNTIME_LIB})

  # Required by onnxmodelbuilder.cpp for building ONNX graphs from .bin.gz.
  # These are intentionally configurable because package layouts vary.
  set(ONNX_INCLUDE_DIR "" CACHE PATH "Directory containing onnx/onnx_pb.h (required for .bin.gz -> ONNX conversion)")
  set(ONNX_PROTO_LIB "" CACHE FILEPATH "Path to onnx_proto library (required for .bin.gz -> ONNX conversion)")
  set(PROTOBUF_INCLUDE_DIR "" CACHE PATH "Directory containing google/protobuf/message.h (required for .bin.gz -> ONNX conversion)")
  set(PROTOBUF_LIB "" CACHE FILEPATH "Path to protobuf library (protobuf-lite or libprotobuf, required for .bin.gz -> ONNX conversion)")
  set(ONNX_PROTO_TARGET "" CACHE STRING "Imported CMake target for onnx_proto (optional, preferred over ONNX_PROTO_LIB)")
  set(PROTOBUF_TARGET "" CACHE STRING "Imported CMake target for protobuf (optional, preferred over PROTOBUF_LIB)")
  mark_as_advanced(CLEAR ONNX_INCLUDE_DIR ONNX_PROTO_LIB PROTOBUF_INCLUDE_DIR PROTOBUF_LIB)

  # Backward compatibility with older cache variable name.
  if(NOT PROTOBUF_LIB AND PROTOBUF_LITE_LIB)
    set(PROTOBUF_LIB "${PROTOBUF_LITE_LIB}")
  endif()

  if(KATAGO_AUTO_FETCH_DEPS)
    set(_katago_vcpkg_installed_root "${KATAGO_VCPKG_ROOT}/installed/${KATAGO_VCPKG_TRIPLET}")
  endif()

  if(KATAGO_AUTO_FETCH_DEPS)
    set(_need_onnx_proto_deps FALSE)
    if(NOT ONNX_INCLUDE_DIR OR NOT ONNX_PROTO_LIB OR NOT PROTOBUF_INCLUDE_DIR OR NOT PROTOBUF_LIB)
      set(_need_onnx_proto_deps TRUE)
    endif()
    if(_need_onnx_proto_deps)
      katago_vcpkg_install_if_needed("onnx")
      katago_vcpkg_install_if_needed("protobuf")
      set(_katago_vcpkg_installed_root "${KATAGO_VCPKG_ROOT}/installed/${KATAGO_VCPKG_TRIPLET}")
      if(EXISTS "${_katago_vcpkg_installed_root}/include/onnx/onnx_pb.h")
        set(ONNX_INCLUDE_DIR "${_katago_vcpkg_installed_root}/include" CACHE PATH "Directory containing onnx/onnx_pb.h (required for .bin.gz -> ONNX conversion)" FORCE)
      endif()
      find_library(_katago_onnx_proto_lib NAMES onnx_proto HINTS "${_katago_vcpkg_installed_root}/lib" "${_katago_vcpkg_installed_root}/debug/lib" NO_DEFAULT_PATH)
      if(_katago_onnx_proto_lib)
        set(ONNX_PROTO_LIB "${_katago_onnx_proto_lib}" CACHE FILEPATH "Path to onnx_proto library (required for .bin.gz -> ONNX conversion)" FORCE)
      endif()
      if(EXISTS "${_katago_vcpkg_installed_root}/include/google/protobuf/message.h")
        set(PROTOBUF_INCLUDE_DIR "${_katago_vcpkg_installed_root}/include" CACHE PATH "Directory containing google/protobuf/message.h (required for .bin.gz -> ONNX conversion)" FORCE)
      endif()
      find_library(_katago_protobuf_lib NAMES protobuf-lite libprotobuf-lite protobuf libprotobuf HINTS "${_katago_vcpkg_installed_root}/lib" "${_katago_vcpkg_installed_root}/debug/lib" NO_DEFAULT_PATH)
      if(_katago_protobuf_lib)
        set(PROTOBUF_LIB "${_katago_protobuf_lib}" CACHE FILEPATH "Path to protobuf library (protobuf-lite or libprotobuf, required for .bin.gz -> ONNX conversion)" FORCE)
      endif()
    endif()
  endif()

  # Prefer config-mode packages (vcpkg provides these) so transitive dependencies
  # like absl/utf8_range are linked automatically.
  if(_katago_vcpkg_installed_root)
    list(APPEND CMAKE_PREFIX_PATH "${_katago_vcpkg_installed_root}")
  endif()
  if(NOT ONNX_PROTO_TARGET)
    find_package(ONNX CONFIG QUIET)
    if(TARGET ONNX::onnx_proto)
      set(ONNX_PROTO_TARGET "ONNX::onnx_proto")
    endif()
  endif()
  if(NOT PROTOBUF_TARGET)
    if(_katago_vcpkg_installed_root AND EXISTS "${_katago_vcpkg_installed_root}/share/protobuf/protobuf-config.cmake")
      set(protobuf_DIR "${_katago_vcpkg_installed_root}/share/protobuf")
    endif()
    find_package(protobuf CONFIG QUIET)
    if(NOT TARGET protobuf::libprotobuf AND NOT TARGET protobuf::libprotobuf-lite)
      find_package(Protobuf CONFIG QUIET)
    endif()
    if(TARGET protobuf::libprotobuf)
      set(PROTOBUF_TARGET "protobuf::libprotobuf")
    elseif(TARGET protobuf::libprotobuf-lite)
      set(PROTOBUF_TARGET "protobuf::libprotobuf-lite")
    endif()
  endif()

  if(NOT ONNX_INCLUDE_DIR)
    find_path(ONNX_INCLUDE_DIR onnx/onnx_pb.h)
  endif()
  if(NOT ONNX_PROTO_LIB AND NOT ONNX_PROTO_TARGET)
    find_library(ONNX_PROTO_LIB onnx_proto)
  endif()
  if(NOT PROTOBUF_INCLUDE_DIR)
    find_path(PROTOBUF_INCLUDE_DIR google/protobuf/message.h)
  endif()
  if(NOT PROTOBUF_LIB AND NOT PROTOBUF_TARGET)
    find_library(PROTOBUF_LIB libprotobuf protobuf protobuf-lite protobuf-lite32)
  endif()

  if(NOT ONNX_INCLUDE_DIR OR NOT PROTOBUF_INCLUDE_DIR OR (NOT ONNX_PROTO_TARGET AND NOT ONNX_PROTO_LIB) OR (NOT PROTOBUF_TARGET AND NOT PROTOBUF_LIB))
    message(FATAL_ERROR
      "ONNX backend requires ONNX protobuf dependencies for .bin.gz model conversion. "
      "Set ONNX_INCLUDE_DIR (contains onnx/onnx_pb.h), ONNX_PROTO_LIB, PROTOBUF_INCLUDE_DIR, and PROTOBUF_LIB.")
  endif()
  target_include_directories(katago SYSTEM PRIVATE "${ONNX_INCLUDE_DIR}")
  target_include_directories(katago SYSTEM PRIVATE "${PROTOBUF_INCLUDE_DIR}")
  if(ONNX_PROTO_TARGET)
    target_link_libraries(katago ${ONNX_PROTO_TARGET})
  else()
    target_link_libraries(katago ${ONNX_PROTO_LIB})
  endif()
  if(PROTOBUF_TARGET)
    target_link_libraries(katago ${PROTOBUF_TARGET})
  else()
    target_link_libraries(katago ${PROTOBUF_LIB})
  endif()

  if(WIN32 AND ONNXRUNTIME_DLLS)
    foreach(_onnxruntime_dll IN LISTS ONNXRUNTIME_DLLS)
      add_custom_command(TARGET katago POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_onnxruntime_dll}"
          $<TARGET_FILE_DIR:katago>
      )
    endforeach()
  endif()
elseif(USE_BACKEND STREQUAL "EIGEN")
  target_compile_definitions(katago PRIVATE USE_EIGEN_BACKEND)
  if(NOT (MSVC))
    find_package(Eigen3 REQUIRED)
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
    message(STATUS "Found Eigen3 at ${EIGEN3_INCLUDE_DIRS}")
  else()
    if(EIGEN3_INCLUDE_DIRS)
      message(STATUS "EIGEN3_INCLUDE_DIRS is set to ${EIGEN3_INCLUDE_DIRS}, assuming that Eigen3 header files are here.")
      include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
    else()
      find_package(Eigen3)
      if(NOT Eigen3_FOUND)
        set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS} CACHE PATH "Directory containing 'Eigen' and 'unsupported' subdirs with Eigen headers")
        message(FATAL_ERROR "*** KataGo: Eigen3 was not found. If you've downloaded Eigen3, you can IGNORE the whole above error from CMake and just set 'EIGEN3_INCLUDE_DIRS' to the root eigen3-3.*.* directory after unzipping it. Among other things it should contain 'Eigen' and 'unsupported' subdirs, which is all KataGo needs. Otherwise, you will need to 'build' Eigen3 even though it is header-only and install it via MSVC in order for cmake to find it via FindEigen3.cmake.")
      else()
        include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
        message(STATUS "Found Eigen3 at ${EIGEN3_INCLUDE_DIRS}")
      endif()
    endif()
  endif()
endif()

if(USE_BIGGER_BOARDS_EXPENSIVE)
  target_compile_definitions(katago PRIVATE COMPILE_MAX_BOARD_LEN=50)
endif()

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  target_compile_definitions(katago PRIVATE NO_GIT_REVISION)
endif()

if(KATAGO_AUTO_FETCH_DEPS)
  set(_need_zlib_deps FALSE)
  if(NOT ZLIB_INCLUDE_DIR OR NOT ZLIB_LIBRARY)
    set(_need_zlib_deps TRUE)
  endif()
  if(_need_zlib_deps)
    katago_vcpkg_install_if_needed("zlib")
    set(_katago_vcpkg_installed_root "${KATAGO_VCPKG_ROOT}/installed/${KATAGO_VCPKG_TRIPLET}")
    if(EXISTS "${_katago_vcpkg_installed_root}/include/zlib.h")
      set(ZLIB_INCLUDE_DIR "${_katago_vcpkg_installed_root}/include" CACHE PATH "Path to directory with zlib.h and other header files" FORCE)
    endif()
    find_library(_katago_zlib_lib NAMES zlib z HINTS "${_katago_vcpkg_installed_root}/lib" "${_katago_vcpkg_installed_root}/debug/lib" NO_DEFAULT_PATH)
    if(_katago_zlib_lib)
      set(ZLIB_LIBRARY "${_katago_zlib_lib}" CACHE FILEPATH "Path to 'libz.so' on Linux or 'libz.lib' on Windows" FORCE)
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

# add_compile_definitions(NDEBUG)

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

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -pedantic -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Wmissing-declarations -Wmissing-include-dirs -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-overflow=1 -Wswitch-default -Wfloat-conversion -Wunused")

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

