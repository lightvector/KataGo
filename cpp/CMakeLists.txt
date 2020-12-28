cmake_minimum_required(VERSION 3.10.2)
project(katago)

set(CMAKE_CXX_STANDARD 14)

include_directories(external)
include_directories(external/tclap-1.2.2/include)
include_directories(SYSTEM external/filesystem-1.3.6/include) #SYSTEM suppresses a few warnings

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0")
  # Suppress a warning message for setting cuda flags that still works fine for now
  cmake_policy(SET CMP0104 OLD)
endif()

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
set_property(CACHE USE_BACKEND PROPERTY STRINGS "" CUDA OPENCL EIGEN)

set(USE_TCMALLOC 0 CACHE BOOL "Use TCMalloc")
set(NO_GIT_REVISION 0 CACHE BOOL "Disable embedding the git revision into the compiled exe")
set(USE_AVX2 0 CACHE BOOL "Compile with AVX2")
set(USE_BIGGER_BOARDS_EXPENSIVE 0 CACHE BOOL "Allow boards up to size 29. Compiling with this Will use more memory and slow down KataGo, even when playing on boards of size 19.")

#--------------------------- NEURAL NET BACKEND ------------------------------------------------------------------------

message(STATUS "Building 'katago' executable for GTP engine and other tools.")
if(USE_BACKEND STREQUAL "CUDA")
  message(STATUS "-DUSE_BACKEND=CUDA, using CUDA backend.")


  # Ensure dynamic cuda linking (Versions prior to 3.17)
  if (${CMAKE_VERSION} VERSION_LESS "3.17")
    set(CMAKE_CUDA_FLAGS "" CACHE STRING "")
    if(CMAKE_CUDA_FLAGS)
      list(REMOVE_ITEM CMAKE_CUDA_FLAGS "-cudart static")
    endif()
    string(APPEND CMAKE_CUDA_FLAGS "-cudart shared")
  endif()

  enable_language(CUDA)

  if(MSVC)
    # Ensure dynamic cuda linking
    if(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES)
      list(REMOVE_ITEM CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "cudart_static")
      list(REMOVE_ITEM CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "cudadevrt")
      list(APPEND CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "cudart")
    endif()
    if(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES)
      list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "cudart_static")
      list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "cudadevrt")
      list(APPEND CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "cudart")
    endif()
  endif()

  set(CUDA_STANDARD 11)

  # Ensure dynamic cuda linking (Versions from 3.17)
  if (${CMAKE_VERSION} VERSION_GREATER "3.16")
    set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
  endif()

  set(NEURALNET_BACKEND_SOURCES neuralnet/cudabackend.cpp neuralnet/cudahelpers.cu)

  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
    set(CMAKE_CUDA_FLAGS
      " \
      -gencode arch=compute_35,code=sm_35       \
      -gencode arch=compute_35,code=compute_35  \
      -gencode arch=compute_37,code=sm_37       \
      -gencode arch=compute_37,code=compute_37  \
      -gencode arch=compute_50,code=sm_50       \
      -gencode arch=compute_50,code=compute_50  \
      -gencode arch=compute_53,code=sm_53       \
      -gencode arch=compute_53,code=compute_53  \
      -gencode arch=compute_60,code=sm_60       \
      -gencode arch=compute_60,code=compute_60  \
      -gencode arch=compute_61,code=sm_61       \
      -gencode arch=compute_61,code=compute_61  \
      -gencode arch=compute_62,code=sm_62       \
      -gencode arch=compute_62,code=compute_62  \
      -gencode arch=compute_70,code=sm_70       \
      -gencode arch=compute_70,code=compute_70  \
      -gencode arch=compute_72,code=sm_72       \
      -gencode arch=compute_72,code=compute_72  \
      -gencode arch=compute_75,code=sm_75       \
      -gencode arch=compute_75,code=compute_75  \
      -gencode arch=compute_80,code=sm_80       \
      -gencode arch=compute_80,code=compute_80  \
      -Wno-deprecated-gpu-targets \
      "
      )
  elseif (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 10.2)
    set(CMAKE_CUDA_FLAGS
      " \
      -gencode arch=compute_30,code=sm_30       \
      -gencode arch=compute_30,code=compute_30  \
      -gencode arch=compute_35,code=sm_35       \
      -gencode arch=compute_35,code=compute_35  \
      -gencode arch=compute_37,code=sm_37       \
      -gencode arch=compute_37,code=compute_37  \
      -gencode arch=compute_50,code=sm_50       \
      -gencode arch=compute_50,code=compute_50  \
      -gencode arch=compute_53,code=sm_53       \
      -gencode arch=compute_53,code=compute_53  \
      -gencode arch=compute_60,code=sm_60       \
      -gencode arch=compute_60,code=compute_60  \
      -gencode arch=compute_61,code=sm_61       \
      -gencode arch=compute_61,code=compute_61  \
      -gencode arch=compute_62,code=sm_62       \
      -gencode arch=compute_62,code=compute_62  \
      -gencode arch=compute_70,code=sm_70       \
      -gencode arch=compute_70,code=compute_70  \
      -gencode arch=compute_72,code=sm_72       \
      -gencode arch=compute_72,code=compute_72  \
      -gencode arch=compute_75,code=sm_75       \
      -gencode arch=compute_75,code=compute_75  \
      -Wno-deprecated-gpu-targets \
      "
      )
  else()
      message(WARNING "CUDA 10.2 or greater is recommended, but attempting to build anyways")
      set(CMAKE_CUDA_FLAGS
        " \
        -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_30,code=compute_30 \
        -gencode arch=compute_37,code=sm_37 \
        -gencode arch=compute_53,code=sm_53 \
        -gencode arch=compute_53,code=compute_53 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_70,code=compute_70 \
        "
        )
  endif()
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
elseif(USE_BACKEND STREQUAL "")
  message(WARNING "${ColorBoldRed}WARNING: Using dummy neural net backend, intended for non-neural-net testing only, will fail on any code path requiring a neural net. To use neural net, specify -DUSE_BACKEND=CUDA or -DUSE_BACKEND=OPENCL or -DUSE_BACKEND=EIGEN to compile with the respective backend.${ColorReset}")
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
    message(STATUS "${ColorRed}NO_GIT_REVISION is set, but but BUILD_DISTRIBUTED is also set and distributed requires git revision, so ignoring NO_GIT_REVISION.${ColorReset}")
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
      message(SEND_ERROR "${ColorBoldRed}Git executable was not found. Either specify GIT_EXECUTABLE as the path to the git executable.${ColorReset}")
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
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    VERBATIM
    )
endif()

#--------------------------- KATAGO COMPILING AND LINKING --------------------------------------------------------------

add_executable(katago
  core/global.cpp
  core/bsearch.cpp
  core/config_parser.cpp
  core/datetime.cpp
  core/elo.cpp
  core/fancymath.cpp
  core/hash.cpp
  core/logger.cpp
  core/makedir.cpp
  core/md5.cpp
  core/multithread.cpp
  core/rand.cpp
  core/rand_helpers.cpp
  core/sha2.cpp
  core/test.cpp
  core/threadsafequeue.cpp
  core/timer.cpp
  game/board.cpp
  game/rules.cpp
  game/boardhistory.cpp
  dataio/sgf.cpp
  dataio/numpywrite.cpp
  dataio/trainingwrite.cpp
  dataio/loadmodel.cpp
  dataio/lzparse.cpp
  dataio/homedata.cpp
  neuralnet/nninputs.cpp
  neuralnet/modelversion.cpp
  neuralnet/nneval.cpp
  neuralnet/desc.cpp
  ${NEURALNET_BACKEND_SOURCES}
  search/timecontrols.cpp
  search/searchparams.cpp
  search/mutexpool.cpp
  search/search.cpp
  search/searchresults.cpp
  search/asyncbot.cpp
  search/distributiontable.cpp
  search/subtreevaluebiastable.cpp
  search/analysisdata.cpp
  program/gtpconfig.cpp
  program/setup.cpp
  program/playutils.cpp
  program/playsettings.cpp
  program/play.cpp
  program/selfplaymanager.cpp
  ${GIT_HEADER_FILE_ALWAYS_UPDATED}
  tests/testboardarea.cpp
  tests/testboardbasic.cpp
  tests/testcommon.cpp
  tests/testrules.cpp
  tests/testscore.cpp
  tests/testsgf.cpp
  tests/testnninputs.cpp
  tests/testownership.cpp
  tests/testsearch.cpp
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
  command/gtp.cpp
  command/lzcost.cpp
  command/match.cpp
  command/matchauto.cpp
  command/misc.cpp
  command/runtests.cpp
  command/sandbox.cpp
  command/selfplay.cpp
  command/tune.cpp
  main.cpp
  )

if(USE_BACKEND STREQUAL "CUDA")
  target_compile_definitions(katago PRIVATE USE_CUDA_BACKEND)
  target_compile_definitions(katago PRIVATE CUDA_TARGET_VERSION=${CMAKE_CUDA_COMPILER_VERSION})
  find_package(CUDA REQUIRED)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES cuda/include include)
  if((NOT CUDNN_INCLUDE_DIR))
    message(FATAL_ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is. ${ColorReset}")
  endif()
  find_library(CUDNN_LIBRARY libcudnn.so PATHS /usr/local/cuda/lib64 /opt/cuda/lib64)
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIR}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
  target_link_libraries(katago ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
elseif(USE_BACKEND STREQUAL "OPENCL")
  target_compile_definitions(katago PRIVATE USE_OPENCL_BACKEND)
  find_package(OpenCL REQUIRED)
  include_directories(${OpenCL_INCLUDE_DIRS})
  link_directories(${OpenCL_LIBRARY})
  target_link_libraries (katago ${OpenCL_LIBRARY})
elseif(USE_BACKEND STREQUAL "EIGEN")
  target_compile_definitions(katago PRIVATE USE_EIGEN_BACKEND)
  if(NOT MSVC)
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
  target_compile_definitions(katago PRIVATE COMPILE_MAX_BOARD_LEN=29)
endif()

if(NO_GIT_REVISION AND (NOT BUILD_DISTRIBUTED))
  target_compile_definitions(katago PRIVATE NO_GIT_REVISION)
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
    message(SEND_ERROR "${ColorBoldRed}WARNING: BUILD_DISTRIBUTED was requested but libzip library was NOT found. KataGo needs this for writing training data so libzip is required. On Linux, install through your normal package manager. On Windows, set LIBZIP_INCLUDE_DIR_ZIP to the directory that includes zip.h and other files, and LIBZIP_INCLUDE_DIR_ZIPCONF to the directory that includes zipconf.h and other files, and LIBZIP_LIBRARY to the libzip.lib file. ${ColorReset}")
  endif()
  target_compile_definitions(katago PRIVATE NO_LIBZIP)
  message(WARNING "${ColorBoldRed}WARNING: libzip library was NOT found. KataGo should still work for GTP/matches/analysis if everything else is good, but selfplay for writing training data will not be possible.${ColorReset}")
  set(LIBZIP_INCLUDE_DIR_ZIP ${LIBZIP_INCLUDE_DIR_ZIP} CACHE PATH "Path to directory with zip.h and other header files")
  set(LIBZIP_INCLUDE_DIR_ZIPCONF ${LIBZIP_INCLUDE_DIR_ZIPCONF} CACHE PATH "Path to directory with zipconf.h and other header files")
  set(LIBZIP_LIBRARY ${LIBZIP_LIBRARY} CACHE FILEPATH "Path to 'libzip.so' on Linux or 'libzip.lib' on Windows")
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
  include_directories(external/httplib)
endif()

#------------------------------------------------------------------------------------

# add_compile_definitions(NDEBUG)

if(MSVC)
  # Suppress min and max macros on windows
  # Also define a few other things for windows
  target_compile_definitions(katago PRIVATE NOMINMAX)
  target_compile_definitions(katago PRIVATE BYTE_ORDER=1234)
  target_compile_definitions(katago PRIVATE LITTLE_ENDIAN=1234)
  target_compile_definitions(katago PRIVATE BIG_ENDIAN=4321)
  # core/rand.cpp uses winsock for a gethostname
  target_link_libraries(katago ws2_32)

  if(USE_AVX2)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /arch:AVX2 -D__FMA__")
    target_compile_definitions(katago PRIVATE USE_AVX2)
  endif()

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:8388608")
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "(arm|aarch32|aarch64)"))
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse")
  endif()
  if(USE_AVX2)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    target_compile_definitions(katago PRIVATE USE_AVX2)
  endif()

  # On g++ it seems like we need to explicitly link threads as well.
  # It seems sometimes this is implied by other options automatically like when we enable CUDA, but we get link errors
  # if we don't explicitly require threads it when attempting to build without CUDA
  if(NOT USE_BACKEND STREQUAL "CUDA")
    find_package (Threads REQUIRED)
    target_link_libraries(katago Threads::Threads)
  endif()

  if(USE_TCMALLOC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -O2 -pedantic -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wstrict-overflow=1 -Wswitch-default -Wfloat-conversion -Wnull-dereference -Wunused -Walloc-zero -Wduplicated-branches -Wduplicated-cond -Wdangling-else -Wrestrict -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
  else()
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -O2 -pedantic -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wstrict-overflow=1 -Wswitch-default -Wfloat-conversion -Wnull-dereference -Wunused -Walloc-zero -Wduplicated-branches -Wduplicated-cond -Wdangling-else -Wrestrict")
  endif()
endif()

target_include_directories(katago PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

