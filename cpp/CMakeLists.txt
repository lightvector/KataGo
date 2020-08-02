cmake_minimum_required(VERSION 3.10.2)
project(katago)

set(CMAKE_CXX_STANDARD 14)

include_directories(external)
include_directories(external/tclap-1.2.2/include)

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

set(USE_BACKEND CACHE STRING "Neural net backend")
string(TOUPPER "${USE_BACKEND}" USE_BACKEND)
set_property(CACHE USE_BACKEND PROPERTY STRINGS "" CUDA OPENCL EIGEN)

set(USE_TCMALLOC 0 CACHE BOOL "Use TCMalloc")
set(NO_GIT_REVISION 0 CACHE BOOL "Disable embedding the git revision into the compiled exe")
set(Boost_USE_STATIC_LIBS_ON 0 CACHE BOOL "Compile against boost statically instead of dynamically")

#--------------------------- NEURAL NET BACKEND ------------------------------------------------------------------------

message("Building 'katago' executable for GTP engine and other tools")
if(USE_BACKEND STREQUAL "CUDA")
  message("-DUSE_BACKEND=CUDA, using CUDA backend")

  # Ensure dynamic cuda linking
  set(CMAKE_CUDA_FLAGS "" CACHE STRING "")
  if(CMAKE_CUDA_FLAGS)
    list(REMOVE_ITEM CMAKE_CUDA_FLAGS "-cudart static")
  endif()
  string(APPEND CMAKE_CUDA_FLAGS "-cudart shared")

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
  set(NEURALNET_BACKEND_SOURCES neuralnet/cudabackend.cpp neuralnet/cudahelpers.cu)
  set(CMAKE_CUDA_FLAGS
    "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70"
    )
elseif(USE_BACKEND STREQUAL "OPENCL")
  message("-DUSE_BACKEND=OPENCL, using OpenCL backend")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/openclbackend.cpp
    neuralnet/openclkernels.cpp
    neuralnet/openclhelpers.cpp
    neuralnet/opencltuner.cpp
    )
elseif(USE_BACKEND STREQUAL "EIGEN")
  message("-DUSE_BACKEND=EIGEN, using Eigen CPU backend")
  set(NEURALNET_BACKEND_SOURCES
    neuralnet/eigenbackend.cpp
    )
elseif(USE_BACKEND STREQUAL "")
  message(WARNING "${ColorBoldRed}WARNING: Using dummy neural net backend, intended for non-neural-net testing only, will fail on any code path requiring a neural net. To use neural net, specify -DUSE_BACKEND=CUDA or -DUSE_BACKEND=OPENCL to compile with the respective backend.${ColorReset}")
  set(NEURALNET_BACKEND_SOURCES neuralnet/dummybackend.cpp)
else()
  message(FATAL_ERROR "Unrecognized backend: " ${USE_BACKEND})
endif()


#--------------------------- TCMALLOC ----------------------------------------------------------------------------------

if(USE_TCMALLOC)
  message("-DUSE_TCMALLOC=1 is set, using tcmalloc as the allocator")
  find_library(TCMALLOC_LIB NAMES tcmalloc_minimal HINTS /usr)
  if(NOT TCMALLOC_LIB)
    message(FATAL_ERROR "Could not find tcmalloc")
  endif()
endif()

# set (Gperftools_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")
# find_package(Gperftools REQUIRED)

#--------------------------- GIT ---------------------------------------------------------------------------------------

if(NO_GIT_REVISION)
  message("-DNO_GIT_REVISION=1 is set, avoiding including the Git revision in compiled executable")
  unset(GIT_HEADER_FILE_ALWAYS_UPDATED)
else()
  message("Including Git revision in the compiled executable, specify -DNO_GIT_REVISION=1 to disable")
  find_package(Git)
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
  command/commandline.cpp
  command/analysis.cpp
  command/benchmark.cpp
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
  find_package(CUDA REQUIRED)
  find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES cuda/include include)
  if((NOT CUDNN_INCLUDE_DIR))
    message(ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is. ${ColorReset}")
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
  find_package(Eigen3 REQUIRED)
  include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
  message("Found Eigen3 at ${EIGEN3_INCLUDE_DIRS}")
endif()

if(NO_GIT_REVISION)
  target_compile_definitions(katago PRIVATE NO_GIT_REVISION)
endif()

find_package(ZLIB REQUIRED)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  target_link_libraries(katago ${ZLIB_LIBRARIES})
endif(ZLIB_FOUND)

find_library(LIBZIP_LIBRARY NAMES zip)
find_path(LIBZIP_INCLUDE_DIR_ZIP NAMES zip.h)
find_path(LIBZIP_INCLUDE_DIR_ZIPCONF NAMES zipconf.h)
if((NOT LIBZIP_LIBRARY) OR (NOT LIBZIP_INCLUDE_DIR_ZIP) OR (NOT LIBZIP_INCLUDE_DIR_ZIPCONF))
  target_compile_definitions(katago PRIVATE NO_LIBZIP)
  message(WARNING "${ColorBoldRed}WARNING: libzip library was NOT found. KataGo should still work for GTP/matches/analysis if everything else is good, but selfplay for writing training data will not be possible.${ColorReset}")
else()
  include_directories(${LIBZIP_INCLUDE_DIR_ZIP})
  include_directories(${LIBZIP_INCLUDE_DIR_ZIPCONF})
  target_link_libraries(katago ${LIBZIP_LIBRARY})
endif()

if(Boost_USE_STATIC_LIBS_ON)
  set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(katago ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})

if(USE_TCMALLOC)
  target_link_libraries(katago ${TCMALLOC_LIB})
endif(USE_TCMALLOC)


#------------------------------------------------------------------------------------

# add_compile_definitions(NDEBUG)

if(MSVC)
  # Getting boost to work on MSVC seems to require forcing it to link
  # with dlls instead of static, and needs this option
  if(NOT Boost_USE_STATIC_LIBS_ON)
    target_compile_definitions(katago PRIVATE BOOST_ALL_DYN_LINK)
  endif()

  # Suppress min and max macros on windows
  # Also define a few other things for windows
  target_compile_definitions(katago PRIVATE NOMINMAX)
  target_compile_definitions(katago PRIVATE BYTE_ORDER=1234)
  target_compile_definitions(katago PRIVATE LITTLE_ENDIAN=1234)
  target_compile_definitions(katago PRIVATE BIG_ENDIAN=4321)
  # core/rand.cpp uses winsock for a gethostname
  target_link_libraries(katago ws2_32)

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:8388608")
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  if(NOT (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm"))
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse")
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
