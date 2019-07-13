cmake_minimum_required(VERSION 3.12.3)
project(writedata)

set(CMAKE_CXX_STANDARD 14)

include_directories(external)
include_directories(external/tclap-1.2.1/include)

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

#------------------------------------------------------------------------------------

set(BUILD_WRITE 0 CACHE BOOL "Build 'write' executable for neural net training data (deprecated)")
set(BUILD_MCTS 1 CACHE BOOL "Build 'katago' for GTP engine and other tools (you probably want this)")
set(USE_OPENCL_BACKEND 0 CACHE BOOL "Use OpenCL backend")
set(USE_CUDA_BACKEND 0 CACHE BOOL "Use CUDA backend")
set(USE_TCMALLOC 0 CACHE BOOL "Use TCMalloc")
set(NO_GIT_REVISION 0 CACHE BOOL "Disable embedding the git revision into the compiled exe")

#------------------------------------------------------------------------------------

if(BUILD_WRITE)
  message("-DBUILD_WRITE=1 is set, building 'write' executable for writing neural net training data from sgfs or LZ data files")
endif()
if(BUILD_MCTS)
  message("-DBUILD_MCTS=1 is set, building 'katago' executable for mcts-backed GTP engine and other tools")

  if((USE_CUDA_BACKEND) AND (USE_OPENCL_BACKEND))
    message(FATAL_ERROR "Please specify only one of -DUSE_CUDA_BACKEND=1 and -DUSE_OPENCL_BACKEND=1")
  endif()

  if(USE_CUDA_BACKEND)
    message("-DUSE_CUDA_BACKEND=1 is set, using CUDA backend")
    enable_language(CUDA)
    set(CUDA_STANDARD 11)
    set(NEURALNET_BACKEND_SOURCES neuralnet/cudabackend.cpp neuralnet/cudahelpers.cu)
    set(CMAKE_CUDA_FLAGS
      "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70"
      )
  elseif(USE_OPENCL_BACKEND)
    message("-DUSE_OPENCL_BACKEND=1 is set, using OpenCL backend")
    set(NEURALNET_BACKEND_SOURCES
      neuralnet/openclbackend.cpp
      neuralnet/openclkernels.cpp
      neuralnet/openclhelpers.cpp
      neuralnet/opencltuner.cpp
      )
  else()
    message(WARNING "${ColorBoldRed}WARNING: Using dummy neural net backend, intended for non-neural-net testing only, will fail on any code path requiring a neural net. Specify -DUSE_CUDA_BACKEND=1 to compile with CUDA to use neural net.${ColorReset}")
    set(NEURALNET_BACKEND_SOURCES neuralnet/dummybackend.cpp)
  endif()
endif()

if((NOT BUILD_WRITE) AND (NOT BUILD_MCTS))
  message(FATAL_ERROR "Please specify one or more of -DBUILD_WRITE=1 or -DBUILD_MCTS=1")
endif()

if(USE_TCMALLOC)
  message("-DUSE_TCMALLOC=1 is set, using tcmalloc as the allocator")
endif()

# set (Gperftools_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")
# find_package(Gperftools REQUIRED)

#------------------------------------------------------------------------------------

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

include(CheckIncludeFile)
check_include_file("dirent.h" HAVE_DIRENT)
if(NOT HAVE_DIRENT)
  unset(HAVE_DIRENT CACHE)
  message(WARNING "dirent.h not found, compiling without it (KataGo should still work fine except for some rare dev-only test tools)")
  add_compile_definitions(NO_DIRENT_H)
endif()

#------------------------------------------------------------------------------------

if(USE_TCMALLOC)
  find_library(TCMALLOC_LIB NAMES tcmalloc_minimal HINTS /usr)
  if(NOT TCMALLOC_LIB)
    message(FATAL_ERROR "Could not find tcmalloc")
  endif()
endif()

#------------------------------------------------------------------------------------

if(BUILD_WRITE)
  add_executable(write
    core/global.cpp
    core/hash.cpp
    core/md5.cpp
    core/rand.cpp
    core/sha2.cpp
    core/timer.cpp
    game/board.cpp
    game/rules.cpp
    game/boardhistory.cpp
    neuralnet/nninputs.cpp
    dataio/datapool.cpp
    dataio/lzparse.cpp
    dataio/sgf.cpp
    ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    write.cpp
    )

  find_package(HDF5 REQUIRED COMPONENTS C CXX)

  if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries (write ${HDF5_LIBRARIES})
  endif (HDF5_FOUND)

  if(NO_GIT_REVISION)
    target_compile_definitions(write PRIVATE NO_GIT_REVISION)
  endif()

  #Suppress min and max macros on windows
  if(MSVC)
    target_compile_definitions(write PRIVATE NOMINMAX)
  endif()

endif()

#------------------------------------------------------------------------------------

if(BUILD_MCTS)
  add_executable(katago
    core/global.cpp
    core/config_parser.cpp
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
    search/asyncbot.cpp
    search/distributiontable.cpp
    search/analysisdata.cpp
    program/setup.cpp
    program/play.cpp
    ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    tests/testboardarea.cpp
    tests/testboardbasic.cpp
    tests/testrules.cpp
    tests/testscore.cpp
    tests/testsgf.cpp
    tests/testnninputs.cpp
    tests/testsearch.cpp
    tests/testtime.cpp
    tests/testtrainingwrite.cpp
    tests/testnn.cpp
    evalsgf.cpp
    gatekeeper.cpp
    gtp.cpp
    match.cpp
    matchauto.cpp
    selfplay.cpp
    misc.cpp
    runtests.cpp
    lzcost.cpp
    sandbox.cpp
    tune.cpp
    main.cpp
    )

  if(USE_CUDA_BACKEND)
    target_compile_definitions(katago PRIVATE USE_CUDA_BACKEND)
    find_package(CUDA REQUIRED)
    find_library(CUDNN_LIBRARY libcudnn.so PATHS /usr/local/cuda/lib64 /opt/cuda/lib64)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
    target_link_libraries(katago ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
  elseif(USE_OPENCL_BACKEND)
    target_compile_definitions(katago PRIVATE USE_OPENCL_BACKEND)
    find_package(OpenCL REQUIRED)
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARY})
    target_link_libraries (katago ${OpenCL_LIBRARY})
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

  find_package(Boost COMPONENTS system filesystem REQUIRED)
  include_directories(${Boost_INCLUDE_DIRS})
  target_link_libraries(katago ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})
  # Getting boost to work on MSVC seems to require forcing it to link
  # with dlls instead of static, and needs these options
  if(MSVC)
    target_compile_definitions(katago PRIVATE BOOST_ALL_NO_LIB)
    target_compile_definitions(katago PRIVATE BOOST_ALL_DYN_LINK)
  endif()
  
  if(USE_TCMALLOC)
    target_link_libraries(katago ${TCMALLOC_LIB})
  endif(USE_TCMALLOC)

  # Suppress min and max macros on windows
  # Also define a few other things for windows
  if(MSVC)
    target_compile_definitions(katago PRIVATE NOMINMAX)
    target_compile_definitions(katago PRIVATE BYTE_ORDER=1234)
    target_compile_definitions(katago PRIVATE LITTLE_ENDIAN=1234)
    target_compile_definitions(katago PRIVATE BIG_ENDIAN=4321)
    # core/rand.cpp uses winsock for a gethostname
    target_link_libraries(katago ws2_32)
  endif()

  # On g++ it seems like we need to explicitly link threads as well.
  # It seems sometimes this is implied by other options automatically like when we enable CUDA, but we get link errors
  # if we don't explicitly require threads it when attempting to build without CUDA
  if(CMAKE_COMPILER_IS_GNUCC AND (NOT USE_CUDA_BACKEND))
    find_package (Threads REQUIRED)
    target_link_libraries(katago Threads::Threads)
  endif()

endif()


#------------------------------------------------------------------------------------

# add_compile_definitions(NDEBUG)

if(CMAKE_COMPILER_IS_GNUCC)
  if(USE_TCMALLOC)
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse -g -O2 -pedantic -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wstrict-overflow=1 -Wswitch-default -Wsizeof-pointer-memaccess -Wuninitialized -Winit-self -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
  else()
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse -g -O2 -pedantic -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wstrict-overflow=1 -Wswitch-default -Wsizeof-pointer-memaccess -Wuninitialized -Winit-self")
  endif()
endif()

target_include_directories(katago PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
