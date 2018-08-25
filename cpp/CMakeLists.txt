cmake_minimum_required(VERSION 3.8.2)
project(writedata)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(tclap-1.2.1/include)

if(BUILD_WRITE)
  message("-DBUILD_WRITE=1 is set, building 'write' executable for writing neural net training data from sgfs or LZ data files")
endif()
if(BUILD_MCTS)
  message("-DBUILD_MCTS=1 is set, building 'main' executable for mcts-backed GTP engine and other tools")
  
  if(USE_TENSORFLOW_BACKEND)
    set(NEURALNET_BACKEND_SOURCES neuralnet/tensorflowbackend.cpp)

  elseif(USE_CUDA_BACKEND)
    enable_language(CUDA)
    set(CUDA_STANDARD 11)
    set(NEURALNET_BACKEND_SOURCES neuralnet/cudabackend.cpp neuralnet/cudahelpers.cu)

  else()
    message(FATAL_ERROR "Please specify -DUSE_TENSORFLOW_BACKEND=0 -DUSE_CUDA_BACKEND=1  OR  -DUSE_TENSORFLOW_BACKEND=1 -DUSE_CUDA_BACKEND=0")
  endif()
endif()

if((NOT BUILD_WRITE) AND (NOT BUILD_MCTS))
  message(FATAL_ERROR "Please specify one or more of -DBUILD_WRITE=1 or -DBUILD_MCTS=1")
endif()



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
    write.cpp
    )

  find_package(HDF5 REQUIRED COMPONENTS C CXX)

  if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries (write ${HDF5_LIBRARIES})
  endif (HDF5_FOUND)
endif()


if(BUILD_MCTS)
  add_executable(main
    core/global.cpp
    core/config_parser.cpp
    core/fancymath.cpp
    core/hash.cpp
    core/logger.cpp
    core/makedir.cpp
    core/md5.cpp
    core/rand.cpp
    core/sha2.cpp
    core/timer.cpp
    game/board.cpp
    game/rules.cpp
    game/boardhistory.cpp
    dataio/sgf.cpp
    neuralnet/nninputs.cpp
    neuralnet/nneval.cpp
    ${NEURALNET_BACKEND_SOURCES}
    search/searchparams.cpp
    search/mutexpool.cpp
    search/search.cpp
    search/asyncbot.cpp
    program/setup.cpp
    tests/testboardarea.cpp
    tests/testboardbasic.cpp
    tests/testrules.cpp
    evalsgf.cpp
    gtp.cpp
    match.cpp
    misc.cpp
    runtests.cpp
    sandbox.cpp
    main.cpp
    )

  if(USE_TENSORFLOW_BACKEND)
    add_definitions(-DUSE_TENSORFLOW_BACKEND)
    find_package(TensorflowCC REQUIRED)
    target_link_libraries(main TensorflowCC::Shared)
    find_package(CUDA)
    if(CUDA_FOUND)
      target_link_libraries(main ${CUDA_LIBRARIES})
    endif()
  endif()

  if(USE_CUDA_BACKEND)
    add_definitions(-DUSE_CUDA_BACKEND)
    find_package(CUDA REQUIRED)
    find_library(CUDNN_LIBRARY libcudnn.so /usr/local/cuda/lib64)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS}) #SYSTEM is for suppressing some compiler warnings in thrust libraries
    target_link_libraries(main ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
  endif()

endif()




if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wsuggest-final-types -Wsuggest-final-methods -Wsizeof-pointer-memaccess -Wuninitialized -Winit-self")
endif ()





# find_package(ZLIB REQUIRED)
# if (ZLIB_FOUND)
#     include_directories(${ZLIB_INCLUDE_DIRS})
#     target_link_libraries (write ${ZLIB_LIBRARIES})
# endif (ZLIB_FOUND)

# find_library(BLOSC_LIB blosc)
# target_link_libraries(write ${BLOSC_LIB})
