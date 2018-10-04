cmake_minimum_required(VERSION 3.8.2)
project(writedata)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(tclap-1.2.1/include)

#------------------------------------------------------------------------------------

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
    set(CMAKE_CUDA_FLAGS
      "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70"
      )

  else()
    message(FATAL_ERROR "Please specify -DUSE_TENSORFLOW_BACKEND=0 -DUSE_CUDA_BACKEND=1  OR  -DUSE_TENSORFLOW_BACKEND=1 -DUSE_CUDA_BACKEND=0")
  endif()
endif()

if((NOT BUILD_WRITE) AND (NOT BUILD_MCTS))
  message(FATAL_ERROR "Please specify one or more of -DBUILD_WRITE=1 or -DBUILD_MCTS=1")
endif()

#------------------------------------------------------------------------------------

find_package(Git)
set(GIT_HEADER_FILE_TEMPLATE_BARE program/gitinfotemplate.h)
set(GIT_HEADER_FILE_ALWAYS_UPDATED_BARE program/gitinfoupdated.h)
set(GIT_HEADER_FILE_BARE program/gitinfo.h)
set(GIT_HEADER_FILE_TEMPLATE ${CMAKE_BINARY_DIR}/${GIT_HEADER_FILE_TEMPLATE_BARE})
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
endif()

#------------------------------------------------------------------------------------

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
    neuralnet/modelversion.cpp
    neuralnet/nneval.cpp
    ${NEURALNET_BACKEND_SOURCES}
    search/searchparams.cpp
    search/mutexpool.cpp
    search/search.cpp
    search/asyncbot.cpp
    search/distributiontable.cpp
    program/setup.cpp
    ${GIT_HEADER_FILE_ALWAYS_UPDATED}
    tests/testboardarea.cpp
    tests/testboardbasic.cpp
    tests/testrules.cpp
    tests/testnninputs.cpp
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


#------------------------------------------------------------------------------------


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
