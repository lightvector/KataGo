cmake_minimum_required(VERSION 3.5)
project(writedata)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)
include_directories(core)
include_directories(tclap-1.2.1)
include_directories(tclap-1.2.1/include)
include_directories(tclap-1.2.1/include/tclap)

add_executable(write
  core/config.h
  core/global.cpp
  core/global.h
  core/hash.cpp
  core/hash.h
  core/md5.cpp
  core/md5.h
  core/multithread.h
  core/rand.cpp
  core/rand.h
  core/rand_helpers.h
  core/sha2.cpp
  core/sha2.h
  core/timer.cpp
  core/timer.h
  tclap-1.2.1/include/tclap/Arg.h
  tclap-1.2.1/include/tclap/ArgException.h
  tclap-1.2.1/include/tclap/ArgTraits.h
  tclap-1.2.1/include/tclap/CmdLine.h
  tclap-1.2.1/include/tclap/CmdLineInterface.h
  tclap-1.2.1/include/tclap/CmdLineOutput.h
  tclap-1.2.1/include/tclap/Constraint.h
  tclap-1.2.1/include/tclap/DocBookOutput.h
  tclap-1.2.1/include/tclap/HelpVisitor.h
  tclap-1.2.1/include/tclap/IgnoreRestVisitor.h
  tclap-1.2.1/include/tclap/MultiArg.h
  tclap-1.2.1/include/tclap/MultiSwitchArg.h
  tclap-1.2.1/include/tclap/OptionalUnlabeledTracker.h
  tclap-1.2.1/include/tclap/StandardTraits.h
  tclap-1.2.1/include/tclap/StdOutput.h
  tclap-1.2.1/include/tclap/SwitchArg.h
  tclap-1.2.1/include/tclap/UnlabeledMultiArg.h
  tclap-1.2.1/include/tclap/UnlabeledValueArg.h
  tclap-1.2.1/include/tclap/ValueArg.h
  tclap-1.2.1/include/tclap/ValuesConstraint.h
  tclap-1.2.1/include/tclap/VersionVisitor.h
  tclap-1.2.1/include/tclap/Visitor.h
  tclap-1.2.1/include/tclap/XorHandler.h
  tclap-1.2.1/include/tclap/ZshCompletionOutput.h
  zstr/src/strict_fstream.hpp
  zstr/src/zstr.hpp
  game/board.cpp
  game/board.h
  game/rules.cpp
  game/rules.h
  game/boardhistory.cpp
  game/boardhistory.h
  neuralnet/nninputs.cpp
  neuralnet/nninputs.h
  dataio/datapool.cpp
  dataio/datapool.h
  dataio/lzparse.cpp
  dataio/lzparse.h
  dataio/sgf.cpp
  dataio/sgf.h
  write.cpp
  )


add_executable(runtests
  core/config.h
  core/global.cpp
  core/global.h
  core/hash.cpp
  core/hash.h
  core/md5.cpp
  core/md5.h
  core/multithread.h
  core/rand.cpp
  core/rand.h
  core/rand_helpers.h
  core/sha2.cpp
  core/sha2.h
  core/timer.cpp
  core/timer.h
  game/board.cpp
  game/board.h
  game/rules.cpp
  game/rules.h
  game/boardhistory.cpp
  game/boardhistory.h
  tests/tests.h
  tests/testboardarea.cpp
  tests/testboardbasic.cpp
  tests/testmisc.cpp
  tests/testrules.cpp
  runtests.cpp
  )

add_executable(example
  core/config.h
  core/global.cpp
  core/global.h
  core/config_parser.cpp
  core/config_parser.h
  core/hash.cpp
  core/hash.h
  core/logger.h
  core/logger.cpp
  core/md5.cpp
  core/md5.h
  core/multithread.h
  core/rand.cpp
  core/rand.h
  core/rand_helpers.h
  core/sha2.cpp
  core/sha2.h
  core/timer.cpp
  core/timer.h
  game/board.cpp
  game/board.h
  game/rules.cpp
  game/rules.h
  game/boardhistory.cpp
  game/boardhistory.h
  neuralnet/nninputs.cpp
  neuralnet/nninputs.h
  neuralnet/nneval.cpp
  neuralnet/nneval.h
  search/searchparams.cpp
  search/searchparams.h
  search/mutexpool.cpp
  search/mutexpool.h
  search/search.cpp
  search/search.h
  search/asyncbot.cpp
  search/asyncbot.h
  example.cpp
  )

add_executable(gtp
  core/config.h
  core/global.cpp
  core/global.h
  core/config_parser.cpp
  core/config_parser.h
  core/hash.cpp
  core/hash.h
  core/logger.h
  core/logger.cpp
  core/md5.cpp
  core/md5.h
  core/multithread.h
  core/rand.cpp
  core/rand.h
  core/rand_helpers.h
  core/sha2.cpp
  core/sha2.h
  core/timer.cpp
  core/timer.h
  game/board.cpp
  game/board.h
  game/rules.cpp
  game/rules.h
  game/boardhistory.cpp
  game/boardhistory.h
  neuralnet/nninputs.cpp
  neuralnet/nninputs.h
  neuralnet/nneval.cpp
  neuralnet/nneval.h
  search/searchparams.cpp
  search/searchparams.h
  search/mutexpool.cpp
  search/mutexpool.h
  search/search.cpp
  search/search.h
  search/asyncbot.cpp
  search/asyncbot.h
  gtp.cpp
  )

find_package(TensorflowCC REQUIRED)
target_link_libraries(example TensorflowCC::Shared)
target_link_libraries(gtp TensorflowCC::Shared)
find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(example ${CUDA_LIBRARIES})
  target_link_libraries(gtp ${CUDA_LIBRARIES})
endif()

find_package(HDF5 REQUIRED COMPONENTS C CXX)

if (HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    target_link_libraries (write ${HDF5_LIBRARIES})
endif (HDF5_FOUND)

if (CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mfpmath=sse -Wall -Wextra -Wno-sign-compare -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls -Wshadow -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wsuggest-final-types -Wsuggest-final-methods -Wsizeof-pointer-memaccess -Wuninitialized -Winit-self")
endif ()

# find_package(ZLIB REQUIRED)
# if (ZLIB_FOUND)
#     include_directories(${ZLIB_INCLUDE_DIRS})
#     target_link_libraries (write ${ZLIB_LIBRARIES})
# endif (ZLIB_FOUND)

# find_library(BLOSC_LIB blosc)
# target_link_libraries(write ${BLOSC_LIB})
