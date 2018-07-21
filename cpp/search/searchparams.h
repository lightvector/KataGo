#ifndef SEARCH_PARAMS_H
#define SEARCH_PARAMS_H

#include "../core/global.h"

struct SearchParams {
  //Utility function parameters
  double winLossUtilityFactor;    //Scaling for [-1,1] value for winning/losing
  double scoreUtilityFactor;      //Scaling for the [-1,1] value for having many/fewer points.
  double noResultUtilityForWhite; //Utility of having a no-result game (simple ko rules or nonterminating territory encore)
  double drawUtilityForWhite;     //Utility of having a jigo

  //Search tree exploration parameters
  double cpuctExploration;  //Constant factor on exploration, should also scale up linearly with magnitude of utility
  double fpuReductionMax;   //Max amount to reduce fpu value for unexplore children

  //Root noise parameters
  bool rootNoiseEnabled;
  double rootDirichletNoiseTotalConcentration; //Same as alpha * board size, to match alphazero this might be 0.03 * 361, total number of balls in the urn
  double rootDirichletNoiseWeight; //Policy at root is this weight * noise + (1 - this weight) * nn policy

  //Randomization. Note - this controls a few things in the search, but a lot of the randomness actually comes from
  //random symmetries of the neural net evaluations, which is separate from the search's rng, see nneval.h
  string randSeed;
  double chosenMoveTemperature; //Make move roughly proportional to visit count ** (1/chosenMoveTemperature)
  double chosenMoveSubtract; //Try to subtract this many playouts from every move prior to applying temperature

  //Misc
  uint32_t mutexPoolSize; //Size of mutex pool for synchronizing access to all search nodes
  int32_t numVirtualLossesPerThread; //Number of virtual losses for one thread to add

  //Asyncbot
  int numThreads; //Number of threads, used in asyncbot layer which spawns threads
  uint64_t maxVisits; //Max number of playouts from the root to think for, counting earlier playouts from tree reuse
  uint64_t maxPlayouts; //Max number of playouts from the root to think for, not counting earlier playouts from tree reuse
  double maxTime; //Max number of seconds to think for if not pondering

  SearchParams();
  ~SearchParams();
};

#endif
