#ifndef SEARCH_PARAMS_H
#define SEARCH_PARAMS_H

#include "../core/global.h"

struct SearchParams {
  //Utility function parameters
  double winLossUtilityFactor;    //Scaling for [-1,1] value for winning/losing
  double scoreUtilityFactor;      //Scaling for the [-1,1] value for having many/fewer points.
  double noResultUtilityForWhite; //Utility of having a no-result game (simple ko rules or nonterminating territory encore)

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

  //Thread-related parameters
  uint32_t mutexPoolSize; //Size of mutex pool for synchronizing access to all search nodes
  int numThreads; //Number of threads, used in asyncbot layer which spawns threads

  SearchParams();
  ~SearchParams();
};

#endif
