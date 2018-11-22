#ifndef SEARCH_PARAMS_H
#define SEARCH_PARAMS_H

#include "../core/global.h"

struct SearchParams {
  //Utility function parameters
  double winLossUtilityFactor;    //Scaling for [-1,1] value for winning/losing
  double scoreUtilityFactor;      //Scaling for the roughly [-1,1] value for having many/fewer points.
  double noResultUtilityForWhite; //Utility of having a no-result game (simple ko rules or nonterminating territory encore)
  double drawEquivalentWinsForWhite; //Consider a draw to be this many wins and one minus this many losses.

  //Search tree exploration parameters
  double cpuctExploration;  //Constant factor on exploration, should also scale up linearly with magnitude of utility
  double fpuReductionMax;   //Max amount to reduce fpu value for unexplore children
  bool fpuUseParentAverage; //Use parent average value for fpu rather than parent nn value.
  double valueWeightExponent; //Amount to apply a downweighting of children with very bad values relative to good ones
  double visitsExponent; //Power with which visits should raise the value weight on a child

  bool scaleParentWeight; //Also scale parent weight when applying valueWeightExponent?

  //Root noise parameters
  bool rootNoiseEnabled;
  double rootDirichletNoiseTotalConcentration; //Same as alpha * board size, to match alphazero this might be 0.03 * 361, total number of balls in the urn
  double rootDirichletNoiseWeight; //Policy at root is this weight * noise + (1 - this weight) * nn policy

  double chosenMoveTemperature; //Make move roughly proportional to visit count ** (1/chosenMoveTemperature)
  double chosenMoveTemperatureEarly; //Temperature at start of game
  double chosenMoveTemperatureHalflife; //Halflife of decay from early temperature to temperature for the rest of the game, scales for board sizes other than 19.
  double chosenMoveSubtract; //Try to subtract this many visits from every move prior to applying temperature
  double chosenMovePrune; //Outright prune moves that have fewer than this many visits

  //Mild behavior hackery
  double rootEndingBonusPoints; //Extra bonus (or penalty) to encourage good passing behavior at the end of the game.
  bool rootPruneUselessSuicides; //Prune moves that are entirely useless suicide moves that prolong the game.

  //Threading-related
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
