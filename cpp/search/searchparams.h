#ifndef SEARCH_SEARCHPARAMS_H_
#define SEARCH_SEARCHPARAMS_H_

#include "../core/global.h"

struct SearchParams {
  //Utility function parameters
  double winLossUtilityFactor;     //Scaling for [-1,1] value for winning/losing
  double staticScoreUtilityFactor; //Scaling for a [-1,1] "scoreValue" for having more/fewer points, centered at 0.
  double dynamicScoreUtilityFactor; //Scaling for a [-1,1] "scoreValue" for having more/fewer points, centered at recent estimated expected score.
  double noResultUtilityForWhite; //Utility of having a no-result game (simple ko rules or nonterminating territory encore)
  double drawEquivalentWinsForWhite; //Consider a draw to be this many wins and one minus this many losses.

  //Search tree exploration parameters
  double cpuctExploration;  //Constant factor on exploration, should also scale up linearly with magnitude of utility
  double fpuReductionMax;   //Max amount to reduce fpu value for unexplore children
  double fpuLossProp; //Scale fpu this proportion of the way towards assuming a move is a loss.
  bool fpuUseParentAverage; //Use parent average value for fpu rather than parent nn value.
  double valueWeightExponent; //Amount to apply a downweighting of children with very bad values relative to good ones
  double visitsExponent; //Power with which visits should raise the value weight on a child

  bool scaleParentWeight; //Also scale parent weight when applying valueWeightExponent?

  //Root parameters
  bool rootNoiseEnabled;
  double rootDirichletNoiseTotalConcentration; //Same as alpha * board size, to match alphazero this might be 0.03 * 361, total number of balls in the urn
  double rootDirichletNoiseWeight; //Policy at root is this weight * noise + (1 - this weight) * nn policy

  double rootPolicyTemperature; //At the root node, scale policy probs by this power
  double rootFpuReductionMax; //Same as fpuReductionMax, but at root
  double rootFpuLossProp; //Same as fpuLossProp, but at root

  //We use the min of these two together, and also excess visits get pruned if the value turns out bad.
  double rootDesiredPerChildVisitsCoeff; //Funnel sqrt(this * policy prob * total visits) down any given child that receives any visits at all at the root

  //Parameters for choosing the move to play
  double chosenMoveTemperature; //Make move roughly proportional to visit count ** (1/chosenMoveTemperature)
  double chosenMoveTemperatureEarly; //Temperature at start of game
  double chosenMoveTemperatureHalflife; //Halflife of decay from early temperature to temperature for the rest of the game, scales for board sizes other than 19.
  double chosenMoveSubtract; //Try to subtract this many visits from every move prior to applying temperature
  double chosenMovePrune; //Outright prune moves that have fewer than this many visits

  bool useLcbForSelection; //Using LCB for move selection?
  double lcbStdevs; //How many stdevs a move needs to be better than another for LCB selection
  double minVisitPropForLCB; //Only use LCB override when a move has this proportion of visits as the top move
  
  //Mild behavior hackery
  double rootEndingBonusPoints; //Extra bonus (or penalty) to encourage good passing behavior at the end of the game.
  bool rootPruneUselessMoves; //Prune moves that are entirely useless moves that prolong the game.

  //Threading-related
  uint32_t mutexPoolSize; //Size of mutex pool for synchronizing access to all search nodes
  int32_t numVirtualLossesPerThread; //Number of virtual losses for one thread to add

  //Asyncbot
  int numThreads; //Number of threads
  int64_t maxVisits; //Max number of playouts from the root to think for, counting earlier playouts from tree reuse
  int64_t maxPlayouts; //Max number of playouts from the root to think for, not counting earlier playouts from tree reuse
  double maxTime; //Max number of seconds to think for

  //Same caps but when pondering
  int64_t maxVisitsPondering; 
  int64_t maxPlayoutsPondering; 
  double maxTimePondering;

  //Amount of time to reserve for lag when using a time control
  double lagBuffer;

  //Human-friendliness
  double searchFactorAfterOnePass; //Multiply playouts and visits and time by this much after a pass by the opponent
  double searchFactorAfterTwoPass; //Multiply playouts and visits and time by this after two passes by the opponent

  SearchParams();
  ~SearchParams();
};

#endif  // SEARCH_SEARCHPARAMS_H_
