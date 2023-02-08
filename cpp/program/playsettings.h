#ifndef PROGRAM_PLAYSETTINGS_H_
#define PROGRAM_PLAYSETTINGS_H_

#include "../core/config_parser.h"

struct PlaySettings {
  //Play a bunch of mostly policy-distributed moves at the start to initialize a game.
  bool initGamesWithPolicy;
  double policyInitAreaProp; //Avg number of moves is this * board area
  double startPosesPolicyInitAreaProp; //Avg number of moves when using a starting position from sgf
  double compensateAfterPolicyInitProb; //Chance to adjust komi to cancel the effect of imbalanced init
  //Occasionally try some alternative moves and search the responses to them.
  double sidePositionProb;

  //Temperature to use for placing handicap stones and for initializing the board position
  double policyInitAreaTemperature;
  double handicapTemperature;

  //Use this many visits in a short search to estimate the score, for adjusting komi
  int compensateKomiVisits;
  //When NOT compensating komi, set the fair komi for white playing first rather than black playing first.
  double flipKomiProbWhenNoCompensate;
  
  //Use this many visits in a short search to estimate the score, for computing lead
  int estimateLeadVisits;
  //On each train position, estimate the lead in points with this probability
  double estimateLeadProb;

  //Occasionally fork an entire new game to try out an experimental move in the opening
  double earlyForkGameProb; //Expected number of early forked games per game
  double earlyForkGameExpectedMoveProp; //Fork on average within the first board area * this prop moves
  double forkGameProb; //Expected number of forked games per game
  int forkGameMinChoices; //Fork between the favorite of this many random legal moves, at minimum
  int earlyForkGameMaxChoices; //Fork between the favorite of this many random legal moves, at maximum
  int forkGameMaxChoices; //Fork between the favorite of this many random legal moves, at maximum

  //Hack to make learning of seki easier - fork positions with different rules when we have sekis
  double sekiForkHackProb;
  //Hack to improve learning of very weird komi and very lopsided positions
  bool fancyKomiVarying;

  //With this probability, use only this many visits for a move, and record it with only this weight
  double cheapSearchProb;
  int cheapSearchVisits;
  float cheapSearchTargetWeight;

  //Attenuate the number of visits used in positions where one player or the other is extremely winning
  bool reduceVisits;
  double reduceVisitsThreshold; //When mcts value is more extreme than this
  int reduceVisitsThresholdLookback; //Value must be more extreme over the last this many turns
  int reducedVisitsMin; //Number of visits at the most extreme winrate
  float reducedVisitsWeight; //Amount of weight to put on the training sample at minimum visits winrate.

  //Probabilistically favor samples that had high policy surprise (kl divergence).
  double policySurpriseDataWeight;
  //Probabilistically favor samples that had high winLossValue surprise (kl divergence).
  double valueSurpriseDataWeight;
  //Scale frequency weights for writing data by this
  double scaleDataWeight;

  //Record positions from within the search tree that had at least this many visits, recording only with this weight.
  bool recordTreePositions;
  int recordTreeThreshold;
  float recordTreeTargetWeight;

  //Don't stochastically integerify target weights
  bool noResolveTargetWeights;

  //Resign conditions
  bool allowResignation;
  double resignThreshold; //Require that mcts win value is less than this
  double resignConsecTurns; //Require that both players have agreed on it continuously for this many turns

  //Enable full data recording and a variety of other minor tweaks applying only for self-play training.
  bool forSelfPlay;

  //Asymmetric playouts training
  double handicapAsymmetricPlayoutProb; //Probability of asymmetric playouts on handicap games
  double normalAsymmetricPlayoutProb; //Probability of asymmetric playouts on normal games
  double maxAsymmetricRatio;
  double minAsymmetricCompensateKomiProb; //Minimum probability to make game fair if asymmetric (other probs will also override)

  //Record time taken per move
  bool recordTimePerMove;

  PlaySettings();
  ~PlaySettings();

  static PlaySettings loadForMatch(ConfigParser& cfg);
  static PlaySettings loadForGatekeeper(ConfigParser& cfg);
  static PlaySettings loadForSelfplay(ConfigParser& cfg, bool isDistributed);
};

#endif // PROGRAM_PLAYSETTINGS_H_
