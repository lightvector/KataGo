#ifndef PLAY_H
#define PLAY_H

#include "../core/global.h"
#include "../core/multithread.h"
#include "../core/rand.h"
#include "../core/config_parser.h"
#include "../core/threadsafequeue.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../dataio/trainingwrite.h"
#include "../search/searchparams.h"
#include "../search/search.h"

//Object choosing random initial rules and board sizes for games. Threadsafe.
class GameInitializer {
 public:
  GameInitializer(ConfigParser& cfg);
  ~GameInitializer();

  GameInitializer(const GameInitializer&) = delete;
  GameInitializer& operator=(const GameInitializer&) = delete;

  //Initialize everything for a new game with random rules
  //Also, mutates params to randomize appropriate things like utilities, but does NOT fill in all the settings.
  //User should make sure the initial params provided makes sense as a mean or baseline.
  void createGame(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack, SearchParams& params);

  //A version that doesn't randomize params
  void createGame(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack);

 private:
  void initShared(ConfigParser& cfg);
  void createGameSharedUnsynchronized(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack);

  std::mutex createGameMutex;
  Rand rand;

  vector<string> allowedKoRuleStrs;
  vector<string> allowedScoringRuleStrs;
  vector<bool> allowedMultiStoneSuicideLegals;

  vector<int> allowedKoRules;
  vector<int> allowedScoringRules;

  vector<int> allowedBSizes;
  vector<double> allowedBSizeRelProbs;

  float komiMean;
  float komiStdev;
  double komiAllowIntegerProb;
  double handicapProb;
  float handicapStoneValue;
  double komiBigStdevProb;
  float komiBigStdev;

  double noResultStdev;
  double drawRandRadius;
};


//Object for generating and servering evenly distributed pairings between different bots. Threadsafe.
class MatchPairer {
 public:
  //Holds pointers to the various nnEvals, but does NOT take ownership for freeing them.
  MatchPairer(
    ConfigParser& cfg,
    int numBots,
    const vector<string>& botNames,
    const vector<NNEvaluator*>& nnEvals,
    const vector<SearchParams>& baseParamss,
    bool forSelfPlay,
    bool forGateKeeper
  );

  ~MatchPairer();

  struct BotSpec {
    int botIdx;
    string botName;
    NNEvaluator* nnEval;
    SearchParams baseParams;
  };

  MatchPairer(const MatchPairer&) = delete;
  MatchPairer& operator=(const MatchPairer&) = delete;

  //Get the total number of games that the matchpairer will generate
  int getNumGamesTotalToGenerate() const;

  //Get next matchup and log stuff
  bool getMatchup(
    int64_t& gameIdx, BotSpec& botSpecB, BotSpec& botSpecW, Logger& logger
  );

 private:
  int numBots;
  vector<string> botNames;
  vector<NNEvaluator*> nnEvals;
  vector<SearchParams> baseParamss;

  vector<int> secondaryBots;
  vector<pair<int,int>> nextMatchups;
  vector<pair<int,int>> nextMatchupsBuf;
  Rand rand;

  int matchRepFactor;
  int repsOfLastMatchup;
  
  int64_t numGamesStartedSoFar;
  int64_t numGamesTotal;
  int64_t logGamesEvery;

  std::mutex getMatchupMutex;

  pair<int,int> getMatchupPairUnsynchronized();
};

struct FancyModes {
  //Play a bunch of mostly policy-distributed moves at the start to initialize a game.
  bool initGamesWithPolicy;
  //Occasionally try some alternative moves and search the responses to them.
  double forkSidePositionProb;

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
  
  //Record positions from within the search tree that had at least this many visits, recording only with this weight.
  bool recordTreePositions;
  int recordTreeThreshold;
  float recordTreeTargetWeight;

  //Resign conditions
  bool allowResignation;
  double resignThreshold; //Require that mcts win value is less than this
  double resignConsecTurns; //Require that both players have agreed on it continuously for this many turns

  FancyModes();
  ~FancyModes();
};

//Functions to run a single game or other things
namespace Play {
  //Use the given bot to play free handicap stones, modifying the board and hist in the process and setting the bot's position to it.
  void playExtraBlack(Search* bot, Logger& logger, int numExtraBlack, Board& board, BoardHistory& hist, double temperature);

  FinishedGameData* runGame(
    const Board& initialBoard, Player pla, const BoardHistory& initialHist, int numExtraBlack,
    const MatchPairer::BotSpec& botSpecB, const MatchPairer::BotSpec& botSpecW,
    const string& searchRandSeed,
    bool doEndGameIfAllPassAlive, bool clearBotAfterSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, vector<std::atomic<bool>*>& stopConditions,
    FancyModes fancyModes, bool recordFullData, int dataPosLen,
    Rand& gameRand,
    std::function<NNEvaluator*()>* checkForNewNNEval
  );

}


//Class for running a game and enqueueing the result as training data.
//Wraps together most of the neural-net-independent parameters to spawn and run a full game.
class GameRunner {
  bool logSearchInfo;
  bool logMoves;
  bool forSelfPlay;
  int maxMovesPerGame;
  bool clearBotAfterSearch;
  string searchRandSeedBase;
  FancyModes fancyModes;
  GameInitializer* gameInit;

public:
  GameRunner(ConfigParser& cfg, const string& searchRandSeedBase, bool forSelfPlay, FancyModes fancyModes);
  ~GameRunner();

  //Will return NULL if stopped before the game completes. The caller is responsible for freeing the data
  //if it isn't NULL.
  FinishedGameData* runGame(
    int64_t gameIdx,
    const MatchPairer::BotSpec& botSpecB,
    const MatchPairer::BotSpec& botSpecW,
    Logger& logger,
    int dataPosLen,
    vector<std::atomic<bool>*>& stopConditions,
    std::function<NNEvaluator*()>* checkForNewNNEval
  );

};


#endif
