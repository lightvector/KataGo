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
  Rand rand;

  int64_t numGamesStartedSoFar;
  int64_t numGamesTotal;
  int64_t logGamesEvery;

  std::mutex getMatchupMutex;

  pair<int,int> getMatchupPair();
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

  FancyModes();
  ~FancyModes();
};

//Functions to run a single game
namespace Play {
  FinishedGameData* runGame(
    const Board& initialBoard, Player pla, const BoardHistory& initialHist, int numExtraBlack,
    MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
    const string& searchRandSeed,
    bool doEndGameIfAllPassAlive, bool clearBotAfterSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, vector<std::atomic<bool>*>& stopConditions,
    FancyModes fancyModes, bool recordFullData, int dataPosLen,
    Rand& gameRand
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

  bool runGame(
    MatchPairer* matchPairer, Logger& logger,
    int dataPosLen,
    ThreadSafeQueue<FinishedGameData*>* finishedGameQueue,
    //reportGame should not hold on to a reference to the finished game data.
    std::function<void(const FinishedGameData&)>* reportGame,
    vector<std::atomic<bool>*>& stopConditions
  );

};


#endif
