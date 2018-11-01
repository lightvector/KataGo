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

//Object choosing random initial rules and board sizes for games. Threadsafe.
class GameInitializer {
 public:
  GameInitializer(ConfigParser& cfg);
  GameInitializer(ConfigParser& cfg, const SearchParams& baseParams);
  ~GameInitializer();

  GameInitializer(const GameInitializer&) = delete;
  GameInitializer& operator=(const GameInitializer&) = delete;

  //Initialize everything for a new game with random rules
  //Also, mutates params to have new rules, but does NOT set all its settings, user
  void createGame(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack);
  void createGame(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack, SearchParams& params);


 private:
  void initShared(ConfigParser& cfg);
  void createGameSharedUnsynchronized(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack);


  std::mutex createGameMutex;
  Rand rand;

  bool hasParams;

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
  double drawStdev;

  SearchParams baseParams;
};


//Object for generating and servering evenly distributed pairings between different bots. Threadsafe.
class MatchPairer {
 public:
  MatchPairer(ConfigParser& cfg, bool forSelfPlay);
  ~MatchPairer();

  MatchPairer(const MatchPairer&) = delete;
  MatchPairer& operator=(const MatchPairer&) = delete;

  //Get next matchup and log stuff
  bool getMatchup(
    int64_t& gameIdx, int& botIdxB, int& botIdxW, Logger& logger,
    const NNEvaluator* nnEvalToLog, const vector<NNEvaluator*>* nnEvalsToLog
  );
  //Convenience usage for self play where there is only one bot.
  bool getMatchup(
    int64_t& gameIdx, Logger& logger,
    const NNEvaluator* nnEvalToLog, const vector<NNEvaluator*>* nnEvalsToLog
  );

 private:
  int numBots;
  vector<int> secondaryBots;
  vector<pair<int,int>> nextMatchups;
  Rand rand;

  int64_t numGamesStartedSoFar;
  int64_t numGamesTotal;
  int64_t logGamesEvery;

  std::mutex getMatchupMutex;

  pair<int,int> getMatchupPair();
};


//Functions to run a single game
namespace Play {
  void runGame(
    Board& board, Player pla, BoardHistory& hist, int numExtraBlack, Search* botB, Search* botW,
    bool doEndGameIfAllPassAlive, bool clearBotAfterSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, std::atomic<bool>& stopSignalReceived,
    bool fancyModes,
    FinishedGameData* gameData, Rand* gameRand
  );

}


//Class for running a game and enqueueing the result as training data.
//Wraps together most of the neural-net-independent parameters to spawn and run a full game.
class GameRunner {
  bool logSearchInfo;
  bool logMoves;
  int maxMovesPerGame;
  string searchRandSeedBase;
  MatchPairer* matchPairer;
  GameInitializer* gameInit;

public:
  GameRunner(ConfigParser& cfg, const string& searchRandSeedBase);
  ~GameRunner();

  bool runGameAndEnqueueData(
    NNEvaluator* nnEval, Logger& logger,
    int dataPosLen, ThreadSafeQueue<FinishedGameData*>& finishedGameQueue,
    std::atomic<bool>& stopSignalReceived
  );
  bool runGameAndEnqueueData(
    NNEvaluator* nnEvalB, NNEvaluator* nnEvalW, Logger& logger,
    int dataPosLen, ThreadSafeQueue<FinishedGameData*>& finishedGameQueue,
    std::atomic<bool>& stopSignalReceived
  );

};


#endif
