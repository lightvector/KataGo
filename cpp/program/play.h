#ifndef PLAY_H
#define PLAY_H

#include "../core/global.h"
#include "../core/multithread.h"
#include "../core/rand.h"
#include "../core/config_parser.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../dataio/trainingwrite.h"

//Object choosing random initial rules and board sizes for games. Threadsafe.
class GameInitializer {
 public:
  GameInitializer(ConfigParser& cfg);
  ~GameInitializer();

  GameInitializer(const GameInitializer&) = delete;
  GameInitializer& operator=(const GameInitializer&) = delete;

  void createGame(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack);


 private:
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
  double komiBigStdev;
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
    FinishedGameData* gameData, Rand* gameRand
  );

}

#endif
