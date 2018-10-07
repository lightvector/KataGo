#ifndef PLAY_H
#define PLAY_H

#include "../core/global.h"
#include "../core/multithread.h"
#include "../core/rand.h"
#include "../core/config_parser.h"
#include "../game/board.h"
#include "../game/boardhistory.h"

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


//Object for generating and servering evenly distributed pairings between different bots.
//NOT threadsafe
class MatchPairer {
 public:
  MatchPairer(int numBots, const vector<int>& secondaryBots);
  ~MatchPairer();

  MatchPairer(const MatchPairer&) = delete;
  MatchPairer& operator=(const MatchPairer&) = delete;

  pair<int,int> getMatchup();

 private:
  int numBots;
  vector<int> secondaryBots;
  vector<pair<int,int>> nextMatchups;
  Rand rand;
};



namespace Play {
  void runGame(
    Board& board, Player pla, BoardHistory& hist, int numExtraBlack, AsyncBot* botB, AsyncBot* botW,
    bool doEndGameIfAllPassAlive, bool clearBotAfterSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, std::atomic<bool>& stopSignalReceived
  );

}



#endif
