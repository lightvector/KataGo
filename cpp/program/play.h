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

struct InitialPosition {
  Board board;
  BoardHistory hist;
  Player pla;

  InitialPosition();
  InitialPosition(const Board& board, const BoardHistory& hist, Player pla);
  ~InitialPosition();
};

STRUCT_NAMED_TRIPLE(int, extraBlack, float, komi, float, komiBase, ExtraBlackAndKomi);

//Object choosing random initial rules and board sizes for games. Threadsafe.
class GameInitializer {
 public:
  GameInitializer(ConfigParser& cfg);
  ~GameInitializer();

  GameInitializer(const GameInitializer&) = delete;
  GameInitializer& operator=(const GameInitializer&) = delete;

  //Initialize everything for a new game with random rules, unless initialPosition is provided, in which case it uses
  //those rules (possibly with noise to the komi given in that position)
  //Also, mutates params to randomize appropriate things like utilities, but does NOT fill in all the settings.
  //User should make sure the initial params provided makes sense as a mean or baseline.
  //Does NOT place handicap stones, users of this function need to place them manually
  void createGame(Board& board, Player& pla, BoardHistory& hist, ExtraBlackAndKomi& extraBlackAndKomi, SearchParams& params, const InitialPosition* initialPosition);

  //A version that doesn't randomize params
  void createGame(Board& board, Player& pla, BoardHistory& hist, ExtraBlackAndKomi& extraBlackAndKomi, const InitialPosition* initialPosition);

 private:
  void initShared(ConfigParser& cfg);
  void createGameSharedUnsynchronized(Board& board, Player& pla, BoardHistory& hist, ExtraBlackAndKomi& extraBlackAndKomi, const InitialPosition* initialPosition);

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
  MatchPairer(
    ConfigParser& cfg,
    int numBots,
    const vector<string>& botNames,
    const vector<NNEvaluator*>& nnEvals,
    const vector<SearchParams>& baseParamss,
    bool forSelfPlay,
    bool forGateKeeper,
    const vector<bool>& excludeBot
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

  vector<bool> excludeBot;
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

  //In handicap games and when forking a whole game - with this probability do NOT adjust the komi to be fair.
  double noCompensateKomiProb;
  //Use this many visits in a short search to estimate the score, for adjusting komi
  int compensateKomiVisits;
  
  //Occasionally fork an entire new game to try out an experimental move in the opening
  double earlyForkGameProb; //Expected number of forked games per game
  double earlyForkGameExpectedMoveProp; //Fork on average within the first board area * this prop moves
  int earlyForkGameMinChoices; //Fork between the favorite of this many random legal moves, at minimum
  int earlyForkGameMaxChoices; //Fork between the favorite of this many random legal moves, at maximum
  
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
  void playExtraBlack(
    Search* bot,
    Logger& logger,
    ExtraBlackAndKomi extraBlackAndKomi,
    Board& board,
    BoardHistory& hist,
    double temperature,
    Rand& gameRand,
    bool adjustKomi,
    int numVisitsForKomi
  );

  //In the case where checkForNewNNEval is provided, will MODIFY the provided botSpecs with any new nneval!
  FinishedGameData* runGame(
    const Board& startBoard, Player pla, const BoardHistory& startHist, ExtraBlackAndKomi extraBlackAndKomi,
    MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
    const string& searchRandSeed,
    bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, vector<std::atomic<bool>*>& stopConditions,
    FancyModes fancyModes, bool recordFullData, int dataXLen, int dataYLen,
    bool allowPolicyInit,
    Rand& gameRand,
    std::function<NNEvaluator*()>* checkForNewNNEval
  );

  //In the case where checkForNewNNEval is provided, will MODIFY the provided botSpecs with any new nneval!
  FinishedGameData* runGame(
    const Board& startBoard, Player pla, const BoardHistory& startHist, ExtraBlackAndKomi extraBlackAndKomi,
    MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
    Search* botB, Search* botW,
    bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, vector<std::atomic<bool>*>& stopConditions,
    FancyModes fancyModes, bool recordFullData, int dataXLen, int dataYLen,
    bool allowPolicyInit,
    Rand& gameRand,
    std::function<NNEvaluator*()>* checkForNewNNEval
  );
  
  void maybeForkGame(
    const FinishedGameData* finishedGameData,
    const InitialPosition** nextInitialPosition,
    const FancyModes& fancyModes,
    Rand& gameRand,
    Search* bot,
    Logger& logger
  );

  Loc chooseRandomPolicyMove(
    const NNOutput* nnOutput,
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    Rand& gameRand,
    double temperature,
    bool allowPass,
    Loc banMove
  );

  void adjustKomiToEven(
    Search* bot,
    const Board& board,
    BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    Logger& logger
  );

  double getSearchFactor(
    double searchFactorWhenWinningThreshold,
    double searchFactorWhenWinning,
    const SearchParams& params,
    const vector<double>& recentWinLossValues,
    Player pla
  );
}


//Class for running a game and enqueueing the result as training data.
//Wraps together most of the neural-net-independent parameters to spawn and run a full game.
class GameRunner {
  bool logSearchInfo;
  bool logMoves;
  bool forSelfPlay;
  int maxMovesPerGame;
  bool clearBotBeforeSearch;
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
    const InitialPosition* initialPosition,
    const InitialPosition** nextInitialPosition,
    Logger& logger,
    int dataXLen,
    int dataYLen,
    vector<std::atomic<bool>*>& stopConditions,
    std::function<NNEvaluator*()>* checkForNewNNEval
  );

};


#endif
