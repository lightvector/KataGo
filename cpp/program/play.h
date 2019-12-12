#ifndef PROGRAM_PLAY_H_
#define PROGRAM_PLAY_H_

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../core/multithread.h"
#include "../core/rand.h"
#include "../core/threadsafequeue.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/sgf.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../search/search.h"
#include "../search/searchparams.h"

struct InitialPosition {
  Board board;
  BoardHistory hist;
  Player pla;

  InitialPosition();
  InitialPosition(const Board& board, const BoardHistory& hist, Player pla);
  ~InitialPosition();
};

//Holds various initial positions that we may start from rather than a whole new game
struct ForkData {
  std::mutex mutex;
  std::vector<const InitialPosition*> forks;
  std::vector<const InitialPosition*> sekiForks;
  ~ForkData();

  void add(const InitialPosition* pos);
  const InitialPosition* get(Rand& rand);

  void addSeki(const InitialPosition* pos, Rand& rand);
  const InitialPosition* getSeki(Rand& rand);
};

struct ExtraBlackAndKomi {
  int extraBlack = 0;
  float komi = 7.5;
  float komiBase = 7.5;
  bool makeGameFair = false;
  bool makeGameFairForEmptyBoard = false;
};

struct OtherGameProperties {
  bool isSgfPos = false;
  bool allowPolicyInit = true;

  //Note: these two behave slightly differently than the ones in searchParams - as properties for the whole
  //game, they make the playouts *actually* vary instead of only making the neural net think they do.
  double playoutDoublingAdvantage = 0.0;
  Player playoutDoublingAdvantagePla = C_EMPTY;
};

struct FancyModes;

//Object choosing random initial rules and board sizes for games. Threadsafe.
class GameInitializer {
 public:
  GameInitializer(ConfigParser& cfg, Logger& logger);
  ~GameInitializer();

  GameInitializer(const GameInitializer&) = delete;
  GameInitializer& operator=(const GameInitializer&) = delete;

  //Initialize everything for a new game with random rules, unless initialPosition is provided, in which case it uses
  //those rules (possibly with noise to the komi given in that position)
  //Also, mutates params to randomize appropriate things like utilities, but does NOT fill in all the settings.
  //User should make sure the initial params provided makes sense as a mean or baseline.
  //Does NOT place handicap stones, users of this function need to place them manually
  void createGame(
    Board& board, Player& pla, BoardHistory& hist,
    ExtraBlackAndKomi& extraBlackAndKomi,
    SearchParams& params,
    const InitialPosition* initialPosition,
    const FancyModes& fancyModes,
    OtherGameProperties& otherGameProps
  );

  //A version that doesn't randomize params
  void createGame(
    Board& board, Player& pla, BoardHistory& hist,
    ExtraBlackAndKomi& extraBlackAndKomi,
    const InitialPosition* initialPosition,
    const FancyModes& fancyModes,
    OtherGameProperties& otherGameProps
  );

  Rules randomizeScoringAndTaxRules(Rules rules, Rand& randToUse) const;

 private:
  void initShared(ConfigParser& cfg, Logger& logger);
  void createGameSharedUnsynchronized(
    Board& board, Player& pla, BoardHistory& hist,
    ExtraBlackAndKomi& extraBlackAndKomi,
    const InitialPosition* initialPosition,
    const FancyModes& fancyModes,
    OtherGameProperties& otherGameProps
  );

  std::mutex createGameMutex;
  Rand rand;

  std::vector<std::string> allowedKoRuleStrs;
  std::vector<std::string> allowedScoringRuleStrs;
  std::vector<std::string> allowedTaxRuleStrs;
  std::vector<bool> allowedMultiStoneSuicideLegals;
  std::vector<bool> allowedButtons;

  std::vector<int> allowedKoRules;
  std::vector<int> allowedScoringRules;
  std::vector<int> allowedTaxRules;

  std::vector<int> allowedBSizes;
  std::vector<double> allowedBSizeRelProbs;

  double allowRectangleProb;

  float komiMean;
  float komiStdev;
  double komiAllowIntegerProb;
  double handicapProb;
  double handicapCompensateKomiProb;
  double forkCompensateKomiProb;
  double komiBigStdevProb;
  float komiBigStdev;
  bool komiAuto;

  double noResultStdev;
  double drawRandRadius;

  std::vector<Sgf::PositionSample> startPoses;
  std::vector<double> startPosCumProbs;
  double startPosesProb;
};


//Object for generating and servering evenly distributed pairings between different bots. Threadsafe.
class MatchPairer {
 public:
  //Holds pointers to the various nnEvals, but does NOT take ownership for freeing them.
  MatchPairer(
    ConfigParser& cfg,
    int numBots,
    const std::vector<std::string>& botNames,
    const std::vector<NNEvaluator*>& nnEvals,
    const std::vector<SearchParams>& baseParamss,
    bool forSelfPlay,
    bool forGateKeeper
  );
  MatchPairer(
    ConfigParser& cfg,
    int numBots,
    const std::vector<std::string>& botNames,
    const std::vector<NNEvaluator*>& nnEvals,
    const std::vector<SearchParams>& baseParamss,
    bool forSelfPlay,
    bool forGateKeeper,
    const std::vector<bool>& excludeBot
  );

  ~MatchPairer();

  struct BotSpec {
    int botIdx;
    std::string botName;
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
  std::vector<std::string> botNames;
  std::vector<NNEvaluator*> nnEvals;
  std::vector<SearchParams> baseParamss;

  std::vector<bool> excludeBot;
  std::vector<int> secondaryBots;
  std::vector<std::pair<int,int>> nextMatchups;
  std::vector<std::pair<int,int>> nextMatchupsBuf;
  Rand rand;

  int matchRepFactor;
  int repsOfLastMatchup;

  int64_t numGamesStartedSoFar;
  int64_t numGamesTotal;
  int64_t logGamesEvery;

  std::mutex getMatchupMutex;

  std::pair<int,int> getMatchupPairUnsynchronized();
};

struct FancyModes {
  //Play a bunch of mostly policy-distributed moves at the start to initialize a game.
  bool initGamesWithPolicy;
  //Occasionally try some alternative moves and search the responses to them.
  double forkSidePositionProb;

  //Use this many visits in a short search to estimate the score, for adjusting komi
  int compensateKomiVisits;

  //Occasionally fork an entire new game to try out an experimental move in the opening
  double earlyForkGameProb; //Expected number of forked games per game
  double earlyForkGameExpectedMoveProp; //Fork on average within the first board area * this prop moves
  int earlyForkGameMinChoices; //Fork between the favorite of this many random legal moves, at minimum
  int earlyForkGameMaxChoices; //Fork between the favorite of this many random legal moves, at maximum

  //Hack to make learning of seki easier - fork positions with different rules when we have sekis
  bool sekiForkHack;

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

  //Record positions from within the search tree that had at least this many visits, recording only with this weight.
  bool recordTreePositions;
  int recordTreeThreshold;
  float recordTreeTargetWeight;

  //Resign conditions
  bool allowResignation;
  double resignThreshold; //Require that mcts win value is less than this
  double resignConsecTurns; //Require that both players have agreed on it continuously for this many turns

  //Enable full data recording and a variety of other minor tweaks applying only for self-play training.
  bool forSelfPlay;
  int dataXLen; //When self-play data recording, the width/height of the tensor
  int dataYLen; //When self-play data recording, the width/height of the tensor

  //Asymmetric playouts training
  double handicapAsymmetricPlayoutProb; //Probability of asymmetric playouts on handicap games
  double normalAsymmetricPlayoutProb; //Probability of asymmetric playouts on normal games
  double maxAsymmetricRatio;
  double minAsymmetricCompensateKomiProb; //Minimum probability to make game fair if asymmetric (other probs will also override)

  FancyModes();
  ~FancyModes();
};

//Functions to run a single game or other things
namespace Play {
  //Use the given bot to play free handicap stones, modifying the board and hist in the process and setting the bot's position to it.
  void playExtraBlack(
    Search* bot,
    int numExtraBlack,
    Board& board,
    BoardHistory& hist,
    double temperature,
    Rand& gameRand
  );

  //In the case where checkForNewNNEval is provided, will MODIFY the provided botSpecs with any new nneval!
  FinishedGameData* runGame(
    const Board& startBoard, Player pla, const BoardHistory& startHist, ExtraBlackAndKomi extraBlackAndKomi,
    MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
    const std::string& searchRandSeed,
    bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, std::vector<std::atomic<bool>*>& stopConditions,
    const FancyModes& fancyModes, const OtherGameProperties& otherGameProps,
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
    int maxMovesPerGame, std::vector<std::atomic<bool>*>& stopConditions,
    const FancyModes& fancyModes, const OtherGameProperties& otherGameProps,
    Rand& gameRand,
    std::function<NNEvaluator*()>* checkForNewNNEval
  );

  void maybeForkGame(
    const FinishedGameData* finishedGameData,
    ForkData* forkData,
    const FancyModes& fancyModes,
    Rand& gameRand,
    Search* bot
  );

  void maybeSekiForkGame(
    const FinishedGameData* finishedGameData,
    ForkData* forkData,
    const FancyModes& fancyModes,
    const GameInitializer* gameInit,
    Rand& gameRand
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
    Search* botB,
    Search* botW,
    const Board& board,
    BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    Logger& logger,
    const OtherGameProperties& otherGameProps,
    Rand& rand
  );

  double getSearchFactor(
    double searchFactorWhenWinningThreshold,
    double searchFactorWhenWinning,
    const SearchParams& params,
    const std::vector<double>& recentWinLossValues,
    Player pla
  );

  int numHandicapStones(const Board& initialBoard, const std::vector<Move>& moveHistory, bool assumeMultipleStartingBlackMovesAreHandicap);

  double getHackedLCBForWinrate(const Search* search, const AnalysisData& data, Player pla);
}


//Class for running a game and enqueueing the result as training data.
//Wraps together most of the neural-net-independent parameters to spawn and run a full game.
class GameRunner {
  bool logSearchInfo;
  bool logMoves;
  int maxMovesPerGame;
  bool clearBotBeforeSearch;
  std::string searchRandSeedBase;
  FancyModes fancyModes;
  GameInitializer* gameInit;

public:
  GameRunner(ConfigParser& cfg, const std::string& searchRandSeedBase, FancyModes fancyModes, Logger& logger);
  ~GameRunner();

  //Will return NULL if stopped before the game completes. The caller is responsible for freeing the data
  //if it isn't NULL.
  FinishedGameData* runGame(
    int64_t gameIdx,
    const MatchPairer::BotSpec& botSpecB,
    const MatchPairer::BotSpec& botSpecW,
    ForkData* forkData,
    Logger& logger,
    std::vector<std::atomic<bool>*>& stopConditions,
    std::function<NNEvaluator*()>* checkForNewNNEval
  );

};


#endif  // PROGRAM_PLAY_H_
