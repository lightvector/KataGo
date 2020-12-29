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
#include "../program/playsettings.h"

struct InitialPosition {
  Board board;
  BoardHistory hist;
  Player pla;
  bool isPlainFork;
  bool isSekiFork;
  bool isHintFork;

  InitialPosition();
  InitialPosition(const Board& board, const BoardHistory& hist, Player pla, bool isPlainFork, bool isSekiFork, bool isHintFork);
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
  bool allowInteger = true;
};

struct OtherGameProperties {
  bool isSgfPos = false;
  bool isHintPos = false;
  bool allowPolicyInit = true;
  bool isFork = false;
  bool isHintFork = false;

  int hintTurn = -1;
  Hash128 hintPosHash;
  Loc hintLoc = Board::NULL_LOC;

  //Note: these two behave slightly differently than the ones in searchParams - as properties for the whole
  //game, they make the playouts *actually* vary instead of only making the neural net think they do.
  double playoutDoublingAdvantage = 0.0;
  Player playoutDoublingAdvantagePla = C_EMPTY;
};

//Object choosing random initial rules and board sizes for games. Threadsafe.
class GameInitializer {
 public:
  GameInitializer(ConfigParser& cfg, Logger& logger);
  GameInitializer(ConfigParser& cfg, Logger& logger, const std::string& randSeed);
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
    const PlaySettings& playSettings,
    OtherGameProperties& otherGameProps,
    const Sgf::PositionSample* startPosSample
  );

  //A version that doesn't randomize params
  void createGame(
    Board& board, Player& pla, BoardHistory& hist,
    ExtraBlackAndKomi& extraBlackAndKomi,
    const InitialPosition* initialPosition,
    const PlaySettings& playSettings,
    OtherGameProperties& otherGameProps,
    const Sgf::PositionSample* startPosSample
  );

  Rules randomizeScoringAndTaxRules(Rules rules, Rand& randToUse) const;

  //Only sample the space of possible rules
  Rules createRules();
  bool isAllowedBSize(int xSize, int ySize);

  std::vector<int> getAllowedBSizes() const;

 private:
  void initShared(ConfigParser& cfg, Logger& logger);
  void createGameSharedUnsynchronized(
    Board& board, Player& pla, BoardHistory& hist,
    ExtraBlackAndKomi& extraBlackAndKomi,
    const InitialPosition* initialPosition,
    const PlaySettings& playSettings,
    OtherGameProperties& otherGameProps,
    const Sgf::PositionSample* startPosSample
  );
  Rules createRulesUnsynchronized();

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
  double sgfCompensateKomiProb;
  double komiBigStdevProb;
  float komiBigStdev;
  bool komiAuto;

  int numExtraBlackFixed;
  double noResultStdev;
  double drawRandRadius;

  std::vector<Sgf::PositionSample> startPoses;
  std::vector<double> startPosCumProbs;
  double startPosesProb;

  std::vector<Sgf::PositionSample> hintPoses;
  std::vector<double> hintPosCumProbs;
  double hintPosesProb;
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
  int64_t getNumGamesTotalToGenerate() const;

  //Get next matchup and log stuff
  bool getMatchup(
    BotSpec& botSpecB, BotSpec& botSpecW, Logger& logger
  );

 private:
  int numBots;
  std::vector<std::string> botNames;
  std::vector<NNEvaluator*> nnEvals;
  std::vector<SearchParams> baseParamss;

  std::vector<bool> excludeBot;
  std::vector<int> secondaryBots;
  std::vector<int> blackPriority;
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


//Functions to run a single game or other things
namespace Play {

  //In the case where checkForNewNNEval is provided, will MODIFY the provided botSpecs with any new nneval!
  FinishedGameData* runGame(
    const Board& startBoard, Player pla, const BoardHistory& startHist, ExtraBlackAndKomi extraBlackAndKomi,
    MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
    const std::string& searchRandSeed,
    bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, std::vector<std::atomic<bool>*>& stopConditions,
    const PlaySettings& playSettings, const OtherGameProperties& otherGameProps,
    Rand& gameRand,
    std::function<NNEvaluator*()> checkForNewNNEval,
    std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const Search*)> onEachMove
  );

  //In the case where checkForNewNNEval is provided, will MODIFY the provided botSpecs with any new nneval!
  FinishedGameData* runGame(
    const Board& startBoard, Player pla, const BoardHistory& startHist, ExtraBlackAndKomi extraBlackAndKomi,
    MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
    Search* botB, Search* botW,
    bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
    Logger& logger, bool logSearchInfo, bool logMoves,
    int maxMovesPerGame, std::vector<std::atomic<bool>*>& stopConditions,
    const PlaySettings& playSettings, const OtherGameProperties& otherGameProps,
    Rand& gameRand,
    std::function<NNEvaluator*()> checkForNewNNEval,
    std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const Search*)> onEachMove
  );

  void maybeForkGame(
    const FinishedGameData* finishedGameData,
    ForkData* forkData,
    const PlaySettings& playSettings,
    Rand& gameRand,
    Search* bot
  );

  void maybeSekiForkGame(
    const FinishedGameData* finishedGameData,
    ForkData* forkData,
    const PlaySettings& playSettings,
    const GameInitializer* gameInit,
    Rand& gameRand
  );

  void maybeHintForkGame(
    const FinishedGameData* finishedGameData,
    ForkData* forkData,
    const OtherGameProperties& otherGameProps
  );

}


//Class for running a game and enqueueing the result as training data.
//Wraps together most of the neural-net-independent parameters to spawn and run a full game.
class GameRunner {
  bool logSearchInfo;
  bool logMoves;
  int maxMovesPerGame;
  bool clearBotBeforeSearch;
  PlaySettings playSettings;
  GameInitializer* gameInit;

public:
  GameRunner(ConfigParser& cfg, PlaySettings playSettings, Logger& logger);
  GameRunner(ConfigParser& cfg, const std::string& gameInitRandSeed, PlaySettings fModes, Logger& logger);
  ~GameRunner();

  //Will return NULL if stopped before the game completes. The caller is responsible for freeing the data
  //if it isn't NULL.
  FinishedGameData* runGame(
    const std::string& seed,
    const MatchPairer::BotSpec& botSpecB,
    const MatchPairer::BotSpec& botSpecW,
    ForkData* forkData,
    const Sgf::PositionSample* startPosSample,
    Logger& logger,
    std::vector<std::atomic<bool>*>& stopConditions,
    std::function<NNEvaluator*()> checkForNewNNEval,
    std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const Search*)> onEachMove,
    bool logOwnership
  );

  const GameInitializer* getGameInitializer() const;

};


#endif  // PROGRAM_PLAY_H_
