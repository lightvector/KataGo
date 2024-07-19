#ifndef DATAIO_SGF_H_
#define DATAIO_SGF_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../dataio/trainingwrite.h"
#include "../game/board.h"
#include "../game/boardhistory.h"

STRUCT_NAMED_TRIPLE(uint8_t,x,uint8_t,y,Player,pla,MoveNoBSize);
STRUCT_NAMED_PAIR(int,x,int,y,XYSize);

struct SgfNode {
  std::map<std::string,std::vector<std::string>>* props;
  MoveNoBSize move;

  SgfNode();
  ~SgfNode();

  SgfNode(const SgfNode& other);
  SgfNode(SgfNode&& other) noexcept;

  SgfNode& operator=(const SgfNode&);
  SgfNode& operator=(SgfNode&&) noexcept;

  bool hasProperty(const char* key) const;
  bool hasProperty(const std::string& key) const;
  std::string getSingleProperty(const char* key) const;
  std::string getSingleProperty(const std::string& key) const;
  const std::vector<std::string> getProperties(const char* key) const;
  const std::vector<std::string> getProperties(const std::string& key) const;

  void addProperty(const std::string& key, const std::string& value);
  void appendComment(const std::string& value);

  bool hasPlacements() const;
  void accumPlacements(std::vector<Move>& moves, int xSize, int ySize) const;
  void accumMoves(std::vector<Move>& moves, int xSize, int ySize) const;

  Color getPLSpecifiedColor() const;
  Rules getRulesFromRUTagOrFail() const;
  Player getSgfWinner() const;
  float getKomiOrFail() const;
  float getKomiOrDefault(float defaultKomi) const;
};

struct Sgf {
  static constexpr int RANK_UNKNOWN = -100000;

  std::string fileName;
  std::vector<SgfNode*> nodes;
  std::vector<Sgf*> children;
  Hash128 hash;

  Sgf();
  ~Sgf();

  Sgf(const Sgf&) = delete;
  Sgf& operator=(const Sgf&) = delete;

  static Sgf* parse(const std::string& str);
  static Sgf* loadFile(const std::string& file);
  static std::vector<Sgf*> loadFiles(const std::vector<std::string>& files);
  static std::vector<Sgf*> loadSgfsFile(const std::string& file);
  static std::vector<Sgf*> loadSgfsFiles(const std::vector<std::string>& files);

  static std::vector<Sgf*> loadSgfOrSgfsLogAndIgnoreErrors(const std::string& file, Logger& logger);

  XYSize getXYSize() const;
  float getKomiOrFail() const;
  float getKomiOrDefault(float defaultKomi) const;
  bool hasRules() const;
  Rules getRulesOrFail() const;
  int getHandicapValue() const;
  Player getSgfWinner() const;
  Color getFirstPlayerColor() const;

  int getRank(Player pla) const; //dan ranks are 1d=0, 2d=1,... 9d=8. Kyu ranks are negative.
  int getRating(Player pla) const;
  std::string getPlayerName(Player pla) const;

  bool hasRootProperty(const std::string& property) const;
  std::string getRootPropertyWithDefault(const std::string& property, const std::string& defaultRet) const;
  std::vector<std::string> getRootProperties(const std::string& property) const;

  void addRootProperty(const std::string& key, const std::string& value);

  void getPlacements(std::vector<Move>& moves, int xSize, int ySize) const;
  void getMoves(std::vector<Move>& moves, int xSize, int ySize) const;

  //Maximum depth of sgf tree in nodes
  int64_t depth() const;
  //Total number of sgf nodes
  int64_t nodeCount() const;
  //Total number of sgf branches (0 for a linear sgf, 1 if there is 1 fork, etc)
  int64_t branchCount() const;

  struct PositionSample {
    Board board;
    Player nextPla;
    //Prior to using the sample, play these moves on to the board.
    //This provides a little bit of history and context, which can also be relevant for setting up ko prohibitions.
    std::vector<Move> moves;
    //Turn number as of the start of board.
    int64_t initialTurnNumber;
    //Hinted move that may be good at the end of position sample, or Board::NULL_LOC
    Loc hintLoc;
    //The weight of this sample, for random selection
    double weight;
    //Arbitrary label or metadata
    std::string metadata;
    //Scaling of training weight in the training data
    double trainingWeight = 1.0;

    static std::string toJsonLine(const PositionSample& sample);
    static PositionSample ofJsonLine(const std::string& s);

    //Return a copy of this sample with all player stones and moves flipped to the opposite color
    Sgf::PositionSample getColorFlipped() const;

    //Return a copy of this sample except one move earlier
    Sgf::PositionSample previousPosition(double newWeight) const;
    bool hasPreviousPositions(int numPrevious) const;

    int64_t getCurrentTurnNumber() const;

    //For the moment, only used in testing since it does extra consistency checks.
    //If we need a version to be used in "prod", we could make an efficient version maybe as operator==.
    bool isEqualForTesting(const PositionSample& other, bool checkNumCaptures, bool checkSimpleKo) const;
  };

  //Loads SGF all unique positions in ALL branches of that SGF.
  //Hashes are used to filter out "identical" positions when loading many files from different SGFs that may have overlapping openings, etc.
  //The hashes are not guaranteed to correspond to position hashes, or anything else external to this function itself.
  //May raise an exception on illegal moves or other SGF issues, only partially appending things on to the boards and hists.
  //If rand is provided, will randomize order of iteration through the SGF.
  //If hashParent is true, will determine uniqueness by the combination of parent hash and own hash.
  void loadAllUniquePositions(
    std::set<Hash128>& uniqueHashes,
    bool hashComments,
    bool hashParent,
    bool flipIfPassOrWFirst,
    bool allowGameOver,
    Rand* rand,
    std::vector<PositionSample>& samples
  ) const;
  //f is allowed to mutate and consume sample.
  void iterAllUniquePositions(
    std::set<Hash128>& uniqueHashes,
    bool hashComments,
    bool hashParent,
    bool flipIfPassOrWFirst,
    bool allowGameOver,
    Rand* rand,
    std::function<void(PositionSample&,const BoardHistory&,const std::string&)> f
  ) const;

  //Same as iterAllUniquePositions, but without the uniqueness. Will re-traverse same positions if they
  //occur multiple times in the SGF.
  //f is allowed to mutate and consume sample.
  void iterAllPositions(
    bool flipIfPassOrWFirst,
    bool allowGameOver,
    Rand* rand,
    std::function<void(PositionSample&,const BoardHistory&,const std::string&)> f
  ) const;

  static std::set<Hash128> readExcludes(const std::vector<std::string>& files);

  private:
  void getMovesHelper(std::vector<Move>& moves, int xSize, int ySize) const;


  void iterAllPositionsHelper(
    Board& board, BoardHistory& hist, Player nextPla,
    const Rules& rules, int xSize, int ySize,
    PositionSample& sampleBuf,
    std::set<Hash128>& uniqueHashes,
    bool requireUnique,
    bool hashComments,
    bool hashParent,
    bool flipIfPassOrWFirst,
    bool allowGameOver,
    bool isRoot,
    Rand* rand,
    std::vector<std::pair<int64_t,int64_t>>& variationTraceNodesBranch,
    std::function<void(PositionSample&,const BoardHistory&,const std::string&)> f
  ) const;
  void samplePositionHelper(
    Board& board, BoardHistory& hist, Player nextPla,
    PositionSample& sampleBuf,
    std::set<Hash128>& uniqueHashes,
    bool requireUnique,
    bool hashComments,
    bool hashParent,
    bool flipIfPassOrWFirst,
    bool allowGameOver,
    const std::string& comments,
    std::function<void(PositionSample&,const BoardHistory&,const std::string&)> f
  ) const;
};

struct CompactSgf {
  std::string fileName;
  SgfNode rootNode;
  std::vector<Move> placements;
  std::vector<Move> moves;
  int xSize;
  int ySize;
  int64_t depth;
  Player sgfWinner;
  Hash128 hash;

  CompactSgf(const Sgf* sgf);
  CompactSgf(Sgf&& sgf);
  ~CompactSgf();

  CompactSgf(const CompactSgf&) = delete;
  CompactSgf& operator=(const CompactSgf&) = delete;

  static CompactSgf* parse(const std::string& str);
  static CompactSgf* loadFile(const std::string& file);
  static std::vector<CompactSgf*> loadFiles(const std::vector<std::string>& files);

  bool hasRules() const;
  Rules getRulesOrFail() const;
  Rules getRulesOrFailAllowUnspecified(const Rules& defaultRules) const;
  Rules getRulesOrWarn(const Rules& defaultRules, std::function<void(const std::string& msg)> f) const;

  void setupInitialBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist) const;
  void playMovesAssumeLegal(Board& board, Player& nextPla, BoardHistory& hist, int64_t turnIdx) const;
  void setupBoardAndHistAssumeLegal(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int64_t turnIdx) const;
  //These throw a StringError upon illegal move.
  void playMovesTolerant(Board& board, Player& nextPla, BoardHistory& hist, int64_t turnIdx, bool preventEncore) const;
  void setupBoardAndHistTolerant(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int64_t turnIdx, bool preventEncore) const;
};

namespace WriteSgf {
  //Write an SGF with no newlines to the given ostream.
  //If startTurnIdx >= 0, write a comment in the SGF root node indicating startTurnIdx, so as to
  //indicate the index of the first turn that should be used for training data. (0 means the whole SGF, 1 means skipping black's first move, etc).
  //If valueTargets is not NULL, also write down after each move the MCTS values following that search move.
  //If overideFinishedScore is not NAN and the game is finished and the result is a score (rather than noresult or resign), set this score instead, from white's perspective.
  void writeSgf(
    std::ostream& out, const std::string& bName, const std::string& wName,
    const BoardHistory& endHist,
    const FinishedGameData* gameData,
    bool tryNicerRulesString,
    bool omitResignPlayerMove
  );
  void writeSgf(
    std::ostream& out, const std::string& bName, const std::string& wName,
    const BoardHistory& endHist,
    const FinishedGameData* gameData,
    bool tryNicerRulesString,
    bool omitResignPlayerMove,
    double overrideFinishedWhiteScore
  );
  void writeSgf(
    std::ostream& out, const std::string& bName, const std::string& wName,
    const BoardHistory& endHist,
    const std::vector<std::string>& extraComments
  );
  void writeSgf(
    std::ostream& out, const std::string& bName, const std::string& wName,
    const BoardHistory& endHist,
    const FinishedGameData* gameData,
    bool tryNicerRulesString,
    bool omitResignPlayerMove,
    double overrideFinishedWhiteScore,
    const std::vector<std::string>& extraComments
  );

  //If hist is a finished game, print the result to out along with SGF tag, else do nothing
  void printGameResult(std::ostream& out, const BoardHistory& hist);
  void printGameResult(std::ostream& out, const BoardHistory& hist, double overrideFinishedWhiteScore);
  //Get the game result without a surrounding sgf tag
  std::string gameResultNoSgfTag(const BoardHistory& hist);
  std::string gameResultNoSgfTag(const BoardHistory& hist, double overrideFinishedWhiteScore);
}

#endif  // DATAIO_SGF_H_
