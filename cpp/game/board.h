/*
 * board.h
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modifications).
 */

#ifndef GAME_BOARD_H_
#define GAME_BOARD_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../external/nlohmann_json/json.hpp"
#include "rules.h"
#include "../core/rand.h"

#ifndef COMPILE_MAX_BOARD_LEN
#define COMPILE_MAX_BOARD_LEN 39
#endif

#define FOREACHADJ(BLOCK) {int ADJOFFSET = -(x_size+1); {BLOCK}; ADJOFFSET = -1; {BLOCK}; ADJOFFSET = 1; {BLOCK}; ADJOFFSET = x_size+1; {BLOCK}};

//TYPES AND CONSTANTS-----------------------------------------------------------------

static constexpr int LEFT_TOP_INDEX = 0;
static constexpr int TOP_INDEX = 1;
static constexpr int RIGHT_TOP_INDEX = 2;
static constexpr int RIGHT_INDEX = 3;
static constexpr int RIGHT_BOTTOM_INDEX = 4;
static constexpr int BOTTOM_INDEX = 5;
static constexpr int LEFT_BOTTOM_INDEX = 6;
static constexpr int LEFT_INDEX = 7;

struct Board;

typedef int8_t State;

static constexpr int PLAYER_BITS_COUNT = 2;
static constexpr State ACTIVE_MASK = (1 << PLAYER_BITS_COUNT) - 1;

static Color getOpp(Color c)
{return c ^ 3;}

Color getActiveColor(State state);

Color getPlacedDotColor(State s);

Color getEmptyTerritoryColor(State s);

bool isGrounded(State state);

//Conversions for players and colors
namespace PlayerIO {
  char colorToChar(Color c);
  std::string playerToStringShort(Player p);
  std::string playerToString(Player p);
  bool tryParsePlayer(const std::string& s, Player& pla);
  Player parsePlayer(const std::string& s);
}

namespace Location
{
  Loc getLoc(int x, int y, int x_size);
  int getX(Loc loc, int x_size);
  int getY(Loc loc, int x_size);

  void getAdjacentOffsets(short adj_offsets[8], int x_size, bool isDots);
  bool isAdjacent(Loc loc0, Loc loc1, int x_size);
  Loc getMirrorLoc(Loc loc, int x_size, int y_size);
  Loc getCenterLoc(int x_size, int y_size);
  Loc getCenterLoc(const Board& b);
  bool isCentral(Loc loc, int x_size, int y_size);
  bool isNearCentral(Loc loc, int x_size, int y_size);
  int distance(Loc loc0, Loc loc1, int x_size);
  int euclideanDistanceSquared(Loc loc0, Loc loc1, int x_size);
  int getGetBigJumpInitialIndex(Loc loc0, Loc loc1, int x_size);
  Loc getNextLocCW(Loc loc0, Loc loc1, int x_size);

  std::string toString(Loc loc, int x_size, int y_size, bool isDots);
  std::string toString(Loc loc, const Board& b);
  std::string toStringMach(Loc loc, int x_size, bool isDots);
  std::string toStringMach(Loc loc, const Board& b);

  bool tryOfString(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfString(const std::string& str, const Board& b, Loc& result);
  Loc ofString(const std::string& str, int x_size, int y_size);
  Loc ofString(const std::string& str, const Board& b);

  //Same, but will parse "null" as Board::NULL_LOC
  bool tryOfStringAllowNull(const std::string& str, int x_size, int y_size, Loc& result);
  bool tryOfStringAllowNull(const std::string& str, const Board& b, Loc& result);
  Loc ofStringAllowNull(const std::string& str, int x_size, int y_size);
  Loc ofStringAllowNull(const std::string& str, const Board& b);

  std::vector<Loc> parseSequence(const std::string& str, const Board& b);

  Loc xm1y(Loc loc);
  Loc xm1ym1(Loc loc, int x_size);
  Loc xym1(Loc loc, int x_size);
  Loc xp1ym1(Loc loc, int x_size);
  Loc xp1y(Loc loc);
  Loc xp1yp1(Loc loc, int x_size);
  Loc xyp1(Loc loc, int x_size);
  Loc xm1yp1(Loc loc, int x_size);
}

//Fast lightweight board designed for playouts and simulations, where speed is essential.
//Simple ko rule only.
//Does not enforce player turn order.

constexpr static int getMaxArrSize(const int x_size, const int y_size) {
  return (x_size+1)*(y_size+2)+1;
}

struct Board
{
  //Initialization------------------------------
  //Initialize the zobrist hash.
  //MUST BE CALLED AT PROGRAM START!
  static void initHash();

  //Board parameters and Constants----------------------------------------

  static constexpr int MAX_LEN_X = //Maximum x edge length allowed for the board
#ifdef COMPILE_MAX_BOARD_LEN_X
    COMPILE_MAX_BOARD_LEN_X;
#else
    COMPILE_MAX_BOARD_LEN;
#endif
  static constexpr int MAX_LEN_Y = //Maximum y edge length allowed for the board
#ifdef COMPILE_MAX_BOARD_LEN_Y
    COMPILE_MAX_BOARD_LEN_Y;
#else
    COMPILE_MAX_BOARD_LEN;
#endif
  static constexpr int MAX_LEN = std::max(MAX_LEN_X, MAX_LEN_Y);  //Maximum edge length allowed for the board
  static constexpr int DEFAULT_LEN_X = std::min(MAX_LEN_X,19); //Default x edge length for board if unspecified
  static constexpr int DEFAULT_LEN_Y = std::min(MAX_LEN_Y,19); //Default y edge length for board if unspecified
  static constexpr int MAX_PLAY_SIZE = MAX_LEN_X * MAX_LEN_Y;  //Maximum number of playable spaces
  static constexpr int MAX_ARR_SIZE = getMaxArrSize(MAX_LEN_X, MAX_LEN_Y); //Maximum size of arrays needed

  //Location used to indicate an invalid spot on the board.
  static constexpr Loc NULL_LOC = 0;
  //Location used to indicate a pass or grounding (Dots game) move is desired.
  static constexpr Loc PASS_LOC = 1;

  //Zobrist Hashing------------------------------
  static bool IS_ZOBRIST_INITALIZED;
  static Hash128 ZOBRIST_SIZE_X_HASH[MAX_LEN_X+1];
  static Hash128 ZOBRIST_SIZE_Y_HASH[MAX_LEN_Y+1];
  static Hash128 ZOBRIST_BOARD_HASH[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_BOARD_HASH2[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_PLAYER_HASH[4];
  static Hash128 ZOBRIST_KO_LOC_HASH[MAX_ARR_SIZE];
  static Hash128 ZOBRIST_KO_MARK_HASH[MAX_ARR_SIZE][4];
  static Hash128 ZOBRIST_ENCORE_HASH[3];
  static Hash128 ZOBRIST_SECOND_ENCORE_START_HASH[MAX_ARR_SIZE][4];
  static const Hash128 ZOBRIST_PASS_ENDS_PHASE;
  static const Hash128 ZOBRIST_GAME_IS_OVER;

  //Structs---------------------------------------

  //Tracks a chain/string/group of stones
  struct ChainData {
    Player owner;        //Owner of chain
    short num_locs;      //Number of stones in chain
    short num_liberties; //Number of liberties in chain
  };

  //Tracks locations for fast random selection
  /* struct PointList { */
  /*   PointList(); */
  /*   PointList(const PointList&); */
  /*   void operator=(const PointList&); */
  /*   void add(Loc); */
  /*   void remove(Loc); */
  /*   int size() const; */
  /*   Loc& operator[](int); */
  /*   bool contains(Loc loc) const; */

  /*   Loc list_[MAX_PLAY_SIZE];   //Locations in the list */
  /*   int indices_[MAX_ARR_SIZE]; //Maps location to index in the list */
  /*   int size_; */
  /* }; */

  struct Base {
    std::vector<Loc> rollback_locations;
    std::vector<State> rollback_states;
    Player pla{};
    bool is_real{};

    Base() = default;
    Base(Player newPla, const std::vector<Loc>& rollbackLocations, const std::vector<State>& rollbackStates, bool isReal);
  };

  //Move data passed back when moves are made to allow for undos
  struct MoveRecord {
    Player pla;
    Loc loc;
    Loc ko_loc;
    uint8_t capDirs; //First 4 bits indicate directions of capture, fifth bit indicates suicide

    // Move data for Dots game
    State previousState;
    std::vector<Base> bases;
    std::vector<Loc> emptyBaseInvalidateLocations;
    std::vector<Loc> groundingLocations;

    MoveRecord() = default;

    // Constructor for Go game
    MoveRecord(
      Loc initLoc,
      Player initPla,
      Loc init_ko_loc,
      uint8_t initCapDirs
    );

    // Constructor for Dots game
    MoveRecord(
      Loc newLoc,
      Player newPla,
      State newPreviousState,
      const std::vector<Base>& newBases,
      const std::vector<Loc>& newEmptyBaseInvalidateLocations,
      const std::vector<Loc>& newGroundingLocations
    );
  };

  //Constructors---------------------------------
  Board();  //Create Board of size (DEFAULT_LEN,DEFAULT_LEN)
  explicit Board(const Rules& rules);
  Board(int x, int y, const Rules& rules); // Create Board of size (x,y) with the specified Rules
  Board(const Board& other);

  Board& operator=(const Board&) = default;

  //Functions------------------------------------

  [[nodiscard]] Color getColor(Loc loc) const;
  [[nodiscard]] State getState(Loc loc) const;
  void setState(Loc loc, State state);
  bool isDots() const;

  template<typename Func> void forEachAdjacent(const Loc loc, Func&& f) const {
    const int stride = x_size + 1;
    f(loc - stride);
    f(loc - 1);
    f(loc + 1);
    f(loc + stride);
  }

  double sqrtBoardArea() const;

  //Gets the number of stones of the chain at loc. Precondition: location must be black or white.
  int getChainSize(Loc loc) const;
  //Gets the number of liberties of the chain at loc. Precondition: location must be black or white.
  int getNumLiberties(Loc loc) const;
  //Returns the number of liberties a new stone placed here would have, or max if it would be >= max.
  int getNumLibertiesAfterPlay(Loc loc, Player pla, int max) const;
  //Returns a fast lower and upper bound on the number of liberties a new stone placed here would have
  void getBoundNumLibertiesAfterPlay(Loc loc, Player pla, int& lowerBound, int& upperBound) const;
  //Gets the number of empty spaces directly adjacent to this location
  int getNumImmediateLiberties(Loc loc) const;

  //Check if moving here would be a self-capture
  bool isSuicide(Loc loc, Player pla) const;
  //Check if moving here would be an illegal self-capture
  bool isIllegalSuicide(Loc loc, Player pla, bool isMultiStoneSuicideLegal) const;
  //Check if moving here is illegal due to simple ko
  bool isKoBanned(Loc loc) const;
  //Check if moving here is legal. Equivalent to isLegalIgnoringKo && !isKoBanned
  bool isLegal(Loc loc, Player pla, bool isMultiStoneSuicideLegal, bool ignoreKo) const;
  //Check if this location is on the board
  bool isOnBoard(Loc loc) const;
  //Check if this location contains a simple eye for the specified player.
  bool isSimpleEye(Loc loc, Player pla) const;
  //Check if a move at this location would be a capture of an opponent group.
  bool wouldBeCapture(Loc loc, Player pla) const;
  //Check if a move at this location would be a capture in a simple ko mouth.
  bool wouldBeKoCapture(Loc loc, Player pla) const;
  Loc getKoCaptureLoc(Loc loc, Player pla) const;
  //Check if this location is adjacent to stones of the specified color
  bool isAdjacentToPla(Loc loc, Player pla) const;
  bool isAdjacentOrDiagonalToPla(Loc loc, Player pla) const;
  //Check if this location is adjacent a given chain.
  bool isAdjacentToChain(Loc loc, Loc chain) const;
  //Does this connect two pla distinct groups that are not both pass-alive and not within opponent pass-alive area either?
  bool isNonPassAliveSelfConnection(Loc loc, Player pla, Color* passAliveArea) const;
  //Is this board empty?
  bool isEmpty() const;
  //Count the number of stones on the board
  int numStonesOnBoard() const;
  int numPlaStonesOnBoard(Player pla) const;

  //Get a hash that combines the position of the board with simple ko prohibition and a player to move.
  Hash128 getSitHashWithSimpleKo(Player pla) const;

  //Lift any simple ko ban recorded on thie board due to an immediate prior ko capture.
  void clearSimpleKoLoc();
  //Directly set that there is a simple ko prohibition on this location. Note that this is not necessarily safe
  //when also using a BoardHistory, since the BoardHistory may not know about this change, or the game could be in cleanup phase, etc.
  void setSimpleKoLoc(Loc loc);

  //Sets the specified stone if possible, including overwriting existing stones.
  //Resolves any captures and/or suicides that result from setting that stone, including deletions of the stone itself.
  //Returns false if location or color were out of range.
  bool setStone(Loc loc, Color color);

  // Set the start pos and use the provided random in case of randomization is used
  // It should be called strictly before handicap placement
  void setStartPos(Rand& rand);
  //Sets the specified stone, including overwriting existing stones, but only if doing so will
  //not result in any captures or zero liberty groups.
  //Returns false if location or color were out of range, or if would cause a zero liberty group.
  //In case of failure, will restore the position, but may result in chain ids or ordering in the board changing.
  //If startPos is true, adds the move to start pos moves to distinguish between start pos and handicap stones
  bool setStoneFailIfNoLibs(Loc loc, Color color, bool startPos = false);
  //Same, but sets multiple stones, and only requires that the final configuration contain no zero-liberty groups.
  //If it does contain a zero liberty group, fails and returns false and leaves the board in an arbitrarily changed but valid state.
  //Also returns false if any location is specified more than once.
  //If startPos is true, adds the placements to start pos moves to distinguish between start pos and handicap stones
  bool setStonesFailIfNoLibs(const std::vector<Move>& placements, bool startPos = false);

  //Attempts to play the specified move. Returns true if successful, returns false if the move was illegal.
  bool playMove(Loc loc, Player pla, bool isMultiStoneSuicideLegal);

  //Plays the specified move, assuming it is legal.
  void playMoveAssumeLegal(Loc loc, Player pla);

  // Plays the specified move, assuming it is legal, and returns a MoveRecord for the move
  MoveRecord playMoveRecorded(Loc loc, Player pla);

  //Undo the move given by record. Moves MUST be undone in the order they were made.
  //Undos will NOT typically restore the precise representation in the board to the way it was. The heads of chains
  //might change, the order of the circular lists might change, etc.
  void undo(MoveRecord& record);

  //Get what the position hash would be if we were to play this move and resolve captures and suicides.
  //Assumes the move is on an empty location.
  Hash128 getPosHashAfterMove(Loc loc, Player pla) const;

  //Returns true if, for a move just played at loc, the sum of the number of stones in loc's group and the sizes of the empty regions it touches
  //are greater than bound. See also https://senseis.xmp.net/?Cycle for some interesting test cases for thinking about this bound.
  //Returns false for passes.
  bool simpleRepetitionBoundGt(Loc loc, int bound) const;

  //Get a random legal move that does not fill a simple eye.
  /* Loc getRandomMCLegal(Player pla); */

  //Check if the given stone is in unescapable atari or can be put into unescapable atari.
  //WILL perform a mutable search - may alter the linked lists or heads, etc.
  bool searchIsLadderCaptured(Loc loc, bool defenderFirst, std::vector<Loc>& buf);
  bool searchIsLadderCapturedAttackerFirst2Libs(Loc loc, std::vector<Loc>& buf, std::vector<Loc>& workingMoves);

  //If a point is a pass-alive stone or pass-alive territory for a color, mark it that color.
  //If nonPassAliveStones, also marks non-pass-alive stones that are not part of the opposing pass-alive territory.
  //If safeBigTerritories, also marks for each pla empty regions bordered by pla stones and no opp stones, where all pla stones are pass-alive.
  //If unsafeBigTerritories, also marks for each pla empty regions bordered by pla stones and no opp stones, regardless.
  //All other points are marked as C_EMPTY.
  //[result] must be a buffer of size MAX_ARR_SIZE and will get filled with the result
  // For Dots game it just calculates grounding
  void calculateArea(
    Color* result,
    bool nonPassAliveStones,
    bool safeBigTerritories,
    bool unsafeBigTerritories,
    bool isMultiStoneSuicideLegal
  ) const;

  int calculateOwnershipAndWhiteScore(Color* result, Color groundingPlayer) const;

  // Calculates the area (including non pass alive stones, safe and unsafe big territories)
  //However, strips out any "seki" regions.
  //Seki regions are that are adjacent to any remaining empty regions.
  //If keepTerritories, then keeps the surrounded territories in seki regions, only strips points for stones.
  //If keepStones, then keeps the stones, only strips points for surrounded territories.
  //whiteMinusBlackIndependentLifeRegionCount - multiply this by two for a group tax.
  void calculateIndependentLifeArea(
    Color* result,
    int& whiteMinusBlackIndependentLifeRegionCount,
    bool keepTerritories,
    bool keepStones,
    bool isMultiStoneSuicideLegal
  ) const;

  void calculateOneMoveCaptureAndBasePositionsForDots(bool isSuicideLegal, std::vector<Color>& captures, std::vector<Color>& bases) const;

  //Run some basic sanity checks on the board state, throws an exception if not consistent, for testing/debugging
  void checkConsistency() const;
  //For the moment, only used in testing since it does extra consistency checks.
  //If we need a version to be used in "prod", we could make an efficient version maybe as operator==.
  bool isEqualForTesting(const Board& other, bool checkNumCaptures = true, bool checkSimpleKo = true, bool checkRules = true) const;

  static Board parseBoard(int xSize, int ySize, const std::string& s, const Rules& rules = Rules::DEFAULT_GO, char lineDelimiter = '\n');
  std::string toString() const;
  static void printBoard(std::ostream& out, const Board& board, Loc markLoc, const std::vector<Move>* hist);
  static std::string toStringSimple(const Board& board, char lineDelimiter = '\n');
  static nlohmann::json toJson(const Board& board);
  static Board ofJson(const nlohmann::json& data);

  //Data--------------------------------------------

  int x_size;                  //Horizontal size of board
  int y_size;                  //Vertical size of board
  Rules rules;
  Color colors[MAX_ARR_SIZE];  //Color of each location on the board.

  Loc ko_loc;   //A simple ko capture was made here, making it illegal to replay here next move

  /* PointList empty_list; //List of all empty locations on board */

  Hash128 pos_hash; //A zobrist hash of the current board position (does not include ko point or player to move)

  int numBlackCaptures; //Number of b stones captured, informational and used by board history when clearing pos
  int numWhiteCaptures; //Number of w stones captured, informational and used by board history when clearing pos

  // Useful for fast calculation of the game result and finishing the game
  int blackScoreIfWhiteGrounds;
  int whiteScoreIfBlackGrounds;

  // Offsets to add to get clockwise traverse
  short adj_offsets[8];

  int numLegalMoves;

  //Every chain of stones has one of its stones arbitrarily designated as the head.
  std::vector<ChainData> chain_data; //For each head stone, the chaindata for the chain under that head. Undefined otherwise.
  std::vector<Loc> chain_head;       //Where is the head of this chain? Undefined if EMPTY or WALL
  std::vector<Loc> next_in_chain;    //Location of next stone in chain. Circular linked list. Undefined if EMPTY or WALL
  std::vector<Move> start_pos_moves; //Moves that are played at the very beginning of the game

  private:

  // Dots game data
  mutable std::vector<Loc> closureOrInvalidateLocsBuffer = std::vector<Loc>();
  mutable std::vector<Loc> territoryLocationsBuffer = std::vector<Loc>();
  mutable std::vector<Loc> walkStack = std::vector<Loc>();
  mutable std::vector<bool> visited_data = std::vector<bool>();

  // Dots game functions
  [[nodiscard]] bool wouldBeCaptureDots(Loc loc, Player pla) const;
  [[nodiscard]] bool isSuicideDots(Loc loc, Player pla) const;
  void playMoveAssumeLegalDots(Loc loc, Player pla);
  MoveRecord playMoveRecordedDots(Loc loc, Player pla);
  MoveRecord tryPlayMoveRecordedDots(Loc loc, Player pla, bool isSuicideLegal);
  void undoDots(MoveRecord& moveRecord);
  std::vector<short> fillGrounding(Loc loc);
  void captureWhenEmptyTerritoryBecomesRealBase(Loc initLoc, Player opp, std::vector<Base>& bases, bool& isGrounded);
  void tryCapture(
    Loc loc,
    Player pla,
    const std::array<Loc, 4>& unconnectedLocations,
    int unconnectedLocationsSize,
    bool& atLeastOneRealBaseIsGrounded,
    std::vector<Base>& bases);
  void ground(Player pla, std::vector<Loc>& emptyBaseInvalidatePositions, std::vector<Base>& bases);
  std::array<Loc, 4> getUnconnectedLocations(Loc loc, Player pla, int& size) const;
  void checkAndAddUnconnectedLocation(
    std::array<Loc, 4>& unconnectedLocationsBuffer,
    int& size,
    Player checkPla,
    Player currentPla,
    Loc addLoc1,
    Loc addLoc2) const;
  void tryGetCounterClockwiseClosure(Loc initialLoc, Loc startLoc, Player pla) const;
  Base buildBase(const std::vector<short>& closure, Player pla);
  void getTerritoryLocations(Player pla, Loc firstLoc, bool grounding, bool& createRealBase) const;
  Base createBaseAndUpdateStates(Player basePla, bool isReal);
  void updateScoreAndHashForTerritory(Loc loc, State state, Player basePla, bool rollback);
  void invalidateAdjacentEmptyTerritoryIfNeeded(Loc loc);
  void makeMoveAndCalculateCapturesAndBases(Player pla, Loc loc, bool isSuicideLegal,
    std::vector<signed char>& captures, std::vector<signed char>& bases) const;

  void setGrounded(Loc loc);
  void clearGrounded(Loc loc);
  bool isVisited(Loc loc) const;
  void setVisited(Loc loc) const;
  void clearVisited(Loc loc) const;
  void clearVisited(const std::vector<short>& locations) const;

  void init(int xS, int yS, const Rules& initRules);
  int countHeuristicConnectionLibertiesX2(Loc loc, Player pla) const;
  bool isLibertyOf(Loc loc, Loc head) const;
  void mergeChains(Loc loc1, Loc loc2);
  int removeChain(Loc loc);
  void removeSingleStone(Loc loc);

  void addChain(Loc loc, Player pla);
  Loc addChainHelper(Loc head, Loc tailTarget, Loc loc, Color color);
  void rebuildChain(Loc loc, Player pla);
  Loc rebuildChainHelper(Loc head, Loc tailTarget, Loc loc, Color color);
  void changeSurroundingLiberties(Loc loc, Color color, int delta);

  friend std::ostream& operator<<(std::ostream& out, const Board& board);

  int findLiberties(Loc loc, std::vector<Loc>& buf, int bufStart, int bufIdx) const;
  int findLibertyGainingCaptures(Loc loc, std::vector<Loc>& buf, int bufStart, int bufIdx) const;
  bool hasLibertyGainingCaptures(Loc loc) const;

  void calculateAreaForPla(
    Player pla,
    bool safeBigTerritories,
    bool unsafeBigTerritories,
    bool isMultiStoneSuicideLegal,
    Color* result
  ) const;

  bool isAdjacentToPlaHead(Player pla, Loc loc, Loc plaHead) const;

  void calculateIndependentLifeAreaHelper(
    const Color* basicArea,
    Color* result,
    int& whiteMinusBlackIndependentLifeRegionCount
  ) const;

  bool countEmptyHelper(bool* emptyCounted, Loc initialLoc, int& count, int bound) const;

  //static void monteCarloOwner(Player player, Board* board, int mc_counts[]);
};




#endif // GAME_BOARD_H_
