#ifndef NEURALNET_NNINPUTS_H_
#define NEURALNET_NNINPUTS_H_

#include <memory>

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"

void setRowBin(float* rowBin, int pos, int feature, float value, int posStride, int featureStride);

namespace NNPos {
  constexpr int MAX_BOARD_LEN_X = Board::MAX_LEN_X;
  constexpr int MAX_BOARD_LEN_Y = Board::MAX_LEN_Y;
  constexpr int MAX_BOARD_LEN = Board::MAX_LEN;
  constexpr int MAX_BOARD_AREA = MAX_BOARD_LEN_X * MAX_BOARD_LEN_Y;
  //Policy output adds +1 for the pass move
  constexpr int MAX_NN_POLICY_SIZE = MAX_BOARD_AREA + 1;
  //Extra score distribution radius, used for writing score in data rows and for the neural net score belief output
  constexpr int EXTRA_SCORE_DISTR_RADIUS = 60;
  //Used various places we clip komi beyond board area.
  constexpr float KOMI_CLIP_RADIUS = 20.0f;

  int xyToPos(int x, int y, int nnXLen);
  int locToPos(Loc loc, int boardXSize, int nnXLen, int nnYLen);
  Loc posToLoc(int pos, int boardXSize, int boardYSize, int nnXLen, int nnYLen);
  int getPassPos(int nnXLen, int nnYLen);
  bool isPassPos(int pos, int nnXLen, int nnYLen);
  int getPolicySize(int nnXLen, int nnYLen);
}

namespace NNInputs {
  constexpr int SYMMETRY_NOTSPECIFIED = -1;
  constexpr int SYMMETRY_ALL = -2;
}

struct MiscNNInputParams {
  double drawEquivalentWinsForWhite = 0.5;
  bool conservativePassAndIsRoot = false;
  bool enablePassingHacks = false;
  double playoutDoublingAdvantage = 0.0;
  float nnPolicyTemperature = 1.0f;
  bool avoidMYTDaggerHack = false;
  // If no symmetry is specified, it will use default or random based on config, unless node is already cached.
  int symmetry = NNInputs::SYMMETRY_NOTSPECIFIED;
  double policyOptimism = 0.0;
  int maxHistory = 1000;

  static const Hash128 ZOBRIST_CONSERVATIVE_PASS;
  static const Hash128 ZOBRIST_FRIENDLY_PASS;
  static const Hash128 ZOBRIST_PASSING_HACKS;
  static const Hash128 ZOBRIST_PLAYOUT_DOUBLINGS;
  static const Hash128 ZOBRIST_NN_POLICY_TEMP;
  static const Hash128 ZOBRIST_AVOID_MYTDAGGER_HACK;
  static const Hash128 ZOBRIST_POLICY_OPTIMISM;
  static const Hash128 ZOBRIST_ZERO_HISTORY;
};

namespace NNInputs {
  const int NUM_FEATURES_SPATIAL_V3 = 22;
  const int NUM_FEATURES_GLOBAL_V3 = 14;

  const int NUM_FEATURES_SPATIAL_V4 = 22;
  const int NUM_FEATURES_GLOBAL_V4 = 14;

  const int NUM_FEATURES_SPATIAL_V5 = 13;
  const int NUM_FEATURES_GLOBAL_V5 = 12;

  const int NUM_FEATURES_SPATIAL_V6 = 22;
  const int NUM_FEATURES_GLOBAL_V6 = 16;

  const int NUM_FEATURES_SPATIAL_V7 = 22;
  const int NUM_FEATURES_GLOBAL_V7 = 19;

  constexpr int NUM_FEATURES_SPATIAL_V_DOTS = 18;
  constexpr int NUM_FEATURES_GLOBAL_V_DOTS = 22;

  constexpr int DOTS_FEATURE_SPATIAL_ON_BOARD = 0; // 0
  constexpr int DOTS_FEATURE_SPATIAL_PLAYER_ACTIVE = 1; // 1
  constexpr int DOTS_FEATURE_SPATIAL_PLAYER_OPP_ACTIVE = 2; // 2
  constexpr int DOTS_FEATURE_SPATIAL_PLAYER_PLACED = 3; // 1
  constexpr int DOTS_FEATURE_SPATIAL_PLAYER_OPP_PLACED = 4; // 2
  constexpr int DOTS_FEATURE_SPATIAL_DEAD_DOTS = 5; // Actually scoring
  constexpr int DOTS_FEATURE_SPATIAL_GROUNDED = 6; // Analogue of territory (18,19)
  constexpr int DOTS_FEATURE_SPATIAL_PLAYER_CAPTURES = 7; // (3,4,5)
  constexpr int DOTS_FEATURE_SPATIAL_PLAYER_SURROUNDINGS = 8; // (3,4,5)
  constexpr int DOTS_FEATURE_SPATIAL_PLAYER_OPP_CAPTURES = 9; // (3,4,5)
  constexpr int DOTS_FEATURE_SPATIAL_PLAYER_OPP_SURROUNDINGS = 10; // (3,4,5)
  constexpr int DOTS_FEATURE_SPATIAL_LAST_MOVE_0 = 10; // 9
  constexpr int DOTS_FEATURE_SPATIAL_LAST_MOVE_1 = 11; // 10
  constexpr int DOTS_FEATURE_SPATIAL_LAST_MOVE_2 = 12; // 11
  constexpr int DOTS_FEATURE_SPATIAL_LAST_MOVE_3 = 13; // 12
  constexpr int DOTS_FEATURE_SPATIAL_LAST_MOVE_4 = 14; // 13
  constexpr int DOTS_FEATURE_SPATIAL_LADDER_CAPTURED = 15; // 14, TODO: implement later
  constexpr int DOTS_FEATURE_SPATIAL_LADDER_CAPTURED_PREVIOUS = 16; // 15, TODO: implement later
  constexpr int DOTS_FEATURE_SPATIAL_LADDER_CAPTURED_PREVIOUS_2 = 17; // 16, TODO: implement later
  constexpr int DOTS_FEATURE_SPATIAL_LADDER_WORKING_MOVES = 18; // 17, TODO: implement later

  constexpr int DOTS_FEATURE_GLOBAL_KOMI = 0; // 5
  constexpr int DOTS_FEATURE_GLOBAL_SUICIDE = 1; // 8
  constexpr int DOTS_FEATURE_GLOBAL_CAPTURE_EMPTY = 2; // 9, 10, 11
  constexpr int DOTS_FEATURE_GLOBAL_OPENING_ENABLED = 3; // Not sure if it's necessary (15)
  constexpr int DOTS_FEATURE_GLOBAL_OPENING_DOTS = 4; // Not sure if it's necessary, number of opening dots? (16)
  constexpr int DOTS_FEATURE_GLOBAL_GROUNDING_WINS_GAME = 5;
  constexpr int DOTS_FEATURE_GLOBAL_FIELD_SIZE_KOMI_PARITY = 6; // Not sure what this is (18)

  Hash128 getHash(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams
  );

  int getNumberOfSpatialFeatures(int version);
  int getNumberOfGlobalFeatures(int version);

  void fillRowVN(
    int version,
    const Board& board, const BoardHistory& hist, Player nextPlayer,
    const MiscNNInputParams& nnInputParams,
    int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV3(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV4(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV5(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV6(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV7(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowVDots(
    const Board& board, const BoardHistory& hist, Player nextPlayer,
    const MiscNNInputParams& nnInputParams,
    int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );

  //If groupTax is specified, for each color region of area, reduce weight on empty spaces equally to reduce the total sum by 2.
  //(but should handle seki correctly)
  void fillScoring(
    const Board& board,
    const Color* area,
    bool groupTax,
    float* scoring
  );
}

struct NNOutput {
  Hash128 nnHash; //NNInputs - getHash

  //Initially from the perspective of the player to move at the time of the eval, fixed up later in nnEval.cpp
  //to be the value from white's perspective.
  //These three are categorial probabilities for each outcome.
  float whiteWinProb;
  float whiteLossProb;
  float whiteNoResultProb;

  //The first two moments of the believed distribution of the expected score at the end of the game, from white's perspective.
  float whiteScoreMean;
  float whiteScoreMeanSq;
  //Points to make game fair
  float whiteLead;
  //Expected arrival time of remaining game variance, in turns, weighted by variance, only when modelVersion >= 9
  float varTimeLeft;
  //A metric indicating the "typical" error in the winloss value or the score that the net expects, relative to the
  //short-term future MCTS value.
  float shorttermWinlossError;
  float shorttermScoreError;

  //Indexed by pos rather than loc
  //Values in here will be set to negative for illegal moves, including superko
  float policyProbs[NNPos::MAX_NN_POLICY_SIZE];
  //The optimism value used for this neural net evaluation.
  float policyOptimismUsed;

  int nnXLen;
  int nnYLen;
  //If not NULL, then this contains a nnXLen*nnYLen-sized map of expected ownership on the board.
  float* whiteOwnerMap;

  //If not NULL, then contains policy with dirichlet noise or any other noise adjustments for this node
  float* noisedPolicyProbs;

  NNOutput(); //Does NOT initialize values
  NNOutput(const NNOutput& other);
  ~NNOutput();

  //Averages the others. Others must be nonempty and share the same nnHash (i.e. be for the same board, except if somehow the hash collides)
  //Does NOT carry over noisedPolicyProbs.
  NNOutput(const std::vector<std::shared_ptr<NNOutput>>& others);

  NNOutput& operator=(const NNOutput&);

  inline float* getPolicyProbsMaybeNoised() { return noisedPolicyProbs != NULL ? noisedPolicyProbs : policyProbs; }
  inline const float* getPolicyProbsMaybeNoised() const { return noisedPolicyProbs != NULL ? noisedPolicyProbs : policyProbs; }
  void debugPrint(std::ostream& out, const Board& board);
  inline int getPos(Loc loc, const Board& board) const { return NNPos::locToPos(loc, board.x_size, nnXLen, nnYLen ); }
};

namespace SymmetryHelpers {
  //A symmetry is 3 bits flipY(bit 0), flipX(bit 1), transpose(bit 2). They are applied in that order.
  //The first four symmetries only reflect, and do not transpose X and Y.
  constexpr int SYMMETRY_NONE = 0;
  constexpr int SYMMETRY_FLIP_Y = 1;
  constexpr int SYMMETRY_FLIP_X = 2;
  constexpr int SYMMETRY_FLIP_Y_X = 3; // Rotate 180
  constexpr int SYMMETRY_TRANSPOSE = 4; // Rotate 90 CW + Flip X; Rotate 90 CCW + FlipY
  constexpr int SYMMETRY_TRANSPOSE_FLIP_X = 5; // Rotate 90 CW
  constexpr int SYMMETRY_TRANSPOSE_FLIP_Y = 6; // Rotate 90 CCW
  constexpr int SYMMETRY_TRANSPOSE_FLIP_Y_X = 7; // Rotate 90 CW + Flip Y; Rotate 90 CCW + FlipX

  constexpr int NUM_SYMMETRIES = 8;
  constexpr int NUM_SYMMETRIES_WITHOUT_TRANSPOSE = 4;

  //These two IGNORE transpose if hSize and wSize do not match. So non-square transposes are disallowed.
  //copyOutputsWithSymmetry performs the inverse of symmetry.
  void copyInputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry);
  void copyOutputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int symmetry);

  //Applies a symmetry to a location
  Loc getSymLoc(int x, int y, const Board& board, int symmetry);
  Loc getSymLoc(Loc loc, const Board& board, int symmetry);
  Loc getSymLoc(int x, int y, int xSize, int ySize, int symmetry);
  Loc getSymLoc(Loc loc, int xSize, int ySize, int symmetry);

  //Applies a symmetry to a board
  Board getSymBoard(const Board& board, int symmetry);

  //Get the inverse of the specified symmetry
  int invert(int symmetry);
  //Get the symmetry equivalent to first applying firstSymmetry and then applying nextSymmetry.
  int compose(int firstSymmetry, int nextSymmetry);
  int compose(int firstSymmetry, int nextSymmetry, int nextNextSymmetry);

  inline bool isTranspose(int symmetry) { return (symmetry & 0x4) != 0; }
  inline bool isFlipX(int symmetry) { return (symmetry & 0x2) != 0; }
  inline bool isFlipY(int symmetry) { return (symmetry & 0x1) != 0; }

  //Fill isSymDupLoc with true on all but one copy of each symmetrically equivalent move, and false everywhere else.
  //isSymDupLocs should be an array of size Board::MAX_ARR_SIZE
  //If onlySymmetries is not NULL, will only consider the symmetries specified there.
  //validSymmetries will be filled with all symmetries of the current board, including using history for checking ko/superko and some encore-related state.
  //This implementation is dependent on specific order of the symmetries (i.e. transpose is coded as 0x4)
  //Will pretend moves that have a nonzero value in avoidMoves do not exist.
  void markDuplicateMoveLocs(
    const Board& board,
    const BoardHistory& hist,
    const std::vector<int>* onlySymmetries,
    const std::vector<int>& avoidMoves,
    bool* isSymDupLoc,
    std::vector<int>& validSymmetries
  );

// For each symmetry, return a metric about the "amount" of difference that board would have with other
// if symmetry were applied to board.
  void getSymmetryDifferences(
    const Board& board, const Board& other, double maxDifferenceToReport, double symmetryDifferences[NUM_SYMMETRIES]
  );

  std::string symmetryToString(int symmetry);
}

//Utility functions for computing the "scoreValue", the unscaled utility of various numbers of points, prior to multiplication by
//staticScoreUtilityFactor or dynamicScoreUtilityFactor (see searchparams.h)
namespace ScoreValue {
  //MUST BE CALLED AT PROGRAM START!
  void initTables();
  void freeTables();

  //The number of wins a game result should count as
  double whiteWinsOfWinner(Player winner, double drawEquivalentWinsForWhite);
  //The score difference that a game result should count as on average
  double whiteScoreDrawAdjust(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite, const BoardHistory& hist);

  //The unscaled utility of achieving a certain score difference
  double whiteScoreValueOfScoreSmooth(double finalWhiteMinusBlackScore, double center, double scale, double drawEquivalentWinsForWhite, double sqrtBoardArea, const BoardHistory& hist);
  double whiteScoreValueOfScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, double scale, double sqrtBoardArea);
  //Approximately invert whiteScoreValueOfScoreSmooth
  double approxWhiteScoreOfScoreValueSmooth(double scoreValue, double center, double scale, double sqrtBoardArea);
  //The derviative of whiteScoreValueOfScoreSmoothNoDrawAdjust with respect to finalWhiteMinusBlackScore.
  double whiteDScoreValueDScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, double scale, double sqrtBoardArea);

  //Compute what the scoreMeanSq should be for a final game result
  //It is NOT simply the same as finalWhiteMinusBlackScore^2 because for integer komi we model it as a distribution where with the appropriate probability
  //you gain or lose 0.5 point to achieve the desired drawEquivalentWinsForWhite, so it actually has some variance.
  double whiteScoreMeanSqOfScoreGridded(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite);

  //The expected unscaled utility of the final score difference, given the mean and stdev of the distribution of that difference,
  //assuming roughly a normal distribution.
  double expectedWhiteScoreValue(double whiteScoreMean, double whiteScoreStdev, double center, double scale, double sqrtBoardArea);

  //Get the standard deviation of score given the E(score) and E(score^2)
  double getScoreStdev(double scoreMeanAvg, double scoreMeanSqAvg);
}

#endif  // NEURALNET_NNINPUTS_H_
