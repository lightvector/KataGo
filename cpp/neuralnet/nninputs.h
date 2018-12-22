#ifndef NNINPUTS_H
#define NNINPUTS_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"

namespace NNPos {
  //Currently, neural net policy output can handle a max of 19x19 boards.
  const int MAX_BOARD_LEN = 19;
  const int MAX_BOARD_AREA = MAX_BOARD_LEN * MAX_BOARD_LEN;
  //Policy output adds +1 for the pass move
  const int MAX_NN_POLICY_SIZE = MAX_BOARD_AREA + 1;
  //Extra score distribution radius, used for writing score in data rows and for the neural net score belief output
  const int EXTRA_SCORE_DISTR_RADIUS = 60;

  int xyToPos(int x, int y, int posLen);
  int locToPos(Loc loc, int boardXSize, int posLen);
  Loc posToLoc(int pos, int boardXSize, int boardYSize, int posLen);
  bool isPassPos(int pos, int posLen);
  int getPolicySize(int posLen);
}

namespace NNInputs {
  const int NUM_SYMMETRY_BOOLS = 3;
  const int NUM_SYMMETRY_COMBINATIONS = 8;

  const int NUM_FEATURES_V0 = 19;
  const int ROW_SIZE_V0 = NNPos::MAX_BOARD_LEN * NNPos::MAX_BOARD_LEN * NUM_FEATURES_V0;

  const int NUM_FEATURES_V1 = 19;
  const int ROW_SIZE_V1 = NNPos::MAX_BOARD_LEN * NNPos::MAX_BOARD_LEN * NUM_FEATURES_V1;

  const int NUM_FEATURES_V2 = 17;
  const int ROW_SIZE_V2 = NNPos::MAX_BOARD_LEN * NNPos::MAX_BOARD_LEN * NUM_FEATURES_V2;

  const int NUM_FEATURES_BIN_V3 = 22;
  const int NUM_FEATURES_GLOBAL_V3 = 14;

  Hash128 getHashV0(
    const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
    Player nextPlayer, float selfKomi
  );
  //Neural net input format that was pre-rules-implementation, doesn't handle superko and
  //doesn't get told about the rules
  void fillRowV0(
    const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
    Player nextPlayer, float selfKomi, int posLen, bool useNHWC, float* row
  );

  //Handles superko and works for tromp-taylor, but otherwise not all rules implemented
  Hash128 getHashV1(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer
  );
  void fillRowV1(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    int posLen, bool useNHWC, float* row
  );

  //Slightly more complete rules support, new ladder features, compressed some features
  Hash128 getHashV2(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer
  );
  void fillRowV2(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    int posLen, bool useNHWC, float* row
  );

  //Ongoing sandbox for full rules support for self play, not stable yet
  Hash128 getHashV3(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    double drawEquivalentWinsForWhite
  );
  void fillRowV3(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    double drawEquivalentWinsForWhite, int posLen, bool useNHWC, float* rowBin, float* rowGlobal
  );

}

struct NNOutput {
  Hash128 nnHash; //NNInputs - getHashV0 or getHashV1, etc.

  //Initially from the perspective of the player to move at the time of the eval, fixed up later in nnEval.cpp
  //to be the value from white's perspective.
  //These three are categorial probabilities for each outcome.
  float whiteWinProb;
  float whiteLossProb;
  float whiteNoResultProb;

  //The first two moments of the believed distribution of the expected score at the end of the game, from white's perspective.
  float whiteScoreMean;
  float whiteScoreMeanSq;

  //Indexed by pos rather than loc
  //Values in here will be set to negative for illegal moves, including superko
  float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

  int posLen;
  //If not NULL, then this contains a posLen*posLen-sized map of expected ownership on the board.
  float* whiteOwnerMap;

  NNOutput(); //Does NOT initialize values
  NNOutput(const NNOutput& other);
  ~NNOutput();

  NNOutput& operator=(const NNOutput&);
};

//Utility functions for computing the "scoreValue", the unscaled utility of various numbers of points, prior to multiplication by
//staticScoreUtilityFactor or dynamicScoreUtilityFactor (see searchparams.h)
namespace ScoreValue {
  //MUST BE CALLED AT PROGRAM START!
  void initTables();

  //The number of wins a game result should count as
  double whiteWinsOfWinner(Player winner, double drawEquivalentWinsForWhite);
  //The score difference that a game result should count as on average
  double whiteScoreDrawAdjust(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite, const BoardHistory& hist);

  //The unscaled utility of achieving a certain score difference
  double whiteScoreValueOfScoreSmooth(double finalWhiteMinusBlackScore, double center, double drawEquivalentWinsForWhite, const Board& b, const BoardHistory& hist);
  double whiteScoreValueOfScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, const Board& b);
  //Approximately invert whiteScoreValueOfScoreSmooth
  double approxWhiteScoreOfScoreValueSmooth(double scoreValue, double center, const Board& b);

  //Compute what the scoreMeanSq should be for a final game result
  //It is NOT simply the same as finalWhiteMinusBlackScore^2 because for integer komi we model it as a distribution where with the appropriate probability
  //you gain or lose 0.5 point to achieve the desired drawEquivalentWinsForWhite, so it actually has some variance.
  double whiteScoreMeanSqOfScoreGridded(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite, const BoardHistory& hist);

  //The expected unscaled utility of the final score difference, given the mean and stdev of the distribution of that difference,
  //assuming roughly a normal distribution.
  double expectedWhiteScoreValue(double whiteScoreMean, double whiteScoreStdev, double center, const Board& b);
}

#endif



