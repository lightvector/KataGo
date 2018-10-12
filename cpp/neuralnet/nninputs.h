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

  const int NUM_FEATURES_BIN_V3 = 20;
  const int NUM_FEATURES_FLOAT_V3 = 9;
  const int ROW_SIZE_BIN_V3 = NNPos::MAX_BOARD_LEN * NNPos::MAX_BOARD_LEN * NUM_FEATURES_BIN_V3;
  const int ROW_SIZE_FLOAT_V3 = NNPos::MAX_BOARD_LEN * NUM_FEATURES_FLOAT_V3;

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
    double drawUtilityForWhite
  );
  void fillRowV3(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    double drawUtilityForWhite, int posLen, bool useNHWC, bool* rowBin, float* rowFloat
  );


}

struct NNOutput {
  Hash128 nnHash; //NNInputs - getHashV0 or getHashV1

  //From the perspective of the player to move at the time of the eval
  float whiteValue;

  //Indexed by pos rather than loc
  //Values in here will be set to negative for illegal moves, including superko
  float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

  NNOutput(); //Does NOT initialize values
  NNOutput(const NNOutput& other);

  //Utility --------------------------------------------------------------------
  //The utility of having a particular winner
  static double whiteValueOfWinner(Player winner, double drawUtilityForWhite);
  //The utility of achieving a certain score difference
  static double whiteValueOfScore(double finalWhiteMinusBlackScore, double drawUtilityForWhite, const Board& b, const BoardHistory& hist);
};

#endif



