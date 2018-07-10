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
  const int NN_POLICY_SIZE = MAX_BOARD_AREA + 1;

  int getOffset(int bSize);
  int xyToPos(int x, int y, int offset);
  int locToPos(Loc loc, int bSize, int offset);
  Loc posToLoc(int pos, int bSize, int offset);
}

namespace NNInputs {
  const int NUM_SYMMETRY_BOOLS = 3;

  const int NUM_FEATURES_V0 = 19;
  const int ROW_SIZE_V0 = NNPos::MAX_BOARD_LEN * NNPos::MAX_BOARD_LEN * NUM_FEATURES_V0;

  const int NUM_FEATURES_V1 = 19;
  const int ROW_SIZE_V1 = NNPos::MAX_BOARD_LEN * NNPos::MAX_BOARD_LEN * NUM_FEATURES_V1;

  Hash128 getHashV0(
    const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
    Player nextPlayer, float selfKomi
  );
  //Neural net input format that was pre-rules-implementation, doesn't handle superko and
  //doesn't get told about the rules or pass-alive stones
  void fillRowV0(
    const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
    Player nextPlayer, float selfKomi, float* row
  );

  Hash128 getHashV1(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer
  );
  void fillRowV1(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer, float* row
  );

}

#endif



