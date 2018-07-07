#ifndef NNINPUTS_H
#define NNINPUTS_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../game/rules.h"

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
  const int NUM_FEATURES = 19;
  const int NUM_SYMMETRIES = 3;
  const int ROW_SIZE = NNPos::MAX_BOARD_LEN * NNPos::MAX_BOARD_LEN * NUM_FEATURES;

  Hash128 getHash(
    const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
    Player nextPlayer, float selfKomi
  );

  void fillRow(
    const Board& board, const vector<Move>& moveHistory, int moveHistoryLen,
    Player nextPlayer, float selfKomi, float* row
  );

}

#endif



