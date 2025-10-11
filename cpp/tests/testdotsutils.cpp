#include "testdotsutils.h"

using namespace std;

Move XYMove::toMove(const int x_size) const {
  return Move(Location::getLoc(x, y, x_size), player);
}

Board parseDotsFieldDefault(const string& input, const vector<XYMove>& extraMoves) {
  return parseDotsField(input, Rules::DEFAULT_DOTS.startPosIsRandom, Rules::DEFAULT_DOTS.multiStoneSuicideLegal, Rules::DEFAULT_DOTS.dotsCaptureEmptyBases, Rules::DEFAULT_DOTS.dotsFreeCapturedDots, extraMoves);
}

Board parseDotsField(const string& input, const bool startPosIsRandom, const bool suicide, const bool captureEmptyBases,
  const bool freeCapturedDots, const vector<XYMove>& extraMoves) {
  int currentXSize = 0;
  int xSize = -1;
  int ySize = 0;
  for(int i = 0; i <= input.length(); i++) {
    if(i == input.length() - 1 || input[i] == '\n') {
      if(i > 0) {
        if(xSize != -1) {
          assert(xSize == currentXSize);
        } else {
          xSize = currentXSize;
        }
        currentXSize = 0;
        ySize++;
      }
    } else {
      currentXSize++;
    }
  }

  Board result = Board::parseBoard(xSize, ySize, input, Rules(Rules::START_POS_EMPTY, startPosIsRandom, suicide, captureEmptyBases, freeCapturedDots));
  playXYMovesAssumeLegal(result, extraMoves);
  return result;
}

void playXYMovesAssumeLegal(Board& board, const vector<XYMove>& moves) {
  for(const XYMove& move : moves) {
    board.playMoveAssumeLegal(Location::getLoc(move.x, move.y, board.x_size), move.player);
  }
}