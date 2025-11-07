#include "testdotsutils.h"

using namespace std;

Board parseDotsFieldDefault(const string& input) {
  return parseDotsField(input, Rules::DEFAULT_DOTS.dotsCaptureEmptyBases, Rules::DEFAULT_DOTS.dotsFreeCapturedDots, vector<XYMove>());
}

Board parseDotsFieldDefault(const string& input, const vector<XYMove>& extraMoves) {
  return parseDotsField(input, Rules::DEFAULT_DOTS.dotsCaptureEmptyBases, Rules::DEFAULT_DOTS.dotsFreeCapturedDots, extraMoves);
}

Board parseDotsField(const string& input, const bool captureEmptyBases,
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

  Board result = Board::parseBoard(xSize, ySize, input, Rules(true, Rules::START_POS_EMPTY, captureEmptyBases, freeCapturedDots));
  for(const XYMove& extraMove : extraMoves) {
    result.playMoveAssumeLegal(Location::getLoc(extraMove.x, extraMove.y, result.x_size), extraMove.player);
  }
  return result;
}