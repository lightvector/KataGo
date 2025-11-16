#include "../game/boardhistory.h"

using namespace std;

int BoardHistory::countDotsScoreWhiteMinusBlack(const Board& board, Color area[Board::MAX_ARR_SIZE]) const {
  assert(rules.isDots);

  const float whiteScore = whiteScoreIfGroundingAlive(board);
  Color groundingPlayer = C_EMPTY;
  if (whiteScore >= 0.0f) { // Don't use EPS comparison because in case of zero result, there is a draw and any player can ground
    groundingPlayer = C_WHITE;
  } else if (whiteScore < 0.0f) {
    groundingPlayer = C_BLACK;
  }
  return board.calculateOwnershipAndWhiteScore(area, groundingPlayer);
}

bool BoardHistory::winOrEffectiveDrawByGrounding(const Board& board, const Player pla, const bool considerDraw) const {
  assert(rules.isDots);

  const float whiteScore = whiteScoreIfGroundingAlive(board);
  return (considerDraw && Global::isZero(whiteScore)) ||
    (pla == P_WHITE && whiteScore > Global::FLOAT_EPS) ||
      (pla == P_BLACK && whiteScore < -Global::FLOAT_EPS);
}

float BoardHistory::whiteScoreIfGroundingAlive(const Board& board) const {
  assert(rules.isDots);

  const float fullWhiteScoreIfBlackGrounds =
       static_cast<float>(board.whiteScoreIfBlackGrounds) + whiteBonusScore + whiteHandicapBonusScore + rules.komi;
  if (fullWhiteScoreIfBlackGrounds < -Global::FLOAT_EPS) {
    // Black already won the game
    return fullWhiteScoreIfBlackGrounds;
  }

  const float fullBlackScoreIfWhiteGrounds =
       static_cast<float>(board.blackScoreIfWhiteGrounds) - whiteBonusScore - whiteHandicapBonusScore - rules.komi;
  if (fullBlackScoreIfWhiteGrounds < -Global::FLOAT_EPS) {
    // White already won the game
    return -fullBlackScoreIfWhiteGrounds;
  }

  if (Global::isZero(fullWhiteScoreIfBlackGrounds) && Global::isZero(fullBlackScoreIfWhiteGrounds)) {
    // Draw by grounding
    return 0.0f;
  }

  return std::numeric_limits<float>::quiet_NaN();
}