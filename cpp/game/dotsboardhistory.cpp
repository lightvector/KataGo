#include "../game/boardhistory.h"

using namespace std;

int BoardHistory::countDotsScoreWhiteMinusBlack(const Board& board, Color area[Board::MAX_ARR_SIZE]) const {
  assert(rules.isDots);

  const float whiteScore = whiteScoreIfGroundingAlive(board);
  Color groundingPlayer = C_EMPTY;
  if (whiteScore >= 0.0f) {
    groundingPlayer = C_WHITE;
  } else if (whiteScore < 0.0f) {
    groundingPlayer = C_BLACK;
  }
  return board.calculateOwnershipAndWhiteScore(area, groundingPlayer);
}

bool BoardHistory::winOrEffectiveDrawByGrounding(const Board& board, const Player pla, const bool considerDraw) const {
  assert(rules.isDots);

  const float whiteScore = whiteScoreIfGroundingAlive(board);
  return considerDraw && whiteScore == 0.0f || pla == P_WHITE && whiteScore > 0.0f || pla == P_BLACK && whiteScore < 0.0f;
}

float BoardHistory::whiteScoreIfGroundingAlive(const Board& board) const {
  assert(rules.isDots);

  const float fullWhiteScoreIfBlackGrounds =
       static_cast<float>(board.whiteScoreIfBlackGrounds) + whiteBonusScore + whiteHandicapBonusScore + rules.komi;
  if (fullWhiteScoreIfBlackGrounds < 0.0f) {
    // Black already won the game
    return fullWhiteScoreIfBlackGrounds;
  }

  const float fullBlackScoreIfWhiteGrounds =
       static_cast<float>(board.blackScoreIfWhiteGrounds) - whiteBonusScore - whiteHandicapBonusScore - rules.komi;
  if (fullBlackScoreIfWhiteGrounds < 0.0f) {
    // White already won the game
    return -fullBlackScoreIfWhiteGrounds;
  }

  if (fullWhiteScoreIfBlackGrounds == 0.0f && fullBlackScoreIfWhiteGrounds == 0.0f) {
    // Draw by grounding
    return 0.0f;
  }

  return std::numeric_limits<float>::quiet_NaN();
}