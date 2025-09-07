#include "../game/boardhistory.h"

using namespace std;

int BoardHistory::countDotsScoreWhiteMinusBlack(const Board& board, Color area[Board::MAX_ARR_SIZE]) const {
  assert(rules.isDots);

  const float whiteScore = whiteScoreIfGroundingAlive(board);
  Color groundingPlayer = C_EMPTY;
  if (whiteScore > 0.0f) {
    groundingPlayer = C_WHITE;
  } else if (whiteScore < 0.0f) {
    groundingPlayer = C_BLACK;
  }
  return board.calculateOwnershipAndWhiteScore(area, groundingPlayer);
}

bool BoardHistory::doesGroundingWinGame(const Board& board, const Player pla) const {
  float whiteScore;
  return doesGroundingWinGame(board, pla, whiteScore);
}

bool BoardHistory::doesGroundingWinGame(const Board& board, const Player pla, float& whiteScore) const {
  assert(rules.isDots);

  whiteScore = whiteScoreIfGroundingAlive(board);
  return pla == P_WHITE && whiteScore > 0.0f || pla == P_BLACK && whiteScore < 0.0f;
}

float BoardHistory::whiteScoreIfGroundingAlive(const Board& board) const {
  assert(rules.isDots);

  if (const float fullWhiteScoreIfBlackGrounds =
       static_cast<float>(board.whiteScoreIfBlackGrounds) + whiteBonusScore + whiteHandicapBonusScore + rules.komi;
     fullWhiteScoreIfBlackGrounds < 0.0f) {
    // Black already won the game
    return fullWhiteScoreIfBlackGrounds;
  }

  if (const float fullBlackScoreIfWhiteGrounds =
       static_cast<float>(board.blackScoreIfWhiteGrounds) - whiteBonusScore - whiteHandicapBonusScore - rules.komi;
     fullBlackScoreIfWhiteGrounds < 0.0f) {
    // White already won the game
    return -fullBlackScoreIfWhiteGrounds;
  }

  return 0.0f;
}