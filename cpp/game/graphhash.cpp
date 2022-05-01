#include "../game/graphhash.h"

Hash128 GraphHash::getStateHash(const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite) {
  const Board& board = hist.getRecentBoard(0);
  Hash128 hash = BoardHistory::getSituationRulesAndKoHash(board, hist, nextPlayer, drawEquivalentWinsForWhite);

  // Fold in whether a pass ends this phase
  bool passEndsPhase = hist.passWouldEndPhase(board,nextPlayer);
  if(passEndsPhase)
    hash ^= Board::ZOBRIST_PASS_ENDS_PHASE;
  // Fold in whether the game is over or not
  if(hist.isGameFinished)
    hash ^= Board::ZOBRIST_GAME_IS_OVER;

  // Fold in consecutive pass count. Probably usually redundant with history tracking. Use some standard LCG constants.
  static constexpr uint64_t CONSECPASS_MULT0 = 2862933555777941757ULL;
  static constexpr uint64_t CONSECPASS_MULT1 = 3202034522624059733ULL;
  hash.hash0 += CONSECPASS_MULT0 * (uint64_t)hist.consecutiveEndingPasses;
  hash.hash1 += CONSECPASS_MULT1 * (uint64_t)hist.consecutiveEndingPasses;
  return hash;
}

Hash128 GraphHash::getGraphHash(Hash128 prevGraphHash, const BoardHistory& hist, Player nextPlayer, int repBound, double drawEquivalentWinsForWhite) {
  const Board& board = hist.getRecentBoard(0);
  Loc prevMoveLoc = hist.moveHistory.size() <= 0 ? Board::NULL_LOC : hist.moveHistory[hist.moveHistory.size()-1].loc;
  if(prevMoveLoc == Board::NULL_LOC || board.simpleRepetitionBoundGt(prevMoveLoc,repBound)) {
    return getStateHash(hist,nextPlayer,drawEquivalentWinsForWhite);
  }
  else {
    Hash128 newHash = prevGraphHash;
    newHash.hash0 = Hash::splitMix64(newHash.hash0 ^ newHash.hash1);
    newHash.hash1 = Hash::nasam(newHash.hash1) + newHash.hash0;
    Hash128 stateHash = getStateHash(hist,nextPlayer,drawEquivalentWinsForWhite);
    newHash.hash0 += stateHash.hash0;
    newHash.hash1 += stateHash.hash1;
    return newHash;
  }
}

Hash128 GraphHash::getGraphHashFromScratch(const BoardHistory& histOrig, Player nextPlayer, int repBound, double drawEquivalentWinsForWhite) {
  BoardHistory hist = histOrig.copyToInitial();
  Board board = hist.getRecentBoard(0);
  Hash128 graphHash = Hash128();

  for(size_t i = 0; i<histOrig.moveHistory.size(); i++) {
    graphHash = getGraphHash(graphHash, hist, histOrig.moveHistory[i].pla, repBound, drawEquivalentWinsForWhite);
    bool suc = hist.makeBoardMoveTolerant(board, histOrig.moveHistory[i].loc, histOrig.moveHistory[i].pla, histOrig.preventEncoreHistory[i]);
    assert(suc);
  }
  assert(
    BoardHistory::getSituationRulesAndKoHash(board, hist, nextPlayer, drawEquivalentWinsForWhite) ==
    BoardHistory::getSituationRulesAndKoHash(histOrig.getRecentBoard(0), histOrig, nextPlayer, drawEquivalentWinsForWhite)
  );

  graphHash = getGraphHash(graphHash, hist, nextPlayer, repBound, drawEquivalentWinsForWhite);
  return graphHash;
}

