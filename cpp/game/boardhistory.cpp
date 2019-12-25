#include "../game/boardhistory.h"

#include <algorithm>

using namespace std;

static Hash128 getKoHash(const Rules& rules, const Board& board, Player pla, int encorePhase, Hash128 koRecapBlockHash) {
  if(rules.koRule == Rules::KO_SITUATIONAL || rules.koRule == Rules::KO_SIMPLE || encorePhase > 0)
    return board.pos_hash ^ Board::ZOBRIST_PLAYER_HASH[pla] ^ koRecapBlockHash;
  else
    return board.pos_hash ^ koRecapBlockHash;
}
static Hash128 getKoHashAfterMoveNonEncore(const Rules& rules, Hash128 posHashAfterMove, Player pla) {
  if(rules.koRule == Rules::KO_SITUATIONAL || rules.koRule == Rules::KO_SIMPLE)
    return posHashAfterMove ^ Board::ZOBRIST_PLAYER_HASH[pla];
  else
    return posHashAfterMove;
}
// static Hash128 getKoHashAfterMove(const Rules& rules, Hash128 posHashAfterMove, Player pla, int encorePhase, Hash128 koRecapBlockHashAfterMove) {
//   if(rules.koRule == Rules::KO_SITUATIONAL || rules.koRule == Rules::KO_SIMPLE || encorePhase > 0)
//     return posHashAfterMove ^ Board::ZOBRIST_PLAYER_HASH[pla] ^ koRecapBlockHashAfterMove;
//   else
//     return posHashAfterMove ^ koRecapBlockHashAfterMove;
// }


BoardHistory::BoardHistory()
  :rules(),
   moveHistory(),koHashHistory(),
   firstTurnIdxWithKoHistory(0),
   initialBoard(),
   initialPla(P_BLACK),
   initialEncorePhase(0),
   initialTurnNumber(0),
   assumeMultipleStartingBlackMovesAreHandicap(false),
   whiteHasMoved(false),
   recentBoards(),
   currentRecentBoardIdx(0),
   presumedNextMovePla(P_BLACK),
   consecutiveEndingPasses(0),
   hashesBeforeBlackPass(),hashesBeforeWhitePass(),
   encorePhase(0),numTurnsThisPhase(0),
   koRecapBlockHash(),
   koCapturesInEncore(),
   whiteBonusScore(0.0f),
   whiteHandicapBonusScore(0.0f),
   hasButton(false),
   isPastNormalPhaseEnd(false),
   isGameFinished(false),winner(C_EMPTY),finalWhiteMinusBlackScore(0.0f),
   isScored(false),isNoResult(false),isResignation(false)
{
  std::fill(wasEverOccupiedOrPlayed, wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, false);
  std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
  std::fill(koRecapBlocked, koRecapBlocked+Board::MAX_ARR_SIZE, false);
  std::fill(secondEncoreStartColors, secondEncoreStartColors+Board::MAX_ARR_SIZE, C_EMPTY);
}

BoardHistory::~BoardHistory()
{}

BoardHistory::BoardHistory(const Board& board, Player pla, const Rules& r, int ePhase)
  :rules(r),
   moveHistory(),koHashHistory(),
   firstTurnIdxWithKoHistory(0),
   initialBoard(),
   initialPla(),
   initialEncorePhase(0),
   initialTurnNumber(0),
   assumeMultipleStartingBlackMovesAreHandicap(false),
   whiteHasMoved(false),
   recentBoards(),
   currentRecentBoardIdx(0),
   presumedNextMovePla(pla),
   consecutiveEndingPasses(0),
   hashesBeforeBlackPass(),hashesBeforeWhitePass(),
   encorePhase(0),numTurnsThisPhase(0),
   koRecapBlockHash(),
   koCapturesInEncore(),
   whiteBonusScore(0.0f),
   whiteHandicapBonusScore(0.0f),
   hasButton(false),
   isPastNormalPhaseEnd(false),
   isGameFinished(false),winner(C_EMPTY),finalWhiteMinusBlackScore(0.0f),
   isScored(false),isNoResult(false),isResignation(false)
{
  std::fill(wasEverOccupiedOrPlayed, wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, false);
  std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
  std::fill(koRecapBlocked, koRecapBlocked+Board::MAX_ARR_SIZE, false);
  std::fill(secondEncoreStartColors, secondEncoreStartColors+Board::MAX_ARR_SIZE, C_EMPTY);

  clear(board,pla,rules,ePhase);
}

BoardHistory::BoardHistory(const BoardHistory& other)
  :rules(other.rules),
   moveHistory(other.moveHistory),koHashHistory(other.koHashHistory),
   firstTurnIdxWithKoHistory(other.firstTurnIdxWithKoHistory),
   initialBoard(other.initialBoard),
   initialPla(other.initialPla),
   initialEncorePhase(other.initialEncorePhase),
   initialTurnNumber(other.initialTurnNumber),
   assumeMultipleStartingBlackMovesAreHandicap(other.assumeMultipleStartingBlackMovesAreHandicap),
   whiteHasMoved(other.whiteHasMoved),
   recentBoards(),
   currentRecentBoardIdx(other.currentRecentBoardIdx),
   presumedNextMovePla(other.presumedNextMovePla),
   consecutiveEndingPasses(other.consecutiveEndingPasses),
   hashesBeforeBlackPass(other.hashesBeforeBlackPass),hashesBeforeWhitePass(other.hashesBeforeWhitePass),
   encorePhase(other.encorePhase),numTurnsThisPhase(other.numTurnsThisPhase),
   koRecapBlockHash(other.koRecapBlockHash),
   koCapturesInEncore(other.koCapturesInEncore),
   whiteBonusScore(other.whiteBonusScore),
   whiteHandicapBonusScore(other.whiteHandicapBonusScore),
   hasButton(other.hasButton),
   isPastNormalPhaseEnd(other.isPastNormalPhaseEnd),
   isGameFinished(other.isGameFinished),winner(other.winner),finalWhiteMinusBlackScore(other.finalWhiteMinusBlackScore),
   isScored(other.isScored),isNoResult(other.isNoResult),isResignation(other.isResignation)
{
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  std::copy(other.wasEverOccupiedOrPlayed, other.wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, wasEverOccupiedOrPlayed);
  std::copy(other.superKoBanned, other.superKoBanned+Board::MAX_ARR_SIZE, superKoBanned);
  std::copy(other.koRecapBlocked, other.koRecapBlocked+Board::MAX_ARR_SIZE, koRecapBlocked);
  std::copy(other.secondEncoreStartColors, other.secondEncoreStartColors+Board::MAX_ARR_SIZE, secondEncoreStartColors);
}


BoardHistory& BoardHistory::operator=(const BoardHistory& other)
{
  if(this == &other)
    return *this;
  rules = other.rules;
  moveHistory = other.moveHistory;
  koHashHistory = other.koHashHistory;
  firstTurnIdxWithKoHistory = other.firstTurnIdxWithKoHistory;
  initialBoard = other.initialBoard;
  initialPla = other.initialPla;
  initialEncorePhase = other.initialEncorePhase;
  initialTurnNumber = other.initialTurnNumber;
  assumeMultipleStartingBlackMovesAreHandicap = other.assumeMultipleStartingBlackMovesAreHandicap;
  whiteHasMoved = other.whiteHasMoved;
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  presumedNextMovePla = other.presumedNextMovePla;
  std::copy(other.wasEverOccupiedOrPlayed, other.wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, wasEverOccupiedOrPlayed);
  std::copy(other.superKoBanned, other.superKoBanned+Board::MAX_ARR_SIZE, superKoBanned);
  consecutiveEndingPasses = other.consecutiveEndingPasses;
  hashesBeforeBlackPass = other.hashesBeforeBlackPass;
  hashesBeforeWhitePass = other.hashesBeforeWhitePass;
  encorePhase = other.encorePhase;
  numTurnsThisPhase = other.numTurnsThisPhase;
  std::copy(other.koRecapBlocked, other.koRecapBlocked+Board::MAX_ARR_SIZE, koRecapBlocked);
  koRecapBlockHash = other.koRecapBlockHash;
  koCapturesInEncore = other.koCapturesInEncore;
  std::copy(other.secondEncoreStartColors, other.secondEncoreStartColors+Board::MAX_ARR_SIZE, secondEncoreStartColors);
  whiteBonusScore = other.whiteBonusScore;
  whiteHandicapBonusScore = other.whiteHandicapBonusScore;
  hasButton = other.hasButton;
  isPastNormalPhaseEnd = other.isPastNormalPhaseEnd;
  isGameFinished = other.isGameFinished;
  winner = other.winner;
  finalWhiteMinusBlackScore = other.finalWhiteMinusBlackScore;
  isScored = other.isScored;
  isNoResult = other.isNoResult;
  isResignation = other.isResignation;

  return *this;
}

BoardHistory::BoardHistory(BoardHistory&& other) noexcept
 :rules(other.rules),
  moveHistory(std::move(other.moveHistory)),koHashHistory(std::move(other.koHashHistory)),
  firstTurnIdxWithKoHistory(other.firstTurnIdxWithKoHistory),
  initialBoard(other.initialBoard),
  initialPla(other.initialPla),
  initialEncorePhase(other.initialEncorePhase),
  initialTurnNumber(other.initialTurnNumber),
  assumeMultipleStartingBlackMovesAreHandicap(other.assumeMultipleStartingBlackMovesAreHandicap),
  whiteHasMoved(other.whiteHasMoved),
  recentBoards(),
  currentRecentBoardIdx(other.currentRecentBoardIdx),
  presumedNextMovePla(other.presumedNextMovePla),
  consecutiveEndingPasses(other.consecutiveEndingPasses),
  hashesBeforeBlackPass(std::move(other.hashesBeforeBlackPass)),hashesBeforeWhitePass(std::move(other.hashesBeforeWhitePass)),
  encorePhase(other.encorePhase),numTurnsThisPhase(other.numTurnsThisPhase),
  koRecapBlockHash(other.koRecapBlockHash),
  koCapturesInEncore(std::move(other.koCapturesInEncore)),
  whiteBonusScore(other.whiteBonusScore),
  whiteHandicapBonusScore(other.whiteHandicapBonusScore),
  hasButton(other.hasButton),
  isPastNormalPhaseEnd(other.isPastNormalPhaseEnd),
  isGameFinished(other.isGameFinished),winner(other.winner),finalWhiteMinusBlackScore(other.finalWhiteMinusBlackScore),
  isScored(other.isScored),isNoResult(other.isNoResult),isResignation(other.isResignation)
{
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  std::copy(other.wasEverOccupiedOrPlayed, other.wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, wasEverOccupiedOrPlayed);
  std::copy(other.superKoBanned, other.superKoBanned+Board::MAX_ARR_SIZE, superKoBanned);
  std::copy(other.koRecapBlocked, other.koRecapBlocked+Board::MAX_ARR_SIZE, koRecapBlocked);
  std::copy(other.secondEncoreStartColors, other.secondEncoreStartColors+Board::MAX_ARR_SIZE, secondEncoreStartColors);
}

BoardHistory& BoardHistory::operator=(BoardHistory&& other) noexcept
{
  rules = other.rules;
  moveHistory = std::move(other.moveHistory);
  koHashHistory = std::move(other.koHashHistory);
  firstTurnIdxWithKoHistory = other.firstTurnIdxWithKoHistory;
  initialBoard = other.initialBoard;
  initialPla = other.initialPla;
  initialEncorePhase = other.initialEncorePhase;
  initialTurnNumber = other.initialTurnNumber;
  assumeMultipleStartingBlackMovesAreHandicap = other.assumeMultipleStartingBlackMovesAreHandicap;
  whiteHasMoved = other.whiteHasMoved;
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  presumedNextMovePla = other.presumedNextMovePla;
  std::copy(other.wasEverOccupiedOrPlayed, other.wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, wasEverOccupiedOrPlayed);
  std::copy(other.superKoBanned, other.superKoBanned+Board::MAX_ARR_SIZE, superKoBanned);
  consecutiveEndingPasses = other.consecutiveEndingPasses;
  hashesBeforeBlackPass = std::move(other.hashesBeforeBlackPass);
  hashesBeforeWhitePass = std::move(other.hashesBeforeWhitePass);
  encorePhase = other.encorePhase;
  numTurnsThisPhase = other.numTurnsThisPhase;
  std::copy(other.koRecapBlocked, other.koRecapBlocked+Board::MAX_ARR_SIZE, koRecapBlocked);
  koRecapBlockHash = other.koRecapBlockHash;
  koCapturesInEncore = std::move(other.koCapturesInEncore);
  std::copy(other.secondEncoreStartColors, other.secondEncoreStartColors+Board::MAX_ARR_SIZE, secondEncoreStartColors);
  whiteBonusScore = other.whiteBonusScore;
  whiteHandicapBonusScore = other.whiteHandicapBonusScore;
  hasButton = other.hasButton;
  isPastNormalPhaseEnd = other.isPastNormalPhaseEnd;
  isGameFinished = other.isGameFinished;
  winner = other.winner;
  finalWhiteMinusBlackScore = other.finalWhiteMinusBlackScore;
  isScored = other.isScored;
  isNoResult = other.isNoResult;
  isResignation = other.isResignation;

  return *this;
}

void BoardHistory::clear(const Board& board, Player pla, const Rules& r, int ePhase) {
  rules = r;
  moveHistory.clear();
  koHashHistory.clear();
  firstTurnIdxWithKoHistory = 0;

  initialBoard = board;
  initialPla = pla;
  initialEncorePhase = ePhase;
  initialTurnNumber = 0;
  assumeMultipleStartingBlackMovesAreHandicap = false;
  whiteHasMoved = false;

  //This makes it so that if we ask for recent boards with a lookback beyond what we have a history for,
  //we simply return copies of the starting board.
  for(int i = 0; i<NUM_RECENT_BOARDS; i++)
    recentBoards[i] = board;
  currentRecentBoardIdx = 0;

  presumedNextMovePla = pla;

  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      wasEverOccupiedOrPlayed[loc] = (board.colors[loc] != C_EMPTY);
    }
  }

  std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
  consecutiveEndingPasses = 0;
  hashesBeforeBlackPass.clear();
  hashesBeforeWhitePass.clear();
  numTurnsThisPhase = 0;
  std::fill(koRecapBlocked, koRecapBlocked+Board::MAX_ARR_SIZE, false);
  koRecapBlockHash = Hash128();
  koCapturesInEncore.clear();
  whiteBonusScore = 0.0f;
  whiteHandicapBonusScore = 0.0f;
  hasButton = rules.hasButton && encorePhase == 0;
  isPastNormalPhaseEnd = false;
  isGameFinished = false;
  winner = C_EMPTY;
  finalWhiteMinusBlackScore = 0.0f;
  isScored = false;
  isNoResult = false;
  isResignation = false;

  //Handle encore phase
  encorePhase = ePhase;
  assert(encorePhase >= 0 && encorePhase <= 2);
  if(encorePhase > 0)
    assert(rules.scoringRule == Rules::SCORING_TERRITORY);
  //Update the few parameters that depend on encore
  if(encorePhase == 2)
    std::copy(board.colors, board.colors+Board::MAX_ARR_SIZE, secondEncoreStartColors);
  else
    std::fill(secondEncoreStartColors, secondEncoreStartColors+Board::MAX_ARR_SIZE, C_EMPTY);

  //Push hash for the new board state
  koHashHistory.push_back(getKoHash(rules,board,pla,encorePhase,koRecapBlockHash));

  if(rules.scoringRule == Rules::SCORING_TERRITORY) {
    //Chill 1 point for every move played
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == P_BLACK)
          whiteBonusScore += 1.0f;
        else if(board.colors[loc] == P_WHITE)
          whiteBonusScore -= 1.0f;
      }
    }
    //If white actually played extra moves that got captured so we don't see them,
    //then chill for those too
    int netWhiteCaptures = board.numWhiteCaptures - board.numBlackCaptures;
    whiteBonusScore -= (float)netWhiteCaptures;
  }
  whiteHandicapBonusScore = computeWhiteHandicapBonus();
}

void BoardHistory::setInitialTurnNumber(int n) {
  initialTurnNumber = n;
}

void BoardHistory::setAssumeMultipleStartingBlackMovesAreHandicap(bool b) {
  assumeMultipleStartingBlackMovesAreHandicap = b;
  whiteHandicapBonusScore = computeWhiteHandicapBonus();
}

static int numHandicapStonesOnBoardHelper(const Board& board, int blackNonPassTurnsToStart) {
  int startBoardNumBlackStones = 0;
  int startBoardNumWhiteStones = 0;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(board.colors[loc] == C_BLACK)
        startBoardNumBlackStones += 1;
      else if(board.colors[loc] == C_WHITE)
        startBoardNumWhiteStones += 1;
    }
  }
  //If we set up in a nontrivial position, then consider it a non-handicap game.
  if(startBoardNumWhiteStones != 0)
    return 0;
  //Add in additional counted stones
  int blackTurnAdvantage = startBoardNumBlackStones + blackNonPassTurnsToStart;

  //If there was only one black move/stone to start, then it was a regular game
  if(blackTurnAdvantage <= 1)
    return 0;
  return blackTurnAdvantage;
}

int BoardHistory::numHandicapStonesOnBoard(const Board& b) {
  return numHandicapStonesOnBoardHelper(b,0);
}

int BoardHistory::computeNumHandicapStones() const {
  int blackNonPassTurnsToStart = 0;
  if(assumeMultipleStartingBlackMovesAreHandicap) {
    //Find the length of the initial sequence of black moves - treat a string of consecutive black
    //moves at the start of the game as "handicap"
    //This is necessary because when loading sgfs or on some servers, (particularly with free placement)
    //handicap is implemented by having black just make a bunch of moves in a row.
    //But if white makes multiple moves in a row after that, then the plays are probably not handicap, someone's setting
    //up a problem position by having black play all moves in a row then white play all moves in a row.
    for(int i = 0; i<moveHistory.size(); i++) {
      Loc moveLoc = moveHistory[i].loc;
      Player movePla = moveHistory[i].pla;
      if(movePla != P_BLACK) {
        //Two white moves in a row? Re-set count.
        if(i+1 < moveHistory.size() && moveHistory[i+1].pla != P_BLACK)
          blackNonPassTurnsToStart = 0;
        break;
      }
      if(moveLoc != Board::PASS_LOC && moveLoc != Board::NULL_LOC)
        blackNonPassTurnsToStart += 1;
    }
  }
  return numHandicapStonesOnBoardHelper(initialBoard,blackNonPassTurnsToStart);
}

int BoardHistory::computeWhiteHandicapBonus() const {
  if(rules.whiteHandicapBonusRule == Rules::WHB_ZERO)
    return 0;
  else {
    int numHandicapStones = computeNumHandicapStones();
    if(rules.whiteHandicapBonusRule == Rules::WHB_N)
      return numHandicapStones;
    else if(rules.whiteHandicapBonusRule == Rules::WHB_N_MINUS_ONE)
      return (numHandicapStones > 1) ? numHandicapStones-1 : 0;
    else
      ASSERT_UNREACHABLE;
    return 0;
  }
}

void BoardHistory::printDebugInfo(ostream& out, const Board& board) const {
  out << board << endl;
  out << "Initial pla " << PlayerIO::playerToString(initialPla) << endl;
  out << "Encore phase " << encorePhase << endl;
  out << "Turns this phase " << numTurnsThisPhase << endl;
  out << "Rules " << rules << endl;
  out << "Ko recap block hash " << koRecapBlockHash << endl;
  out << "White bonus score " << whiteBonusScore << endl;
  out << "White handicap bonus score " << whiteHandicapBonusScore << endl;
  out << "Has button " << hasButton << endl;
  out << "Presumed next pla " << PlayerIO::playerToString(presumedNextMovePla) << endl;
  out << "Past normal phase end " << isPastNormalPhaseEnd << endl;
  out << "Game result " << isGameFinished << " " << PlayerIO::playerToString(winner) << " "
      << finalWhiteMinusBlackScore << " " << isScored << " " << isNoResult << " " << isResignation << endl;
  out << "Last moves ";
  for(int i = 0; i<moveHistory.size(); i++)
    out << Location::toString(moveHistory[i].loc,board) << " ";
  out << endl;
  assert(firstTurnIdxWithKoHistory + koHashHistory.size() == moveHistory.size() + 1);
}


const Board& BoardHistory::getRecentBoard(int numMovesAgo) const {
  assert(numMovesAgo >= 0 && numMovesAgo < NUM_RECENT_BOARDS);
  int idx = (currentRecentBoardIdx - numMovesAgo + NUM_RECENT_BOARDS) % NUM_RECENT_BOARDS;
  return recentBoards[idx];
}


void BoardHistory::setKomi(float newKomi) {
  float oldKomi = rules.komi;
  rules.komi = newKomi;

  //Recompute the game result due to the new komi
  if(isGameFinished && isScored)
    setFinalScoreAndWinner(finalWhiteMinusBlackScore - oldKomi + newKomi);
}


//If rootKoHashTable is provided, will take advantage of rootKoHashTable rather than search within the first
//rootKoHashTable->size() moves of koHashHistory.
//ALSO counts the most recent ko hash!
bool BoardHistory::koHashOccursInHistory(Hash128 koHash, const KoHashTable* rootKoHashTable) const {
  size_t start = 0;
  size_t koHashHistorySize = koHashHistory.size();
  if(rootKoHashTable != NULL &&
     firstTurnIdxWithKoHistory == rootKoHashTable->firstTurnIdxWithKoHistory
  ) {
    size_t tableSize = rootKoHashTable->size();
    assert(firstTurnIdxWithKoHistory + koHashHistory.size() == moveHistory.size() + 1);
    assert(tableSize <= koHashHistorySize);
    if(rootKoHashTable->containsHash(koHash))
      return true;
    start = tableSize;
  }
  for(size_t i = start; i < koHashHistorySize; i++)
    if(koHashHistory[i] == koHash)
      return true;
  return false;
}

//If rootKoHashTable is provided, will take advantage of rootKoHashTable rather than search within the first
//rootKoHashTable->size() moves of koHashHistory.
//ALSO counts the most recent ko hash!
int BoardHistory::numberOfKoHashOccurrencesInHistory(Hash128 koHash, const KoHashTable* rootKoHashTable) const {
  int count = 0;
  size_t start = 0;
  size_t koHashHistorySize = koHashHistory.size();
  if(rootKoHashTable != NULL &&
     firstTurnIdxWithKoHistory == rootKoHashTable->firstTurnIdxWithKoHistory
  ) {
    size_t tableSize = rootKoHashTable->size();
    assert(firstTurnIdxWithKoHistory + koHashHistory.size() == moveHistory.size() + 1);
    assert(tableSize <= koHashHistorySize);
    count += rootKoHashTable->numberOfOccurrencesOfHash(koHash);
    start = tableSize;
  }
  for(size_t i = start; i < koHashHistorySize; i++)
    if(koHashHistory[i] == koHash)
      count++;
  return count;
}

float BoardHistory::whiteKomiAdjustmentForDraws(double drawEquivalentWinsForWhite) const {
  //We fold the draw utility into the komi, for input into things like the neural net.
  //Basically we model it as if the final score were jittered by a uniform draw from [-0.5,0.5].
  //E.g. if komi from self perspective is 7 and a draw counts as 0.75 wins and 0.25 losses,
  //then komi input should be as if it was 7.25, which in a jigo game when jittered by 0.5 gives white 75% wins and 25% losses.
  bool komiIsInteger = ((int)rules.komi == rules.komi);
  float drawAdjustment = !komiIsInteger ? 0.0f : (float)(drawEquivalentWinsForWhite - 0.5);
  return drawAdjustment;
}

float BoardHistory::currentSelfKomi(Player pla, double drawEquivalentWinsForWhite) const {
  float whiteKomiAdjusted = whiteBonusScore + whiteHandicapBonusScore + rules.komi + whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite);

  if(pla == P_WHITE)
    return whiteKomiAdjusted;
  else if(pla == P_BLACK)
    return -whiteKomiAdjusted;
  else {
    assert(false);
    return 0.0f;
  }
}

int BoardHistory::countAreaScoreWhiteMinusBlack(const Board& board, Color area[Board::MAX_ARR_SIZE]) const {
  int score = 0;
  if(rules.taxRule == Rules::TAX_NONE) {
    bool nonPassAliveStones = true;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = true;
    board.calculateArea(
      area,
      nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules.multiStoneSuicideLegal
    );
  }
  else if(rules.taxRule == Rules::TAX_SEKI || rules.taxRule == Rules::TAX_ALL) {
    bool keepTerritories = false;
    bool keepStones = true;
    int whiteMinusBlackIndependentLifeRegionCount = 0;
    board.calculateIndependentLifeArea(
      area,whiteMinusBlackIndependentLifeRegionCount,
      keepTerritories,
      keepStones,
      rules.multiStoneSuicideLegal
    );
    if(rules.taxRule == Rules::TAX_ALL)
      score -= 2 * whiteMinusBlackIndependentLifeRegionCount;
  }
  else
    ASSERT_UNREACHABLE;

  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(area[loc] == C_WHITE)
        score += 1;
      else if(area[loc] == C_BLACK)
        score -= 1;
    }
  }

  return score;
}

//ALSO makes area color the points that were not pass alive but were scored for a side.
int BoardHistory::countTerritoryAreaScoreWhiteMinusBlack(const Board& board, Color area[Board::MAX_ARR_SIZE]) const {
  int score = 0;
  bool keepTerritories;
  bool keepStones;
  if(rules.taxRule == Rules::TAX_NONE) {
    keepTerritories = true;
    keepStones = false;
  }
  else if(rules.taxRule == Rules::TAX_SEKI || rules.taxRule == Rules::TAX_ALL) {
    keepTerritories = false;
    keepStones = false;
  }
  else
    ASSERT_UNREACHABLE;

  int whiteMinusBlackIndependentLifeRegionCount = 0;
  board.calculateIndependentLifeArea(
    area,whiteMinusBlackIndependentLifeRegionCount,
    keepTerritories,
    keepStones,
    rules.multiStoneSuicideLegal
  );

  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(area[loc] == C_WHITE)
        score += 1;
      else if(area[loc] == C_BLACK)
        score -= 1;
      else {
        //Checking encorePhase < 2 allows us to get the correct score if we directly end the game before the second
        //encore such that we never actually fill secondEncoreStartColors. This matters for premature termination
        //of the game like ending due to a move limit and such.
        if(board.colors[loc] == C_WHITE && (encorePhase < 2 || secondEncoreStartColors[loc] == C_WHITE)) {
          score += 1;
          area[loc] = C_WHITE;
        }
        if(board.colors[loc] == C_BLACK && (encorePhase < 2 || secondEncoreStartColors[loc] == C_BLACK)) {
          score -= 1;
          area[loc] = C_BLACK;
        }
      }
    }
  }
  if(rules.taxRule == Rules::TAX_ALL)
    score -= 2 * whiteMinusBlackIndependentLifeRegionCount;
  return score;
}

void BoardHistory::setFinalScoreAndWinner(float score) {
  finalWhiteMinusBlackScore = score;
  if(finalWhiteMinusBlackScore > 0.0f)
    winner = C_WHITE;
  else if(finalWhiteMinusBlackScore < 0.0f)
    winner = C_BLACK;
  else
    winner = C_EMPTY;
}

void BoardHistory::getAreaNow(const Board& board, Color area[Board::MAX_ARR_SIZE]) const {
  if(rules.scoringRule == Rules::SCORING_AREA)
    countAreaScoreWhiteMinusBlack(board,area);
  else if(rules.scoringRule == Rules::SCORING_TERRITORY)
    countTerritoryAreaScoreWhiteMinusBlack(board,area);
  else
    ASSERT_UNREACHABLE;
}

void BoardHistory::endAndScoreGameNow(const Board& board, Color area[Board::MAX_ARR_SIZE]) {
  int boardScore;
  if(rules.scoringRule == Rules::SCORING_AREA)
    boardScore = countAreaScoreWhiteMinusBlack(board,area);
  else if(rules.scoringRule == Rules::SCORING_TERRITORY)
    boardScore = countTerritoryAreaScoreWhiteMinusBlack(board,area);
  else
    ASSERT_UNREACHABLE;

  if(hasButton) {
    hasButton = false;
    whiteBonusScore += (presumedNextMovePla == P_WHITE ? 0.5f : -0.5f);
  }

  setFinalScoreAndWinner(boardScore + whiteBonusScore + whiteHandicapBonusScore + rules.komi);
  isScored = true;
  isNoResult = false;
  isResignation = false;
  isGameFinished = true;
  isPastNormalPhaseEnd = false;
}

void BoardHistory::endAndScoreGameNow(const Board& board) {
  Color area[Board::MAX_ARR_SIZE];
  endAndScoreGameNow(board,area);
}

void BoardHistory::endGameIfAllPassAlive(const Board& board) {
  int boardScore = 0;
  bool nonPassAliveStones = false;
  bool safeBigTerritories = false;
  bool unsafeBigTerritories = false;
  Color area[Board::MAX_ARR_SIZE];
  board.calculateArea(
    area,
    nonPassAliveStones, safeBigTerritories, unsafeBigTerritories, rules.multiStoneSuicideLegal
  );

  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(area[loc] == C_WHITE)
        boardScore += 1;
      else if(area[loc] == C_BLACK)
        boardScore -= 1;
      else
        return;
    }
  }

  //In the case that we have a group tax, rescore normally to actually count the group tax
  if(rules.taxRule == Rules::TAX_ALL)
    endAndScoreGameNow(board);
  else {
    if(hasButton) {
      hasButton = false;
      whiteBonusScore += (presumedNextMovePla == P_WHITE ? 0.5f : -0.5f);
    }
    setFinalScoreAndWinner(boardScore + whiteBonusScore + whiteHandicapBonusScore + rules.komi);
    isScored = true;
    isNoResult = false;
    isResignation = false;
    isGameFinished = true;
    isPastNormalPhaseEnd = false;
  }
}

void BoardHistory::setWinnerByResignation(Player pla) {
  isGameFinished = true;
  isPastNormalPhaseEnd = false;
  isScored = false;
  isNoResult = false;
  isResignation = true;
  winner = pla;
  finalWhiteMinusBlackScore = 0.0f;
}

void BoardHistory::setKoRecapBlocked(Loc loc, bool b) {
  if(koRecapBlocked[loc] != b) {
    koRecapBlocked[loc] = b;
    //We used to have per-color marks, so the zobrist was for both. Just combine them.
    koRecapBlockHash ^= Board::ZOBRIST_KO_MARK_HASH[loc][C_BLACK] ^ Board::ZOBRIST_KO_MARK_HASH[loc][C_WHITE];
  }
}

bool BoardHistory::isLegal(const Board& board, Loc moveLoc, Player movePla) const {
  //Ko-moves in the encore that are recapture blocked are interpreted as pass-for-ko, so they are legal
  if(encorePhase > 0 && moveLoc >= 0 && moveLoc < Board::MAX_ARR_SIZE && moveLoc != Board::PASS_LOC) {
    Loc koCaptureLoc = board.getKoCaptureLoc(moveLoc,movePla);
    if(koCaptureLoc != Board::NULL_LOC && koRecapBlocked[koCaptureLoc] && board.colors[koCaptureLoc] == getOpp(movePla))
      return true;
  }

  if(!board.isLegal(moveLoc,movePla,rules.multiStoneSuicideLegal))
    return false;
  if(superKoBanned[moveLoc])
    return false;

  return true;
}

bool BoardHistory::isPassForKo(const Board& board, Loc moveLoc, Player movePla) const {
  if(encorePhase > 0 && moveLoc >= 0 && moveLoc < Board::MAX_ARR_SIZE && moveLoc != Board::PASS_LOC) {
    Loc koCaptureLoc = board.getKoCaptureLoc(moveLoc,movePla);
    if(koCaptureLoc != Board::NULL_LOC && koRecapBlocked[koCaptureLoc] && board.colors[koCaptureLoc] == getOpp(movePla))
      return true;
  }
  return false;
}

//Return the number of consecutive game-ending passes there would be if a pass was made
int BoardHistory::newConsecutiveEndingPassesAfterPass() const {
  int newConsecutiveEndingPasses = consecutiveEndingPasses;
  if(encorePhase > 0)
    newConsecutiveEndingPasses++;
  else {
    switch(rules.koRule) {
    case Rules::KO_SIMPLE:
    case Rules::KO_POSITIONAL:
    case Rules::KO_SITUATIONAL:
      newConsecutiveEndingPasses++;
      break;
    case Rules::KO_SPIGHT:
      newConsecutiveEndingPasses = 0;
      break;
    default:
      ASSERT_UNREACHABLE;
      break;
    }
  }
  return newConsecutiveEndingPasses;
}

//Returns true if the rules of the game specify that passes should clear history for the purposes
//of ko rules checking and for no-result infinite cycles. Also implies that the phase will end
//spightlight - i.e. upon a pass where the same player has passed in the same situation before.
bool BoardHistory::phaseHasSpightlikeEndingAndPassHistoryClearing() const {
  return encorePhase > 0
    || rules.koRule == Rules::KO_SIMPLE
    || rules.koRule == Rules::KO_SPIGHT;
}

//Returns true if this move would be a pass that causes spight-style ending of the phase
//(i.e. ending that ignores the number of consecutive passes)
bool BoardHistory::wouldBeSpightlikeEndingPass(Player movePla, Hash128 koHashBeforeMove) const {
  if(phaseHasSpightlikeEndingAndPassHistoryClearing()) {
    if(movePla == P_BLACK && std::find(hashesBeforeBlackPass.begin(), hashesBeforeBlackPass.end(), koHashBeforeMove) != hashesBeforeBlackPass.end())
      return true;
    if(movePla == P_WHITE && std::find(hashesBeforeWhitePass.begin(), hashesBeforeWhitePass.end(), koHashBeforeMove) != hashesBeforeWhitePass.end())
      return true;
  }
  return false;
}

bool BoardHistory::passWouldEndPhase(const Board& board, Player movePla) const {
  Hash128 koHashBeforeMove = getKoHash(rules, board, movePla, encorePhase, koRecapBlockHash);
  if(newConsecutiveEndingPassesAfterPass() >= 2 ||
     wouldBeSpightlikeEndingPass(movePla,koHashBeforeMove))
    return true;
  return false;
}

bool BoardHistory::passWouldEndGame(const Board& board, Player movePla) const {
  return passWouldEndPhase(board,movePla) && (
    rules.scoringRule == Rules::SCORING_AREA
    || (rules.scoringRule == Rules::SCORING_TERRITORY && encorePhase >= 2)
  );
}

void BoardHistory::makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla, const KoHashTable* rootKoHashTable) {
  makeBoardMoveAssumeLegal(board,moveLoc,movePla,rootKoHashTable,false);
}

void BoardHistory::makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla, const KoHashTable* rootKoHashTable, bool preventEncore) {
  Hash128 posHashBeforeMove = board.pos_hash;

  //If somehow we're making a move after the game was ended, just clear those values and continue
  isGameFinished = false;
  isPastNormalPhaseEnd = false;
  winner = C_EMPTY;
  finalWhiteMinusBlackScore = 0.0f;
  isScored = false;
  isNoResult = false;
  isResignation = false;

  //Update consecutiveEndingPasses and button
  bool isSpightlikeEndingPass = false;
  if(moveLoc != Board::PASS_LOC)
    consecutiveEndingPasses = 0;
  else if(hasButton) {
    assert(encorePhase == 0 && rules.hasButton);
    hasButton = false;
    whiteBonusScore += (movePla == P_WHITE ? 0.5f : -0.5f);
    consecutiveEndingPasses = 0;
    //Taking the button clears all ko hash histories (this is equivalent to not clearing them and treating buttonless
    //state as different than buttonful state)
    hashesBeforeBlackPass.clear();
    hashesBeforeWhitePass.clear();
    koHashHistory.clear();
    //The first turn idx with history will be the one RESULTING from this move.
    firstTurnIdxWithKoHistory = moveHistory.size()+1;
  }
  else {
    //Passes clear ko history in the main phase with spight ko rules and in the encore
    //This lifts bans in spight ko rules and lifts 3-fold-repetition checking in the encore for no-resultifying infinite cycles
    //They also clear in simple ko rules for the purpose of no-resulting long cycles. Long cycles with passes do not no-result.
    if(phaseHasSpightlikeEndingAndPassHistoryClearing()) {
      koHashHistory.clear();
      //The first turn idx with history will be the one RESULTING from this move.
      firstTurnIdxWithKoHistory = moveHistory.size()+1;
      //Does not clear hashesBeforeBlackPass or hashesBeforeWhitePass. Passes lift ko bans, but
      //still repeated positions after pass end the game or phase, which these arrays are used to check.
    }

    Hash128 koHashBeforeThisMove = getKoHash(rules,board,movePla,encorePhase,koRecapBlockHash);
    consecutiveEndingPasses = newConsecutiveEndingPassesAfterPass();
    //Check if we have a game-ending pass BEFORE updating hashesBeforeBlackPass and hashesBeforeWhitePass
    isSpightlikeEndingPass = wouldBeSpightlikeEndingPass(movePla,koHashBeforeThisMove);

    //Update hashesBeforeBlackPass and hashesBeforeWhitePass
    if(movePla == P_BLACK)
      hashesBeforeBlackPass.push_back(koHashBeforeThisMove);
    else if(movePla == P_WHITE)
      hashesBeforeWhitePass.push_back(koHashBeforeThisMove);
    else
      ASSERT_UNREACHABLE;
  }

  //Handle pass-for-ko moves in the encore. Pass for ko lifts a ko recapture block and does nothing else.
  bool wasPassForKo = false;
  if(encorePhase > 0 && moveLoc != Board::PASS_LOC) {
    Loc koCaptureLoc = board.getKoCaptureLoc(moveLoc,movePla);
    if(koCaptureLoc != Board::NULL_LOC && koRecapBlocked[koCaptureLoc] && board.colors[koCaptureLoc] == getOpp(movePla)) {
      setKoRecapBlocked(koCaptureLoc,false);
      wasPassForKo = true;
      //Clear simple ko loc just in case
      //Since we aren't otherwise touching the board, from the board's perspective a player will be moving twice in a row.
      board.clearSimpleKoLoc();
    }
  }
  //Otherwise handle regular moves
  if(!wasPassForKo) {
    board.playMoveAssumeLegal(moveLoc,movePla);

    if(encorePhase > 0) {
      //Update ko recapture blocks and record that this was a ko capture
      if(board.ko_loc != Board::NULL_LOC) {
        setKoRecapBlocked(moveLoc,true);
        koCapturesInEncore.push_back(EncoreKoCapture(posHashBeforeMove,moveLoc,movePla));
        //Clear simple ko loc now that we've absorbed the ko loc information into the korecap blocks
        //Once we have that, the simple ko loc plays no further role in game state or legality
        board.clearSimpleKoLoc();
      }
      //Unmark all ko recap blocks not on stones
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          if(board.colors[loc] == C_EMPTY && koRecapBlocked[loc])
            setKoRecapBlocked(loc,false);
        }
      }
    }
  }

  //Update recent boards
  currentRecentBoardIdx = (currentRecentBoardIdx + 1) % NUM_RECENT_BOARDS;
  recentBoards[currentRecentBoardIdx] = board;

  Hash128 koHashAfterThisMove = getKoHash(rules,board,getOpp(movePla),encorePhase,koRecapBlockHash);
  koHashHistory.push_back(koHashAfterThisMove);
  moveHistory.push_back(Move(moveLoc,movePla));
  numTurnsThisPhase += 1;
  presumedNextMovePla = getOpp(movePla);

  if(moveLoc != Board::PASS_LOC)
    wasEverOccupiedOrPlayed[moveLoc] = true;

  //Mark all locations that are superko-illegal for the next player, by iterating and testing each point.
  Player nextPla = getOpp(movePla);
  if(encorePhase <= 0 && rules.koRule != Rules::KO_SIMPLE) {
    assert(koRecapBlockHash == Hash128());
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        //Cannot be superko banned if it's not a pseudolegal move in the first place, or we would already ban the move under simple ko.
        if(board.colors[loc] != C_EMPTY || board.isIllegalSuicide(loc,nextPla,rules.multiStoneSuicideLegal) || loc == board.ko_loc)
          superKoBanned[loc] = false;
        //Also cannot be superko banned if a stone was never there or played there before AND the move is not suicide, because that means
        //the move results in a new stone there and if no stone was ever there in the past the it must be a new position.
        else if(!wasEverOccupiedOrPlayed[loc] && !board.isSuicide(loc,nextPla))
          superKoBanned[loc] = false;
        else {
          Hash128 posHashAfterMove = board.getPosHashAfterMove(loc,nextPla);
          Hash128 koHashAfterMove = getKoHashAfterMoveNonEncore(rules, posHashAfterMove, getOpp(nextPla));
          superKoBanned[loc] = koHashOccursInHistory(koHashAfterMove,rootKoHashTable);
        }
      }
    }
  }
  else if(encorePhase > 0) {
    //During the encore, only one capture of each ko in a given position by a given player
    std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
    for(size_t i = 0; i<koCapturesInEncore.size(); i++) {
      const EncoreKoCapture& ekc = koCapturesInEncore[i];
      if(ekc.posHashBeforeMove == board.pos_hash && ekc.movePla == nextPla)
        superKoBanned[ekc.moveLoc] = true;
    }
  }

  //Territory scoring - chill 1 point per move in main phase and first encore
  if(rules.scoringRule == Rules::SCORING_TERRITORY && encorePhase <= 1 && moveLoc != Board::PASS_LOC && !wasPassForKo) {
    if(movePla == P_BLACK)
      whiteBonusScore += 1.0f;
    else if(movePla == P_WHITE)
      whiteBonusScore -= 1.0f;
    else
      ASSERT_UNREACHABLE;
  }

  //Handicap bonus score
  if(movePla == P_WHITE)
    whiteHasMoved = true;
  if(assumeMultipleStartingBlackMovesAreHandicap && !whiteHasMoved && movePla == P_BLACK && rules.whiteHandicapBonusRule != Rules::WHB_ZERO) {
    whiteHandicapBonusScore = computeWhiteHandicapBonus();
  }

  //Phase transitions and game end
  if(consecutiveEndingPasses >= 2 || isSpightlikeEndingPass) {
    if(rules.scoringRule == Rules::SCORING_AREA) {
      assert(encorePhase <= 0);
      endAndScoreGameNow(board);
    }
    else if(rules.scoringRule == Rules::SCORING_TERRITORY) {
      if(encorePhase >= 2)
        endAndScoreGameNow(board);
      else {
        if(preventEncore) {
          isPastNormalPhaseEnd = true;
        }
        else {
          encorePhase += 1;
          numTurnsThisPhase = 0;
          if(encorePhase == 2)
            std::copy(board.colors, board.colors+Board::MAX_ARR_SIZE, secondEncoreStartColors);

          std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
          consecutiveEndingPasses = 0;
          hashesBeforeBlackPass.clear();
          hashesBeforeWhitePass.clear();
          std::fill(koRecapBlocked, koRecapBlocked+Board::MAX_ARR_SIZE, false);
          koRecapBlockHash = Hash128();
          koCapturesInEncore.clear();

          koHashHistory.clear();
          koHashHistory.push_back(getKoHash(rules,board,getOpp(movePla),encorePhase,koRecapBlockHash));
          //The first ko hash history is the one for the move we JUST appended to the move history earlier.
          firstTurnIdxWithKoHistory = moveHistory.size();
        }
      }
    }
    else
      ASSERT_UNREACHABLE;
  }

  //Break long cycles with no-result
  if(moveLoc != Board::PASS_LOC && (encorePhase > 0 || rules.koRule == Rules::KO_SIMPLE)) {
    if(numberOfKoHashOccurrencesInHistory(koHashHistory[koHashHistory.size()-1], rootKoHashTable) >= 3) {
      isNoResult = true;
      isGameFinished = true;
    }
  }

}

KoHashTable::KoHashTable()
  :koHashHistorySortedByLowBits(),
   firstTurnIdxWithKoHistory(0)
{
  idxTable = new uint32_t[TABLE_SIZE];
}
KoHashTable::~KoHashTable() {
  delete[] idxTable;
}

size_t KoHashTable::size() const {
  return koHashHistorySortedByLowBits.size();
}

void KoHashTable::recompute(const BoardHistory& history) {
  koHashHistorySortedByLowBits = history.koHashHistory;
  firstTurnIdxWithKoHistory = history.firstTurnIdxWithKoHistory;

  auto cmpFirstByLowBits = [](const Hash128& a, const Hash128& b) {
    if((a.hash0 & TABLE_MASK) < (b.hash0 & TABLE_MASK))
      return true;
    if((a.hash0 & TABLE_MASK) > (b.hash0 & TABLE_MASK))
      return false;
    return a < b;
  };

  std::sort(koHashHistorySortedByLowBits.begin(),koHashHistorySortedByLowBits.end(),cmpFirstByLowBits);

  //Just in case, since we're using 32 bits for indices.
  if(koHashHistorySortedByLowBits.size() > 1000000000)
    throw StringError("Board history length longer than 1000000000, not supported");
  uint32_t size = (uint32_t)koHashHistorySortedByLowBits.size();

  uint32_t idx = 0;
  for(uint32_t bits = 0; bits<TABLE_SIZE; bits++) {
    while(idx < size && ((koHashHistorySortedByLowBits[idx].hash0 & TABLE_MASK) < bits))
      idx++;
    idxTable[bits] = idx;
  }
}

bool KoHashTable::containsHash(Hash128 hash) const {
  uint32_t bits = hash.hash0 & TABLE_MASK;
  size_t idx = idxTable[bits];
  size_t size = koHashHistorySortedByLowBits.size();
  while(idx < size && ((koHashHistorySortedByLowBits[idx].hash0 & TABLE_MASK) == bits)) {
    if(hash == koHashHistorySortedByLowBits[idx])
      return true;
    idx++;
  }
  return false;
}

int KoHashTable::numberOfOccurrencesOfHash(Hash128 hash) const {
  uint32_t bits = hash.hash0 & TABLE_MASK;
  size_t idx = idxTable[bits];
  size_t size = koHashHistorySortedByLowBits.size();
  int count = 0;
  while(idx < size && ((koHashHistorySortedByLowBits[idx].hash0 & TABLE_MASK) == bits)) {
    if(hash == koHashHistorySortedByLowBits[idx])
      count++;
    idx++;
  }
  return count;
}
