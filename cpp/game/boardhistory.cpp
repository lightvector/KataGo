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

BoardHistory::BoardHistory() : BoardHistory(Rules::DEFAULT_GO) {}

BoardHistory::BoardHistory(const Rules& rules)
  : rules(rules),
    moveHistory(),
    preventEncoreHistory(),
    koHashHistory(),
    firstTurnIdxWithKoHistory(0),
    initialBoard(rules),
    initialPla(P_BLACK),
    initialEncorePhase(0),
    initialTurnNumber(0),
    assumeMultipleStartingBlackMovesAreHandicap(false),
    whiteHasMoved(false),
    overrideNumHandicapStones(-1),
    currentRecentBoardIdx(0),
    presumedNextMovePla(P_BLACK),
    consecutiveEndingPasses(0),
    hashesBeforeBlackPass(),
    hashesBeforeWhitePass(),
    encorePhase(0),
    numTurnsThisPhase(0),
    numApproxValidTurnsThisPhase(0),
    numConsecValidTurnsThisGame(0),
    koRecapBlockHash(),
    koCapturesInEncore(),
    whiteBonusScore(0.0f),
    whiteHandicapBonusScore(0.0f),
    hasButton(false),
    isPastNormalPhaseEnd(false),
    isGameFinished(false),
    winner(C_EMPTY),
    finalWhiteMinusBlackScore(0.0f),
    isScored(false),
    isNoResult(false),
    isResignation(false),
    isPassAliveFinished(false) {
  for(int i = 0; i < NUM_RECENT_BOARDS; i++) {
    recentBoards.emplace_back(rules);
  }
  if(!rules.isDots) {
    wasEverOccupiedOrPlayed.resize(Board::MAX_ARR_SIZE, false);
    superKoBanned.resize(Board::MAX_ARR_SIZE, false);
    koRecapBlocked.resize(Board::MAX_ARR_SIZE, false);
    secondEncoreStartColors.resize(Board::MAX_ARR_SIZE, C_EMPTY);
  }
}

BoardHistory::~BoardHistory()
{}

BoardHistory::BoardHistory(const Board& board) : BoardHistory(board, P_BLACK, board.rules, 0) {}

BoardHistory::BoardHistory(const Board& board, Player pla, const Rules& r, int ePhase)
  :rules(r),
   moveHistory(),
   preventEncoreHistory(),
   koHashHistory(),
   firstTurnIdxWithKoHistory(0),
   initialBoard(rules),
   initialPla(),
   initialEncorePhase(0),
   initialTurnNumber(0),
   assumeMultipleStartingBlackMovesAreHandicap(false),
   whiteHasMoved(false),
   overrideNumHandicapStones(-1),
   recentBoards(),
   currentRecentBoardIdx(0),
   presumedNextMovePla(pla),
   consecutiveEndingPasses(0),
   hashesBeforeBlackPass(),hashesBeforeWhitePass(),
   encorePhase(0),
   numTurnsThisPhase(0),
   numApproxValidTurnsThisPhase(0),
   numConsecValidTurnsThisGame(0),
   koRecapBlockHash(),
   koCapturesInEncore(),
   whiteBonusScore(0.0f),
   whiteHandicapBonusScore(0.0f),
   hasButton(false),
   isPastNormalPhaseEnd(false),
   isGameFinished(false),winner(C_EMPTY),finalWhiteMinusBlackScore(0.0f),
   isScored(false),isNoResult(false),isResignation(false),isPassAliveFinished(false)
{
  for(int i = 0; i < NUM_RECENT_BOARDS; i++) {
    recentBoards.emplace_back(rules);
  }
  if (!rules.isDots) {
    wasEverOccupiedOrPlayed.resize(Board::MAX_ARR_SIZE, false);
    superKoBanned.resize(Board::MAX_ARR_SIZE, false);
    koRecapBlocked.resize(Board::MAX_ARR_SIZE, false);
    secondEncoreStartColors.resize(Board::MAX_ARR_SIZE, C_EMPTY);
  }

  clear(board,pla,rules,ePhase);
}

BoardHistory::BoardHistory(const BoardHistory& other)
  :rules(other.rules),
   moveHistory(other.moveHistory),
   preventEncoreHistory(other.preventEncoreHistory),
   koHashHistory(other.koHashHistory),
   firstTurnIdxWithKoHistory(other.firstTurnIdxWithKoHistory),
   initialBoard(other.initialBoard),
   initialPla(other.initialPla),
   initialEncorePhase(other.initialEncorePhase),
   initialTurnNumber(other.initialTurnNumber),
   assumeMultipleStartingBlackMovesAreHandicap(other.assumeMultipleStartingBlackMovesAreHandicap),
   whiteHasMoved(other.whiteHasMoved),
   overrideNumHandicapStones(other.overrideNumHandicapStones),
   currentRecentBoardIdx(other.currentRecentBoardIdx),
   presumedNextMovePla(other.presumedNextMovePla),
   consecutiveEndingPasses(other.consecutiveEndingPasses),
   hashesBeforeBlackPass(other.hashesBeforeBlackPass),hashesBeforeWhitePass(other.hashesBeforeWhitePass),
   encorePhase(other.encorePhase),
   numTurnsThisPhase(other.numTurnsThisPhase),
   numApproxValidTurnsThisPhase(other.numApproxValidTurnsThisPhase),
   numConsecValidTurnsThisGame(other.numConsecValidTurnsThisGame),
   koRecapBlockHash(other.koRecapBlockHash),
   koCapturesInEncore(other.koCapturesInEncore),
   whiteBonusScore(other.whiteBonusScore),
   whiteHandicapBonusScore(other.whiteHandicapBonusScore),
   hasButton(other.hasButton),
   isPastNormalPhaseEnd(other.isPastNormalPhaseEnd),
   isGameFinished(other.isGameFinished),winner(other.winner),finalWhiteMinusBlackScore(other.finalWhiteMinusBlackScore),
   isScored(other.isScored),isNoResult(other.isNoResult),isResignation(other.isResignation),isPassAliveFinished(other.isPassAliveFinished)
{
  recentBoards = other.recentBoards;
  wasEverOccupiedOrPlayed = other.wasEverOccupiedOrPlayed;
  superKoBanned = other.superKoBanned;
  koRecapBlocked = other.koRecapBlocked;
  secondEncoreStartColors = other.secondEncoreStartColors;
}


BoardHistory& BoardHistory::operator=(const BoardHistory& other)
{
  if(this == &other)
    return *this;
  rules = other.rules;
  moveHistory = other.moveHistory;
  preventEncoreHistory = other.preventEncoreHistory;
  koHashHistory = other.koHashHistory;
  firstTurnIdxWithKoHistory = other.firstTurnIdxWithKoHistory;
  initialBoard = other.initialBoard;
  initialPla = other.initialPla;
  initialEncorePhase = other.initialEncorePhase;
  initialTurnNumber = other.initialTurnNumber;
  assumeMultipleStartingBlackMovesAreHandicap = other.assumeMultipleStartingBlackMovesAreHandicap;
  whiteHasMoved = other.whiteHasMoved;
  overrideNumHandicapStones = other.overrideNumHandicapStones;
  recentBoards = other.recentBoards;
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  presumedNextMovePla = other.presumedNextMovePla;
  wasEverOccupiedOrPlayed = other.wasEverOccupiedOrPlayed;
  superKoBanned = other.superKoBanned;
  consecutiveEndingPasses = other.consecutiveEndingPasses;
  hashesBeforeBlackPass = other.hashesBeforeBlackPass;
  hashesBeforeWhitePass = other.hashesBeforeWhitePass;
  encorePhase = other.encorePhase;
  numTurnsThisPhase = other.numTurnsThisPhase;
  numApproxValidTurnsThisPhase = other.numApproxValidTurnsThisPhase;
  numConsecValidTurnsThisGame = other.numConsecValidTurnsThisGame;
  koRecapBlocked = other.koRecapBlocked;
  koRecapBlockHash = other.koRecapBlockHash;
  koCapturesInEncore = other.koCapturesInEncore;
  secondEncoreStartColors = other.secondEncoreStartColors;
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
  isPassAliveFinished = other.isPassAliveFinished;

  return *this;
}

BoardHistory::BoardHistory(BoardHistory&& other) noexcept
 :rules(other.rules),
  moveHistory(std::move(other.moveHistory)),
  preventEncoreHistory(std::move(other.preventEncoreHistory)),
  koHashHistory(std::move(other.koHashHistory)),
  firstTurnIdxWithKoHistory(other.firstTurnIdxWithKoHistory),
  initialBoard(other.initialBoard),
  initialPla(other.initialPla),
  initialEncorePhase(other.initialEncorePhase),
  initialTurnNumber(other.initialTurnNumber),
  assumeMultipleStartingBlackMovesAreHandicap(other.assumeMultipleStartingBlackMovesAreHandicap),
  whiteHasMoved(other.whiteHasMoved),
  overrideNumHandicapStones(other.overrideNumHandicapStones),
  currentRecentBoardIdx(other.currentRecentBoardIdx),
  presumedNextMovePla(other.presumedNextMovePla),
  consecutiveEndingPasses(other.consecutiveEndingPasses),
  hashesBeforeBlackPass(std::move(other.hashesBeforeBlackPass)),hashesBeforeWhitePass(std::move(other.hashesBeforeWhitePass)),
  encorePhase(other.encorePhase),
  numTurnsThisPhase(other.numTurnsThisPhase),
  numApproxValidTurnsThisPhase(other.numApproxValidTurnsThisPhase),
  numConsecValidTurnsThisGame(other.numConsecValidTurnsThisGame),
  koRecapBlockHash(other.koRecapBlockHash),
  koCapturesInEncore(std::move(other.koCapturesInEncore)),
  whiteBonusScore(other.whiteBonusScore),
  whiteHandicapBonusScore(other.whiteHandicapBonusScore),
  hasButton(other.hasButton),
  isPastNormalPhaseEnd(other.isPastNormalPhaseEnd),
  isGameFinished(other.isGameFinished),winner(other.winner),finalWhiteMinusBlackScore(other.finalWhiteMinusBlackScore),
  isScored(other.isScored),isNoResult(other.isNoResult),isResignation(other.isResignation),isPassAliveFinished(other.isPassAliveFinished)
{
  recentBoards = other.recentBoards;
  wasEverOccupiedOrPlayed = other.wasEverOccupiedOrPlayed;
  superKoBanned = other.superKoBanned;
  koRecapBlocked = other.koRecapBlocked;
  secondEncoreStartColors = other.secondEncoreStartColors;
}

BoardHistory& BoardHistory::operator=(BoardHistory&& other) noexcept
{
  rules = other.rules;
  moveHistory = std::move(other.moveHistory);
  preventEncoreHistory = std::move(other.preventEncoreHistory);
  koHashHistory = std::move(other.koHashHistory);
  firstTurnIdxWithKoHistory = other.firstTurnIdxWithKoHistory;
  initialBoard = other.initialBoard;
  initialPla = other.initialPla;
  initialEncorePhase = other.initialEncorePhase;
  initialTurnNumber = other.initialTurnNumber;
  assumeMultipleStartingBlackMovesAreHandicap = other.assumeMultipleStartingBlackMovesAreHandicap;
  whiteHasMoved = other.whiteHasMoved;
  overrideNumHandicapStones = other.overrideNumHandicapStones;
  recentBoards = other.recentBoards;
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  presumedNextMovePla = other.presumedNextMovePla;
  wasEverOccupiedOrPlayed = other.wasEverOccupiedOrPlayed;
  superKoBanned = other.superKoBanned;
  consecutiveEndingPasses = other.consecutiveEndingPasses;
  hashesBeforeBlackPass = std::move(other.hashesBeforeBlackPass);
  hashesBeforeWhitePass = std::move(other.hashesBeforeWhitePass);
  encorePhase = other.encorePhase;
  numTurnsThisPhase = other.numTurnsThisPhase;
  numApproxValidTurnsThisPhase = other.numApproxValidTurnsThisPhase;
  numConsecValidTurnsThisGame = other.numConsecValidTurnsThisGame;
  koRecapBlocked = other.koRecapBlocked;
  koRecapBlockHash = other.koRecapBlockHash;
  koCapturesInEncore = std::move(other.koCapturesInEncore);
  secondEncoreStartColors = other.secondEncoreStartColors;
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
  isPassAliveFinished = other.isPassAliveFinished;

  return *this;
}

void BoardHistory::clear(const Board& board, Player pla, const Rules& r, int ePhase) {
  rules = r;
  moveHistory.clear();
  preventEncoreHistory.clear();
  koHashHistory.clear();
  firstTurnIdxWithKoHistory = 0;

  initialBoard = board;
  initialPla = pla;
  initialEncorePhase = ePhase;
  initialTurnNumber = 0;
  assumeMultipleStartingBlackMovesAreHandicap = false;
  whiteHasMoved = false;
  overrideNumHandicapStones = -1;

  //This makes it so that if we ask for recent boards with a lookback beyond what we have a history for,
  //we simply return copies of the starting board.
  for(int i = 0; i<NUM_RECENT_BOARDS; i++)
    recentBoards[i] = board;
  currentRecentBoardIdx = 0;

  presumedNextMovePla = pla;
  consecutiveEndingPasses = 0;
  hashesBeforeBlackPass.clear();
  hashesBeforeWhitePass.clear();
  numTurnsThisPhase = 0;
  numApproxValidTurnsThisPhase = 0;
  numConsecValidTurnsThisGame = 0;
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
  isPassAliveFinished = false;

  if (!rules.isDots) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        wasEverOccupiedOrPlayed[loc] = (board.colors[loc] != C_EMPTY);
      }
    }
    std::fill(superKoBanned.begin(), superKoBanned.end(), false);
    std::fill(koRecapBlocked.begin(), koRecapBlocked.end(), false);

    //Handle encore phase
    encorePhase = ePhase;
    assert(encorePhase >= 0 && encorePhase <= 2);
    if(encorePhase > 0)
      assert(rules.scoringRule == Rules::SCORING_TERRITORY);
    //Update the few parameters that depend on encore
    if(encorePhase == 2)
      std::copy_n(board.colors, Board::MAX_ARR_SIZE, secondEncoreStartColors.begin());
    else
      std::fill(secondEncoreStartColors.begin(), secondEncoreStartColors.end(), C_EMPTY);

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
    whiteHandicapBonusScore = (float)computeWhiteHandicapBonus();
  }
}

BoardHistory BoardHistory::copyToInitial() const {
  BoardHistory hist(initialBoard, initialPla, rules, initialEncorePhase);
  hist.setInitialTurnNumber(initialTurnNumber);
  hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
  hist.setOverrideNumHandicapStones(overrideNumHandicapStones);
  return hist;
}

void BoardHistory::setInitialTurnNumber(int64_t n) {
  initialTurnNumber = n;
}

void BoardHistory::setAssumeMultipleStartingBlackMovesAreHandicap(bool b) {
  assumeMultipleStartingBlackMovesAreHandicap = b;
  whiteHandicapBonusScore = static_cast<float>(computeWhiteHandicapBonus());
}

void BoardHistory::setOverrideNumHandicapStones(int n) {
  overrideNumHandicapStones = n;
  whiteHandicapBonusScore = static_cast<float>(computeWhiteHandicapBonus());
}


static int numHandicapStonesOnBoardHelper(const Board& board, const int blackNonPassTurnsToStart) {
  int startBoardNumBlackStones, startBoardNumWhiteStones;
  board.getCurrentMoves(startBoardNumBlackStones, startBoardNumWhiteStones, false);

  //If we set up in a nontrivial position, then consider it a non-handicap game.
  if(startBoardNumWhiteStones != 0)
    return 0;
  // Add in additional counted stones
  const int blackTurnAdvantage = startBoardNumBlackStones + blackNonPassTurnsToStart;

  //If there was only one black move/stone to start, then it was a regular game
  if(blackTurnAdvantage <= 1)
    return 0;
  return blackTurnAdvantage;
}

int BoardHistory::numHandicapStonesOnBoard(const Board& b) {
  return numHandicapStonesOnBoardHelper(b,0);
}

int BoardHistory::computeNumHandicapStones() const {
  if(overrideNumHandicapStones >= 0)
    return overrideNumHandicapStones;

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
        //Two white moves in a row? Re-set count and quit out.
        if(i+1 < moveHistory.size() && moveHistory[i+1].pla != P_BLACK) {
          blackNonPassTurnsToStart = 0;
          break;
        }
        //White move is a single isolated pass? Assume that it's still a handicap game, it's just that the black
        //moves are interleaved with white passes. Ignore it and continue.
        if(moveLoc == Board::PASS_LOC)
          continue;
        if (moveLoc == Board::RESIGN_LOC)
          continue; // Actually shouldn't be here
        //Otherwise quit out, we have a normal white move.
        break;
      }
      if(moveLoc != Board::PASS_LOC && moveLoc != Board::NULL_LOC && moveLoc != Board::RESIGN_LOC)
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

void BoardHistory::printBasicInfo(ostream& out, const Board& board) const {
  Board::printBoard(out, board, Board::NULL_LOC, &moveHistory, false);
  const bool isDots = rules.isDots;
  assert(isDots == board.rules.isDots);
  out << "Next player: " << PlayerIO::playerToString(presumedNextMovePla, isDots) << endl;
  if(encorePhase > 0)
    out << "Game phase: " << encorePhase << endl;
  out << "Rules: " << rules.toJsonString() << endl;
  if(whiteHandicapBonusScore != 0)
    out << "Handicap bonus score: " << whiteHandicapBonusScore << endl;

  const auto firstPlayerName = PlayerIO::playerToString(P_BLACK, isDots);
  const auto secondPlayerName = PlayerIO::playerToString(P_WHITE, isDots);
  if (!isDots) {
    out << firstPlayerName << " stones captured: " << board.numBlackCaptures << endl;
    out << secondPlayerName << " stones captured: " << board.numWhiteCaptures << endl;
  } else {
    out << firstPlayerName << " score: " << board.numWhiteCaptures << endl;
    out << secondPlayerName << " score: " << board.numBlackCaptures << endl;
  }
  if (isGameFinished) {
    out << "Game is finished, winner: " << PlayerIO::playerToString(winner, isDots) << ", score: " << finalWhiteMinusBlackScore << ", resign: " << boolalpha << isResignation << endl;
  }
}

void BoardHistory::printDebugInfo(ostream& out, const Board& board) const {
  out << board << endl;
  const bool isDots = board.rules.isDots;
  if (!isDots) {
    out << "Initial pla " << PlayerIO::playerToString(initialPla, rules.isDots) << endl;
    out << "Encore phase " << encorePhase << endl;
    out << "Turns this phase " << numTurnsThisPhase << endl;
    out << "Approx valid turns this phase " << numApproxValidTurnsThisPhase << endl;
    out << "Approx consec valid turns this game " << numConsecValidTurnsThisGame << endl;
  } else {
    assert(0 == encorePhase);
  }
  out << "Rules " << rules << endl;
  if (!isDots) {
    out << "Ko recap block hash " << koRecapBlockHash << endl;
  } else {
    assert(Hash128() == koRecapBlockHash);
  }
  out << "White bonus score " << whiteBonusScore << endl;
  if (!isDots) {
    out << "White handicap bonus score " << whiteHandicapBonusScore << endl;
    out << "Has button " << hasButton << endl;
  } else {
    assert(0.0f == whiteHandicapBonusScore);
    assert(!hasButton);
  }
  out << "Presumed next pla " << PlayerIO::playerToString(presumedNextMovePla, rules.isDots) << endl;
  if (!isDots) {
    out << "Past normal phase end " << isPastNormalPhaseEnd << endl;
  } else {
    assert(0 == isPastNormalPhaseEnd);
  }
  out << "Game result " << isGameFinished << " " << PlayerIO::playerToString(winner, rules.isDots) << " "
      << finalWhiteMinusBlackScore << " " << isScored << " " << isNoResult << " " << isResignation;
  if (isPassAliveFinished) {
    out << " " << isPassAliveFinished;
  }
  out << endl;
  out << "Last moves ";
  for (const auto i : moveHistory)
    out << Location::toString(i.loc,board) << " ";
  out << endl;
  if (!isDots) {
    assert(firstTurnIdxWithKoHistory + koHashHistory.size() == moveHistory.size() + 1);
  }
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
  assert(!rules.isDots);

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
  float drawAdjustment = rules.gameResultWillBeInteger() ? (float)(drawEquivalentWinsForWhite - 0.5) : 0.0f;
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
  assert(rules.isDots == board.isDots() && !rules.isDots);

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
  assert(rules.isDots == board.isDots() && !rules.isDots);

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

void BoardHistory::setFinalScoreAndWinner(const float score) {
  finalWhiteMinusBlackScore = score;
  if(finalWhiteMinusBlackScore > Global::FLOAT_EPS)
    winner = C_WHITE;
  else if(finalWhiteMinusBlackScore < -Global::FLOAT_EPS)
    winner = C_BLACK;
  else
    winner = C_EMPTY;
}

void BoardHistory::endAndScoreGameNow(const Board& board, Color area[Board::MAX_ARR_SIZE]) {
  assert(rules.isDots == board.isDots());

  int boardScore = 0;
  if(rules.isDots)
    boardScore = countDotsScoreWhiteMinusBlack(board,area);
  else if(rules.scoringRule == Rules::SCORING_AREA)
    boardScore = countAreaScoreWhiteMinusBlack(board,area);
  else if(rules.scoringRule == Rules::SCORING_TERRITORY)
    boardScore = countTerritoryAreaScoreWhiteMinusBlack(board,area);
  else
    ASSERT_UNREACHABLE;

  if(hasButton) {
    assert(!board.rules.isDots);
    hasButton = false;
    whiteBonusScore += (presumedNextMovePla == P_WHITE ? 0.5f : -0.5f);
  }

  setFinalScoreAndWinner(static_cast<float>(boardScore) + whiteBonusScore + whiteHandicapBonusScore + rules.komi);
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
  assert(rules.isDots == board.isDots());

  if (rules.isDots) {
    if (const float whiteScoreAfterGrounding = whiteScoreIfGroundingAlive(board); whiteScoreAfterGrounding != std::numeric_limits<float>::quiet_NaN()) {
      setFinalScoreAndWinner(whiteScoreAfterGrounding);
      isScored = true;
      isNoResult = false;
      isResignation = false;
      isGameFinished = true;
      isPastNormalPhaseEnd = false;
      isPassAliveFinished = true;
    }
  } else {
    Color area[Board::MAX_ARR_SIZE];
    int boardScore = 0;

    bool nonPassAliveStones = false;
    bool safeBigTerritories = false;
    bool unsafeBigTerritories = false;
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
      isPassAliveFinished = true;
    }
  }
}

void BoardHistory::endGameIfNoLegalMoves(const Board& board) {
  if (board.numLegalMovesIfSuiAllowed == 0) {
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        const Loc loc = Location::getLoc(x, y, board.x_size);
        assert(!board.isLegal(loc, P_BLACK, rules.multiStoneSuicideLegal, true));
        assert(!board.isLegal(loc, P_WHITE, rules.multiStoneSuicideLegal, true));
      }
    }
    endAndScoreGameNow(board);
  }
}

void BoardHistory::setWinnerByResignation(Player pla) {
  isGameFinished = true;
  isPastNormalPhaseEnd = false;
  isScored = false;
  isNoResult = false;
  isResignation = true;
  isPassAliveFinished = false;
  winner = pla;
  finalWhiteMinusBlackScore = 0.0f;
}

void BoardHistory::setKoRecapBlocked(Loc loc, bool b) {
  assert(!rules.isDots);
  if(koRecapBlocked[loc] != b) {
    koRecapBlocked[loc] = b;
    //We used to have per-color marks, so the zobrist was for both. Just combine them.
    koRecapBlockHash ^= Board::ZOBRIST_KO_MARK_HASH[loc][C_BLACK] ^ Board::ZOBRIST_KO_MARK_HASH[loc][C_WHITE];
  }
}

bool BoardHistory::isLegal(const Board& board, Loc moveLoc, Player movePla) const {
  assert(board.isDots() == rules.isDots);

  if(movePla != presumedNextMovePla)
    return false;

  if (rules.isDots) {
    // Ko is not relevant for Dots game
    return board.isLegal(moveLoc, movePla, rules.multiStoneSuicideLegal, false);
  }

  //Ko-moves in the encore that are recapture blocked are interpreted as pass-for-ko, so they are legal
  if(encorePhase > 0) {
    if(moveLoc >= 0 && moveLoc < Board::MAX_ARR_SIZE && moveLoc != Board::PASS_LOC && moveLoc != Board::RESIGN_LOC) {
      if(board.colors[moveLoc] == getOpp(movePla) && koRecapBlocked[moveLoc] && board.getChainSize(moveLoc) == 1 && board.getNumLiberties(moveLoc) == 1)
        return true;
      Loc koCaptureLoc = board.getKoCaptureLoc(moveLoc,movePla);
      if(koCaptureLoc != Board::NULL_LOC && koRecapBlocked[koCaptureLoc] && board.colors[koCaptureLoc] == getOpp(movePla))
        return true;
    }
  }
  else {
    //Only check ko bans during normal play.
    //Ko mechanics in the encore are totally different, we ignore simple ko loc.
    if(board.isKoBanned(moveLoc))
      return false;
  }
  if(!board.isLegal(moveLoc, movePla, rules.multiStoneSuicideLegal, true))
    return false;
  if(superKoBanned[moveLoc])
    return false;

  return true;
}

bool BoardHistory::isPassForKo(const Board& board, Loc moveLoc, Player movePla) const {
  assert(rules.isDots == board.isDots());
  if (rules.isDots) return false;

  if(encorePhase > 0 && moveLoc >= 0 && moveLoc < Board::MAX_ARR_SIZE && moveLoc != Board::PASS_LOC && moveLoc != Board::RESIGN_LOC) {
    if(board.colors[moveLoc] == getOpp(movePla) && koRecapBlocked[moveLoc] && board.getChainSize(moveLoc) == 1 && board.getNumLiberties(moveLoc) == 1)
      return true;

    Loc koCaptureLoc = board.getKoCaptureLoc(moveLoc,movePla);
    if(koCaptureLoc != Board::NULL_LOC && koRecapBlocked[koCaptureLoc] && board.colors[koCaptureLoc] == getOpp(movePla))
      return true;
  }
  return false;
}

int64_t BoardHistory::getCurrentTurnNumber() const {
  return std::max((int64_t)0,initialTurnNumber + (int64_t)moveHistory.size());
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
  assert(rules.isDots == board.isDots());
  if (rules.isDots) return false;
  
  // TODO: probably add the assert? assert(!board.isDots());
  Hash128 koHashBeforeMove = getKoHash(rules, board, movePla, encorePhase, koRecapBlockHash);
  if(newConsecutiveEndingPassesAfterPass() >= 2 ||
     wouldBeSpightlikeEndingPass(movePla,koHashBeforeMove))
    return true;
  return false;
}

bool BoardHistory::passWouldEndGame(const Board& board, Player movePla) const {
  if (board.rules.isDots) {
    return true; // Pass in Dots game is grounding move that always ends the game
  }
  return passWouldEndPhase(board,movePla) && (
    rules.scoringRule == Rules::SCORING_AREA
    || (rules.scoringRule == Rules::SCORING_TERRITORY && encorePhase >= 2)
  );
}

bool BoardHistory::shouldSuppressEndGameFromFriendlyPass(const Board& board, Player movePla) const {
  return rules.friendlyPassOk &&
    rules.scoringRule == Rules::SCORING_AREA &&
    newConsecutiveEndingPassesAfterPass() == 2 &&
    !wouldBeSpightlikeEndingPass(movePla, getKoHash(rules, board, movePla, encorePhase, koRecapBlockHash));
}

bool BoardHistory::isFinalPhase() const {
  return
    rules.isDots
    || rules.scoringRule == Rules::SCORING_AREA
    || (rules.scoringRule == Rules::SCORING_TERRITORY && encorePhase >= 2);
}

bool BoardHistory::isLegalTolerant(const Board& board, Loc moveLoc, Player movePla) const {
  // Allow either side to move during tolerant play, but still check that a player is specified
  if(movePla != P_BLACK && movePla != P_WHITE)
    return false;
  constexpr bool multiStoneSuicideLegal = true; // Tolerate suicide and ko regardless of rules
  constexpr bool ignoreKo = true;
  if(!isPassForKo(board, moveLoc, movePla) && !board.isLegal(moveLoc,movePla,multiStoneSuicideLegal,ignoreKo))
    return false;
  return true;
}
bool BoardHistory::makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla) {
  // Allow either side to move during tolerant play, but still check that a player is specified
  if(movePla != P_BLACK && movePla != P_WHITE)
    return false;
  bool multiStoneSuicideLegal = true; // Tolerate suicide and ko regardless of rules
  constexpr bool ignoreKo = true;
  if(!isPassForKo(board, moveLoc, movePla) && !board.isLegal(moveLoc,movePla,multiStoneSuicideLegal,ignoreKo))
    return false;
  makeBoardMoveAssumeLegal(board,moveLoc,movePla,NULL);
  return true;
}
bool BoardHistory::makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla, bool preventEncore) {
  // Allow either side to move during tolerant play, but still check that a player is specified
  if(movePla != P_BLACK && movePla == presumedNextMovePla && movePla != P_WHITE)
    return false;
  bool multiStoneSuicideLegal = true; // Tolerate suicide and ko regardless of rules
  constexpr bool ignoreKo = true;
  if(!isPassForKo(board, moveLoc, movePla) && !board.isLegal(moveLoc,movePla,multiStoneSuicideLegal,ignoreKo))
    return false;
  makeBoardMoveAssumeLegal(board,moveLoc,movePla,NULL,preventEncore);
  return true;
}

void BoardHistory::makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla, const KoHashTable* rootKoHashTable, bool preventEncore) {
  Hash128 posHashBeforeMove = board.pos_hash;

  //Handle if somehow we're making a move after the game or phase was ended
  if(isGameFinished || isPastNormalPhaseEnd) {
    //Cap at 1 - do include the latest move that likely ended the game by itself since by itself
    //absent any history of passes it should be valid still.
    numApproxValidTurnsThisPhase = std::min(numApproxValidTurnsThisPhase,1);
    numConsecValidTurnsThisGame = std::min(numConsecValidTurnsThisGame,1);
  }

  const bool moveIsIllegal = !isLegal(board,moveLoc,movePla);

  //And if somehow we're making a move after the game was ended, just clear those values and continue.
  isGameFinished = false;
  isPastNormalPhaseEnd = false;
  winner = C_EMPTY;
  finalWhiteMinusBlackScore = 0.0f;
  isScored = false;
  isNoResult = false;
  isResignation = false;
  isPassAliveFinished = false;

  //Update consecutiveEndingPasses and button
  bool isSpightlikeEndingPass = false;
  bool wasPassForKo = false;

  if (moveLoc == Board::RESIGN_LOC) {
    setWinnerByResignation(getOpp(movePla));
  } else if (rules.isDots) {
    //Dots game
    board.playMoveAssumeLegal(moveLoc, movePla);
    if (moveLoc == Board::PASS_LOC) {
      isScored = true;
      isNoResult = false;
      isResignation = false;
      isGameFinished = true;
      isPastNormalPhaseEnd = false;
      const auto whiteMinusBlackScore = static_cast<float>(board.numBlackCaptures - board.numWhiteCaptures);
      setFinalScoreAndWinner(whiteMinusBlackScore + whiteBonusScore + whiteHandicapBonusScore + rules.komi);
    }
  } else {
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
    if(encorePhase > 0 && moveLoc != Board::PASS_LOC) {
      if(board.colors[moveLoc] == getOpp(movePla) && koRecapBlocked[moveLoc]) {
        setKoRecapBlocked(moveLoc,false);
        wasPassForKo = true;
        //Clear simple ko loc just in case
        //Since we aren't otherwise touching the board, from the board's perspective a player will be moving twice in a row.
        board.clearSimpleKoLoc();
      }
      else {
        Loc koCaptureLoc = board.getKoCaptureLoc(moveLoc,movePla);
        if(koCaptureLoc != Board::NULL_LOC && koRecapBlocked[koCaptureLoc] && board.colors[koCaptureLoc] == getOpp(movePla)) {
          setKoRecapBlocked(koCaptureLoc,false);
          wasPassForKo = true;
          //Clear simple ko loc just in case
          //Since we aren't otherwise touching the board, from the board's perspective a player will be moving twice in a row.
          board.clearSimpleKoLoc();
        }
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
  }

  //Update recent boards
  currentRecentBoardIdx = (currentRecentBoardIdx + 1) % NUM_RECENT_BOARDS;
  recentBoards[currentRecentBoardIdx] = board;
  moveHistory.emplace_back(moveLoc,movePla);
  presumedNextMovePla = getOpp(movePla);

  numTurnsThisPhase += 1;
  numApproxValidTurnsThisPhase += 1;
  numConsecValidTurnsThisGame += 1;

  if(moveIsIllegal)
    numConsecValidTurnsThisGame = 0;

  if (!rules.isDots) {
    Hash128 koHashAfterThisMove = getKoHash(rules,board,getOpp(movePla),encorePhase,koRecapBlockHash);
    koHashHistory.push_back(koHashAfterThisMove);
    preventEncoreHistory.push_back(preventEncore);

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
      std::fill(superKoBanned.begin(), superKoBanned.end(), false);
      for(size_t i = 0; i<koCapturesInEncore.size(); i++) {
        const EncoreKoCapture& ekc = koCapturesInEncore[i];
        if(ekc.posHashBeforeMove == board.pos_hash && ekc.movePla == nextPla)
          superKoBanned[ekc.moveLoc] = true;
      }
    }


    //Territory scoring - chill 1 point per move in main phase and first encore
    if(rules.scoringRule == Rules::SCORING_TERRITORY && encorePhase <= 1 && moveLoc != Board::PASS_LOC && moveLoc != Board::RESIGN_LOC && !wasPassForKo) {
      if(movePla == P_BLACK)
        whiteBonusScore += 1.0f;
      else if(movePla == P_WHITE)
        whiteBonusScore -= 1.0f;
      else
        ASSERT_UNREACHABLE;
    }

    //Handicap bonus score
    if(movePla == P_WHITE && moveLoc != Board::PASS_LOC && moveLoc != Board::RESIGN_LOC)
      whiteHasMoved = true;
    if(assumeMultipleStartingBlackMovesAreHandicap && !whiteHasMoved && movePla == P_BLACK && rules.whiteHandicapBonusRule != Rules::WHB_ZERO) {
      whiteHandicapBonusScore = (float)computeWhiteHandicapBonus();
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
            //Cap at 1 - do include just the single pass here by itself since the single pass by itself
            //absent any history of passes before that should be valid still.
            numApproxValidTurnsThisPhase = std::min(numApproxValidTurnsThisPhase,1);
            numConsecValidTurnsThisGame = std::min(numConsecValidTurnsThisGame,1);
          }
          else {
            encorePhase += 1;
            numTurnsThisPhase = 0;
            numApproxValidTurnsThisPhase = 0;
            if(encorePhase == 2)
              std::copy_n(board.colors, Board::MAX_ARR_SIZE, secondEncoreStartColors.begin());

            std::fill(superKoBanned.begin(), superKoBanned.end(), false);
            consecutiveEndingPasses = 0;
            hashesBeforeBlackPass.clear();
            hashesBeforeWhitePass.clear();
            std::fill(koRecapBlocked.begin(), koRecapBlocked.end(), false);
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
    if(moveLoc != Board::PASS_LOC && moveLoc != Board::RESIGN_LOC && (encorePhase > 0 || rules.koRule == Rules::KO_SIMPLE)) {
      if(numberOfKoHashOccurrencesInHistory(koHashHistory[koHashHistory.size()-1], rootKoHashTable) >= 3) {
        isNoResult = true;
        isGameFinished = true;
      }
    }
  }
}


bool BoardHistory::hasBlackPassOrWhiteFirst() const {
  //First move was made by white this game, on an empty board.
  if(initialBoard.isStartPos() && moveHistory.size() > 0 && moveHistory[0].pla == P_WHITE)
    return true;
  //Black passed exactly once or white doublemoved
  int numBlackPasses = 0;
  int numWhitePasses = 0;
  int numBlackDoubleMoves = 0;
  int numWhiteDoubleMoves = 0;
  for(int i = 0; i<moveHistory.size(); i++) {
    if(moveHistory[i].loc == Board::PASS_LOC && moveHistory[i].pla == P_BLACK)
      numBlackPasses++;
    if(moveHistory[i].loc == Board::PASS_LOC && moveHistory[i].pla == P_WHITE)
      numWhitePasses++;
    if(i > 0 && moveHistory[i].pla == P_BLACK && moveHistory[i-1].pla == P_BLACK)
      numBlackDoubleMoves++;
    if(i > 0 && moveHistory[i].pla == P_WHITE && moveHistory[i-1].pla == P_WHITE)
      numWhiteDoubleMoves++;
  }
  if(numBlackPasses == 1 && numWhitePasses == 0 && numBlackDoubleMoves == 0 && numWhiteDoubleMoves == 0)
    return true;
  if(numBlackPasses == 0 && numWhitePasses == 0 && numBlackDoubleMoves == 0 && numWhiteDoubleMoves == 1)
    return true;

  return false;
}

Hash128 BoardHistory::getSituationAndSimpleKoHash(const Board& board, Player nextPlayer) {
  //Note that board.pos_hash also incorporates the size of the board.
  Hash128 hash = board.pos_hash;
  hash ^= Board::ZOBRIST_PLAYER_HASH[nextPlayer];
  if (board.isDots()) {
    assert(board.ko_loc == Board::NULL_LOC);
  }
  if(board.ko_loc != Board::NULL_LOC)
    hash ^= Board::ZOBRIST_KO_LOC_HASH[board.ko_loc];
  return hash;
}

Hash128 BoardHistory::getSituationAndSimpleKoAndPrevPosHash(const Board& board, const BoardHistory& hist, Player nextPlayer) {
  //Note that board.pos_hash also incorporates the size of the board.
  Hash128 hash = board.pos_hash;
  hash ^= Board::ZOBRIST_PLAYER_HASH[nextPlayer];
  if (board.isDots()) {
    assert(board.ko_loc == Board::NULL_LOC);
  }
  if(board.ko_loc != Board::NULL_LOC)
    hash ^= Board::ZOBRIST_KO_LOC_HASH[board.ko_loc];

  Hash128 mixed;
  mixed.hash1 = Hash::rrmxmx(hash.hash0);
  mixed.hash0 = Hash::splitMix64(hash.hash1);
  if(hist.moveHistory.size() > 0)
    mixed ^= hist.getRecentBoard(1).pos_hash;
  return mixed;
}

Hash128 BoardHistory::getSituationRulesAndKoHash(const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite) {
  int xSize = board.x_size;
  int ySize = board.y_size;

  //Note that board.pos_hash also incorporates the size of the board.
  Hash128 hash = board.pos_hash;
  hash ^= Board::ZOBRIST_PLAYER_HASH[nextPlayer];

  if (!board.isDots()) {
    assert(hist.encorePhase >= 0 && hist.encorePhase <= 2);
    hash ^= Board::ZOBRIST_ENCORE_HASH[hist.encorePhase];

    if(hist.encorePhase == 0) {
      if(board.ko_loc != Board::NULL_LOC)
        hash ^= Board::ZOBRIST_KO_LOC_HASH[board.ko_loc];
      for(int y = 0; y<ySize; y++) {
        for(int x = 0; x<xSize; x++) {
          Loc loc = Location::getLoc(x,y,xSize);
          if(hist.superKoBanned[loc] && loc != board.ko_loc)
            hash ^= Board::ZOBRIST_KO_LOC_HASH[loc];
        }
      }
    }
    else {
      for(int y = 0; y<ySize; y++) {
        for(int x = 0; x<xSize; x++) {
          Loc loc = Location::getLoc(x,y,xSize);
          if(hist.superKoBanned[loc])
            hash ^= Board::ZOBRIST_KO_LOC_HASH[loc];
          if(hist.koRecapBlocked[loc])
            hash ^= Board::ZOBRIST_KO_MARK_HASH[loc][P_BLACK] ^ Board::ZOBRIST_KO_MARK_HASH[loc][P_WHITE];
        }
      }
      if(hist.encorePhase == 2) {
        for(int y = 0; y<ySize; y++) {
          for(int x = 0; x<xSize; x++) {
            Loc loc = Location::getLoc(x,y,xSize);
            Color c = hist.secondEncoreStartColors[loc];
            if(c != C_EMPTY)
              hash ^= Board::ZOBRIST_SECOND_ENCORE_START_HASH[loc][c];
          }
        }
      }
    }
  }

  float selfKomi = hist.currentSelfKomi(nextPlayer,drawEquivalentWinsForWhite);

  //Discretize the komi for the purpose of matching hash
  int64_t komiDiscretized = (int64_t)(selfKomi*256.0f);
  uint64_t komiHash = Hash::murmurMix((uint64_t)komiDiscretized);
  hash.hash0 ^= komiHash;
  hash.hash1 ^= Hash::basicLCong(komiHash);

  //Fold in the ko, scoring, and suicide rules
  if(hist.rules.multiStoneSuicideLegal)
    hash ^= Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH;

  if (!board.isDots()) {
    hash ^= Rules::ZOBRIST_KO_RULE_HASH[hist.rules.koRule];
    hash ^= Rules::ZOBRIST_SCORING_RULE_HASH[hist.rules.scoringRule];
    hash ^= Rules::ZOBRIST_TAX_RULE_HASH[hist.rules.taxRule];
    if(hist.hasButton)
      hash ^= Rules::ZOBRIST_BUTTON_HASH;
    if(hist.rules.friendlyPassOk)
      hash ^= Rules::ZOBRIST_FRIENDLY_PASS_OK_HASH;
  } else {
    hash ^= Rules::ZOBRIST_DOTS_GAME_HASH;
    if (hist.rules.dotsCaptureEmptyBases) {
      hash ^= Rules::ZOBRIST_DOTS_CAPTURE_EMPTY_BASES_HASH;
    }
  }

  return hash;
}



KoHashTable::KoHashTable()
  :koHashHistorySortedByLowBits(),
   firstTurnIdxWithKoHistory(0)
{
  idxTable = new uint32_t[TABLE_SIZE];
  std::fill(idxTable,idxTable+TABLE_SIZE,(uint32_t)(0));
}
KoHashTable::~KoHashTable() {
  delete[] idxTable;
}

size_t KoHashTable::size() const {
  return koHashHistorySortedByLowBits.size();
}

void KoHashTable::recompute(const BoardHistory& history) {
  assert(!history.rules.isDots);

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
