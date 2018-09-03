
#include <algorithm>
#include "../game/boardhistory.h"

static Hash128 getKoHash(const Rules& rules, const Board& board, Player pla, int encorePhase, Hash128 koProhibitHash) {
  if(rules.koRule == Rules::KO_SITUATIONAL || encorePhase > 0)
    return board.pos_hash ^ Board::ZOBRIST_PLAYER_HASH[pla] ^ koProhibitHash;
  else
    return board.pos_hash ^ koProhibitHash;
}
static Hash128 getKoHashAfterMoveNonEncore(const Rules& rules, Hash128 posHashAfterMove, Player pla) {
  if(rules.koRule == Rules::KO_SITUATIONAL)
    return posHashAfterMove ^ Board::ZOBRIST_PLAYER_HASH[pla];
  else
    return posHashAfterMove;
}

BoardHistory::BoardHistory()
  :rules(),
   moveHistory(),koHashHistory(),
   koHistoryLastClearedBeginningMoveIdx(0),
   recentBoards(),
   currentRecentBoardIdx(0),
   consecutiveEndingPasses(0),
   hashesAfterBlackPass(),hashesAfterWhitePass(),
   encorePhase(0),koProhibitHash(),
   koCapturesInEncore(),
   whiteBonusScore(0),
   isGameFinished(false),winner(C_EMPTY),finalWhiteMinusBlackScore(0.0f),isNoResult(false)
{
  std::fill(wasEverOccupiedOrPlayed, wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, false);
  std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
  std::fill(blackKoProhibited, blackKoProhibited+Board::MAX_ARR_SIZE, false);
  std::fill(whiteKoProhibited, whiteKoProhibited+Board::MAX_ARR_SIZE, false);
  std::fill(secondEncoreStartColors, secondEncoreStartColors+Board::MAX_ARR_SIZE, C_EMPTY);
}

BoardHistory::~BoardHistory()
{}

BoardHistory::BoardHistory(const Board& board, Player pla, const Rules& r)
  :rules(r),
   moveHistory(),koHashHistory(),
   koHistoryLastClearedBeginningMoveIdx(0),
   recentBoards(),
   currentRecentBoardIdx(0),
   consecutiveEndingPasses(0),
   hashesAfterBlackPass(),hashesAfterWhitePass(),
   encorePhase(0),koProhibitHash(),
   koCapturesInEncore(),
   whiteBonusScore(0),
   isGameFinished(false),winner(C_EMPTY),finalWhiteMinusBlackScore(0.0f),isNoResult(false)
{
  std::fill(wasEverOccupiedOrPlayed, wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, false);
  std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
  std::fill(blackKoProhibited, blackKoProhibited+Board::MAX_ARR_SIZE, false);
  std::fill(whiteKoProhibited, whiteKoProhibited+Board::MAX_ARR_SIZE, false);
  std::fill(secondEncoreStartColors, secondEncoreStartColors+Board::MAX_ARR_SIZE, C_EMPTY);

  clear(board,pla,rules);
}

BoardHistory::BoardHistory(const BoardHistory& other)
  :rules(other.rules),
   moveHistory(other.moveHistory),koHashHistory(other.koHashHistory),
   koHistoryLastClearedBeginningMoveIdx(other.koHistoryLastClearedBeginningMoveIdx),
   recentBoards(other.recentBoards),
   currentRecentBoardIdx(other.currentRecentBoardIdx),
   consecutiveEndingPasses(other.consecutiveEndingPasses),
   hashesAfterBlackPass(other.hashesAfterBlackPass),hashesAfterWhitePass(other.hashesAfterWhitePass),
   encorePhase(other.encorePhase),koProhibitHash(other.koProhibitHash),
   koCapturesInEncore(other.koCapturesInEncore),
   whiteBonusScore(other.whiteBonusScore),
   isGameFinished(other.isGameFinished),winner(other.winner),finalWhiteMinusBlackScore(other.finalWhiteMinusBlackScore),isNoResult(other.isNoResult)
{
  std::copy(other.wasEverOccupiedOrPlayed, other.wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, wasEverOccupiedOrPlayed);
  std::copy(other.superKoBanned, other.superKoBanned+Board::MAX_ARR_SIZE, superKoBanned);
  std::copy(other.blackKoProhibited, other.blackKoProhibited+Board::MAX_ARR_SIZE, blackKoProhibited);
  std::copy(other.whiteKoProhibited, other.whiteKoProhibited+Board::MAX_ARR_SIZE, whiteKoProhibited);
  std::copy(other.secondEncoreStartColors, other.secondEncoreStartColors+Board::MAX_ARR_SIZE, secondEncoreStartColors);
}


BoardHistory& BoardHistory::operator=(const BoardHistory& other)
{
  if(this == &other)
    return *this;
  rules = other.rules;
  moveHistory = other.moveHistory;
  koHashHistory = other.koHashHistory;
  koHistoryLastClearedBeginningMoveIdx = other.koHistoryLastClearedBeginningMoveIdx;
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  std::copy(other.wasEverOccupiedOrPlayed, other.wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, wasEverOccupiedOrPlayed);
  std::copy(other.superKoBanned, other.superKoBanned+Board::MAX_ARR_SIZE, superKoBanned);
  consecutiveEndingPasses = other.consecutiveEndingPasses;
  hashesAfterBlackPass = other.hashesAfterBlackPass;
  hashesAfterWhitePass = other.hashesAfterWhitePass;
  encorePhase = other.encorePhase;
  std::copy(other.blackKoProhibited, other.blackKoProhibited+Board::MAX_ARR_SIZE, blackKoProhibited);
  std::copy(other.whiteKoProhibited, other.whiteKoProhibited+Board::MAX_ARR_SIZE, whiteKoProhibited);
  koProhibitHash = other.koProhibitHash;
  koCapturesInEncore = other.koCapturesInEncore;
  std::copy(other.secondEncoreStartColors, other.secondEncoreStartColors+Board::MAX_ARR_SIZE, secondEncoreStartColors);
  whiteBonusScore = other.whiteBonusScore;
  isGameFinished = other.isGameFinished;
  winner = other.winner;
  finalWhiteMinusBlackScore = other.finalWhiteMinusBlackScore;
  isNoResult = other.isNoResult;

  return *this;
}

BoardHistory::BoardHistory(BoardHistory&& other) noexcept
 :rules(other.rules),
  moveHistory(std::move(other.moveHistory)),koHashHistory(std::move(other.koHashHistory)),
  koHistoryLastClearedBeginningMoveIdx(other.koHistoryLastClearedBeginningMoveIdx),
  recentBoards(other.recentBoards),
  currentRecentBoardIdx(other.currentRecentBoardIdx),
  consecutiveEndingPasses(other.consecutiveEndingPasses),
  hashesAfterBlackPass(std::move(other.hashesAfterBlackPass)),hashesAfterWhitePass(std::move(other.hashesAfterWhitePass)),
  encorePhase(other.encorePhase),koProhibitHash(other.koProhibitHash),
  koCapturesInEncore(std::move(other.koCapturesInEncore)),
  whiteBonusScore(other.whiteBonusScore),
  isGameFinished(other.isGameFinished),winner(other.winner),finalWhiteMinusBlackScore(other.finalWhiteMinusBlackScore),isNoResult(other.isNoResult)
{
  std::copy(other.wasEverOccupiedOrPlayed, other.wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, wasEverOccupiedOrPlayed);
  std::copy(other.superKoBanned, other.superKoBanned+Board::MAX_ARR_SIZE, superKoBanned);
  std::copy(other.blackKoProhibited, other.blackKoProhibited+Board::MAX_ARR_SIZE, blackKoProhibited);
  std::copy(other.whiteKoProhibited, other.whiteKoProhibited+Board::MAX_ARR_SIZE, whiteKoProhibited);
  std::copy(other.secondEncoreStartColors, other.secondEncoreStartColors+Board::MAX_ARR_SIZE, secondEncoreStartColors);
}

BoardHistory& BoardHistory::operator=(BoardHistory&& other) noexcept
{
  rules = other.rules;
  moveHistory = std::move(other.moveHistory);
  koHashHistory = std::move(other.koHashHistory);
  koHistoryLastClearedBeginningMoveIdx = other.koHistoryLastClearedBeginningMoveIdx;
  std::copy(other.recentBoards, other.recentBoards+NUM_RECENT_BOARDS, recentBoards);
  currentRecentBoardIdx = other.currentRecentBoardIdx;
  std::copy(other.wasEverOccupiedOrPlayed, other.wasEverOccupiedOrPlayed+Board::MAX_ARR_SIZE, wasEverOccupiedOrPlayed);
  std::copy(other.superKoBanned, other.superKoBanned+Board::MAX_ARR_SIZE, superKoBanned);
  consecutiveEndingPasses = other.consecutiveEndingPasses;
  hashesAfterBlackPass = std::move(other.hashesAfterBlackPass);
  hashesAfterWhitePass = std::move(other.hashesAfterWhitePass);
  encorePhase = other.encorePhase;
  std::copy(other.blackKoProhibited, other.blackKoProhibited+Board::MAX_ARR_SIZE, blackKoProhibited);
  std::copy(other.whiteKoProhibited, other.whiteKoProhibited+Board::MAX_ARR_SIZE, whiteKoProhibited);
  koProhibitHash = other.koProhibitHash;
  koCapturesInEncore = std::move(other.koCapturesInEncore);
  std::copy(other.secondEncoreStartColors, other.secondEncoreStartColors+Board::MAX_ARR_SIZE, secondEncoreStartColors);
  whiteBonusScore = other.whiteBonusScore;
  isGameFinished = other.isGameFinished;
  winner = other.winner;
  finalWhiteMinusBlackScore = other.finalWhiteMinusBlackScore;
  isNoResult = other.isNoResult;

  return *this;
}

void BoardHistory::clear(const Board& board, Player pla, const Rules& r) {
  rules = r;
  moveHistory.clear();
  koHashHistory.clear();
  koHistoryLastClearedBeginningMoveIdx = 0;

  for(int i = 0; i<NUM_RECENT_BOARDS; i++)
    recentBoards[i] = board;
  currentRecentBoardIdx = 0;

  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      wasEverOccupiedOrPlayed[loc] = (board.colors[loc] != C_EMPTY);
    }
  }

  std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
  consecutiveEndingPasses = 0;
  hashesAfterBlackPass.clear();
  hashesAfterWhitePass.clear();
  encorePhase = 0;
  std::fill(blackKoProhibited, blackKoProhibited+Board::MAX_ARR_SIZE, false);
  std::fill(whiteKoProhibited, whiteKoProhibited+Board::MAX_ARR_SIZE, false);
  koProhibitHash = Hash128();
  koCapturesInEncore.clear();
  std::fill(secondEncoreStartColors, secondEncoreStartColors+Board::MAX_ARR_SIZE, C_EMPTY);
  whiteBonusScore = 0;
  isGameFinished = false;
  winner = C_EMPTY;
  finalWhiteMinusBlackScore = 0.0f;
  isNoResult = false;

  if(rules.scoringRule == Rules::SCORING_TERRITORY) {
    //Chill 1 point for every move played
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == P_BLACK)
          whiteBonusScore += 1;
        else if(board.colors[loc] == P_WHITE)
          whiteBonusScore -= 1;
      }
    }
    //If white actually played extra moves that got captured so we don't see them,
    //then chill for those too
    int netWhiteCaptures = board.numWhiteCaptures - board.numBlackCaptures;
    whiteBonusScore -= netWhiteCaptures;
  }

  //Push hash for the new board state
  koHashHistory.push_back(getKoHash(rules,board,pla,encorePhase,koProhibitHash));
}

const Board& BoardHistory::getRecentBoard(int numMovesAgo) const {
  assert(numMoveAgo >= 0 && numMovesAgo < NUM_RECENT_BOARDS);
  int idx = (currentRecentBoardIdx + numMovesAgo) % NUM_RECENT_BOARDS;
  return recentBoards[idx];
}


void BoardHistory::setKomi(float newKomi) {
  rules.komi = newKomi;
  isGameFinished = false;
  winner = C_EMPTY;
  finalWhiteMinusBlackScore = 0.0f;
  isNoResult = false;
}

//If rootKoHashTable is provided, will take advantage of rootKoHashTable rather than search within the first
//rootKoHashTable->size() moves of koHashHistory.
//ALSO counts the most recent ko hash!
bool BoardHistory::koHashOccursInHistory(Hash128 koHash, const KoHashTable* rootKoHashTable) const {
  size_t start = 0;
  if(rootKoHashTable != NULL &&
     koHistoryLastClearedBeginningMoveIdx == rootKoHashTable->koHistoryLastClearedBeginningMoveIdx
  ) {
    size_t tableSize = rootKoHashTable->size();
    assert(tableSize < koHashHistory.size());
    if(rootKoHashTable->containsHash(koHash))
      return true;
    start = tableSize;
  }

  size_t koHashHistorySize = koHashHistory.size();
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
  if(rootKoHashTable != NULL &&
     koHistoryLastClearedBeginningMoveIdx == rootKoHashTable->koHistoryLastClearedBeginningMoveIdx
  ) {
    size_t tableSize = rootKoHashTable->size();
    assert(tableSize < koHashHistory.size());
    count += rootKoHashTable->numberOfOccurrencesOfHash(koHash);
    start = tableSize;
  }
  size_t koHashHistorySize = koHashHistory.size();
  for(size_t i = start; i < koHashHistorySize; i++)
    if(koHashHistory[i] == koHash)
      count++;
  return count;
}

float BoardHistory::currentSelfKomi(Player pla) const {
  float whiteKomi = whiteBonusScore + rules.komi;
  if(pla == P_WHITE)
    return whiteKomi;
  else if(pla == P_BLACK)
    return -whiteKomi;
  else {
    assert(false);
    return 0.0f;
  }
}

int BoardHistory::countAreaScoreWhiteMinusBlack(const Board& board) const {
  int score = 0;
  bool nonPassAliveStones = true;
  bool safeBigTerritories = true;
  bool unsafeBigTerritories = true;
  Color area[Board::MAX_ARR_SIZE];
  board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules.multiStoneSuicideLegal);
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

int BoardHistory::countTerritoryAreaScoreWhiteMinusBlack(const Board& board) const {
  int score = 0;
  bool nonPassAliveStones = false;
  bool safeBigTerritories = true;
  bool unsafeBigTerritories = false;
  Color area[Board::MAX_ARR_SIZE];
  board.calculateArea(area,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,rules.multiStoneSuicideLegal);
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(area[loc] == C_WHITE)
        score += 1;
      else if(area[loc] == C_BLACK)
        score -= 1;
      else {
        if(board.colors[loc] == C_WHITE && secondEncoreStartColors[loc] == C_WHITE)
          score += 1;
        if(board.colors[loc] == C_BLACK && secondEncoreStartColors[loc] == C_BLACK)
          score -= 1;
      }
    }
  }
  return score;
}

void BoardHistory::setKoProhibited(Player pla, Loc loc, bool b) {
  if(pla == P_BLACK) {
    if(blackKoProhibited[loc] != b) {
      blackKoProhibited[loc] = b;
      koProhibitHash ^= Board::ZOBRIST_KO_MARK_HASH[loc][pla];
    }
  }
  else if(pla == P_WHITE) {
    if(whiteKoProhibited[loc] != b) {
      whiteKoProhibited[loc] = b;
      koProhibitHash ^= Board::ZOBRIST_KO_MARK_HASH[loc][pla];
    }
  }
  else
    assert(false);
}

bool BoardHistory::isLegal(const Board& board, Loc moveLoc, Player movePla) const {
  //Moves in the encore on ko-prohibited spots are treated as pass-for-ko, so they are legal
  //They might also be simply not even ko-moves, if surrounding moves have caused the move on the
  //ko-prohibited spot to not be a ko-move due to now capturing more than one stone.
  if(encorePhase > 0 && moveLoc >= 0 && moveLoc < Board::MAX_ARR_SIZE) {
    if(movePla == P_BLACK && blackKoProhibited[moveLoc])
      return true;
    else if(movePla == P_WHITE && whiteKoProhibited[moveLoc])
      return true;
  }

  if(!board.isLegal(moveLoc,movePla,rules.multiStoneSuicideLegal))
    return false;
  if(superKoBanned[moveLoc])
    return false;

  return true;
}

void BoardHistory::makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla, const KoHashTable* rootKoHashTable) {
  Loc koLocBeforeMove = board.ko_loc;
  Hash128 posHashBeforeMove = board.pos_hash;

  //If somehow we're making a move after the game was ended, just clear those values and continue
  isGameFinished = false;
  winner = C_EMPTY;
  finalWhiteMinusBlackScore = 0.0f;
  isNoResult = false;

  //Handle pass-for-ko moves in the encore. Pass for ko lifts a ko prohibition and does nothing else.
  bool wasPassForKo = false;
  if(encorePhase > 0) {
    if((movePla == P_BLACK && blackKoProhibited[moveLoc] && board.wouldBeKoCapture(moveLoc,P_BLACK)) ||
       (movePla == P_WHITE && whiteKoProhibited[moveLoc] && board.wouldBeKoCapture(moveLoc,P_WHITE))) {
      setKoProhibited(movePla,moveLoc,false);
      wasPassForKo = true;
    }
  }
  //Otherwise handle regular moves
  if(!wasPassForKo) {
    board.playMoveAssumeLegal(moveLoc,movePla);

    if(encorePhase > 0) {
      //Update ko prohibitions and record that this was a ko capture
      if(board.ko_loc != Board::NULL_LOC) {
        setKoProhibited(getOpp(movePla),board.ko_loc,true);
        koCapturesInEncore.push_back(EncoreKoCapture(posHashBeforeMove,moveLoc,movePla));
      }
      //Unmark for the opponent all points that aren't ko moves
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          if(movePla == P_BLACK && whiteKoProhibited[loc] && !board.wouldBeKoCapture(loc,P_WHITE))
            setKoProhibited(getOpp(movePla),loc,false);
          if(movePla == P_WHITE && blackKoProhibited[loc] && !board.wouldBeKoCapture(loc,P_BLACK))
            setKoProhibited(getOpp(movePla),loc,false);
        }
      }
      //Unmark ko prohibitions on the board just played for oneself (clear dangling ko prohibitions
      //resulting from playing moves that used to be ko moves but are no longer due to the move
      //now capturing additional stones as a result).
      setKoProhibited(movePla,moveLoc,false);
    }
  }

  //Update recent boards
  currentRecentBoardIdx = (currentRecentBoardIdx + 1) % NUM_RECENT_BOARDS;
  recentBoards[currentRecentBoardIdx] = board;

  //Passes clear ko history in the main phase with spight ko rules and in the encore
  //This lifts bans in spight ko rules and lifts 3-fold-repetition checking in the encore for no-resultifying infinite cycles
  //They also clear in simple ko rules for the purpose of no-resulting long cycles, long cycles with passes do not no-result.
  if(moveLoc == Board::PASS_LOC && (encorePhase > 0 || rules.koRule == Rules::KO_SIMPLE || rules.koRule == Rules::KO_SPIGHT)) {
    koHashHistory.clear();
    koHistoryLastClearedBeginningMoveIdx = moveHistory.size()+1;
    //Does not clear hashesAfterBlackPass or hashesAfterWhitePass. Passes lift ko bans, but
    //still repeated positions after pass wnd the game or phase, which these arrays are used to check.
  }

  koHashHistory.push_back(getKoHash(rules,board,getOpp(movePla),encorePhase,koProhibitHash));
  moveHistory.push_back(Move(moveLoc,movePla));
  wasEverOccupiedOrPlayed[moveLoc] = true;

  //Mark all locations that are superko-illegal for the next player
  Player nextPla = getOpp(movePla);
  if(encorePhase <= 0 && rules.koRule != Rules::KO_SIMPLE) {
    assert(koProhibitHash == Hash128());
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        //Cannot be superko banned if it's not a legal move in the first place, or if there was never a stone there
        //was never played before, or we would already ban the move under simple ko
        if(board.colors[loc] != C_EMPTY || board.isIllegalSuicide(loc,nextPla,rules.multiStoneSuicideLegal) || loc == board.ko_loc)
          superKoBanned[loc] = false;
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

  //Update consecutiveEndingPasses
  if(moveLoc != Board::PASS_LOC)
    consecutiveEndingPasses = 0;
  else if(encorePhase > 0)
    consecutiveEndingPasses++;
  else {
    switch(rules.koRule) {
    case Rules::KO_SIMPLE:
      if(koLocBeforeMove == Board::NULL_LOC)
        consecutiveEndingPasses++;
      else
        consecutiveEndingPasses = 0;
      break;
    case Rules::KO_POSITIONAL:
    case Rules::KO_SITUATIONAL:
      consecutiveEndingPasses++;
      break;
    case Rules::KO_SPIGHT:
      consecutiveEndingPasses = 0;
      break;
    default:
      assert(false);
      break;
    }
  }

  //Check if we have a game-ending pass before updating hashesAfterBlackPass and hashesAfterWhitePass
  bool isSimpleSpightOrEncoreEndingPass = false;
  if(moveLoc == Board::PASS_LOC && (encorePhase > 0 || rules.koRule == Rules::KO_SIMPLE || rules.koRule == Rules::KO_SPIGHT)) {
    Hash128 lastHash = koHashHistory[koHashHistory.size()-1];
    if(movePla == P_BLACK && std::find(hashesAfterBlackPass.begin(), hashesAfterBlackPass.end(), lastHash) != hashesAfterBlackPass.end())
      isSimpleSpightOrEncoreEndingPass = true;
    if(movePla == P_WHITE && std::find(hashesAfterWhitePass.begin(), hashesAfterWhitePass.end(), lastHash) != hashesAfterWhitePass.end())
      isSimpleSpightOrEncoreEndingPass = true;
  }

  //Update hashesAfterBlackPass and hashesAfterWhitePass
  if(moveLoc == Board::PASS_LOC) {
    if(movePla == P_BLACK)
      hashesAfterBlackPass.push_back(koHashHistory[koHashHistory.size()-1]);
    else if(movePla == P_WHITE)
      hashesAfterWhitePass.push_back(koHashHistory[koHashHistory.size()-1]);
    else
      assert(false);
  }

  //Territory scoring - chill 1 point per move in main phase and first encore
  if(rules.scoringRule == Rules::SCORING_TERRITORY && encorePhase <= 1 && moveLoc != Board::PASS_LOC && !wasPassForKo) {
    if(movePla == P_BLACK)
      whiteBonusScore += 1;
    else if(movePla == P_WHITE)
      whiteBonusScore -= 1;
    else
      assert(false);
  }

  //Phase transitions and game end
  if(consecutiveEndingPasses >= 2 || isSimpleSpightOrEncoreEndingPass) {
    if(rules.scoringRule == Rules::SCORING_AREA) {
      assert(encorePhase <= 0);
      int boardScore = countAreaScoreWhiteMinusBlack(board);
      finalWhiteMinusBlackScore = boardScore + whiteBonusScore + rules.komi;
      if(finalWhiteMinusBlackScore > 0.0f)
        winner = C_WHITE;
      else if(finalWhiteMinusBlackScore < 0.0f)
        winner = C_BLACK;
      isNoResult = false;
      isGameFinished = true;
    }
    else if(rules.scoringRule == Rules::SCORING_TERRITORY) {
      if(encorePhase >= 2) {
        int boardScore = countTerritoryAreaScoreWhiteMinusBlack(board);
        finalWhiteMinusBlackScore = boardScore + whiteBonusScore + rules.komi;
        if(finalWhiteMinusBlackScore > 0.0f)
          winner = C_WHITE;
        else if(finalWhiteMinusBlackScore < 0.0f)
          winner = C_BLACK;
        isNoResult = false;
        isGameFinished = true;
      }
      else {
        encorePhase += 1;
        if(encorePhase == 2)
          std::copy(board.colors, board.colors+Board::MAX_ARR_SIZE, secondEncoreStartColors);

        std::fill(superKoBanned, superKoBanned+Board::MAX_ARR_SIZE, false);
        consecutiveEndingPasses = 0;
        hashesAfterBlackPass.clear();
        hashesAfterWhitePass.clear();
        std::fill(blackKoProhibited, blackKoProhibited+Board::MAX_ARR_SIZE, false);
        std::fill(whiteKoProhibited, whiteKoProhibited+Board::MAX_ARR_SIZE, false);
        koProhibitHash = Hash128();
        koCapturesInEncore.clear();

        koHashHistory.clear();
        koHistoryLastClearedBeginningMoveIdx = moveHistory.size();
        koHashHistory.push_back(getKoHash(rules,board,getOpp(movePla),encorePhase,koProhibitHash));
      }
    }
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
   koHistoryLastClearedBeginningMoveIdx(0)
{
  idxTable = new uint16_t[TABLE_SIZE];
}
KoHashTable::~KoHashTable() {
  delete[] idxTable;
}

size_t KoHashTable::size() const {
  return koHashHistorySortedByLowBits.size();
}

void KoHashTable::recompute(const BoardHistory& history) {
  koHashHistorySortedByLowBits = history.koHashHistory;
  koHistoryLastClearedBeginningMoveIdx = history.koHistoryLastClearedBeginningMoveIdx;

  auto cmpFirstByLowBits = [](const Hash128& a, const Hash128& b) {
    if((a.hash0 & TABLE_MASK) < (b.hash0 & TABLE_MASK))
      return true;
    if((a.hash0 & TABLE_MASK) > (b.hash0 & TABLE_MASK))
      return false;
    return a < b;
  };

  std::sort(koHashHistorySortedByLowBits.begin(),koHashHistorySortedByLowBits.end(),cmpFirstByLowBits);

  //Just in case, since we're using 16 bits for indices.
  assert(koHashHistorySortedByLowBits.size() < 30000);
  uint16_t size = (uint16_t)koHashHistorySortedByLowBits.size();

  uint16_t idx = 0;
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


