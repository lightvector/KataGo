#include "../search/patternbonustable.h"

#include "../core/rand.h"
#include "../core/multithread.h"
#include "../core/fileutils.h"
#include "../neuralnet/nninputs.h"
#include "../search/localpattern.h"
#include "../dataio/sgf.h"
#include "../dataio/files.h"

using namespace std;

static std::mutex initMutex;
static std::atomic<bool> isInited(false);
static LocalPatternHasher patternHasher;
static Hash128 ZOBRIST_MOVE_LOCS[Board::MAX_ARR_SIZE];

static void initIfNeeded() {
  if(isInited)
    return;
  std::lock_guard<std::mutex> lock(initMutex);
  if(isInited)
    return;
  Rand rand("PatternBonusTable ZOBRIST STUFF");
  patternHasher.init(9,9,rand);

  rand.init("Reseed PatternBonusTable zobrist so that zobrists don't change when Board::MAX_ARR_SIZE changes");
  for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    ZOBRIST_MOVE_LOCS[i] = Hash128(h0,h1);
  }
  isInited = true;
}

PatternBonusTable::PatternBonusTable() {
  initIfNeeded();
  entries.resize(1024);
}
PatternBonusTable::PatternBonusTable(int32_t numShards) {
  initIfNeeded();
  entries.resize(numShards);
}
PatternBonusTable::PatternBonusTable(const PatternBonusTable& other) {
  initIfNeeded();
  entries = other.entries;
}
PatternBonusTable::~PatternBonusTable() {
}

Hash128 PatternBonusTable::getHash(Player pla, Loc moveLoc, const Board& board) const {
  //We don't want to over-trigger this on a ko that repeats the same pattern over and over
  //So we just disallow this on ko fight
  //Also no bonuses for passing.
  if(moveLoc == Board::NULL_LOC || moveLoc == Board::PASS_LOC || board.wouldBeKoCapture(moveLoc,pla))
    return Hash128();

  Hash128 hash = patternHasher.getHash(board,moveLoc,pla);
  hash ^= ZOBRIST_MOVE_LOCS[moveLoc];
  hash ^= Board::ZOBRIST_SIZE_X_HASH[board.x_size];
  hash ^= Board::ZOBRIST_SIZE_Y_HASH[board.y_size];

  return hash;
}

PatternBonusEntry PatternBonusTable::get(Hash128 hash) const {
  //Hash 0 indicates to not do anything. If anything legit collides with it, then it will do nothing
  //but this should be very rare.
  if(hash == Hash128())
    return PatternBonusEntry();

  auto subMapIdx = hash.hash0 % entries.size();

  const std::map<Hash128,PatternBonusEntry>& subMap = entries[subMapIdx];
  auto iter = subMap.find(hash);
  if(iter == subMap.end())
    return PatternBonusEntry();
  return iter->second;
}

PatternBonusEntry PatternBonusTable::get(Player pla, Loc moveLoc, const Board& board) const {
  Hash128 hash = getHash(pla, moveLoc, board);
  return get(hash);
}

void PatternBonusTable::addBonus(Player pla, Loc moveLoc, const Board& board, double bonus, int symmetry, bool flipColors, std::set<Hash128>& hashesThisGame) {
  //We don't want to over-trigger this on a ko that repeats the same pattern over and over
  //So we just disallow this on ko fight
  //Also no bonuses for passing.
  if(moveLoc == Board::NULL_LOC || moveLoc == Board::PASS_LOC || board.wouldBeKoCapture(moveLoc,pla))
    return;

  Hash128 hash = patternHasher.getHashWithSym(board,moveLoc,pla,symmetry,flipColors);
  hash ^= ZOBRIST_MOVE_LOCS[SymmetryHelpers::getSymLoc(moveLoc,board,symmetry)];
  if(SymmetryHelpers::isTranspose(symmetry)) {
    hash ^= Board::ZOBRIST_SIZE_X_HASH[board.y_size];
    hash ^= Board::ZOBRIST_SIZE_Y_HASH[board.x_size];
  }
  else {
    hash ^= Board::ZOBRIST_SIZE_X_HASH[board.x_size];
    hash ^= Board::ZOBRIST_SIZE_Y_HASH[board.y_size];
  }

  if(contains(hashesThisGame,hash))
    return;
  hashesThisGame.insert(hash);

  auto subMapIdx = hash.hash0 % entries.size();

  std::map<Hash128,PatternBonusEntry>& subMap = entries[subMapIdx];
  subMap[hash].utilityBonus += bonus;
}

void PatternBonusTable::addBonusForGameMoves(const BoardHistory& game, double bonus) {
  addBonusForGameMoves(game,bonus,C_EMPTY);
}

void PatternBonusTable::addBonusForGameMoves(const BoardHistory& game, double bonus, Player onlyPla) {
  std::set<Hash128> hashesThisGame;
  Board board = game.initialBoard;
  BoardHistory hist(board, game.initialPla, game.rules, game.initialEncorePhase);
  for(size_t i = 0; i<game.moveHistory.size(); i++) {
    Player pla = game.moveHistory[i].pla;
    Loc loc = game.moveHistory[i].loc;
    //We first play the move to see if it's a move we can accept
    bool suc = hist.makeBoardMoveTolerant(board, loc, pla);
    if(!suc)
      break;
    if(onlyPla == C_EMPTY || onlyPla == pla) {
      for(int flipColors = 0; flipColors < 2; flipColors++) {
        for(int symmetry = 0; symmetry < 8; symmetry++) {
          //getRecentBoard(1) - the convention is to pattern match on the board BEFORE the move is played.
          //This is also more pricipled than convening on the board after since with different captures, moves
          //may have different effects even while leading to the same position.
          addBonus(pla, loc, hist.getRecentBoard(1), bonus, symmetry, (bool)flipColors, hashesThisGame);
        }
      }
    }
  }
}

void PatternBonusTable::avoidRepeatedSgfMoves(
  const vector<string>& sgfsDirsOrFiles,
  double penalty,
  double decayOlderFilesLambda,
  int64_t minTurnNumber,
  size_t maxFiles,
  const vector<string>& allowedPlayerNames,
  Logger& logger,
  const string& logSource
) {
  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirsOrFiles(sgfsDirsOrFiles,sgfFiles);
  FileHelpers::sortNewestToOldest(sgfFiles);

  double factor = 1.0;
  for(size_t i = 0; i<sgfFiles.size() && i < maxFiles; i++) {
    const string& fileName = sgfFiles[i];
    Sgf* sgf = NULL;
    try {
      sgf = Sgf::loadFile(fileName);
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + fileName + ": " + e.what());
      continue;
    }

    bool blackOkay = allowedPlayerNames.size() <= 0 || contains(allowedPlayerNames, sgf->getPlayerName(P_BLACK));
    bool whiteOkay = allowedPlayerNames.size() <= 0 || contains(allowedPlayerNames, sgf->getPlayerName(P_WHITE));

    std::set<Hash128> hashesThisGame;

    std::function<void(Sgf::PositionSample&, const BoardHistory&, const string&)> posHandler = [&](
      Sgf::PositionSample& posSample, const BoardHistory& hist, const string& comments
    ) {
      (void)posSample;
      if(comments.size() > 0 && comments.find("%SKIP%") != string::npos)
        return;
      if(hist.moveHistory.size() <= 0)
        return;
      if(hist.moveHistory.size() < minTurnNumber)
        return;
      Loc moveLoc = hist.moveHistory[hist.moveHistory.size()-1].loc;
      Player movePla = hist.moveHistory[hist.moveHistory.size()-1].pla;
      if(movePla == P_BLACK && !blackOkay)
        return;
      if(movePla == P_WHITE && !whiteOkay)
        return;

      for(int flipColorsInt = 0; flipColorsInt < 2; flipColorsInt++) {
        for(int symmetry = 0; symmetry < 8; symmetry++) {
          //getRecentBoard(1) - the convention is to pattern match on the board BEFORE the move is played.
          //This is also more pricipled than convening on the board after since with different captures, moves
          //may have different effects even while leading to the same position.
          bool flipColors = (bool)flipColorsInt;
          Player symPla = flipColors ? getOpp(movePla) : movePla;
          double bonus = symPla == P_WHITE ? -penalty*factor : penalty*factor;
          addBonus(movePla, moveLoc, hist.getRecentBoard(1), bonus, symmetry, flipColors, hashesThisGame);
        }
      }
    };

    bool hashComments = true;
    bool hashParent = true;
    bool flipIfPassOrWFirst = false;
    bool allowGameOver = false;
    std::set<Hash128> uniqueHashes;
    sgf->iterAllUniquePositions(uniqueHashes, hashComments, hashParent, flipIfPassOrWFirst, allowGameOver, NULL, posHandler);
    logger.write("Added " + Global::uint64ToString(hashesThisGame.size()) + " shapes to penalize repeats for " + logSource + " from " + fileName);

    delete sgf;
    factor *= decayOlderFilesLambda;
  }
}


void PatternBonusTable::avoidRepeatedPosMovesAndDeleteExcessFiles(
  const vector<string>& posesDirsToLoadAndPrune,
  double penalty,
  double decayOlderPosesLambda,
  int64_t minTurnNumber,
  int64_t maxTurnNumber,
  size_t maxPoses,
  Logger& logger,
  const string& logSource
) {
  vector<string> posFiles;
  FileHelpers::collectPosesFromDirs(posesDirsToLoadAndPrune,posFiles);
  FileHelpers::sortNewestToOldest(posFiles);

  size_t numPosesUsed = 0;
  size_t numPosesInvalid = 0;
  size_t numPosLoadErrors = 0; //May be due to concurrent access and pruning of the dir, that's fine, but we count it.
  double factor = 1.0;

  Sgf::PositionSample posSample;
  size_t i = 0;
  for(; i<posFiles.size(); i++) {
    if(numPosesUsed >= maxPoses)
      break;
    std::set<Hash128> hashesThisGame;
    const string& fileName = posFiles[i];
    vector<string> lines = FileUtils::readFileLines(fileName,'\n');
    for(size_t j = 0; j<lines.size(); j++) {
      string line = Global::trim(lines[j]);
      if(line.size() > 0) {
        try {
          posSample = Sgf::PositionSample::ofJsonLine(line);
        }
        catch(const StringError& err) {
          (void)err;
          numPosLoadErrors += 1;
          continue;
        }

        const bool isMultiStoneSuicideLegal = true;
        int64_t turnNumber = posSample.getCurrentTurnNumber();
        if(
          turnNumber < minTurnNumber ||
          turnNumber > maxTurnNumber ||
          posSample.moves.size() != 0 || // Right now auto pattern avoid expects moveless records
          !posSample.board.isLegal(posSample.hintLoc, posSample.nextPla, isMultiStoneSuicideLegal)
        ) {
          numPosesInvalid += 1;
          continue;
        }

        for(int flipColorsInt = 0; flipColorsInt < 2; flipColorsInt++) {
          for(int symmetry = 0; symmetry < 8; symmetry++) {
            //getRecentBoard(1) - the convention is to pattern match on the board BEFORE the move is played.
            //This is also more pricipled than convening on the board after since with different captures, moves
            //may have different effects even while leading to the same position.
            bool flipColors = (bool)flipColorsInt;
            Player symPla = flipColors ? getOpp(posSample.nextPla) : posSample.nextPla;
            double bonus = symPla == P_WHITE ? -penalty*factor : penalty*factor;
            addBonus(posSample.nextPla, posSample.hintLoc, posSample.board, bonus, symmetry, flipColors, hashesThisGame);
          }
        }
        numPosesUsed += 1;
        factor *= decayOlderPosesLambda;
      }
    }
  }
  for(; i<posFiles.size(); i++) {
    logger.write("Removing old pos file: " + posFiles[i]);
    FileUtils::tryRemoveFile(posFiles[i]);
  }

  logger.write("Loaded avoid poses from " + logSource);
  logger.write("numPosesUsed = " + Global::uint64ToString(numPosesUsed));
  logger.write("numPosesInvalid = " + Global::uint64ToString(numPosesInvalid));
  logger.write("numPosLoadErrors = " + Global::uint64ToString(numPosLoadErrors));
}
