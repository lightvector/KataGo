#include "../search/patternbonustable.h"

#include "../core/rand.h"
#include "../core/multithread.h"
#include "../search/localpattern.h"

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
  patternHasher.init(5,5,rand);

  rand.init("Reseed PatternBonusTable zobrist so that zobrists don't change when Board::MAX_ARR_SIZE changes");
  for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    ZOBRIST_MOVE_LOCS[i] = Hash128(h0,h1);
  }
  isInited = true;
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

PatternBonusEntry PatternBonusTable::get(Player pla, Loc prevMoveLoc, const Board& board) const {
  //We don't want to over-trigger this on a ko that repeats the same pattern over and over
  //So we just disallow this on ko fight
  if(board.ko_loc != Board::NULL_LOC)
    return PatternBonusEntry();

  Hash128 hash = ZOBRIST_MOVE_LOCS[prevMoveLoc];
  hash ^= patternHasher.getHash(board,prevMoveLoc,pla);
  auto subMapIdx = hash.hash0 % entries.size();

  const std::map<Hash128,PatternBonusEntry>& subMap = entries[subMapIdx];
  auto iter = subMap.find(hash);
  if(iter == subMap.end())
    return PatternBonusEntry();
  return iter->second;
}

void PatternBonusTable::addBonus(Player pla, Loc prevMoveLoc, const Board& board, double bonus, std::set<Hash128>& hashesThisGame) {
  //We don't want to over-trigger this on a ko that repeats the same pattern over and over
  //So we just disallow this on ko fight
  if(board.ko_loc != Board::NULL_LOC)
    return;

  Hash128 hash = ZOBRIST_MOVE_LOCS[prevMoveLoc];
  hash ^= patternHasher.getHash(board,prevMoveLoc,pla);
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
  for(size_t i = 0; i<hist.moveHistory.size(); i++) {
    Player pla = hist.moveHistory[i].pla;
    Loc loc = hist.moveHistory[i].loc;
    bool suc = hist.makeBoardMoveTolerant(board, loc, pla);
    if(!suc)
      break;
    if(onlyPla == C_EMPTY || onlyPla == pla)
      addBonus(pla, loc, board, bonus, hashesThisGame);
  }
}

