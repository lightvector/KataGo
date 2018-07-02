
#include <algorithm>
#include "core/global.h"
#include "boardhistory.h"

static Hash128 getKoHash(const Rules& rules, const FastBoard& board, Player pla) {
  if(rules.koRule == Rules::KO_SITUATIONAL)
    return board.pos_hash ^ FastBoard::ZOBRIST_PLAYER_HASH[pla];
  else
    return board.pos_hash;
}

BoardHistory::BoardHistory()
  :moveHistory(),koHashHistory()
{
  for(int i = 0; i<FastBoard::MAX_ARR_SIZE; i++)
    wasEverOccupiedOrPlayed[i] = false;
}

BoardHistory::~BoardHistory()
{}

BoardHistory::BoardHistory(const Rules& rules, const FastBoard& board, Player pla)
  :moveHistory(),koHashHistory()
{
  for(int i = 0; i<FastBoard::MAX_ARR_SIZE; i++)
    wasEverOccupiedOrPlayed[i] = false;

  clear(rules,board,pla);
}

BoardHistory::BoardHistory(const BoardHistory& other)
  :moveHistory(other.moveHistory),koHashHistory(other.koHashHistory)
{
  for(int i = 0; i<FastBoard::MAX_ARR_SIZE; i++)
    wasEverOccupiedOrPlayed[i] = other.wasEverOccupiedOrPlayed[i];
}


BoardHistory& BoardHistory::operator=(const BoardHistory& other)
{
  moveHistory = other.moveHistory;
  koHashHistory = other.koHashHistory;
  for(int i = 0; i<FastBoard::MAX_ARR_SIZE; i++)
    wasEverOccupiedOrPlayed[i] = other.wasEverOccupiedOrPlayed[i];
  return *this;
}

BoardHistory::BoardHistory(BoardHistory&& other) noexcept
  :moveHistory(std::move(other.moveHistory)),koHashHistory(std::move(other.koHashHistory))
{
  for(int i = 0; i<FastBoard::MAX_ARR_SIZE; i++)
    wasEverOccupiedOrPlayed[i] = other.wasEverOccupiedOrPlayed[i];
}

BoardHistory& BoardHistory::operator=(BoardHistory&& other) noexcept
{
  moveHistory = std::move(other.moveHistory);
  koHashHistory = std::move(other.koHashHistory);
  for(int i = 0; i<FastBoard::MAX_ARR_SIZE; i++)
    wasEverOccupiedOrPlayed[i] = other.wasEverOccupiedOrPlayed[i];
}

void BoardHistory::clear(const Rules& rules, const FastBoard& board, Player pla) {
  moveHistory.clear();
  koHashHistory.clear();
  koHashHistory.push_back(getKoHash(rules,board,pla));

  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      wasEverOccupiedOrPlayed[loc] = (board.colors[loc] != C_EMPTY);
    }
  }
}

void BoardHistory::updateAfterMove(const Rules& rules, const FastBoard& board, Loc moveLoc, Player movePla) {
  koHashHistory.push_back(getKoHash(rules,board,getOpp(movePla)));
  moveHistory.push_back(Move(moveLoc,movePla));
  wasEverOccupiedOrPlayed[moveLoc] = true;
}



KoHashTable::KoHashTable()
  :koHashHistorySortedByLowBits()
{
  idxTable = new uint16_t[TABLE_SIZE];
}
KoHashTable::~KoHashTable() {
  delete[] idxTable;
}

void KoHashTable::recompute(const BoardHistory& history) {
  koHashHistorySortedByLowBits = history.koHashHistory;

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


