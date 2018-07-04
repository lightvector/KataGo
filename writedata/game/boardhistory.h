#ifndef BOARDHISTORY_H_
#define BOARDHISTORY_H_

#include "core/global.h"
#include "core/hash.h"
#include "fastboard.h"
#include "rules.h"

//A data structure enabling fast checking if a move would be illegal due to superko.
struct BoardHistory {
  //Chronological history of moves
  vector<Move> moveHistory;
  //Chronological history of hashes, including the latest board's hash.
  vector<Hash128> koHashHistory;

  //Did this board location ever have a stone there before, or was it ever played?
  //(Also includes locations of suicides)
  bool wasEverOccupiedOrPlayed[FastBoard::MAX_ARR_SIZE];

  BoardHistory();
  ~BoardHistory();

  BoardHistory(const Rules& rules, const FastBoard& board, Player pla);

  BoardHistory(const BoardHistory& other);
  BoardHistory& operator=(const BoardHistory& other);

  BoardHistory(BoardHistory&& other) noexcept;
  BoardHistory& operator=(BoardHistory&& other) noexcept;

  void clear(const Rules& rules, const FastBoard& board, Player pla);

  void updateAfterMove(const Rules& rules, const FastBoard& board, Loc moveLoc, Player movePla);

};

struct KoHashTable {
  uint16_t* idxTable;
  vector<Hash128> koHashHistorySortedByLowBits;

  static const int TABLE_SIZE = 1 << 10;
  static const uint64_t TABLE_MASK = TABLE_SIZE-1;

  KoHashTable();
  ~KoHashTable();

  KoHashTable(const KoHashTable& other) = delete;
  KoHashTable& operator=(const KoHashTable& other) = delete;
  KoHashTable(KoHashTable&& other) = delete;
  KoHashTable& operator=(KoHashTable&& other) = delete;

  void recompute(const BoardHistory& history);
  bool containsHash(Hash128 hash) const;

};


#endif
