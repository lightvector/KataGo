#ifndef SEARCH_PATTERNBONUSTABLE_H
#define SEARCH_PATTERNBONUSTABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../game/boardhistory.h"

struct PatternBonusEntry {
  double utilityBonus = 0.0;
};

struct PatternBonusTable {
  std::vector<std::map<Hash128,PatternBonusEntry>> entries;

  PatternBonusTable(int32_t numShards);
  PatternBonusTable(const PatternBonusTable& other);
  ~PatternBonusTable();

  PatternBonusEntry get(Player pla, Loc prevMoveLoc, const Board& board) const;
  void addBonus(Player pla, Loc prevMoveLoc, const Board& board, double bonus, std::set<Hash128>& hashesThisGame);
  void addBonusForGameMoves(const BoardHistory& game, double bonus);
  void addBonusForGameMoves(const BoardHistory& game, double bonus, Player onlyPla);
};

#endif


