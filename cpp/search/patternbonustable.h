#ifndef SEARCH_PATTERNBONUSTABLE_H
#define SEARCH_PATTERNBONUSTABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../game/boardhistory.h"

//All bonuses are bonuses to white's utility for the pattern occuring on the board.
struct PatternBonusEntry {
  double utilityBonus = 0.0;
};

struct PatternBonusTable {
  std::vector<std::map<Hash128,PatternBonusEntry>> entries;

  PatternBonusTable();
  PatternBonusTable(int32_t numShards);
  PatternBonusTable(const PatternBonusTable& other);
  ~PatternBonusTable();

  PatternBonusEntry get(Player pla, Loc prevMoveLoc, const Board& board) const;
  PatternBonusEntry get(Hash128 hash) const;
  Hash128 getHash(Player pla, Loc prevMoveLoc, const Board& board) const;

  //All bonuses are bonuses to white's utility for the pattern occuring on the board.
  void addBonus(Player pla, Loc prevMoveLoc, const Board& board, double bonus, int symmetry, std::set<Hash128>& hashesThisGame);
  void addBonusForGameMoves(const BoardHistory& game, double bonus);
  void addBonusForGameMoves(const BoardHistory& game, double bonus, Player onlyPla);
};

#endif


