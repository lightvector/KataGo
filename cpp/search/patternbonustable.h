#ifndef SEARCH_PATTERNBONUSTABLE_H
#define SEARCH_PATTERNBONUSTABLE_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/logger.h"
#include "../game/boardhistory.h"

// All bonuses are bonuses to white's utility for the pattern occuring on the board.
struct PatternBonusEntry {
  double utilityBonus = 0.0;
};

struct PatternBonusTable {
  std::vector<std::map<Hash128,PatternBonusEntry>> entries;

  PatternBonusTable();
  PatternBonusTable(int32_t numShards);
  PatternBonusTable(const PatternBonusTable& other);
  ~PatternBonusTable();

  // The board specified here is expected to be the board BEFORE the move is played.
  PatternBonusEntry get(Player pla, Loc moveLoc, const Board& board) const;
  PatternBonusEntry get(Hash128 hash) const;
  // The board specified here is expected to be the board BEFORE the move is played.
  Hash128 getHash(Player pla, Loc moveLoc, const Board& board) const;

  // All bonuses are bonuses to white's utility for the pattern occuring on the board.
  // The board specified here is expected to be the board BEFORE the move is played.
  void addBonus(Player pla, Loc moveLoc, const Board& board, double bonus, int symmetry, bool flipColors, std::set<Hash128>& hashesThisGame);

  void addBonusForGameMoves(const BoardHistory& game, double bonus);
  void addBonusForGameMoves(const BoardHistory& game, double bonus, Player onlyPla);

  void avoidRepeatedSgfMoves(
    const std::vector<std::string>& sgfsDirsOrFiles,
    double penalty,
    double decayOlderFilesLambda,
    int minTurnNumber,
    size_t maxFiles,
    const std::vector<std::string>& allowedPlayerNames,
    Logger& logger
  );
};

#endif


