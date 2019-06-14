#ifndef DATAIO_SGF_H_
#define DATAIO_SGF_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../dataio/trainingwrite.h"
#include "../game/board.h"
#include "../game/boardhistory.h"

STRUCT_NAMED_TRIPLE(uint8_t,x,uint8_t,y,Player,pla,MoveNoBSize);
STRUCT_NAMED_PAIR(int,x,int,y,XYSize);

struct SgfNode {
  std::map<std::string,std::vector<std::string>>* props;
  MoveNoBSize move;

  SgfNode();
  ~SgfNode();

  SgfNode(const SgfNode& other);
  SgfNode(SgfNode&& other) noexcept;

  SgfNode& operator=(const SgfNode&);
  SgfNode& operator=(SgfNode&&) noexcept;

  bool hasProperty(const char* key) const;
  std::string getSingleProperty(const char* key) const;

  bool hasPlacements() const;
  void accumPlacements(std::vector<Move>& moves, int xSize, int ySize) const;
  void accumMoves(std::vector<Move>& moves, int xSize, int ySize) const;

  Color getPLSpecifiedColor() const;
  Rules getRules(const Rules& defaultRules) const;
};

struct Sgf {
  std::string fileName;
  std::vector<SgfNode*> nodes;
  std::vector<Sgf*> children;
  Hash128 hash;

  Sgf();
  ~Sgf();

  Sgf(const Sgf&) = delete;
  Sgf& operator=(const Sgf&) = delete;

  static Sgf* parse(const std::string& str);
  static Sgf* loadFile(const std::string& file);
  static std::vector<Sgf*> loadFiles(const std::vector<std::string>& files);
  static std::vector<Sgf*> loadSgfsFile(const std::string& file);
  static std::vector<Sgf*> loadSgfsFiles(const std::vector<std::string>& files);

  XYSize getXYSize() const;
  float getKomi() const;
  Rules getRules(const Rules& defaultRules) const;

  void getPlacements(std::vector<Move>& moves, int xSize, int ySize) const;
  void getMoves(std::vector<Move>& moves, int xSize, int ySize) const;

  int depth() const;

  private:
  void getMovesHelper(std::vector<Move>& moves, int xSize, int ySize) const;

};

struct CompactSgf {
  std::string fileName;
  SgfNode rootNode;
  std::vector<Move> placements;
  std::vector<Move> moves;
  int xSize;
  int ySize;
  int depth;
  float komi;
  Hash128 hash;

  CompactSgf(const Sgf* sgf);
  CompactSgf(Sgf&& sgf);
  ~CompactSgf();

  CompactSgf(const CompactSgf&) = delete;
  CompactSgf& operator=(const CompactSgf&) = delete;

  static CompactSgf* parse(const std::string& str);
  static CompactSgf* loadFile(const std::string& file);
  static std::vector<CompactSgf*> loadFiles(const std::vector<std::string>& files);

  Rules getRulesFromSgf(const Rules& defaultRules);
  void setupInitialBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist);
  void setupBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int turnNumber);
};

namespace WriteSgf {
  //Write an SGF with no newlines to the given ostream.
  //If startTurnIdx >= 0, write a comment in the SGF root node indicating startTurnIdx, so as to
  //indicate the index of the first turn that should be used for training data. (0 means the whole SGF, 1 means skipping black's first move, etc).
  //If valueTargets is not NULL, also write down after each move the MCTS values following that search move.
  void writeSgf(
    std::ostream& out, const std::string& bName, const std::string& wName, const Rules& rules,
    const BoardHistory& hist,
    const FinishedGameData* gameData
  );

  //If hist is a finished game, print the result to out, else do nothing
  void printGameResult(std::ostream& out, const BoardHistory& hist);
}

#endif  // DATAIO_SGF_H_
