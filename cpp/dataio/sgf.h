#ifndef SGF_H_
#define SGF_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../dataio/trainingwrite.h"

STRUCT_NAMED_TRIPLE(uint8_t,x,uint8_t,y,Player,pla,MoveNoBSize);
STRUCT_NAMED_PAIR(int,x,int,y,XYSize);

struct SgfNode {
  map<string,vector<string>>* props;
  MoveNoBSize move;

  SgfNode();
  ~SgfNode();

  SgfNode(const SgfNode& other);
  SgfNode(SgfNode&& other) noexcept;

  SgfNode& operator=(const SgfNode&);
  SgfNode& operator=(SgfNode&&) noexcept;

  bool hasProperty(const char* key) const;
  string getSingleProperty(const char* key) const;

  bool hasPlacements() const;
  void accumPlacements(vector<Move>& moves, int xSize, int ySize) const;
  void accumMoves(vector<Move>& moves, int xSize, int ySize) const;

  Rules getRules(const Rules& defaultRules) const;
};

struct Sgf {
  string fileName;
  vector<SgfNode*> nodes;
  vector<Sgf*> children;
  Hash128 hash;

  Sgf();
  ~Sgf();

  Sgf(const Sgf&) = delete;
  Sgf& operator=(const Sgf&) = delete;

  static Sgf* parse(const string& str);
  static Sgf* loadFile(const string& file);
  static vector<Sgf*> loadFiles(const vector<string>& files);
  static vector<Sgf*> loadSgfsFile(const string& file);
  static vector<Sgf*> loadSgfsFiles(const vector<string>& files);

  XYSize getXYSize() const;
  float getKomi() const;
  Rules getRules(const Rules& defaultRules) const;

  void getPlacements(vector<Move>& moves, int xSize, int ySize) const;
  void getMoves(vector<Move>& moves, int xSize, int ySize) const;

  int depth() const;

  private:
  void getMovesHelper(vector<Move>& moves, int xSize, int ySize) const;

};

struct CompactSgf {
  string fileName;
  SgfNode rootNode;
  vector<Move> placements;
  vector<Move> moves;
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

  static CompactSgf* parse(const string& str);
  static CompactSgf* loadFile(const string& file);
  static vector<CompactSgf*> loadFiles(const vector<string>& files);

  void setupInitialBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist);
  void setupBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int turnNumber);
};

namespace WriteSgf {
  //Write an SGF with no newlines to the given ostream.
  //If startTurnIdx >= 0, write a comment in the SGF root node indicating startTurnIdx, so as to
  //indicate the index of the first turn that should be used for training data. (0 means the whole SGF, 1 means skipping black's first move, etc).
  //If valueTargets is not NULL, also write down after each move the MCTS values following that search move.
  void writeSgf(
    ostream& out, const string& bName, const string& wName, const Rules& rules,
    const BoardHistory& hist,
    const FinishedGameData* gameData
  );

  //If hist is a finished game, print the result to out, else do nothing
  void printGameResult(ostream& out, const BoardHistory& hist);
}

#endif
