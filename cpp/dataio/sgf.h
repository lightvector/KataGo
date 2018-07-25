#ifndef SGF_H_
#define SGF_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../game/board.h"
#include "../game/boardhistory.h"

STRUCT_NAMED_TRIPLE(uint8_t,x,uint8_t,y,Player,pla,MoveNoBSize);

struct SgfNode {
  map<string,vector<string>>* props;
  MoveNoBSize move;

  SgfNode();
  SgfNode(const SgfNode& other);
  ~SgfNode();

  bool hasProperty(const char* key) const;
  string getSingleProperty(const char* key) const;

  bool hasPlacements() const;
  void accumPlacements(vector<Move>& moves, int bSize) const;
  void accumMoves(vector<Move>& moves, int bSize) const;
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

  int getBSize() const;
  float getKomi() const;

  void getPlacements(vector<Move>& moves, int bSize) const;
  void getMoves(vector<Move>& moves, int bSize) const;

  int depth() const;

  private:
  void getMovesHelper(vector<Move>& moves, int bSize) const;

};

struct CompactSgf {
  string fileName;
  SgfNode rootNode;
  vector<Move> placements;
  vector<Move> moves;
  int bSize;
  int depth;
  float komi;
  Hash128 hash;

  CompactSgf(const Sgf* sgf);
  ~CompactSgf();

  CompactSgf(const CompactSgf&) = delete;
  CompactSgf& operator=(const CompactSgf&) = delete;

  static CompactSgf* loadFile(const string& file);
  static vector<CompactSgf*> loadFiles(const vector<string>& files);
};

namespace WriteSgf {
  void writeSgf(
    ostream& out, const string& bName, const string& wName, const Rules& rules,
    const Board& initialBoard, const BoardHistory& hist
  );
}

#endif
