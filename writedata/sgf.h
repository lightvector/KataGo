#ifndef SGF_H_
#define SGF_H_

#include "core/global.h"
#include "fastboard.h"

STRUCT_NAMED_PAIR(Loc,loc,Player,pla,Move);
STRUCT_NAMED_TRIPLE(uint8_t,x,uint8_t,y,Player,pla,MoveNoBSize);
struct Hash128 {
  uint64_t hash0;
  uint64_t hash1;
  inline Hash128(): hash0(), hash1() {}
  inline Hash128(uint64_t h0, uint64_t h1): hash0(h0), hash1(h1) {}
  inline bool operator==(const Hash128& other) const { return hash0 == other.hash0 && hash1 == other.hash1; }
  inline bool operator<(const Hash128& other) const { return hash0 < other.hash0 || (hash0 == other.hash0 && hash1 < other.hash1); }
};

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

  static Sgf* parse(const string& str);
  static Sgf* loadFile(const string& file);
  static vector<Sgf*> loadFiles(const vector<string>& files);

  int getBSize() const;

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
  Hash128 hash;

  CompactSgf(const Sgf* sgf);
  ~CompactSgf();

  static CompactSgf* loadFile(const string& file);
  static vector<CompactSgf*> loadFiles(const vector<string>& files);
};


#endif
