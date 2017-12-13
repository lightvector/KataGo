#ifndef SGF_H_
#define SGF_H_

#include "core/global.h"
#include "fastboard.h"

STRUCT_NAMED_PAIR(Loc,loc,Player,pla,Move);

struct Sgf;

struct SgfNode {
  map<string,vector<string>> props;
  Sgf* parent;
  SgfNode();
  ~SgfNode();

  string getSingleProperty(const char* key) const;

  bool hasPlacements() const;
  void accumPlacements(vector<Move>& moves, int bSize) const;
  void accumMoves(vector<Move>& moves, int bSize) const;
};

struct Sgf {
  string fileName;
  vector<SgfNode*> nodes;
  vector<Sgf*> children;

  Sgf();
  ~Sgf();

  static Sgf* parse(const string& str);
  static Sgf* loadFile(const string& file);
  static vector<Sgf*> loadFiles(const vector<string>& files);

  int getBSize() const;

  void getPlacements(vector<Move>& moves, int bSize) const;
  void getMoves(vector<Move>& moves, int bSize) const;

  int depth() const;
};

#endif
