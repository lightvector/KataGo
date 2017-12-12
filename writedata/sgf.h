#ifndef SGF_H_
#define SGF_H_

#include "core/global.h"
#include "fastboard.h"

STRUCT_NAMED_PAIR(Loc,loc,Player,pla,Move);

struct SgfNode {
  map<string,vector<string>> props;
  SgfNode();
  ~SgfNode();
};

struct Sgf {
  vector<SgfNode*> nodes;
  vector<Sgf*> children;

  Sgf();
  ~Sgf();

  static Sgf* parse(const string& str);
  static Sgf* loadFile(const string& file);
  static vector<Sgf*> loadFiles(const vector<string>& files);
};

#endif
