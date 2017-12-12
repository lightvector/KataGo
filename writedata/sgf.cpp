#include "core/global.h"
#include "fastboard.h"
#include "sgf.h"

SgfNode::SgfNode()
{}
SgfNode::~SgfNode()
{}

Sgf::Sgf()
{}
Sgf::~Sgf() {
  for(int i = 0; i<nodes.size(); i++)
    delete nodes[i];
  for(int i = 0; i<children.size(); i++)
    delete children[i];
}

static void sgfFail(const string& msg, const string& str, int pos) {
  throw IOError(msg + " (pos " + Global::intToString(pos) + "):" + str);
}
static void sgfFail(const char* msg, const string& str, int pos) {
  sgfFail(string(msg),str,pos);
}

static char nextSgfTextChar(const string& str, int& pos) {
  if(pos >= str.length()) sgfFail("Unexpected end of str", str,pos);
  return str[pos++];
}
static char nextSgfChar(const string& str, int& pos) {
  while(true) {
    if(pos >= str.length()) sgfFail("Unexpected end of str", str,pos);
    char c = str[pos++];
    if(!Global::isWhitespace(c))
      return c;
  }
}

static string parseTextValue(const string& str, int& pos) {
  string acc;
  bool escaping = false;
  while(true) {
    char c = nextSgfTextChar(str,pos);
    if(!escaping && c == ']') {
      pos--;
      break;
    }
    if(!escaping && c == '\\') {
      escaping = true;
      continue;
    }
    if(escaping && (c == '\n' || c == '\r')) {
      while(c == '\n' || c == '\r')
        c = nextSgfTextChar(str,pos);
      pos--;
      escaping = false;
      continue;
    }
    if(c == '\t') {
      escaping = false;
      acc += ' ';
      continue;
    }

    escaping = false;
    acc += c;
  }
  return acc;
}

static bool maybeParseProperty(SgfNode* node, const string& str, int& pos) {
  int keystart = pos;
  while(Global::isAlpha(nextSgfChar(str,pos))) {}
  pos--;
  int keystop = pos;
  string key = str.substr(keystart,keystop-keystart);
  if(key.length() <= 0)
    return false;

  vector<string>& contents = node->props[key];

  bool parsedAtLeastOne = false;
  while(true) {
    if(nextSgfChar(str,pos) != '[') {
      pos--;
      break;
    }
    contents.push_back(parseTextValue(str,pos));
    if(nextSgfChar(str,pos) != ']') sgfFail("Expected closing bracket",str,pos);

    parsedAtLeastOne = true;
  }
  if(!parsedAtLeastOne)
    sgfFail("No property values for property " + key,str,pos);
  return true;
}

static SgfNode* maybeParseNode(const string& str, int& pos) {
  if(nextSgfChar(str,pos) != ';') {
    pos--;
    return NULL;
  }
  SgfNode* node = new SgfNode();
  try {
    while(true) {
      bool suc = maybeParseProperty(node,str,pos);
      if(!suc)
        break;
    }
  }
  catch(...) {
    delete node;
    throw;
  }
  return node;
}

static Sgf* maybeParseSgf(const string& str, int& pos) {
  if(pos >= str.length() || str[pos] != '(')
    return NULL;
  nextSgfChar(str,pos); //Discard the '('

  Sgf* sgf = new Sgf();

  try {
    while(true) {
      SgfNode* node = maybeParseNode(str,pos);
      if(node == NULL)
        break;
      sgf->nodes.push_back(node);
    }
    while(true) {
      Sgf* child = maybeParseSgf(str,pos);
      if(child == NULL)
        break;
      sgf->children.push_back(child);
    }
    if(nextSgfChar(str,pos) != ')')
      sgfFail("Expected closing paren for sgf tree",str,pos);
  }
  catch (...) {
    delete sgf;
    throw;
  }
  return sgf;
}


Sgf* Sgf::parse(const string& str) {
  int pos = 0;
  Sgf* sgf = maybeParseSgf(str,pos);
  if(sgf == NULL)
    sgfFail("Empty sgf",str,0);
  return sgf;
}

Sgf* Sgf::loadFile(const string& file) {
  return parse(Global::readFile(file));
}

vector<Sgf*> Sgf::loadFiles(const vector<string>& files) {
  vector<Sgf*> sgfs;
  try {
    for(int i = 0; i<files.size(); i++) {
      try {
        Sgf* sgf = loadFile(files[i]);
        sgfs.push_back(sgf);
      }
      catch(const IOError& e) {
        cout << "Skipping sgf file: " << files[i] << ": " << e.message << endl;
      }
    }
  }
  catch(...) {
    for(int i = 0; i<sgfs.size(); i++) {
      delete sgfs[i];
    }
    throw;
  }
  return sgfs;
}
