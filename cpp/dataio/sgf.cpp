#include "../dataio/sgf.h"

#include "../core/sha2.h"

using namespace std;

SgfNode::SgfNode()
  :props(NULL),move(0,0,C_EMPTY)
{}
SgfNode::SgfNode(const SgfNode& other)
  :props(NULL),move(0,0,C_EMPTY)
{
  if(other.props != NULL)
    props = new map<string,vector<string>>(*(other.props));
  move = other.move;
}
SgfNode::SgfNode(SgfNode&& other) noexcept
  :props(NULL),move(0,0,C_EMPTY)
{
  props = other.props;
  other.props = NULL;
  move = other.move;
}
SgfNode::~SgfNode()
{
  if(props != NULL)
    delete props;
}

SgfNode& SgfNode::operator=(const SgfNode& other) {
  if(this == &other)
    return *this;
  if(props != NULL)
    delete props;
  if(other.props != NULL)
    props = new map<string,vector<string>>(*(other.props));
  else
    props = NULL;
  move = other.move;
  return *this;
}
SgfNode& SgfNode::operator=(SgfNode&& other) noexcept {
  if(props != NULL)
    delete props;
  props = other.props;
  other.props = NULL;
  move = other.move;
  return *this;
}


static void propertyFail(const string& msg) {
  throw IOError(msg);
}
static void propertyFail(const char* msg) {
  propertyFail(string(msg));
}

static int parseSgfCoord(char c) {
  if(c >= 'a' && c <= 'z')
    return (int)c - (int)'a';
  if(c >= 'A' && c <= 'Z')
    return (int)c - (int)'A' + 26;
  return -1;
}

//MoveNoBSize uses only single bytes
//If both coords are COORD_MAX, that indicates pass
static const int COORD_MAX = 128;

static MoveNoBSize parseSgfLocOrPassNoSize(const string& s, Player pla) {
  if(s.length() == 0)
    return MoveNoBSize(COORD_MAX,COORD_MAX,pla);
  if(s.length() != 2)
    propertyFail("Invalid location: " + s);

  int x = parseSgfCoord(s[0]);
  int y = parseSgfCoord(s[1]);

  if(x < 0 || y < 0 || x >= COORD_MAX || y >= COORD_MAX)
    propertyFail("Invalid location: " + s);
  return MoveNoBSize(x,y,pla);
}

static Loc parseSgfLoc(const string& s, int xSize, int ySize) {
  if(s.length() != 2)
    propertyFail("Invalid location: " + s);

  int x = parseSgfCoord(s[0]);
  int y = parseSgfCoord(s[1]);

  if(x < 0 || x >= xSize || y < 0 || y >= ySize)
    propertyFail("Invalid location: " + s);
  return Location::getLoc(x,y,xSize);
}

static Loc parseSgfLocOrPass(const string& s, int xSize, int ySize) {
  if(s.length() == 0 || (s == "tt" && (xSize <= 19 || ySize <= 19)))
    return Board::PASS_LOC;
  return parseSgfLoc(s,xSize,ySize);
}

static void writeSgfLoc(ostream& out, Loc loc, int xSize, int ySize) {
  if(xSize >= 53 || ySize >= 53)
    throw StringError("Writing coordinates for SGF files for board sizes >= 53 is not implemented");
  if(loc == Board::PASS_LOC || loc == Board::NULL_LOC)
    return;
  int x = Location::getX(loc,xSize);
  int y = Location::getY(loc,xSize);
  const char* chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  out << chars[x];
  out << chars[y];
}

bool SgfNode::hasProperty(const char* key) const {
  if(props == NULL)
    return false;
  return contains(*props,key);
}

string SgfNode::getSingleProperty(const char* key) const {
  if(props == NULL)
    propertyFail("SGF does not contain property: " + string(key));
  if(!contains(*props,key))
    propertyFail("SGF does not contain property: " + string(key));
  const vector<string>& prop = map_get(*props,key);
  if(prop.size() != 1)
    propertyFail("SGF property is not a singleton: " + string(key));
  return prop[0];
}

bool SgfNode::hasPlacements() const {
  return props != NULL && (contains(*props,"AB") || contains(*props,"AW") || contains(*props,"AE"));
}

void SgfNode::accumPlacements(vector<Move>& moves, int xSize, int ySize) const {
  if(props == NULL)
    return;
  if(contains(*props,"AB")) {
    const vector<string>& ab = map_get(*props,"AB");
    int len = ab.size();
    for(int i = 0; i<len; i++) {
      Loc loc = parseSgfLoc(ab[i],xSize,ySize);
      moves.push_back(Move(loc,P_BLACK));
    }
  }
  if(contains(*props,"AW")) {
    const vector<string>& aw = map_get(*props,"AW");
    int len = aw.size();
    for(int i = 0; i<len; i++) {
      Loc loc = parseSgfLoc(aw[i],xSize,ySize);
      moves.push_back(Move(loc,P_WHITE));
    }
  }
  if(contains(*props,"AE")) {
    const vector<string>& ae = map_get(*props,"AE");
    int len = ae.size();
    for(int i = 0; i<len; i++) {
      Loc loc = parseSgfLoc(ae[i],xSize,ySize);
      moves.push_back(Move(loc,C_EMPTY));
    }
  }
}

void SgfNode::accumMoves(vector<Move>& moves, int xSize, int ySize) const {
  if(move.pla == C_BLACK) {
    if((move.x == COORD_MAX && move.y == COORD_MAX) ||
       (move.x == 19 && move.y == 19 && (xSize <= 19 || ySize <= 19))) //handle "tt"
      moves.push_back(Move(Board::PASS_LOC,move.pla));
    else {
      if(move.x >= xSize || move.y >= ySize) propertyFail("Move out of bounds: " + Global::intToString(move.x) + "," + Global::intToString(move.y));
      moves.push_back(Move(Location::getLoc(move.x,move.y,xSize),move.pla));
    }
  }
  if(props != NULL && contains(*props,"B")) {
    const vector<string>& b = map_get(*props,"B");
    int len = b.size();
    for(int i = 0; i<len; i++) {
      Loc loc = parseSgfLocOrPass(b[i],xSize,ySize);
      moves.push_back(Move(loc,P_BLACK));
    }
  }
  if(move.pla == C_WHITE) {
    if((move.x == COORD_MAX && move.y == COORD_MAX) ||
       (move.x == 19 && move.y == 19 && (xSize <= 19 || ySize <= 19))) //handle "tt"
      moves.push_back(Move(Board::PASS_LOC,move.pla));
    else {
      if(move.x >= xSize || move.y >= ySize) propertyFail("Move out of bounds: " + Global::intToString(move.x) + "," + Global::intToString(move.y));
      moves.push_back(Move(Location::getLoc(move.x,move.y,xSize),move.pla));
    }
  }
  if(props != NULL && contains(*props,"W")) {
    const vector<string>& w = map_get(*props,"W");
    int len = w.size();
    for(int i = 0; i<len; i++) {
      Loc loc = parseSgfLocOrPass(w[i],xSize,ySize);
      moves.push_back(Move(loc,P_WHITE));
    }
  }
}

Color SgfNode::getPLSpecifiedColor() const {
  if(!hasProperty("PL"))
    return C_EMPTY;
  string s = Global::toLower(getSingleProperty("PL"));
  if(s == "b" || s == "black")
    return C_BLACK;
  if(s == "w" || s == "white")
    return C_WHITE;
  return C_EMPTY;
}

Rules SgfNode::getRules(const Rules& defaultRules) const {
  Rules rules = defaultRules;
  if(!hasProperty("RU"))
    return rules;
  string s = Global::toLower(getSingleProperty("RU"));
  if(s == "japanese") {
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.koRule = Rules::KO_SIMPLE;
    rules.multiStoneSuicideLegal = false;
  }
  else if(s == "chinese") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SIMPLE;
    rules.multiStoneSuicideLegal = false;
  }
  else if(s == "aga") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.multiStoneSuicideLegal = false;
  }
  else if(s == "nz") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.multiStoneSuicideLegal = true;
  }
  else if(s == "tromp-taylor" || s == "tromp taylor" || s == "tromptaylor") {
    rules.scoringRule = Rules::SCORING_AREA;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.multiStoneSuicideLegal = true;
  }
  else {
    string origS = s;
    auto startsWithAndStrip = [](string& str, const string& prefix) {
      bool matches = str.length() >= prefix.length() && str.substr(0,prefix.length()) == prefix;
      if(matches)
        str = str.substr(prefix.length());
      return matches;
    };
    auto fail = [&origS]() {
      throw StringError("Could not parse rules in sgf: " + origS);
    };

    if(startsWithAndStrip(s,"ko")) {
      if(startsWithAndStrip(s,"simple")) rules.koRule = Rules::KO_SIMPLE;
      else if(startsWithAndStrip(s,"positional")) rules.koRule = Rules::KO_POSITIONAL;
      else if(startsWithAndStrip(s,"situational")) rules.koRule = Rules::KO_SITUATIONAL;
      else if(startsWithAndStrip(s,"spight")) rules.koRule = Rules::KO_SPIGHT;
      else fail();

      bool b;
      b = startsWithAndStrip(s,"score");
      if(!b) fail();

      if(startsWithAndStrip(s,"area")) rules.scoringRule = Rules::SCORING_AREA;
      else if(startsWithAndStrip(s,"territory")) rules.scoringRule = Rules::SCORING_TERRITORY;
      else fail();

      b = startsWithAndStrip(s,"sui");
      if(!b) fail();
      if(startsWithAndStrip(s,"1")) rules.multiStoneSuicideLegal = true;
      else if(startsWithAndStrip(s,"0")) rules.multiStoneSuicideLegal = false;
      else fail();
    }
  }
  return rules;
}


Sgf::Sgf()
{}
Sgf::~Sgf() {
  for(int i = 0; i<nodes.size(); i++)
    delete nodes[i];
  for(int i = 0; i<children.size(); i++)
    delete children[i];
}


int Sgf::depth() const {
  int maxChildDepth = 0;
  for(int i = 0; i<children.size(); i++) {
    int childDepth = children[i]->depth();
    if(childDepth > maxChildDepth)
      maxChildDepth = childDepth;
  }
  return maxChildDepth + nodes.size();
}

static void checkNonEmpty(const vector<SgfNode*>& nodes) {
  if(nodes.size() <= 0)
    throw StringError("Empty sgf");
}

XYSize Sgf::getXYSize() const {
  checkNonEmpty(nodes);
  int xSize;
  int ySize;
  if(!nodes[0]->hasProperty("SZ"))
    return XYSize(19,19); //Some SGF files don't specify, in that case assume 19

  const string& s = nodes[0]->getSingleProperty("SZ");
  if(contains(s,':')) {
    vector<string> pieces = Global::split(s,':');
    if(pieces.size() != 2)
      propertyFail("Could not parse board size in sgf: " + s);
    bool suc = Global::tryStringToInt(pieces[0], xSize) && Global::tryStringToInt(pieces[1], ySize);
    if(!suc)
      propertyFail("Could not parse board size in sgf: " + s);
  }
  else {
    bool suc = Global::tryStringToInt(s, xSize);
    if(!suc)
      propertyFail("Could not parse board size in sgf: " + s);
    ySize = xSize;
  }

  if(xSize <= 0 || ySize <= 0)
    propertyFail("Board size in sgf is <= 0: " + s);
  if(xSize > Board::MAX_LEN || ySize > Board::MAX_LEN)
    propertyFail(
      "Board size in sgf is > Board::MAX_LEN = " + Global::intToString((int)Board::MAX_LEN) +
      ", if larger sizes are desired, consider increasing and recompiling: " + s
    );
  return XYSize(xSize,ySize);
}

float Sgf::getKomi() const {
  checkNonEmpty(nodes);

  //Default, if SGF doesn't specify
  if(!nodes[0]->hasProperty("KM"))
    return 7.5f;

  float komi;
  bool suc = Global::tryStringToFloat(nodes[0]->getSingleProperty("KM"), komi);
  if(!suc)
    propertyFail("Could not parse komi in sgf");
  if(!Rules::komiIsIntOrHalfInt(komi))
    propertyFail("Komi in sgf is not integer or half-integer");
  return komi;
}

Rules Sgf::getRules(const Rules& defaultRules) const {
  checkNonEmpty(nodes);
  return nodes[0]->getRules(defaultRules);
}

void Sgf::getPlacements(vector<Move>& moves, int xSize, int ySize) const {
  moves.clear();
  checkNonEmpty(nodes);
  nodes[0]->accumPlacements(moves,xSize,ySize);
}

//Gets the longest child if the sgf has branches
void Sgf::getMoves(vector<Move>& moves, int xSize, int ySize) const {
  moves.clear();
  getMovesHelper(moves,xSize,ySize);
}

void Sgf::getMovesHelper(vector<Move>& moves, int xSize, int ySize) const {
  checkNonEmpty(nodes);
  for(int i = 0; i<nodes.size(); i++) {
    if(i > 0 && nodes[i]->hasPlacements())
      propertyFail("Found stone placements after the root, game records that are not simply ordinary play not currently supported");
    nodes[i]->accumMoves(moves,xSize,ySize);
  }

  int maxChildDepth = 0;
  Sgf* maxChild = NULL;
  for(int i = 0; i<children.size(); i++) {
    int childDepth = children[i]->depth();
    if(childDepth > maxChildDepth) {
      maxChildDepth = childDepth;
      maxChild = children[i];
    }
  }

  if(maxChild != NULL) {
    maxChild->getMovesHelper(moves,xSize,ySize);
  }
}



//PARSING---------------------------------------------------------------------

static void sgfFail(const string& msg, const string& str, int pos) {
  throw IOError(msg + " (pos " + Global::intToString(pos) + "):\n" + str);
}
static void sgfFail(const char* msg, const string& str, int pos) {
  sgfFail(string(msg),str,pos);
}
static void sgfFail(const string& msg, const string& str, int entryPos, int pos) {
  throw IOError(msg + " (entryPos " + Global::intToString(entryPos) + "):" + " (pos " + Global::intToString(pos) + "):\n" + str);
}
static void sgfFail(const char* msg, const string& str, int entryPos, int pos) {
  sgfFail(string(msg),str,entryPos,pos);
}

static void consume(const string& str, int& pos, int& newPos) {
  (void)str;
  pos = newPos;
  //cout << "CHAR: " << str[newPos-1] << endl;
}

static char peekSgfTextChar(const string& str, int& pos, int& newPos) {
  newPos = pos;
  if(newPos >= str.length()) sgfFail("Unexpected end of str", str,newPos);
  return str[newPos++];
}
static char peekSgfChar(const string& str, int& pos, int& newPos) {
  newPos = pos;
  while(true) {
    if(newPos >= str.length()) sgfFail("Unexpected end of str", str,newPos);
    char c = str[newPos++];
    if(!Global::isWhitespace(c))
      return c;
  }
}

static string parseTextValue(const string& str, int& pos) {
  string acc;
  bool escaping = false;
  int newPos;
  while(true) {
    char c = peekSgfTextChar(str,pos,newPos);
    if(!escaping && c == ']') {
      break;
    }
    consume(str,pos,newPos);

    if(!escaping && c == '\\') {
      escaping = true;
      continue;
    }
    if(c == '\n' || c == '\r') {
      while(true) {
        c = peekSgfTextChar(str,pos,newPos);
        if(c == '\n' || c == '\r')
          consume(str,pos,newPos);
        else
          break;
      }
      if(!escaping)
        acc += '\n';
      escaping = false;
      continue;
    }
    if(c == '\t' || c == '\v' || c == '\f') {
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
  string key;
  while(true) {
    int newPos;
    char c = peekSgfChar(str,pos,newPos);
    if(Global::isAlpha(c)) {
      key += c;
      consume(str,pos,newPos);
    }
    else
      break;
  }
  if(key.length() <= 0)
    return false;

  bool parsedAtLeastOne = false;
  while(true) {
    int newPos;
    if(peekSgfChar(str,pos,newPos) != '[')
      break;
    consume(str,pos,newPos);

    if(node->move.pla == C_EMPTY && key == "B") {
      node->move = parseSgfLocOrPassNoSize(parseTextValue(str,pos),P_BLACK);
    }
    else if(node->move.pla == C_EMPTY && key == "W") {
      node->move = parseSgfLocOrPassNoSize(parseTextValue(str,pos),P_WHITE);
    }
    else {
      if(node->props == NULL)
        node->props = new map<string,vector<string>>();
      vector<string>& contents = (*(node->props))[key];
      string value = parseTextValue(str,pos);
      contents.push_back(value);
    }
    if(peekSgfChar(str,pos,newPos) != ']')
      sgfFail("Expected closing bracket",str,pos);
    consume(str,pos,newPos);

    parsedAtLeastOne = true;
  }
  if(!parsedAtLeastOne)
    sgfFail("No property values for property " + key,str,pos);

  return true;
}

static SgfNode* maybeParseNode(const string& str, int& pos) {
  int newPos;
  if(peekSgfChar(str,pos,newPos) != ';')
    return NULL;
  consume(str,pos,newPos);

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
  if(pos >= str.length())
    return NULL;
  int newPos;
  char c = peekSgfChar(str,pos,newPos);
  if(c != '(')
    return NULL;
  consume(str,pos,newPos);

  int entryPos = pos;
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
    c = peekSgfChar(str,pos,newPos);
    if(c != ')')
      sgfFail("Expected closing paren for sgf tree",str,entryPos,pos);
    consume(str,pos,newPos);
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
  uint64_t hash[4];
  SHA2::get256(str.c_str(),hash);
  if(sgf == NULL || sgf->nodes.size() == 0)
    sgfFail("Empty or invalid sgf (is the opening parenthesis missing?)",str,0);
  sgf->hash = Hash128(hash[0],hash[1]);
  return sgf;
}

Sgf* Sgf::loadFile(const string& file) {
  Sgf* sgf = parse(Global::readFile(file));
  if(sgf != NULL)
    sgf->fileName = file;
  return sgf;
}

vector<Sgf*> Sgf::loadFiles(const vector<string>& files) {
  vector<Sgf*> sgfs;
  try {
    for(int i = 0; i<files.size(); i++) {
      if(i % 10000 == 0)
        cout << "Loaded " << i << "/" << files.size() << " files" << endl;
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

vector<Sgf*> Sgf::loadSgfsFile(const string& file) {
  vector<Sgf*> sgfs;
  vector<string> lines = Global::readFileLines(file,'\n');
  try {
    for(size_t i = 0; i<lines.size(); i++) {
      string line = Global::trim(lines[i]);
      if(line.length() <= 0)
        continue;
      Sgf* sgf = parse(line);
      sgf->fileName = file;
      sgfs.push_back(sgf);
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


vector<Sgf*> Sgf::loadSgfsFiles(const vector<string>& files) {
  vector<Sgf*> sgfs;
  try {
    for(int i = 0; i<files.size(); i++) {
      if(i % 500 == 0)
        cout << "Loaded " << i << "/" << files.size() << " files" << endl;
      try {
        vector<Sgf*> s = loadSgfsFile(files[i]);
        sgfs.insert(sgfs.end(),s.begin(),s.end());
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



CompactSgf::CompactSgf(const Sgf* sgf)
  :fileName(sgf->fileName),
   rootNode(),
   placements(),
   moves(),
   xSize(),
   ySize(),
   depth()
{
  XYSize size = sgf->getXYSize();
  xSize = size.x;
  ySize = size.y;
  depth = sgf->depth();
  komi = sgf->getKomi();
  hash = sgf->hash;

  sgf->getPlacements(placements, xSize, ySize);
  sgf->getMoves(moves, xSize, ySize);

  checkNonEmpty(sgf->nodes);
  rootNode = *(sgf->nodes[0]);
}

CompactSgf::CompactSgf(Sgf&& sgf)
  :fileName(),
   rootNode(),
   placements(),
   moves(),
   xSize(),
   ySize(),
   depth()
{
  XYSize size = sgf.getXYSize();
  xSize = size.x;
  ySize = size.y;
  depth = sgf.depth();
  komi = sgf.getKomi();
  hash = sgf.hash;

  sgf.getPlacements(placements, xSize, ySize);
  sgf.getMoves(moves, xSize, ySize);

  fileName = std::move(sgf.fileName);
  checkNonEmpty(sgf.nodes);
  rootNode = std::move(*sgf.nodes[0]);
  for(int i = 0; i<sgf.nodes.size(); i++) {
    delete sgf.nodes[i];
    sgf.nodes[i] = NULL;
  }
  for(int i = 0; i<sgf.children.size(); i++) {
    delete sgf.children[i];
    sgf.children[i] = NULL;
  }
}

CompactSgf::~CompactSgf() {
}


CompactSgf* CompactSgf::parse(const string& str) {
  Sgf* sgf = Sgf::parse(str);
  CompactSgf* compact = new CompactSgf(std::move(*sgf));
  delete sgf;
  return compact;
}

CompactSgf* CompactSgf::loadFile(const string& file) {
  Sgf* sgf = Sgf::loadFile(file);
  CompactSgf* compact = new CompactSgf(std::move(*sgf));
  delete sgf;
  return compact;
}

vector<CompactSgf*> CompactSgf::loadFiles(const vector<string>& files) {
  vector<CompactSgf*> sgfs;
  try {
    for(int i = 0; i<files.size(); i++) {
      if(i % 10000 == 0)
        cout << "Loaded " << i << "/" << files.size() << " files" << endl;
      try {
        CompactSgf* sgf = loadFile(files[i]);
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

Rules CompactSgf::getRulesFromSgf(const Rules& defaultRules) {
  Rules rules = defaultRules;
  rules.komi = komi;
  rules = rootNode.getRules(rules);
  return rules;
}

void CompactSgf::setupInitialBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist) {
  Color plPlayer = rootNode.getPLSpecifiedColor();
  if(plPlayer == P_BLACK || plPlayer == P_WHITE)
    nextPla = plPlayer;
  else {
    bool hasBlack = false;
    bool allBlack = true;
    for(int i = 0; i<placements.size(); i++) {
      if(placements[i].pla == P_BLACK)
        hasBlack = true;
      else
        allBlack = false;
    }
    if(hasBlack && !allBlack)
      nextPla = P_WHITE;
    else
      nextPla = P_BLACK;
  }

  board = Board(xSize,ySize);
  for(int i = 0; i<placements.size(); i++) {
    board.setStone(placements[i].loc,placements[i].pla);
  }

  hist = BoardHistory(board,nextPla,initialRules,0);
}

void CompactSgf::setupBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int turnNumber) {
  setupInitialBoardAndHist(initialRules, board, nextPla, hist);

  if(turnNumber < 0 || turnNumber > moves.size())
    throw StringError(
      Global::strprintf(
        "Attempting to set up position from SGF for invalid turn number %d, valid values are %d to %d",
        (int)turnNumber, 0, (int)moves.size()
      )
    );

  for(size_t i = 0; i<turnNumber; i++) {
    hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
    nextPla = getOpp(moves[i].pla);
  }
}

void WriteSgf::printGameResult(ostream& out, const BoardHistory& hist) {
  if(hist.isGameFinished) {
    out << "RE[";
    if(hist.isNoResult)
      out << "Void";
    else if(hist.isResignation && hist.winner == C_BLACK)
      out << "B+R";
    else if(hist.isResignation && hist.winner == C_WHITE)
      out << "W+R";
    else if(hist.winner == C_BLACK)
      out << "B+" << (-hist.finalWhiteMinusBlackScore);
    else if(hist.winner == C_WHITE)
      out << "W+" << hist.finalWhiteMinusBlackScore;
    else if(hist.winner == C_EMPTY)
      out << "0";
    else
      ASSERT_UNREACHABLE;
    out << "]";
  }
}

void WriteSgf::writeSgf(
  ostream& out, const string& bName, const string& wName, const Rules& rules,
  const BoardHistory& hist,
  const FinishedGameData* gameData
) {
  const Board& initialBoard = hist.initialBoard;
  int xSize = initialBoard.x_size;
  int ySize = initialBoard.y_size;
  out << "(;FF[4]GM[1]";
  if(xSize == ySize)
    out << "SZ[" << xSize << "]";
  else
    out << "SZ[" << xSize << ":" << ySize << "]";
  out << "PB[" << bName << "]";
  out << "PW[" << wName << "]";

  int handicap = 0;
  bool hasWhite = false;
  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      if(initialBoard.colors[loc] == C_BLACK)
        handicap += 1;
      if(initialBoard.colors[loc] == C_WHITE)
        hasWhite = true;
    }
  }
  if(hasWhite)
    handicap = 0;

  out << "HA[" << handicap << "]";
  out << "KM[" << rules.komi << "]";
  out << "RU[ko" << Rules::writeKoRule(rules.koRule)
      << "score" << Rules::writeScoringRule(rules.scoringRule)
      << "sui" << rules.multiStoneSuicideLegal << "]";
  printGameResult(out,hist);

  bool hasAB = false;
  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      if(initialBoard.colors[loc] == C_BLACK) {
        if(!hasAB) {
          out << "AB";
          hasAB = true;
        }
        out << "[";
        writeSgfLoc(out,loc,xSize,ySize);
        out << "]";
      }
    }
  }

  bool hasAW = false;
  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      Loc loc = Location::getLoc(x,y,xSize);
      if(initialBoard.colors[loc] == C_WHITE) {
        if(!hasAW) {
          out << "AW";
          hasAW = true;
        }
        out << "[";
        writeSgfLoc(out,loc,xSize,ySize);
        out << "]";
      }
    }
  }

  int startTurnIdx = 0;
  if(gameData != NULL) {
    startTurnIdx = gameData->startHist.moveHistory.size();
    out << "C[startTurnIdx=" << startTurnIdx
        << "," << "mode=" << gameData->mode
        << "," << "modeM1=" << gameData->modeMeta1
        << "," << "modeM2=" << gameData->modeMeta2;
    for(int j = 0; j<gameData->changedNeuralNets.size(); j++) {
      out << ",newNeuralNetTurn" << gameData->changedNeuralNets[j]->turnNumber
          << "=" << gameData->changedNeuralNets[j]->name;
    }
    out << "]";
    assert(hist.moveHistory.size() - startTurnIdx <= gameData->whiteValueTargetsByTurn.size());
  }

  for(size_t i = 0; i<hist.moveHistory.size(); i++) {
    if(hist.moveHistory[i].pla == P_BLACK)
      out << ";B[";
    else
      out << ";W[";
    writeSgfLoc(out,hist.moveHistory[i].loc,xSize,ySize);
    out << "]";

    if(gameData != NULL) {
      if(i >= startTurnIdx) {
        const ValueTargets& targets = gameData->whiteValueTargetsByTurn[i-startTurnIdx];
        char winBuf[32];
        char lossBuf[32];
        char noResultBuf[32];
        char scoreBuf[32];
        sprintf(winBuf,"%.2f",targets.win);
        sprintf(lossBuf,"%.2f",targets.loss);
        sprintf(noResultBuf,"%.2f",targets.noResult);
        sprintf(scoreBuf,"%.1f",targets.score);
        out << "C["
            << winBuf << " "
            << lossBuf << " "
            << noResultBuf << " "
            << scoreBuf << "]";
      }
    }
  }
  out << ")";
}
