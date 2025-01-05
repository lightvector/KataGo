#include "../dataio/sgf.h"

#include "../core/fileutils.h"
#include "../core/sha2.h"
#include "../dataio/files.h"
#include "../program/playutils.h"

#include "../external/nlohmann_json/json.hpp"

using namespace std;
using json = nlohmann::json;

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

static void parseSgfLocRectangle(const string& s, int xSize, int ySize, int& x1, int& y1, int& x2, int& y2) {
  if(contains(s,':')) {
    if(s.length() != 5 || s[2] != ':')
      propertyFail("Invalid location rect: " + s);
    x1 = parseSgfCoord(s[0]);
    y1 = parseSgfCoord(s[1]);
    x2 = parseSgfCoord(s[3]);
    y2 = parseSgfCoord(s[4]);
  }
  else {
    if(s.length() != 2)
      propertyFail("Invalid location: " + s);

    x1 = parseSgfCoord(s[0]);
    y1 = parseSgfCoord(s[1]);
    x2 = x1;
    y2 = y1;
  }
  if(x1 < 0 || x1 >= xSize || y1 < 0 || y1 >= ySize ||
     x2 < 0 || x2 >= xSize || y2 < 0 || y2 >= ySize ||
     x1 > x2 || y1 > y2)
    propertyFail("Invalid location or location rect: " + s);
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
bool SgfNode::hasProperty(const string& key) const {
  return hasProperty(key.c_str());
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
string SgfNode::getSingleProperty(const string& key) const {
  return getSingleProperty(key.c_str());
}

const vector<string> SgfNode::getProperties(const char* key) const {
  if(props == NULL)
    propertyFail("SGF does not contain property: " + string(key));
  if(!contains(*props,key))
    propertyFail("SGF does not contain property: " + string(key));
  return map_get(*props,key);
}
const vector<string> SgfNode::getProperties(const string& key) const {
  return getProperties(key.c_str());
}

void SgfNode::addProperty(const string& key, const string& value) {
  if(props == NULL)
    props = new map<string,vector<string>>();
  vector<string>& contents = (*props)[key];
  contents.push_back(value);
}

void SgfNode::appendComment(const string& value) {
  if(props == NULL)
    props = new map<string,vector<string>>();
  vector<string>& contents = (*props)["C"];
  if(contents.size() == 0)
    contents.push_back(value);
  else {
    contents[contents.size()-1] = contents[contents.size()-1] + value;
  }
}

bool SgfNode::hasPlacements() const {
  return props != NULL && (contains(*props,"AB") || contains(*props,"AW") || contains(*props,"AE"));
}

void SgfNode::accumPlacements(vector<Move>& moves, int xSize, int ySize) const {
  if(props == NULL)
    return;

  auto handleRectangleList = [&](const vector<string>& elts, Player color) {
    size_t len = elts.size();
    for(size_t i = 0; i<len; i++) {
      int x1; int y1;
      int x2; int y2;
      parseSgfLocRectangle(elts[i],xSize,ySize,x1,y1,x2,y2);
      for(int x = x1; x <= x2; x++) {
        for(int y = y1; y <= y2; y++) {
          Loc loc = Location::getLoc(x,y,xSize);
          moves.push_back(Move(loc,color));
        }
      }
    }
  };

  if(contains(*props,"AB")) {
    const vector<string>& ab = map_get(*props,"AB");
    handleRectangleList(ab,P_BLACK);
  }
  if(contains(*props,"AW")) {
    const vector<string>& aw = map_get(*props,"AW");
    handleRectangleList(aw,P_WHITE);
  }
  if(contains(*props,"AE")) {
    const vector<string>& ae = map_get(*props,"AE");
    handleRectangleList(ae,C_EMPTY);
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
    size_t len = b.size();
    for(size_t i = 0; i<len; i++) {
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
    size_t len = w.size();
    for(size_t i = 0; i<len; i++) {
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

Rules SgfNode::getRulesFromRUTagOrFail() const {
  if(!hasProperty("RU"))
    throw StringError("SGF file does not specify rules");
  string s = getSingleProperty("RU");

  Rules parsed;
  bool suc = Rules::tryParseRules(s,parsed);
  if(!suc)
    throw StringError("Could not parse rules in sgf: " + s);
  return parsed;
}

Player SgfNode::getSgfWinner() const {
  if(!hasProperty("RE"))
    return C_EMPTY;
  string s = Global::toLower(getSingleProperty("RE"));
  if(Global::isPrefix(s,"b+") || Global::isPrefix(s,"black+"))
    return P_BLACK;
  if(Global::isPrefix(s,"w+") || Global::isPrefix(s,"white+"))
    return P_WHITE;
  return C_EMPTY;
}

string SgfNode::getPlayerName(Player pla) const {
  if(pla == P_BLACK) {
    if(!hasProperty("PB"))
      return "";
    return getSingleProperty("PB");
  }
  else if(pla == P_WHITE) {
    if(!hasProperty("PW"))
      return "";
    return getSingleProperty("PW");
  }
  return "";
}

Sgf::Sgf()
{}
Sgf::~Sgf() {
  for(int i = 0; i<nodes.size(); i++)
    delete nodes[i];
  for(int i = 0; i<children.size(); i++)
    delete children[i];
}


int64_t Sgf::depth() const {
  int64_t maxChildDepth = 0;
  for(int i = 0; i<children.size(); i++) {
    int64_t childDepth = children[i]->depth();
    if(childDepth > maxChildDepth)
      maxChildDepth = childDepth;
  }
  return maxChildDepth + (int64_t)nodes.size();
}

int64_t Sgf::nodeCount() const {
  int64_t count = 0;
  for(int i = 0; i<children.size(); i++) {
    count += children[i]->nodeCount();
  }
  return count + (int64_t)nodes.size();
}

int64_t Sgf::branchCount() const {
  int64_t count = 0;
  for(int i = 0; i<children.size(); i++) {
    count += children[i]->branchCount();
  }
  if(children.size() > 1)
    count += (int64_t)children.size()-1;
  return count;
}

static void checkNonEmpty(const vector<SgfNode*>& nodes) {
  if(nodes.size() <= 0)
    throw StringError("Empty sgf");
}

XYSize Sgf::getXYSize() const {
  checkNonEmpty(nodes);
  int xSize = 0; //Initialize to 0 to suppress spurious clang compiler warning.
  int ySize = 0; //Initialize to 0 to suppress spurious clang compiler warning.
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

  if(xSize <= 1 || ySize <= 1)
    propertyFail("Board size in sgf is <= 1: " + s);
  if(xSize > Board::MAX_LEN || ySize > Board::MAX_LEN)
    propertyFail(
      "Board size in sgf is > Board::MAX_LEN = " + Global::intToString((int)Board::MAX_LEN) +
      ", if larger sizes are desired, consider increasing and recompiling: " + s
    );
  return XYSize(xSize,ySize);
}
float Sgf::getKomiOrFail() const {
  checkNonEmpty(nodes);
  return nodes[0]->getKomiOrFail();
}

float Sgf::getKomiOrDefault(float defaultKomi) const {
  checkNonEmpty(nodes);
  return nodes[0]->getKomiOrDefault(defaultKomi);
}

float SgfNode::getKomiOrFail() const {
  if(!hasProperty("KM"))
    propertyFail("Sgf does not specify komi");
  return getKomiOrDefault(0.0f);
}

float SgfNode::getKomiOrDefault(float defaultKomi) const {
   //Default, if SGF doesn't specify
  if(!hasProperty("KM"))
    return defaultKomi;

  float komi;
  bool suc = Global::tryStringToFloat(getSingleProperty("KM"), komi);
  if(!suc)
    propertyFail("Could not parse komi in sgf");

  if(!Rules::komiIsIntOrHalfInt(komi)) {
    //Hack - if the komi is a quarter integer and it looks like a Chinese GoGoD file, then double komi and accept
    if(Rules::komiIsIntOrHalfInt(komi*2.0f) && hasProperty("US") && hasProperty("RU") &&
       Global::isPrefix(getSingleProperty("US"),"GoGoD") &&
       (
         Global::toLower(getSingleProperty("RU")) == "chinese" ||
         Global::toLower(getSingleProperty("RU")) == "chinese, pair go"
       )
    )
      komi *= 2.0f;
    else
      propertyFail("Komi in sgf is not integer or half-integer");
  }

  //Hack - check for foxwq sgfs with weird komis
  if(hasProperty("AP") && contains(getProperties("AP"),"foxwq")) {
    if(komi == 550 || komi == 275)
      komi = 5.5f;
    else if(komi == 325 || komi == 650)
      komi = 6.5f;
    else if(komi == 375 || komi == 750)
      komi = 7.5f;
    else if(komi == 350 || komi == 700)
      komi = 7.0f;
    else if(komi == 0)
      komi = 0.0f;
    else if(komi == 6.5 || komi == 7.5 || komi == 7)
    {}
    else
      propertyFail("Currently no case implemented for foxwq komi: " + Global::floatToString(komi));
  }

  return komi;
}

int Sgf::getHandicapValue() const {
  checkNonEmpty(nodes);
  //Default, if SGF doesn't specify
  if(!nodes[0]->hasProperty("HA"))
    return 0;

  int handicapValue = 0;
  bool suc = Global::tryStringToInt(nodes[0]->getSingleProperty("HA"), handicapValue);
  if(!suc)
    propertyFail("Could not parse handicap value in sgf");
  return handicapValue;
}

bool Sgf::hasRules() const {
  checkNonEmpty(nodes);
  return nodes[0]->hasProperty("RU");
}

Rules Sgf::getRulesOrFail() const {
  checkNonEmpty(nodes);
  Rules rules = nodes[0]->getRulesFromRUTagOrFail();
  rules.komi = getKomiOrFail();
  return rules;
}

Player Sgf::getSgfWinner() const {
  checkNonEmpty(nodes);
  return nodes[0]->getSgfWinner();
}

Color Sgf::getFirstPlayerColor() const {
  checkNonEmpty(nodes);
  Color plColor = nodes[0]->getPLSpecifiedColor();
  if(plColor == C_BLACK || plColor == C_WHITE)
    return plColor;
  XYSize size = getXYSize();
  int xSize = size.x;
  int ySize = size.y;
  vector<Move> moves;
  getMoves(moves,xSize,ySize);
  if(moves.size() > 0)
    return moves[0].pla;
  return C_BLACK;
}

int Sgf::getRank(Player pla) const {
  checkNonEmpty(nodes);
  string rankStr;
  if(pla == P_BLACK) {
    if(!nodes[0]->hasProperty("BR"))
      return Sgf::RANK_UNKNOWN;
    rankStr = nodes[0]->getSingleProperty("BR");
  }
  else if(pla == P_WHITE) {
    if(!nodes[0]->hasProperty("WR"))
      return Sgf::RANK_UNKNOWN;
    rankStr = nodes[0]->getSingleProperty("WR");
  }
  else {
    assert(false);
    return Sgf::RANK_UNKNOWN;
  }
  int rank;
  static constexpr int TOP_DAN = 13;
  static constexpr int BOTTOM_KYU = 50;
  string rankStrLower = Global::toLower(rankStr);

  if(Global::isSuffix(rankStrLower,"d")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower,"d"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return rank-1;
  }
  if(Global::isSuffix(rankStrLower," d")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower," d"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return rank-1;
  }
  if(Global::isSuffix(rankStrLower,"dan")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower,"dan"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return rank-1;
  }
  if(Global::isSuffix(rankStrLower," dan")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower," dan"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return rank-1;
  }
  // \346\256\265 is UTF8 for the chinese "duan" character.
  if(Global::isSuffix(rankStr,"\346\256\265")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStr,"\346\256\265"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return rank-1;
  }
  if(Global::isSuffix(rankStrLower,"p")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower,"p"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return std::max(rank,9)-1;
  }
  if(Global::isSuffix(rankStrLower," p")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower," p"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return std::max(rank,9)-1;
  }
  if(Global::isSuffix(rankStrLower,"pro")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower,"pro"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return std::max(rank,9)-1;
  }
  if(Global::isSuffix(rankStrLower," pro")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower," pro"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return std::max(rank,9)-1;
  }
  if(Global::isPrefix(rankStr,"P") && Global::isSuffix(rankStr,"\346\256\265")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(Global::chopPrefix(rankStr,"P"),"\346\256\265"),rank);
    if(suc && rank >= 1 && rank <= TOP_DAN)
      return std::max(rank,9)-1;
  }
  if(Global::isSuffix(rankStrLower,"k")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower,"k"),rank);
    if(suc && rank >= 1 && rank <= BOTTOM_KYU)
      return -rank;
  }
  if(Global::isSuffix(rankStrLower," k")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower," k"),rank);
    if(suc && rank >= 1 && rank <= BOTTOM_KYU)
      return -rank;
  }
  if(Global::isSuffix(rankStrLower,"kyu")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower,"kyu"),rank);
    if(suc && rank >= 1 && rank <= BOTTOM_KYU)
      return -rank;
  }
  if(Global::isSuffix(rankStrLower," kyu")) {
    bool suc = Global::tryStringToInt(Global::chopSuffix(rankStrLower," kyu"),rank);
    if(suc && rank >= 1 && rank <= BOTTOM_KYU)
      return -rank;
  }
  propertyFail("Could not parse rank in sgf: " + rankStr);
  return Sgf::RANK_UNKNOWN;
}

int Sgf::getRating(Player pla) const {
  checkNonEmpty(nodes);
  string ratingStr;
  if(pla == P_BLACK) {
    if(!nodes[0]->hasProperty("BR"))
      propertyFail("Could not parse rating in sgf");
    ratingStr = nodes[0]->getSingleProperty("BR");
  }
  else if(pla == P_WHITE) {
    if(!nodes[0]->hasProperty("WR"))
      propertyFail("Could not find rating in sgf");
    ratingStr = nodes[0]->getSingleProperty("WR");
  }
  else {
    assert(false);
    propertyFail("Could not find rating in sgf");
  }

  int rating;
  bool suc = Global::tryStringToInt(ratingStr,rating);
  if(!suc)
    propertyFail("Could not parse rating in sgf: " + ratingStr);
  return rating;
}


string Sgf::getPlayerName(Player pla) const {
  if(pla == P_BLACK) {
    if(!nodes[0]->hasProperty("PB"))
      return "";
    return nodes[0]->getSingleProperty("PB");
  }
  else if(pla == P_WHITE) {
    if(!nodes[0]->hasProperty("PW"))
      return "";
    return nodes[0]->getSingleProperty("PW");
  }
  assert(false);
  return "";
}

bool Sgf::hasRootProperty(const std::string& property) const {
  if(nodes.size() <= 0)
    return false;
  return nodes[0]->hasProperty(property);
}

std::string Sgf::getRootPropertyWithDefault(const std::string& property, const std::string& defaultRet) const {
  if(nodes.size() <= 0)
    return defaultRet;
  if(!nodes[0]->hasProperty(property))
    return defaultRet;
  return nodes[0]->getSingleProperty(property);
}

std::vector<std::string> Sgf::getRootProperties(const std::string& property) const {
  if(nodes.size() <= 0)
    return std::vector<std::string>();
  return nodes[0]->getProperties(property);
}

void Sgf::addRootProperty(const std::string& key, const std::string& value) {
  checkNonEmpty(nodes);
  nodes[0]->addProperty(key,value);
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

  int64_t maxChildDepth = 0;
  Sgf* maxChild = NULL;
  for(int i = 0; i<children.size(); i++) {
    int64_t childDepth = children[i]->depth();
    if(childDepth > maxChildDepth) {
      maxChildDepth = childDepth;
      maxChild = children[i];
    }
  }

  if(maxChild != NULL) {
    maxChild->getMovesHelper(moves,xSize,ySize);
  }
}


void Sgf::loadAllUniquePositions(
  std::set<Hash128>& uniqueHashes,
  bool hashComments,
  bool hashParent,
  bool flipIfPassOrWFirst,
  bool allowGameOver,
  Rand* rand,
  vector<PositionSample>& samples
) const {
  std::function<void(PositionSample&, const BoardHistory&, const string&)> f = [&samples](PositionSample& sample, const BoardHistory& hist, const string& comments) {
    (void)hist;
    (void)comments;
    samples.push_back(sample);
  };

  iterAllUniquePositions(uniqueHashes,hashComments,hashParent,flipIfPassOrWFirst,allowGameOver,rand,f);
}

void Sgf::iterAllUniquePositions(
  std::set<Hash128>& uniqueHashes,
  bool hashComments,
  bool hashParent,
  bool flipIfPassOrWFirst,
  bool allowGameOver,
  Rand* rand,
  std::function<void(PositionSample&,const BoardHistory&,const std::string&)> f
) const {
  XYSize size = getXYSize();
  int xSize = size.x;
  int ySize = size.y;

  Board board(xSize,ySize);
  Player nextPla = nodes.size() > 0 ? nodes[0]->getPLSpecifiedColor() : C_EMPTY;
  if(nextPla == C_EMPTY)
    nextPla = C_BLACK;
  Rules rules = Rules::getTrompTaylorish();
  rules.koRule = Rules::KO_SITUATIONAL;
  rules.multiStoneSuicideLegal = true;
  BoardHistory hist(board,nextPla,rules,0);

  PositionSample sampleBuf;
  std::vector<std::pair<int64_t,int64_t>> variationTraceNodesBranch;
  bool isRoot = true;
  bool requireUnique = true;
  iterAllPositionsHelper(
    board,hist,nextPla,rules,xSize,ySize,sampleBuf,uniqueHashes,requireUnique,hashComments,hashParent,flipIfPassOrWFirst,allowGameOver,isRoot,rand,variationTraceNodesBranch,f
  );
}
void Sgf::iterAllPositions(
  bool flipIfPassOrWFirst,
  bool allowGameOver,
  Rand* rand,
  std::function<void(PositionSample&,const BoardHistory&,const std::string&)> f
) const {
  XYSize size = getXYSize();
  int xSize = size.x;
  int ySize = size.y;

  Board board(xSize,ySize);
  Player nextPla = nodes.size() > 0 ? nodes[0]->getPLSpecifiedColor() : C_EMPTY;
  if(nextPla == C_EMPTY)
    nextPla = C_BLACK;
  Rules rules = Rules::getTrompTaylorish();
  rules.koRule = Rules::KO_SITUATIONAL;
  rules.multiStoneSuicideLegal = true;
  BoardHistory hist(board,nextPla,rules,0);

  PositionSample sampleBuf;
  std::vector<std::pair<int64_t,int64_t>> variationTraceNodesBranch;
  std::set<Hash128> uniqueHashes;
  bool isRoot = true;
  bool requireUnique = false;
  bool hashComments = false;
  bool hashParent = false;
  iterAllPositionsHelper(
    board,hist,nextPla,rules,xSize,ySize,sampleBuf,uniqueHashes,requireUnique,hashComments,hashParent,flipIfPassOrWFirst,allowGameOver,isRoot,rand,variationTraceNodesBranch,f
  );
}

void Sgf::iterAllPositionsHelper(
  Board& board, BoardHistory& hist, Player nextPla,
  const Rules& rules, int xSize, int ySize,
  PositionSample& sampleBuf,
  std::set<Hash128>& uniqueHashes,
  bool requireUnique,
  bool hashComments,
  bool hashParent,
  bool flipIfPassOrWFirst,
  bool allowGameOver,
  bool isRoot,
  Rand* rand,
  std::vector<std::pair<int64_t,int64_t>>& variationTraceNodesBranch,
  std::function<void(PositionSample&,const BoardHistory&,const std::string&)> f
) const {
  vector<Move> buf;
  for(size_t i = 0; i<nodes.size(); i++) {
    string comments;
    if(nodes[i]->hasProperty("C"))
      comments = nodes[i]->getSingleProperty("C");

    //Do the root node even if it has no placements since nothing else will do it.
    if(isRoot && i == 0 && !nodes[i]->hasPlacements()) {
      samplePositionHelper(board,hist,nextPla,sampleBuf,uniqueHashes,requireUnique,hashComments,hashParent,flipIfPassOrWFirst,allowGameOver,comments,f);
    }

    //Handle placements
    if(nodes[i]->hasPlacements()) {
      buf.clear();
      nodes[i]->accumPlacements(buf,xSize,ySize);
      if(buf.size() > 0) {
        int netStonesAdded = 0;
        for(size_t j = 0; j<buf.size(); j++) {
          if(board.colors[buf[j].loc] != C_EMPTY && buf[j].pla == C_EMPTY)
            netStonesAdded--;
          if(board.colors[buf[j].loc] == C_EMPTY && buf[j].pla != C_EMPTY)
            netStonesAdded++;
        }
        bool suc = board.setStonesFailIfNoLibs(buf);
        if(!suc) {
          ostringstream trace;
          for(size_t s = 0; s < variationTraceNodesBranch.size(); s++) {
            trace << "forward " << variationTraceNodesBranch[s].first << " ";
            trace << "branch " << variationTraceNodesBranch[s].second << " ";
          }
          trace << "forward " << i;

          throw StringError(
            "Illegal placements in " + fileName + " SGF trace (branches 0-indexed): " + trace.str()
          );
        }

        board.clearSimpleKoLoc();
        //Clear history any time placements happen, but make sure we track the initial turn number.
        int64_t initialTurnNumber = hist.initialTurnNumber;
        initialTurnNumber += (int64_t)hist.moveHistory.size();

        //If stones were net added, count each such stone as half of an initial turn.
        //Sort of hacky, but improves the correlation between initial turn and how full the board is compared
        //to not doing it.
        if(netStonesAdded > 0)
          initialTurnNumber += (netStonesAdded+1)/2;
        //Also make sure the turn number is at least as large as the number of stones in the board
        if(board.numStonesOnBoard() > initialTurnNumber)
          initialTurnNumber = board.numStonesOnBoard();

        hist.clear(board,nextPla,rules,0);
        hist.setInitialTurnNumber(initialTurnNumber);
      }
      samplePositionHelper(board,hist,nextPla,sampleBuf,uniqueHashes,requireUnique,hashComments,hashParent,flipIfPassOrWFirst,allowGameOver,comments,f);
    }

    //Handle actual moves
    buf.clear();
    nodes[i]->accumMoves(buf,xSize,ySize);

    for(size_t j = 0; j<buf.size(); j++) {
      bool suc = hist.makeBoardMoveTolerant(board,buf[j].loc,buf[j].pla);
      if(!suc) {
        ostringstream trace;
        for(size_t s = 0; s < variationTraceNodesBranch.size(); s++) {
          trace << "forward " << variationTraceNodesBranch[s].first << " ";
          trace << "branch " << variationTraceNodesBranch[s].second << " ";
        }
        trace << "forward " << i;

        // hist.printBasicInfo(trace, board);
        // hist.printDebugInfo(trace, board);
        // trace << Location::toString(buf[j].loc,board) << endl;

        throw StringError(
          "Illegal move in " + fileName + " effective turn " + Global::int64ToString((int64_t)(hist.moveHistory.size())+hist.initialTurnNumber) + " move " +
          Location::toString(buf[j].loc, board.x_size, board.y_size) + " SGF trace (branches 0-indexed): " + trace.str()
        );
      }
      if(hist.moveHistory.size() > 0x3FFFFFFF)
        throw StringError("too many moves in sgf");
      nextPla = getOpp(buf[j].pla);
      samplePositionHelper(board,hist,nextPla,sampleBuf,uniqueHashes,requireUnique,hashComments,hashParent,flipIfPassOrWFirst,allowGameOver,comments,f);
    }
  }


  std::vector<size_t> permutation(children.size());
  for(size_t i = 0; i<children.size(); i++)
    permutation[i] = i;
  if(rand != NULL) {
    rand->shuffle(permutation);
  }

  for(size_t c = 0; c<children.size(); c++) {
    size_t i = permutation[c];
    std::unique_ptr<Board> copy = std::make_unique<Board>(board);
    std::unique_ptr<BoardHistory> histCopy = std::make_unique<BoardHistory>(hist);
    variationTraceNodesBranch.push_back(std::make_pair((int64_t)nodes.size(),(int64_t)i));
    children[i]->iterAllPositionsHelper(
      *copy,*histCopy,nextPla,rules,xSize,ySize,sampleBuf,uniqueHashes,requireUnique,hashComments,hashParent,flipIfPassOrWFirst,allowGameOver,false,rand,variationTraceNodesBranch,f
    );
    assert(variationTraceNodesBranch.size() > 0);
    variationTraceNodesBranch.erase(variationTraceNodesBranch.begin()+(variationTraceNodesBranch.size()-1));
  }
}

void Sgf::samplePositionHelper(
  Board& board, BoardHistory& hist, Player nextPla,
  PositionSample& sampleBuf,
  std::set<Hash128>& uniqueHashes,
  bool requireUnique,
  bool hashComments,
  bool hashParent,
  bool flipIfPassOrWFirst,
  bool allowGameOver,
  const std::string& comments,
  std::function<void(PositionSample&,const BoardHistory&,const std::string&)> f
) const {
  //If the game is over or there were two consecutive passes, skip
  if(!allowGameOver) {
    if(hist.isGameFinished || (
         hist.moveHistory.size() >= 2
         && hist.moveHistory[hist.moveHistory.size()-1].loc == Board::PASS_LOC
         && hist.moveHistory[hist.moveHistory.size()-2].loc == Board::PASS_LOC
       ))
      return;
  }

  //Hash based on position, player, and simple ko
  Hash128 situationHash = board.pos_hash;
  situationHash ^= Board::ZOBRIST_PLAYER_HASH[nextPla];
  assert(hist.encorePhase == 0);
  if(board.ko_loc != Board::NULL_LOC)
    situationHash ^= Board::ZOBRIST_KO_LOC_HASH[board.ko_loc];

  if(hashComments)
    situationHash.hash0 += Hash::simpleHash(comments.c_str());

  if(hashParent) {
    Hash128 parentHash = Hash128();
    if(hist.moveHistory.size() > 0) {
      const Board& prevBoard = hist.getRecentBoard(1);
      parentHash = prevBoard.pos_hash;
      if(prevBoard.ko_loc != Board::NULL_LOC)
        parentHash ^= Board::ZOBRIST_KO_LOC_HASH[prevBoard.ko_loc];
    }
    //Mix in a blended up hash of the previous board state to avoid zobrist cancellation, also swapping halves
    Hash128 mixed = Hash128(Hash::murmurMix(parentHash.hash1),Hash::splitMix64(parentHash.hash0));
    situationHash ^= mixed;
  }

  if(requireUnique && contains(uniqueHashes,situationHash))
    return;
  uniqueHashes.insert(situationHash);

  //Snap the position 5 turns ago so as to include 5 moves of history.
  assert(BoardHistory::NUM_RECENT_BOARDS > 5);
  int turnsAgoToSnap = 0;
  while(turnsAgoToSnap < 5) {
    if(turnsAgoToSnap >= hist.moveHistory.size())
      break;
    //If a player played twice in a row, then instead snap so as not to have a move history
    //with a double move by the same player.
    if(turnsAgoToSnap > 0 && hist.moveHistory[hist.moveHistory.size() - turnsAgoToSnap - 1].pla == hist.moveHistory[hist.moveHistory.size() - turnsAgoToSnap].pla)
      break;
    if(turnsAgoToSnap == 0 && hist.moveHistory[hist.moveHistory.size() - turnsAgoToSnap - 1].pla == nextPla)
      break;
    turnsAgoToSnap++;
  }
  if(hist.moveHistory.size() > 0x3FFFFFFF)
    throw StringError("hist has too many moves");
  int64_t startTurnIdx = (int64_t)hist.moveHistory.size() - turnsAgoToSnap;

  sampleBuf.board = hist.getRecentBoard(turnsAgoToSnap);
  if(startTurnIdx < hist.moveHistory.size())
    sampleBuf.nextPla = hist.moveHistory[startTurnIdx].pla;
  else
    sampleBuf.nextPla = nextPla;
  sampleBuf.moves.clear();
  for(int64_t i = startTurnIdx; i<(int64_t)hist.moveHistory.size(); i++)
    sampleBuf.moves.push_back(hist.moveHistory[i]);
  sampleBuf.initialTurnNumber = hist.initialTurnNumber + startTurnIdx;
  sampleBuf.hintLoc = Board::NULL_LOC;
  sampleBuf.weight = 1.0;

  if(flipIfPassOrWFirst) {
    if(hist.hasBlackPassOrWhiteFirst())
      sampleBuf = sampleBuf.getColorFlipped();
  }

  f(sampleBuf,hist,comments);
}

static uint64_t parseHex64(const string& str) {
  assert(str.length() == 16);
  uint64_t x = 0;
  for(int i = 0; i<16; i++) {
    x *= 16;
    if(str[i] >= '0' && str[i] <= '9')
      x += str[i] - '0';
    else if(str[i] >= 'a' && str[i] <= 'f')
      x += str[i] - 'a' + 10;
    else if(str[i] >= 'A' && str[i] <= 'F')
      x += str[i] - 'A' + 10;
    else
      assert(false);
  }
  return x;
}

set<Hash128> Sgf::readExcludes(const vector<string>& files) {
  set<Hash128> excludeHashes;
  for(const string& file: files) {
    string excludeHashesFile = Global::trim(file);
    if(excludeHashesFile.size() <= 0)
      continue;
    vector<string> hashes = FileUtils::readFileLines(excludeHashesFile,'\n');
    for(const string& hashStr: hashes) {
      string hash128 = Global::trim(Global::stripComments(hashStr));
      if(hash128.length() <= 0)
        continue;
      if(hash128.length() != 32)
        throw IOError("Could not parse hashpair in exclude hashes file: " + hash128);

      uint64_t hash0 = parseHex64(hash128.substr(0,16));
      uint64_t hash1 = parseHex64(hash128.substr(16,16));
      excludeHashes.insert(Hash128(hash0,hash1));
    }
  }
  return excludeHashes;
}

string Sgf::PositionSample::toJsonLine(const Sgf::PositionSample& sample) {
  json data;
  data["xSize"] = sample.board.x_size;
  data["ySize"] = sample.board.y_size;
  data["board"] = Board::toStringSimple(sample.board,'/');
  data["nextPla"] = PlayerIO::playerToStringShort(sample.nextPla);
  vector<string> moveLocs;
  vector<string> movePlas;
  for(size_t i = 0; i<sample.moves.size(); i++)
    moveLocs.push_back(Location::toString(sample.moves[i].loc,sample.board));
  for(size_t i = 0; i<sample.moves.size(); i++)
    movePlas.push_back(PlayerIO::playerToStringShort(sample.moves[i].pla));

  data["moveLocs"] = moveLocs;
  data["movePlas"] = movePlas;
  data["initialTurnNumber"] = sample.initialTurnNumber;
  data["hintLoc"] = Location::toString(sample.hintLoc,sample.board);
  data["weight"] = sample.weight;
  if(sample.metadata.size() > 0)
    data["metadata"] = sample.metadata;
  if(sample.trainingWeight != 1.0)
    data["trainingWeight"] = sample.trainingWeight;
  return data.dump();
}

Sgf::PositionSample Sgf::PositionSample::ofJsonLine(const string& s) {
  json data = json::parse(s);
  PositionSample sample;
  try {
    int xSize = data["xSize"].get<int>();
    int ySize = data["ySize"].get<int>();
    sample.board = Board::parseBoard(xSize,ySize,data["board"].get<string>(),'/');
    sample.nextPla = PlayerIO::parsePlayer(data["nextPla"].get<string>());
    vector<string> moveLocs = data["moveLocs"].get<vector<string>>();
    vector<string> movePlas = data["movePlas"].get<vector<string>>();
    if(moveLocs.size() != movePlas.size())
      throw StringError("moveLocs.size() != movePlas.size()");
    for(size_t i = 0; i<moveLocs.size(); i++) {
      Loc moveLoc = Location::ofString(moveLocs[i],sample.board);
      Player movePla = PlayerIO::parsePlayer(movePlas[i]);
      sample.moves.push_back(Move(moveLoc,movePla));
    }
    sample.initialTurnNumber = data["initialTurnNumber"].get<int64_t>();
    string hintLocStr = Global::toLower(Global::trim(data["hintLoc"].get<string>()));
    if(hintLocStr == "" || hintLocStr == "''" || hintLocStr == "\"\"" ||
       hintLocStr == "null" || hintLocStr == "'null'" || hintLocStr == "\"null\"")
      sample.hintLoc = Board::NULL_LOC;
    else
      sample.hintLoc = Location::ofString(data["hintLoc"].get<string>(),sample.board);

    if(data.find("weight") != data.end())
      sample.weight = data["weight"].get<double>();
    else
      sample.weight = 1.0;

    if(data.find("metadata") != data.end())
      sample.metadata = data["metadata"].get<string>();
    else
      sample.metadata = string();

    if(data.find("trainingWeight") != data.end())
      sample.trainingWeight = data["trainingWeight"].get<double>();
    else
      sample.trainingWeight = 1.0;
  }
  catch(nlohmann::detail::exception& e) {
    throw StringError("Error parsing position sample json\n" + s + "\n" + e.what());
  }
  return sample;
}

Sgf::PositionSample Sgf::PositionSample::getColorFlipped() const {
  Sgf::PositionSample other = *this;
  Board newBoard(other.board.x_size,other.board.y_size);
  for(int y = 0; y < other.board.y_size; y++) {
    for(int x = 0; x < other.board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,other.board.x_size);
      if(other.board.colors[loc] == C_BLACK || other.board.colors[loc] == C_WHITE) {
        bool suc = newBoard.setStoneFailIfNoLibs(loc, getOpp(other.board.colors[loc]));
        assert(suc);
        (void)suc;
      }
    }
  }
  other.board = newBoard;
  other.nextPla = getOpp(other.nextPla);
  for(int i = 0; i<other.moves.size(); i++)
    other.moves[i].pla = getOpp(other.moves[i].pla);

  return other;
}

bool Sgf::PositionSample::hasPreviousPositions(int numPrevious) const {
  return moves.size() >= numPrevious;
}

Sgf::PositionSample Sgf::PositionSample::previousPosition(double newWeight) const {
  Sgf::PositionSample other = *this;
  if(other.moves.size() > 0) {
    other.moves.pop_back();
    other.hintLoc = Board::NULL_LOC;
    other.weight = newWeight;
  }
  return other;
}

bool Sgf::PositionSample::tryGetCurrentBoardHistory(const Rules& rules, Player& nextPlaToMove, BoardHistory& hist) const {
  int encorePhase = 0;
  Player pla = nextPla;
  Board boardCopy = board;
  hist.clear(boardCopy,pla,rules,encorePhase);
  int numSampleMoves = (int)moves.size();
  for(int i = 0; i<numSampleMoves; i++) {
    if(!hist.isLegal(boardCopy,moves[i].loc,moves[i].pla))
      return false;
    assert(moves[i].pla == pla);
    hist.makeBoardMoveAssumeLegal(boardCopy,moves[i].loc,moves[i].pla,NULL);
    pla = getOpp(pla);
  }
  nextPlaToMove = pla;
  return true;
}

int64_t Sgf::PositionSample::getCurrentTurnNumber() const {
  return std::max((int64_t)0, initialTurnNumber + (int64_t)moves.size());
}

bool Sgf::PositionSample::isEqualForTesting(const Sgf::PositionSample& other, bool checkNumCaptures, bool checkSimpleKo) const {
  if(!board.isEqualForTesting(other.board,checkNumCaptures,checkSimpleKo))
    return false;
  if(nextPla != other.nextPla)
    return false;
  if(moves.size() != other.moves.size())
    return false;
  for(size_t i = 0; i<moves.size(); i++) {
    if(moves[i].pla != other.moves[i].pla)
      return false;
    if(moves[i].loc != other.moves[i].loc)
      return false;
  }
  if(initialTurnNumber != other.initialTurnNumber)
    return false;
  if(hintLoc != other.hintLoc)
    return false;
  if(weight != other.weight)
    return false;
  return true;
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
    //Skip any BOM at the start of the file
    if(newPos == 0 && (str.length() >= 3 && str[0] == (char)0xEF && str[1] == (char)0xBB && str[2] == (char)0xBF)) {
      newPos += 3;
      continue;
    }

    char c = str[newPos++];

    //Skip whitespace
    if(Global::isWhitespace(c))
      continue;
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
      node->addProperty(key,parseTextValue(str,pos));
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

  // Hack for missing handicap placements in fox
  int handicap = 0;
  if(sgf->nodes.size() > 1
     && sgf->nodes[0]->hasProperty("AP")
     && (
       contains(sgf->nodes[0]->getProperties("AP"),"foxwq")
       || (
         contains(sgf->nodes[0]->getProperties("AP"),"GNU Go:3.8") // Some older fox games are labeled as gnugo only
         && sgf->getRootPropertyWithDefault("GN","-") == "" // But also have this identifying characteristic
       )
     )
     && sgf->getRootPropertyWithDefault("SZ","") == "19"
     && !sgf->nodes[0]->hasPlacements()
     && sgf->nodes[0]->move.pla == C_EMPTY
     && sgf->nodes[1]->move.pla == C_WHITE
     && Global::tryStringToInt(sgf->getRootPropertyWithDefault("HA",""),handicap)
     && handicap >= 2
     && handicap <= 9
  ) {
    Board board(19,19);
    PlayUtils::placeFixedHandicap(board, handicap);
    // Older fox sgfs used handicaps with side stones on the north and south rather than east and west
    if(handicap == 6 || handicap == 7) {
      if(sgf->hasRootProperty("DT")) {
        bool suc = false;
        SimpleDate date;
        try {
          date = SimpleDate(sgf->getRootPropertyWithDefault("DT",""));
          suc = true;
        }
        catch(const StringError&) {}
        if(suc && date < SimpleDate(2018,1,1)) {
          board = SymmetryHelpers::getSymBoard(board,4);
          }
      }
    }

    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_BLACK) {
          ostringstream out;
          writeSgfLoc(out, Location::getLoc(x,y,board.x_size), board.x_size, board.y_size);
          sgf->addRootProperty("AB",out.str());
        }
      }
    }
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
  Sgf* sgf = parse(FileUtils::readFile(file));
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
  vector<string> lines = FileUtils::readFileLines(file,'\n');
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

std::vector<Sgf*> Sgf::loadSgfOrSgfsLogAndIgnoreErrors(const string& fileName, Logger& logger) {
  if(FileHelpers::isMultiSgfs(fileName)) {
    try {
      std::vector<Sgf*> loaded = Sgf::loadSgfsFile(fileName);
      return loaded;
    }
    catch(const StringError& e) {
      logger.write("Invalid SGFS " + fileName + ": " + e.what());
      return std::vector<Sgf*>();
    }
  }
  else {
    Sgf* sgf = NULL;
    try {
      sgf = Sgf::loadFile(fileName);
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + fileName + ": " + e.what());
      return std::vector<Sgf*>();
    }
    std::vector<Sgf*> ret;
    ret.push_back(sgf);
    return ret;
  }
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
  hash = sgf->hash;

  sgf->getPlacements(placements, xSize, ySize);
  sgf->getMoves(moves, xSize, ySize);

  checkNonEmpty(sgf->nodes);
  rootNode = *(sgf->nodes[0]);

  sgfWinner = rootNode.getSgfWinner();
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

  sgfWinner = rootNode.getSgfWinner();
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

bool CompactSgf::hasRules() const {
  return rootNode.hasProperty("RU");
}

Rules CompactSgf::getRulesOrFail() const {
  Rules rules = rootNode.getRulesFromRUTagOrFail();
  rules.komi = rootNode.getKomiOrFail();
  return rules;
}

Rules CompactSgf::getRulesOrFailAllowUnspecified(const Rules& defaultRules) const {
  Rules rules;
  if(!hasRules())
    rules = defaultRules;
  else
    rules = rootNode.getRulesFromRUTagOrFail();

  if(rootNode.hasProperty("KM"))
    rules.komi = rootNode.getKomiOrFail();
  return rules;
}

Rules CompactSgf::getRulesOrWarn(const Rules& defaultRules, std::function<void(const string& msg)> f) const {
  if(!hasRules()) {
    Rules rules = defaultRules;
    if(rootNode.hasProperty("KM")) {
      try {
        rules.komi = rootNode.getKomiOrFail();
      }
      catch(const std::exception& e) {
        f("Sgf has no rules, also there was an error parsing komi, using default rules: " + rules.toString());
        return rules;
      }
      f("Sgf has no rules, using default rules with SGF-specified komi: " + rules.toString());
      return rules;
    }
    else {
      f("Sgf has no rules or komi, using default rules: " + rules.toString());
      return rules;
    }
  }

  Rules rules;
  try {
    rules = rootNode.getRulesFromRUTagOrFail();
  }
  catch(const std::exception& e) {
    rules = defaultRules;
    if(rootNode.hasProperty("KM")) {
      try {
        rules.komi = rootNode.getKomiOrFail();
      }
      catch(const std::exception&) {}
    }
    f("WARNING: using default rules " + rules.toString() + " because could not parse sgf rules: " + e.what());
    return rules;
  }

  if(rootNode.hasProperty("KM")) {
    try {
      rules.komi = rootNode.getKomiOrFail();
    }
    catch(const std::exception& e) {
      f("There was an error parsing komi, using default komi with rules: " + rules.toString());
    }
  }
  return rules;
}


void CompactSgf::setupInitialBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist) const {
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
    if(hasBlack && allBlack)
      nextPla = P_WHITE;
    else
      nextPla = P_BLACK;
  }

  // Override with the actual color of the move, if it exists
  if(moves.size() > 0)
    nextPla = moves[0].pla;

  board = Board(xSize,ySize);
  bool suc = board.setStonesFailIfNoLibs(placements);
  if(!suc)
    throw StringError("setupInitialBoardAndHist: initial board position contains invalid stones or zero-liberty stones");
  hist = BoardHistory(board,nextPla,initialRules,0);
  if(hist.initialTurnNumber < board.numStonesOnBoard())
    hist.initialTurnNumber = board.numStonesOnBoard();
}

void CompactSgf::playMovesAssumeLegal(Board& board, Player& nextPla, BoardHistory& hist, int64_t turnIdx) const {
  if(turnIdx < 0 || turnIdx > (int64_t)moves.size())
    throw StringError(
      Global::strprintf(
        "Attempting to set up position from SGF for invalid turn idx %lld, valid values are %lld to %lld",
        (long long)turnIdx, (long long)0, (long long)moves.size()
      )
    );

  for(int64_t i = 0; i<turnIdx; i++) {
    hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
    nextPla = getOpp(moves[i].pla);
  }
}

void CompactSgf::playMovesTolerant(Board& board, Player& nextPla, BoardHistory& hist, int64_t turnIdx, bool preventEncore) const {
  if(turnIdx < 0 || turnIdx > (int64_t)moves.size())
    throw StringError(
      Global::strprintf(
        "Attempting to set up position from SGF for invalid turn idx %lld, valid values are %lld to %lld",
        (long long)turnIdx, (long long)0, (long long)moves.size()
      )
    );

  for(int64_t i = 0; i<turnIdx; i++) {
    bool suc = hist.makeBoardMoveTolerant(board,moves[i].loc,moves[i].pla,preventEncore);
    if(!suc)
      throw StringError("Illegal move in " + fileName + " turn " + Global::int64ToString(i) + " move " + Location::toString(moves[i].loc, board.x_size, board.y_size));
    nextPla = getOpp(moves[i].pla);
  }
}

void CompactSgf::setupBoardAndHistAssumeLegal(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int64_t turnIdx) const {
  setupInitialBoardAndHist(initialRules, board, nextPla, hist);
  playMovesAssumeLegal(board, nextPla, hist, turnIdx);
}

void CompactSgf::setupBoardAndHistTolerant(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int64_t turnIdx, bool preventEncore) const {
  setupInitialBoardAndHist(initialRules, board, nextPla, hist);
  playMovesTolerant(board, nextPla, hist, turnIdx, preventEncore);
}


void WriteSgf::printGameResult(ostream& out, const BoardHistory& hist)
{
  printGameResult(out,hist,std::numeric_limits<double>::quiet_NaN());
}
void WriteSgf::printGameResult(ostream& out, const BoardHistory& hist, double overrideFinishedWhiteScore) {
  if(hist.isGameFinished) {
    out << "RE[";
    out << WriteSgf::gameResultNoSgfTag(hist, overrideFinishedWhiteScore);
    out << "]";
  }
}

string WriteSgf::gameResultNoSgfTag(const BoardHistory& hist) {
  return gameResultNoSgfTag(hist,std::numeric_limits<double>::quiet_NaN());
}
string WriteSgf::gameResultNoSgfTag(const BoardHistory& hist, double overrideFinishedWhiteScore) {
  if(!hist.isGameFinished)
    return "";
  else if(hist.isNoResult)
    return "Void";
  else if(hist.isResignation && hist.winner == C_BLACK)
    return "B+R";
  else if(hist.isResignation && hist.winner == C_WHITE)
    return "W+R";

  if(!std::isnan(overrideFinishedWhiteScore)) {
    if(overrideFinishedWhiteScore < 0)
      return "B+" + Global::doubleToString(-overrideFinishedWhiteScore);
    else if(overrideFinishedWhiteScore > 0)
      return "W+" + Global::doubleToString(overrideFinishedWhiteScore);
    else
      return "0";
  }
  else {
    if(hist.winner == C_BLACK)
      return "B+" + Global::doubleToString(-hist.finalWhiteMinusBlackScore);
    else if(hist.winner == C_WHITE)
      return "W+" + Global::doubleToString(hist.finalWhiteMinusBlackScore);
    else if(hist.winner == C_EMPTY)
      return "0";
    else
      ASSERT_UNREACHABLE;
  }
  return "";
}
void WriteSgf::writeSgf(
  ostream& out, const string& bName, const string& wName,
  const BoardHistory& endHist,
  const FinishedGameData* gameData,
  bool tryNicerRulesString,
  bool omitResignPlayerMove
) {
  writeSgf(
    out,
    bName,
    wName,
    endHist,
    gameData,
    tryNicerRulesString,
    omitResignPlayerMove,
    std::numeric_limits<double>::quiet_NaN()
  );
}

void WriteSgf::writeSgf(
  ostream& out, const string& bName, const string& wName,
  const BoardHistory& endHist,
  const FinishedGameData* gameData,
  bool tryNicerRulesString,
  bool omitResignPlayerMove,
  double overrideFinishedWhiteScore
) {
  writeSgf(
    out,
    bName,
    wName,
    endHist,
    gameData,
    tryNicerRulesString,
    omitResignPlayerMove,
    overrideFinishedWhiteScore,
    std::vector<std::string>()
  );
}


void WriteSgf::writeSgf(
  ostream& out, const string& bName, const string& wName,
  const BoardHistory& endHist,
  const std::vector<std::string>& extraComments
) {
  writeSgf(
    out,
    bName,
    wName,
    endHist,
    NULL,
    false,
    false,
    std::numeric_limits<double>::quiet_NaN(),
    extraComments
  );
}

void WriteSgf::writeSgf(
  ostream& out, const string& bName, const string& wName,
  const BoardHistory& endHist,
  const FinishedGameData* gameData,
  bool tryNicerRulesString,
  bool omitResignPlayerMove,
  double overrideFinishedWhiteScore,
  const std::vector<std::string>& extraComments
) {
  const Board& initialBoard = endHist.initialBoard;
  const Rules& rules = endHist.rules;

  int xSize = initialBoard.x_size;
  int ySize = initialBoard.y_size;
  out << "(;FF[4]GM[1]";
  if(xSize == ySize)
    out << "SZ[" << xSize << "]";
  else
    out << "SZ[" << xSize << ":" << ySize << "]";
  out << "PB[" << bName << "]";
  out << "PW[" << wName << "]";

  if(gameData != NULL) {
    out << "HA[" << gameData->handicapForSgf << "]";
  }
  else {
    BoardHistory histCopy(endHist);
    //Always use true for computing the handicap value that goes into an sgf
    histCopy.setAssumeMultipleStartingBlackMovesAreHandicap(true);
    out << "HA[" << histCopy.computeNumHandicapStones() << "]";
  }

  out << "KM[" << rules.komi << "]";
  out << "RU[" << (tryNicerRulesString ? rules.toStringNoKomiMaybeNice() : rules.toStringNoKomi()) << "]";
  printGameResult(out,endHist,overrideFinishedWhiteScore);

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

  size_t startTurnIdx = 0;
  ostringstream commentOut;
  if(gameData != NULL) {
    startTurnIdx = gameData->startHist.moveHistory.size();
    commentOut << "startTurnIdx=" << startTurnIdx;
    commentOut << ",initTurnNum=" << gameData->startHist.initialTurnNumber;
    commentOut << ",gameHash=" << gameData->gameHash;

    static_assert(FinishedGameData::NUM_MODES == 8, "");
    if(gameData->mode == FinishedGameData::MODE_NORMAL)
      commentOut << "," << "gtype=normal";
    else if(gameData->mode == FinishedGameData::MODE_CLEANUP_TRAINING)
      commentOut << "," << "gtype=cleanuptraining";
    else if(gameData->mode == FinishedGameData::MODE_FORK)
      commentOut << "," << "gtype=fork";
    else if(gameData->mode == FinishedGameData::MODE_HANDICAP)
      commentOut << "," << "gtype=handicap";
    else if(gameData->mode == FinishedGameData::MODE_SGFPOS)
      commentOut << "," << "gtype=sgfpos";
    else if(gameData->mode == FinishedGameData::MODE_HINTPOS)
      commentOut << "," << "gtype=hintpos";
    else if(gameData->mode == FinishedGameData::MODE_HINTFORK)
      commentOut << "," << "gtype=hintfork";
    else if(gameData->mode == FinishedGameData::MODE_ASYM)
      commentOut << "," << "gtype=asym";
    else
      commentOut << "," << "gtype=other";

    if(gameData->beganInEncorePhase != 0)
      commentOut << "," << "beganInEncorePhase=" << gameData->beganInEncorePhase;
    if(gameData->usedInitialPosition != 0)
      commentOut << "," << "usedInitialPosition=" << gameData->usedInitialPosition;
    if(gameData->playoutDoublingAdvantage != 0)
      commentOut << "," << "pdaWhite=" << ((gameData->playoutDoublingAdvantagePla == P_WHITE ? 1 : -1) * gameData->playoutDoublingAdvantage);

    for(int j = 0; j<gameData->changedNeuralNets.size(); j++) {
      commentOut << ",newNeuralNetTurn" << gameData->changedNeuralNets[j]->turnIdx
          << "=" << gameData->changedNeuralNets[j]->name;
    }
    if(gameData->bTimeUsed > 0 || gameData->wTimeUsed > 0) {
      commentOut << "," << "bTimeUsed=" << gameData->bTimeUsed;
      commentOut << "," << "wTimeUsed=" << gameData->wTimeUsed;
    }
    assert(endHist.moveHistory.size() <= startTurnIdx + gameData->whiteValueTargetsByTurn.size());
  }

  if(extraComments.size() > 0) {
    if(commentOut.str().length() > 0)
      commentOut << " ";
    commentOut << extraComments[0];
  }

  if(commentOut.str().length() > 0)
    out << "C[" << commentOut.str() << "]";

  string comment;
  Board board(initialBoard);
  BoardHistory hist(board,endHist.initialPla,endHist.rules,endHist.initialEncorePhase);
  for(size_t i = 0; i<endHist.moveHistory.size(); i++) {
    comment.clear();
    out << ";";

    Loc loc = endHist.moveHistory[i].loc;
    Player pla = endHist.moveHistory[i].pla;

    bool isResignMove = endHist.isGameFinished && endHist.isResignation && endHist.winner == getOpp(pla) && i+1 == endHist.moveHistory.size();
    if(!(omitResignPlayerMove && isResignMove)) {
      if(pla == P_BLACK)
        out << "B[";
      else
        out << "W[";

      bool isPassForKo = hist.isPassForKo(board,loc,pla);
      if(isPassForKo)
        writeSgfLoc(out,Board::PASS_LOC,xSize,ySize);
      else
        writeSgfLoc(out,loc,xSize,ySize);
      out << "]";

      if(isPassForKo) {
        out << "TR[";
        writeSgfLoc(out,loc,xSize,ySize);
        out << "]";
        comment += "Pass for ko";
      }
    }

    if(gameData != NULL && i >= startTurnIdx) {
      size_t turnAfterStart = i-startTurnIdx;
      if(turnAfterStart < gameData->whiteValueTargetsByTurn.size()) {
        const ValueTargets& targets = gameData->whiteValueTargetsByTurn[turnAfterStart];
        char winBuf[32];
        char lossBuf[32];
        char noResultBuf[32];
        char scoreBuf[32];
        sprintf(winBuf,"%.2f",targets.win);
        sprintf(lossBuf,"%.2f",targets.loss);
        sprintf(noResultBuf,"%.2f",targets.noResult);
        sprintf(scoreBuf,"%.1f",targets.score);
        if(comment.length() > 0)
          comment += " ";
        comment += winBuf;
        comment += " ";
        comment += lossBuf;
        comment += " ";
        comment += noResultBuf;
        comment += " ";
        comment += scoreBuf;
      }
      if(turnAfterStart < gameData->policyTargetsByTurn.size()) {
        char visitsBuf[32];
        sprintf(visitsBuf,"%d",(int)(gameData->policyTargetsByTurn[turnAfterStart].unreducedNumVisits));
        if(comment.length() > 0)
          comment += " ";
        comment += "v=";
        comment += visitsBuf;
      }
      if(turnAfterStart < gameData->targetWeightByTurnUnrounded.size()) {
        char weightBuf[32];
        sprintf(weightBuf,"%.2f",gameData->targetWeightByTurnUnrounded[turnAfterStart]);
        if(comment.length() > 0)
          comment += " ";
        comment += "weight=";
        comment += weightBuf;
      }
    }

    if(endHist.isGameFinished && i+1 == endHist.moveHistory.size()) {
      if(comment.length() > 0)
        comment += " ";
      comment += "result=" + WriteSgf::gameResultNoSgfTag(endHist,overrideFinishedWhiteScore);
    }

    if(extraComments.size() > i+1) {
      if(comment.length() > 0)
        comment += " ";
      comment += extraComments[i+1];
    }

    if(comment.length() > 0)
      out << "C[" << comment << "]";

    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);

  }
  out << ")";
}
