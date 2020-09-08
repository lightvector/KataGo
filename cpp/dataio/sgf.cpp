#include "../dataio/sgf.h"

#include "../core/sha2.h"

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

  if(!Rules::komiIsIntOrHalfInt(komi)) {
    //Hack - if the komi is a quarter integer and it looks like a Chines GoGoD file, then double komi and accept
    if(Rules::komiIsIntOrHalfInt(komi*2.0f) && nodes[0]->hasProperty("US") && nodes[0]->hasProperty("RU") &&
       Global::isPrefix(nodes[0]->getSingleProperty("US"),"GoGoD") &&
       Global::toLower(nodes[0]->getSingleProperty("RU")) == "chinese")
      komi *= 2.0f;
    else
      propertyFail("Komi in sgf is not integer or half-integer");
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
  //In SGF files the komi comes from the KM tag.
  rules.komi = getKomi();
  return rules;
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


void Sgf::loadAllUniquePositions(
  std::set<Hash128>& uniqueHashes, vector<PositionSample>& samples
) const {
  std::function<void(PositionSample&, const BoardHistory&)> f = [&samples](PositionSample& sample, const BoardHistory& hist) {
    (void)hist;
    samples.push_back(sample);
  };

  iterAllUniquePositions(uniqueHashes,f);
}

void Sgf::iterAllUniquePositions(
  std::set<Hash128>& uniqueHashes, std::function<void(PositionSample&,const BoardHistory&)> f
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
  iterAllUniquePositionsHelper(board,hist,nextPla,rules,xSize,ySize,sampleBuf,0,uniqueHashes,f);
}

void Sgf::iterAllUniquePositionsHelper(
  Board& board, BoardHistory& hist, Player nextPla,
  const Rules& rules, int xSize, int ySize,
  PositionSample& sampleBuf,
  int initialTurnNumber,
  std::set<Hash128>& uniqueHashes,
  std::function<void(PositionSample&,const BoardHistory&)> f
) const {
  vector<Move> buf;
  for(int i = 0; i<nodes.size(); i++) {

    //Handle placements
    if(nodes[i]->hasPlacements()) {
      buf.clear();
      nodes[i]->accumPlacements(buf,xSize,ySize);
      if(buf.size() > 0) {
        int netStonesAdded = 0;
        for(int j = 0; j<buf.size(); j++) {
          if(board.colors[buf[j].loc] != C_EMPTY && buf[j].pla == C_EMPTY)
            netStonesAdded--;
          if(board.colors[buf[j].loc] == C_EMPTY && buf[j].pla != C_EMPTY)
            netStonesAdded++;
          board.setStone(buf[j].loc, buf[j].pla);
        }
        board.clearSimpleKoLoc();
        //Clear history any time placements happen, but make sure we track the initial turn number.
        initialTurnNumber += hist.moveHistory.size();

        //If stones were net added, count each such stone as half of an initial turn.
        //Sort of hacky, but improves the correlation between initial turn and how full the board is compared
        //to not doing it.
        if(netStonesAdded > 0)
          initialTurnNumber += (netStonesAdded+1)/2;

        hist.clear(board,nextPla,rules,0);
      }
      samplePositionIfUniqueHelper(board,hist,nextPla,sampleBuf,initialTurnNumber,uniqueHashes,f);
    }

    //Handle actual moves
    buf.clear();
    nodes[i]->accumMoves(buf,xSize,ySize);

    for(int j = 0; j<buf.size(); j++) {
      bool suc = hist.makeBoardMoveTolerant(board,buf[j].loc,buf[j].pla);
      if(!suc)
        throw StringError("Illegal move in " + fileName + " turn " + Global::intToString(j) + " move " + Location::toString(buf[j].loc, board.x_size, board.y_size));
      nextPla = getOpp(buf[j].pla);
      samplePositionIfUniqueHelper(board,hist,nextPla,sampleBuf,initialTurnNumber,uniqueHashes,f);
    }
  }
  // if(children.size() > 1) {
  //   cout << "IN " << children.size() << endl;
  // }

  for(int i = 0; i<children.size(); i++) {
    Board copy = board;
    BoardHistory histCopy = hist;
    // if(children.size() > 1)
    //   cout << "BEGIN " << i << "/" << children.size() << endl;
    children[i]->iterAllUniquePositionsHelper(copy,histCopy,nextPla,rules,xSize,ySize,sampleBuf,initialTurnNumber,uniqueHashes,f);
  }

  // if(children.size() > 1) {
  //   cout << "OUT " << children.size() << endl;
  // }
}

void Sgf::samplePositionIfUniqueHelper(
  Board& board, BoardHistory& hist, Player nextPla,
  PositionSample& sampleBuf,
  int initialTurnNumber,
  std::set<Hash128>& uniqueHashes,
  std::function<void(PositionSample&,const BoardHistory&)> f
) const {
  //If the game is over or there were two consecutive passes, skip
  if(hist.isGameFinished || (
       hist.moveHistory.size() >= 2
       && hist.moveHistory[hist.moveHistory.size()-1].loc == Board::PASS_LOC
       && hist.moveHistory[hist.moveHistory.size()-2].loc == Board::PASS_LOC
     ))
    return;

  //Hash based on position, player, and simple ko
  Hash128 situationHash = board.pos_hash;
  situationHash ^= Board::ZOBRIST_PLAYER_HASH[nextPla];
  assert(hist.encorePhase == 0);
  if(board.ko_loc != Board::NULL_LOC)
    situationHash ^= Board::ZOBRIST_KO_LOC_HASH[board.ko_loc];

  if(contains(uniqueHashes,situationHash))
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
  int startTurn = hist.moveHistory.size() - turnsAgoToSnap;

  sampleBuf.board = hist.getRecentBoard(turnsAgoToSnap);
  if(startTurn < hist.moveHistory.size())
    sampleBuf.nextPla = hist.moveHistory[startTurn].pla;
  else
    sampleBuf.nextPla = nextPla;
  sampleBuf.moves.clear();
  for(int i = startTurn; i<hist.moveHistory.size(); i++)
    sampleBuf.moves.push_back(hist.moveHistory[i]);
  sampleBuf.initialTurnNumber = initialTurnNumber + startTurn;
  sampleBuf.hintLoc = Board::NULL_LOC;
  sampleBuf.weight = 1.0;
  f(sampleBuf,hist);
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
  for(int i = 0; i<files.size(); i++) {
    string excludeHashesFile = Global::trim(files[i]);
    if(excludeHashesFile.size() <= 0)
      continue;
    vector<string> hashes = Global::readFileLines(excludeHashesFile,'\n');
    for(int64_t j = 0; j < hashes.size(); j++) {
      const string& hash128 = Global::trim(Global::stripComments(hashes[j]));
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
    sample.initialTurnNumber = data["initialTurnNumber"].get<int>();
    sample.hintLoc = Location::ofString(data["hintLoc"].get<string>(),sample.board);

    if(data.find("weight") != data.end())
      sample.weight = data["weight"].get<double>();
    else
      sample.weight= 1.0;
  }
  catch(nlohmann::detail::exception& e) {
    throw StringError("Error parsing position sample json\n" + s + "\n" + e.what());
  }
  return sample;
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
  rules.komi = komi;
  return rules;
}

Rules CompactSgf::getRulesOrFailAllowUnspecified(const Rules& defaultRules) const {
  //But still carry over the komi no matter what!
  Rules rules = defaultRules;
  rules.komi = komi;

  if(!hasRules()) {
    return rules;
  }
  return getRulesOrFail();
}

Rules CompactSgf::getRulesOrWarn(const Rules& defaultRules, std::function<void(const string& msg)> f) const {
  //But still carry over the komi no matter what!
  Rules rules = defaultRules;
  rules.komi = komi;

  if(!hasRules()) {
    f("Sgf has no rules, using default rules: " + rules.toString());
    return rules;
  }
  try {
    return getRulesOrFail();
  }
  catch(const std::exception& e) {
    f("WARNING: using default rules " + rules.toString() + " because could not parse sgf rules: " + e.what());
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

void CompactSgf::playMovesAssumeLegal(Board& board, Player& nextPla, BoardHistory& hist, int turnNumber) const {
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

void CompactSgf::playMovesTolerant(Board& board, Player& nextPla, BoardHistory& hist, int turnNumber, bool preventEncore) const {
  if(turnNumber < 0 || turnNumber > moves.size())
    throw StringError(
      Global::strprintf(
        "Attempting to set up position from SGF for invalid turn number %d, valid values are %d to %d",
        (int)turnNumber, 0, (int)moves.size()
      )
    );

  for(size_t i = 0; i<turnNumber; i++) {
    bool suc = hist.makeBoardMoveTolerant(board,moves[i].loc,moves[i].pla,preventEncore);
    if(!suc)
      throw StringError("Illegal move in " + fileName + " turn " + Global::int64ToString(i) + " move " + Location::toString(moves[i].loc, board.x_size, board.y_size));
    nextPla = getOpp(moves[i].pla);
  }
}

void CompactSgf::setupBoardAndHistAssumeLegal(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int turnNumber) const {
  setupInitialBoardAndHist(initialRules, board, nextPla, hist);
  playMovesAssumeLegal(board, nextPla, hist, turnNumber);
}

void CompactSgf::setupBoardAndHistTolerant(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int turnNumber, bool preventEncore) const {
  setupInitialBoardAndHist(initialRules, board, nextPla, hist);
  playMovesTolerant(board, nextPla, hist, turnNumber, preventEncore);
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

string WriteSgf::gameResultNoSgfTag(const BoardHistory& hist) {
  if(!hist.isGameFinished)
    return "";
  else if(hist.isNoResult)
    return "Void";
  else if(hist.isResignation && hist.winner == C_BLACK)
    return "B+R";
  else if(hist.isResignation && hist.winner == C_WHITE)
    return "W+R";
  else if(hist.winner == C_BLACK)
    return "B+" + Global::doubleToString(-hist.finalWhiteMinusBlackScore);
  else if(hist.winner == C_WHITE)
    return "W+" + Global::doubleToString(hist.finalWhiteMinusBlackScore);
  else if(hist.winner == C_EMPTY)
    return "0";
  else
    ASSERT_UNREACHABLE;
  return "";
}

void WriteSgf::writeSgf(
  ostream& out, const string& bName, const string& wName,
  const BoardHistory& endHist,
  const FinishedGameData* gameData,
  bool tryNicerRulesString
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
  out << "RU[" << (tryNicerRulesString ? rules.toStringNoKomiMaybeNice() : rules.toStringNoKomi()) << "]";
  printGameResult(out,endHist);

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
  if(gameData != NULL) {
    startTurnIdx = gameData->startHist.moveHistory.size();
    out << "C[startTurnIdx=" << startTurnIdx;
    out << ",gameHash=" << gameData->gameHash;

    static_assert(FinishedGameData::NUM_MODES == 6, "");
    if(gameData->mode == FinishedGameData::MODE_NORMAL)
      out << "," << "mode=normal";
    else if(gameData->mode == FinishedGameData::MODE_CLEANUP_TRAINING)
      out << "," << "mode=cleanuptraining";
    else if(gameData->mode == FinishedGameData::MODE_FORK)
      out << "," << "mode=fork";
    else if(gameData->mode == FinishedGameData::MODE_HANDICAP)
      out << "," << "mode=handicap";
    else if(gameData->mode == FinishedGameData::MODE_SGFPOS)
      out << "," << "mode=sgfpos";
    else if(gameData->mode == FinishedGameData::MODE_HINTPOS)
      out << "," << "mode=hintpos";
    else
      out << "," << "mode=other";

    if(gameData->beganInEncorePhase != 0)
      out << "," << "beganInEncorePhase=" << gameData->beganInEncorePhase;
    if(gameData->usedInitialPosition != 0)
      out << "," << "usedInitialPosition=" << gameData->usedInitialPosition;

    for(int j = 0; j<gameData->changedNeuralNets.size(); j++) {
      out << ",newNeuralNetTurn" << gameData->changedNeuralNets[j]->turnNumber
          << "=" << gameData->changedNeuralNets[j]->name;
    }
    out << "]";
    assert(endHist.moveHistory.size() <= startTurnIdx + gameData->whiteValueTargetsByTurn.size());
  }

  string comment;
  Board board(initialBoard);
  BoardHistory hist(board,endHist.initialPla,endHist.rules,endHist.initialEncorePhase);
  for(size_t i = 0; i<endHist.moveHistory.size(); i++) {
    comment.clear();
    out << ";";

    Loc loc = endHist.moveHistory[i].loc;
    Player pla = endHist.moveHistory[i].pla;

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
    }

    if(comment.length() > 0)
      out << "C[" << comment << "]";

    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);

  }
  out << ")";
}
