
#include "../book/book.h"

#include <fstream>
#include <thread>
#include "../core/makedir.h"
#include "../core/fancymath.h"
#include "../core/fileutils.h"
#include "../game/graphhash.h"
#include "../neuralnet/nninputs.h"
#include "../external/nlohmann_json/json.hpp"

//------------------------
#include "../core/using.h"
//------------------------

using nlohmann::json;

static double square(double x) {
  return x * x;
}
static double pow3(double x) {
  return x * x * x;
}
static double pow7(double x) {
  double cube = x * x * x;
  return cube * cube * x;
}

// Clamp the score at contradicting the winloss too much for purposes of sorting
static double clampScoreForSorting(double score, double winLoss) {
  winLoss = std::max(-1.0, std::min(1.0, winLoss));
  double scoreLowerBound = (winLoss - 1.0) / (winLoss + 1.0 + 0.0001) * 2;
  double scoreUpperBound = -(-winLoss - 1.0) / (-winLoss + 1.0 + 0.0001) * 2;
  return std::max(scoreLowerBound, std::min(scoreUpperBound, score));
}

BookHash::BookHash()
:historyHash(),stateHash()
{}

BookHash::BookHash(Hash128 hhash, Hash128 shash)
:historyHash(hhash),stateHash(shash)
{}

bool BookHash::operator==(const BookHash other) const
{return historyHash == other.historyHash && stateHash == other.stateHash;}

bool BookHash::operator!=(const BookHash other) const
{return historyHash != other.historyHash || stateHash != other.stateHash;}

bool BookHash::operator>(const BookHash other) const
{
  if(stateHash > other.stateHash) return true;
  if(stateHash < other.stateHash) return false;
  return historyHash > other.historyHash;
}
bool BookHash::operator>=(const BookHash other) const
{
  if(stateHash > other.stateHash) return true;
  if(stateHash < other.stateHash) return false;
  return historyHash >= other.historyHash;
}
bool BookHash::operator<(const BookHash other) const
{
  if(stateHash < other.stateHash) return true;
  if(stateHash > other.stateHash) return false;
  return historyHash < other.historyHash;
}
bool BookHash::operator<=(const BookHash other) const
{
  if(stateHash < other.stateHash) return true;
  if(stateHash > other.stateHash) return false;
  return historyHash <= other.historyHash;
}

BookHash BookHash::operator^(const BookHash other) const {
  return BookHash(historyHash ^ other.historyHash, stateHash ^ other.stateHash);
}
BookHash BookHash::operator|(const BookHash other) const {
  return BookHash(historyHash | other.historyHash, stateHash | other.stateHash);
}
BookHash BookHash::operator&(const BookHash other) const {
  return BookHash(historyHash & other.historyHash, stateHash & other.stateHash);
}
BookHash& BookHash::operator^=(const BookHash other) {
  historyHash ^= other.historyHash;
  stateHash ^= other.stateHash;
  return *this;
}
BookHash& BookHash::operator|=(const BookHash other) {
  historyHash |= other.historyHash;
  stateHash |= other.stateHash;
  return *this;
}
BookHash& BookHash::operator&=(const BookHash other) {
  historyHash &= other.historyHash;
  stateHash &= other.stateHash;
  return *this;
}

std::ostream& operator<<(std::ostream& out, const BookHash other)
{
  out << other.stateHash << other.historyHash;
  return out;
}

std::string BookHash::toString() const {
  return stateHash.toString() + historyHash.toString();
}

BookHash BookHash::ofString(const string& s) {
  if(s.size() != 64)
    throw IOError("Could not parse as BookHash: " + s);
  Hash128 stateHash = Hash128::ofString(s.substr(0,32));
  Hash128 historyHash = Hash128::ofString(s.substr(32,32));
  return BookHash(historyHash,stateHash);
}

// Just to fill out the extra 128 bits we have with another independent zobrist
static Hash128 getExtraPosHash(const Board& board) {
  Hash128 hash;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      hash ^= Board::ZOBRIST_BOARD_HASH2[loc][board.colors[loc]];
    }
  }
  return hash;
}

void BookHash::getHashAndSymmetry(const BoardHistory& hist, int repBound, BookHash& hashRet, int& symmetryToAlignRet, vector<int>& symmetriesRet, int bookVersion) {
  Board boardsBySym[SymmetryHelpers::NUM_SYMMETRIES];
  BoardHistory histsBySym[SymmetryHelpers::NUM_SYMMETRIES];
  Hash128 accums[SymmetryHelpers::NUM_SYMMETRIES];

  // Make sure the book all matches orientation for rectangular boards.
  // Don't consider symmetries that change the lengths of x and y.
  // This also lets us be fairly lazy in the rest of the book implementation and not have to carefully consider every case whether
  // we need to swap xSize and ySize when passing into getSymLoc.
  int numSymmetries = (hist.getRecentBoard(0).x_size != hist.getRecentBoard(0).y_size) ?
    SymmetryHelpers::NUM_SYMMETRIES_WITHOUT_TRANSPOSE : SymmetryHelpers::NUM_SYMMETRIES;

  for(int symmetry = 0; symmetry < numSymmetries; symmetry++) {
    boardsBySym[symmetry] = SymmetryHelpers::getSymBoard(hist.initialBoard,symmetry);
    histsBySym[symmetry] = BoardHistory(boardsBySym[symmetry], hist.initialPla, hist.rules, hist.initialEncorePhase);
    accums[symmetry] = Hash128();
  }

  for(size_t i = 0; i<hist.moveHistory.size(); i++) {
    for(int symmetry = 0; symmetry < numSymmetries; symmetry++) {
      Loc moveLoc = SymmetryHelpers::getSymLoc(hist.moveHistory[i].loc, boardsBySym[symmetry], symmetry);
      Player movePla = hist.moveHistory[i].pla;
      // Add the next
      Hash128 nextHash;

      if(bookVersion >= 2) {
        constexpr double drawEquivalentWinsForWhite = 0.5;
        nextHash = GraphHash::getStateHash(histsBySym[symmetry],histsBySym[symmetry].presumedNextMovePla,drawEquivalentWinsForWhite);
      }
      // Old less-rigorous hashing
      else
        nextHash = boardsBySym[symmetry].pos_hash ^ Board::ZOBRIST_PLAYER_HASH[movePla];

      accums[symmetry].hash0 += nextHash.hash0;
      accums[symmetry].hash1 += nextHash.hash1;
      // Mix it up
      accums[symmetry].hash0 = Hash::splitMix64(accums[symmetry].hash0);
      accums[symmetry].hash1 = Hash::nasam(accums[symmetry].hash1);

      // Assume legal since we're only replaying moves from another history.
      histsBySym[symmetry].makeBoardMoveAssumeLegal(boardsBySym[symmetry], moveLoc, movePla, nullptr);
      if(boardsBySym[symmetry].simpleRepetitionBoundGt(moveLoc, repBound)) {
        accums[symmetry] = Hash128();
      }
    }
  }

  BookHash hashes[SymmetryHelpers::NUM_SYMMETRIES];
  for(int symmetry = 0; symmetry < numSymmetries; symmetry++) {
    Player nextPlayer = hist.presumedNextMovePla;
    constexpr double drawEquivalentWinsForWhite = 0.5;
    Hash128 stateHash = GraphHash::getStateHash(histsBySym[symmetry],nextPlayer,drawEquivalentWinsForWhite);
    hashes[symmetry] = BookHash(accums[symmetry] ^ getExtraPosHash(boardsBySym[symmetry]), stateHash);
  }

  if(bookVersion >= 2) {
    for(int symmetry = 0; symmetry < numSymmetries; symmetry++) {
      hashes[symmetry].historyHash.hash0 = Hash::murmurMix(hashes[symmetry].historyHash.hash0);
      hashes[symmetry].historyHash.hash1 = Hash::murmurMix(hashes[symmetry].historyHash.hash1);
      hashes[symmetry].stateHash.hash0 = Hash::murmurMix(hashes[symmetry].stateHash.hash0);
      hashes[symmetry].stateHash.hash1 = Hash::murmurMix(hashes[symmetry].stateHash.hash1);
    }
  }

  // Use the smallest symmetry that gives us the same hash
  int smallestSymmetry = 0;
  BookHash smallestHash;
  for(int symmetry = 0; symmetry < numSymmetries; symmetry++) {
    if(symmetry == 0 || hashes[symmetry] < smallestHash) {
      smallestSymmetry = symmetry;
      smallestHash = hashes[symmetry];
    }
  }

  hashRet = smallestHash;
  symmetryToAlignRet = smallestSymmetry;

  // Find all symmetries that preserve the smallestHash.
  symmetriesRet.clear();
  for(int symmetry = 0; symmetry < numSymmetries; symmetry++) {
    if(hashes[SymmetryHelpers::compose(smallestSymmetry,symmetry)] == smallestHash)
      symmetriesRet.push_back(symmetry);
  }
}

double BookValues::getAdjustedWinLossError(const Rules& rules) const {
  (void)rules;
  // Handle the problem where old model versions don't support error estimates and just treat as no error.
  if(winLossError < 0)
    return 0;
  return winLossError;
}

double BookValues::getAdjustedScoreError(const Rules& rules) const {
  // Handle the problem where old model versions don't support error estimates and just treat as no error.
  if(scoreError < 0)
    return 0;
  if(rules.gameResultWillBeInteger()) {
    double scoreVariance = scoreStdev * scoreStdev;
    // KataGo's formalization of the score variance in draw-allowed games will be systematically too high by 0.25
    // due to blurring of the score on the half-integer gridpoints.
    // Assumes drawEquivalentWinsForWhite = 0.5.
    double adjustedScoreVariance = scoreVariance - 0.25;
    // Make sure we don't go smaller than some tiny values
    if(adjustedScoreVariance < scoreVariance * 0.05)
      adjustedScoreVariance = scoreVariance * 0.05;
    return std::min(sqrt(adjustedScoreVariance),scoreError);
  }
  else {
    return std::min(scoreStdev,scoreError);
  }
}


// -----------------------------------------------------------------------------------------------------------

BookMove::BookMove()
  :move(Board::NULL_LOC),
   symmetryToAlign(0),
   hash(),
   rawPolicy(0.0),
   costFromRoot(0.0),
   isWLPV(false),
   biggestWLCostFromRoot(0.0)
{}

BookMove::BookMove(Loc mv, int s, BookHash h, double rp)
  :move(mv),
   symmetryToAlign(s),
   hash(h),
   rawPolicy(rp),
   costFromRoot(0.0),
   isWLPV(false),
   biggestWLCostFromRoot(0.0)
{}

BookMove BookMove::getSymBookMove(int symmetry, int xSize, int ySize) {
  BookMove ret(
    SymmetryHelpers::getSymLoc(move,xSize,ySize,symmetry),
    // This needs to be the symmetry that transform's retspace -> childspace
    // symmetry is the transform from orig -> ret
    // symmetryToAlign is the transform from orig -> child.
    // Therefore invert(symmetry) + symmetryToAlign is correct.
    SymmetryHelpers::compose(SymmetryHelpers::invert(symmetry),symmetryToAlign),
    hash,
    rawPolicy
  );
  ret.costFromRoot = costFromRoot;
  ret.isWLPV = isWLPV;
  ret.biggestWLCostFromRoot = biggestWLCostFromRoot;
  return ret;
}

// -----------------------------------------------------------------------------------------------------------


BookNode::BookNode(BookHash h, Book* b, Player p, const vector<int>& syms)
  :hash(h),
   book(b),
   pla(p),
   symmetries(syms),
   thisValuesNotInBook(),
   canExpand(true),
   canReExpand(true),
   moves(),
   parents(),
   bestParentIdx(0),
   recursiveValues(),
   minDepthFromRoot(0),
   minCostFromRoot(0),
   thisNodeExpansionCost(0),
   minCostFromRootWLPV(0),
   expansionIsWLPV(false),
   biggestWLCostFromRoot(0)
{}

BookNode::~BookNode() {
}

// -----------------------------------------------------------------------------------------------------------

SymBookNode::SymBookNode()
  :node(nullptr),
   symmetryOfNode(0),
   invSymmetryOfNode(0)
{}
ConstSymBookNode::ConstSymBookNode()
  :node(nullptr),
   symmetryOfNode(0),
   invSymmetryOfNode(0)
{}
SymBookNode::SymBookNode(std::nullptr_t)
  :node(nullptr),
   symmetryOfNode(0),
   invSymmetryOfNode(0)
{}
ConstSymBookNode::ConstSymBookNode(std::nullptr_t)
  :node(nullptr),
   symmetryOfNode(0),
   invSymmetryOfNode(0)
{}
SymBookNode::SymBookNode(BookNode* n, int s)
  :node(n),
   symmetryOfNode(s),
   invSymmetryOfNode(SymmetryHelpers::invert(s))
{}
ConstSymBookNode::ConstSymBookNode(const BookNode* n, int s)
  :node(n),
   symmetryOfNode(s),
   invSymmetryOfNode(SymmetryHelpers::invert(s))
{}
ConstSymBookNode::ConstSymBookNode(const SymBookNode& other)
  :node(other.node),
   symmetryOfNode(other.symmetryOfNode),
   invSymmetryOfNode(other.invSymmetryOfNode)
{}
ConstSymBookNode& ConstSymBookNode::operator=(const SymBookNode& other) {
  node = other.node;
  symmetryOfNode = other.symmetryOfNode;
  invSymmetryOfNode = other.invSymmetryOfNode;
  return *this;
}

bool SymBookNode::isNull() {
  return node == nullptr;
}
bool ConstSymBookNode::isNull() {
  return node == nullptr;
}

SymBookNode SymBookNode::applySymmetry(int symmetry) {
  // symmetry is the map from thisspace -> retspace
  // symmetryOfNode is the map from nodespace -> thisspace
  // The constructor will want the map from nodespace -> retspace
  return SymBookNode(node,SymmetryHelpers::compose(symmetryOfNode,symmetry));
}
ConstSymBookNode ConstSymBookNode::applySymmetry(int symmetry) {
  return ConstSymBookNode(node,SymmetryHelpers::compose(symmetryOfNode,symmetry));
}

Player SymBookNode::pla() {
  return node->pla;
}
Player ConstSymBookNode::pla() {
  return node->pla;
}

BookHash SymBookNode::hash() {
  return node->hash;
}
BookHash ConstSymBookNode::hash() {
  return node->hash;
}

vector<int> SymBookNode::getSymmetries() {
  vector<int> symmetries;
  for(int symmetry: node->symmetries)
    symmetries.push_back(SymmetryHelpers::compose(invSymmetryOfNode, symmetry, symmetryOfNode));
  return symmetries;
}
vector<int> ConstSymBookNode::getSymmetries() {
  vector<int> symmetries;
  for(int symmetry: node->symmetries)
    symmetries.push_back(SymmetryHelpers::compose(invSymmetryOfNode, symmetry, symmetryOfNode));
  return symmetries;
}


bool SymBookNode::isMoveInBook(Loc move) {
  return ConstSymBookNode(*this).isMoveInBook(move);
}
bool ConstSymBookNode::isMoveInBook(Loc move) {
  assert(node != nullptr);
  for(int symmetry: node->symmetries) {
    // invSymmetryOfNode is the map (symbooknodespace -> nodespace)
    // symmetry is the map nodespace -> nodespace that we should look for alternative versions of a move.
    // There is only one way to compose them, which is to left-compose invSymmetryOfNode, so we left-compose it (and not right-compose it).
    symmetry = SymmetryHelpers::compose(invSymmetryOfNode, symmetry);
    if(contains(node->moves, SymmetryHelpers::getSymLoc(move, node->book->initialBoard, symmetry)))
      return true;
  }
  return false;
}

int SymBookNode::numUniqueMovesInBook() {
  return ConstSymBookNode(*this).numUniqueMovesInBook();
}
int ConstSymBookNode::numUniqueMovesInBook() {
  assert(node != nullptr);
  return (int)(node->moves.size());
}

vector<BookMove> SymBookNode::getUniqueMovesInBook() {
  return ConstSymBookNode(*this).getUniqueMovesInBook();
}
vector<BookMove> ConstSymBookNode::getUniqueMovesInBook() {
  assert(node != nullptr);
  vector<BookMove> ret;
  for(std::pair<Loc,BookMove> kv: node->moves) {
    ret.push_back(kv.second.getSymBookMove(symmetryOfNode, node->book->initialBoard.x_size, node->book->initialBoard.y_size));
  }
  return ret;
}

BookValues& SymBookNode::thisValuesNotInBook() {
  assert(node != nullptr);
  return node->thisValuesNotInBook;
}
const BookValues& ConstSymBookNode::thisValuesNotInBook() {
  assert(node != nullptr);
  return node->thisValuesNotInBook;
}

bool& SymBookNode::canExpand() {
  assert(node != nullptr);
  return node->canExpand;
}
bool ConstSymBookNode::canExpand() {
  assert(node != nullptr);
  return node->canExpand;
}
bool& SymBookNode::canReExpand() {
  assert(node != nullptr);
  return node->canReExpand;
}
bool ConstSymBookNode::canReExpand() {
  assert(node != nullptr);
  return node->canReExpand;
}


const RecursiveBookValues& SymBookNode::recursiveValues() {
  assert(node != nullptr);
  return node->recursiveValues;
}
const RecursiveBookValues& ConstSymBookNode::recursiveValues() {
  assert(node != nullptr);
  return node->recursiveValues;
}

int SymBookNode::minDepthFromRoot() {
  assert(node != nullptr);
  return node->minDepthFromRoot;
}
int ConstSymBookNode::minDepthFromRoot() {
  assert(node != nullptr);
  return node->minDepthFromRoot;
}
double SymBookNode::minCostFromRoot() {
  assert(node != nullptr);
  return node->minCostFromRoot;
}
double ConstSymBookNode::minCostFromRoot() {
  assert(node != nullptr);
  return node->minCostFromRoot;
}
double SymBookNode::totalExpansionCost() {
  assert(node != nullptr);
  return node->minCostFromRoot + node->thisNodeExpansionCost;
}
double ConstSymBookNode::totalExpansionCost() {
  assert(node != nullptr);
  return node->minCostFromRoot + node->thisNodeExpansionCost;
}

SymBookNode SymBookNode::canonicalParent() {
  if(node->parents.size() <= 0)
    return SymBookNode(nullptr);
  int64_t bestParentIdx = node->bestParentIdx;
  if(bestParentIdx < 0 || bestParentIdx >= node->parents.size())
    bestParentIdx = 0;
  BookNode* parent = node->book->get(node->parents[bestParentIdx].first);
  if(parent == nullptr)
    return SymBookNode(nullptr);
  auto iter = parent->moves.find(node->parents[bestParentIdx].second);
  if(iter == parent->moves.end())
    return SymBookNode(nullptr);
  const BookMove& moveFromParent = iter->second;
  // moveFromParent.symmetryToAlign is the map parentspace -> nodespace
  // symmetryOfNode is the map nodespace -> displayspace
  // For the constructor, we need the map parentspace -> displayspace
  return SymBookNode(parent,SymmetryHelpers::compose(moveFromParent.symmetryToAlign,symmetryOfNode));
}
ConstSymBookNode ConstSymBookNode::canonicalParent() {
  if(node->parents.size() <= 0)
    return ConstSymBookNode(nullptr);
  int64_t bestParentIdx = node->bestParentIdx;
  if(bestParentIdx < 0 || bestParentIdx >= node->parents.size())
    bestParentIdx = 0;
  const BookNode* parent = node->book->get(node->parents[bestParentIdx].first);
  if(parent == nullptr)
    return ConstSymBookNode(nullptr);
  auto iter = parent->moves.find(node->parents[bestParentIdx].second);
  if(iter == parent->moves.end())
    return ConstSymBookNode(nullptr);
  const BookMove& moveFromParent = iter->second;
  // moveFromParent.symmetryToAlign is the map parentspace -> nodespace
  // symmetryOfNode is the map nodespace -> displayspace
  // For the constructor, we need the map parentspace -> displayspace
  return ConstSymBookNode(parent,SymmetryHelpers::compose(moveFromParent.symmetryToAlign,symmetryOfNode));
}

SymBookNode SymBookNode::follow(Loc move) {
  assert(node != nullptr);
  for(int symmetry: node->symmetries) {
    // Same logic here, invSymmetryOfNode maps symbooknodespace -> nodespace
    symmetry = SymmetryHelpers::compose(invSymmetryOfNode, symmetry);
    auto iter = node->moves.find(SymmetryHelpers::getSymLoc(move, node->book->initialBoard, symmetry));
    if(iter != node->moves.end()) {
      const BookMove& bookMove = iter->second;
      BookNode* child = node->book->get(bookMove.hash);
      // Symmetry maps symbooknodespace -> nodespace orientation in which the move is found.
      // symmetryToAlign maps nodespace -> childspace
      // The constructor for this child SymBookNode expects the map childspace -> symbooknodespace
      // So we need to compose and invert.
      return SymBookNode(child,SymmetryHelpers::invert(SymmetryHelpers::compose(symmetry,bookMove.symmetryToAlign)));
    }
  }
  return SymBookNode(nullptr);
}
ConstSymBookNode ConstSymBookNode::follow(Loc move) {
  assert(node != nullptr);
  for(int symmetry: node->symmetries) {
    // Same logic here, invSymmetryOfNode maps symbooknodespace -> nodespace
    symmetry = SymmetryHelpers::compose(invSymmetryOfNode, symmetry);
    auto iter = node->moves.find(SymmetryHelpers::getSymLoc(move, node->book->initialBoard, symmetry));
    if(iter != node->moves.end()) {
      const BookMove& bookMove = iter->second;
      const BookNode* child = node->book->get(bookMove.hash);
      // Same logic here, compose and invert.
      return ConstSymBookNode(child,SymmetryHelpers::invert(SymmetryHelpers::compose(symmetry,bookMove.symmetryToAlign)));
    }
  }
  return ConstSymBookNode(nullptr);
}

SymBookNode SymBookNode::playMove(Board& board, BoardHistory& hist, Loc move) {
  SymBookNode ret = follow(move);
  if(ret.isNull())
    return SymBookNode(nullptr);
  if(!hist.isLegal(board,move,node->pla))
    return SymBookNode(nullptr);
  hist.makeBoardMoveAssumeLegal(board,move,node->pla,nullptr);
  return ret;
}
ConstSymBookNode ConstSymBookNode::playMove(Board& board, BoardHistory& hist, Loc move) {
  ConstSymBookNode ret = follow(move);
  if(ret.isNull())
    return ConstSymBookNode(nullptr);
  if(!hist.isLegal(board,move,node->pla))
    return ConstSymBookNode(nullptr);
  hist.makeBoardMoveAssumeLegal(board,move,node->pla,nullptr);
  return ret;
}

SymBookNode SymBookNode::playAndAddMove(Board& board, BoardHistory& hist, Loc move, double rawPolicy, bool& childIsTransposing) {
  assert(node != nullptr);
  assert(!isMoveInBook(move));
  childIsTransposing = false;

  if(!hist.isLegal(board,move,node->pla))
    return SymBookNode(nullptr);

  int xSize = node->book->initialBoard.x_size;
  int ySize = node->book->initialBoard.y_size;

  // Transform the move into the space of the node
  Loc symMove = SymmetryHelpers::getSymLoc(move,xSize,ySize,invSymmetryOfNode);

  // Find the symmetry for move that prefers the upper right corner if possible.
  // Maximize x first, then minimize y next
  // Although, this will only work politely for the initial empty board. Past that, it will really be just whatever orientation gets
  // chosen canonically for the board. Which is fine, I guess.
  // Even if the symmetry equivalence code isn't perfect, this should never choose an outright illegal move except for true hash
  // collisions because the state hash accounts for all ko and superko prohibitions, so at worst we'll be
  // just playing the wrong child node out.
  Loc bestLoc = symMove;
  int bestSymmetry = 0;
  for(int symmetry: node->symmetries) {
    if(symmetry == 0)
      continue;
    Loc symLoc = SymmetryHelpers::getSymLoc(symMove,xSize,ySize,symmetry);
    int symX = Location::getX(symLoc,xSize);
    int symY = Location::getY(symLoc,xSize);
    int bestX = Location::getX(bestLoc,xSize);
    int bestY = Location::getY(bestLoc,xSize);
    if(symX > bestX || (symX == bestX && symY < bestY)) {
      bestLoc = symLoc;
      bestSymmetry = symmetry;
    }
  }

  hist.makeBoardMoveAssumeLegal(board,move,node->pla,nullptr);
  BookHash childHash;
  int symmetryToAlignToChild;
  vector<int> symmetriesOfChild;
  BookHash::getHashAndSymmetry(hist, node->book->repBound, childHash, symmetryToAlignToChild, symmetriesOfChild, node->book->bookVersion);

  // Okay...
  // A: invSymmetryOfNode is the transform from SymBookNode space -> BookNode space
  // B: bestSymmetry is the transform from BookNode space -> polite BookNode space
  // C: symmetryToAlignToChild is the transform from SymBookNode space -> child space
  //
  // We need to fill newBookMove with the transform from polite BookNode space -> child space.
  // So we need to compose invert(B) and invert(A) and C
  // We need to return back to the user for the child SymBookNode the symmetry mapping child space -> SymBookNode space
  // So we need to use invert(C)

  BookNode* child = node->book->get(childHash);
  if(child == nullptr) {
    child = new BookNode(childHash, node->book, hist.presumedNextMovePla, symmetriesOfChild);
    bool suc = node->book->add(childHash,child);
    assert(suc);
    (void)suc;
    childIsTransposing = false;
  }
  else {
    childIsTransposing = true;
  }
  child->parents.push_back(std::make_pair(node->hash, bestLoc));

  BookMove newBookMove(
    bestLoc,
    SymmetryHelpers::compose(SymmetryHelpers::invert(bestSymmetry),symmetryOfNode,symmetryToAlignToChild),
    childHash,
    rawPolicy
  );
  node->moves[bestLoc] = newBookMove;
  return SymBookNode(child,SymmetryHelpers::invert(symmetryToAlignToChild));
}

bool SymBookNode::getBoardHistoryReachingHere(BoardHistory& ret, vector<Loc>& moveHistoryRet) {
  vector<double> winLossRet;
  return getBoardHistoryReachingHere(ret,moveHistoryRet,winLossRet);
}
bool ConstSymBookNode::getBoardHistoryReachingHere(BoardHistory& ret, vector<Loc>& moveHistoryRet) {
  vector<double> winLossRet;
  return getBoardHistoryReachingHere(ret,moveHistoryRet,winLossRet);
}
bool SymBookNode::getBoardHistoryReachingHere(BoardHistory& ret, vector<Loc>& moveHistoryRet, vector<double>& winlossRet) {
  return ConstSymBookNode(*this).getBoardHistoryReachingHere(ret,moveHistoryRet,winlossRet);
}
bool ConstSymBookNode::getBoardHistoryReachingHere(BoardHistory& ret, vector<Loc>& moveHistoryRet, vector<double>& winlossRet) {
  assert(node != nullptr);
  const Book* book = node->book;
  vector<const BookNode*> pathFromRoot;
  vector<Loc> movesFromRoot;
  bool suc = node->book->reverseDepthFirstSearchWithMoves(
    node,
    true, // Prefer low cost parents
    [&book,&pathFromRoot,&movesFromRoot](const vector<const BookNode*>& stack, const vector<Loc>& moveStack) {
      if(stack.back() == book->root) {
        pathFromRoot = vector<const BookNode*>(stack.rbegin(),stack.rend());
        movesFromRoot = vector<Loc>(moveStack.rbegin(),moveStack.rend());
        return Book::DFSAction::abort;
      }
      return Book::DFSAction::recurse;
    }
  );
  (void)suc;
  assert(suc);
  assert(pathFromRoot.size() >= 1);
  assert(movesFromRoot.size() == pathFromRoot.size());

  winlossRet.clear();
  for(const BookNode* pathNode: pathFromRoot)
    winlossRet.push_back(pathNode->recursiveValues.winLossValue);
  winlossRet.push_back(node->recursiveValues.winLossValue);

  // Find the total composed symmetry that we will have to apply as we walk down.
  int symmetryAcc = 0;
  for(size_t i = 0; i < pathFromRoot.size()-1; i++) {
    auto iter = pathFromRoot[i]->moves.find(movesFromRoot[i]);
    assert(iter != pathFromRoot[i]->moves.end());
    symmetryAcc = SymmetryHelpers::compose(symmetryAcc,iter->second.symmetryToAlign);
  }
  // At the end, we'll need this symmetry to transform it into SymBookNode space.
  symmetryAcc = SymmetryHelpers::compose(symmetryAcc,symmetryOfNode);
  // Additionally we need to apply the symmetry that maps the book's initial hist into its root node space.
  symmetryAcc = SymmetryHelpers::compose(node->book->initialSymmetry,symmetryAcc);

  // symmetryAcc is the map from initialSpace to desired final histSpace
  // Start out with a board permuted with this symmetry so that the initial stones all end up in the right orientation
  BoardHistory hist = node->book->getInitialHist(symmetryAcc);
  Board board = hist.getRecentBoard(0);
  moveHistoryRet.clear();

  // Invariant: during loop iteration i, symmetryPathNodeToHist is the transform from pathFromRoot[i] space to hist space.
  // symmetryAcc is the map from initialSpace -> histSpace
  // initialSymmetry is the map from initialSpace -> rootSpace.
  // We need to start with the map from rootSpace -> histSpace
  int symmetryPathNodeToHist = SymmetryHelpers::compose(SymmetryHelpers::invert(node->book->initialSymmetry),symmetryAcc);
  for(size_t i = 0; i < pathFromRoot.size()-1; i++) {
    auto iter = pathFromRoot[i]->moves.find(movesFromRoot[i]);
    // Convert move into space of hist and play it
    Loc symMove = SymmetryHelpers::getSymLoc(movesFromRoot[i], node->book->initialBoard, symmetryPathNodeToHist);
    moveHistoryRet.push_back(symMove);

    // Use tolerant rules so that if something weird happens regarding superko in cycles, we just plow through.
    if(!hist.isLegalTolerant(board, symMove, pathFromRoot[i]->pla)) {
      // Something is very wrong, probably a corrupted book data structure.
      return false;
    }
    hist.makeBoardMoveAssumeLegal(board, symMove, pathFromRoot[i]->pla, nullptr);

    // Update symmetryPathNodeToHist for the next loop.
    // symmetryPathNodeToHist is currently space[i] -> histSpace, and needs to become space[i+1] -> histSpace
    // symmetryToAlign[i] is space[i] -> space[i+1].
    symmetryPathNodeToHist = SymmetryHelpers::compose(SymmetryHelpers::invert(iter->second.symmetryToAlign), symmetryPathNodeToHist);
  }

  // Yay
  ret = hist;
  return true;
}

static double invSigmoid(double proportion) {
  if(proportion <= 0.0)
    return -std::numeric_limits<double>::infinity();
  double d = 1.0/proportion - 1.0;
  if(d <= 0.0)
    return std::numeric_limits<double>::infinity();
  return -log(d);
}
static double sigmoid(double x) {
  if(x >= 50.0)
    return 1.0;
  if(x <= -50.0)
    return 0.0;
  return 1.0 / (1.0 + exp(-x));
}

BookParams BookParams::loadFromCfg(ConfigParser& cfg, int64_t maxVisits, int64_t maxVisitsForLeaves) {
  BookParams cfgParams;
  cfgParams.errorFactor = cfg.getDouble("errorFactor",0.01,100.0);
  cfgParams.costPerMove = cfg.getDouble("costPerMove",0.0,1000000.0);
  cfgParams.costPerUCBWinLossLoss = cfg.getDouble("costPerUCBWinLossLoss",0.0,1000000.0);
  cfgParams.costPerUCBWinLossLossPow3 = cfg.getDouble("costPerUCBWinLossLossPow3",0.0,1000000.0);
  cfgParams.costPerUCBWinLossLossPow7 = cfg.getDouble("costPerUCBWinLossLossPow7",0.0,1000000.0);
  cfgParams.costPerUCBScoreLoss = cfg.getDouble("costPerUCBScoreLoss",0.0,1000000.0);
  cfgParams.costPerLogPolicy = cfg.getDouble("costPerLogPolicy",0.0,1000000.0);
  cfgParams.costPerMovesExpanded = cfg.getDouble("costPerMovesExpanded",0.0,1000000.0);
  cfgParams.costPerSquaredMovesExpanded = cfg.getDouble("costPerSquaredMovesExpanded",0.0,1000000.0);
  cfgParams.costWhenPassFavored = cfg.getDouble("costWhenPassFavored",0.0,1000000.0);
  cfgParams.bonusPerWinLossError = cfg.getDouble("bonusPerWinLossError",0.0,1000000.0);
  cfgParams.bonusPerScoreError = cfg.getDouble("bonusPerScoreError",0.0,1000000.0);
  cfgParams.bonusPerSharpScoreDiscrepancy = cfg.getDouble("bonusPerSharpScoreDiscrepancy",0.0,1000000.0);
  cfgParams.bonusPerExcessUnexpandedPolicy = cfg.getDouble("bonusPerExcessUnexpandedPolicy",0.0,1000000.0);
  cfgParams.bonusPerUnexpandedBestWinLoss = cfg.getDouble("bonusPerUnexpandedBestWinLoss",0.0,1000000.0);
  cfgParams.bonusForWLPV1 = cfg.contains("bonusForWLPV1") ? cfg.getDouble("bonusForWLPV1",0.0,1000000.0) : 0.0;
  cfgParams.bonusForWLPV2 = cfg.contains("bonusForWLPV2") ? cfg.getDouble("bonusForWLPV2",0.0,1000000.0) : 0.0;
  cfgParams.bonusForWLPVFinalProp = cfg.contains("bonusForWLPVFinalProp") ? cfg.getDouble("bonusForWLPVFinalProp",0.0,1.0) : 0.5;
  cfgParams.bonusForBiggestWLCost = cfg.contains("bonusForBiggestWLCost") ? cfg.getDouble("bonusForBiggestWLCost",0.0,1000000.0) : 0.0;
  cfgParams.bonusBehindInVisitsScale = cfg.contains("bonusBehindInVisitsScale") ? cfg.getDouble("bonusBehindInVisitsScale",0.0,1000000.0) : 0.0;
  cfgParams.scoreLossCap = cfg.getDouble("scoreLossCap",0.0,1000000.0);
  cfgParams.earlyBookCostReductionFactor = cfg.contains("earlyBookCostReductionFactor") ? cfg.getDouble("earlyBookCostReductionFactor",0.0,1.0) : 0.0;
  cfgParams.earlyBookCostReductionLambda = cfg.contains("earlyBookCostReductionLambda") ? cfg.getDouble("earlyBookCostReductionLambda",0.0,1.0) : 0.5;
  cfgParams.utilityPerScore = cfg.getDouble("utilityPerScore",0.0,1000000.0);
  cfgParams.policyBoostSoftUtilityScale = cfg.getDouble("policyBoostSoftUtilityScale",0.0,1000000.0);
  cfgParams.utilityPerPolicyForSorting = cfg.getDouble("utilityPerPolicyForSorting",0.0,1000000.0);
  cfgParams.adjustedVisitsWLScale = cfg.contains("adjustedVisitsWLScale") ? cfg.getDouble("adjustedVisitsWLScale",0.0,1000000.0) : 0.05;
  cfgParams.maxVisitsForReExpansion = cfg.contains("maxVisitsForReExpansion") ? cfg.getDouble("maxVisitsForReExpansion",0.0,1e50) : 0.0;
  cfgParams.visitsScale = cfg.contains("visitsScale") ? cfg.getDouble("visitsScale") : (maxVisits + 1) / 2;
  cfgParams.visitsScaleLeaves = cfg.contains("visitsScaleLeaves") ? cfg.getDouble("visitsScaleLeaves") : maxVisitsForLeaves;
  cfgParams.sharpScoreOutlierCap = cfg.getDouble("sharpScoreOutlierCap",0.0,1000000.0);
  return cfgParams;
}

void BookParams::randomizeParams(Rand& rand, double stdev) {
  errorFactor *= exp(0.5 * stdev * rand.nextGaussianTruncated(3.0));
  costPerMove *= exp(0.5 * stdev * rand.nextGaussianTruncated(3.0));
  costPerUCBWinLossLoss *= exp(stdev * rand.nextGaussianTruncated(3.0));
  costPerUCBWinLossLossPow3 *= exp(stdev * rand.nextGaussianTruncated(3.0));
  costPerUCBWinLossLossPow7 *= exp(stdev * rand.nextGaussianTruncated(3.0));
  costPerUCBScoreLoss *= exp(stdev * rand.nextGaussianTruncated(3.0));
  costPerLogPolicy *= exp(stdev * rand.nextGaussianTruncated(3.0));
  costPerMovesExpanded *= exp(stdev * rand.nextGaussianTruncated(3.0));
  costPerSquaredMovesExpanded *= exp(stdev * rand.nextGaussianTruncated(3.0));
  bonusPerWinLossError *= exp(0.5 * stdev * rand.nextGaussianTruncated(3.0));
  bonusPerScoreError *= exp(0.5 * stdev * rand.nextGaussianTruncated(3.0));
  bonusPerSharpScoreDiscrepancy *= exp(0.5 * stdev * rand.nextGaussianTruncated(3.0));
  bonusPerExcessUnexpandedPolicy *= exp(stdev * rand.nextGaussianTruncated(3.0));
  bonusPerUnexpandedBestWinLoss *= exp(stdev * rand.nextGaussianTruncated(3.0));
  bonusBehindInVisitsScale *= exp(0.5 * stdev * rand.nextGaussianTruncated(3.0));

  bonusForWLPV1 = sigmoid(invSigmoid(bonusForWLPV1) + 0.5*stdev*rand.nextGaussianTruncated(3.0));
  bonusForWLPV2 *= sigmoid(invSigmoid(bonusForWLPV2) + 0.5*stdev*rand.nextGaussianTruncated(3.0));
  bonusForWLPVFinalProp *= sigmoid(invSigmoid(bonusForWLPVFinalProp) + stdev*rand.nextGaussianTruncated(3.0));
  bonusForBiggestWLCost *= sigmoid(invSigmoid(bonusForBiggestWLCost) + 0.5*stdev*rand.nextGaussianTruncated(3.0));
  earlyBookCostReductionFactor *= sigmoid(invSigmoid(earlyBookCostReductionFactor) + 0.5*stdev*rand.nextGaussianTruncated(3.0));
  earlyBookCostReductionLambda *= sigmoid(invSigmoid(earlyBookCostReductionLambda) + 0.5*stdev*rand.nextGaussianTruncated(3.0));
}


Book::Book(
  int bversion,
  const Board& b,
  Rules r,
  Player p,
  int rb,
  BookParams bp
) : bookVersion(bversion),
    initialBoard(b),
    initialRules(r),
    initialPla(p),
    repBound(rb),
    params(bp),
    initialSymmetry(0),
    root(nullptr),
    nodes(),
    nodeIdxMapsByHash(nullptr)
{
  nodeIdxMapsByHash = new std::map<BookHash,int64_t>[NUM_HASH_BUCKETS];

  BookHash rootHash;
  int symmetryToAlign;
  vector<int> rootSymmetries;

  int initialEncorePhase = 0;
  BoardHistory initialHist(initialBoard, initialPla, initialRules, initialEncorePhase);
  BookHash::getHashAndSymmetry(initialHist, repBound, rootHash, symmetryToAlign, rootSymmetries, bookVersion);

  initialSymmetry = symmetryToAlign;
  root = new BookNode(rootHash, this, initialPla, rootSymmetries);
  nodeIdxMapsByHash[rootHash.stateHash.hash0 % NUM_HASH_BUCKETS][rootHash] = (int64_t)nodes.size();
  nodes.push_back(root);
}

Book::~Book() {
  for(BookNode* node: nodes)
    delete node;
  delete[] nodeIdxMapsByHash;
}

BoardHistory Book::getInitialHist() const {
  return getInitialHist(0);
}
BoardHistory Book::getInitialHist(int symmetry) const {
  int initialEncorePhase = 0;
  return BoardHistory(SymmetryHelpers::getSymBoard(initialBoard,symmetry), initialPla, initialRules, initialEncorePhase);
}

size_t Book::size() const {
  return nodes.size();
}

BookParams Book::getParams() const { return params; }
void Book::setParams(const BookParams& p) { params = p; }
std::map<BookHash,double> Book::getBonusByHash() const { return bonusByHash; }
void Book::setBonusByHash(const std::map<BookHash,double>& d) { bonusByHash = d; }
std::map<BookHash,double> Book::getExpandBonusByHash() const { return expandBonusByHash; }
void Book::setExpandBonusByHash(const std::map<BookHash,double>& d) { expandBonusByHash = d; }
std::map<BookHash,double> Book::getVisitsRequiredByHash() const { return visitsRequiredByHash; }
void Book::setVisitsRequiredByHash(const std::map<BookHash,double>& d) { visitsRequiredByHash = d; }
std::map<BookHash,int> Book::getBranchRequiredByHash() const { return branchRequiredByHash; }
void Book::setBranchRequiredByHash(const std::map<BookHash,int>& d) { branchRequiredByHash = d; }


SymBookNode Book::getRoot() {
  // Invert because SymBookNode needs map from node space -> initial space, but initialSymmetry is map initial space -> node space.
  return SymBookNode(root, SymmetryHelpers::invert(initialSymmetry));
}
ConstSymBookNode Book::getRoot() const {
  return ConstSymBookNode(root, SymmetryHelpers::invert(initialSymmetry));
}

SymBookNode Book::get(const BoardHistory& hist) {
  SymBookNode node = getRoot();
  for(Move move: hist.moveHistory) {
    node = node.follow(move.loc);
    if(node.isNull())
      return node;
  }
  return node;
}
ConstSymBookNode Book::get(const BoardHistory& hist) const {
  ConstSymBookNode node = getRoot();
  for(Move move: hist.moveHistory) {
    node = node.follow(move.loc);
    if(node.isNull())
      return node;
  }
  return node;
}

void Book::recompute(const vector<SymBookNode>& newAndChangedNodes) {
  // Walk up from all changed nodes and mark all parents dirty recursively.
  std::set<BookHash> dirtyNodes;
  std::function<DFSAction(BookNode*)> markDirty = [&dirtyNodes](BookNode* node) {
    // cout << "Mark dirty " << node->hash << " " << node << endl;
    if(contains(dirtyNodes, node->hash))
      return DFSAction::skip;
    dirtyNodes.insert(node->hash);
    return DFSAction::recurse;
  };
  for(SymBookNode node: newAndChangedNodes) {
    reverseDepthFirstSearchWithPostF(node.node, markDirty, std::function<void(BookNode*)>(nullptr));
  }

  // Walk down through all dirty nodes recomputing values
  bool allDirty = false;
  iterateDirtyNodesPostOrder(
    dirtyNodes,
    allDirty,
    [this](BookNode* node) {
      recomputeNodeValues(node);
    }
  );

  // Walk down through entire book, recomputing costs.
  iterateEntireBookPreOrder(
    [this](BookNode* node) {
      recomputeNodeCost(node);
    }
  );
}

void Book::recomputeEverything() {
  // Walk down through all nodes recomputing values
  bool allDirty = true;
  iterateDirtyNodesPostOrder(
    std::set<BookHash>(),
    allDirty,
    [this](BookNode* node) {
      recomputeNodeValues(node);
    }
  );

  // Walk down through entire book, recomputing costs.
  iterateEntireBookPreOrder(
    [this](BookNode* node) {
      recomputeNodeCost(node);
    }
  );
}

vector<SymBookNode> Book::getNextNToExpand(int n) {
  vector<BookNode*> toExpand(n);
  auto end = std::partial_sort_copy(
    nodes.begin(),
    nodes.end(),
    toExpand.begin(),
    toExpand.end(),
    [](BookNode* n0, BookNode* n1) {
      return n0->minCostFromRoot + n0->thisNodeExpansionCost < n1->minCostFromRoot + n1->thisNodeExpansionCost;
    }
  );
  toExpand.resize(end-toExpand.begin());

  vector<SymBookNode> ret;
  for(BookNode* node: toExpand) {
    if(node->canExpand)
      ret.push_back(SymBookNode(node,0));
  }
  return ret;
}

vector<SymBookNode> Book::getAllLeaves(double minVisits) {
  vector<SymBookNode> ret;
  for(BookNode* node: nodes) {
    if(node->recursiveValues.visits >= minVisits) {
      bool allChildrenLess = true;
      for(auto iter = node->moves.begin(); iter != node->moves.end(); ++iter) {
        const BookNode* child = get(iter->second.hash);
        if(child->recursiveValues.visits >= minVisits) {
          allChildrenLess = false;
          break;
        }
      }
      if(allChildrenLess) {
        ret.push_back(SymBookNode(node,0));
      }
    }
  }
  return ret;
}

std::vector<SymBookNode> Book::getAllNodes() {
  vector<SymBookNode> ret;
  for(BookNode* node: nodes) {
    ret.push_back(SymBookNode(node,0));
  }
  return ret;
}

int64_t Book::getIdx(BookHash hash) const {
  std::map<BookHash,int64_t>& nodeIdxMap = nodeIdxMapsByHash[hash.stateHash.hash0 % NUM_HASH_BUCKETS];
  auto iter = nodeIdxMap.find(hash);
  if(iter == nodeIdxMap.end())
    throw StringError("Node idx not found for hash");
  return iter->second;
}
BookNode* Book::get(BookHash hash) {
  std::map<BookHash,int64_t>& nodeIdxMap = nodeIdxMapsByHash[hash.stateHash.hash0 % NUM_HASH_BUCKETS];
  auto iter = nodeIdxMap.find(hash);
  if(iter == nodeIdxMap.end())
    return nullptr;
  return nodes[iter->second];
}
const BookNode* Book::get(BookHash hash) const {
  const std::map<BookHash,int64_t>& nodeIdxMap = nodeIdxMapsByHash[hash.stateHash.hash0 % NUM_HASH_BUCKETS];
  auto iter = nodeIdxMap.find(hash);
  if(iter == nodeIdxMap.end())
    return nullptr;
  return nodes[iter->second];
}

SymBookNode Book::getByHash(BookHash hash) {
  BookNode* node = get(hash);
  if(node == nullptr)
    return SymBookNode(nullptr);
  return SymBookNode(node,0);
}
ConstSymBookNode Book::getByHash(BookHash hash) const {
  const BookNode* node = get(hash);
  if(node == nullptr)
    return ConstSymBookNode(nullptr);
  return ConstSymBookNode(node,0);
}


bool Book::add(BookHash hash, BookNode* node) {
  std::map<BookHash,int64_t>& nodeIdxMap = nodeIdxMapsByHash[hash.stateHash.hash0 % NUM_HASH_BUCKETS];
  auto iter = nodeIdxMap.find(hash);
  if(iter != nodeIdxMap.end())
    return false;
  nodeIdxMap[hash] = (int64_t)nodes.size();
  nodes.push_back(node);
  return true;
}


// Walk the entire reverse-subtree starting at node, calling f with the current (stack,moveStack) at each step.
// stack is the stack of moves searched. moveStack[i] is the move to make from stack[i] to reach stack[i-1], in stack[i]'s symmetry alignment.
// moveStack[0] is always NULL_LOC.
// Will never walk to any node more than once, if there are backward paths that reverse-transpose or have cycles.
// Stop searching and immediately return true if f ever returns abort.
// Returns false at the end of the whole search if f never aborts.
// Also this particular function supports trying to follow low-cost parents first.
bool Book::reverseDepthFirstSearchWithMoves(
  const BookNode* initialNode,
  bool preferLowCostParents,
  const std::function<Book::DFSAction(const vector<const BookNode*>&, const vector<Loc>&)>& f
) const {
  vector<const BookNode*> stack;
  vector<Loc> moveStack;
  vector<int64_t> nextParentIdxToTry;
  std::set<BookHash> visitedHashes;
  stack.push_back(initialNode);
  Loc nullLoc = Board::NULL_LOC; // Workaround for c++14 wonkiness fixed in c++17
  moveStack.push_back(nullLoc);
  if(preferLowCostParents)
    nextParentIdxToTry.push_back(-1); // -1 encodes "try the best parent" first.
  else
    nextParentIdxToTry.push_back(0);
  visitedHashes.insert(initialNode->hash);

  while(true) {
    // Handle new found node
    DFSAction action = f(stack,moveStack);
    if(action == DFSAction::abort)
      return true;
    else if(action == DFSAction::skip) {
      nextParentIdxToTry.back() = std::numeric_limits<int64_t>::max();
    }

    // Attempt to find the next node
    while(true) {
      // Try walk to next parent
      const BookNode* node = stack.back();
      int64_t nextParentIdx = nextParentIdxToTry.back();
      // -1 encodes "try the best parent" first.
      if(nextParentIdx == -1)
        nextParentIdx = node->bestParentIdx;
      if(nextParentIdx < node->parents.size()) {
        BookHash nextParentHash = node->parents[nextParentIdx].first;
        Loc nextParentLoc = node->parents[nextParentIdx].second;
        nextParentIdxToTry.back() += 1;
        if(!contains(visitedHashes, nextParentHash)) {
          const BookNode* nextParent = get(nextParentHash);
          stack.push_back(nextParent);
          moveStack.push_back(nextParentLoc);
          if(preferLowCostParents)
            nextParentIdxToTry.push_back(-1); // -1 encodes "try the best parent" first.
          else
            nextParentIdxToTry.push_back(0);
          visitedHashes.insert(nextParentHash);
          // Found next node, break out of attempt to find the next node.
          break;
        }
        else {
          // Parent already visited, continue at same node to try next parent.
          continue;
        }
      }
      else {
        // Exhausted all parents at this node, go up one level.
        stack.pop_back();
        moveStack.pop_back();
        nextParentIdxToTry.pop_back();
        // If we go up a level and there's nothing, we're done.
        if(stack.size() <= 0)
          return false;
      }
    }
  }
}

// Walk the entire reverse-subtree starting at node, calling f with the current node at each step.
// Will never walk to any node more than once, if there are backward paths that reverse-transpose or have cycles.
// Stop searching and immediately return true if f ever returns abort.
// Returns false at the end of the whole search if f never aborts.
// If postF is povided, calls postF after returning from a node.
bool Book::reverseDepthFirstSearchWithPostF(
  BookNode* initialNode,
  const std::function<Book::DFSAction(BookNode*)>& f,
  const std::function<void(BookNode*)>& postF
) {
  vector<BookNode*> stack;
  vector<size_t> nextParentIdxToTry;
  std::set<BookHash> visitedHashes;
  stack.push_back(initialNode);
  nextParentIdxToTry.push_back(0);
  visitedHashes.insert(initialNode->hash);

  while(true) {
    // Handle new found node
    DFSAction action = f(stack.back());
    if(action == DFSAction::abort)
      return true;
    else if(action == DFSAction::skip) {
      nextParentIdxToTry.back() = std::numeric_limits<size_t>::max();
    }

    // Attempt to find the next node
    while(true) {
      // Try walk to next parent
      BookNode* node = stack.back();
      size_t nextParentIdx = nextParentIdxToTry.back();
      if(nextParentIdx < node->parents.size()) {
        BookHash nextParentHash = node->parents[nextParentIdx].first;
        nextParentIdxToTry.back() += 1;
        if(!contains(visitedHashes, nextParentHash)) {
          BookNode* nextParent = get(nextParentHash);
          stack.push_back(nextParent);
          nextParentIdxToTry.push_back(0);
          visitedHashes.insert(nextParentHash);
          // Found next node, break out of attempt to find the next node.
          break;
        }
        else {
          // Parent already visited, continue at same node to try next parent.
          continue;
        }
      }
      else {
        if(postF)
          postF(stack.back());

        // Exhausted all parents at this node, go up one level.
        stack.pop_back();
        nextParentIdxToTry.pop_back();
        // If we go up a level and there's nothing, we're done.
        if(stack.size() <= 0)
          return false;
      }
    }
  }
}

// Precondition: dirtyNodes has the property that if a node n is in dirtyNodes, all parents of n are in dirtyNodes.
// Calls f on every node in dirtyNodes in an order where all children of any node are called before a node.
// Except if somehow the book has mangaged to be cyclic, in which case the cycle is broken arbitrarily.
// If allDirty is true, ignores dirtyNodes and treats all nodes in the entiry book as dirty.
void Book::iterateDirtyNodesPostOrder(
  const std::set<BookHash>& dirtyNodes,
  bool allDirty,
  const std::function<void(BookNode* node)>& f
) {
  vector<BookNode*> stack;
  vector<std::map<Loc,BookMove>::iterator> nextChildToTry;
  std::set<BookHash> visitedHashes;

  if(!allDirty && dirtyNodes.size() <= 0)
    return;
  assert(allDirty || contains(dirtyNodes, root->hash));

  stack.push_back(root);
  nextChildToTry.push_back(root->moves.begin());
  visitedHashes.insert(root->hash);

  while(true) {
    // Attempt to find the next node
    while(true) {
      // Try walk to next parent
      BookNode* node = stack.back();
      std::map<Loc,BookMove>::iterator iter = nextChildToTry.back();

      if(iter != node->moves.end()) {
        BookHash nextChildHash = iter->second.hash;
        ++nextChildToTry.back();
        if(!contains(visitedHashes,nextChildHash) && (allDirty || contains(dirtyNodes,nextChildHash))) {
          BookNode* nextChild = get(nextChildHash);
          stack.push_back(nextChild);
          nextChildToTry.push_back(nextChild->moves.begin());
          visitedHashes.insert(nextChildHash);
          // Found next node, break out of attempt to find the next node.
          break;
        }
        else {
          // Child already visited, continue at same node to try next child.
          continue;
        }
      }
      else {
        // Exhausted all childs at this node. Call f since postorder is done.
        f(node);

        //Then pop up a level.
        stack.pop_back();
        nextChildToTry.pop_back();
        // If we go up a level and there's nothing, we're done.
        if(stack.size() <= 0)
          return;
      }
    }
  }
}

void Book::iterateEntireBookPreOrder(
  const std::function<void(BookNode*)>& f
) {
  std::set<BookHash> visitedHashes;
  for(BookNode* initialNode: nodes) {
    if(contains(visitedHashes, initialNode->hash))
      continue;
    reverseDepthFirstSearchWithPostF(
      initialNode,
      [&visitedHashes](BookNode* node) {
        if(contains(visitedHashes, node->hash))
          return DFSAction::skip;
        return DFSAction::recurse;
      },
      [&visitedHashes,&f](BookNode* node) {
        if(contains(visitedHashes, node->hash))
          return;
        visitedHashes.insert(node->hash);
        f(node);
      }
    );
  }
}

void Book::recomputeAdjustedVisits(
  BookNode* node,
  double notInBookVisits,
  double notInBookMaxRawPolicy,
  double notInBookWL,
  double notInBookScoreMean,
  double notInBookSharpScoreMean,
  double notInBookScoreLCB,
  double notInBookScoreUCB
) {
  std::vector<int> sortIdxBuf;
  std::vector<double> sortValuesBuf;
  std::vector<double> childAdjustedVisitsBuf;

  const double plaFactor = node->pla == P_WHITE ? 1.0 : -1.0;

  // Compute the values for sorting all children, AND the not in book values.
  int numItems;
  {
    int i = 0;
    for(auto iter = node->moves.begin(); iter != node->moves.end(); ++iter) {
      const BookNode* child = get(iter->second.hash);
      const RecursiveBookValues& vals = child->recursiveValues;
      sortIdxBuf.push_back(i);
      sortValuesBuf.push_back(
        getSortingValue(plaFactor,vals.winLossValue,vals.scoreMean,vals.sharpScoreMean,vals.scoreLCB,vals.scoreUCB,iter->second.rawPolicy)
      );
      childAdjustedVisitsBuf.push_back(vals.adjustedVisits);
      i += 1;
    }
    sortIdxBuf.push_back(i);
    sortValuesBuf.push_back(
      getSortingValue(plaFactor,notInBookWL,notInBookScoreMean,notInBookSharpScoreMean,notInBookScoreLCB,notInBookScoreUCB,notInBookMaxRawPolicy)
    );
    childAdjustedVisitsBuf.push_back(notInBookVisits);
    i += 1;
    numItems = i;
  }

  // Sort them from worst to best
  std::sort(
    sortIdxBuf.begin(),sortIdxBuf.end(),
    [&](const int& idx0,
        const int& idx1) {
      return sortValuesBuf[idx0] < sortValuesBuf[idx1];
    }
  );

  // Compute an exponentially weighted moving average of visits, where "time" is measured by the number of
  // units of params.adjustedVisitsWLScale.
  // This blurs a little so that we ignore minor fluctuations if visits are not in ascending order.
  // EWMA is in geometric average space.
  double wsum = 0.0;
  double wvsum = 0.0;
  double prevSortingValue = -1e100;
  std::vector<double> caps;
  for(int i = 0; i<numItems; i++) {
    double timeElapsed = sortValuesBuf[sortIdxBuf[i]] - prevSortingValue;
    prevSortingValue = sortValuesBuf[sortIdxBuf[i]];
    double factor = exp(-timeElapsed);
    wsum *= factor;
    wvsum *= factor;
    wsum += 1.0;
    wvsum += log(1.0 + params.visitsScale*0.05 + childAdjustedVisitsBuf[sortIdxBuf[i]]);
    double ewmaVisits = exp(wvsum / wsum);
    // Count the larger of the actual number of visits and the ewmaVisits
    // Cap all visits of moves worse than this at this amount.
    double adjustedVisitsCap = std::max(ewmaVisits, childAdjustedVisitsBuf[sortIdxBuf[i]]);
    caps.push_back(adjustedVisitsCap);
  }

  // Okay, now iterate through in reverse order and sum up capped visits, allowing for params.visitsScale
  // and other tolerance in the visits so that while visits are small and chunky or the differences aren't
  // relatively that large we don't penalize distributions being weird
  double adjustedVisits = 0;
  double lowestCapSoFar = 1e100;
  for(int i = numItems-1; i>= 0; i--) {
    lowestCapSoFar = std::min(caps[i],lowestCapSoFar);
    adjustedVisits += std::min(4.0 * lowestCapSoFar + params.visitsScale, childAdjustedVisitsBuf[sortIdxBuf[i]]);
  }
  node->recursiveValues.adjustedVisits = adjustedVisits;
}

void Book::recomputeNodeValues(BookNode* node) {
  double winLossValue;
  double scoreMean;
  double sharpScoreMean;
  double winLossLCB;
  double scoreLCB;
  double scoreFinalLCB;
  double winLossUCB;
  double scoreUCB;
  double scoreFinalUCB;
  double weight = 0.0;
  double visits = 0.0;

  {
    BookValues& values = node->thisValuesNotInBook;
    double scoreError = values.getAdjustedScoreError(node->book->initialRules);
    double winLossError = values.getAdjustedWinLossError(node->book->initialRules);
    winLossValue = values.winLossValue;
    scoreMean = values.scoreMean;
    sharpScoreMean = values.sharpScoreMeanRaw;
    winLossLCB = values.winLossValue - params.errorFactor * winLossError;
    scoreLCB = values.scoreMean - params.errorFactor * scoreError;
    scoreFinalLCB = values.scoreMean - params.errorFactor * values.scoreStdev;
    winLossUCB = values.winLossValue + params.errorFactor * winLossError;
    scoreUCB = values.scoreMean + params.errorFactor * scoreError;
    scoreFinalUCB = values.scoreMean + params.errorFactor * values.scoreStdev;
    weight += values.weight;
    visits += values.visits;

    // A quick hack to limit the issue of outliers from sharpScore, and adjust the LCB/UCB to reflect the uncertainty
    // Skip scoreUCB/scoreLCB adjustment if there isn't any error at all, where the net doesn't support it.
    if(scoreError > 0) {
      if(sharpScoreMean > scoreUCB)
        scoreUCB = sharpScoreMean;
      if(sharpScoreMean < scoreLCB)
        scoreLCB = sharpScoreMean;
    }
    if(sharpScoreMean > scoreMean + params.sharpScoreOutlierCap)
      sharpScoreMean = scoreMean + params.sharpScoreOutlierCap;
    if(sharpScoreMean < scoreMean - params.sharpScoreOutlierCap)
      sharpScoreMean = scoreMean - params.sharpScoreOutlierCap;

    values.sharpScoreMeanClamped = sharpScoreMean;
  }

  // Recompute at this point when the all the values are the not-in-book values.
  recomputeAdjustedVisits(
    node,
    visits,
    node->thisValuesNotInBook.maxPolicy,
    winLossValue,
    scoreMean,
    sharpScoreMean,
    scoreLCB,
    scoreUCB
  );

  for(auto iter = node->moves.begin(); iter != node->moves.end(); ++iter) {
    const BookNode* child = get(iter->second.hash);
    // cout << "pulling values from child " << child << " hash " << child->hash << endl;
    const RecursiveBookValues& values = child->recursiveValues;
    if(node->pla == P_WHITE) {
      winLossValue = std::max(winLossValue, values.winLossValue);
      scoreMean = std::max(scoreMean, values.scoreMean);
      sharpScoreMean = std::max(sharpScoreMean, values.sharpScoreMean);
      winLossLCB = std::max(winLossLCB, values.winLossLCB);
      scoreLCB = std::max(scoreLCB, values.scoreLCB);
      scoreFinalLCB = std::max(scoreFinalLCB, values.scoreFinalLCB);
      winLossUCB = std::max(winLossUCB, values.winLossUCB);
      scoreUCB = std::max(scoreUCB, values.scoreUCB);
      scoreFinalUCB = std::max(scoreFinalUCB, values.scoreFinalUCB);
      weight += values.weight;
      visits += values.visits;
    }
    else {
      winLossValue = std::min(winLossValue, values.winLossValue);
      scoreMean = std::min(scoreMean, values.scoreMean);
      sharpScoreMean = std::min(sharpScoreMean, values.sharpScoreMean);
      winLossLCB = std::min(winLossLCB, values.winLossLCB);
      scoreLCB = std::min(scoreLCB, values.scoreLCB);
      scoreFinalLCB = std::min(scoreFinalLCB, values.scoreFinalLCB);
      winLossUCB = std::min(winLossUCB, values.winLossUCB);
      scoreUCB = std::min(scoreUCB, values.scoreUCB);
      scoreFinalUCB = std::min(scoreFinalUCB, values.scoreFinalUCB);
      weight += values.weight;
      visits += values.visits;
    }
  }

  RecursiveBookValues& values = node->recursiveValues;
  values.winLossValue = winLossValue;
  values.scoreMean = scoreMean;
  values.sharpScoreMean = sharpScoreMean;
  values.winLossLCB = winLossLCB;
  values.scoreLCB = scoreLCB;
  values.scoreFinalLCB = scoreFinalLCB;
  values.winLossUCB = winLossUCB;
  values.scoreUCB = scoreUCB;
  values.scoreFinalUCB = scoreFinalUCB;
  values.weight = weight;
  values.visits = visits;

  // cout << "Setting " << node->hash << " values" << endl;
  // cout << "Values " << values.winLossLCB << " " << values.winLossValue << " " << values.winLossUCB << endl;
  // cout << "Score " << values.scoreLCB << " " << values.scoreMean << " " << values.scoreUCB << endl;
}

double Book::getUtility(const RecursiveBookValues& values) const {
  return values.winLossValue + values.scoreMean * params.utilityPerScore;
}

void Book::recomputeNodeCost(BookNode* node) {
  // Update this node's minCostFromRoot based on cost for moves from parents.
  if(node == root) {
    node->minDepthFromRoot = 0;
    node->minCostFromRoot = 0.0;
    node->minCostFromRootWLPV = 0.0;
    node->biggestWLCostFromRoot = 0.0;
  }
  else {
    // cout << "Recomputing cost " << node->hash << endl;
    int minDepth = 0x3FFFFFFF;
    double minCost = 1e100;
    double minCostWLPV = 1e100;
    double bestBiggestWLCostFromRoot = 1e100;
    size_t bestParentIdx = 0;
    for(size_t parentIdx = 0; parentIdx<node->parents.size(); parentIdx++) {
      std::pair<BookHash,Loc>& parentInfo = node->parents[parentIdx];
      const BookNode* parent = get(parentInfo.first);
      auto parentLocAndBookMove = parent->moves.find(parentInfo.second);
      assert(parentLocAndBookMove != parent->moves.end());
      int depth = parent->minDepthFromRoot + 1;
      double cost = parentLocAndBookMove->second.costFromRoot;
      double biggestWLCostFromRoot = parentLocAndBookMove->second.biggestWLCostFromRoot;
      if(cost < minCost) {
        minCost = cost;
        bestBiggestWLCostFromRoot = biggestWLCostFromRoot;
        bestParentIdx = parentIdx;
      }
      if(parentLocAndBookMove->second.isWLPV) {
        if(parent->minCostFromRootWLPV < minCostWLPV)
          minCostWLPV = parent->minCostFromRootWLPV;
      }
      if(depth < minDepth)
        minDepth = depth;
    }
    node->minDepthFromRoot = minDepth;
    node->minCostFromRoot = minCost;
    node->minCostFromRootWLPV = minCostWLPV;
    node->biggestWLCostFromRoot = bestBiggestWLCostFromRoot;
    node->bestParentIdx = (int64_t)bestParentIdx;
  }

  // cout << "-----------------------------------------------------------------------" << endl;
  // cout << "Initial min cost from root " << node->minCostFromRoot << endl;

  // Apply user-specified bonuses
  if(contains(bonusByHash, node->hash)) {
    double bonus = bonusByHash[node->hash];
    node->minCostFromRoot -= bonus;

    // cout << "Applying user bonus " << bonus << " cost is now " << node->minCostFromRoot << endl;
  }

  if(contains(visitsRequiredByHash, node->hash)) {
    double visitsRequired = visitsRequiredByHash[node->hash];
    if(node->recursiveValues.visits < visitsRequired ||
       node->recursiveValues.adjustedVisits < 0.5 * visitsRequired / std::max(1.0, pow(visitsRequired / params.visitsScale, 0.1))
    ) {
      node->minCostFromRoot -= 500.0;
    }
  }

  if(node->minCostFromRoot < node->minCostFromRootWLPV)
    node->minCostFromRootWLPV = node->minCostFromRoot;

  double bestWinLossThisPerspective = -1e100;
  Loc bestWinLossMove = Board::NULL_LOC;
  const BookNode* bestWinLossChild = NULL;
  // Find the winloss PV for this node
  {
    for(auto& locAndBookMove: node->moves) {
      locAndBookMove.second.isWLPV = false;
      const BookNode* child = get(locAndBookMove.second.hash);
      double winLossThisPerspective = (node->pla == P_WHITE ? child->recursiveValues.winLossValue : -child->recursiveValues.winLossValue);
      if(winLossThisPerspective > bestWinLossThisPerspective) {
        bestWinLossThisPerspective = winLossThisPerspective;
        bestWinLossMove = locAndBookMove.first;
        bestWinLossChild = child;
      }
    }
    {
      node->expansionIsWLPV = false;
      double winLossThisPerspective = (node->pla == P_WHITE ? node->thisValuesNotInBook.winLossValue : -node->thisValuesNotInBook.winLossValue);
      if(winLossThisPerspective > bestWinLossThisPerspective) {
        bestWinLossThisPerspective = winLossThisPerspective;
        bestWinLossMove = Board::NULL_LOC;
        bestWinLossChild = NULL;
      }
    }
    if(bestWinLossMove == Board::NULL_LOC)
      node->expansionIsWLPV = true;
    else
      node->moves[bestWinLossMove].isWLPV = true;
  }
  double bestWinLoss = (node->pla == P_WHITE ? bestWinLossThisPerspective : -bestWinLossThisPerspective);


  // Look at other children whose policy is higher, and if this is better than those by a lot
  // softly boost the policy of this move.
  auto boostLogRawPolicy = [&](double logRawPolicy, double childUtility, double rawPolicy) {
    double boostedLogRawPolicy = logRawPolicy;
    for(auto& otherLocAndBookMove: node->moves) {
      if(otherLocAndBookMove.second.rawPolicy <= rawPolicy)
        continue;
      const BookNode* otherChild = get(otherLocAndBookMove.second.hash);
      double otherChildUtility = getUtility(otherChild->recursiveValues);
      double gainOverOtherChild =
        (node->pla == P_WHITE) ?
        childUtility - otherChildUtility :
        otherChildUtility - childUtility;
      if(gainOverOtherChild <= 0)
        continue;
      double policyBoostFactor = 2.0/(1.0 + exp(-gainOverOtherChild / params.policyBoostSoftUtilityScale)) - 1.0;
      // Explicit shift to make boost factor count a little bit immediately if at all better
      policyBoostFactor = 0.1 + 0.9 * policyBoostFactor;
      double otherLogRawPolicy = log(otherLocAndBookMove.second.rawPolicy + 1e-100);
      double p = logRawPolicy + policyBoostFactor * (otherLogRawPolicy - logRawPolicy);
      if(p > boostedLogRawPolicy)
        boostedLogRawPolicy = p;
      // cout << "Boosting policy " << logRawPolicy << " " << otherLogRawPolicy << " " << p << " " << gainOverOtherChild << endl;
    }
    return boostedLogRawPolicy;
  };

  // Figure out whether pass is the favored move
  double passPolicy = 0.0;
  double passUtility = node->pla == P_WHITE ? -1e100 : 1e100;
  const Loc passLoc = Board::PASS_LOC;
  if(node->moves.find(passLoc) != node->moves.end()) {
    passPolicy = node->moves[passLoc].rawPolicy;
    passUtility = getUtility(get(node->moves[passLoc].hash)->recursiveValues);
  }

  // Update cost for moves for children to reference.
  double smallestCostFromUCB = 1e100;
  for(auto& locAndBookMove: node->moves) {
    const BookNode* child = get(locAndBookMove.second.hash);
    double ucbWinLossLoss =
      (node->pla == P_WHITE) ?
      node->recursiveValues.winLossUCB - child->recursiveValues.winLossUCB :
      child->recursiveValues.winLossLCB - node->recursiveValues.winLossLCB;
    double ucbWinLossLossPow3 =
      (node->pla == P_WHITE) ?
      pow3(node->recursiveValues.winLossUCB) - pow3(child->recursiveValues.winLossUCB) :
      pow3(child->recursiveValues.winLossLCB) - pow3(node->recursiveValues.winLossLCB);
    double ucbWinLossLossPow7 =
      (node->pla == P_WHITE) ?
      pow7(node->recursiveValues.winLossUCB) - pow7(child->recursiveValues.winLossUCB) :
      pow7(child->recursiveValues.winLossLCB) - pow7(node->recursiveValues.winLossLCB);
    double ucbScoreLoss =
      (node->pla == P_WHITE) ?
      node->recursiveValues.scoreUCB - child->recursiveValues.scoreUCB :
      child->recursiveValues.scoreLCB - node->recursiveValues.scoreLCB;
    if(ucbScoreLoss > params.scoreLossCap)
      ucbScoreLoss = params.scoreLossCap;
    double rawPolicy = locAndBookMove.second.rawPolicy;
    double logRawPolicy = log(rawPolicy + 1e-100);
    double childUtility = getUtility(child->recursiveValues);
    double boostedLogRawPolicy = boostLogRawPolicy(logRawPolicy, childUtility, locAndBookMove.second.rawPolicy);
    bool passFavored = passPolicy > 0.15 && passPolicy > rawPolicy * 0.8 && (
      (node->pla == P_WHITE && passUtility > childUtility - 0.02) ||
      (node->pla == P_BLACK && passUtility < childUtility + 0.02)
    );

    double costFromWL =
      ucbWinLossLoss * params.costPerUCBWinLossLoss
      + ucbWinLossLossPow3 * params.costPerUCBWinLossLossPow3
      + ucbWinLossLossPow7 * params.costPerUCBWinLossLossPow7;
    if(costFromWL > node->biggestWLCostFromRoot)
      costFromWL -= params.bonusForBiggestWLCost * (costFromWL - node->biggestWLCostFromRoot);
    double costFromUCB =
      costFromWL
      + ucbScoreLoss * params.costPerUCBScoreLoss;

    double cost =
      node->minCostFromRoot
      + params.costPerMove
      + costFromUCB
      + (-boostedLogRawPolicy * params.costPerLogPolicy)
      + (passFavored ? params.costWhenPassFavored : 0.0);
    locAndBookMove.second.costFromRoot = cost;
    locAndBookMove.second.biggestWLCostFromRoot = std::max(node->biggestWLCostFromRoot, costFromWL);

    // cout << "Setting child " << (int)locAndBookMove.first << " cost from root, parentMinCostFromRoot " << node->minCostFromRoot
    //      << " costPerMove " << costPerMove
    //      << " costFromUCB " << costFromUCB
    //      << " cost due to log policy (" << rawPolicy << ") " << (-boostedLogRawPolicy * costPerLogPolicy)
    //      << " passFavored " << (passFavored ? costWhenPassFavored : 0.0)
    //      << " total " << cost
    //      << endl;

    if(costFromUCB < smallestCostFromUCB)
      smallestCostFromUCB = costFromUCB;
  }

  if(!node->canExpand) {
    node->thisNodeExpansionCost = 1e100;
    // cout << "Can't expand this node" << endl;
  }
  else if(node->canReExpand && node->recursiveValues.visits <= params.maxVisitsForReExpansion) {
    double m = node->recursiveValues.visits / std::max(1.0, params.maxVisitsForReExpansion);
    node->thisNodeExpansionCost = m * params.costPerMovesExpanded + m * m * params.costPerSquaredMovesExpanded;
    smallestCostFromUCB = 0;
    // cout << "maxVisitsForReExpansion met, this node expansion cost is free" << endl;
  }
  else {
    double scoreError = node->thisValuesNotInBook.getAdjustedScoreError(node->book->initialRules);
    double winLossError = node->thisValuesNotInBook.getAdjustedWinLossError(node->book->initialRules);
    double ucbWinLossLoss =
      (node->pla == P_WHITE) ?
      (node->recursiveValues.winLossUCB - (node->thisValuesNotInBook.winLossValue + params.errorFactor * winLossError)) :
      ((node->thisValuesNotInBook.winLossValue - params.errorFactor * winLossError) - node->recursiveValues.winLossLCB);
    double ucbWinLossLossPow3 =
      (node->pla == P_WHITE) ?
      (pow3(node->recursiveValues.winLossUCB) - pow3(node->thisValuesNotInBook.winLossValue + params.errorFactor * winLossError)) :
      (pow3(node->thisValuesNotInBook.winLossValue - params.errorFactor * winLossError) - pow3(node->recursiveValues.winLossLCB));
    double ucbWinLossLossPow7 =
      (node->pla == P_WHITE) ?
      (pow7(node->recursiveValues.winLossUCB) - pow7(node->thisValuesNotInBook.winLossValue + params.errorFactor * winLossError)) :
      (pow7(node->thisValuesNotInBook.winLossValue - params.errorFactor * winLossError) - pow7(node->recursiveValues.winLossLCB));
    double ucbScoreLoss =
      (node->pla == P_WHITE) ?
      (node->recursiveValues.scoreUCB - (node->thisValuesNotInBook.scoreMean + params.errorFactor * scoreError)) :
      ((node->thisValuesNotInBook.scoreMean - params.errorFactor * scoreError) - node->recursiveValues.scoreLCB);
    if(ucbScoreLoss > params.scoreLossCap)
      ucbScoreLoss = params.scoreLossCap;
    double rawPolicy = node->thisValuesNotInBook.maxPolicy;
    double logRawPolicy = log(rawPolicy + 1e-100);
    double notInBookUtility = node->thisValuesNotInBook.winLossValue + node->thisValuesNotInBook.scoreMean * params.utilityPerScore;
    double boostedLogRawPolicy = boostLogRawPolicy(logRawPolicy, notInBookUtility, node->thisValuesNotInBook.maxPolicy);
    bool passFavored = passPolicy > 0.15 && passPolicy > rawPolicy * 0.8 && (
      (node->pla == P_WHITE && passUtility > notInBookUtility - 0.02) ||
      (node->pla == P_BLACK && passUtility < notInBookUtility + 0.02)
    );

    // For computing moves expanded penalty
    double movesExpanded = (double)node->moves.size();

    // If the proposed expansion is significantly better in utility than most expanded moves, the penalty should not be as large.
    double movesExpandedCap = 0.5;
    for(auto& otherLocAndBookMove: node->moves) {
      if(movesExpandedCap >= movesExpanded)
        break;
      const BookNode* otherChild = get(otherLocAndBookMove.second.hash);
      double otherChildUtility = getUtility(otherChild->recursiveValues);
      double gainOverOtherChild =
        (node->pla == P_WHITE) ?
        notInBookUtility - otherChildUtility :
        otherChildUtility - notInBookUtility;

      double proportionToNotCount;
      if(gainOverOtherChild <= 0)
        proportionToNotCount = 0.0;
      else {
        // Reuse params.policyBoostSoftUtilityScale for scaling
        proportionToNotCount = 2.0/(1.0 + exp(-gainOverOtherChild / params.policyBoostSoftUtilityScale)) - 1.0;
      }

      movesExpandedCap += 1.5 * (1.0 - proportionToNotCount);
    }
    if(movesExpanded > movesExpandedCap) {
      // cout << "Capping movesExpanded " << movesExpanded << " to " << movesExpandedCap << endl;
      // for(auto& otherLocAndBookMove: node->moves) {
      //   const BookNode* otherChild = get(otherLocAndBookMove.second.hash);
      //   double otherChildUtility = getUtility(otherChild->recursiveValues);
      //   double gainOverOtherChild =
      //     (node->pla == P_WHITE) ?
      //     notInBookUtility - otherChildUtility :
      //     otherChildUtility - notInBookUtility;
      //   cout << "gainOverOtherChild " << gainOverOtherChild << endl;
      // }
      movesExpanded = movesExpandedCap;
    }


    // If we have more than 1/N of unexpanded policy, we cap the penalty for expanded moves as if we had N.
    if(movesExpanded > 1.0 / (rawPolicy + 1e-30)) {
      movesExpanded = 1.0 / (rawPolicy + 1e-30);
    }



    // cout << "Expansion thisValues " <<
    //   node->thisValuesNotInBook.winLossValue - errorFactor * winLossError << " " <<
    //   node->thisValuesNotInBook.winLossValue << " " <<
    //   node->thisValuesNotInBook.winLossValue + errorFactor * winLossError << endl;
    // cout << "Expansion thisScores " <<
    //   node->thisValuesNotInBook.scoreMean - errorFactor * scoreError << " " <<
    //   node->thisValuesNotInBook.scoreMean << " " <<
    //   node->thisValuesNotInBook.scoreMean + errorFactor * scoreError << endl;
    // cout << "Recursive values " <<
    //   node->recursiveValues.winLossLCB << " " <<
    //   node->recursiveValues.winLossValue << " " <<
    //   node->recursiveValues.winLossUCB << endl;
    // cout << "Recursive scores " <<
    //   node->recursiveValues.scoreLCB << " " <<
    //   node->recursiveValues.scoreMean << " " <<
    //   node->recursiveValues.scoreUCB << endl;
    // cout << "Expansion stats " << ucbWinLossLoss << " " << ucbScoreLoss << " " << rawPolicy << endl;

    double costFromWL =
      ucbWinLossLoss * params.costPerUCBWinLossLoss
      + ucbWinLossLossPow3 * params.costPerUCBWinLossLossPow3
      + ucbWinLossLossPow7 * params.costPerUCBWinLossLossPow7;
    if(costFromWL > node->biggestWLCostFromRoot)
      costFromWL -= params.bonusForBiggestWLCost * (costFromWL - node->biggestWLCostFromRoot);
    double costFromUCB =
      costFromWL
      + ucbScoreLoss * params.costPerUCBScoreLoss;

    node->thisNodeExpansionCost =
      params.costPerMove
      + costFromUCB
      + (-boostedLogRawPolicy * params.costPerLogPolicy)
      + movesExpanded * params.costPerMovesExpanded
      + movesExpanded * movesExpanded * params.costPerSquaredMovesExpanded
      + (passFavored ? params.costWhenPassFavored : 0.0);

    // cout << "Setting this node expansion cost "
    //      << " costPerMove " << costPerMove
    //      << " costFromUCB " << costFromUCB
    //      << " cost due to log policy (" << rawPolicy << ") " << (-boostedLogRawPolicy * costPerLogPolicy)
    //      << " moves expanded cost " << (movesExpanded * costPerMovesExpanded + movesExpanded * movesExpanded * costPerSquaredMovesExpanded)
    //      << " passFavored " << (passFavored ? costWhenPassFavored : 0.0)
    //      << " total " << node->thisNodeExpansionCost
    //      << endl;

    if(costFromUCB < smallestCostFromUCB)
      smallestCostFromUCB = costFromUCB;
  }

  // Partly replenish moves based on ucb cost conficting, since cost conflicting probably means actually the node is
  // interesting for further expansion.
  if(smallestCostFromUCB > 1e-100) {
    // cout << "Replenishing due to smallest cost from UCB " << smallestCostFromUCB << endl;
    for(auto& locAndBookMove: node->moves) {
      // cout << "Child " << (int)locAndBookMove.first
      //      << " cost " << locAndBookMove.second.costFromRoot
      //      << " becomes " <<  (locAndBookMove.second.costFromRoot - 0.8 * smallestCostFromUCB) << endl;
      locAndBookMove.second.costFromRoot -= 0.8 * smallestCostFromUCB;
    }
    // cout << "This node expansion cost " << node->thisNodeExpansionCost
    //      << " becomes " <<  (node->thisNodeExpansionCost - 0.8 * smallestCostFromUCB) << endl;
    node->thisNodeExpansionCost -= 0.8 * smallestCostFromUCB;
  }

  // For each move, in order, if its plain winrate is a lot better than the winrate of other moves, then its cost can't be too much worse.
  for(auto& locAndBookMove: node->moves) {
    const BookNode* child = get(locAndBookMove.second.hash);
    double winLoss = (node->pla == P_WHITE) ? child->recursiveValues.winLossValue : -child->recursiveValues.winLossValue;
    double bestOtherCostFromRoot = locAndBookMove.second.costFromRoot;
    for(auto& locAndBookMoveOther: node->moves) {
      if(locAndBookMoveOther.second.costFromRoot < bestOtherCostFromRoot) {
        const BookNode* otherChild = get(locAndBookMoveOther.second.hash);
        double winLossOther = (node->pla == P_WHITE) ? otherChild->recursiveValues.winLossValue : -otherChild->recursiveValues.winLossValue;
        // At least 1.5% better
        if(winLoss > winLossOther + 0.03)
          bestOtherCostFromRoot = locAndBookMoveOther.second.costFromRoot;
      }
    }
    // Reduce 70% of cost towards the move that we're better than.
    if(bestOtherCostFromRoot < locAndBookMove.second.costFromRoot) {
      // cout << "Child " << (int)locAndBookMove.first
      //      << " cost " << locAndBookMove.second.costFromRoot
      //      << " reduced best cost of moves it beats " << bestOtherCostFromRoot
      //      << " becomes " << locAndBookMove.second.costFromRoot + 0.50 * (bestOtherCostFromRoot - locAndBookMove.second.costFromRoot) << endl;
      locAndBookMove.second.costFromRoot += 0.70 * (bestOtherCostFromRoot - locAndBookMove.second.costFromRoot);
    }
  }
  {
    double winLoss = (node->pla == P_WHITE) ? node->thisValuesNotInBook.winLossValue : -node->thisValuesNotInBook.winLossValue;
    double bestOtherCostFromRoot = node->thisNodeExpansionCost + node->minCostFromRoot;
    for(auto& locAndBookMoveOther: node->moves) {
      if(locAndBookMoveOther.second.costFromRoot < bestOtherCostFromRoot) {
        const BookNode* otherChild = get(locAndBookMoveOther.second.hash);
        double winLossOther = (node->pla == P_WHITE) ? otherChild->recursiveValues.winLossValue : -otherChild->recursiveValues.winLossValue;
        // At least 1.5% better
        if(winLoss > winLossOther + 0.03)
          bestOtherCostFromRoot = locAndBookMoveOther.second.costFromRoot;
      }
    }
    // Reduce 70% of cost towards the move that we're better than.
    if(bestOtherCostFromRoot - node->minCostFromRoot < node->thisNodeExpansionCost) {
      // cout << "This node expansion cost " << node->thisNodeExpansionCost
      //      << " reduced best cost of moves it beats " << bestOtherCostFromRoot - node->minCostFromRoot
      //      << " becomes " << node->thisNodeExpansionCost + 0.50 * (bestOtherCostFromRoot - node->minCostFromRoot - node->thisNodeExpansionCost) << endl;
      node->thisNodeExpansionCost += 0.70 * (bestOtherCostFromRoot - node->minCostFromRoot - node->thisNodeExpansionCost);
    }
  }

  // Apply bonuses to moves now. Apply fully up to 0.75 of the cost.
  for(auto& locAndBookMove: node->moves) {
    const BookNode* child = get(locAndBookMove.second.hash);
    double winLossError = std::fabs(child->recursiveValues.winLossUCB - child->recursiveValues.winLossLCB) / params.errorFactor / 2.0;
    double scoreError = std::fabs(child->recursiveValues.scoreUCB - child->recursiveValues.scoreLCB) / params.errorFactor / 2.0;
    double sharpScoreDiscrepancy = std::fabs(child->recursiveValues.sharpScoreMean - child->recursiveValues.scoreMean);
    double bonus =
      params.bonusPerWinLossError * winLossError +
      params.bonusPerScoreError * scoreError +
      params.bonusPerSharpScoreDiscrepancy * sharpScoreDiscrepancy;
    double bonusCap1 = (locAndBookMove.second.costFromRoot - node->minCostFromRoot) * 0.75;
    if(bonus > bonusCap1)
      bonus = bonusCap1;
    // cout << "Child " << (int)locAndBookMove.first
    //      << " cost " << locAndBookMove.second.costFromRoot
    //      << " errors " << winLossError << " " << scoreError << " " << sharpScoreDiscrepancy
    //      << " bonus " << bonus
    //      << " becomes " <<  (locAndBookMove.second.costFromRoot - bonus) << endl;
    locAndBookMove.second.costFromRoot -= bonus;

    if(locAndBookMove.second.isWLPV) {
      double wlPVBonusScale = (locAndBookMove.second.costFromRoot - node->minCostFromRoot) * (1.0-params.bonusForWLPVFinalProp);
      if(wlPVBonusScale > 0.0) {
        double factor1 = std::max(0.0, 1.0 - square(child->recursiveValues.winLossValue));
        double factor2 = 4.0 * std::max(0.0, 0.25 - square(0.5 - std::fabs(child->recursiveValues.winLossValue)));
        double wlPVBonus = wlPVBonusScale * tanh(factor1 * params.bonusForWLPV1 + factor2 * params.bonusForWLPV2);
        // cout << "Child " << (int)locAndBookMove.first
        //      << " cost " << locAndBookMove.second.costFromRoot
        //      << " wlpv factors " << factor1 << " " << factor2
        //      << " becomes " <<  (locAndBookMove.second.costFromRoot - wlPVBonus) << endl;
        locAndBookMove.second.costFromRoot -= wlPVBonus;
      }
    }
  }
  {
    double winLossError = node->thisValuesNotInBook.getAdjustedWinLossError(node->book->initialRules);
    double scoreError = node->thisValuesNotInBook.getAdjustedScoreError(node->book->initialRules);
    double sharpScoreDiscrepancy = std::fabs(node->thisValuesNotInBook.sharpScoreMeanRaw - node->thisValuesNotInBook.scoreMean);

    // If there's an unusually large share of the policy not expanded, add a mild bonus for it.
    // For the final node expansion cost, sharp score discrepancy beyond 1 point is not capped, to encourage expanding the
    // search to better resolve the difference, since sharp score can have some weird outliers.
    double movesExpanded = (double)node->moves.size();
    double excessUnexpandedPolicy = 0.0;
    if(movesExpanded > 0 && node->thisValuesNotInBook.maxPolicy > 1.0 / movesExpanded)
      excessUnexpandedPolicy = node->thisValuesNotInBook.maxPolicy - 1.0 / movesExpanded;
    double bonus =
      params.bonusPerWinLossError * winLossError +
      params.bonusPerScoreError * scoreError +
      params.bonusPerSharpScoreDiscrepancy * std::min(sharpScoreDiscrepancy, 1.0) +
      params.bonusPerExcessUnexpandedPolicy * excessUnexpandedPolicy;
    double bonusCap1 = node->thisNodeExpansionCost * 0.75;
    if(bonus > bonusCap1)
      bonus = bonusCap1;

    // Sharp score discrepancy is an uncapped bonus at the very final node
    bonus += params.bonusPerSharpScoreDiscrepancy * std::max(0.0, sharpScoreDiscrepancy - 1.0);
    // cout << "This node expansion cost " << node->thisNodeExpansionCost
    //      << " errors " << winLossError << " " << scoreError << " " << sharpScoreDiscrepancy
    //      << " bonus " << bonus
    //      << " becomes " <<  (node->thisNodeExpansionCost - bonus) << endl;
    node->thisNodeExpansionCost -= bonus;

    // bonusPerUnexpandedBestWinLoss is an uncapped bonus
    // Also if a move is better at all we offset it as if it were additionally better by this much
    const double BEST_WINLOSS_OFFSET = 0.02;
    {
      double winLoss = (node->pla == P_WHITE) ? node->thisValuesNotInBook.winLossValue : -node->thisValuesNotInBook.winLossValue;
      bool anyOtherWinLossFound = false;
      double bestOtherWinLoss = 0.0;
      double bestOtherVisits = 0.0;
      double totalOtherVisits = 0.0;
      for(auto& locAndBookMoveOther: node->moves) {
        const BookNode* otherChild = get(locAndBookMoveOther.second.hash);
        double winLossOther = (node->pla == P_WHITE) ? otherChild->recursiveValues.winLossValue : -otherChild->recursiveValues.winLossValue;
        if(!anyOtherWinLossFound || winLossOther > bestOtherWinLoss) {
          bestOtherWinLoss = winLossOther;
          bestOtherVisits = otherChild->recursiveValues.visits;
          anyOtherWinLossFound = true;
        }
        totalOtherVisits += otherChild->recursiveValues.visits;
      }
      if(anyOtherWinLossFound && winLoss > bestOtherWinLoss) {
        double visitsFactor = 0.5 * (
          std::min(1.0, sqrt(bestOtherVisits / std::max(1.0, params.visitsScale))) +
          std::min(1.0, sqrt(totalOtherVisits / std::max(1.0, params.visitsScale)))
        );
        node->thisNodeExpansionCost -= params.bonusPerUnexpandedBestWinLoss * (winLoss - bestOtherWinLoss + BEST_WINLOSS_OFFSET) * visitsFactor;
      }
    }
    if(node->moves.size() >= 2) {
      // Also things eligible for reexpansion should get a bonus if they are way better than other stuff that has a lot of visits.
      // But it counts 0.75 times as much.
      if(bestWinLossChild != NULL && bestWinLossChild->recursiveValues.visits <= params.maxVisitsForReExpansion) {
        bool anyOtherWinLossFound = false;
        double bestOtherWinLossThisPerspective = 0.0;
        double bestOtherVisits = 0.0;
        double totalOtherVisits = 0.0;
        for(auto& locAndBookMoveOther: node->moves) {
          const BookNode* otherChild = get(locAndBookMoveOther.second.hash);
          if(otherChild != bestWinLossChild) {
            double winLossOtherThisPerspective =
              (node->pla == P_WHITE) ? otherChild->recursiveValues.winLossValue : -otherChild->recursiveValues.winLossValue;
            if(!anyOtherWinLossFound || winLossOtherThisPerspective > bestOtherWinLossThisPerspective) {
              bestOtherWinLossThisPerspective = winLossOtherThisPerspective;
              bestOtherVisits = otherChild->recursiveValues.visits;
              anyOtherWinLossFound = true;
            }
            totalOtherVisits += otherChild->recursiveValues.visits;
          }
        }

        // The best child has fewer visits than the second best
        if(
          anyOtherWinLossFound && bestWinLossThisPerspective > bestOtherWinLossThisPerspective &&
          bestWinLossChild->recursiveValues.visits < bestOtherVisits
        ) {
          double visitsFactor = 0.5 * (
            std::min(1.0, sqrt(bestOtherVisits / std::max(1.0, params.visitsScale))) +
            std::min(1.0, sqrt(totalOtherVisits / std::max(1.0, params.visitsScale)))
          );
          // Subtract off for what was actually explored
          visitsFactor -= std::min(1.0, sqrt(bestWinLossChild->recursiveValues.visits / std::max(1.0, params.visitsScale)));

          for(auto& locAndBookMove: node->moves) {
            const BookNode* child = get(locAndBookMove.second.hash);
            if(child == bestWinLossChild) {
              locAndBookMove.second.costFromRoot -=
                0.75 * params.bonusPerUnexpandedBestWinLoss * (bestWinLossThisPerspective - bestOtherWinLossThisPerspective + BEST_WINLOSS_OFFSET) * visitsFactor;
              break;
            }
          }
        }
      }

      // Look to see if a child has a LOT fewer visits than one that is worse or very close - add a bonus.
      auto behindInVisitsBonus = [&](double childWinLoss, double adjustedVisits) {
        double maxBonus = 0.0;
        for(auto& otherLocAndBookMove: node->moves) {
          const BookNode* otherChild = get(otherLocAndBookMove.second.hash);
          double otherVisits = otherChild->recursiveValues.adjustedVisits;
          if(otherVisits <= 30.0 * adjustedVisits)
            continue;
          double otherChildWinLoss = otherChild->recursiveValues.winLossValue;
          double gainOverOtherChild =
            (node->pla == P_WHITE) ?
            childWinLoss - otherChildWinLoss :
            otherChildWinLoss - childWinLoss;
          if(gainOverOtherChild <= -params.policyBoostSoftUtilityScale)
            continue;
          double thisBonus =
            log10(otherVisits / (30.0 * adjustedVisits))
            - 0.5 * log10(std::max(adjustedVisits,params.visitsScaleLeaves) / params.visitsScaleLeaves);
          if(gainOverOtherChild < 0.0) {
            double factor = (gainOverOtherChild + params.policyBoostSoftUtilityScale) / (params.policyBoostSoftUtilityScale + 1e-10);
            thisBonus = thisBonus * factor * factor;
          }
          maxBonus = std::max(maxBonus, thisBonus);
        }
        if(maxBonus <= 0.0)
          return 0.0;

        // Also scale down by how far we are from the best, and scale down by if we are close to losing so nothing matters
        double gainOverBest =
          (node->pla == P_WHITE) ?
          childWinLoss - bestWinLoss :
          bestWinLoss - childWinLoss;
        // Just in case
        gainOverBest = std::min(gainOverBest, 0.0);
        double losingScale = std::min(1.0, (node->pla == P_WHITE) ? childWinLoss + 1.0 : 1.0 - childWinLoss);
        return maxBonus * exp(gainOverBest / (2.0 * params.policyBoostSoftUtilityScale)) * losingScale;
      };

      // double anyBehindInVisits = false;
      for(auto& locAndBookMove: node->moves) {
        const BookNode* child = get(locAndBookMove.second.hash);
        double childWinLoss = child->recursiveValues.winLossValue;
        double adjustedVisits = child->recursiveValues.adjustedVisits;
        double childBonus = behindInVisitsBonus(childWinLoss,adjustedVisits);
        // if(childBonus > 0.25)
        //   anyBehindInVisits = true;
        locAndBookMove.second.costFromRoot -= childBonus;
      }
      {
        double childWinLoss = node->thisValuesNotInBook.winLossValue;
        double adjustedVisits = node->thisValuesNotInBook.visits;
        double childBonus = behindInVisitsBonus(childWinLoss,adjustedVisits);
        // if(childBonus > 0.25)
        //   anyBehindInVisits = true;
        node->thisNodeExpansionCost -= childBonus;
      }

      // if(anyBehindInVisits) {
      //   cout << "ANY BEHINDINVISITS" << endl;
      //   for(auto& locAndBookMove: node->moves) {
      //     const BookNode* child = get(locAndBookMove.second.hash);
      //     double childWinLoss = child->recursiveValues.winLossValue;
      //     double childWinLossPersp = (node->pla == P_WHITE) ? child->recursiveValues.winLossValue : -child->recursiveValues.winLossValue;
      //     double adjustedVisits = child->recursiveValues.adjustedVisits;
      //     double childBonus = behindInVisitsBonus(childWinLoss,adjustedVisits);
      //     cout << "child " << childWinLossPersp << " " << adjustedVisits << " " << childBonus << endl;
      //   }
      //   {
      //     double childWinLoss = node->thisValuesNotInBook.winLossValue;
      //     double childWinLossPersp = (node->pla == P_WHITE) ? node->thisValuesNotInBook.winLossValue : -node->thisValuesNotInBook.winLossValue;
      //     double adjustedVisits = node->thisValuesNotInBook.visits;
      //     double childBonus = behindInVisitsBonus(childWinLoss,adjustedVisits);
      //     cout << "xpand " << childWinLossPersp << " " << adjustedVisits << " " << childBonus << endl;
      //   }
      // }
    }
  }

  // Uncapped
  if(node->expansionIsWLPV || (node->canReExpand && node->recursiveValues.visits <= params.maxVisitsForReExpansion)) {
    double wlPVBonusScale = node->thisNodeExpansionCost + std::max(0.0, node->minCostFromRoot - node->minCostFromRootWLPV) * params.bonusForWLPVFinalProp;
    if(wlPVBonusScale > 0.0) {
      double factor1 = std::max(0.0, 1.0 - square(node->thisValuesNotInBook.winLossValue));
      double factor2 = 4.0 * std::max(0.0, 0.25 - square(0.5 - std::fabs(node->thisValuesNotInBook.winLossValue)));
      double wlPVBonus = wlPVBonusScale * tanh(factor1 * params.bonusForWLPV1 + factor2 * params.bonusForWLPV2);
      // cout << "This node expansion cost " << node->thisNodeExpansionCost
      //      << " wlpv factors " << factor1 << " " << factor2
      //      << " becomes " <<  (node->thisNodeExpansionCost - wlPVBonus) << endl;
      node->thisNodeExpansionCost -= wlPVBonus;
    }
  }

  double depthFromRootFactor = 1.0 - params.earlyBookCostReductionFactor * pow(params.earlyBookCostReductionLambda, node->minDepthFromRoot);
  for(auto& locAndBookMove: node->moves) {
    locAndBookMove.second.costFromRoot = node->minCostFromRoot + (locAndBookMove.second.costFromRoot - node->minCostFromRoot) * depthFromRootFactor;
  }
  {
    node->thisNodeExpansionCost = node->thisNodeExpansionCost * depthFromRootFactor;
  }

  if(contains(expandBonusByHash, node->hash)) {
    double bonus = expandBonusByHash[node->hash];
    node->thisNodeExpansionCost -= bonus;
  }
  if(contains(branchRequiredByHash, node->hash)) {
    int requiredBranch = branchRequiredByHash[node->hash];
    if(node->moves.size() < requiredBranch) {
      node->thisNodeExpansionCost -= 700.0;
    }
    else {
      // If we require branch, also ensure that branching children have enough visits
      int childEnoughVisitsCount = 0;
      for(auto& locAndBookMove: node->moves) {
        const BookNode* child = get(locAndBookMove.second.hash);
        double childVisits = child->recursiveValues.visits;
        if(childVisits > params.maxVisitsForReExpansion) {
          childEnoughVisitsCount += 1;
        }
      }
      if(childEnoughVisitsCount < requiredBranch) {
        std::vector<int> sortIdxBuf;
        std::vector<double> sortValuesBuf;
        std::vector<Loc> locBuf;
        const double plaFactor = node->pla == P_WHITE ? 1.0 : -1.0;
        for(auto& locAndBookMove: node->moves) {
          const BookNode* child = get(locAndBookMove.second.hash);
          const RecursiveBookValues& vals = child->recursiveValues;
          sortIdxBuf.push_back((int)sortIdxBuf.size());
          sortValuesBuf.push_back(
            getSortingValue(plaFactor,vals.winLossValue,vals.scoreMean,vals.sharpScoreMean,vals.scoreLCB,vals.scoreUCB,locAndBookMove.second.rawPolicy)
          );
          locBuf.push_back(locAndBookMove.first);
        }
        // Sort from best to worst
        std::sort(
          sortIdxBuf.begin(),sortIdxBuf.end(),
          [&](const int& idx0,
              const int& idx1) {
            return sortValuesBuf[idx0] > sortValuesBuf[idx1];
          }
        );
        // In order, bonus the costs of the most promising moves
        int numBonused = 0;
        for(const int& idx: sortIdxBuf) {
          if(numBonused + childEnoughVisitsCount >= requiredBranch)
            break;
          Loc loc = locBuf[idx];
          BookMove& bookMove = node->moves[loc];
          const BookNode* child = get(bookMove.hash);
          double childVisits = child->recursiveValues.visits;
          if(childVisits <= params.maxVisitsForReExpansion) {
            numBonused += 1;
            bookMove.costFromRoot -= 200.0;
          }
        }
      }
    }
  }

  // cout << "Setting cost " << node->hash << " " << node->minCostFromRoot << " " << node->thisNodeExpansionCost << endl;
  // cout << "TOTAL THIS NODE COST " << node->minCostFromRoot + node->thisNodeExpansionCost << endl;
}

static const string HTML_TEMPLATE = R"%%(
<html>
<header>
<link rel="stylesheet" href="../book.css">
<script>
$$DATA_VARS
</script>
<script type="text/javascript" src="../book.js"></script>
</header>
<body>
</body>
</html>
)%%";

double Book::getSortingValue(
  double plaFactor,
  double winLossValue,
  double scoreMean,
  double sharpScoreMeanClamped,
  double scoreLCB,
  double scoreUCB,
  double rawPolicy
) const {
  double score = 0.5 * (sharpScoreMeanClamped + scoreMean);
  double sortingValue =
    plaFactor * (winLossValue + clampScoreForSorting(score, winLossValue) * params.utilityPerScore * 0.75)
    + plaFactor * clampScoreForSorting(0.5*(plaFactor+1.0) * scoreLCB + 0.5*(1.0-plaFactor) * scoreUCB, winLossValue) * 0.25 * params.utilityPerScore
    + params.utilityPerPolicyForSorting * (0.75 * rawPolicy + 0.5 * log10(rawPolicy + 0.0001)/4.0) * (1.0 + winLossValue*winLossValue);
  return sortingValue;
}


int64_t Book::exportToHtmlDir(
  const string& dirName,
  const string& rulesLabel,
  const string& rulesLink,
  bool devMode,
  double htmlMinVisits,
  Logger& logger
) {
  MakeDir::make(dirName);
  const char* hexChars = "0123456789ABCDEF";
  for(int i = 0; i<16; i++) {
    for(int j = 0; j<16; j++) {
      MakeDir::make(dirName + "/" + hexChars[i] + hexChars[j]);
    }
  }
  MakeDir::make(dirName + "/root");
  {
    std::ofstream out;
    FileUtils::open(out, dirName + "/book.js");
    out << "const rulesLabel = \"" + rulesLabel + "\";\n";
    out << "const rulesLink = \"" + rulesLink + "\";\n";
    out << "const devMode = " + (devMode ? string("true") : string("false")) + ";\n";
    out << "const bSizeX = " + Global::intToString(initialBoard.x_size) + ";\n";
    out << "const bSizeY = " + Global::intToString(initialBoard.y_size) + ";\n";
    out << Book::BOOK_JS1;
    out << Book::BOOK_JS2;
    out << Book::BOOK_JS3;
    out.close();
  }
  {
    std::ofstream out;
    FileUtils::open(out, dirName + "/book.css");
    out << Book::BOOK_CSS;
    out.close();
  }

  char toStringBuf[1024];
  auto doubleToStringFourDigits = [&](double x) {
    std::sprintf(toStringBuf,"%.4f",x);
    return string(toStringBuf);
  };
  auto doubleToStringTwoDigits = [&](double x) {
    std::sprintf(toStringBuf,"%.2f",x);
    return string(toStringBuf);
  };
  auto doubleToStringZeroDigits = [&](double x) {
    std::sprintf(toStringBuf,"%.0f",x);
    return string(toStringBuf);
  };

  auto getFilePath = [&](BookNode* node, bool relative) {
    string path = relative ? "" : dirName + "/";
    if(node == root)
      path += "root/root";
    else {
      string hashStr = node->hash.toString();
      assert(hashStr.size() > 10);
      // Pull from the middle of the string, to avoid the fact that the hashes are
      // biased towards small hashes due to finding the minimum of the 8 symmetries.
      path += hashStr.substr(8,2) + "/" + node->hash.toString();
    }
    path += ".html";
    return path;
  };

  int64_t numFilesWritten = 0;

  std::function<void(BookNode*)> f = [&](BookNode* node) {
    // Entirely omit exporting nodes that are simply leaves, to save on the number of files we have to produce and serve.
    // if(node != root && node->moves.size() == 0)
    //   return;

    // Omit exporting nodes that have too few visits
    if(node->recursiveValues.visits < htmlMinVisits)
      return;

    string filePath = getFilePath(node, false);
    string html = HTML_TEMPLATE;
    auto replace = [&](const string& key, const string& replacement) {
      size_t pos = html.find(key);
      assert(pos != string::npos);
      html.replace(pos, key.size(), replacement);
    };

    const int symmetry = 0;
    SymBookNode symNode(node, symmetry);

    BoardHistory hist;
    vector<Loc> moveHistory;
    bool suc = symNode.getBoardHistoryReachingHere(hist,moveHistory);
    if(!suc) {
      logger.write("WARNING: Failed to get board history reaching node when trying to export to html, probably there is some bug");
      logger.write("or else some hash collision or something else is wrong.");
      logger.write("BookHash of node unable to export: " + symNode.hash().toString());
      std::ostringstream movesOut;
      for(Loc move: moveHistory)
        movesOut << Location::toString(move,initialBoard) << " ";
      logger.write("Moves:");
      logger.write(movesOut.str());
      return;
    }

    // Omit exporting nodes that are past the normal game end.
    if(hist.encorePhase > 0)
      return;

    Board board = hist.getRecentBoard(0);

    if(contains(rulesLabel,'"') || contains(rulesLabel,'\n'))
      throw StringError("rulesLabel cannot contain quotes or newlines");
    if(contains(rulesLink,'"') || contains(rulesLink,'\n'))
      throw StringError("rulesLink cannot contain quotes or newlines");
    string dataVarsStr;
    dataVarsStr += "const nextPla = " + Global::intToString(node->pla) + ";\n";
    {
      SymBookNode parent = symNode.canonicalParent();
      if(parent.isNull()) {
        dataVarsStr += "const pLink = '';\n";
        dataVarsStr += "const pSym = 0;\n";
      }
      else {
        string parentPath = getFilePath(parent.node, true);
        dataVarsStr += "const pLink = '../" + parentPath + "';\n";
        dataVarsStr += "const pSym = " + Global::intToString(parent.symmetryOfNode) + ";\n";
      }
    }
    dataVarsStr += "const board = [";
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        dataVarsStr += Global::intToString(board.colors[loc]) + ",";
      }
    }
    dataVarsStr += "];\n";
    dataVarsStr += "const links = {";
    string linkSymmetriesStr = "const linkSyms = {";
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        SymBookNode child = symNode.follow(loc);
        // Entirely omit linking children that are simply leaves, to save on the number of files we have to produce and serve.
        // if(!child.isNull() && child.node->moves.size() > 0) {
        if(!child.isNull()) {
          // Omit exporting nodes that have too few visits
          if(child.recursiveValues().visits >= htmlMinVisits) {
            string childPath = getFilePath(child.node, true);
            dataVarsStr += Global::intToString(x+y*board.x_size) + ":'../" + childPath + "',";
            linkSymmetriesStr += Global::intToString(x+y*board.x_size) + ":" + Global::intToString(child.symmetryOfNode) + ",";
          }
        }
      }
    }
    // Also handle pass, pass get indexed one after the legal moves
    {
      Loc loc = Board::PASS_LOC;
      // Avoid linking children that would end the phase
      if(!hist.passWouldEndPhase(board,node->pla)) {
        SymBookNode child = symNode.follow(loc);
        // Entirely omit linking children that are simply leaves, to save on the number of files we have to produce and serve.
        // if(!child.isNull() && child.node->moves.size() > 0) {
        if(!child.isNull()) {
          // Omit exporting nodes that have too few visits
          if(child.recursiveValues().visits >= htmlMinVisits) {
            string childPath = getFilePath(child.node, true);
            dataVarsStr += Global::intToString(board.y_size*board.x_size) + ":'../" + childPath + "',";
            linkSymmetriesStr += Global::intToString(board.y_size*board.x_size) + ":" + Global::intToString(child.symmetryOfNode) + ",";
          }
        }
      }
    }
    dataVarsStr += "};\n";
    linkSymmetriesStr += "};\n";
    dataVarsStr += linkSymmetriesStr;

    vector<BookMove> uniqueMovesInBook = symNode.getUniqueMovesInBook();
    vector<RecursiveBookValues> uniqueChildValues;
    vector<double> uniqueChildCosts;
    vector<double> uniqueChildCostsWLPV;
    vector<double> uniqueChildBiggestWLCost;
    vector<size_t> uniqueMoveIdxs;
    vector<double> sortingValues;
    for(BookMove& bookMove: uniqueMovesInBook) {
      SymBookNode child = symNode.follow(bookMove.move);
      uniqueChildValues.push_back(child.node->recursiveValues);
      uniqueChildCosts.push_back(child.node->minCostFromRoot);
      uniqueChildCostsWLPV.push_back(child.node->minCostFromRootWLPV);
      uniqueChildBiggestWLCost.push_back(child.node->biggestWLCostFromRoot);
      uniqueMoveIdxs.push_back(uniqueMoveIdxs.size());

      RecursiveBookValues& vals = child.node->recursiveValues;
      double plaFactor = node->pla == P_WHITE ? 1.0 : -1.0;
      double sortingValue = getSortingValue(plaFactor,vals.winLossValue,vals.scoreMean,vals.sharpScoreMean,vals.scoreLCB,vals.scoreUCB,bookMove.rawPolicy);
      sortingValues.push_back(sortingValue);
    }

    std::sort(
      uniqueMoveIdxs.begin(),uniqueMoveIdxs.end(),
      [&](const size_t& idx0,
          const size_t& idx1) {
        return sortingValues[idx0] > sortingValues[idx1];
      }
    );

    vector<int> equivalentSymmetries = symNode.getSymmetries();
    std::set<Loc> locsHandled;

    dataVarsStr += "const moves = [";
    for(size_t idx: uniqueMoveIdxs) {
      dataVarsStr += "{";
      const Loc passLoc = Board::PASS_LOC;
      if(uniqueMovesInBook[idx].move != passLoc) {
        dataVarsStr += "'xy':[";
        for(int s: equivalentSymmetries) {
          Loc symMove = SymmetryHelpers::getSymLoc(uniqueMovesInBook[idx].move,initialBoard,s);
          if(contains(locsHandled, symMove))
            continue;
          locsHandled.insert(symMove);
          dataVarsStr += "[";
          dataVarsStr += Global::intToString(Location::getX(symMove, initialBoard.x_size));
          dataVarsStr += ",";
          dataVarsStr += Global::intToString(Location::getY(symMove, initialBoard.x_size));
          dataVarsStr += "],";
        }
        dataVarsStr += "],";
      }
      if(uniqueMovesInBook[idx].move == passLoc)
        dataVarsStr += "'move':'" + Location::toString(uniqueMovesInBook[idx].move, initialBoard) + "',";
      dataVarsStr += "'p':" + doubleToStringFourDigits(uniqueMovesInBook[idx].rawPolicy) + ",";
      dataVarsStr += "'wl':" + doubleToStringFourDigits(uniqueChildValues[idx].winLossValue) + ",";
      if(devMode) {
        dataVarsStr += "'wlUCB':" + doubleToStringFourDigits(uniqueChildValues[idx].winLossUCB) + ",";
        dataVarsStr += "'wlLCB':" + doubleToStringFourDigits(uniqueChildValues[idx].winLossLCB) + ",";
        dataVarsStr += "'sM':" + doubleToStringTwoDigits(uniqueChildValues[idx].scoreMean) + ",";
        dataVarsStr += "'ssM':" + doubleToStringTwoDigits(uniqueChildValues[idx].sharpScoreMean) + ",";
        dataVarsStr += "'sUCB':" + doubleToStringTwoDigits(uniqueChildValues[idx].scoreUCB) + ",";
        dataVarsStr += "'sLCB':" + doubleToStringTwoDigits(uniqueChildValues[idx].scoreLCB) + ",";
        //dataVarsStr += "'w':" + doubleToStringZeroDigits(uniqueChildValues[idx].weight) + ",";
        dataVarsStr += "'v':" + doubleToStringZeroDigits(uniqueChildValues[idx].visits) + ",";
        dataVarsStr += "'av':" + doubleToStringZeroDigits(uniqueChildValues[idx].adjustedVisits) + ",";
        dataVarsStr += "'cost':" + doubleToStringFourDigits(uniqueMovesInBook[idx].costFromRoot - node->minCostFromRoot) + ",";
        dataVarsStr += "'costRoot':" + doubleToStringFourDigits(uniqueChildCosts[idx]) + ",";
        dataVarsStr += "'costWLPV':" + doubleToStringFourDigits(uniqueChildCostsWLPV[idx]) + ",";
        dataVarsStr += "'bigWLC':" + doubleToStringFourDigits(uniqueChildBiggestWLCost[idx]) + ",";
      }
      else {
        dataVarsStr += "'ssM':" + doubleToStringTwoDigits(0.5*(uniqueChildValues[idx].scoreMean+uniqueChildValues[idx].sharpScoreMean)) + ",";
        dataVarsStr += "'wlRad':" + doubleToStringFourDigits(0.5*(uniqueChildValues[idx].winLossUCB - uniqueChildValues[idx].winLossLCB)) + ",";
        dataVarsStr += "'sRad':" + doubleToStringTwoDigits(0.5*(uniqueChildValues[idx].scoreUCB - uniqueChildValues[idx].scoreLCB)) + ",";
        dataVarsStr += "'v':" + doubleToStringZeroDigits(uniqueChildValues[idx].visits) + ",";
        dataVarsStr += "'av':" + doubleToStringZeroDigits(uniqueChildValues[idx].adjustedVisits) + ",";
      }
      dataVarsStr += "},";
    }
    {
      BookValues& values = node->thisValuesNotInBook;
      if(values.maxPolicy > 0.0) {
        double scoreError = values.getAdjustedScoreError(node->book->initialRules);
        double winLossError = values.getAdjustedWinLossError(node->book->initialRules);
        double winLossValueUCB = values.winLossValue + params.errorFactor * winLossError;
        double winLossValueLCB = values.winLossValue - params.errorFactor * winLossError;
        double scoreUCB = values.scoreMean + params.errorFactor * scoreError;
        double scoreLCB = values.scoreMean - params.errorFactor * scoreError;
        // double scoreFinalUCB = values.scoreMean + errorFactor * values.scoreStdev;
        // double scoreFinalLCB = values.scoreMean - errorFactor * values.scoreStdev;

        double scoreMean = values.scoreMean;
        // Adjust the LCB/UCB to reflect the uncertainty from sharp score
        // Skip scoreUCB/scoreLCB adjustment if there isn't any error at all, where the net doesn't support it.
        if(scoreError > 0) {
          if(values.sharpScoreMeanRaw > scoreUCB)
            scoreUCB = values.sharpScoreMeanRaw;
          if(values.sharpScoreMeanRaw < scoreLCB)
            scoreLCB = values.sharpScoreMeanRaw;
        }
        double sharpScoreMean = values.sharpScoreMeanClamped;

        dataVarsStr += "{";
        dataVarsStr += "'move':'other',";
        dataVarsStr += "'p':" + doubleToStringFourDigits(values.maxPolicy) + ",";
        dataVarsStr += "'wl':" + doubleToStringFourDigits(values.winLossValue) + ",";
        if(devMode) {
          dataVarsStr += "'wlUCB':" + doubleToStringFourDigits(winLossValueUCB) + ",";
          dataVarsStr += "'wlLCB':" + doubleToStringFourDigits(winLossValueLCB) + ",";
          dataVarsStr += "'sM':" + doubleToStringTwoDigits(scoreMean) + ",";
          dataVarsStr += "'ssM':" + doubleToStringTwoDigits(sharpScoreMean) + ",";
          dataVarsStr += "'sUCB':" + doubleToStringTwoDigits(scoreUCB) + ",";
          dataVarsStr += "'sLCB':" + doubleToStringTwoDigits(scoreLCB) + ",";
          dataVarsStr += "'w':" + doubleToStringZeroDigits(values.weight) + ",";
          dataVarsStr += "'v':" + doubleToStringZeroDigits(values.visits) + ",";
          dataVarsStr += "'av':" + doubleToStringZeroDigits(values.visits) + ",";
          dataVarsStr += "'cost':" + doubleToStringFourDigits(node->thisNodeExpansionCost) + ",";
          dataVarsStr += "'costRoot':" + doubleToStringFourDigits(node->minCostFromRoot + node->thisNodeExpansionCost) + ",";
          dataVarsStr += "'costWLPV':" + doubleToStringFourDigits(node->expansionIsWLPV ? node->minCostFromRootWLPV : node->minCostFromRoot + node->thisNodeExpansionCost) + ",";
          dataVarsStr += "'bigWLC':" + doubleToStringFourDigits(node->biggestWLCostFromRoot) + ",";
        }
        else {
          dataVarsStr += "'ssM':" + doubleToStringTwoDigits(0.5*(scoreMean+sharpScoreMean)) + ",";
          dataVarsStr += "'wlRad':" + doubleToStringFourDigits(0.5*(winLossValueUCB-winLossValueLCB)) + ",";
          dataVarsStr += "'sRad':" + doubleToStringTwoDigits(0.5*(scoreUCB-scoreLCB)) + ",";
          dataVarsStr += "'v':" + doubleToStringZeroDigits(values.visits) + ",";
          dataVarsStr += "'av':" + doubleToStringZeroDigits(values.visits) + ",";
        }
        dataVarsStr += "},";
      }
    }
    dataVarsStr += "];\n";

    replace("$$DATA_VARS",dataVarsStr);

    std::ofstream out;
    FileUtils::open(out, filePath);
    out << html;
    out.close();
    numFilesWritten += 1;
  };
  iterateEntireBookPreOrder(f);
  return numFilesWritten;
}

static const char BOARD_LINE_DELIMITER = '|';

static double roundDouble(double x, double invMinPrec) {
  return round(x * invMinPrec) / invMinPrec;
}

void Book::saveToFile(const string& fileName) const {
  string tmpFileName = fileName + ".tmp";
  std::ofstream out;
  FileUtils::open(out, tmpFileName);

  {
    json paramsDump;
    paramsDump["version"] = bookVersion;
    paramsDump["initialBoard"] = Board::toJson(initialBoard);
    paramsDump["initialRules"] = initialRules.toJson();
    paramsDump["initialPla"] = PlayerIO::playerToString(initialPla);
    paramsDump["repBound"] = repBound;
    paramsDump["errorFactor"] = params.errorFactor;
    paramsDump["costPerMove"] = params.costPerMove;
    paramsDump["costPerUCBWinLossLoss"] = params.costPerUCBWinLossLoss;
    paramsDump["costPerUCBWinLossLossPow3"] = params.costPerUCBWinLossLossPow3;
    paramsDump["costPerUCBWinLossLossPow7"] = params.costPerUCBWinLossLossPow7;
    paramsDump["costPerUCBScoreLoss"] = params.costPerUCBScoreLoss;
    paramsDump["costPerLogPolicy"] = params.costPerLogPolicy;
    paramsDump["costPerMovesExpanded"] = params.costPerMovesExpanded;
    paramsDump["costPerSquaredMovesExpanded"] = params.costPerSquaredMovesExpanded;
    paramsDump["costWhenPassFavored"] = params.costWhenPassFavored;
    paramsDump["bonusPerWinLossError"] = params.bonusPerWinLossError;
    paramsDump["bonusPerScoreError"] = params.bonusPerScoreError;
    paramsDump["bonusPerSharpScoreDiscrepancy"] = params.bonusPerSharpScoreDiscrepancy;
    paramsDump["bonusPerExcessUnexpandedPolicy"] = params.bonusPerExcessUnexpandedPolicy;
    paramsDump["bonusPerUnexpandedBestWinLoss"] = params.bonusPerUnexpandedBestWinLoss;
    paramsDump["bonusForWLPV1"] = params.bonusForWLPV1;
    paramsDump["bonusForWLPV2"] = params.bonusForWLPV2;
    paramsDump["bonusForWLPVFinalProp"] = params.bonusForWLPVFinalProp;
    paramsDump["bonusForBiggestWLCost"] = params.bonusForBiggestWLCost;
    paramsDump["bonusBehindInVisitsScale"] = params.bonusBehindInVisitsScale;
    paramsDump["scoreLossCap"] = params.scoreLossCap;
    paramsDump["earlyBookCostReductionFactor"] = params.earlyBookCostReductionFactor;
    paramsDump["earlyBookCostReductionLambda"] = params.earlyBookCostReductionLambda;
    paramsDump["utilityPerScore"] = params.utilityPerScore;
    paramsDump["policyBoostSoftUtilityScale"] = params.policyBoostSoftUtilityScale;
    paramsDump["utilityPerPolicyForSorting"] = params.utilityPerPolicyForSorting;
    paramsDump["adjustedVisitsWLScale"] = params.adjustedVisitsWLScale;
    paramsDump["maxVisitsForReExpansion"] = params.maxVisitsForReExpansion;
    paramsDump["visitsScale"] = params.visitsScale;
    paramsDump["visitsScaleLeaves"] = params.visitsScaleLeaves;
    paramsDump["sharpScoreOutlierCap"] = params.sharpScoreOutlierCap;
    paramsDump["initialSymmetry"] = initialSymmetry;
    out << paramsDump.dump() << endl;
  }

  // Interning of hash specific to this save file, to shorten file size and save/load times
  // We don't rely on nodeIdxs to be constant across different saves and loads, although in practice
  // they might be unless someone manually edits the save files.
  if(bookVersion >= 2) {
    out << nodes.size() << "\n";
    for(size_t nodeIdx = 0; nodeIdx<nodes.size(); nodeIdx++) {
      out << nodes[nodeIdx]->hash.toString() << "\n";
    }
  }
  out << std::flush;

  for(size_t nodeIdx = 0; nodeIdx<nodes.size(); nodeIdx++) {
    const BookNode* node = nodes[nodeIdx];
    json nodeData = json::object();
    if(bookVersion >= 2) {
      nodeData["id"] = nodeIdx;
      nodeData["pla"] = PlayerIO::playerToStringShort(node->pla);
      nodeData["syms"] = node->symmetries;
      nodeData["wl"] = roundDouble(node->thisValuesNotInBook.winLossValue, 100000000);
      nodeData["sM"] = roundDouble(node->thisValuesNotInBook.scoreMean, 1000000);
      nodeData["ssM"] = roundDouble(node->thisValuesNotInBook.sharpScoreMeanRaw, 1000000);
      nodeData["wlE"] = roundDouble(node->thisValuesNotInBook.winLossError, 100000000);
      nodeData["sE"] = roundDouble(node->thisValuesNotInBook.scoreError, 1000000);
      nodeData["sStd"] = roundDouble(node->thisValuesNotInBook.scoreStdev, 1000000);
      nodeData["maxP"] = node->thisValuesNotInBook.maxPolicy;
      nodeData["w"] = roundDouble(node->thisValuesNotInBook.weight, 1000);
      nodeData["v"] = node->thisValuesNotInBook.visits;
      nodeData["cEx"] = node->canExpand;
      // Don't record reexpansion prohibition, since this can change with the user's multi-ply search settings
      // nodeData["cRx"] = node->canReExpand;
    }
    else {
      nodeData["hash"] = node->hash.toString();
      nodeData["pla"] = PlayerIO::playerToString(node->pla);
      nodeData["symmetries"] = node->symmetries;
      nodeData["winLossValue"] = node->thisValuesNotInBook.winLossValue;
      nodeData["scoreMean"] = node->thisValuesNotInBook.scoreMean;
      nodeData["sharpScoreMean"] = node->thisValuesNotInBook.sharpScoreMeanRaw;
      nodeData["winLossError"] = node->thisValuesNotInBook.winLossError;
      nodeData["scoreError"] = node->thisValuesNotInBook.scoreError;
      nodeData["scoreStdev"] = node->thisValuesNotInBook.scoreStdev;
      nodeData["maxPolicy"] = node->thisValuesNotInBook.maxPolicy;
      nodeData["weight"] = node->thisValuesNotInBook.weight;
      nodeData["visits"] = node->thisValuesNotInBook.visits;
      nodeData["canExpand"] = node->canExpand;
    }

    if(bookVersion >= 2) {
      nodeData["mvs"] = json::array();
      for(auto& locAndBookMove: node->moves) {
        json moveData;
        moveData["m"] = Location::toString(locAndBookMove.second.move,initialBoard);
        moveData["sym"] = locAndBookMove.second.symmetryToAlign;
        moveData["id"] = getIdx(locAndBookMove.second.hash);
        moveData["rP"] = locAndBookMove.second.rawPolicy;
        nodeData["mvs"].push_back(moveData);
      }
    }
    else {
      nodeData["moves"] = json::array();
      for(auto& locAndBookMove: node->moves) {
        json moveData;
        moveData["move"] = Location::toString(locAndBookMove.second.move,initialBoard);
        moveData["symmetryToAlign"] = locAndBookMove.second.symmetryToAlign;
        moveData["hash"] = locAndBookMove.second.hash.toString();
        moveData["rawPolicy"] = locAndBookMove.second.rawPolicy;
        nodeData["moves"].push_back(moveData);
      }
    }

    if(bookVersion >= 2) {
      nodeData["par"] = json::array();
      for(auto& hashAndLoc: node->parents) {
        json parentData;
        parentData["id"] = getIdx(hashAndLoc.first);
        parentData["loc"] = Location::toString(hashAndLoc.second,initialBoard);
        nodeData["par"].push_back(parentData);
      }
    }
    else {
      nodeData["parents"] = json::array();
      for(auto& hashAndLoc: node->parents) {
        json parentData;
        parentData["hash"] = hashAndLoc.first.toString();
        parentData["loc"] = Location::toString(hashAndLoc.second,initialBoard);
        nodeData["parents"].push_back(parentData);
      }
    }

    out << nodeData << "\n";
  }
  out << std::flush;
  out.close();

  // Just in case, avoid any possible racing for file system
  std::this_thread::sleep_for(std::chrono::duration<double>(1));
  FileUtils::rename(tmpFileName,fileName);
}

Book* Book::loadFromFile(const std::string& fileName) {
  std::ifstream in;
  FileUtils::open(in, fileName);
  std::string line;
  Book* ret = NULL;
  try {
    getline(in,line);
    if(!in)
      throw IOError("Could not load initial metadata line");
    auto assertContains = [&](const json& data, const string& key) {
      if(!data.contains(key))
        throw IOError("Could not parse json or find expected key " + key);
    };

    std::unique_ptr<Book> book;
    {
      json params = json::parse(line);
      assertContains(params,"version");
      int bookVersion = params["version"].get<int>();
      if(bookVersion != 1 && bookVersion != 2)
        throw IOError("Unsupported book version: " + Global::intToString(bookVersion));

      assertContains(params,"initialBoard");
      Board initialBoard = Board::ofJson(params["initialBoard"]);
      assertContains(params,"initialRules");
      Rules initialRules = Rules::parseRules(params["initialRules"].dump());
      Player initialPla = PlayerIO::parsePlayer(params["initialPla"].get<string>());
      int repBound = params["repBound"].get<int>();

      BookParams bookParams;
      bookParams.errorFactor = params["errorFactor"].get<double>();
      bookParams.costPerMove = params["costPerMove"].get<double>();
      bookParams.costPerUCBWinLossLoss = params["costPerUCBWinLossLoss"].get<double>();
      bookParams.costPerUCBWinLossLossPow3 = params.contains("costPerUCBWinLossLossPow3") ? params["costPerUCBWinLossLossPow3"].get<double>() : 0.0;
      bookParams.costPerUCBWinLossLossPow7 = params.contains("costPerUCBWinLossLossPow7") ? params["costPerUCBWinLossLossPow7"].get<double>() : 0.0;
      bookParams.costPerUCBScoreLoss = params["costPerUCBScoreLoss"].get<double>();
      bookParams.costPerLogPolicy = params["costPerLogPolicy"].get<double>();
      bookParams.costPerMovesExpanded = params["costPerMovesExpanded"].get<double>();
      bookParams.costPerSquaredMovesExpanded = params["costPerSquaredMovesExpanded"].get<double>();
      bookParams.costWhenPassFavored = params["costWhenPassFavored"].get<double>();
      bookParams.bonusPerWinLossError = params["bonusPerWinLossError"].get<double>();
      bookParams.bonusPerScoreError = params.contains("bonusPerScoreError") ? params["bonusPerScoreError"].get<double>() : 0.0;
      bookParams.bonusPerSharpScoreDiscrepancy = params.contains("bonusPerSharpScoreDiscrepancy") ? params["bonusPerSharpScoreDiscrepancy"].get<double>() : 0.0;
      bookParams.bonusPerExcessUnexpandedPolicy = params.contains("bonusPerExcessUnexpandedPolicy") ? params["bonusPerExcessUnexpandedPolicy"].get<double>() : 0.0;
      bookParams.bonusPerUnexpandedBestWinLoss = params.contains("bonusPerUnexpandedBestWinLoss") ? params["bonusPerUnexpandedBestWinLoss"].get<double>() : 0.0;
      bookParams.bonusForWLPV1 = params.contains("bonusForWLPV1") ? params["bonusForWLPV1"].get<double>() : 0.0;
      bookParams.bonusForWLPV2 = params.contains("bonusForWLPV2") ? params["bonusForWLPV2"].get<double>() : 0.0;
      bookParams.bonusForWLPVFinalProp = params.contains("bonusForWLPVFinalProp") ? params["bonusForWLPVFinalProp"].get<double>() : 0.5;
      bookParams.bonusForBiggestWLCost = params.contains("bonusForBiggestWLCost") ? params["bonusForBiggestWLCost"].get<double>() : 0.0;
      bookParams.bonusBehindInVisitsScale = params.contains("bonusBehindInVisitsScale") ? params["bonusBehindInVisitsScale"].get<double>() : 0.0;
      bookParams.scoreLossCap = params["scoreLossCap"].get<double>();
      bookParams.earlyBookCostReductionFactor = params.contains("earlyBookCostReductionFactor") ? params["earlyBookCostReductionFactor"].get<double>() : 0.0;
      bookParams.earlyBookCostReductionLambda = params.contains("earlyBookCostReductionLambda") ? params["earlyBookCostReductionLambda"].get<double>() : 0.0;
      bookParams.utilityPerScore = params["utilityPerScore"].get<double>();
      bookParams.policyBoostSoftUtilityScale = params["policyBoostSoftUtilityScale"].get<double>();
      bookParams.utilityPerPolicyForSorting = params["utilityPerPolicyForSorting"].get<double>();
      bookParams.adjustedVisitsWLScale = params.contains("adjustedVisitsWLScale") ? params["adjustedVisitsWLScale"].get<double>() : 0.05;
      bookParams.maxVisitsForReExpansion = params.contains("maxVisitsForReExpansion") ? params["maxVisitsForReExpansion"].get<double>() : 0.0;
      bookParams.visitsScale = params.contains("visitsScale") ? params["visitsScale"].get<double>() : 1.0;
      bookParams.visitsScaleLeaves = params.contains("visitsScaleLeaves") ? params["visitsScaleLeaves"].get<double>() : 1.0;
      bookParams.sharpScoreOutlierCap = params.contains("sharpScoreOutlierCap") ? params["sharpScoreOutlierCap"].get<double>() : 10000.0;

      book = std::make_unique<Book>(
        bookVersion,
        initialBoard,
        initialRules,
        initialPla,
        repBound,
        bookParams
      );

      int initialSymmetry = params["initialSymmetry"].get<int>();
      if(book->initialSymmetry != initialSymmetry)
        throw IOError("Inconsistent initial symmetry with initialization");
    }

    std::vector<BookHash> hashDict;
    if(book->bookVersion >= 2) {
      getline(in,line);
      if(!in)
        throw IOError("Could not load book hash list size");
      size_t hashDictSize = Global::stringToUInt64(line);
      for(size_t nodeIdx = 0; nodeIdx<hashDictSize; nodeIdx++) {
        getline(in,line);
        if(!in)
          throw IOError("Book hash list ended early");
        // Strip extra whitespace (e.g. carriage returns from windows)
        hashDict.push_back(BookHash::ofString(Global::trim(line)));
      }
    }

    while(getline(in,line)) {
      if(line.size() <= 0)
        break;
      json nodeData = json::parse(line);

      BookHash hash;
      Player pla;
      vector<int> symmetries;

      if(book->bookVersion >= 2) {
        size_t nodeIdx = nodeData["id"].get<size_t>();
        hash = hashDict[nodeIdx];
        pla = PlayerIO::parsePlayer(nodeData["pla"].get<string>());
        symmetries = nodeData["syms"].get<vector<int>>();
      }
      else {
        hash = BookHash::ofString(nodeData["hash"].get<string>());
        pla = PlayerIO::parsePlayer(nodeData["pla"].get<string>());
        symmetries = nodeData["symmetries"].get<vector<int>>();
      }

      BookNode* node = book->get(hash);
      if(node != NULL) {
        if(node->hash != hash) throw IOError("Inconsistent hash for root node with initialization");
        if(node->pla != pla) throw IOError("Inconsistent pla for root node with initialization");
        if(node->symmetries != symmetries) throw IOError("Inconsistent symmetries for root node with initialization");
      }
      else {
        node = new BookNode(hash, book.get(), pla, symmetries);
        book->add(hash, node);
      }

      if(book->bookVersion >= 2) {
        node->thisValuesNotInBook.winLossValue = nodeData["wl"].get<double>();
        node->thisValuesNotInBook.scoreMean = nodeData["sM"].get<double>();
        node->thisValuesNotInBook.sharpScoreMeanRaw = nodeData["ssM"].get<double>();
        node->thisValuesNotInBook.sharpScoreMeanClamped = node->thisValuesNotInBook.sharpScoreMeanRaw;
        node->thisValuesNotInBook.winLossError = nodeData["wlE"].get<double>();
        node->thisValuesNotInBook.scoreError = nodeData["sE"].get<double>();
        node->thisValuesNotInBook.scoreStdev = nodeData["sStd"].get<double>();
        node->thisValuesNotInBook.maxPolicy = nodeData["maxP"].get<double>();
        node->thisValuesNotInBook.weight = nodeData["w"].get<double>();
        node->thisValuesNotInBook.visits = nodeData["v"].get<double>();
      }
      else {
        node->thisValuesNotInBook.winLossValue = nodeData["winLossValue"].get<double>();
        node->thisValuesNotInBook.scoreMean = nodeData["scoreMean"].get<double>();
        node->thisValuesNotInBook.sharpScoreMeanRaw = nodeData["sharpScoreMean"].get<double>();
        node->thisValuesNotInBook.sharpScoreMeanClamped = node->thisValuesNotInBook.sharpScoreMeanRaw;
        node->thisValuesNotInBook.winLossError = nodeData["winLossError"].get<double>();
        node->thisValuesNotInBook.scoreError = nodeData["scoreError"].get<double>();
        node->thisValuesNotInBook.scoreStdev = nodeData["scoreStdev"].get<double>();
        node->thisValuesNotInBook.maxPolicy = nodeData["maxPolicy"].get<double>();
        node->thisValuesNotInBook.weight = nodeData["weight"].get<double>();
        node->thisValuesNotInBook.visits = nodeData["visits"].get<double>();
      }

      // Older versions had some buggy conditions under which they would set this incorrectly, and nodes would be stuck not expanding.
      // So force it true on old versions.
      // Parameter changes can alter whether a node is expandable or not (e.g. whether it's considered done given all its visits)
      // But it's not much harm to set a node as non-expandable, since except for error cases this only happens when a node
      // has explored all possible legal moves, in which case we might as well not use this node either for reexpansions,
      // if reexpansions can only target child nodes, that's fine.
      if(book->bookVersion >= 2)
        node->canExpand = nodeData["cEx"].get<bool>();
      else
        node->canExpand = true;

      // Don't record reexpansion prohibition, since this can change with the user's multi-ply search settings
      // node->canReExpand = nodeData.find("cRx") != nodeData.end() ? nodeData["cRx"].get<bool>() : true;

      if(book->bookVersion >= 2) {
        for(json& moveData: nodeData["mvs"]) {
          BookMove move;
          move.move = Location::ofString(moveData["m"].get<string>(),book->initialBoard);
          move.symmetryToAlign = moveData["sym"].get<int>();
          size_t nodeIdx = moveData["id"].get<size_t>();
          move.hash = hashDict[nodeIdx];
          move.rawPolicy = moveData["rP"].get<double>();
          node->moves[move.move] = move;
        }
      }
      else {
        for(json& moveData: nodeData["moves"]) {
          BookMove move;
          move.move = Location::ofString(moveData["move"].get<string>(),book->initialBoard);
          move.symmetryToAlign = moveData["symmetryToAlign"].get<int>();
          move.hash = BookHash::ofString(moveData["hash"].get<string>());
          move.rawPolicy = moveData["rawPolicy"].get<double>();
          node->moves[move.move] = move;
        }
      }

      if(book->bookVersion >= 2) {
        for(json& parentData: nodeData["par"]) {
          size_t nodeIdx = parentData["id"].get<size_t>();
          BookHash parentHash = hashDict[nodeIdx];
          Loc loc = Location::ofString(parentData["loc"].get<string>(),book->initialBoard);
          node->parents.push_back(std::make_pair(parentHash,loc));
        }
      }
      else {
        for(json& parentData: nodeData["parents"]) {
          BookHash parentHash = BookHash::ofString(parentData["hash"].get<string>());
          Loc loc = Location::ofString(parentData["loc"].get<string>(),book->initialBoard);
          node->parents.push_back(std::make_pair(parentHash,loc));
        }
      }
    }
    book->recomputeEverything();
    ret = book.release();
  }
  catch(const std::exception& e) {
    throw IOError("When parsing book file " + fileName + ": " + e.what() + "\nFurthest line read was:\n" + line.substr(0,10000));
  }
  return ret;
}


