
#include "../book/book.h"

#include <fstream>
#include "../core/makedir.h"
#include "../neuralnet/nninputs.h"

using std::vector;
using std::map;
using std::cout;
using std::endl;
using std::string;

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

static Hash128 getStateHash(const BoardHistory& hist) {
  const Board& board = hist.getRecentBoard(0);
  Player nextPlayer = hist.presumedNextMovePla;
  double drawEquivalentWinsForWhite = 0.5;
  Hash128 hash = BoardHistory::getSituationRulesAndKoHash(board, hist, nextPlayer, drawEquivalentWinsForWhite);

  // Fold in whether a pass ends this phase
  bool passEndsPhase = hist.passWouldEndPhase(board,nextPlayer);
  if(passEndsPhase)
    hash ^= Board::ZOBRIST_PASS_ENDS_PHASE;
  // Fold in whether the game is over or not
  if(hist.isGameFinished)
    hash ^= Board::ZOBRIST_GAME_IS_OVER;

  // Fold in consecutive pass count. Probably usually redundant with history tracking. Use some standard LCG constants.
  static constexpr uint64_t CONSECPASS_MULT0 = 2862933555777941757ULL;
  static constexpr uint64_t CONSECPASS_MULT1 = 3202034522624059733ULL;
  hash.hash0 += CONSECPASS_MULT0 * (uint64_t)hist.consecutiveEndingPasses;
  hash.hash1 += CONSECPASS_MULT1 * (uint64_t)hist.consecutiveEndingPasses;
  return hash;
}

void BookHash::getHashAndSymmetry(const BoardHistory& hist, int repBound, BookHash& hashRet, int& symmetryToAlignRet, vector<int>& symmetriesRet) {
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
      Hash128 nextHash = boardsBySym[symmetry].pos_hash ^ Board::ZOBRIST_PLAYER_HASH[movePla];
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
  for(int symmetry = 0; symmetry < numSymmetries; symmetry++)
    hashes[symmetry] = BookHash(accums[symmetry] ^ getExtraPosHash(boardsBySym[symmetry]), getStateHash(histsBySym[symmetry]));

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

// -----------------------------------------------------------------------------------------------------------

BookMove::BookMove()
  :move(Board::NULL_LOC),
   symmetryToAlign(0),
   hash(),
   rawPolicy(0.0)
{}

BookMove::BookMove(Loc mv, int s, BookHash h, double rp)
  :move(mv),
   symmetryToAlign(s),
   hash(h),
   rawPolicy(rp)
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
   moves(),
   parents(),
   recursiveValues(),
   minCostFromRoot(0),
   thisNodeExpansionCost(0)
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


const RecursiveBookValues& SymBookNode::recursiveValues() {
  assert(node != nullptr);
  return node->recursiveValues;
}
const RecursiveBookValues& ConstSymBookNode::recursiveValues() {
  assert(node != nullptr);
  return node->recursiveValues;
}

double SymBookNode::minCostFromRoot() {
  assert(node != nullptr);
  return node->minCostFromRoot;
}
double ConstSymBookNode::minCostFromRoot() {
  assert(node != nullptr);
  return node->minCostFromRoot;
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

SymBookNode SymBookNode::playAndAddMove(Board& board, BoardHistory& hist, Loc move, double rawPolicy) {
  assert(node != nullptr);
  assert(!isMoveInBook(move));

  if(!hist.isLegal(board,move,node->pla))
    return SymBookNode(nullptr);

  int xSize = node->book->initialBoard.x_size;
  int ySize = node->book->initialBoard.y_size;

  // Transform the move into the space of the node
  Loc symMove = SymmetryHelpers::getSymLoc(move,xSize,ySize,invSymmetryOfNode);

  // Find the symmetry for move that prefers the upper right corner if possible.
  // Maximize x first, then minimize y next
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
  BookHash::getHashAndSymmetry(hist, node->book->repBound, childHash, symmetryToAlignToChild, symmetriesOfChild);

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
  return ConstSymBookNode(*this).getBoardHistoryReachingHere(ret,moveHistoryRet);
}
bool ConstSymBookNode::getBoardHistoryReachingHere(BoardHistory& ret, vector<Loc>& moveHistoryRet) {
  assert(node != nullptr);
  const Book* book = node->book;
  vector<const BookNode*> pathFromRoot;
  vector<Loc> movesFromRoot;
  bool suc = node->book->reverseDepthFirstSearchWithMoves(
    node,
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


Book::Book(
  const Board& b,
  Rules r,
  Player p,
  int rb,
  double sf,
  double cpm,
  double cpucbwl,
  double cpucbsl,
  double cplp,
  double ups
) : initialBoard(b),
    initialRules(r),
    initialPla(p),
    repBound(rb),
    errorFactor(sf),
    costPerMove(cpm),
    costPerUCBWinLossLoss(cpucbwl),
    costPerUCBScoreLoss(cpucbsl),
    costPerLogPolicy(cplp),
    utilityPerScore(ups),
    initialSymmetry(0),
    root(nullptr),
    nodes(),
    nodeIdxMapsByHash(nullptr)
{
  nodeIdxMapsByHash = new map<BookHash,int64_t>[NUM_HASH_BUCKETS];

  BookHash rootHash;
  int symmetryToAlign;
  vector<int> rootSymmetries;

  int initialEncorePhase = 0;
  BoardHistory initialHist(initialBoard, initialPla, initialRules, initialEncorePhase);
  BookHash::getHashAndSymmetry(initialHist, repBound, rootHash, symmetryToAlign, rootSymmetries);

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

double Book::getErrorFactor() const { return errorFactor; }
void Book::setErrorFactor(double d) { errorFactor = d; }
double Book::getCostPerMove() const { return costPerMove; }
void Book::setCostPerMove(double d) { costPerMove = d; }
double Book::getCostPerUCBWinLossLoss() const { return costPerUCBWinLossLoss; }
void Book::setCostPerUCBWinLossLoss(double d) { costPerUCBWinLossLoss = d; }
double Book::getCostPerUCBScoreLoss() const { return costPerUCBScoreLoss; }
void Book::setCostPerUCBScoreLoss(double d) { costPerUCBScoreLoss = d; }
double Book::getCostPerLogPolicy() const { return costPerLogPolicy; }
void Book::setCostPerLogPolicy(double d) { costPerLogPolicy = d; }
double Book::getUtilityPerScore() const { return utilityPerScore; }
void Book::setUtilityPerScore(double d) { utilityPerScore = d; }


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

BookNode* Book::get(BookHash hash) {
  map<BookHash,int64_t>& nodeIdxMap = nodeIdxMapsByHash[hash.stateHash.hash0 % NUM_HASH_BUCKETS];
  auto iter = nodeIdxMap.find(hash);
  if(iter == nodeIdxMap.end())
    return nullptr;
  return nodes[iter->second];
}
const BookNode* Book::get(BookHash hash) const {
  const map<BookHash,int64_t>& nodeIdxMap = nodeIdxMapsByHash[hash.stateHash.hash0 % NUM_HASH_BUCKETS];
  auto iter = nodeIdxMap.find(hash);
  if(iter == nodeIdxMap.end())
    return nullptr;
  return nodes[iter->second];
}

bool Book::add(BookHash hash, BookNode* node) {
  map<BookHash,int64_t>& nodeIdxMap = nodeIdxMapsByHash[hash.stateHash.hash0 % NUM_HASH_BUCKETS];
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
bool Book::reverseDepthFirstSearchWithMoves(
  const BookNode* initialNode,
  const std::function<Book::DFSAction(const vector<const BookNode*>&, const vector<Loc>&)>& f
) const {
  vector<const BookNode*> stack;
  vector<Loc> moveStack;
  vector<size_t> nextParentIdxToTry;
  std::set<BookHash> visitedHashes;
  stack.push_back(initialNode);
  Loc nullLoc = Board::NULL_LOC; //Workaround for c++14 wonkiness fixed in c++17
  moveStack.push_back(nullLoc);
  nextParentIdxToTry.push_back(0);
  visitedHashes.insert(initialNode->hash);

  while(true) {
    // Handle new found node
    DFSAction action = f(stack,moveStack);
    if(action == DFSAction::abort)
      return true;
    else if(action == DFSAction::skip) {
      nextParentIdxToTry.back() = std::numeric_limits<size_t>::max();
    }

    // Attempt to find the next node
    while(true) {
      // Try walk to next parent
      const BookNode* node = stack.back();
      size_t nextParentIdx = nextParentIdxToTry.back();
      if(nextParentIdx < node->parents.size()) {
        BookHash nextParentHash = node->parents[nextParentIdx].first;
        Loc nextParentLoc = node->parents[nextParentIdx].second;
        nextParentIdxToTry.back() += 1;
        if(!contains(visitedHashes, nextParentHash)) {
          const BookNode* nextParent = get(nextParentHash);
          stack.push_back(nextParent);
          moveStack.push_back(nextParentLoc);
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
  vector<map<Loc,BookMove>::iterator> nextChildToTry;
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
      map<Loc,BookMove>::iterator iter = nextChildToTry.back();

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

void Book::recomputeNodeValues(BookNode* node) {
  double winLossValue;
  double scoreMean;
  double lead;
  double winLossLCB;
  double scoreLCB;
  double winLossUCB;
  double scoreUCB;
  double weight = 0.0;
  int64_t visits = 0;

  {
    const BookValues& values = node->thisValuesNotInBook;
    winLossValue = values.winLossValue;
    scoreMean = values.scoreMean;
    lead = values.lead;
    winLossLCB = values.winLossValue - errorFactor * values.winLossError;
    scoreLCB = values.scoreMean - errorFactor * values.scoreError;
    winLossUCB = values.winLossValue + errorFactor * values.winLossError;
    scoreUCB = values.scoreMean + errorFactor * values.scoreError;
    weight += values.weight;
    visits += values.visits;
  }

  for(auto iter = node->moves.begin(); iter != node->moves.end(); ++iter) {
    const BookNode* child = get(iter->second.hash);
    const RecursiveBookValues& values = child->recursiveValues;
    if(node->pla == P_WHITE) {
      winLossValue = std::max(winLossValue, values.winLossValue);
      scoreMean = std::max(scoreMean, values.scoreMean);
      lead = std::max(lead, values.lead);
      winLossLCB = std::max(winLossLCB, values.winLossLCB);
      scoreLCB = std::max(scoreLCB, values.scoreLCB);
      winLossUCB = std::max(winLossUCB, values.winLossUCB);
      scoreUCB = std::max(scoreUCB, values.scoreUCB);
      weight += values.weight;
      visits += values.visits;
    }
    else {
      winLossValue = std::min(winLossValue, values.winLossValue);
      scoreMean = std::min(scoreMean, values.scoreMean);
      lead = std::min(lead, values.lead);
      winLossLCB = std::min(winLossLCB, values.winLossLCB);
      scoreLCB = std::min(scoreLCB, values.scoreLCB);
      winLossUCB = std::min(winLossUCB, values.winLossUCB);
      scoreUCB = std::min(scoreUCB, values.scoreUCB);
      weight += values.weight;
      visits += values.visits;
    }
  }

  RecursiveBookValues& values = node->recursiveValues;
  values.winLossValue = winLossValue;
  values.scoreMean = scoreMean;
  values.lead = lead;
  values.winLossLCB = winLossLCB;
  values.scoreLCB = scoreLCB;
  values.winLossUCB = winLossUCB;
  values.scoreUCB = scoreUCB;
  values.weight = weight;
  values.visits = visits;
}

void Book::recomputeNodeCost(BookNode* node) {
  if(node == root)
    node->minCostFromRoot = 0.0;
  else {
    double minCost = 1e100;
    for(std::pair<BookHash,Loc>& parentInfo: node->parents) {
      const BookNode* parent = get(parentInfo.first);
      double ucbWinLossLoss =
        (parent->pla == P_WHITE) ?
        parent->recursiveValues.winLossUCB - node->recursiveValues.winLossUCB :
        node->recursiveValues.winLossLCB - parent->recursiveValues.winLossLCB;
      double ucbScoreLoss =
        (parent->pla == P_WHITE) ?
        parent->recursiveValues.scoreUCB - node->recursiveValues.scoreUCB :
        node->recursiveValues.scoreLCB - parent->recursiveValues.scoreLCB;
      auto parentLocAndBookMove = parent->moves.find(parentInfo.second);
      assert(parentLocAndBookMove != parent->moves.end());
      double rawPolicy = parentLocAndBookMove->second.rawPolicy;

      double cost =
        parent->minCostFromRoot
        + costPerMove
        + ucbWinLossLoss * costPerUCBWinLossLoss
        + ucbScoreLoss * costPerUCBScoreLoss
        + (-log(rawPolicy + 1e-100) * costPerLogPolicy);

      if(cost < minCost)
        minCost = cost;
    }
    node->minCostFromRoot = minCost;
  }

  if(!node->canExpand) {
    node->thisNodeExpansionCost = 1e100;
  }
  else {
    double ucbWinLossLoss =
      (node->pla == P_WHITE) ?
      (node->recursiveValues.winLossUCB - (node->thisValuesNotInBook.winLossValue + errorFactor * node->thisValuesNotInBook.winLossError)) :
      ((node->thisValuesNotInBook.winLossValue - errorFactor * node->thisValuesNotInBook.winLossError) - node->recursiveValues.winLossLCB);
    double ucbScoreLoss =
      (node->pla == P_WHITE) ?
      (node->recursiveValues.scoreUCB - (node->thisValuesNotInBook.scoreMean + errorFactor * node->thisValuesNotInBook.scoreError)) :
      ((node->thisValuesNotInBook.scoreMean - errorFactor * node->thisValuesNotInBook.scoreError) - node->recursiveValues.scoreLCB);
    double rawPolicy = node->thisValuesNotInBook.maxPolicy;

    node->thisNodeExpansionCost =
      costPerMove
      + ucbWinLossLoss * costPerUCBWinLossLoss
      + ucbScoreLoss * costPerUCBScoreLoss
      + (-log(rawPolicy + 1e-100) * costPerLogPolicy);
  }
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


void Book::exportToHtmlDir(const string& dirName, Logger& logger) {
  MakeDir::make(dirName);
  const char* hexChars = "0123456789ABCDEF";
  for(int i = 0; i<16; i++) {
    for(int j = 0; j<16; j++) {
      MakeDir::make(dirName + "/" + hexChars[i] + hexChars[j]);
    }
  }
  MakeDir::make(dirName + "/root");
  {
    std::ofstream out(dirName + "/book.js");
    out << Book::BOOK_JS;
    out.close();
  }
  {
    std::ofstream out(dirName + "/book.css");
    out << Book::BOOK_CSS;
    out.close();
  }

  auto getFilePath = [&](BookNode* node, int symmetry, vector<int>& handledSymmetries, bool relative) {
    vector<int> equivalentSymmetries;
    for(int nodeSymmetry: node->symmetries) {
      int s = SymmetryHelpers::compose(symmetry,nodeSymmetry);
      equivalentSymmetries.push_back(s);
      handledSymmetries.push_back(s);
    }
    std::sort(equivalentSymmetries.begin(),equivalentSymmetries.end());
    string path = relative ? "" : dirName + "/";
    if(node == root)
      path += "root/root";
    else {
      string hashStr = node->hash.toString();
      assert(hashStr.size() > 2);
      path += hashStr.substr(0,2) + "/" + node->hash.toString();
    }
    path += "_";
    for(int equivalentSymmetry: equivalentSymmetries)
      path += Global::intToString(equivalentSymmetry);
    path += ".html";
    return path;
  };

  std::function<void(BookNode*)> f = [&](BookNode* node) {
    vector<int> handledSymmetries;
    int numSymmetries = (initialBoard.x_size != initialBoard.y_size) ? SymmetryHelpers::NUM_SYMMETRIES_WITHOUT_TRANSPOSE : SymmetryHelpers::NUM_SYMMETRIES;
    for(int symmetry = 0; symmetry < numSymmetries; symmetry++) {
      if(contains(handledSymmetries,symmetry))
        return;
      string filePath = getFilePath(node, symmetry, handledSymmetries, false);
      string html = HTML_TEMPLATE;
      auto replace = [&](const string& key, const string& replacement) {
        size_t pos = html.find(key);
        assert(pos != string::npos);
        html.replace(pos, key.size(), replacement);
      };

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
      Board board = hist.getRecentBoard(0);

      string dataVarsStr;
      dataVarsStr += "const nextPlayer = " + Global::intToString(node->pla) + ";\n";
      dataVarsStr += "const boardSizeX = " + Global::intToString(board.x_size) + ";\n";
      dataVarsStr += "const boardSizeY = " + Global::intToString(board.y_size) + ";\n";
      dataVarsStr += "const board = [";
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          dataVarsStr += Global::intToString(board.colors[loc]) + ",";
        }
      }
      dataVarsStr += "];\n";
      dataVarsStr += "const links = [";
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          SymBookNode child = symNode.follow(loc);
          if(child.isNull())
            dataVarsStr += "'',";
          else {
            vector<int> handledSymmetriesDummy; // Not actually used, just needed for arg
            string childPath = getFilePath(child.node, child.symmetryOfNode, handledSymmetriesDummy, true);
            dataVarsStr += "'../" + childPath + "',";
          }
        }
      }
      dataVarsStr += "];\n";

      vector<std::pair<BookMove,RecursiveBookValues>> movesAndValues;
      vector<BookMove> uniqueMovesInBook = symNode.getUniqueMovesInBook();
      for(BookMove& bookMove: uniqueMovesInBook) {
        SymBookNode child = symNode.follow(bookMove.move);
        movesAndValues.push_back(std::make_pair(bookMove, child.node->recursiveValues));
      }
      std::sort(
        movesAndValues.begin(),movesAndValues.end(),
        [&](const std::pair<BookMove,RecursiveBookValues>& mv0,
            const std::pair<BookMove,RecursiveBookValues>& mv1) {
          return node->pla == P_WHITE ?
            (mv0.second.winLossValue + mv0.second.scoreMean * utilityPerScore >
             mv1.second.winLossValue + mv1.second.scoreMean * utilityPerScore)
            :
            (mv0.second.winLossValue + mv0.second.scoreMean * utilityPerScore <
             mv1.second.winLossValue + mv1.second.scoreMean * utilityPerScore)
            ;
        }
      );

      vector<int> equivalentSymmetries = symNode.getSymmetries();
      std::set<Loc> locsHandled;

      dataVarsStr += "const moves = [";
      for(auto& moveAndValue: movesAndValues) {
        dataVarsStr += "{";
        if(moveAndValue.first.move != Board::PASS_LOC) {
          dataVarsStr += "'xy':[";
          for(int s: equivalentSymmetries) {
            Loc symMove = SymmetryHelpers::getSymLoc(moveAndValue.first.move,initialBoard,s);
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
        dataVarsStr += "'move':'" + Location::toString(moveAndValue.first.move, initialBoard) + "',";
        dataVarsStr += "'policy':" + Global::doubleToString(moveAndValue.first.rawPolicy) + ",";
        dataVarsStr += "'winLossValue':" + Global::doubleToString(moveAndValue.second.winLossValue) + ",";
        dataVarsStr += "'winLossUCB':" + Global::doubleToString(moveAndValue.second.winLossUCB) + ",";
        dataVarsStr += "'winLossLCB':" + Global::doubleToString(moveAndValue.second.winLossLCB) + ",";
        dataVarsStr += "'scoreMean':" + Global::doubleToString(moveAndValue.second.scoreMean) + ",";
        dataVarsStr += "'lead':" + Global::doubleToString(moveAndValue.second.lead) + ",";
        dataVarsStr += "'scoreUCB':" + Global::doubleToString(moveAndValue.second.scoreUCB) + ",";
        dataVarsStr += "'scoreLCB':" + Global::doubleToString(moveAndValue.second.scoreLCB) + ",";
        dataVarsStr += "'weight':" + Global::doubleToString(moveAndValue.second.weight) + ",";
        dataVarsStr += "'visits':" + Global::doubleToString(moveAndValue.second.visits) + ",";
        dataVarsStr += "},";
      }
      {
        BookValues& values = node->thisValuesNotInBook;
        if(values.maxPolicy > 0.0) {
          double winLossValueUCB = values.winLossValue + errorFactor * values.winLossError;
          double winLossValueLCB = values.winLossValue - errorFactor * values.winLossError;
          double scoreUCB = values.scoreMean + errorFactor * values.scoreError;
          double scoreLCB = values.scoreMean - errorFactor * values.scoreError;

          dataVarsStr += "{";
          dataVarsStr += "'move':'other',";
          dataVarsStr += "'policy':" + Global::doubleToString(values.maxPolicy) + ",";
          dataVarsStr += "'winLossValue':" + Global::doubleToString(values.winLossValue) + ",";
          dataVarsStr += "'winLossUCB':" + Global::doubleToString(winLossValueUCB) + ",";
          dataVarsStr += "'winLossLCB':" + Global::doubleToString(winLossValueLCB) + ",";
          dataVarsStr += "'scoreMean':" + Global::doubleToString(values.scoreMean) + ",";
          dataVarsStr += "'lead':" + Global::doubleToString(values.lead) + ",";
          dataVarsStr += "'scoreUCB':" + Global::doubleToString(scoreUCB) + ",";
          dataVarsStr += "'scoreLCB':" + Global::doubleToString(scoreLCB) + ",";
          dataVarsStr += "'weight':" + Global::doubleToString(values.weight) + ",";
          dataVarsStr += "'visits':" + Global::doubleToString(values.visits) + ",";
          dataVarsStr += "},";
        }
      }
      dataVarsStr += "];\n";

      replace("$$DATA_VARS",dataVarsStr);

      std::ofstream out(filePath);
      out << html;
      out.close();
    }
  };
  iterateEntireBookPreOrder(f);
}
