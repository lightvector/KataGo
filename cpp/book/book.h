
#ifndef BOOK_BOOK_H_
#define BOOK_BOOK_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../game/boardhistory.h"

struct BookHash {
  Hash128 historyHash;
  Hash128 stateHash;

  BookHash();
  BookHash(Hash128 historyHash, Hash128 situationHash);

  bool operator<(const BookHash other) const;
  bool operator>(const BookHash other) const;
  bool operator<=(const BookHash other) const;
  bool operator>=(const BookHash other) const;
  bool operator==(const BookHash other) const;
  bool operator!=(const BookHash other) const;

  BookHash operator^(const BookHash other) const;
  BookHash operator|(const BookHash other) const;
  BookHash operator&(const BookHash other) const;
  BookHash& operator^=(const BookHash other);
  BookHash& operator|=(const BookHash other);
  BookHash& operator&=(const BookHash other);

  // Get book hash, and the symmetry to apply so that hist aligns with book nodes for its current position, (histspace -> nodespace)
  // and the list of symmetries such that book node is invariant under those symmetries.
  static void getHashAndSymmetry(const BoardHistory& hist, int repBound, BookHash& hashRet, int& symmetryToAlignRet, std::vector<int>& symmetriesRet);

  friend std::ostream& operator<<(std::ostream& out, const BookHash other);
  std::string toString() const;
};

struct BookMove {
  // If you play *move* from this position, you should go to the child node with this hash.
  Loc move;
  // The symmetry you need to apply to a board aligned with this node so that it aligns with the child node. (nodespace -> childnodespace)
  int symmetryToAlign;
  BookHash hash;

  // The policy prior of this move from the neural net.
  double rawPolicy;

  BookMove();
  BookMove(Loc move, int symmetryToAlign, BookHash hash, double rawPolicy);

  // Apply symmetry to this BookMove.
  BookMove getSymBookMove(int symmetry, int xSize, int ySize);
};

struct BookValues {
  // All values for which the player perspective matters are from white's perspective.
  double winLossValue = 0.0;
  double scoreMean = 0.0;
  double lead = 0.0;

  // Average short term error of nodes in the search. Probably correlated with the confidence
  // of this node, but not necessarily being a meaningful measure of it directly.
  double winLossError = 0.0;
  double scoreError = 0.0;

  double maxPolicy = 0.0;
  double weight = 0.0;
  int64_t visits = 0;
};
struct RecursiveBookValues {
  // Recursively computed via minimax
  double winLossValue = 0.0;
  double scoreMean = 0.0;
  double lead = 0.0;
  double winLossLCB = 0.0; // minimaxing winLossValue - winLossError * errorFactor
  double scoreLCB = 0.0;   // minimaxing scoreMean - scoreError * errorFactor
  double winLossUCB = 0.0; // minimaxing winLossValue + winLossError * errorFactor
  double scoreUCB = 0.0;   // minimaxing scoreMean + scoreError * errorFactor

  // Weighted by sum
  double weight = 0.0;
  int64_t visits = 0;
};

class SymBookNode;
class ConstSymBookNode;
class Book;

class BookNode {
  const BookHash hash;
  Book* const book;  // Non-owning pointer to the book that has this node.
  const Player pla;
  const std::vector<int> symmetries; // List of symmetries under which this position is invariant

  // -----------------------------------------------------------------------------------------------------------
  // Values that the book construction algorithm should set
  // -----------------------------------------------------------------------------------------------------------
  BookValues thisValuesNotInBook;  // Based on a search of this node alone, excluding all current book nodes.
  bool canExpand; // Set to false to never attempt to add more children to this node.

  // -----------------------------------------------------------------------------------------------------------
  // Values maintained by the book
  // -----------------------------------------------------------------------------------------------------------
  std::map<Loc,BookMove> moves;
  std::vector<std::pair<BookHash,Loc>> parents; // Locations are in the parent's alignment space

  RecursiveBookValues recursiveValues;  // Based on minimaxing over the book nodes
  double minCostFromRoot;  // Minimum sum of BookMove cost to reach this node from root
  double thisNodeExpansionCost;  //The cost for picking this node to expand further.

  BookNode(BookHash hash, Book* book, Player pla, const std::vector<int>& symmetries);
  ~BookNode();
  BookNode(const BookNode&) = delete;
  BookNode& operator=(const BookNode&) = delete;
  BookNode(BookNode&& other) = delete;
  BookNode& operator=(BookNode&& other) = delete;

  friend class ConstSymBookNode;
  friend class SymBookNode;
  friend class Book;
};

class SymBookNode {
  BookNode* node;
  int symmetryOfNode;  // The symmetry applied to node to get this SymBookNode. (nodespace -> symbooknodespace)
  int invSymmetryOfNode;  // The symmetry to apply to SymBookNode locs to convert them to node locs. (symbooknodespace -> nodespace)

  SymBookNode(std::nullptr_t);
  SymBookNode(BookNode* node, int symmetry);

 public:
  SymBookNode();
  SymBookNode(const SymBookNode& other) = default;
  SymBookNode& operator=(const SymBookNode& other) = default;

  bool isNull();
  SymBookNode applySymmetry(int symmetry);

  bool isMoveInBook(Loc move);
  std::vector<BookMove> getUniqueMovesInBook();

  Player pla();
  BookHash hash();
  BookValues& thisValuesNotInBook();
  bool& canExpand();
  const RecursiveBookValues& recursiveValues();
  double minCostFromRoot();

  SymBookNode follow(Loc move);

  // Returns NULL if the move is not legal OR the move is not in the book.
  SymBookNode playMove(Board& board, BoardHistory& hist, Loc move);
  // Returns NULL if the move is not legal. The move *must* not be in the book.
  SymBookNode playAndAddMove(Board& board, BoardHistory& hist, Loc move, double rawPolicy);

  // Returns false and does not modify ret if playing the moves in the book to reach here hit an illegal move.
  // This should only happen if a book was loaded from disk that is corrupted, or else only astronomically rarely on hash collisions.
  bool getBoardHistoryReachingHere(BoardHistory& ret);

  friend class ConstSymBookNode;
  friend class Book;
};

class ConstSymBookNode {
  const BookNode* node;
  int symmetryOfNode;  // The symmetry applied to node to get this ConstSymBookNode. (nodespace -> symbooknodespace)
  int invSymmetryOfNode;  // The symmetry to apply to ConstSymBookNode locs to convert them to node locs. (symbooknodespace -> nodespace)

  ConstSymBookNode(std::nullptr_t);
  ConstSymBookNode(const BookNode* node, int symmetry);

 public:
  ConstSymBookNode();
  ConstSymBookNode(const SymBookNode& other);
  ConstSymBookNode(const ConstSymBookNode& other) = default;
  ConstSymBookNode& operator=(const SymBookNode& other);
  ConstSymBookNode& operator=(const ConstSymBookNode& other) = default;

  bool isNull();
  ConstSymBookNode applySymmetry(int symmetry);

  Player pla();
  BookHash hash();
  bool isMoveInBook(Loc move);
  std::vector<BookMove> getUniqueMovesInBook();

  const BookValues& thisValuesNotInBook();
  bool canExpand();
  const RecursiveBookValues& recursiveValues();
  double minCostFromRoot();

  ConstSymBookNode follow(Loc move);
  // Returns NULL if the move is not legal OR the move is not in the book.
  ConstSymBookNode playMove(Board& board, BoardHistory& hist, Loc move);

  // Returns false and does not modify ret if playing the moves in the book to reach here hit an illegal move.
  // This should only happen if a book was loaded from disk that is corrupted, or else only astronomically rarely on hash collisions.
  bool getBoardHistoryReachingHere(BoardHistory& ret);

  friend class Book;
};

// Book object for storing and minimaxing stats based on deep search, supports full tranposing symmetry handling.
// In the case where initialBoard is non-square, transpose symmetries are disallowed everywhere, so
// every node in the book always agrees on these. The tradeoff is that books don't work for
// rectangular boards in the wrong orientation, it's up to the user to manually do that.
class Book {
  static constexpr size_t NUM_HASH_BUCKETS = 2048;

 public:
  const Board initialBoard;
  const Rules initialRules;
  const Player initialPla;
  const int repBound;

 private:
  double errorFactor;
  double costPerMove;
  double costPerUCBWinLossLoss;
  double costPerUCBScoreLoss;
  double costPerLogPolicy;
  double utilityPerScore; //Currently just for sorting

  int initialSymmetry; // The symmetry that needs to be applied to initialBoard to align it with rootNode. (initialspace -> rootnodespace)
  BookNode* root;
  std::vector<BookNode*> nodes;
  std::map<BookHash,int64_t>* nodeIdxMapsByHash;
 public:
  Book(
    const Board& board,
    Rules rules,
    Player initialPla,
    int repBound,
    double errorFactor,
    double costPerMove,
    double costPerUCBWinLossLoss,
    double costPerUCBScoreLoss,
    double costPerLogPolicy,
    double utilityPerScore
  );
  ~Book();

  Book(const Book&) = delete;
  Book& operator=(const Book&) = delete;
  Book(Book&& other) = delete;
  Book& operator=(Book&& other) = delete;

  BoardHistory getInitialHist() const;
  // Get the initial history, with symmetry applied
  BoardHistory getInitialHist(int symmetry) const;

  double getErrorFactor() const;
  void setErrorFactor(double d);
  double getCostPerMove() const;
  void setCostPerMove(double d);
  double getCostPerUCBWinLossLoss() const;
  void setCostPerUCBWinLossLoss(double d);
  double getCostPerUCBScoreLoss() const;
  void setCostPerUCBScoreLoss(double d);
  double getCostPerLogPolicy() const;
  void setCostPerLogPolicy(double d);
  double getUtilityPerScore() const;
  void setUtilityPerScore(double d);

  // Gets the root node, in the orientation of the initial board.
  SymBookNode getRoot();
  ConstSymBookNode getRoot() const;

  // Walk down the book following hist and get the final node.
  // Returns a null SymBookNode if hist goes off the end of the book.
  SymBookNode get(const BoardHistory& hist);
  ConstSymBookNode get(const BoardHistory& hist) const;

  void recompute(const std::vector<SymBookNode>& newAndChangedNodes);
  void recomputeEverything();
  std::vector<SymBookNode> getNextNToExpand(int n);

  void exportToHtmlDir(const std::string& dirName);

 private:
  BookNode* get(BookHash hash);
  const BookNode* get(BookHash hash) const;
  bool add(BookHash hash, BookNode* node);

  enum class DFSAction {
    recurse, // Recursively search this node
    skip,    // Don't recurse into this node but keep searching
    abort    // Abort the entire search
  };

  bool reverseDepthFirstSearchWithMoves(
    const BookNode* initialNode,
    const std::function<DFSAction(const std::vector<const BookNode*>&, const std::vector<Loc>&)>& f
  ) const;

  bool reverseDepthFirstSearchWithPostF(
    BookNode* initialNode,
    const std::function<DFSAction(BookNode*)>& f,
    const std::function<void(BookNode*)>& postF
  );

  void iterateDirtyNodesPostOrder(
    const std::set<BookHash>& dirtyNodes,
    bool allDirty,
    const std::function<void(BookNode* node)>& f
  );

  void iterateEntireBookPreOrder(
    const std::function<void(BookNode*)>& f
  );

  void recomputeNodeValues(BookNode* node);
  void recomputeNodeCost(BookNode* node);

  friend class BookNode;
  friend class SymBookNode;
  friend class ConstSymBookNode;
};


#endif // BOOK_BOOK_H_
