#include <chrono>

#include "../tests/tests.h"
#include "../tests/testdotsutils.h"

#include "../game/graphhash.h"
#include "../program/playutils.h"

using namespace std;
using namespace std::chrono;
using namespace TestCommon;

void checkDotsField(const string& description, const string& input, bool captureEmptyBases, bool freeCapturedDots, const std::function<void(BoardWithMoveRecords&)>& check) {
  cout << "  " << description << endl;

  auto moveRecords = vector<Board::MoveRecord>();

  Board initialBoard = parseDotsField(input, captureEmptyBases, freeCapturedDots, {});

  Board board = Board(initialBoard);

  BoardWithMoveRecords boardWithMoveRecords = BoardWithMoveRecords(board, moveRecords);
  check(boardWithMoveRecords);

  while (!moveRecords.empty()) {
    board.undo(moveRecords.back());
    moveRecords.pop_back();
  }
  testAssert(initialBoard.isEqualForTesting(board, true, true));
}

void checkDotsFieldDefault(const string& description, const string& input, const std::function<void(BoardWithMoveRecords&)>& check) {
  checkDotsField(description, input, Rules::DEFAULT_DOTS.dotsCaptureEmptyBases, Rules::DEFAULT_DOTS.dotsFreeCapturedDots, check);
}

void Tests::runDotsFieldTests() {
  cout << "Running dots basic tests: " << endl;

  checkDotsFieldDefault("Simple capturing",
    R"(
.x.
xox
...
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
  boardWithMoveRecords.playMove(1, 2, P_BLACK);
  testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
});

  checkDotsFieldDefault("Capturing with empty loc inside",
    R"(
.oo.
ox..
.oo.
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    testAssert(boardWithMoveRecords.isLegal(2, 1, P_BLACK));
    testAssert(boardWithMoveRecords.isLegal(2, 1, P_WHITE));

    boardWithMoveRecords.playMove(3, 1, P_WHITE);
    testAssert(1 == boardWithMoveRecords.board.numBlackCaptures);
    testAssert(!boardWithMoveRecords.isLegal(2, 1, P_BLACK));
    testAssert(!boardWithMoveRecords.isLegal(2, 1, P_WHITE));
});

  checkDotsFieldDefault("Triple capture",
    R"(
.x.x.
xo.ox
.xox.
..x..
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
  boardWithMoveRecords.playMove(2, 1, P_BLACK);
  testAssert(3 == boardWithMoveRecords.board.numWhiteCaptures);
});

  checkDotsFieldDefault("Base inside base inside base",
    R"(
.xxxxxxx.
x..ooo..x
x.o.x.o.x
x.oxoxo.x
x.o...o.x
x..o.o..x
.xxx.xxx.
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
  boardWithMoveRecords.playMove(4, 4, P_BLACK);
  testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
  testAssert(0 == boardWithMoveRecords.board.numBlackCaptures);

  boardWithMoveRecords.playMove(4, 5, P_WHITE);
  testAssert(0 == boardWithMoveRecords.board.numWhiteCaptures);
  testAssert(4 == boardWithMoveRecords.board.numBlackCaptures);

  boardWithMoveRecords.playMove(4, 6, P_BLACK);
  testAssert(13 == boardWithMoveRecords.board.numWhiteCaptures);
  testAssert(0 == boardWithMoveRecords.board.numBlackCaptures);
});

  /*checkDotsField("Base inside base inside base don't free captured dots",
  R"(
.xxxxxxxxx..
x..oooooo.x.
x.o.xx...o.x
x.oxo.xo.o.x
x.o.x.o..o.x
x..o....ox.x
x...o.oo...x
.xxxx.xxxxx.
)", true, false, [](const BoardWithMoveRecords& boardWithMoveRecords) {
boardWithMoveRecords.playMove(5, 4, P_BLACK);
testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
testAssert(0 == boardWithMoveRecords.board.numBlackCaptures);

boardWithMoveRecords.playMove(5, 6, P_WHITE);
testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures); // Don't free the captured dot
testAssert(6 == boardWithMoveRecords.board.numBlackCaptures); // Ignore owned color dots

boardWithMoveRecords.playMove(5, 7, P_BLACK);
testAssert(21 == boardWithMoveRecords.board.numWhiteCaptures); // Don't count already counted dots
testAssert(6 == boardWithMoveRecords.board.numBlackCaptures);  // Don't free the captured dot
});*/

  checkDotsFieldDefault("Empty bases and suicide",
    R"(
.x..o.
x.xo.o
.x..o.
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    // Suicide move is not capture
    testAssert(!boardWithMoveRecords.wouldBeCapture(1, 1, P_WHITE));
    testAssert(!boardWithMoveRecords.wouldBeCapture(1, 1, P_BLACK));
    testAssert(!boardWithMoveRecords.wouldBeCapture(4, 1, P_WHITE));
    testAssert(!boardWithMoveRecords.wouldBeCapture(4, 1, P_BLACK));

    testAssert(boardWithMoveRecords.isSuicide(1, 1, P_WHITE));
    testAssert(!boardWithMoveRecords.isSuicide(1, 1, P_BLACK));
    boardWithMoveRecords.playMove(1, 1, P_WHITE);
    testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);

    testAssert(boardWithMoveRecords.isSuicide(4, 1, P_BLACK));
    testAssert(!boardWithMoveRecords.isSuicide(4, 1, P_WHITE));
    boardWithMoveRecords.playMove(4, 1, P_BLACK);
    testAssert(1 == boardWithMoveRecords.board.numBlackCaptures);
});

  checkDotsField("Empty bases when they are allowed",
  R"(
.x..o.
x.xo.o
......
)", true, true, [](const BoardWithMoveRecords& boardWithMoveRecords) {
  boardWithMoveRecords.playMove(1, 2, P_BLACK);
  boardWithMoveRecords.playMove(4, 2, P_WHITE);

  // Suicide is not possible in this mode
  testAssert(!boardWithMoveRecords.isSuicide(1, 1, P_WHITE));
  testAssert(!boardWithMoveRecords.isSuicide(1, 1, P_BLACK));
  testAssert(!boardWithMoveRecords.isSuicide(4, 1, P_BLACK));
  testAssert(!boardWithMoveRecords.isSuicide(4, 1, P_WHITE));

  testAssert(0 == boardWithMoveRecords.board.numWhiteCaptures);
  testAssert(0 == boardWithMoveRecords.board.numBlackCaptures);
});

  checkDotsFieldDefault("Capture wins suicide",
    R"(
.xo.
xo.o
.xo.
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    testAssert(!boardWithMoveRecords.isSuicide(2, 1, P_BLACK));
    boardWithMoveRecords.playMove(2, 1, P_BLACK);
    testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
});

  checkDotsFieldDefault("Single dot doesn't break searching inside empty base",
    R"(
.oooo.
o....o
o.o..o
o....o
.oooo.
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playMove(4, 2, P_BLACK);
    testAssert(1 == boardWithMoveRecords.board.numBlackCaptures);
  });

  checkDotsFieldDefault("Ignored already surrounded territory",
    R"(
..xxx...
.x...x..
x..x..x.
x.x.x..x
x..x..x.
.x...x..
..xxx...
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playMove(3, 3, P_WHITE);
    testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);

    boardWithMoveRecords.playMove(6, 3, P_WHITE);
    testAssert(2 == boardWithMoveRecords.board.numWhiteCaptures);
});

  checkDotsFieldDefault("Invalidation of empty base locations",
    R"(
.oox.
o..ox
.oox.
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playMove(2, 1, P_BLACK);
    boardWithMoveRecords.playMove(1, 1, P_BLACK);
    testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
  });

  checkDotsFieldDefault("Invalidation of empty base locations ignoring borders",
    R"(
..xxx....
.x...x...
x..x..xo.
x.x.x..xo
x..x..xo.
.x...x...
..xxx....
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playMove(6, 3, P_WHITE);
    testAssert(1 == boardWithMoveRecords.board.numBlackCaptures);

    boardWithMoveRecords.playMove(1, 3, P_WHITE);
    testAssert(1 == boardWithMoveRecords.board.numBlackCaptures);

    boardWithMoveRecords.playMove(3, 3, P_WHITE);
    testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
  });

  checkDotsFieldDefault("Dangling dots removing",
    R"(
.xx.xx.
x..xo.x
x.x.x.x
x..x..x
.x...x.
..x.x..
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
      boardWithMoveRecords.playMove(3, 5, P_BLACK);
      testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);

      testAssert(!boardWithMoveRecords.isLegal(3, 2, P_BLACK));
      testAssert(!boardWithMoveRecords.isLegal(3, 2, P_WHITE));
    });

  checkDotsFieldDefault("Recalculate square during dangling dots removing",
    R"(
.ooo..
o...o.
o.o..o
..xo.o
o.o..o
o...o.
.ooo..
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
      boardWithMoveRecords.playMove(1, 3, P_WHITE);
      testAssert(1 == boardWithMoveRecords.board.numBlackCaptures);

      boardWithMoveRecords.playMove(4, 3, P_BLACK);
      testAssert(2 == boardWithMoveRecords.board.numBlackCaptures);
    });

  checkDotsFieldDefault("Base sorting by size",
    R"(
..xxx..
.x...x.
x..x..x
x.xox.x
x.....x
.xx.xx.
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
      boardWithMoveRecords.playMove(3, 4, P_BLACK);
      testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);

      boardWithMoveRecords.playMove(4, 1, P_WHITE);
      testAssert(2 == boardWithMoveRecords.board.numWhiteCaptures);
    });

  checkDotsFieldDefault("Number of legal moves",
  R"(
....
....
....
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
testAssert(12 == boardWithMoveRecords.board.numLegalMoves);
});

  checkDotsFieldDefault("Game over because of absence of legal moves",
    R"(
xxxx
xo.x
xx.x
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playMove(2, 2, P_BLACK);
    testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
    testAssert(0 == boardWithMoveRecords.board.numLegalMoves);
  });
}

void Tests::runDotsGroundingTests() {
  cout << "Running dots grounding tests:" << endl;

    checkDotsFieldDefault("Grounding propagation",
R"(
.x..
o.o.
.x..
.xo.
..x.
....
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    testAssert(2 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(3 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    // Dot adjacent to WALL is already grounded
    testAssert(isGrounded(boardWithMoveRecords.getState(1, 0)));

    // Ignore enemy's dots
    testAssert(isGrounded(boardWithMoveRecords.getState(0, 1)));
    testAssert(!isGrounded(boardWithMoveRecords.getState(2, 1)));

    // Not yet grounded
    testAssert(!isGrounded(boardWithMoveRecords.getState(1, 2)));
    testAssert(!isGrounded(boardWithMoveRecords.getState(1, 3)));

    boardWithMoveRecords.playMove(1, 1, P_BLACK);

    testAssert(2 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(1 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    testAssert(isGrounded(boardWithMoveRecords.getState(1, 1)));

    // Check grounding propagation
    testAssert(isGrounded(boardWithMoveRecords.getState(1, 2)));
    testAssert(isGrounded(boardWithMoveRecords.getState(1, 3)));
    // Diagonal connection is not actual
    testAssert(!isGrounded(boardWithMoveRecords.getState(2, 4)));

    // Ignore enemy's dots
    testAssert(isGrounded(boardWithMoveRecords.getState(0, 1)));
    testAssert(!isGrounded(boardWithMoveRecords.getState(2, 1)));
    testAssert(!isGrounded(boardWithMoveRecords.getState(2, 3)));
}
  );

  checkDotsFieldDefault("Grounding propagation with empty base",
  R"(
..x..
.x.x.
.x.x.
..x..
.....
)",
  [](const BoardWithMoveRecords& boardWithMoveRecords) {
    testAssert(0 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(5 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    testAssert(!isGrounded(boardWithMoveRecords.getState(1, 2)));
    testAssert(!isGrounded(boardWithMoveRecords.getState(3, 2)));
    testAssert(!isGrounded(boardWithMoveRecords.getState(2, 3)));

    boardWithMoveRecords.playMove(2, 2, P_WHITE);

    testAssert(1 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(-1 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    testAssert(isGrounded(boardWithMoveRecords.getState(2, 2)));

    testAssert(isGrounded(boardWithMoveRecords.getState(1, 2)));
    testAssert(isGrounded(boardWithMoveRecords.getState(3, 2)));
    testAssert(isGrounded(boardWithMoveRecords.getState(2, 3)));
  });

  checkDotsFieldDefault("Grounding score with grounded base",
R"(
.x.
xox
...
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playMove(1, 2, P_BLACK);

    testAssert(1 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(-1 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);
}
);

  checkDotsFieldDefault("Grounding score with ungrounded base",
R"(
.....
..o..
.oxo.
.....
.....
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playMove(2, 3, P_WHITE);

    testAssert(4 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(1 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);
}
);

  checkDotsFieldDefault("Grounding score with grounded and ungrounded bases",
R"(
.x.....
xox.o..
...oxo.
.......
.......
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playMove(1, 2, P_BLACK);
    boardWithMoveRecords.playMove(4, 3, P_WHITE);

    testAssert(5 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(0 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);
}
);

  checkDotsFieldDefault("Grounding draw with ungrounded bases",
R"(
.........
..x...o..
.xox.oxo.
.........
.........
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playMove(2, 3, P_BLACK);
    boardWithMoveRecords.playMove(6, 3, P_WHITE);

    testAssert(1 == boardWithMoveRecords.board.numBlackCaptures);
    testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
    testAssert(5 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(5 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);
}
);


  checkDotsFieldDefault("Grounding of real and empty adjacent bases",
R"(
..x..
..x..
.xox.
.....
.x.x.
..x..
.....
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    testAssert(1 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(5 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    testAssert(!isGrounded(boardWithMoveRecords.getState(2, 2)));

    boardWithMoveRecords.playMove(2, 3, P_BLACK);
    testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);

    testAssert(1 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(2 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    // Real base becomes grounded
    testAssert(isGrounded(boardWithMoveRecords.getState(2, 2)));
    testAssert(isGrounded(boardWithMoveRecords.getState(2, 3)));

    // Grounding does not affect an empty location
    testAssert(!isGrounded(boardWithMoveRecords.getState(2, 4)));
    // Grounding does not affect empty surrounding
    testAssert(!isGrounded(boardWithMoveRecords.getState(3, 4)));
}
);

  checkDotsFieldDefault("Grounding of real base when it touches grounded",
R"(
..x..
..x..
.....
.xox.
..x..
.....
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    testAssert(1 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(3 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    testAssert(!isGrounded(boardWithMoveRecords.getState(2, 3)));
    testAssert(!isGrounded(boardWithMoveRecords.getState(2, 4)));

    boardWithMoveRecords.playMove(2, 2, P_BLACK);

    testAssert(1 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(-1 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    testAssert(isGrounded(boardWithMoveRecords.getState(2, 3)));
    testAssert(isGrounded(boardWithMoveRecords.getState(2, 4)));
}
);

  checkDotsFieldDefault("Base inside base inside base and grounding score",
R"(
.......
..ooo..
.o.x.o.
.oxoxo.
.o...o.
..o.o..
.......
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
  testAssert(12 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
  testAssert(3 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

  boardWithMoveRecords.playMove(3, 4, P_BLACK);

  testAssert(12 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
  testAssert(4 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

  boardWithMoveRecords.playMove(3, 5, P_WHITE);

  testAssert(13 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
  testAssert(4 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

  boardWithMoveRecords.playMove(3, 6, P_WHITE);

  testAssert(-4 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
  testAssert(4 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);
});

  checkDotsFieldDefault("Ground empty territory in case of dangling dots removing",
R"(
.........
..xxx....
.x....x..
.x.xx..x.
.x.x.x.x.
.x.xxx.x.
.x..xo.x.
..xxxxx..
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    testAssert(!isGrounded(boardWithMoveRecords.getState(4, 4)));

    boardWithMoveRecords.playMove(5, 1, P_BLACK);
    boardWithMoveRecords.playGroundingMove(P_BLACK);

    // TODO: it should be grounded, however currently it's not possible to set the state correctly due to limitation of grounding algorithm.
    //testAssert(isGrounded(boardWithMoveRecords.getState(4, 4)));
});

  checkDotsFieldDefault("Simple",
  R"(
.....
.xxo.
.....
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playGroundingMove(P_BLACK);

    testAssert(2 == boardWithMoveRecords.board.numBlackCaptures);

    testAssert(1 == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    testAssert(boardWithMoveRecords.getWhiteScore() == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    boardWithMoveRecords.undo();

    boardWithMoveRecords.playGroundingMove(P_WHITE);

    testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);

    testAssert(2 == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);
    testAssert(boardWithMoveRecords.getBlackScore() == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);

    boardWithMoveRecords.undo();
  }
);

  checkDotsFieldDefault("Draw",
R"(
.x...
.xxo.
...o.
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    boardWithMoveRecords.playGroundingMove(P_BLACK);
    testAssert(0 == boardWithMoveRecords.board.numBlackCaptures);
    testAssert(boardWithMoveRecords.getWhiteScore() == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);
    boardWithMoveRecords.undo();

    boardWithMoveRecords.playGroundingMove(P_WHITE);
    testAssert(0 == boardWithMoveRecords.board.numWhiteCaptures);
    testAssert(boardWithMoveRecords.getBlackScore() == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
    boardWithMoveRecords.undo();
}
);

  checkDotsFieldDefault("Bases",
R"(
.........
..xx...x.
.xo.x.xox
..x......
.........
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
  boardWithMoveRecords.playMove(3, 3, P_BLACK);
  boardWithMoveRecords.playMove(7, 3, P_BLACK);
  testAssert(2 == boardWithMoveRecords.board.numWhiteCaptures);

  boardWithMoveRecords.playGroundingMove(P_BLACK);
  testAssert(6 == boardWithMoveRecords.board.numBlackCaptures);
  testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
  testAssert(boardWithMoveRecords.getWhiteScore() == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);
}
);

  checkDotsFieldDefault("Multiple groups",
R"(
......
xxo..o
.ox...
x...oo
...o..
......
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
  boardWithMoveRecords.playGroundingMove(P_BLACK);
  testAssert(1 == boardWithMoveRecords.board.numBlackCaptures);
  testAssert(0 == boardWithMoveRecords.board.numWhiteCaptures);
  testAssert(boardWithMoveRecords.getWhiteScore() == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);
  boardWithMoveRecords.undo();

  boardWithMoveRecords.playGroundingMove(P_WHITE);
  testAssert(0 == boardWithMoveRecords.board.numBlackCaptures);
  testAssert(3 == boardWithMoveRecords.board.numWhiteCaptures);
  testAssert(boardWithMoveRecords.getBlackScore() == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);
  boardWithMoveRecords.undo();
}
);

  checkDotsFieldDefault("Invalidate empty territory",
R"(
......
..oo..
.o..o.
..oo..
......
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    Board board = boardWithMoveRecords.board;

    State state = boardWithMoveRecords.board.getState(Location::getLoc(2, 2, board.x_size));
    testAssert(C_WHITE == getEmptyTerritoryColor(state));

    state = boardWithMoveRecords.board.getState(Location::getLoc(3, 2, board.x_size));
    testAssert(C_WHITE == getEmptyTerritoryColor(state));

    boardWithMoveRecords.playGroundingMove(P_WHITE);
    testAssert(0 == boardWithMoveRecords.board.numBlackCaptures);
    testAssert(6 == boardWithMoveRecords.board.numWhiteCaptures);
    testAssert(boardWithMoveRecords.getBlackScore() == boardWithMoveRecords.board.blackScoreIfWhiteGrounds);

    state = boardWithMoveRecords.board.getState(Location::getLoc(2, 2, board.x_size));
    testAssert(C_EMPTY == getEmptyTerritoryColor(state));

    state = boardWithMoveRecords.board.getState(Location::getLoc(3, 2, board.x_size));
    testAssert(C_EMPTY == getEmptyTerritoryColor(state));
}
);

  checkDotsFieldDefault("Don't invalidate empty territory for strong connection",
R"(
.x.
x.x
.x.
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
    Board board = boardWithMoveRecords.board;

    boardWithMoveRecords.playGroundingMove(P_BLACK);
    testAssert(0 == boardWithMoveRecords.board.numBlackCaptures);
    testAssert(0 == boardWithMoveRecords.board.numWhiteCaptures);
    testAssert(boardWithMoveRecords.getWhiteScore() == boardWithMoveRecords.board.whiteScoreIfBlackGrounds);

    State state = boardWithMoveRecords.board.getState(Location::getLoc(1, 1, board.x_size));
    testAssert(C_BLACK == getEmptyTerritoryColor(state));

    state = boardWithMoveRecords.board.getState(Location::getLoc(0, 0, board.x_size));
    testAssert(C_EMPTY == getEmptyTerritoryColor(state));
}
);
}

void Tests::runDotsPosHashTests() {
   cout << "Running dots pos hash tests" << endl;

  {
    Board dotsFieldWithEmptyBase = parseDotsFieldDefault(R"(
.x.
x.x
.x.
)", { XYMove(1, 1, P_WHITE) });

    Board dotsFieldWithRealBase = parseDotsFieldDefault(R"(
.x.
xox
...
)", { XYMove(1, 2, P_BLACK) });

    testAssert(dotsFieldWithEmptyBase.pos_hash == dotsFieldWithRealBase.pos_hash);
  }

  {
    Board dotsFieldWithSurrounding = parseDotsFieldDefault(R"(
..xxxxxx..
.x......x.
x..x..o..x
x.xoxoxo.x
x........x
.x......x.
..xxx.xx..
)", { XYMove(3, 4, P_BLACK), XYMove(6, 4, P_WHITE), XYMove(5, 6, P_BLACK) });
    testAssert(5 == dotsFieldWithSurrounding.numWhiteCaptures);
    testAssert(0 == dotsFieldWithSurrounding.numBlackCaptures);

    Board dotsFieldWithErasedTerritory = parseDotsFieldDefault(R"(
..xxxxxx..
.xxxxxxxx.
xxxxxxxxxx
xxxxxxxxxx
xxxxxxxxxx
.xxxxxxxx.
..xxxxxx..
)");

    testAssert(dotsFieldWithErasedTerritory.pos_hash == dotsFieldWithSurrounding.pos_hash);
  }
}

string moveRecordsToSgf(const Board& initialBoard, const vector<Board::MoveRecord>& moveRecords) {
  Board boardCopy(initialBoard);
  BoardHistory boardHistory(boardCopy, P_BLACK, boardCopy.rules, 0);
  for (const Board::MoveRecord& moveRecord : moveRecords) {
    boardHistory.makeBoardMoveAssumeLegal(boardCopy, moveRecord.loc, moveRecord.pla, nullptr);
  }
  std::ostringstream sgfStringStream;
  WriteSgf::writeSgf(sgfStringStream, "blue", "red", boardHistory, {});
  return sgfStringStream.str();
}

/**
 * Calculates the grounding and result captures without using the grounding flag and incremental calculations.
 * It's used for testing to verify incremental grounding algorithms.
 */
void validateGrounding(
  const Board& boardBeforeGrounding,
  const Board& boardAfterGrounding,
  const Player pla,
  const vector<Board::MoveRecord>& moveRecords) {
  unordered_set<Loc> visited_locs;
  assert(pla == P_BLACK || pla == P_WHITE);

  int expectedNumBlackCaptures = 0;
  int expectedNumWhiteCaptures = 0;
  const Player opp = getOpp(pla);
  for (int y = 0; y < boardBeforeGrounding.y_size; y++) {
    for (int x = 0; x < boardBeforeGrounding.x_size; x++) {
      Loc loc = Location::getLoc(x, y, boardBeforeGrounding.x_size);
      const State state = boardBeforeGrounding.getState(Location::getLoc(x, y, boardBeforeGrounding.x_size));

      if (const Color activeColor = getActiveColor(state); activeColor == pla) {
        if (visited_locs.count(loc) > 0)
          continue;

        bool grounded = false;

        vector<Loc> walkStack;
        vector<Loc> baseLocs;
        walkStack.push_back(loc);

        // Find active territory and calculate its grounding state.
        while (!walkStack.empty()) {
          Loc curLoc = walkStack.back();
          walkStack.pop_back();

          if (const Color curActiveColor = getActiveColor(boardBeforeGrounding.getState(curLoc)); curActiveColor == pla) {
            if (visited_locs.count(curLoc) == 0) {
              visited_locs.insert(curLoc);
              baseLocs.push_back(curLoc);
              boardBeforeGrounding.forEachAdjacent(curLoc, [&](const Loc& adjLoc) {
                walkStack.push_back(adjLoc);
              });
            }
          } else if (curActiveColor == C_WALL) {
            grounded = true;
          }
        }

        for (const Loc& baseLoc : baseLocs) {
          const Color placedDotColor = getPlacedDotColor(boardBeforeGrounding.getState(baseLoc));

          if (!grounded) {
            // If the territory is not grounded, it becomes dead.
            // Freed dots don't count because they don't add a value to the opp score (assume they just become be placed).
            if (placedDotColor == pla) {
              if (pla == P_BLACK) {
                expectedNumBlackCaptures++;
              } else {
                expectedNumWhiteCaptures++;
              }
            }
          } else {
            State baseLocState = boardAfterGrounding.getState(baseLoc);
            // This check on placed dot color is redundant.
            // However, currently it's not possible to always ground empty locs in some rare cases due to limitations of incremental grounding algorithm.
            // Fortunately, they don't affect the resulting score.
            if (!isGrounded(baseLocState) && getPlacedDotColor(baseLocState) != C_EMPTY) {
              Global::fatalError("Loc (" + to_string(Location::getX(baseLoc, boardBeforeGrounding.x_size)) + "; " +
                 to_string(Location::getY(baseLoc, boardBeforeGrounding.x_size)) + ") " +
                " should be grounded. Sgf: " + moveRecordsToSgf(boardBeforeGrounding, moveRecords));
            }

            // If the territory is grounded, count dead dots of the opp player.
            if (placedDotColor == opp) {
              if (pla == P_BLACK) {
                expectedNumWhiteCaptures++;
              } else {
                expectedNumBlackCaptures++;
              }
            }
          }
        }
      } else if (activeColor == opp) { // In the case of opp active color, counts only captured dots
        if (getPlacedDotColor(state) == pla) {
          if (pla == P_BLACK) {
            expectedNumBlackCaptures++;
          } else {
            expectedNumWhiteCaptures++;
          }
        }
      }
    }
  }

  if (expectedNumBlackCaptures != boardAfterGrounding.numBlackCaptures || expectedNumWhiteCaptures != boardAfterGrounding.numWhiteCaptures) {
    Global::fatalError("expectedNumBlackCaptures (" + to_string(expectedNumBlackCaptures) + ")" +
      " == board.numBlackCaptures (" + to_string(boardAfterGrounding.numBlackCaptures) + ")" +
      " && expectedNumWhiteCaptures (" + to_string(expectedNumWhiteCaptures) + ")" +
      " == board.numWhiteCaptures (" + to_string(boardAfterGrounding.numWhiteCaptures) + ")" +
      " check is failed. Sgf: " + moveRecordsToSgf(boardBeforeGrounding, moveRecords));
  }
}

void runDotsStressTestsInternal(
  int x_size,
  int y_size,
  int gamesCount,
  bool dotsGame,
  int startPos,
  bool dotsCaptureEmptyBase,
  float komi,
  bool suicideAllowed,
  float groundingStartCoef,
  float groundingEndCoef,
  bool performExtraChecks
  ) {
  assert(groundingStartCoef >= 0 && groundingStartCoef <= 1);
  assert(groundingEndCoef >= 0 && groundingEndCoef <= 1);
  assert(groundingEndCoef >= groundingStartCoef);

  cout << "  Random games" <<  endl;
  cout << "    Game type: " << (dotsGame ? "Dots" : "Go") << endl;
  cout << "    Start position: " << Rules::writeStartPosRule(startPos) << endl;
  if (dotsGame) {
    cout << "    Capture empty bases: " << boolalpha << dotsCaptureEmptyBase << endl;
  }
  cout << "    Extra checks: " << boolalpha << performExtraChecks << endl;
#ifdef NDEBUG
  cout << "    Build: Release" << endl;
#else
  cout << "    Build: Debug" << endl;
#endif
  cout << "    Size: " << x_size << ":" << y_size << endl;
  cout << "    Komi: " << komi << endl;
  cout << "    Suicide: " << boolalpha << suicideAllowed << endl;
  cout << "    Games count: " << gamesCount << endl;

  const auto start = high_resolution_clock::now();

  Rand rand("runDotsStressTests");

  Rules rules = dotsGame ? Rules(dotsGame, startPos, dotsCaptureEmptyBase, Rules::DEFAULT_DOTS.dotsFreeCapturedDots) : Rules();
  auto initialBoard = Board(x_size, y_size, rules);

  vector<Loc> randomMoves = vector<Loc>();
  randomMoves.reserve(initialBoard.numLegalMoves);

  for(int y = 0; y < initialBoard.y_size; y++) {
    for(int x = 0; x < initialBoard.x_size; x++) {
      Loc loc = Location::getLoc(x, y, initialBoard.x_size);
      if (initialBoard.getColor(loc) == C_EMPTY) { // Filter out initial poses
        randomMoves.push_back(Location::getLoc(x, y, initialBoard.x_size));
      }
    }
  }

  assert(randomMoves.size() == initialBoard.numLegalMoves);

  int movesCount = 0;
  int blackWinsCount = 0;
  int whiteWinsCount = 0;
  int drawsCount = 0;
  int groundingCount = 0;

  auto moveRecords = vector<Board::MoveRecord>();

  for (int n = 0; n < gamesCount; n++) {
    rand.shuffle(randomMoves);
    moveRecords.clear();

    auto board = Board(initialBoard.x_size, initialBoard.y_size, rules);

    Loc lastLoc = Board::NULL_LOC;

    int tryGroundingAfterMove = (groundingStartCoef + rand.nextDouble() * (groundingEndCoef - groundingStartCoef)) * initialBoard.numLegalMoves;
    Player pla = P_BLACK;
    for(size_t index = 0; index < randomMoves.size(); index++) {
      lastLoc = moveRecords.size() >= tryGroundingAfterMove ? Board::PASS_LOC : randomMoves[index];

      if (board.isLegal(lastLoc, pla, suicideAllowed, false)) {
        Board::MoveRecord moveRecord = board.playMoveRecorded(lastLoc, pla);
        movesCount++;
        moveRecords.push_back(moveRecord);
        pla = getOpp(pla);
      }

      if (lastLoc == Board::PASS_LOC) {
        groundingCount++;
        int scoreDiff;
        int oppScoreIfGrounding;
        Player lastPla = moveRecords.back().pla;
        if (lastPla == P_BLACK) {
          scoreDiff = board.numBlackCaptures - board.numWhiteCaptures;
          oppScoreIfGrounding = board.whiteScoreIfBlackGrounds;
        } else {
          scoreDiff = board.numWhiteCaptures - board.numBlackCaptures;
          oppScoreIfGrounding = board.blackScoreIfWhiteGrounds;
        }
        if (scoreDiff != oppScoreIfGrounding) {
          Global::fatalError("scoreDiff (" + to_string(scoreDiff) + ") == oppScoreIfGrounding (" + to_string(oppScoreIfGrounding) + ") check is failed. " +
            "Sgf: " + moveRecordsToSgf(initialBoard, moveRecords));
        }
        if (performExtraChecks) {
          Board boardBeforeGrounding(board);
          boardBeforeGrounding.undo(moveRecords.back());
          validateGrounding(boardBeforeGrounding, board, lastPla, moveRecords);
        }
        break;
      }
    }

    if (dotsGame && suicideAllowed && lastLoc != Board::PASS_LOC) {
      testAssert(0 == board.numLegalMoves);
    }

    if (float whiteScore = board.numBlackCaptures - board.numWhiteCaptures + komi; whiteScore > 0.0f) {
      whiteWinsCount++;
    } else if (whiteScore < 0) {
      blackWinsCount++;
    } else {
      drawsCount++;
    }

    if (performExtraChecks) {
      while (!moveRecords.empty()) {
        board.undo(moveRecords.back());
        moveRecords.pop_back();
      }

      testAssert(initialBoard.isEqualForTesting(board, true, false));
    }
  }

  const auto end = high_resolution_clock::now();
  auto durationNs = duration_cast<nanoseconds>(end - start);

  cout.precision(4);
  cout << "    Elapsed time: " << duration_cast<milliseconds>(durationNs).count() << " ms" << endl;
  cout << "    Number of games per second: " << static_cast<int>(static_cast<double>(gamesCount) / durationNs.count() * 1000000000) << endl;
  cout << "    Number of moves per second: " << static_cast<int>(static_cast<double>(movesCount) / durationNs.count() * 1000000000) << endl;
  cout << "    Number of moves per game: " << static_cast<int>(static_cast<double>(movesCount) / gamesCount) << endl;
  cout << "    Time per game: " << static_cast<double>(durationNs.count()) / gamesCount / 1000000 << " ms" << endl;
  cout << "    Black wins: " << blackWinsCount << " (" << static_cast<double>(blackWinsCount) / gamesCount << ")" << endl;
  cout << "    White wins: " << whiteWinsCount << " (" << static_cast<double>(whiteWinsCount) / gamesCount << ")" << endl;
  cout << "    Draws: " << drawsCount << " (" << static_cast<double>(drawsCount) / gamesCount << ")" << endl;
  cout << "    Groundings: " << groundingCount << " (" << static_cast<double>(groundingCount) / gamesCount << ")" << endl;
}

void Tests::runDotsStressTests() {
  cout << "Running dots stress tests" << endl;

  cout << "  Max territory" << endl;
  Board board = Board(39, 32, Rules::DEFAULT_DOTS);
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      const Player pla = y == 0 || y == board.y_size - 1 || x == 0 || x == board.x_size - 1 ? P_BLACK : P_WHITE;
      board.playMoveAssumeLegal(Location::getLoc(x, y, board.x_size), pla);
    }
  }
  testAssert((board.x_size - 2) * (board.y_size - 2) == board.numWhiteCaptures);
  testAssert(0 == board.numLegalMoves);

  runDotsStressTestsInternal(39, 32, 3000, true, Rules::START_POS_CROSS, false, 0.0f, true, 0.8f, 1.0f, true);
  runDotsStressTestsInternal(39, 32, 3000, true, Rules::START_POS_CROSS_4, true, 0.5f, false, 0.8f, 1.0f, true);

  runDotsStressTestsInternal(39, 32, 100000, true, Rules::START_POS_CROSS, false, 0.0f, true, 0.8f, 1.0f, false);
}