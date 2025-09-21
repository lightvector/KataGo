#include "../tests/tests.h"
#include "../tests/testdotsutils.h"

#include "../game/graphhash.h"
#include "../program/playutils.h"

using namespace std;
using namespace TestCommon;

void checkDotsField(const string& description, const string& input, bool captureEmptyBases, bool freeCapturedDots, const std::function<void(BoardWithMoveRecords&)>& check) {
  cout << "  " << description << endl;

  auto moveRecords = vector<Board::MoveRecord>();

  Board initialBoard = parseDotsField(input, false, captureEmptyBases, freeCapturedDots, {});

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

void Tests::runDotsBoardHistoryGroundingTests() {
  {
    const Board board = parseDotsFieldDefault(R"(
....
.xo.
.ox.
....
)");
    const auto boardHistory = BoardHistory(board);
    float whiteScoreAfterGrounding;
    testAssert(!boardHistory.doesGroundingWinGame(board, P_BLACK, whiteScoreAfterGrounding));
    testAssert(!boardHistory.doesGroundingWinGame(board, P_WHITE, whiteScoreAfterGrounding));
    testAssert(0.0f == whiteScoreAfterGrounding);
    testAssert(0.0f == boardHistory.whiteScoreIfGroundingAlive(board));
  }

  {
    Board board = parseDotsFieldDefault(R"(
.....
..o..
.oxo.
.....
)");
    board.playMoveAssumeLegal(Location::getLoc(2, 3, board.x_size), P_WHITE);
    testAssert(1 == board.numBlackCaptures);
    const auto boardHistory = BoardHistory(board);
    float whiteScoreAfterGrounding;
    testAssert(!boardHistory.doesGroundingWinGame(board, P_BLACK, whiteScoreAfterGrounding));
    testAssert(boardHistory.doesGroundingWinGame(board, P_WHITE, whiteScoreAfterGrounding));
    testAssert(1.0f == whiteScoreAfterGrounding);
    testAssert(1.0f == boardHistory.whiteScoreIfGroundingAlive(board));
  }

  {
    Board board = parseDotsFieldDefault(R"(
.....
..x..
.xox.
.....
)");
    board.playMoveAssumeLegal(Location::getLoc(2, 3, board.x_size), P_BLACK);
    testAssert(1 == board.numWhiteCaptures);
    const auto boardHistory = BoardHistory(board);
    float whiteScoreAfterGrounding;
    testAssert(boardHistory.doesGroundingWinGame(board, P_BLACK, whiteScoreAfterGrounding));
    testAssert(!boardHistory.doesGroundingWinGame(board, P_WHITE, whiteScoreAfterGrounding));
    testAssert(-1.0f == whiteScoreAfterGrounding);
    testAssert(-1.0f == boardHistory.whiteScoreIfGroundingAlive(board));
  }

  {
    Board board = parseDotsFieldDefault(R"(
.....
..x..
.xox.
.....
.....
)");
    board.playMoveAssumeLegal(Location::getLoc(2, 3, board.x_size), P_BLACK);
    testAssert(1 == board.numWhiteCaptures);
    const auto boardHistory = BoardHistory(board);
    float whiteScoreAfterGrounding;
    testAssert(!boardHistory.doesGroundingWinGame(board, P_BLACK, whiteScoreAfterGrounding));
    testAssert(!boardHistory.doesGroundingWinGame(board, P_WHITE, whiteScoreAfterGrounding));
    testAssert(0.0f == whiteScoreAfterGrounding);
    testAssert(0.0f == boardHistory.whiteScoreIfGroundingAlive(board));
  }
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