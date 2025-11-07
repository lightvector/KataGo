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

  checkDotsFieldDefault("Simple",
  R"(
.....
.xxo.
.....
)", [](const BoardWithMoveRecords& boardWithMoveRecords) {
      boardWithMoveRecords.playGroundingMove(P_BLACK);
      testAssert(2 == boardWithMoveRecords.board.numBlackCaptures);
      boardWithMoveRecords.undo();

      boardWithMoveRecords.playGroundingMove(P_WHITE);
      testAssert(1 == boardWithMoveRecords.board.numWhiteCaptures);
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
    boardWithMoveRecords.undo();

    boardWithMoveRecords.playGroundingMove(P_WHITE);
    testAssert(0 == boardWithMoveRecords.board.numWhiteCaptures);
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
  boardWithMoveRecords.undo();

  boardWithMoveRecords.playGroundingMove(P_WHITE);
  testAssert(0 == boardWithMoveRecords.board.numBlackCaptures);
  testAssert(3 == boardWithMoveRecords.board.numWhiteCaptures);
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



void runDotsStressTestsInternal(int x_size, int y_size, int gamesCount, float groundingAfterCoef, float groundingProb, float komi, bool suicideAllowed, bool checkRollback) {
  // TODO: add tests with grounding
  cout << "  Random games" <<  endl;
  cout << "    Check rollback: " << boolalpha << checkRollback << endl;
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

  Rules rules = Rules(false);
  Board initialBoard = Board(x_size, y_size, rules);

  int tryGroundingAfterMove = groundingAfterCoef * initialBoard.numLegalMoves;

  vector<Loc> randomMoves = vector<Loc>();
  randomMoves.reserve(initialBoard.numLegalMoves);

  for(int y = 0; y < initialBoard.y_size; y++) {
    for(int x = 0; x < initialBoard.x_size; x++) {
      randomMoves.push_back(Location::getLoc(x, y, initialBoard.x_size));
    }
  }

  int movesCount = 0;
  int blackWinsCount = 0;
  int whiteWinsCount = 0;
  int drawsCount = 0;

  auto moveRecords = vector<Board::MoveRecord>();

  for (int n = 0; n < gamesCount; n++) {
    rand.shuffle(randomMoves);

    auto board = Board(initialBoard.x_size, initialBoard.y_size, rules);

    Player pla = P_BLACK;
    for (Loc loc : randomMoves) {
      if (board.isLegal(loc, pla, suicideAllowed, false)) {
        Board::MoveRecord moveRecord = board.playMoveRecorded(loc, pla);
        movesCount++;
        if (checkRollback) {
          moveRecords.push_back(moveRecord);
        }
        pla = getOpp(pla);
      }
    }

    /*if (suicideAllowed) {
      testAssert(0 == board.numLegalMoves);
    }*/

    if (float whiteScore = board.numBlackCaptures - board.numWhiteCaptures + komi; whiteScore > 0.0f) {
      whiteWinsCount++;
    } else if (whiteScore < 0) {
      blackWinsCount++;
    } else {
      drawsCount++;
    }

    if (checkRollback) {
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
}

void Tests::runDotsStressTests() {
  cout << "Running dots stress tests" << endl;

  cout << "  Max territory" << endl;
  Board board = Board(39, 32, Rules::DEFAULT_DOTS);
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Player pla = y == 0 || y == board.y_size - 1 || x == 0 || x == board.x_size - 1 ? P_BLACK : P_WHITE;
      board.playMoveAssumeLegal(Location::getLoc(x, y, board.x_size), pla);
    }
  }
  testAssert((board.x_size - 2) * (board.y_size - 2) == board.numWhiteCaptures);
  testAssert(0 == board.numLegalMoves);

  //runDotsStressTestsInternal(39, 32, 100000, 0.8f, 0.01f, 0.0f, true, false);
  runDotsStressTestsInternal(39, 32, 10000, 0.8f, 0.01f, 0.0f, true, true);
}