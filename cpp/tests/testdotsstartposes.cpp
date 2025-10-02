#include <chrono>

#include "../tests/tests.h"
#include "testdotsutils.h"

#include "../game/graphhash.h"
#include "../program/playutils.h"

using namespace std;
using namespace std::chrono;
using namespace TestCommon;

void writeToSgfAndCheckStartPosFromSgfProp(const int startPos, const bool startPosIsRandom, const Board& board) {
  std::ostringstream sgfStringStream;
  const BoardHistory boardHistory(board, P_BLACK, board.rules, 0);
  WriteSgf::writeSgf(sgfStringStream, "black", "white", boardHistory, {});
  const string sgfString = sgfStringStream.str();
  cout << ";  Sgf: " << sgfString << endl;

  const auto deserializedSgf = Sgf::parse(sgfString);
  const Rules newRules = deserializedSgf->getRulesOrFail();
  testAssert(startPos == newRules.startPos);
  testAssert(startPosIsRandom == newRules.startPosIsRandom);
}

void checkStartPos(const string& description, const int startPos, const bool startPosIsRandom, const int x_size, const int y_size, const string& expectedBoard = "", const vector<XYMove>& extraMoves = {}) {
  cout << "  " << description << " (" << to_string(x_size) << "," << to_string(y_size) << ")";

  auto board = Board(x_size, y_size, Rules(startPos, startPosIsRandom, Rules::DEFAULT_DOTS.multiStoneSuicideLegal, Rules::DEFAULT_DOTS.dotsCaptureEmptyBases, Rules::DEFAULT_DOTS.dotsFreeCapturedDots));
  board.setStartPos(DOTS_RANDOM);
  for (const XYMove& extraMove : extraMoves) {
    board.playMoveAssumeLegal(Location::getLoc(extraMove.x, extraMove.y, board.x_size), extraMove.player);
  }

  std::ostringstream oss;
  Board::printBoard(oss, board, Board::NULL_LOC, nullptr);

  if (!expectedBoard.empty()) {
    expect(description.c_str(), oss, expectedBoard);
  }

  writeToSgfAndCheckStartPosFromSgfProp(startPos, startPosIsRandom, board);
}

void checkStartPosRecognition(const string& description, const int expectedStartPos, const int startPosIsRandom, const string& inputBoard) {
  const Board board = parseDotsField(inputBoard, startPosIsRandom, Rules::DEFAULT_DOTS.multiStoneSuicideLegal, Rules::DEFAULT_DOTS.dotsCaptureEmptyBases, Rules::DEFAULT_DOTS.dotsFreeCapturedDots, {});

  cout << "  " << description << " (" << to_string(board.x_size) << "," << to_string(board.y_size) << ")";

  writeToSgfAndCheckStartPosFromSgfProp(expectedStartPos, startPosIsRandom, board);
}

void checkGenerationAndRecognition(const int startPos, const int startPosIsRandom) {
  const auto generatedMoves = Rules::generateStartPos(startPos, startPosIsRandom ? &DOTS_RANDOM : nullptr, 39, 32);
  bool actualRandomized;
  testAssert(startPos == Rules::tryRecognizeStartPos(generatedMoves, 39, 32, actualRandomized));
  // We can't reliably check in case of randomization is not detected because random generator can
  // generate static poses in rare cases.
  if (actualRandomized) {
    testAssert(startPosIsRandom);
  }
}

void Tests::runDotsStartPosTests() {
  cout << "Running dots start pos tests" << endl;

  Rand rand("runDotsStartPosTests");

  checkStartPos("Cross on minimal size", Rules::START_POS_CROSS, false, 2, 2, R"(
HASH: EC100709447890A116AFC8952423E3DD
   1  2
 2 X  O
 1 O  X
)");

  checkStartPos("Extra dots with cross (for instance, a handicap game)", Rules::START_POS_CROSS, false, 4, 4, R"(
HASH: A130436FBD93FF473AB4F3B84DD304DB
   1  2  3  4
 4 .  .  .  .
 3 .  X  O  .
 2 .  O  X  .
 1 .  .  X  .
)", {XYMove(2, 3, P_BLACK)});

  checkStartPosRecognition("Not enough dots for cross", Rules::START_POS_CUSTOM, false, R"(
....
.xo.
.o..
....
)");

  checkStartPosRecognition("Reversed cross should be recognized as random", Rules::START_POS_CROSS, true, R"(
....
.ox.
.xo.
....
)");

  checkStartPos("Cross on odd size", Rules::START_POS_CROSS, false, 3, 3, R"(
HASH: 3B29F9557D2712A5BC982D218680927D
   1  2  3
 3 .  X  O
 2 .  O  X
 1 .  .  .
)");

  checkStartPos("Cross on standard size", Rules::START_POS_CROSS, false, 39, 32, R"(
HASH: 516E1ABBA0D6B69A0B3D17C9E34E52F7
   1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
32 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
31 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
30 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
29 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
28 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
27 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
26 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
25 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
24 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
23 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
22 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
21 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
20 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
19 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
18 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
17 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X  O  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
16 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  O  X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
15 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
14 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
13 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
12 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
11 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
10 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 9 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 8 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 7 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 6 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 5 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 4 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 3 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 2 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 1 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
)");

  checkStartPos("Double cross on minimal size", Rules::START_POS_CROSS_2, false, 4, 2, R"(
HASH: 43FD769739F2AA27A8A1DAB1F4278229
   1  2  3  4
 2 X  O  O  X
 1 O  X  X  O
)");

  checkStartPos("Double cross on odd size", Rules::START_POS_CROSS_2, false, 5, 3, R"(
HASH: AAA969B8135294A3D1ADAA07BEA9A987
   1  2  3  4  5
 3 .  X  O  O  X
 2 .  O  X  X  O
 1 .  .  .  .  .
)");

  checkStartPos("Double cross", Rules::START_POS_CROSS_2, false, 6, 4, R"(
HASH: D599CEA39B1378D29883145CA4C016FC
   1  2  3  4  5  6
 4 .  .  .  .  .  .
 3 .  X  O  O  X  .
 2 .  O  X  X  O  .
 1 .  .  .  .  .  .
)");

  checkStartPos("Double cross", Rules::START_POS_CROSS_2, false, 7, 4, R"(
HASH: 249F175819EA8FDE47F8676E655A06DE
   1  2  3  4  5  6  7
 4 .  .  .  .  .  .  .
 3 .  .  X  O  O  X  .
 2 .  .  O  X  X  O  .
 1 .  .  .  .  .  .  .
)");

    checkStartPos("Double cross on standard size", Rules::START_POS_CROSS_2, false, 39, 32, R"(
HASH: CAD72FD407955308CEFCBD7A9B14B35B
   1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
32 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
31 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
30 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
29 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
28 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
27 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
26 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
25 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
24 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
23 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
22 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
21 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
20 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
19 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
18 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
17 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  X  O  O  X  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
16 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  O  X  X  O  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
15 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
14 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
13 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
12 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
11 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
10 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 9 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 8 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 7 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 6 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 5 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 4 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 3 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 2 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 1 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
)");

  checkStartPos("Quadruple cross", Rules::START_POS_CROSS_4, false, 5, 5, R"(
HASH: 0C2DD637AAE5FA7E1469BF5829BE922B
   1  2  3  4  5
 5 X  O  .  X  O
 4 O  X  .  O  X
 3 .  .  .  .  .
 2 X  O  .  X  O
 1 O  X  .  O  X
)");

  checkStartPos("Quadruple cross", Rules::START_POS_CROSS_4, false, 7, 7, R"(
HASH: 89CBCA85E94AF1B6C376E6BCBC443A48
   1  2  3  4  5  6  7
 7 .  .  .  .  .  .  .
 6 .  X  O  .  X  O  .
 5 .  O  X  .  O  X  .
 4 .  .  .  .  .  .  .
 3 .  X  O  .  X  O  .
 2 .  O  X  .  O  X  .
 1 .  .  .  .  .  .  .
)");

  checkStartPos("Quadruple cross", Rules::START_POS_CROSS_4, false, 8, 8, R"(
HASH: 445D50D7A61C47CE2730BBB97A2B3C96
   1  2  3  4  5  6  7  8
 8 .  .  .  .  .  .  .  .
 7 .  X  O  .  .  X  O  .
 6 .  O  X  .  .  O  X  .
 5 .  .  .  .  .  .  .  .
 4 .  .  .  .  .  .  .  .
 3 .  X  O  .  .  X  O  .
 2 .  O  X  .  .  O  X  .
 1 .  .  .  .  .  .  .  .
)");

  checkStartPos("Quadruple cross on standard size", Rules::START_POS_CROSS_4, false, 39, 32, R"(
HASH: 2A9AE7F967F17B42D9B9CB45B735E9C6
   1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
32 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
31 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
30 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
29 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
28 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
27 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
26 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
25 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
24 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
23 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
22 .  .  .  .  .  .  .  .  .  .  .  X  O  .  .  .  .  .  .  .  .  .  .  .  .  .  X  O  .  .  .  .  .  .  .  .  .  .  .
21 .  .  .  .  .  .  .  .  .  .  .  O  X  .  .  .  .  .  .  .  .  .  .  .  .  .  O  X  .  .  .  .  .  .  .  .  .  .  .
20 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
19 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
18 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
17 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
16 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
15 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
14 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
13 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
12 .  .  .  .  .  .  .  .  .  .  .  X  O  .  .  .  .  .  .  .  .  .  .  .  .  .  X  O  .  .  .  .  .  .  .  .  .  .  .
11 .  .  .  .  .  .  .  .  .  .  .  O  X  .  .  .  .  .  .  .  .  .  .  .  .  .  O  X  .  .  .  .  .  .  .  .  .  .  .
10 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 9 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 8 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 7 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 6 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 5 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 4 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 3 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 2 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
 1 .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
)");

  checkStartPos("Random quadruple cross on standard size", Rules::START_POS_CROSS_4, true, 39, 32);

  checkGenerationAndRecognition(Rules::START_POS_CROSS_4, false);
  checkGenerationAndRecognition(Rules::START_POS_CROSS_4, true);
}