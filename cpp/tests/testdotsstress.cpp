#include <chrono>

#include "../tests/tests.h"
#include "../tests/testdotsutils.h"

using namespace std;
using namespace std::chrono;
using namespace TestCommon;

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

void validateStatesAndCaptures(const Board& board, const vector<Board::MoveRecord>& moveRecords) {
  int expectedNumBlackCaptures = 0;
  int expectedNumWhiteCaptures = 0;
  int expectedPlacedDotsCount = -board.rules.getNumOfStartPosStones();

  for (int y = 0; y < board.y_size; y++) {
    for (int x = 0; x < board.x_size; x++) {
      const State state = board.getState(Location::getLoc(x, y, board.x_size));
      const Color activeColor = getActiveColor(state);
      const Color placedDotColor = getPlacedDotColor(state);
      const Color emptyTerritoryColor = getEmptyTerritoryColor(state);

      if (placedDotColor != C_EMPTY) {
        expectedPlacedDotsCount++;
      }

      if (activeColor == C_BLACK) {
        assert(C_EMPTY == emptyTerritoryColor);
        if (placedDotColor == C_WHITE) {
          expectedNumWhiteCaptures++;
        }
      } else if (activeColor == C_WHITE) {
        assert(C_EMPTY == emptyTerritoryColor);
        if (placedDotColor == C_BLACK) {
          expectedNumBlackCaptures++;
        }
      } else {
        assert(placedDotColor == C_EMPTY);
        //assert(!isTerritory(state));
      }
    }
  }

  const int actualPlacedDotsCount = moveRecords.size() - (moveRecords.back().loc == Board::PASS_LOC ? 1 : 0);
  assert(expectedPlacedDotsCount == actualPlacedDotsCount);
  assert(expectedNumBlackCaptures == board.numBlackCaptures);
  assert(expectedNumWhiteCaptures == board.numWhiteCaptures);
}

void runDotsStressTestsInternal(
  int x_size,
  int y_size,
  int gamesCount,
  bool dotsGame,
  int startPos,
  bool startPosIsRandom,
  bool dotsCaptureEmptyBase,
  float komi,
  bool suicideAllowed,
  float groundingStartCoef,
  float groundingEndCoef,
  bool performExtraChecks) {
  assert(groundingStartCoef >= 0 && groundingStartCoef <= 1);
  assert(groundingEndCoef >= 0 && groundingEndCoef <= 1);
  assert(groundingEndCoef >= groundingStartCoef);

  cout << "  Random games" <<  endl;
  cout << "    Game type: " << (dotsGame ? "Dots" : "Go") << endl;
  cout << "    Start position: " << Rules::writeStartPosRule(startPos) << endl;
  cout << "    Start position is random: " << boolalpha << startPosIsRandom << endl;
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

  Rules rules = dotsGame ? Rules(dotsGame, startPos, startPosIsRandom, dotsCaptureEmptyBase, Rules::DEFAULT_DOTS.dotsFreeCapturedDots) : Rules();
  int numLegalMoves = x_size * y_size - rules.getNumOfStartPosStones();

  vector<Loc> randomMoves = vector<Loc>();
  randomMoves.reserve(numLegalMoves);

  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      randomMoves.push_back(Location::getLoc(x, y, x_size));
    }
  }

  int movesCount = 0;
  int blackWinsCount = 0;
  int whiteWinsCount = 0;
  int drawsCount = 0;
  int groundingCount = 0;

  auto moveRecords = vector<Board::MoveRecord>();

  for (int n = 0; n < gamesCount; n++) {
    rand.shuffle(randomMoves);
    moveRecords.clear();

    auto initialBoard = Board(x_size, y_size, rules);
    initialBoard.setStartPos(DOTS_RANDOM);
    auto board = initialBoard;

    Loc lastLoc = Board::NULL_LOC;

    int tryGroundingAfterMove = (groundingStartCoef + rand.nextDouble() * (groundingEndCoef - groundingStartCoef)) * numLegalMoves;
    Player pla = P_BLACK;
    int currentGameMovesCount = 0;
    for(short randomMove : randomMoves) {
      lastLoc = currentGameMovesCount >= tryGroundingAfterMove ? Board::PASS_LOC : randomMove;

      if (board.isLegal(lastLoc, pla, suicideAllowed, false)) {
        if (performExtraChecks) {
          Board::MoveRecord moveRecord = board.playMoveRecorded(lastLoc, pla);
          moveRecords.push_back(moveRecord);
        } else {
          board.playMoveAssumeLegal(lastLoc, pla);
        }
        currentGameMovesCount++;
        pla = getOpp(pla);
      }

      if (lastLoc == Board::PASS_LOC) {
        groundingCount++;
        int scoreDiff;
        int oppScoreIfGrounding;
        if (Player lastPla = getOpp(pla); lastPla == P_BLACK) {
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
        break;
      }
    }

    if (performExtraChecks) {
      if (lastLoc == Board::PASS_LOC) {
        Board boardBeforeGrounding(board);
        boardBeforeGrounding.undo(moveRecords.back());
        validateGrounding(boardBeforeGrounding, board, moveRecords.back().pla, moveRecords);
      }
      validateStatesAndCaptures(board, moveRecords);
    }

    if (dotsGame && suicideAllowed && lastLoc != Board::PASS_LOC) {
      testAssert(0 == board.numLegalMoves);
    }

    movesCount += currentGameMovesCount;
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
  auto board = Board(39, 32, Rules(true, Rules::START_POS_EMPTY, Rules::DEFAULT_DOTS.startPosIsRandom, Rules::DEFAULT_DOTS.dotsCaptureEmptyBases, Rules::DEFAULT_DOTS.dotsFreeCapturedDots));
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      const Player pla = y == 0 || y == board.y_size - 1 || x == 0 || x == board.x_size - 1 ? P_BLACK : P_WHITE;
      board.playMoveAssumeLegal(Location::getLoc(x, y, board.x_size), pla);
    }
  }
  testAssert((board.x_size - 2) * (board.y_size - 2) == board.numWhiteCaptures);
  testAssert(0 == board.numLegalMoves);

  runDotsStressTestsInternal(39, 32, 3000, true, Rules::START_POS_CROSS, false, false, 0.0f, true, 0.8f, 1.0f, true);
  runDotsStressTestsInternal(39, 32, 3000, true, Rules::START_POS_CROSS_4, true, true, 0.5f, false, 0.8f, 1.0f, true);

  runDotsStressTestsInternal(39, 32, 50000, true, Rules::START_POS_CROSS, false, false, 0.0f, true, 0.8f, 1.0f, false);
  runDotsStressTestsInternal(39, 32, 50000, true, Rules::START_POS_CROSS_4, true, false, 0.0f, true, 0.8f, 1.0f, false);
}