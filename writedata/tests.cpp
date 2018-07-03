
#include <sstream>
#include "core/global.h"
#include "core/rand.h"
#include "fastboard.h"
#include "sgf.h"
#include "lzparse.h"
#include "datapool.h"

static void testAssertFailed(const char *msg, const char *file, int line) {
  Global::fatalError(string("Failed test assert: ") + string(msg) + "\n" + string("file: ") + string(file) + "\n" + string("line: ") + Global::intToString(line));
}
//A version of assert that's always defined, regardless of NDEBUG
#define testAssert(EX) (void)((EX) || (testAssertFailed(#EX, __FILE__, __LINE__),0))

static void runRandHashTests() {
  cout << "Running rng and hash tests" << endl;
  Rand::test();
}

static bool boardColorsEqual(const FastBoard& b1, const FastBoard& b2) {
  for(int i = 0; i<FastBoard::MAX_ARR_SIZE; i++)
    if(b1.colors[i] != b2.colors[i])
      return false;
  return true;
}

static void runBoardStressTests() {
  cout << "Running board stress tests" << endl;
  Rand rand("runBoardStressTests");

  int numBoards = 4;
  FastBoard boards[numBoards];
  boards[0] = FastBoard();
  boards[1] = FastBoard(9,16,false);
  boards[2] = FastBoard(13,7,true);
  boards[3] = FastBoard(4,4,false);
  FastBoard copies[numBoards];
  Player pla = C_BLACK;
  int suicideCount = 0;
  int koBanCount = 0;
  int koCaptureCount = 0;
  int passCount = 0;
  int regularMoveCount = 0;
  for(int n = 0; n < 20000; n++) {
    Loc locs[numBoards];

    //Sometimes generate garbage input
    if(n % 2 == 0) {
      locs[0] = (Loc)rand.nextInt(-10,500);
      locs[1] = (Loc)rand.nextInt(-10,250);
      locs[2] = (Loc)rand.nextInt(-10,200);
      locs[3] = (Loc)rand.nextInt(-10,50);
    }
    //Sometimes select only from empty points, to greatly increase the chance of real moves
    else {
      for(int i = 0; i<numBoards; i++) {
        int size = boards[i].empty_list.size();
        testAssert(size > 0);
        locs[i] = boards[i].empty_list[rand.nextUInt(size)];
      }
    }

    for(int i = 0; i<numBoards; i++)
      copies[i] = boards[i];

    bool isLegal[numBoards];
    bool suc[numBoards];
    for(int i = 0; i<numBoards; i++) {
      isLegal[i] = boards[i].isLegal(locs[i],pla);
      testAssert(boardColorsEqual(copies[i],boards[i]));
      suc[i] = boards[i].playMove(locs[i],pla);
    }
    
    for(int i = 0; i<numBoards; i++) {
      testAssert(isLegal[i] == suc[i]);
      boards[i].checkConsistency();

      const FastBoard& board = boards[i];
      const FastBoard& copy = copies[i];
      Loc loc = locs[i];
      if(!suc[i]) {
        if(board.isOnBoard(loc)) {
          testAssert(boardColorsEqual(copy,board));
          testAssert(loc < 0 || loc >= FastBoard::MAX_ARR_SIZE || board.colors[loc] != C_EMPTY || board.isIllegalSuicide(loc,pla) || board.isKoBanned(loc));
          if(board.isKoBanned(loc)) {
            testAssert(board.colors[loc] == C_EMPTY && (board.wouldBeKoCapture(loc,C_BLACK) || board.wouldBeKoCapture(loc,C_WHITE)));
            koBanCount++;
          }
        }
      }
      else {
        if(loc == FastBoard::PASS_LOC) {
          testAssert(boardColorsEqual(copy,board));
          testAssert(board.ko_loc == FastBoard::NULL_LOC);
          passCount++;
        }
        else if(copy.isSuicide(loc,pla)) {
          testAssert(board.colors[loc] == C_EMPTY);
          testAssert(board.isLegal(loc,pla));
          testAssert(board.isMultiStoneSuicideLegal);
          suicideCount++;
        }
        else {
          testAssert(board.colors[loc] == pla);
          testAssert(board.getNumLiberties(loc) == copy.getNumLibertiesAfterPlay(loc,pla,1000));
          testAssert(std::min(2,board.getNumLiberties(loc)) == copy.getNumLibertiesAfterPlay(loc,pla,2));
          testAssert(std::min(4,board.getNumLiberties(loc)) == copy.getNumLibertiesAfterPlay(loc,pla,4));
          if(board.ko_loc != FastBoard::NULL_LOC) {
            koCaptureCount++;
            testAssert(copy.wouldBeKoCapture(loc,pla));
          }
          else
            testAssert(!copy.wouldBeKoCapture(loc,pla));

          regularMoveCount++;
        }
      }
    }
    
    pla = getOpp(pla);
  }

  ostringstream out;
  out << endl;
  out << "regularMoveCount " << regularMoveCount << endl;
  out << "passCount " << passCount << endl;
  out << "koCaptureCount " << koCaptureCount << endl;
  out << "koBanCount " << koBanCount << endl;
  out << "suicideCount " << suicideCount << endl;

  string expected = R"%%(

regularMoveCount 25755
passCount 269
koCaptureCount 15
koBanCount 3
suicideCount 145

)%%";
  if(Global::trim(out.str()) != Global::trim(expected)) {
    cout << "Expect test failure, got " << endl;
    cout << out.str();
    testAssert(false);
  }
  
}



int main(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
  testAssert(sizeof(size_t) == 8);
  FastBoard::initHash();

  runRandHashTests();
  runBoardStressTests();

  cout << "All tests passed" << endl;
  return 0;
}
