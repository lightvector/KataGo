
#include <sstream>
#include "core/global.h"
#include "core/rand.h"
#include "game/board.h"

static void testAssertFailed(const char *msg, const char *file, int line) {
  Global::fatalError(string("Failed test assert: ") + string(msg) + "\n" + string("file: ") + string(file) + "\n" + string("line: ") + Global::intToString(line));
}
//A version of assert that's always defined, regardless of NDEBUG
#define testAssert(EX) (void)((EX) || (testAssertFailed(#EX, __FILE__, __LINE__),0))

static void expect(const char* name, ostringstream& out, const string& expected) {
  if(Global::trim(out.str()) != Global::trim(expected)) {
    cout << "Expect test failure!" << endl;
    cout << name << endl;
    cout << "Expected===============================================================" << endl;
    cout << expected << endl;
    cout << "Got====================================================================:" << endl;
    cout << out.str() << endl;
    exit(1);
  }
}


static void runRandHashTests() {
  cout << "Running rng and hash tests" << endl;
  Rand::test();
}

static bool boardColorsEqual(const Board& b1, const Board& b2) {
  for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
    if(b1.colors[i] != b2.colors[i])
      return false;
  return true;
}

static Board parseBoard(int xSize, int ySize, const string& s) {
  Board board(xSize,ySize,false);
  vector<string> lines = Global::split(Global::trim(s),'\n');
  assert(lines.size() == ySize);
  for(int y = 0; y<ySize; y++) {
    assert(lines[y].length() == xSize);
    for(int x = 0; x<xSize; x++) {
      char c = lines[y][x];
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(c == '.' || c == ' ' || c == '*' || c == ',' || c == '`')
        continue;
      else if(c == 'o' || c == 'O')
        board.setStone(loc,P_WHITE);
      else if(c == 'x' || c == 'X')
        board.setStone(loc,P_BLACK);
      else
        assert(false);
    }
  }
  return board;
}

static void runBoardTests() {
  cout << "Running board tests" << endl;
  ostringstream out;

  //============================================================================
  {
    const char* name = "Liberties";
    Board board = parseBoard(9,9,R"%%(
.........
.....x...
..oo..x..
..x......
......xx.
..x..ox..
.oxoo.oxx
xxoo.o.ox
.x.....oo
)%%");
    
    out << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY)
          out << board.getNumLiberties(loc);
        else
          out << ".";
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
.........
.....4...
..55..4..
..3......
......55.
..3..35..
.2366.222
3366.4.22
.3.....22

)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  //============================================================================
  {
    const char* name = "Liberties after move";
    Board board = parseBoard(9,9,R"%%(
.........
.....x...
..oo..x..
..x......
......xx.
..x..ox..
.oxoo.oxx
xxoo.o.ox
.x.....oo
)%%");

    out << endl;
    out << "After black" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_EMPTY)
          out << board.getNumLibertiesAfterPlay(loc,P_BLACK,100);
        else
          out << "-";
      }
      out << endl;
    }
    out << endl;
    out << "After white" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_EMPTY)
          out << board.getNumLibertiesAfterPlay(loc,P_WHITE,100);
        else
          out << "-";
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
After black
233335332
34336-743
33--37-63
35-444863
346446--6
34-42--52
3----0---
----1-1--
2-32322--

After white
233332332
34663-243
37--72-33
33-644233
342444--2
33-67--12
2----8---
----8-4--
0-56352--
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  //============================================================================
  {
    const char* name = "Ladders 1 Lib";
    vector<Loc> buf;
    Board board = parseBoard(9,9,R"%%(
xo.x..oxo
xoxo..o..
xxo......
..o.x....
xo..xox..
o..ooxo..
.....xo..
xoox..xo.
.xxoo.xxo
)%%");

    out << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY)
          out << (int)board.searchIsLadderCaptured(loc,true,buf);
        else
          out << ".";
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
01.0..010
0100..0..
000......
..0.0....
10..000..
0..0000..
.....00..
0000..00.
.1100.001
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  //============================================================================
  {
    const char* name = "Ladders 2 Libs";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = parseBoard(9,9,R"%%(
xo.x..oxo
xo.o..o..
xxo......
..o.x....
xo..xo...
...ooxo..
.....xo..
xoox..xo.
.xx.o.xxo
)%%");

    out << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY)
          out << (int)board.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,buf2);
        else
          out << ".";
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
11.1..000
11.0..0..
110......
..0.0....
10..00...
...0010..
.....10..
1110..01.
.11.0.000

)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  //============================================================================
  {
    const char* name = "LaddersKo-1";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = parseBoard(18,9,R"%%(
..................
..................
....ox.......ox...
..xooox....xooox..
..xoxox....xoxox..
..xx.x......x.x...
...ox.......ox....
....o.............
..................
)%%");

    out << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY)
          out << (int)board.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,buf2);
        else
          out << ".";
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
..................
..................
....10.......00...
..01110....00000..
..01010....00000..
..00.0......0.0...
...00.......00....
....0.............
..................

)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  //============================================================================
  {
    const char* name = "LaddersKo-2";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = parseBoard(18,9,R"%%(
.............xo.oo
....x........xxooo
...x.xx.......xxo.
..xoxoxo.......xxo
..xoooxo........xx
...xo......o......
..................
..................
.....x.oo.........
)%%");
    Board board2 = parseBoard(18,9,R"%%(
..................
....x.............
...x.xx...........
..xoxoxo..........
..xoooxo..........
...xo......o......
.................x
..................
.....x.oo.........
)%%");
    Board board3 = parseBoard(18,9,R"%%(
....xo.......xo...
....xox......xox..
...xo.ox....xo.ox.
...xxoox....xxoox.
..xooox....xooox..
.xo.ox....xo.ox...
..xox......xox....
..xox......xox....
............o.....
)%%");

    out << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY)
          out << (int)board.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,buf2);
        else
          out << ".";
      }
      out << endl;
    }
    out << endl;
    for(int y = 0; y<board2.y_size; y++) {
      for(int x = 0; x<board2.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board2.x_size);
        if(board2.colors[loc] != C_EMPTY)
          out << (int)board2.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,buf2);
        else
          out << ".";
      }
      out << endl;
    }
    out << endl;
    for(int y = 0; y<board3.y_size; y++) {
      for(int x = 0; x<board3.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board3.x_size);
        if(board3.colors[loc] != C_EMPTY)
          out << (int)board3.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,buf2);
        else
          out << ".";
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(

.............00.11
....0........00111
...0.00.......001.
..000000.......000
..000000........00
...00......0......
..................
..................
.....0.00.........

..................
....0.............
...0.00...........
..010100..........
..011100..........
...01......0......
.................0
..................
.....0.00.........

....01.......01...
....011......011..
...00.10....00.00.
...00110....00000.
..01110....00000..
.00.10....00.00...
..010......000....
..010......000....
............0.....

)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  
  //============================================================================
  {
    const char* name = "Pass-alive 1";
    Color result[Board::MAX_ARR_SIZE];
    Board board = parseBoard(9,9,R"%%(
..o.o.xx.
.oooo.x.x
oo.....xx
.........
.........
x......oo
.xx...o.o
x.x..oo.o
.x.x.o.o.
)%%");
    
    out << endl;
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
    board.setMultiStoneSuicideLegal(true);
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  //============================================================================
  {
    const char* name = "Pass-alive 2";
    Color result[Board::MAX_ARR_SIZE];
    Board board = parseBoard(9,9,R"%%(
x.oooooo.
oox..xx.o
o...xox.o
o...x.x.o
oxxx.xx.o
ox..x...o
o.xox...o
o.xxx...o
.ooooooo.
)%%");
    Board board2 = parseBoard(9,9,R"%%(
..oooooo.
oox..xx.o
o...xox.o
o...x.x.o
oxxx.xx.o
ox..x...o
o.x.x...o
o.xxx...o
.ooooooo.
)%%");
    
    out << endl;
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
    board2.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
    board.setMultiStoneSuicideLegal(true);
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

.........
.........
.........
.........
.........
.........
.........
.........
.........

.........
.........
.........
.........
.........
.........
.........
.........
.........
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  //============================================================================
  {
    const char* name = "Pass-alive 3";
    Color result[Board::MAX_ARR_SIZE];
    Board board = parseBoard(19,19,R"%%(
o.o....xx......o..o
.oo.xxx.x......ooo.
xo..x..xx........oo
oo.`xxx..`.....`...
...................
.......xx..........
....xxx.x......xxx.
....xo.xx......x.x.
....xxx........x.x.
xxx`....xxxxx..`x.x
..x.....x...x...xx.
o.xxx...x...x....xx
o.x.x...x...xxx....
..xxx...x...x.x....
xxx.....xxxxxxx....
oo.`.....`.....`...
.o.....oo.oo......o
.oooo.o.o.o.o.oooo.
o..o.o.oo.oo.o.o.o.
)%%");
    out << endl;
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
    board.setMultiStoneSuicideLegal(true);
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
OOO................
OOO................
OO.................
OO.................
...................
.......XX..........
....XXXXX......XXX.
....XXXXX......XXX.
....XXX........XXX.
XXX.............XXX
..X.............XXX
..XXX............XX
..XXX..............
..XXX..............
XXX................
...................
..........OO.......
..........OOO.OOOO.
..........OOOOOOOO.

...................
...................
...................
...................
...................
...................
...............XXX.
...............XXX.
...............XXX.
................XXX
................XXX
.................XX
...................
...................
...................
...................
..........OO.......
..........OOO.OOOO.
..........OOOOOOOO.
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  //============================================================================
  {
    const char* name = "Pass-alive 4";
    Color result[Board::MAX_ARR_SIZE];
    Board board = parseBoard(19,19,R"%%(
.x.x.xxxx.xxxx.x.x.
x.xxxx..x.x..xxxx.x
xx.x..x.x.x.x..x.x.
x.x`xx.xx`xx.xx`x.x
.x.xx.xx...xx.xx.x.
.x.x...........x.x.
x.x.............x.x
.x...............x.
...................
xxx`....xxxxx..`...
..x.....x...x......
oxxxx...x..xx......
o.x.x...x.o.xxx....
..xxx...x...x.x....
xxx.....xxxxxxx....
...`.....`....o`...
ooooo....ooooo.oooo
.o..oo...o...ooo...
o.o.xo...o.o.o.o.oo
)%%");
    out << endl;
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
    board.setMultiStoneSuicideLegal(true);
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
XXXXXXXXX..........
XXXXXX..X..........
XX.X....X..........
X......XX..........
......XX...........
...................
...................
...................
...................
XXX.....XXXXX......
XXX.....XXXXX......
XXXXX...XXXXX......
XXXXX...XXXXXXX....
XXXXX...XXXXXXX....
XXX.....XXXXXXX....
...................
...................
...................
...................

XXXXXXXXX..........
XXXXXX..X..........
XX.X....X..........
X......XX..........
......XX...........
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  //============================================================================
  {
    const char* name = "Pass-alive 5";
    Color result[Board::MAX_ARR_SIZE];
    Board board = parseBoard(19,13,R"%%(
...................
...................
...............xx..
........xxxxxxxx.x.
.....xxxx....x..ox.
..xxx...x.....xxxx.
..x..xxx........x..
..x.ox..xxxxx...x..
..x.oxxx.oo.xxxxx..
..x.ooo.x..x...x...
..xxx...xxxxxx.x...
....xxxxx.....xx...
...................
)%%");
    out << endl;
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
    board.setMultiStoneSuicideLegal(true);
    board.calculatePassAliveTerritory(result);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << getCharOfColor(result[loc]);
      }
      out << endl;
    }
    out << endl;
     
    string expected = R"%%(
...................
...................
...............XX..
........XXXXXXXXXX.
.....XXXX....XXXXX.
..XXXXXXX.....XXXX.
..XXXXXX........X..
..XXXXXXXXXXX...X..
..XXXXXXXXXXXXXXX..
..XXXXXXXXXXXXXX...
..XXXXXXXXXXXXXX...
....XXXXX.....XX...
...................

...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }
  
}

static void runBoardStressTest() {
  cout << "Running board stress test" << endl;
  Rand rand("runBoardStressTests");

  int numBoards = 4;
  Board boards[numBoards];
  boards[0] = Board();
  boards[1] = Board(9,16,false);
  boards[2] = Board(13,7,true);
  boards[3] = Board(4,4,false);
  Board copies[numBoards];
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

      const Board& board = boards[i];
      const Board& copy = copies[i];
      Loc loc = locs[i];
      if(!suc[i]) {
        if(board.isOnBoard(loc)) {
          testAssert(boardColorsEqual(copy,board));
          testAssert(loc < 0 || loc >= Board::MAX_ARR_SIZE || board.colors[loc] != C_EMPTY || board.isIllegalSuicide(loc,pla) || board.isKoBanned(loc));
          if(board.isKoBanned(loc)) {
            testAssert(board.colors[loc] == C_EMPTY && (board.wouldBeKoCapture(loc,C_BLACK) || board.wouldBeKoCapture(loc,C_WHITE)));
            koBanCount++;
          }
        }
      }
      else {
        if(loc == Board::PASS_LOC) {
          testAssert(boardColorsEqual(copy,board));
          testAssert(board.ko_loc == Board::NULL_LOC);
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
          if(board.ko_loc != Board::NULL_LOC) {
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
  expect("Board stress test move counts",out,expected);
}



int main(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
  testAssert(sizeof(size_t) == 8);
  Board::initHash();

  runRandHashTests();
  runBoardTests();
  runBoardStressTest();

  cout << "All tests passed" << endl;
  return 0;
}
