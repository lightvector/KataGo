#include "../tests/tests.h"
#include "../neuralnet/nninputs.h"

using namespace std;
using namespace TestCommon;

void Tests::runBoardAreaTests() {
  cout << "Running board area tests" << endl;
  ostringstream out;

  //============================================================================
  auto printAreas = [&out](const Board& board, Color result[Board::MAX_ARR_SIZE]) {
    bool safeBigTerritoriesBuf[7] =        {false, true,  true,  true,  true,  true,  true};
    bool unsafeBigTerritoriesBuf[7] =      {false, false, true,  true,  false, true,  true};
    bool nonPassAliveStonesBuf[7] =        {false, false, false, true,  false, false, true};
    bool recursivelyReachesSafeBuf[7] =    {false, false, false, false, true,  true,  true};

    for(int mode = 0; mode < 14; mode++) {
      bool multiStoneSuicideLegal = (mode % 2 == 1);
      bool safeBigTerritories = safeBigTerritoriesBuf[mode/2];
      bool unsafeBigTerritories = unsafeBigTerritoriesBuf[mode/2];
      bool nonPassAliveStones = nonPassAliveStonesBuf[mode/2];
      bool recursivelyReachesSafe = recursivelyReachesSafeBuf[mode/2];
      int whiteMinusBlackSafeRegionCount = 0;
      Board copy(board);
      copy.calculateArea(result,whiteMinusBlackSafeRegionCount,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,recursivelyReachesSafe,multiStoneSuicideLegal);
      out << "Safe big territories " << safeBigTerritories << " "
      << "Unsafe big territories " << unsafeBigTerritories << " "
      << "Non pass alive stones " << nonPassAliveStones << " "
      << "Recusively reaches safe " << recursivelyReachesSafe << " "
      << "Suicide " << multiStoneSuicideLegal << endl;
      out << "whiteMinusBlackSafeRegionCount " << whiteMinusBlackSafeRegionCount << endl;
      for(int y = 0; y<copy.y_size; y++) {
        for(int x = 0; x<copy.x_size; x++) {
          Loc loc = Location::getLoc(x,y,copy.x_size);
          out << PlayerIO::colorToChar(result[loc]);
        }
        out << endl;
      }
      out << endl;
      testAssert(boardsSeemEqual(copy,board));
      copy.checkConsistency();
    }
  };

  //============================================================================
  {
    const char* name = "Area 1";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(9,9,R"%%(
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

    printAreas(board,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OO.O..XXX
O.....XXX
.......XX
.........
.........
.......OO
X.....OOO
.X...OOOO
X.X..OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
OO.O..XXX
O.....XXX
.......XX
.........
.........
.......OO
X.....OOO
.X...OOOO
X.X..OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOO.XXX
OOOOO.XXX
OO.....XX
.........
.........
X......OO
XXX...OOO
XXX..OOOO
XXXX.OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOO.XXX
OOOOO.XXX
OO.....XX
.........
.........
X......OO
XXX...OOO
XXX..OOOO
XXXX.OOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OO.O..XXX
O.....XXX
.......XX
.........
.........
.......OO
X.....OOO
.X...OOOO
X.X..OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
OO.O..XXX
O.....XXX
.......XX
.........
.........
.......OO
X.....OOO
.X...OOOO
X.X..OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOO.XXX
OOOOO.XXX
OO.....XX
.........
.........
X......OO
XXX...OOO
XXX..OOOO
XXXX.OOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOO.XXX
OOOOO.XXX
OO.....XX
.........
.........
X......OO
XXX...OOO
XXX..OOOO
XXXX.OOOO
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Area 2";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(9,9,R"%%(
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
    Board board2 = Board::parseBoard(9,9,R"%%(
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

    printAreas(board,result);
    out << "-----" << endl;
    printAreas(board2,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
........O
.........
.........
.........
....X....
.........
.........
.........
O.......O

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OOX..XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
X.OOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OX..X...O
O.XOX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
........O
.........
.........
.........
....X....
.........
.........
.........
O.......O

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OOX..XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
X.OOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OX..X...O
O.XOX...O
O.XXX...O
OOOOOOOOO

-----
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OO......O
.........
.........
.........
....X....
..XX.....
...X.....
.........
O.......O

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
OO......O
.........
.........
.........
....X....
..XX.....
...X.....
.........
O.......O

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OO......O
.........
.........
.........
....X....
..XX.....
...X.....
.........
O.......O

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
OO......O
.........
.........
.........
....X....
..XX.....
...X.....
.........
O.......O

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Area 3";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(19,19,R"%%(
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

    printAreas(board,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOO.............OO.
OOO....X..........O
OO...XX............
OO.................
...................
.......XX..........
....XXXXX......XXX.
....XXXXX......XXX.
....XXX........XXX.
XXX.............XXX
..X......XXX....XXX
..XXX....XXX.....XX
..XXX....XXX.......
..XXX....XXX.X.....
XXX................
...................
O.........OO.......
O......O..OOO.OOOOO
.OO.O.O...OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.O..............OO.
.......X..........O
.....XX............
...................
...................
...................
.......X.......XXX.
...............XXX.
...............XXX.
................XXX
.........XXX....XXX
.........XXX.....XX
...X.....XXX.......
.........XXX.X.....
...................
...................
O.........OO.......
O......O..OOO.OOOOO
.OO.O.O...OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOO....XX......OOOO
OOO.XXXXX......OOOO
OO..XXXXX........OO
OO..XXX............
...................
.......XX..........
....XXXXX......XXX.
....XXXXX......XXX.
....XXX........XXX.
XXX.....XXXXX...XXX
..X.....XXXXX...XXX
O.XXX...XXXXX....XX
O.XXX...XXXXXXX....
..XXX...XXXXXXX....
XXX.....XXXXXXX....
OO.................
OO.....OO.OO......O
OOOOO.OOO.OOO.OOOOO
OOOOOOOOO.OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOO....XX......OOOO
.OO.XXXXX......OOOO
XO..XXXXX........OO
OO..XXX............
...................
.......XX..........
....XXXXX......XXX.
....XO.XX......XXX.
....XXX........XXX.
XXX.....XXXXX...XXX
..X.....XXXXX...XXX
O.XXX...XXXXX....XX
O.XXX...XXXXXXX....
..XXX...XXXXXXX....
XXX.....XXXXXXX....
OO.................
OO.....OO.OO......O
OOOOO.OOO.OOO.OOOOO
OOOOOOOOO.OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -1
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
..........OO......O
..........OOO.OOOOO
..........OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
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
..........OO......O
..........OOO.OOOOO
..........OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -1
OOO.............OO.
OOO....X..........O
OO...XX............
OO.................
...................
.......XX..........
....XXXXX......XXX.
....XXXXX......XXX.
....XXX........XXX.
XXX.............XXX
..X......XXX....XXX
..XXX....XXX.....XX
..XXX....XXX.......
..XXX....XXX.X.....
XXX................
...................
O.........OO......O
O......O..OOO.OOOOO
.OO.O.O...OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
.O..............OO.
.......X..........O
.....XX............
...................
...................
...................
.......X.......XXX.
...............XXX.
...............XXX.
................XXX
.........XXX....XXX
.........XXX.....XX
...X.....XXX.......
.........XXX.X.....
...................
...................
O.........OO......O
O......O..OOO.OOOOO
.OO.O.O...OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -1
OOO....XX......OOOO
OOO.XXXXX......OOOO
OO..XXXXX........OO
OO..XXX............
...................
.......XX..........
....XXXXX......XXX.
....XXXXX......XXX.
....XXX........XXX.
XXX.....XXXXX...XXX
..X.....XXXXX...XXX
O.XXX...XXXXX....XX
O.XXX...XXXXXXX....
..XXX...XXXXXXX....
XXX.....XXXXXXX....
OO.................
OO.....OO.OO......O
OOOOO.OOO.OOO.OOOOO
OOOOOOOOO.OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOO....XX......OOOO
.OO.XXXXX......OOOO
XO..XXXXX........OO
OO..XXX............
...................
.......XX..........
....XXXXX......XXX.
....XO.XX......XXX.
....XXX........XXX.
XXX.....XXXXX...XXX
..X.....XXXXX...XXX
O.XXX...XXXXX....XX
O.XXX...XXXXXXX....
..XXX...XXXXXXX....
XXX.....XXXXXXX....
OO.................
OO.....OO.OO......O
OOOOO.OOO.OOO.OOOOO
OOOOOOOOO.OOOOOOOOO
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Area 4";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(19,19,R"%%(
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
..x.....x...x.xxxxx
oxxxx...x..xx.x....
o.x.x...x.o.xxx....
..xxx...x...x.x....
xxx.....xxxxxxxxxxx
...`.....`....o`...
ooooo....ooooo.oooo
.o..oo...o...ooo...
o.o.xo...o.o.o.o.oo
)%%");

    printAreas(board,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
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
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.X....
XXXXX...XXXXXXX....
XXXXX...XXXXXXX....
XXX.....XXXXXXXXXXX
...................
...................
...................
...................

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
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
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.XXXXX
XXXXX...XXXXXXXXXXX
XXXXX...XXXXXXXXXXX
XXX.....XXXXXXXXXXX
...................
...................
...................
...................

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX.....X.X.X
XXXXXXXXX..XX....X.
XXXXXX.XX..X.XX.X.X
XX.X..XXX...X..X.X.
X.X...XX........X.X
X.X.............X.X
.X...............X.
...................
...................
XXX.....XXXXX......
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.XXXXX
XXXXX...XXXXXXXXXXX
XXXXX...XXXXXXXXXXX
XXX.....XXXXXXXXXXX
...................
..............O....
O.........OOO...OOO
.O........O.O.O.O..

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX.....X.X.X
XXXXXXXXX..XX....X.
XXXXXX.XX..X.XX.X.X
XX.X..XXX...X..X.X.
X.X...XX........X.X
X.X.............X.X
.X...............X.
...................
...................
...................
...................
...............XXXX
...X...........XXXX
.............X.XXXX
...................
...................
..............O....
O.........OOO...OOO
.O........O.O.O.O..

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXX.XX...XX.XXXXX
XXXX...........XXXX
XXX.............XXX
.X...............X.
...................
XXX.....XXXXX......
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.XXXXX
XXXXX...XXXXXXXXXXX
XXXXX...XXXXXXXXXXX
XXX.....XXXXXXXXXXX
..............O....
OOOOO....OOOOOOOOOO
OO..OO...OOOOOOOOOO
OOO.XO...OOOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXX.XX...XX.XXXXX
XXXX...........XXXX
XXX.............XXX
.X...............X.
...................
XXX.....XXXXX......
..X.....X...X.XXXXX
OXXXX...X..XX.XXXXX
O.XXX...X.O.XXXXXXX
..XXX...X...XXXXXXX
XXX.....XXXXXXXXXXX
..............O....
OOOOO....OOOOOOOOOO
OO..OO...OOOOOOOOOO
OOO.XO...OOOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -3
XXXXXXXXX..........
XXXXXXXXX..........
XXXXXXXXX..........
XXXXXXXXX..........
XXXXX.XX...........
XXXX...............
XXX................
.X.................
...................
XXX.....XXXXX......
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.XXXXX
XXXXX...XXXXXXXXXXX
XXXXX...XXXXXXXXXXX
XXX.....XXXXXXXXXXX
...................
...................
...................
...................

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount -1
XXXXXXXXX..........
XXXXXXXXX..........
XXXXXXXXX..........
XXXXXXXXX..........
XXXXX.XX...........
XXXX...............
XXX................
.X.................
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

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -3
XXXXXXXXX.....X.X.X
XXXXXXXXX..XX....X.
XXXXXXXXX..X.XX.X.X
XXXXXXXXX...X..X.X.
XXXXX.XX........X.X
XXXX............X.X
XXX..............X.
.X.................
...................
XXX.....XXXXX......
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.XXXXX
XXXXX...XXXXXXXXXXX
XXXXX...XXXXXXXXXXX
XXX.....XXXXXXXXXXX
...................
..............O....
O.........OOO...OOO
.O........O.O.O.O..

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount -1
XXXXXXXXX.....X.X.X
XXXXXXXXX..XX....X.
XXXXXXXXX..X.XX.X.X
XXXXXXXXX...X..X.X.
XXXXX.XX........X.X
XXXX............X.X
XXX..............X.
.X.................
...................
...................
...................
...............XXXX
...X...........XXXX
.............X.XXXX
...................
...................
..............O....
O.........OOO...OOO
.O........O.O.O.O..

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -3
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXX.XX...XX.XXXXX
XXXX...........XXXX
XXX.............XXX
.X...............X.
...................
XXX.....XXXXX......
XXX.....XXXXX.XXXXX
XXXXX...XXXXX.XXXXX
XXXXX...XXXXXXXXXXX
XXXXX...XXXXXXXXXXX
XXX.....XXXXXXXXXXX
..............O....
OOOOO....OOOOOOOOOO
OO..OO...OOOOOOOOOO
OOO.XO...OOOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount -1
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXXXXXX.XXXXXXXXX
XXXXX.XX...XX.XXXXX
XXXX...........XXXX
XXX.............XXX
.X...............X.
...................
XXX.....XXXXX......
..X.....X...X.XXXXX
OXXXX...X..XX.XXXXX
O.XXX...X.O.XXXXXXX
..XXX...X...XXXXXXX
XXX.....XXXXXXXXXXX
..............O....
OOOOO....OOOOOOOOOO
OO..OO...OOOOOOOOOO
OOO.XO...OOOOOOOOOO
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Area 5";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(19,13,R"%%(
...................
...................
...............xx..
........xxxxxxxx.x.
.....xxxx....x..ox.
..xxx...x..o..xxxx.
..x..xxx........x..
..x.ox..xxxxx...x..
..x.oxxx.oo.xxxxx..
..x.ooo.x..x...x...
..xxx...xxxxxx.x...
....xxxxx.....xx...
...................
)%%");

    printAreas(board,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXX....XXXXXX
XXXXXXXXX.....XXXXX
XXXXXXXX........XXX
XXXXXXXXXXXXX...XXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXX....XXXXXX
XXXXXXXXX.....XXXXX
XXXXXXXX........XXX
XXXXXXXXXXXXX...XXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX..XX
XXXXXXXX..........X
XXXXX.............X
XX...XXX..........X
XX...............XX
XX....XX.........XX
XX...............XX
XX..........XXX.XXX
XX............X.XXX
XXXX.....XXXXX..XXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXX....XXXXXX
XXXXXXXXX..O..XXXXX
XXXXXXXX........XXX
XXXXXXXXXXXXX...XXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXX.XX
XXXXXXXXX....X..OXX
XXXXXXXXX..O..XXXXX
XXX..XXX........XXX
XXX.OXXXXXXXX...XXX
XXX.OXXX.OO.XXXXXXX
XXX.OOO.X..XXXXXXXX
XXXXX...XXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -1
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXX....XXXXXX
XXXXXXXXX.....XXXXX
XXXXXXXX........XXX
XXXXXXXXXXXXX...XXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
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

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -1
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXX....XXXXXX
XXXXXXXXX.....XXXXX
XXXXXXXX........XXX
XXXXXXXXXXXXX...XXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXX..XX
XXXXXXXX..........X
XXXXX.............X
XX...XXX..........X
XX...............XX
XX....XX.........XX
XX...............XX
XX..........XXX.XXX
XX............X.XXX
XXXX.....XXXXX..XXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -1
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXX....XXXXXX
XXXXXXXXX..O..XXXXX
XXXXXXXX........XXX
XXXXXXXXXXXXX...XXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXX.XX
XXXXXXXXX....X..OXX
XXXXXXXXX..O..XXXXX
XXX..XXX........XXX
XXX.OXXXXXXXX...XXX
XXX.OXXX.OO.XXXXXXX
XXX.OOO.X..XXXXXXXX
XXXXX...XXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXX
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Area Rect";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(12,3,R"%%(
x.ooxxxo.xx.
oo.ox.xoox.x
ooxox.xo.ox.
)%%");

    printAreas(board,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOO.....XXX
OOOO.....XXX
OOOO......XX

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.........XXX
.........XXX
..........XX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOO.....XXX
OOOO.....XXX
OOOO......XX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.........XXX
.........XXX
..........XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOO.....XXX
OOOO.X...XXX
OOOO.X..O.XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.........XXX
.....X...XXX
.....X..O.XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOXXXO.XXX
OOOOXXXOOXXX
OOOOXXXOOOXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
X.OOXXXO.XXX
OO.OXXXOOXXX
OOXOXXXOOOXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOO.....XXX
OOOO.....XXX
OOOO......XX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount -1
.........XXX
.........XXX
..........XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOO.....XXX
OOOO.X...XXX
OOOO.X..O.XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount -1
.........XXX
.....X...XXX
.....X..O.XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOXXXO.XXX
OOOOXXXOOXXX
OOOOXXXOOOXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount -1
X.OOXXXO.XXX
OO.OXXXOOXXX
OOXOXXXOOOXX
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Pass alive bug-derived test case";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(9,9,R"%%(
   A B C D E F G H J
 9 . O . O O X X O .
 8 O . O . O O X X O
 7 O O O O O O X . X
 6 O X O . O O X X X
 5 X X O O O O X X .
 4 . X O O O X X X .
 3 . X O O O O O O X
 2 O X X O X X X X X
 1 . X X X . X . . .
)%%");
    printAreas(board,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
OOOOOXXXX
OOOOOOXXX
OOOOOOXXX
OXOOOOXXX
XXOOOOXXX
XXOOOXXXX
XXOOOOOOX
XXXOXXXXX
XXXXXXXXX
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "One more simple test case";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(7,7,R"%%(
..ooo..
..xxx..
xxxxxxx
.....x.
ooooox.
.o..oxx
.x..ox.
)%%");
    printAreas(board,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
..OOO..
..XXX..
XXXXXXX
.....XX
OOOOOXX
.O..OXX
.X..OXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
..OOO..
..XXX..
XXXXXXX
.....XX
OOOOOXX
.O..OXX
.X..OXX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -1
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount -1
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -1
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount -1
.......
..XXX..
XXXXXXX
.....XX
.....XX
.....XX
.....XX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount -1
..OOO..
..XXX..
XXXXXXX
.....XX
OOOOOXX
.O..OXX
.X..OXX

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount -1
..OOO..
..XXX..
XXXXXXX
.....XX
OOOOOXX
.O..OXX
.X..OXX
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Bug-derived test case, in more colors and orientations";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(9,9,R"%%(
.oxo.oxo.
ooxoooxxo
xxxxxxxxx
.x.x.....
xxxoooooo
..o.xo.o.
ooooooooo
xooxxxoxx
.xox.xox.
)%%");
    printAreas(board,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXXXXXXXX
XXXXXXXXX
XXXXXXXXX
XXXX.....
XXXOOOOOO
..OOOOOOO
OOOOOOOOO
OOOOOOOOO
OOOOOOOOO
)%%";
    expect(name,out,expected);
  }


  //============================================================================
  {
    const char* name = "More tests for recursive safe";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(19,19,R"%%(
.xx.....o.o.o...xx.
x.x.x...ooooo.x.x.x
xx.x..oo...o...x.xx
..x`.o.o.`.o...`x..
.x.x..oo...o...x.x.
.x.x....ooo....xox.
o.x...oo...o....x..
.x.x.o.o...o...x.x.
..x...oo...o....x..
.xx`....o.o....`...
xooo.....xxxx......
.x....xxxx..xx.....
x.....x.x....x.....
...o...x.x..xx.o.o.
o.o.oo.xxxxxxoo.o.o
.o.o...xooooo..o.o.
oox.o..xoxo.o.o.xoo
o.oo..xxoxoxox.oo.o
.oo..x..x.x.x...oo.
)%%");

    printAreas(board,result);

    string expected = R"%%(
Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XX.........O.....XX
...........O.......
...........O.......
...................
...................
...................
...................
...................
.........XXXX......
......XXXX..XX.....
......XXX....X.....
.......XXX..XX.....
.......XXXXXX......
.O.....X.........O.
OO.....X.........OO
OOOO..XX.......OOOO
OOO.............OOO

Safe big territories 0 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XX.........O.....XX
...........O.......
...........O.......
...................
...................
...................
...................
...................
.........XXXX......
......XXXX..XX.....
......XXX....X.....
.......XXX..XX.....
.......XXXXXX......
.O.....X.........O.
OO.....X.........OO
OOOO..XX.......OOOO
OOO.............OOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XX.........O.....XX
...........O.......
...........O.......
...................
...................
...................
...................
...................
.........XXXX......
......XXXXXXXX.....
......XXXXXXXX.....
.......XXXXXXX.....
.......XXXXXX......
.O.....X.........O.
OO.....X.........OO
OOOO..XX.......OOOO
OOO.............OOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XX.........O.....XX
...........O.......
...........O.......
...................
...................
...................
...................
...................
.........XXXX......
......XXXXXXXX.....
......XXXXXXXX.....
.......XXXXXXX.....
.......XXXXXX......
.O.....X.........O.
OO.....X.........OO
OOOO..XX.......OOOO
OOO.............OOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XXX.....OOOO....XXX
......O.OOOO.......
..X.....OOOO.......
..X................
...................
..X...O.........X..
...................
...................
.........XXXX......
X.....XXXXXXXX.....
......XXXXXXXX.....
.......XXXXXXX.....
...O...XXXXXX..O.O.
OO.....X.........OO
OO.....X.........OO
OOOO..XX.......OOOO
OOO...XX.X.X....OOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XXX.....OOOO....XXX
......O.OOOO.......
..X.....OOOO.......
..X................
...................
..X...O.........X..
...................
...................
.........XXXX......
X.....XXXXXXXX.....
......XXXXXXXX.....
.......XXXXXXX.....
...O...XXXXXX..O.O.
OO.....X.........OO
OO.....X.........OO
OOOO..XX.......OOOO
OOO...XX.X.X....OOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.X...OOOOO.X.XXX
XXXX..OOOOOO...XXXX
..X..OOOOOOO....X..
.XXX..OOOOOO...X.X.
.XXX....OOO....XOX.
O.X...OO...O....X..
.XXX.OOO...O...XXX.
..X...OO...O....X..
.XX.....O.O........
XOOO.....XXXX......
XX....XXXXXXXX.....
X.....XXXXXXXX.....
...O...XXXXXXX.O.O.
O.OOOO.XXXXXXOOOOOO
OO.O...XOOOOO..O.OO
OOX.O..XOXO.O.O.XOO
OOOO..XXOXOXOX.OOOO
OOO..XXXXXXXX...OOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 0 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.X...OOOOO.X.XXX
XXXX..OOOOOO...XXXX
..X..OOOOOOO....X..
.XXX..OOOOOO...X.X.
.XXX....OOO....XOX.
O.X...OO...O....X..
.XXX.OOO...O...XXX.
..X...OO...O....X..
.XX.....O.O........
XOOO.....XXXX......
XX....XXXXXXXX.....
X.....XXXXXXXX.....
...O...XXXXXXX.O.O.
O.OOOO.XXXXXXOOOOOO
OO.O...XOOOOO..O.OO
OOX.O..XOXO.O.O.XOO
OOOO..XXOXOXOX.OOOO
OOO..XXXXXXXX...OOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XXXX..OOOOOO...XXXX
..X..OOOOOOO....X..
.XXX..OOOOOO.......
.XXX....OOO........
..X................
.XXX...............
..X................
.XX................
.........XXXX......
......XXXXXXXX.....
......XXXXXXXX.....
.......XXXXXXX.O.O.
O......XXXXXXOOOOOO
OO.....X.......O.OO
OO.....X.X.......OO
OOOO..XX.X.X...OOOO
OOO..XXXXXXXX...OOO

Safe big territories 1 Unsafe big territories 0 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XXXX..OOOOOO...XXXX
..X..OOOOOOO....X..
.XXX..OOOOOO.......
.XXX....OOO........
..X................
.XXX...............
..X................
.XX................
.........XXXX......
......XXXXXXXX.....
......XXXXXXXX.....
.......XXXXXXX.O.O.
O......XXXXXXOOOOOO
OO.....X.......O.OO
OO.....X.X.......OO
OOOO..XX.X.X...OOOO
OOO..XXXXXXXX...OOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XXXX..OOOOOO...XXXX
..X..OOOOOOO....X..
.XXX..OOOOOO.......
.XXX....OOO........
..X................
.XXX..O.........X..
..X................
.XX................
.........XXXX......
X.....XXXXXXXX.....
......XXXXXXXX.....
.......XXXXXXX.O.O.
O..O...XXXXXXOOOOOO
OO.....X.......O.OO
OO.....X.X.......OO
OOOO..XX.X.X...OOOO
OOO..XXXXXXXX...OOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 0 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.....OOOOO...XXX
XXXX..OOOOOO...XXXX
..X..OOOOOOO....X..
.XXX..OOOOOO.......
.XXX....OOO........
..X................
.XXX..O.........X..
..X................
.XX................
.........XXXX......
X.....XXXXXXXX.....
......XXXXXXXX.....
.......XXXXXXX.O.O.
O..O...XXXXXXOOOOOO
OO.....X.......O.OO
OO.....X.X.......OO
OOOO..XX.X.X...OOOO
OOO..XXXXXXXX...OOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 0
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.X...OOOOO.X.XXX
XXXX..OOOOOO...XXXX
..X..OOOOOOO....X..
.XXX..OOOOOO...X.X.
.XXX....OOO....XOX.
O.X...OO...O....X..
.XXX.OOO...O...XXX.
..X...OO...O....X..
.XX.....O.O........
XOOO.....XXXX......
XX....XXXXXXXX.....
X.....XXXXXXXX.....
...O...XXXXXXX.O.O.
O.OOOO.XXXXXXOOOOOO
OO.O...XOOOOO..O.OO
OOX.O..XOXO.O.O.XOO
OOOO..XXOXOXOX.OOOO
OOO..XXXXXXXX...OOO

Safe big territories 1 Unsafe big territories 1 Non pass alive stones 1 Recusively reaches safe 1 Suicide 1
whiteMinusBlackSafeRegionCount 0
XXX.....OOOOO...XXX
XXX.X...OOOOO.X.XXX
XXXX..OOOOOO...XXXX
..X..OOOOOOO....X..
.XXX..OOOOOO...X.X.
.XXX....OOO....XOX.
O.X...OO...O....X..
.XXX.OOO...O...XXX.
..X...OO...O....X..
.XX.....O.O........
XOOO.....XXXX......
XX....XXXXXXXX.....
X.....XXXXXXXX.....
...O...XXXXXXX.O.O.
O.OOOO.XXXXXXOOOOOO
OO.O...XOOOOO..O.OO
OOX.O..XOXO.O.O.XOO
OOOO..XXOXOXOX.OOOO
OOO..XXXXXXXX...OOO

)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "isNonPassAliveSelfConnection";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(9,9,R"%%(
.x.oo.xx.
xoo.oxox.
.o.ooooox
ooo....x.
..o..x.xx
oo.o...x.
...xxxoxx
xxxx.x.x.
o.o.xxoxo
)%%");

    bool multiStoneSuicideLegal = false;
    bool nonPassAliveStones = false;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = false;
    bool recursivelyReachesSafe = false;
    int whiteMinusBlackSafeRegionCount = 0;
    board.calculateArea(result,whiteMinusBlackSafeRegionCount,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,recursivelyReachesSafe,multiStoneSuicideLegal);

    out << endl;
    out << "NonPassAliveSelfConn black" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_EMPTY)
          out << board.isNonPassAliveSelfConnection(loc,P_BLACK,result);
        else {
          testAssert(board.isNonPassAliveSelfConnection(loc,P_BLACK,result) == false);
          out << "-";
        }
      }
      out << endl;
    }
    out << endl;
    out << "NonPassAliveSelfConn white" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_EMPTY)
          out << board.isNonPassAliveSelfConnection(loc,P_WHITE,result);
        else {
          testAssert(board.isNonPassAliveSelfConnection(loc,P_WHITE,result) == false);
          out << "-";
        }
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(

NonPassAliveSelfConn black
0-0--1--0
---0----1
0-0------
---0000-1
00-00-1--
--0-010-0
000------
----0-0-0
-0-0-----

NonPassAliveSelfConn white
0-0--0--0
---0----0
0-0------
---0000-0
11-10-0--
--1-000-0
000------
----0-1-0
-0-0-----

)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "groupTaxOwnership";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(13,13,R"%%(
.xo.o.o...xx.
xoooooo...x.x
oo.........xx
..xx.........
xx..x........
.x..x........
ox..x........
.xx.x........
.x.x.........
xxxx........x
oooxooo...xx.
o.oxo.o...x.x
.o.o.o....xx.
)%%");

    int nnXLen = board.x_size;
    int nnYLen = board.y_size;

    bool multiStoneSuicideLegal = false;
    bool nonPassAliveStones = false;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = false;
    bool recursivelyReachesSafe = true;
    int whiteMinusBlackSafeRegionCount = 0;
    board.calculateArea(result,whiteMinusBlackSafeRegionCount,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,recursivelyReachesSafe,multiStoneSuicideLegal);

    float ownership[NNPos::MAX_BOARD_AREA];

    out << endl;
    out << "No group tax" << endl;
    NNInputs::fillOwnership(board,result,false,nnXLen,nnYLen,ownership);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        out << Global::strprintf("%4.0f ", ownership[pos]*100);
      }
      out << endl;
    }
    out << endl;
    out << "Group tax" << endl;
    NNInputs::fillOwnership(board,result,true,nnXLen,nnYLen,ownership);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        out << Global::strprintf("%4.0f ", ownership[pos]*100);
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(

No group tax
 100  100  100  100  100  100  100    0    0    0 -100 -100 -100
 100  100  100  100  100  100  100    0    0    0 -100 -100 -100
 100  100    0    0    0    0    0    0    0    0    0 -100 -100
   0    0 -100 -100    0    0    0    0    0    0    0    0    0
-100 -100 -100 -100 -100    0    0    0    0    0    0    0    0
-100 -100 -100 -100 -100    0    0    0    0    0    0    0    0
-100 -100 -100 -100 -100    0    0    0    0    0    0    0    0
-100 -100 -100 -100 -100    0    0    0    0    0    0    0    0
-100 -100 -100 -100    0    0    0    0    0    0    0    0    0
-100 -100 -100 -100    0    0    0    0    0    0    0    0 -100
 100  100  100 -100  100  100  100    0    0    0 -100 -100 -100
 100  100  100 -100  100  100  100    0    0    0 -100 -100 -100
 100  100  100  100  100  100    0    0    0    0 -100 -100 -100

Group tax
  60   60  100   60  100   60  100    0    0    0 -100 -100    0
  60  100  100  100  100  100  100    0    0    0 -100    0 -100
 100  100    0    0    0    0    0    0    0    0    0 -100 -100
   0    0 -100 -100    0    0    0    0    0    0    0    0    0
-100 -100  -83  -83 -100    0    0    0    0    0    0    0    0
 -83 -100  -83  -83 -100    0    0    0    0    0    0    0    0
 -83 -100  -83  -83 -100    0    0    0    0    0    0    0    0
 -83 -100 -100  -83 -100    0    0    0    0    0    0    0    0
 -83 -100  -83 -100    0    0    0    0    0    0    0    0    0
-100 -100 -100 -100    0    0    0    0    0    0    0    0 -100
 100  100  100 -100  100  100  100    0    0    0 -100 -100  -33
 100   60  100 -100  100   60  100    0    0    0 -100  -33 -100
  60  100   60  100   60  100    0    0    0    0 -100 -100  -33

)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "groupTaxOwnership2";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(13,13,R"%%(
.xo.ox.oooooo
x.ooox.oxxxxo
oooxxx.oxo..x
xxxx...oxo..x
oooxx..oxxxxo
.o.ox..oooooo
.ooxx........
ooxx.........
.............
oooxx....o...
oxxoxxxxxoxxx
ox.oxooooxooo
ox.oxo.o.x.o.
)%%");

    int nnXLen = board.x_size;
    int nnYLen = board.y_size;

    bool multiStoneSuicideLegal = true;
    bool nonPassAliveStones = true;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = false;
    bool recursivelyReachesSafe = true;
    int whiteMinusBlackSafeRegionCount = 0;
    board.calculateArea(result,whiteMinusBlackSafeRegionCount,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,recursivelyReachesSafe,multiStoneSuicideLegal);

    float ownership[NNPos::MAX_BOARD_AREA];

    out << endl;
    out << "No group tax" << endl;
    NNInputs::fillOwnership(board,result,false,nnXLen,nnYLen,ownership);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        out << Global::strprintf("%4.0f ", ownership[pos]*100);
      }
      out << endl;
    }
    out << endl;
    out << "Group tax" << endl;
    NNInputs::fillOwnership(board,result,true,nnXLen,nnYLen,ownership);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        out << Global::strprintf("%4.0f ", ownership[pos]*100);
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(

No group tax
   0 -100  100    0  100 -100    0  100  100  100  100  100  100
-100    0  100  100  100 -100    0  100 -100 -100 -100 -100  100
 100  100  100 -100 -100 -100    0  100 -100  100    0    0 -100
-100 -100 -100 -100    0    0    0  100 -100  100    0    0 -100
 100  100  100 -100 -100    0    0  100 -100 -100 -100 -100  100
   0  100    0  100 -100    0    0  100  100  100  100  100  100
   0  100  100 -100 -100    0    0    0    0    0    0    0    0
 100  100 -100 -100    0    0    0    0    0    0    0    0    0
   0    0    0    0    0    0    0    0    0    0    0    0    0
 100  100  100 -100 -100    0    0    0    0  100    0    0    0
 100 -100 -100  100 -100 -100 -100 -100 -100  100 -100 -100 -100
 100 -100    0  100 -100  100  100  100  100 -100  100  100  100
 100 -100    0  100 -100  100    0  100    0 -100    0  100    0

Group tax
   0 -100  100    0  100 -100    0  100  100  100  100  100  100
-100    0  100  100  100 -100    0  100 -100 -100 -100 -100  100
 100  100  100 -100 -100 -100    0  100 -100  100    0    0 -100
-100 -100 -100 -100    0    0    0  100 -100  100    0    0 -100
 100  100  100 -100 -100    0    0  100 -100 -100 -100 -100  100
   0  100    0  100 -100    0    0  100  100  100  100  100  100
   0  100  100 -100 -100    0    0    0    0    0    0    0    0
 100  100 -100 -100    0    0    0    0    0    0    0    0    0
   0    0    0    0    0    0    0    0    0    0    0    0    0
 100  100  100 -100 -100    0    0    0    0  100    0    0    0
 100 -100 -100  100 -100 -100 -100 -100 -100  100 -100 -100 -100
 100 -100    0  100 -100  100  100  100  100 -100  100  100  100
 100 -100    0  100 -100  100    0  100    0 -100    0  100    0

)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "groupTaxOwnership2 but with unsafe territories on";
    Color result[Board::MAX_ARR_SIZE];
    Board board = Board::parseBoard(13,13,R"%%(
.xo.ox.oooooo
x.ooox.oxxxxo
oooxxx.oxo..x
xxxx...oxo..x
oooxx..oxxxxo
.o.ox..oooooo
.ooxx........
ooxx.........
.............
oooxx....o...
oxxoxxxxxoxxx
ox.oxooooxooo
ox.oxo.o.x.o.
)%%");

    int nnXLen = board.x_size;
    int nnYLen = board.y_size;

    bool multiStoneSuicideLegal = true;
    bool nonPassAliveStones = true;
    bool safeBigTerritories = true;
    bool unsafeBigTerritories = true;
    bool recursivelyReachesSafe = true;
    int whiteMinusBlackSafeRegionCount = 0;
    board.calculateArea(result,whiteMinusBlackSafeRegionCount,nonPassAliveStones,safeBigTerritories,unsafeBigTerritories,recursivelyReachesSafe,multiStoneSuicideLegal);

    float ownership[NNPos::MAX_BOARD_AREA];

    out << endl;
    out << "No group tax" << endl;
    NNInputs::fillOwnership(board,result,false,nnXLen,nnYLen,ownership);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        out << Global::strprintf("%4.0f ", ownership[pos]*100);
      }
      out << endl;
    }
    out << endl;
    out << "Group tax" << endl;
    NNInputs::fillOwnership(board,result,true,nnXLen,nnYLen,ownership);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        out << Global::strprintf("%4.0f ", ownership[pos]*100);
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(
No group tax
-100 -100  100  100  100 -100    0  100  100  100  100  100  100
-100    0  100  100  100 -100    0  100 -100 -100 -100 -100  100
 100  100  100 -100 -100 -100    0  100 -100  100    0    0 -100
-100 -100 -100 -100    0    0    0  100 -100  100    0    0 -100
 100  100  100 -100 -100    0    0  100 -100 -100 -100 -100  100
 100  100  100  100 -100    0    0  100  100  100  100  100  100
 100  100  100 -100 -100    0    0    0    0    0    0    0    0
 100  100 -100 -100    0    0    0    0    0    0    0    0    0
   0    0    0    0    0    0    0    0    0    0    0    0    0
 100  100  100 -100 -100    0    0    0    0  100    0    0    0
 100 -100 -100  100 -100 -100 -100 -100 -100  100 -100 -100 -100
 100 -100    0  100 -100  100  100  100  100 -100  100  100  100
 100 -100    0  100 -100  100  100  100    0 -100    0  100  100

Group tax
   0 -100  100    0  100 -100    0  100  100  100  100  100  100
-100    0  100  100  100 -100    0  100 -100 -100 -100 -100  100
 100  100  100 -100 -100 -100    0  100 -100  100    0    0 -100
-100 -100 -100 -100    0    0    0  100 -100  100    0    0 -100
 100  100  100 -100 -100    0    0  100 -100 -100 -100 -100  100
  33  100   33  100 -100    0    0  100  100  100  100  100  100
  33  100  100 -100 -100    0    0    0    0    0    0    0    0
 100  100 -100 -100    0    0    0    0    0    0    0    0    0
   0    0    0    0    0    0    0    0    0    0    0    0    0
 100  100  100 -100 -100    0    0    0    0  100    0    0    0
 100 -100 -100  100 -100 -100 -100 -100 -100  100 -100 -100 -100
 100 -100    0  100 -100  100  100  100  100 -100  100  100  100
 100 -100    0  100 -100  100    0  100    0 -100    0  100    0

)%%";
    expect(name,out,expected);
  }

}
