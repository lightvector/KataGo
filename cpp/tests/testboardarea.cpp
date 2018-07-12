#include "../tests/tests.h"
using namespace TestCommon;

void Tests::runBoardAreaTests() {
  cout << "Running board area tests" << endl;
  ostringstream out;

  //============================================================================
  auto printAreas = [&out](const Board& board, Color result[Board::MAX_ARR_SIZE]) {
    for(int mode = 0; mode < 4; mode++) {
      bool multiStoneSuicideLegal = (mode % 2 == 1);
      bool requirePassAlive = (mode <= 1);
      Board copy(board);
      copy.calculateArea(result,requirePassAlive,multiStoneSuicideLegal);
      out << "Require pass alive " << requirePassAlive << " Suicide " << multiStoneSuicideLegal << endl;
      for(int y = 0; y<copy.y_size; y++) {
        for(int x = 0; x<copy.x_size; x++) {
          Loc loc = Location::getLoc(x,y,copy.x_size);
          out << colorToChar(result[loc]);
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
Require pass alive 1 Suicide 0
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Require pass alive 1 Suicide 1
......XXX
......XXX
.......XX
.........
.........
.......OO
......OOO
.....OOOO
.....OOOO

Require pass alive 0 Suicide 0
OOOOO.XXX
OOOOO.XXX
OO.....XX
.........
.........
X......OO
XXX...OOO
XXX..OOOO
XXXX.OOOO

Require pass alive 0 Suicide 1
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
    out.str("");
    out.clear();
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
    printAreas(board2,result);

    string expected = R"%%(
Require pass alive 1 Suicide 0
OOOOOOOOO
OO...XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Require pass alive 1 Suicide 1
.........
.........
.........
.........
.........
.........
.........
.........
.........

Require pass alive 0 Suicide 0
OOOOOOOOO
OOX..XX.O
O...XXX.O
O...XXX.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Require pass alive 0 Suicide 1
X.OOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OX..X...O
O.XOX...O
O.XXX...O
OOOOOOOOO

Require pass alive 1 Suicide 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Require pass alive 1 Suicide 1
.........
.........
.........
.........
.........
.........
.........
.........
.........

Require pass alive 0 Suicide 0
OOOOOOOOO
OOX..XX.O
O...XOX.O
O...X.X.O
OXXXXXX.O
OXXXX...O
O.XXX...O
O.XXX...O
OOOOOOOOO

Require pass alive 0 Suicide 1
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
    out.str("");
    out.clear();
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
Require pass alive 1 Suicide 0
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

Require pass alive 1 Suicide 1
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

Require pass alive 0 Suicide 0
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

Require pass alive 0 Suicide 1
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
    out.str("");
    out.clear();
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
Require pass alive 1 Suicide 0
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

Require pass alive 1 Suicide 1
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

Require pass alive 0 Suicide 0
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

Require pass alive 0 Suicide 1
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
    out.str("");
    out.clear();
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
Require pass alive 1 Suicide 0
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

Require pass alive 1 Suicide 1
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

Require pass alive 0 Suicide 0
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

Require pass alive 0 Suicide 1
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
    out.str("");
    out.clear();
  }
}
