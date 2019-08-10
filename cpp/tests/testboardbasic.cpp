#include "../tests/tests.h"

using namespace std;
using namespace TestCommon;

void Tests::runBoardIOTests() {
  cout << "Running board IO tests" << endl;
  ostringstream out;

  //============================================================================
  {
    const char* name = "Location parse test";
    auto testLoc = [&out](const char* s, int xSize, int ySize) {
      try {
        Loc loc = Location::ofString(s,xSize,ySize);
        out << s << " " << Location::toString(loc,xSize,ySize) << " x " << Location::getX(loc,xSize) << " y " << Location::getY(loc,xSize) << endl;
      }
      catch(const StringError& e) {
        out << e.what() << endl;
      }
    };

    constexpr int numSizes = 4;
    int sizes[numSizes] = {9,19,26,70};
    for(int i = 0; i<numSizes; i++) {
      for(int j = 0; j<numSizes; j++) {
        if(i-j > 1 || j-i > 1)
          continue;
        int xSize = sizes[i];
        int ySize = sizes[j];
        out << "----------------------------------" << endl;
        out << xSize << " " << ySize << endl;

        testLoc("A1",xSize,ySize);
        testLoc("A0",xSize,ySize);
        testLoc("B2",xSize,ySize);
        testLoc("b2",xSize,ySize);
        testLoc("A",xSize,ySize);
        testLoc("B",xSize,ySize);
        testLoc("1",xSize,ySize);
        testLoc("pass",xSize,ySize);
        testLoc("H9",xSize,ySize);
        testLoc("I9",xSize,ySize);
        testLoc("J9",xSize,ySize);
        testLoc("J10",xSize,ySize);
        testLoc("K8",xSize,ySize);
        testLoc("k19",xSize,ySize);
        testLoc("a22",xSize,ySize);
        testLoc("y1",xSize,ySize);
        testLoc("z1",xSize,ySize);
        testLoc("aa1",xSize,ySize);
        testLoc("AA26",xSize,ySize);
        testLoc("AZ26",xSize,ySize);
        testLoc("BC50",xSize,ySize);
      }
    }

    string expected = R"%%(
----------------------------------
9 9
A1 A1 x 0 y 8
Could not parse board location: A0
B2 B2 x 1 y 7
b2 B2 x 1 y 7
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 0
Could not parse board location: I9
J9 J9 x 8 y 0
Could not parse board location: J10
Could not parse board location: K8
Could not parse board location: k19
Could not parse board location: a22
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
9 19
A1 A1 x 0 y 18
Could not parse board location: A0
B2 B2 x 1 y 17
b2 B2 x 1 y 17
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 10
Could not parse board location: I9
J9 J9 x 8 y 10
J10 J10 x 8 y 9
Could not parse board location: K8
Could not parse board location: k19
Could not parse board location: a22
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
19 9
A1 A1 x 0 y 8
Could not parse board location: A0
B2 B2 x 1 y 7
b2 B2 x 1 y 7
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 0
Could not parse board location: I9
J9 J9 x 8 y 0
Could not parse board location: J10
K8 K8 x 9 y 1
Could not parse board location: k19
Could not parse board location: a22
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
19 19
A1 A1 x 0 y 18
Could not parse board location: A0
B2 B2 x 1 y 17
b2 B2 x 1 y 17
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 10
Could not parse board location: I9
J9 J9 x 8 y 10
J10 J10 x 8 y 9
K8 K8 x 9 y 11
k19 K19 x 9 y 0
Could not parse board location: a22
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
19 26
A1 A1 x 0 y 25
Could not parse board location: A0
B2 B2 x 1 y 24
b2 B2 x 1 y 24
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 17
Could not parse board location: I9
J9 J9 x 8 y 17
J10 J10 x 8 y 16
K8 K8 x 9 y 18
k19 K19 x 9 y 7
a22 A22 x 0 y 4
Could not parse board location: y1
Could not parse board location: z1
Could not parse board location: aa1
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
26 19
A1 A1 x 0 y 18
Could not parse board location: A0
B2 B2 x 1 y 17
b2 B2 x 1 y 17
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 10
Could not parse board location: I9
J9 J9 x 8 y 10
J10 J10 x 8 y 9
K8 K8 x 9 y 11
k19 K19 x 9 y 0
Could not parse board location: a22
y1 Y1 x 23 y 18
z1 Z1 x 24 y 18
aa1 AA1 x 25 y 18
Could not parse board location: AA26
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
26 26
A1 A1 x 0 y 25
Could not parse board location: A0
B2 B2 x 1 y 24
b2 B2 x 1 y 24
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 17
Could not parse board location: I9
J9 J9 x 8 y 17
J10 J10 x 8 y 16
K8 K8 x 9 y 18
k19 K19 x 9 y 7
a22 A22 x 0 y 4
y1 Y1 x 23 y 25
z1 Z1 x 24 y 25
aa1 AA1 x 25 y 25
AA26 AA26 x 25 y 0
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
26 70
A1 A1 x 0 y 69
Could not parse board location: A0
B2 B2 x 1 y 68
b2 B2 x 1 y 68
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 61
Could not parse board location: I9
J9 J9 x 8 y 61
J10 J10 x 8 y 60
K8 K8 x 9 y 62
k19 K19 x 9 y 51
a22 A22 x 0 y 48
y1 Y1 x 23 y 69
z1 Z1 x 24 y 69
aa1 AA1 x 25 y 69
AA26 AA26 x 25 y 44
Could not parse board location: AZ26
Could not parse board location: BC50
----------------------------------
70 26
A1 A1 x 0 y 25
Could not parse board location: A0
B2 B2 x 1 y 24
b2 B2 x 1 y 24
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 17
Could not parse board location: I9
J9 J9 x 8 y 17
J10 J10 x 8 y 16
K8 K8 x 9 y 18
k19 K19 x 9 y 7
a22 A22 x 0 y 4
y1 Y1 x 23 y 25
z1 Z1 x 24 y 25
aa1 AA1 x 25 y 25
AA26 AA26 x 25 y 0
AZ26 AZ26 x 49 y 0
Could not parse board location: BC50
----------------------------------
70 70
A1 A1 x 0 y 69
Could not parse board location: A0
B2 B2 x 1 y 68
b2 B2 x 1 y 68
Could not parse board location: A
Could not parse board location: B
Could not parse board location: 1
pass pass x 0 y -1
H9 H9 x 7 y 61
Could not parse board location: I9
J9 J9 x 8 y 61
J10 J10 x 8 y 60
K8 K8 x 9 y 62
k19 K19 x 9 y 51
a22 A22 x 0 y 48
y1 Y1 x 23 y 69
z1 Z1 x 24 y 69
aa1 AA1 x 25 y 69
AA26 AA26 x 25 y 44
AZ26 AZ26 x 49 y 44
BC50 BC50 x 52 y 20
)%%";
    expect(name,out,expected);
  }


  //============================================================================
  {
    const char* name = "Parse test";
    Board board = Board::parseBoard(6,5,R"%%(
 ABCDEF
5......
4......
3......
2......
1......
)%%");
    Board board2 = Board::parseBoard(6,5,R"%%(
   A B C D E F
10 . . . . . .
 9 . . . . . .
 8 . . . . . .
 7 . X . . . .
 6 . . . . . .
)%%");

    board.playMove(Location::ofString("B2",board),P_BLACK,true);
    board2.playMove(Location::ofString("F1",board),P_WHITE,true);
    out << board << endl;
    out << board2 << endl;

    string expected = R"%%(
HASH: FF41A6A8C248603FA60347F93F085846
   A B C D E F
 5 . . . . . .
 4 . . . . . .
 3 . . . . . .
 2 . X . . . .
 1 . . . . . .


HASH: 6B93B11D3BA70C1DF1D07EE065566210
   A B C D E F
 5 . . . . . .
 4 . . . . . .
 3 . . . . . .
 2 . X . . . .
 1 . . . . . O

)%%";

    expect(name,out,expected);
  }
}

void Tests::runBoardBasicTests() {
  cout << "Running board basic tests" << endl;
  ostringstream out;

  //============================================================================
  {
    const char* name = "Liberties";
    Board board = Board::parseBoard(9,9,R"%%(
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
  }

  //============================================================================
  {
    const char* name = "Liberties after move";
    Board board = Board::parseBoard(9,9,R"%%(
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
  }

  //============================================================================
  {
    const char* name = "Liberties after move capped at 2";
    Board board = Board::parseBoard(9,9,R"%%(
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
          out << board.getNumLibertiesAfterPlay(loc,P_BLACK,2);
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
          out << board.getNumLibertiesAfterPlay(loc,P_WHITE,2);
        else
          out << "-";
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(
After black
222222222
22222-222
22--22-22
22-222222
222222--2
22-22--22
2----0---
----1-1--
2-22222--

After white
222222222
22222-222
22--22-22
22-222222
222222--2
22-22--12
2----2---
----2-2--
0-22222--
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Liberties after move capped at 3";
    Board board = Board::parseBoard(9,9,R"%%(
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
          out << board.getNumLibertiesAfterPlay(loc,P_BLACK,3);
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
          out << board.getNumLibertiesAfterPlay(loc,P_WHITE,3);
        else
          out << "-";
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(
After black
233333332
33333-333
33--33-33
33-333333
333333--3
33-32--32
3----0---
----1-1--
2-32322--

After white
233332332
33333-233
33--32-33
33-333233
332333--2
33-33--12
2----3---
----3-3--
0-33332--
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Liberties after move 2";
    Board board = Board::parseBoard(9,9,R"%%(
x.xx...xx
oooxo.oxo
xxxxo.ox.
ooooo..ox
..xxx..o.
........o
x......xo
ox....xox
.ox...xo.
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
-4--232--
-----2---
-----2--3
-----32--
26---73-1
34656442-
-444445--
--4444---
2--333--2

After white
-1--634--
-----8---
-----7--0
-----77--
66---35-3
24333444-
-244442--
--2443---
0--232--1
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Liberties after move 2 capped at 3";
    Board board = Board::parseBoard(9,9,R"%%(
x.xx...xx
oooxo.oxo
xxxxo.ox.
ooooo..ox
..xxx..o.
........o
x......xo
ox....xox
.ox...xo.
)%%");

    out << endl;
    out << "After black" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_EMPTY)
          out << board.getNumLibertiesAfterPlay(loc,P_BLACK,3);
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
          out << board.getNumLibertiesAfterPlay(loc,P_WHITE,3);
        else
          out << "-";
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(
After black
-3--232--
-----2---
-----2--3
-----32--
23---33-1
33333332-
-333333--
--3333---
2--333--2

After white
-1--333--
-----3---
-----3--0
-----33--
33---33-3
23333333-
-233332--
--2333---
0--232--1
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "IsAdjacentToPla";
    Board board = Board::parseBoard(9,9,R"%%(
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
    out << "Adj black" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << (int)board.isAdjacentToPla(loc,P_BLACK);
      }
      out << endl;
    }
    out << endl;
    out << "Adj white" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << (int)board.isAdjacentToPla(loc,P_WHITE);
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(
Adj black
000001000
000010100
001001010
010100110
001001111
011101111
111100111
111000011
111000001

Adj white
000000000
001100000
011110000
001100000
000001000
010110100
101111010
011110111
001101111

)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "wouldBeKoCapture";
    Board board = Board::parseBoard(9,9,R"%%(
.....oxx.
..o.o.oox
.oxo.oxx.
.o.o..x..
.xox.x.xo
..x..oxo.
....o.oxx
xo...oxox
.xo..x.oo
)%%");

    out << endl;
    out << "WouldBeKo black" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << (int)board.wouldBeKoCapture(loc,P_BLACK);
      }
      out << endl;
    }
    out << endl;
    out << "WouldBeKo white" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << (int)board.wouldBeKoCapture(loc,P_WHITE);
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(
WouldBeKo black
000000000
000000000
000000000
000000000
000000000
000000000
000001000
000000000
000000000

WouldBeKo white
000000000
000000000
000000000
000000000
000000100
000000000
000000000
000000000
100000000

)%%";
    expect(name,out,expected);
  }


  //============================================================================
  {
    const char* name = "Ladders 1 Lib";
    vector<Loc> buf;
    Board board = Board::parseBoard(9,9,R"%%(
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
    Board startBoard = board;
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
    board.checkConsistency();
    testAssert(boardsSeemEqual(board,startBoard));

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
  }

  //============================================================================
  {
    const char* name = "Ladders 2 Libs";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = Board::parseBoard(9,9,R"%%(
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
    Board startBoard = board;
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
    board.checkConsistency();
    testAssert(boardsSeemEqual(board,startBoard));

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
  }

  //============================================================================
  {
    const char* name = "LaddersKo-1";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = Board::parseBoard(18,9,R"%%(
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
    Board startBoard = board;
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
    board.checkConsistency();
    testAssert(boardsSeemEqual(board,startBoard));

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
  }

  //============================================================================
  {
    const char* name = "LaddersKo-2";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = Board::parseBoard(18,9,R"%%(
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
    Board board2 = Board::parseBoard(18,9,R"%%(
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
    Board board3 = Board::parseBoard(18,9,R"%%(
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
    board.checkConsistency();
    board2.checkConsistency();
    board3.checkConsistency();

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
  }

  //============================================================================
  {
    const char* name = "LaddersMultiKo";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = Board::parseBoard(9,9,R"%%(
.xxxxxxo.
xoxox.xo.
ooo.oxoo.
..oooo...
.xxxx....
xx.x.xx..
xoxoxox..
xoooooxx.
o.ooo.ox.
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
    board.checkConsistency();

    string expected = R"%%(
.0000000.
00000.00.
000.0000.
..0000...
.0000....
00.0.00..
0000000..
00000000.
0.000.00.
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "LaddersBigBoard";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . . . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . . . . . . . . . . . . X O O . .
 3 . . . O . . . . . . . . . . O X X . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .
)%%");

    out << endl;
    Board startBoard = board;
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
    board.checkConsistency();
    testAssert(boardsSeemEqual(board,startBoard));

    string expected = R"%%(
...................
...................
...................
...0...........0...
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
..............000..
...0..........000..
...................
...................
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "WholeBoardLadder";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 . . . . O . . . . . . . . . . . . . .
18 O . . . . . . . . . . . . . . . . . O
17 . . . . . . . . . . . . . . . . . . O
16 . . . O O X O . . . . . . . . . . . .
15 O . . . O X O O O O . O . . . . . O .
14 O X X X X X X X X X O . . . . . . . .
13 X O O O . X . . . . X . . . . . . . O
12 X . O O O X . . . . X O . . . . . . .
11 X O O O O X . . . . X . . . . . . . .
10 O X X X X X X X X X O . . . . . . . O
 9 O O . . . X O . . O . . . . . . . . .
 8 . O . O . X . O . . . . . . . . . . .
 7 . . . . . . O . . . . . . . . . . . .
 6 . . . . O . . . . X . . . . . . . . .
 5 O . . . . X . . . X . . . X O . . . .
 4 . . . . . X . . . X . . . X . . . . .
 3 . . . . . X . . . X . . . X . . . . .
 2 . . . . O . X X X X X X X O . . . O .
 1 . O . . . . O O O O O O O . O . . . .
)%%");

    out << endl;
    Board startBoard = board;
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
    board.checkConsistency();
    testAssert(boardsSeemEqual(board,startBoard));

    string expected = R"%%(
....0..............
0.................0
..................0
...0000............
0...000000.0.....0.
00000000000........
0000.0....0.......0
0.0000....00.......
000000....0........
00000000000.......0
00...00..0.........
.0.0.0.0...........
......0............
....0....0.........
0....0...0...00....
.....0...0...0.....
.....0...0...0.....
....0.00000000...0.
.0....1111111.0....
)%%";
    expect(name,out,expected);
  }


  //============================================================================
  {
    const char* name = "CubicLadder not far from max node budget";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 . . . O O O O O O O O . . . . . O . O
18 X X X . X X X X X . . . O O O O O . O
17 O . . . . . O O . . . . . . . X X X .
16 . . . . . . . O O O . . . . . . O . .
15 O . O . . . . . . . . . . . . . . . .
14 . . O . . . . . . O . . . . . . O . O
13 . . O . . . . . . . X . . . . . O . .
12 . . . . . . . . . O X . . . . . . . .
11 . . . . . . . X . X . O . O . . . . .
10 O X X . . . . . X . O . . . . . . . .
 9 O X . . . . . . X O . . . . . . . . .
 8 . . X O . . . . . X . . . . . . . . .
 7 O . . . X . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 O . . . . . . . . . . . . . . . O . .
 3 . . . . . . . . . X O . . . . . O . .
 2 . . . O O . . . . X O X . X . . . X .
 1 . . . . . . X . O O O X . X . . O O O
)%%");

    out << endl;
    Board startBoard = board;
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
    board.checkConsistency();
    testAssert(boardsSeemEqual(board,startBoard));

    string expected = R"%%(
...00000000.....0.0
000.00000...00000.0
0.....00.......000.
.......000......0..
0.0................
..0......0......0.0
..0.......0.....0..
.........00........
.......0.0.0.0.....
000.....0.0........
00......01.........
..00.....0.........
0...0..............
...................
...................
0...............0..
.........00.....0..
...00....000.0...0.
......0.0000.0..000
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    //Requires a node budget of somewhere between 2.5M and 25M nodes
    const char* name = "Failing polynomial ladder due to max node budget";
    vector<Loc> buf;
    vector<Loc> buf2;
    Board board = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 X . O O O O X . O O O O X . O O O O O
18 O X X X X O O X X X X O O X X X X X O
17 O X . . X X O X . . X X O X . . . X O
16 O . . . . X O X . . . X O X . . . X O
15 . . . . . . O X . . . . O X . . . X O
14 . . . . . . . . . . . . . . . . . X O
13 . . . . . . . . . . . . . . . X X O O
12 . . . . . . . . . . . . . . . O O O X
11 . . . . . . . . . . . . . . . . X X .
10 . . . . . . . . . . . . . . . . . X O
 9 . . . . . . . . . . . . . . . . . X O
 8 . . . . . . . . . . . . . . . . . X O
 7 . . . . . . . . . . . . . . . X X O O
 6 . . . . . . . . . . . . . . . O O O X
 5 . . . . . . . . . . . . . . . . X X .
 4 . . . . . . . . . . . . . . . . . X O
 3 . . . . . . . . . . . . . . . . . X O
 2 . X . . . X . . . . . . . . . . X . O
 1 . . . . . . . . . . . . . . . O O O .
)%%");

    out << endl;
    Board startBoard = board;
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
    board.checkConsistency();
    testAssert(boardsSeemEqual(board,startBoard));

    string expected = R"%%(
0.00000.00000.00000
0000000000000000000
00..0000..0000...00
0....000...000...00
......00....00...00
.................00
...............0000
...............0000
................00.
.................00
.................00
.................00
...............0000
...............0000
................00.
.................00
.................00
.0...0..........0.0
...............000.
)%%";
    expect(name,out,expected);
  }

}


void Tests::runBoardUndoTest() {
  cout << "Running board undo test" << endl;
  Rand rand("runBoardUndoTests");

  int suicideCount = 0;
  int koCaptureCount = 0;
  int passCount = 0;
  int regularMoveCount = 0;
  auto run = [&](const Board& startBoard, bool multiStoneSuicideLegal) {
    static const int steps = 1000;
    Board* boards = new Board[steps+1];
    Board::MoveRecord records[steps];

    boards[0] = startBoard;
    for(int n = 1; n <= steps; n++) {
      boards[n] = boards[n-1];
      Loc loc;
      Player pla;
      while(true) {
        pla = rand.nextUInt(2) == 0 ? P_BLACK : P_WHITE;
        //Maximum range of board location values when 19x19:
        int numLocs = (19+1)*(19+2)+1;
        loc = (Loc)rand.nextUInt(numLocs);
        if(boards[n].isLegal(loc,pla,multiStoneSuicideLegal))
          break;
      }

      records[n-1] = boards[n].playMoveRecorded(loc,pla);

      if(loc == Board::PASS_LOC)
        passCount++;
      else if(boards[n-1].isSuicide(loc,pla))
        suicideCount++;
      else {
        if(boards[n].ko_loc != Board::NULL_LOC)
          koCaptureCount++;
        regularMoveCount++;
      }
    }

    Board board = boards[steps];
    for(int n = steps-1; n >= 0; n--) {
      board.undo(records[n]);
      testAssert(boardsSeemEqual(boards[n],board));
      board.checkConsistency();
    }
    delete[] boards;
  };

  run(Board(19,19),true);
  run(Board(4,4),true);
  run(Board(4,4),false);

  ostringstream out;
  out << endl;
  out << "regularMoveCount " << regularMoveCount << endl;
  out << "passCount " << passCount << endl;
  out << "koCaptureCount " << koCaptureCount << endl;
  out << "suicideCount " << suicideCount << endl;

  string expected = R"%%(

regularMoveCount 2431
passCount 482
koCaptureCount 25
suicideCount 87

)%%";
  expect("Board undo test move counts",out,expected);
}


void Tests::runBoardStressTest() {
  cout << "Running board stress test" << endl;
  Rand rand("runBoardStressTests");

  static const int numBoards = 4;
  Board boards[numBoards];
  boards[0] = Board();
  boards[1] = Board(9,16);
  boards[2] = Board(13,7);
  boards[3] = Board(4,4);
  bool multiStoneSuicideLegal[4] = {false,false,true,false};
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
      isLegal[i] = boards[i].isLegal(locs[i],pla,multiStoneSuicideLegal[i]);
      testAssert(boardsSeemEqual(copies[i],boards[i]));
      suc[i] = boards[i].playMove(locs[i],pla,multiStoneSuicideLegal[i]);
    }

    for(int i = 0; i<numBoards; i++) {
      testAssert(isLegal[i] == suc[i]);
      boards[i].checkConsistency();

      const Board& board = boards[i];
      const Board& copy = copies[i];
      Loc loc = locs[i];
      if(!suc[i]) {
        if(board.isOnBoard(loc)) {
          testAssert(boardsSeemEqual(copy,board));
          testAssert(loc < 0 || loc >= Board::MAX_ARR_SIZE || board.colors[loc] != C_EMPTY || board.isIllegalSuicide(loc,pla,multiStoneSuicideLegal[i]) || board.isKoBanned(loc));
          if(board.isKoBanned(loc)) {
            testAssert(board.colors[loc] == C_EMPTY && (board.wouldBeKoCapture(loc,C_BLACK) || board.wouldBeKoCapture(loc,C_WHITE)));
            koBanCount++;
          }
        }
      }
      else {
        if(loc == Board::PASS_LOC) {
          testAssert(boardsSeemEqual(copy,board));
          testAssert(board.ko_loc == Board::NULL_LOC);
          passCount++;
        }
        else if(copy.isSuicide(loc,pla)) {
          testAssert(board.colors[loc] == C_EMPTY);
          testAssert(board.isLegal(loc,pla,multiStoneSuicideLegal[i]));
          testAssert(multiStoneSuicideLegal[i]);
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

    pla = rand.nextUInt(2) == 0 ? getOpp(pla) : pla;
  }

  ostringstream out;
  out << endl;
  out << "regularMoveCount " << regularMoveCount << endl;
  out << "passCount " << passCount << endl;
  out << "koCaptureCount " << koCaptureCount << endl;
  out << "koBanCount " << koBanCount << endl;
  out << "suicideCount " << suicideCount << endl;

  for(int i = 0; i<4; i++)
    out << "Caps " << boards[i].numBlackCaptures << " " << boards[i].numWhiteCaptures << endl;
  string expected = R"%%(

regularMoveCount 37692
passCount 280
koCaptureCount 164
koBanCount 25
suicideCount 445
Caps 4862 4732
Caps 4590 4745
Caps 4890 5071
Caps 4364 4393

)%%";
  expect("Board stress test move counts",out,expected);
}
