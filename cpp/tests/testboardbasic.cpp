#include "../tests/tests.h"

#include "../game/graphhash.h"
#include "../program/playutils.h"

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
HASH: 27902CF3F972B1855303DB453BF8DE63
   A B C D E F
 5 . . . . . .
 4 . . . . . .
 3 . . . . . .
 2 . X . . . .
 1 . . . . . .


HASH: BC864FD8F525EE9D6B590472C800841A
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
    const char* name = "Distance";
    Board board(17,12);

    auto testDistance = [&](int x0, int y0, int x1, int y1) {
      out << "distance (" << x0 << "," << y0 << ") (" << x1 << "," << y1 << ") = " <<
      Location::distance(Location::getLoc(x0,y0,board.x_size),Location::getLoc(x1,y1,board.x_size),board.x_size) << endl;
    };
    auto testEuclideanDistance = [&](int x0, int y0, int x1, int y1) {
      out << "euclideanSq (" << x0 << "," << y0 << ") (" << x1 << "," << y1 << ") = " <<
      Location::euclideanDistanceSquared(Location::getLoc(x0,y0,board.x_size),Location::getLoc(x1,y1,board.x_size),board.x_size) << endl;
    };
    testDistance(13,6,12,3);
    testDistance(13,6,12,4);
    testDistance(13,6,12,5);
    testDistance(13,6,12,6);
    testDistance(13,6,12,7);
    testDistance(13,6,13,3);
    testDistance(13,6,13,4);
    testDistance(13,6,13,5);
    testDistance(13,6,13,6);
    testDistance(13,6,13,7);
    testDistance(13,6,14,3);
    testDistance(13,6,14,4);
    testDistance(13,6,14,5);
    testDistance(13,6,14,6);
    testDistance(13,6,14,7);
    testDistance(13,6,15,3);
    testDistance(13,6,15,4);
    testDistance(13,6,15,5);
    testDistance(13,6,15,6);
    testDistance(13,6,15,7);
    testDistance(13,6,0,0);
    testDistance(13,6,16,11);
    testDistance(13,6,0,11);
    testDistance(13,6,16,0);
    testEuclideanDistance(13,6,12,3);
    testEuclideanDistance(13,6,12,4);
    testEuclideanDistance(13,6,12,5);
    testEuclideanDistance(13,6,12,6);
    testEuclideanDistance(13,6,12,7);
    testEuclideanDistance(13,6,13,3);
    testEuclideanDistance(13,6,13,4);
    testEuclideanDistance(13,6,13,5);
    testEuclideanDistance(13,6,13,6);
    testEuclideanDistance(13,6,13,7);
    testEuclideanDistance(13,6,14,3);
    testEuclideanDistance(13,6,14,4);
    testEuclideanDistance(13,6,14,5);
    testEuclideanDistance(13,6,14,6);
    testEuclideanDistance(13,6,14,7);
    testEuclideanDistance(13,6,15,3);
    testEuclideanDistance(13,6,15,4);
    testEuclideanDistance(13,6,15,5);
    testEuclideanDistance(13,6,15,6);
    testEuclideanDistance(13,6,15,7);
    testEuclideanDistance(13,6,0,0);
    testEuclideanDistance(13,6,16,11);
    testEuclideanDistance(13,6,0,11);
    testEuclideanDistance(13,6,16,0);

    string expected = R"%%(
distance (13,6) (12,3) = 4
distance (13,6) (12,4) = 3
distance (13,6) (12,5) = 2
distance (13,6) (12,6) = 1
distance (13,6) (12,7) = 2
distance (13,6) (13,3) = 3
distance (13,6) (13,4) = 2
distance (13,6) (13,5) = 1
distance (13,6) (13,6) = 0
distance (13,6) (13,7) = 1
distance (13,6) (14,3) = 4
distance (13,6) (14,4) = 3
distance (13,6) (14,5) = 2
distance (13,6) (14,6) = 1
distance (13,6) (14,7) = 2
distance (13,6) (15,3) = 5
distance (13,6) (15,4) = 4
distance (13,6) (15,5) = 3
distance (13,6) (15,6) = 2
distance (13,6) (15,7) = 3
distance (13,6) (0,0) = 19
distance (13,6) (16,11) = 8
distance (13,6) (0,11) = 18
distance (13,6) (16,0) = 9
euclideanSq (13,6) (12,3) = 10
euclideanSq (13,6) (12,4) = 5
euclideanSq (13,6) (12,5) = 2
euclideanSq (13,6) (12,6) = 1
euclideanSq (13,6) (12,7) = 2
euclideanSq (13,6) (13,3) = 9
euclideanSq (13,6) (13,4) = 4
euclideanSq (13,6) (13,5) = 1
euclideanSq (13,6) (13,6) = 0
euclideanSq (13,6) (13,7) = 1
euclideanSq (13,6) (14,3) = 10
euclideanSq (13,6) (14,4) = 5
euclideanSq (13,6) (14,5) = 2
euclideanSq (13,6) (14,6) = 1
euclideanSq (13,6) (14,7) = 2
euclideanSq (13,6) (15,3) = 13
euclideanSq (13,6) (15,4) = 8
euclideanSq (13,6) (15,5) = 5
euclideanSq (13,6) (15,6) = 4
euclideanSq (13,6) (15,7) = 5
euclideanSq (13,6) (0,0) = 205
euclideanSq (13,6) (16,11) = 34
euclideanSq (13,6) (0,11) = 194
euclideanSq (13,6) (16,0) = 45
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
    const char* name = "wouldBeCapture";
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
    out << "WouldBeCapture black" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << (int)board.wouldBeCapture(loc,P_BLACK);
      }
      out << endl;
    }
    out << endl;
    out << "WouldBeCapture white" << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << (int)board.wouldBeCapture(loc,P_WHITE);
      }
      out << endl;
    }
    out << endl;

    string expected = R"%%(
WouldBeCapture black
000000000
000001000
000000000
001000000
000000000
000000001
000001000
000000000
000000100

WouldBeCapture white
000000001
000000000
000000000
001000000
000000100
000000001
000000000
000000000
100000100
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

  //============================================================================
  {
    const char* name = "simpleRepetitionBoundOpen";
    Board board = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . . . . . . . . . . . . . O . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . O . . . . . . . . . . . . . . . .
 6 . . X O . . . . . . . . . . . . . . .
 5 . . X . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . . . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .
)%%");

    Board startBoard = board;
    for(Player pla = 0; pla <= 2; pla++) {
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          if(board.colors[loc] != C_EMPTY) {
            if(pla != C_EMPTY) {
              out << "   " << PlayerIO::colorToChar(board.colors[loc]);
            }
            else {
              int bound;
              for(bound = 0; bound < 362; bound++) {
                if(!board.simpleRepetitionBoundGt(loc,bound))
                  break;
              }
              out << Global::strprintf(" %3d", bound);
              for(; bound < 362; bound++) {
                testAssert(!board.simpleRepetitionBoundGt(loc,bound));
              }
            }
            board.checkConsistency();
          }
          else {
            if(pla == C_EMPTY) {
              out << "   .";
            }
            else {
              Board boardCopy = board;
              boardCopy.clearSimpleKoLoc();
              bool isMultiStoneSuicideLegal = true;
              bool suc = boardCopy.playMove(loc, pla, isMultiStoneSuicideLegal);
              if(!suc)
                out << "   .";
              else {
                int bound;
                for(bound = 0; bound < 362; bound++) {
                  if(!boardCopy.simpleRepetitionBoundGt(loc,bound))
                    break;
                }
                out << Global::strprintf(" %3d", bound);
                for(; bound < 362; bound++) {
                  testAssert(!boardCopy.simpleRepetitionBoundGt(loc,bound));
                }
              }
              boardCopy.checkConsistency();
            }
          }
        }
        out << endl;
      }
      out << endl;
    }
    board.checkConsistency();
    testAssert(boardsSeemEqual(board,startBoard));

    string expected = R"%%(
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   . 356   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   . 356   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   . 357 356   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   . 357   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   . 356   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .
   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .

 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355   O 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355   O 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 357   X   O 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 357   X 358 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 358   X 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355

 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 356 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 356   O 356 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 356 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 356   O 357 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355   X   O 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355   X 356 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355   X 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355 355
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "simpleRepetitionBound";
    Board board = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 . . X O . X O . X O O . . . . O X . .
18 . . . X X O O X . . X O O . O X X X X
17 . X . . X X X X X X X . . O O O X O .
16 X . . X O X O O O O X X O O X O X X .
15 O X X . O O . O . . X . O X X X O X .
14 . O O O . . X O X X X O O . X X O X O
13 O . . . . O O X O O O O . O X O O O .
12 X O O O O . X X . . O X X . X O . O X
11 . X . X . X . . X . O . X . X X X O .
10 X . . X . . X O X X O O X O . X O . O
 9 . X . O X . X X O . X O . O X . O . .
 8 . X . O . . X O O . X O O X X X O X .
 7 . O X . . O O . . X X O . O X X O O .
 6 X X O X . . . . X . X O O . . X X O .
 5 O . O O O O O X . . . X O O O X O X .
 4 . O . O X X . O X X X X X O X X O . .
 3 . . . O X O O O X . . X O X . X O . .
 2 . . O X X O X X O . X . O X . . O X .
 1 . . O . . X . O O . . X O . . . O . .
)%%");

    Board startBoard = board;
    for(Player pla = 0; pla <= 2; pla++) {
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          if(board.colors[loc] != C_EMPTY) {
            if(pla != C_EMPTY) {
              out << "   " << PlayerIO::colorToChar(board.colors[loc]);
            }
            else {
              int bound;
              for(bound = 0; bound < 362; bound++) {
                if(!board.simpleRepetitionBoundGt(loc,bound))
                  break;
              }
              out << Global::strprintf(" %3d", bound);
              for(; bound < 362; bound++) {
                testAssert(!board.simpleRepetitionBoundGt(loc,bound));
              }
            }
            board.checkConsistency();
          }
          else {
            if(pla == C_EMPTY) {
              out << "   .";
            }
            else {
              Board boardCopy = board;
              boardCopy.clearSimpleKoLoc();
              bool isMultiStoneSuicideLegal = true;
              bool suc = boardCopy.playMove(loc, pla, isMultiStoneSuicideLegal);
              if(!suc)
                out << "   .";
              else {
                int bound;
                for(bound = 0; bound < 362; bound++) {
                  if(!boardCopy.simpleRepetitionBoundGt(loc,bound))
                    break;
                }
                out << Global::strprintf(" %3d", bound);
                for(; bound < 362; bound++) {
                  testAssert(!boardCopy.simpleRepetitionBoundGt(loc,bound));
                }
              }
              boardCopy.checkConsistency();
            }
          }
        }
        out << endl;
      }
      out << endl;
    }
    board.checkConsistency();
    testAssert(boardsSeemEqual(board,startBoard));

    string expected = R"%%(
   .   .  11   2   .   2   4   .   4   9   9   .   .   .   .   6  15   .   .
   .   .   .  37  37   4   4  37   .   .  37   9   9   .  46  15  15  15  15
   .  11   .   .  37  37  37  37  37  37  37   .   .  46  46  46  15   4   .
  11   .   .  12  11  37   9   9   9   9  37  37  46  46  18  46  15  15   .
   2  13  13   .  11  11   .   9   .   .  37   .  46  18  18  18  25  15   .
   .  11  11  11   .   .   8   9  37  37  37  46  46   .  18  18  25  15   5
   8   .   .   .   .   9   9   9  46  46  46  46   .   5  18  25  25  25   .
   2  30  30  30  30   .   9   9   .   .  46   9   9   .  18  25   .  25   3
   .   7   .  21   .  18   .   .  10   .  46   .   9   .  18  18  18  25   .
  10   .   .  21   .   .  20   3  10  10  46  46   9   6   .  18  21   .  16
   .  10   .  21  15   .  20  20  19   .  25  46   .   6  22   .  21   .   .
   .  10   .  21   .   .  20  19  19   .  25  46  46  22  22  22  21  15   .
   .   4  20   .   .  16  16   .   .  25  25  46   .   4  22  22  21  21   .
   6   6  33  15   .   .   .   .  19   .  25  46  46   .   .  22  22  21   .
  11   .  33  33  33  33  33  19   .   .   .  18  46  46  46  22  25  15   .
   .  11   .  33   8   8   .   6  18  18  18  18  18  46  22  22  25   .   .
   .   .   .  33   8   6   6   6  18   .   .  18  10   8   .  22  25   .   .
   .   .  13   8   8   6   3   3   9   .   7   .  10   8   .   .  25  15   .
   .   .  13   .   .   4   .   9   9   .   .   7  10   .   .   .  25   .   .

  10  11   X   O  39   X   O  41   X   O   O   5   5   5   6   O   X  15  15
  10  11  38   X   X   O   O   X  38  37   X   O   O   5   O   X   X   X   X
  12   X  11  39   X   X   X   X   X   X   X  37   2   O   O   O   X   O  16
   X  15  14   X   O   X   O   O   O   O   X   X   O   O   X   O   X   X  15
   O   X   X  14   O   O   8   O  37  37   X  37   O   X   X   X   O   X  15
   2   O   O   O   6   8   X   O   X   X   X   O   O  18   X   X   O   X   O
   O   6   6   6   6   O   O   X   O   O   O   O   9   O   X   O   O   O   3
   X   O   O   O   O  24   X   X  14   3   O   X   X  25   X   O  18   O   X
  12   X  23   X  25   X  28  15   X  10   O   9   X  25   X   X   X   O   3
   X  13  21   X  22  22   X   O   X   X   O   O   X   O  38   X   O  14   O
  12   X  10   O   X  21   X   X   O  33   X   O   9   O   X  38   O  15  14
  10   X  25   O  15  20   X   O   O  25   X   O   O   X   X   X   O   X  15
   7   O   X  21  14   O   O  14  26   X   X   O   .   O   X   X   O   O  14
   X   X   O   X  15  14  14  20   X  26   X   O   O   2  22   X   X   O  14
   O   6   O   O   O   O   O   X  34  18  39   X   O   O   O   X   O   X  15
   9   O   9   O   X   X  13   O   X   X   X   X   X   O   X   X   O  15  14
   9   9   9   O   X   O   O   O   X  18  19   X   O   X  24   X   O  15  14
   9   9   O   X   X   O   X   X   O   7   X  20   O   X   8  22   O   X  15
   9   9   O   8  10   X   6   O   O   5   8   X   O   8   6   6   O  15  14

  10  10   X   O   3   X   O   4   X   O   O  13   9   5  47   O   X   2   2
  10  10  10   X   X   O   O   X   2   9   X   O   O  48   O   X   X   X   X
  10   X  10  10   X   X   X   X   X   X   X   9  48   O   O   O   X   O   4
   X  10  10   X   O   X   O   O   O   O   X   X   O   O   X   O   X   X   3
   O   X   X  15   O   O  19   O   9   9   X  46   O   X   X   X   O   X   5
  13   O   O   O  15  14   X   O   X   X   X   O   O  49   X   X   O   X   O
   O  36  35  35  32   O   O   X   O   O   O   O  49   O   X   O   O   O  29
   X   O   O   O   O  32   X   X  46  46   O   X   X   5   X   O  25   O   X
   2   X  30   X  30   X   2   3   X  46   O  46   X   6   X   X   X   O  26
   X   5   5   X  14  14   X   O   X   X   O   O   X   O   6   X   O  33   O
   3   X  21   O   X  14   X   X   O  19   X   O  51   O   X  21   O  21  16
   3   X  21   O  21  16   X   O   O  19   X   O   O   X   X   X   O   X  14
   4   O   X  21  16   O   O  21  19   X   X   O  47   O   X   X   O   O  21
   X   X   O   X  33  35  35  14   X   4   X   O   O  47  46   X   X   O  21
   O  35   O   O   O   O   O   X   4   4   4   X   O   O   O   X   O   X  14
  12   O  34   O   X   X  38   O   X   X   X   X   X   O   X   X   O  25  14
   9  11  37   O   X   O   O   O   X   5   5   X   O   X   6   X   O  25  14
   9  13   O   X   X   O   X   X   O   9   X  10   O   X   6  25   O   X  14
   9  13   O  13   2   X  11   O   O   9   5   X   O  10   6  25   O  25  14
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

regularMoveCount 2446
passCount 475
koCaptureCount 24
suicideCount 79

)%%";
  expect("Board undo test move counts",out,expected);
}

void Tests::runBoardHandicapTest() {
  cout << "Running board handicap test" << endl;
  {
    Board board = Board(19,19);
    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("chinese");
    BoardHistory hist(board,nextPla,rules,0);

    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,3,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,4,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,5,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
    testAssert(hist.computeNumHandicapStones() == 3);
    testAssert(hist.computeWhiteHandicapBonus() == 3);
  }

  {
    Board board = Board(19,19);
    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("chinese");
    BoardHistory hist(board,nextPla,rules,0);

    hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
    testAssert(hist.computeNumHandicapStones() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,3,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,4,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 2);
    testAssert(hist.computeWhiteHandicapBonus() == 2);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,5,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 3);
    testAssert(hist.computeWhiteHandicapBonus() == 3);
  }

  {
    Board board = Board(19,19);
    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("aga");
    BoardHistory hist(board,nextPla,rules,0);

    hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
    testAssert(hist.computeNumHandicapStones() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,3,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,4,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 2);
    testAssert(hist.computeWhiteHandicapBonus() == 1);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,5,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 3);
    testAssert(hist.computeWhiteHandicapBonus() == 2);
  }

  {
    Board board = Board(19,19);
    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("aga");
    BoardHistory hist(board,nextPla,rules,0);

    hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
    testAssert(hist.computeNumHandicapStones() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,3,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,4,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 2);
    testAssert(hist.computeWhiteHandicapBonus() == 1);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    testAssert(hist.computeNumHandicapStones() == 2);
    testAssert(hist.computeWhiteHandicapBonus() == 1);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,5,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 3);
    testAssert(hist.computeWhiteHandicapBonus() == 2);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,6,board.x_size), P_WHITE, NULL);
    testAssert(hist.computeNumHandicapStones() == 3);
    testAssert(hist.computeWhiteHandicapBonus() == 2);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,7,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 3);
    testAssert(hist.computeWhiteHandicapBonus() == 2);
  }

  {
    Board board = Board(19,19);
    Player nextPla = P_BLACK;
    Rules rules = Rules::parseRules("chinese");
    BoardHistory hist(board,nextPla,rules,0);

    hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
    testAssert(hist.computeNumHandicapStones() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,3,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,4,board.x_size), P_WHITE, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,5,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,6,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,7,board.x_size), P_BLACK, NULL);
    testAssert(hist.computeNumHandicapStones() == 0);
    testAssert(hist.computeWhiteHandicapBonus() == 0);
  }

}

void Tests::runBoardStressTest() {
  cout << "Running board stress test" << endl;

  {
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
      Loc emptyBuf[Board::MAX_ARR_SIZE];

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
          int emptyCount = 0;
          Loc end = Location::getLoc(boards[i].x_size-1,boards[i].y_size-1,boards[i].x_size);
          for(Loc j = 0; j<=end; j++) {
            if(boards[i].colors[j] == C_EMPTY)
              emptyBuf[emptyCount++] = j;
          }
          testAssert(emptyCount > 0);
          locs[i] = emptyBuf[rand.nextUInt(emptyCount)];
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
              testAssert(board.colors[loc] == C_EMPTY);
              testAssert(board.wouldBeKoCapture(loc,C_BLACK) || board.wouldBeKoCapture(loc,C_WHITE));
              testAssert(board.wouldBeCapture(loc,C_BLACK) || board.wouldBeCapture(loc,C_WHITE));
              if(board.isAdjacentToPla(loc,getOpp(pla))) {
                testAssert(board.wouldBeKoCapture(loc,pla));
                testAssert(board.wouldBeCapture(loc,pla));
              }
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
            testAssert(!copy.wouldBeCapture(loc,pla));
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
              testAssert(copy.wouldBeCapture(loc,pla));
            }
            else
              testAssert(!copy.wouldBeKoCapture(loc,pla));
            if(!board.isAdjacentToPla(loc,getOpp(pla)) && copy.isAdjacentToPla(loc,getOpp(pla)))
              testAssert(copy.wouldBeCapture(loc,pla));
            if(!copy.isAdjacentToPla(loc,getOpp(pla)))
              testAssert(!copy.wouldBeCapture(loc,pla));

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

regularMoveCount 38017
passCount 273
koCaptureCount 212
koBanCount 45
suicideCount 440
Caps 4753 5024
Caps 4821 4733
Caps 4995 5041
Caps 4420 4335

)%%";
    expect("Board stress test move counts",out,expected);
  }

  {
    Rand rand("runBoardSetStoneTests");

    auto boardToPlacements = [&](const Board& b) {
      std::vector<Move> placements;
      for(int y = 0; y<b.y_size; y++) {
        for(int x = 0; x<b.x_size; x++) {
          Loc loc = Location::getLoc(x,y,b.x_size);
          placements.push_back(Move(loc,b.colors[loc]));
        }
      }
      for(int i = 1; i<placements.size(); i++)
        std::swap(placements[i],placements[rand.nextUInt(i+1)]);
      return placements;
    };
    auto boardToNonEmptyPlacements = [&](const Board& b) {
      std::vector<Move> placements;
      for(int y = 0; y<b.y_size; y++) {
        for(int x = 0; x<b.x_size; x++) {
          Loc loc = Location::getLoc(x,y,b.x_size);
          if(b.colors[loc] != C_EMPTY)
            placements.push_back(Move(loc,b.colors[loc]));
        }
      }
      for(int i = 1; i<placements.size(); i++)
        std::swap(placements[i],placements[rand.nextUInt(i+1)]);
      return placements;
    };

    for(int rep = 0; rep<1000; rep++) {
      Board board0(5 + rand.nextUInt(14), 5 + rand.nextUInt(14));
      Board board1 = board0;
      Board board2 = board0;
      Board board3 = board0;
      Board board4 = board0;
      Board board5 = board0;
      Board board6 = board0;
      int numMoves1 = rand.nextUInt(1000);
      int numMoves2 = rand.nextUInt(1000);
      for(int i = 0; i<numMoves1; i++) {
        Loc loc = Location::getLoc(rand.nextUInt(board1.x_size),rand.nextUInt(board1.y_size),board1.x_size);
        Player pla = rand.nextBool(0.5) ? P_BLACK : P_WHITE;
        if(board1.isLegal(loc,pla,true)) {
          bool suc4 = board4.setStoneFailIfNoLibs(loc,pla);
          testAssert(suc4 == !(board1.wouldBeCapture(loc,pla) || board1.isSuicide(loc,pla)));
          if(!suc4) {
            board4.playMoveAssumeLegal(loc,pla);
          }
          else {
            testAssert(board4.colors[loc] == pla);
          }

          board1.playMoveAssumeLegal(loc,pla);
          bool suc3 = board3.setStone(loc,pla);
          testAssert(suc3);
        }
        else {
          Color oldColor = board4.colors[loc];
          bool suc4 = board4.setStoneFailIfNoLibs(loc,pla);
          if(suc4) {
            testAssert(board4.colors[loc] == pla);
            bool suc4_2 = board4.setStoneFailIfNoLibs(loc,oldColor);
            testAssert(suc4_2);
          }
        }
      }
      board1.checkConsistency();
      board3.checkConsistency();
      board4.checkConsistency();
      testAssert(board1.pos_hash == board3.pos_hash);
      testAssert(board1.pos_hash == board4.pos_hash);

      double emptyProb = rand.nextDouble() * 0.5;
      for(int i = 0; i<numMoves2; i++) {
        Loc loc = Location::getLoc(rand.nextUInt(board2.x_size),rand.nextUInt(board2.y_size),board2.x_size);
        Color color = rand.nextBool(emptyProb) ? C_EMPTY : rand.nextBool(0.5) ? P_BLACK : P_WHITE;
        bool suc2 = board2.setStone(loc,color);
        testAssert(suc2);
        bool suc5 = board5.setStoneFailIfNoLibs(loc,color);
        bool suc6 = board6.setStoneFailIfNoLibs(loc,color);
        if(color == C_EMPTY) {
          testAssert(suc5);
          testAssert(suc6);
        }
        if(!suc5)
          board5.setStone(loc,color);
        else {
          testAssert(board5.colors[loc] == color);
        }
        if(suc6) {
          testAssert(board6.colors[loc] == color);
        }
        else {
          testAssert(board6.colors[loc] != color);
        }
        board2.checkConsistency();
        board5.checkConsistency();
        board6.checkConsistency();
        testAssert(board2.pos_hash == board5.pos_hash);
      }

      bool suc0 = board0.setStonesFailIfNoLibs(boardToNonEmptyPlacements(board1));
      testAssert(suc0);
      board0.checkConsistency();
      testAssert(board0.pos_hash == board1.pos_hash);

      bool suc3 = board3.setStonesFailIfNoLibs(boardToPlacements(board2));
      testAssert(suc3);
      board3.checkConsistency();
      testAssert(board3.pos_hash == board2.pos_hash);
    }
  }
  {
    Rand rand("runBoardSetStoneTests2");
    for(int rep = 0; rep<1000; rep++) {
      Board board(1 + rand.nextUInt(18), 1 + rand.nextUInt(18));
      std::vector<Move> placements;
      for(int i = 0; i<1000; i++) {
        Loc loc = Location::getLoc(rand.nextUInt(board.x_size),rand.nextUInt(board.y_size),board.x_size);
        Player pla = rand.nextBool(0.5) ? P_BLACK : P_WHITE;
        if(board.isLegal(loc,pla,true)) {
          placements.push_back(Move(loc,pla));
          bool anyCaps = board.wouldBeCapture(loc,pla) || board.isSuicide(loc,pla);
          board.playMoveAssumeLegal(loc,pla);
          Board copy(board.x_size,board.y_size);
          bool suc = copy.setStonesFailIfNoLibs(placements);
          testAssert(suc == !anyCaps);
          copy.checkConsistency();
          if(!suc)
            break;
          testAssert(board.pos_hash == copy.pos_hash);
        }
      }
    }
  }
  {
    Rand rand("runBoardSetStoneTests3");
    for(int rep = 0; rep<1000; rep++) {
      Board board(1 + rand.nextUInt(18), 1 + rand.nextUInt(18));
      for(int i = 0; i<300; i++) {
        Loc loc = Location::getLoc(rand.nextUInt(board.x_size),rand.nextUInt(board.y_size),board.x_size);
        Color color = rand.nextBool(0.25) ? C_EMPTY : rand.nextBool(0.5) ? P_BLACK : P_WHITE;
        board.setStone(loc,color);
      }

      Board orig = board;
      std::set<Loc> prevPlacedLocs;
      std::vector<Move> placements;
      for(int i = 0; i<1000; i++) {
        Loc loc = Location::getLoc(rand.nextUInt(board.x_size),rand.nextUInt(board.y_size),board.x_size);
        Color color = rand.nextBool(0.25) ? C_EMPTY : rand.nextBool(0.5) ? P_BLACK : P_WHITE;

        placements.push_back(Move(loc,color));
        if(prevPlacedLocs.find(loc) != prevPlacedLocs.end()) {
          Board copy(board.x_size,board.y_size);
          bool suc = copy.setStonesFailIfNoLibs(placements);
          testAssert(!suc);
          placements.pop_back();
        }
        else {
          prevPlacedLocs.insert(loc);
          Board prev = board;
          board.setStone(loc,color);

          bool anyCaps = false;
          for(int y = 0; y<board.y_size; y++) {
            for(int x = 0; x<board.x_size; x++) {
              Loc l = Location::getLoc(x,y,board.x_size);
              if(l != loc && board.colors[l] == C_EMPTY && prev.colors[l] != C_EMPTY)
                anyCaps = true;
            }
          }

          Board copy = orig;
          bool suc = copy.setStonesFailIfNoLibs(placements);
          if(!(suc == (!anyCaps && board.colors[loc] == color))) {
            cout << suc << " " << anyCaps << " " << (board.colors[loc] == color) << endl;
            cout << orig << endl;
            cout << prev << endl;
            cout << board << endl;
            cout << i << endl;
            for(const auto& placement: placements) {
              cout << Location::toString(placement.loc,board) << " " << (int)placement.pla << endl;
            }
          }
          testAssert(suc == (!anyCaps && board.colors[loc] == color));
          if(!suc)
            break;
        }
      }
    }
  }
}

void Tests::runBoardReplayTest() {
  cout << "Running board replay test" << endl;
  Rand rand("runBoardReplayTest");

  Board base = Board::parseBoard(9,5,R"%%(
.xo.o.ooo
xo.oooooo
.xooxxxo.
oxxxxx.xo
.o.xx.xo.
)%%");

  for(int rep = 0; rep<6000; rep++) {
    Board board(base);
    Player pla = rand.nextBool(0.5) ? P_BLACK : P_WHITE;
    int initialEncorePhase = rand.nextBool(0.9) ? 0 : rand.nextBool(0.5) ? 1 : 2;
    Rules rules = rand.nextBool(0.8) ? Rules::parseRules("japanese") : Rules::parseRules("chinese");
    if(rules.scoringRule == Rules::SCORING_AREA)
      initialEncorePhase = 0;
    if(rand.nextBool(0.1))
      rules.koRule = rand.nextBool(0.5) ? Rules::KO_SITUATIONAL : Rules::KO_POSITIONAL;
    if(rand.nextBool(0.2))
      rules.taxRule = rand.nextBool(0.5) ? Rules::TAX_SEKI : rand.nextBool(0.5) ? Rules::TAX_NONE : Rules::TAX_ALL;
    BoardHistory hist(board,pla,rules,initialEncorePhase);
    hist.setInitialTurnNumber(rand.nextInt(0,40));
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(rand.nextBool(0.5));

    double drawEquivalentWinsForWhite = 0.7;
    int repBound = rand.nextInt(1,5);

    Hash128 graphHash = GraphHash::getGraphHashFromScratch(hist, pla, repBound, drawEquivalentWinsForWhite);

    bool anyKomiSet = false;
    double passProb = rand.nextDouble(0.05,0.80);
    int numSteps = rand.nextInt(6,15);
    for(int i = 0; i<numSteps; i++) {
      BoardHistory tmpHist(board,pla,rules,hist.encorePhase);
      Loc moveLoc;
      if(rand.nextBool(passProb))
        moveLoc = Board::PASS_LOC;
      else {
        moveLoc = PlayUtils::chooseRandomLegalMove(board,tmpHist,pla,rand,Board::NULL_LOC);
        //Resample to increase likelihood of capture moves.
        if(!board.wouldBeCapture(moveLoc,pla))
          moveLoc = PlayUtils::chooseRandomLegalMove(board,tmpHist,pla,rand,Board::NULL_LOC);
        //Resample to decrease likelihood of moves not adjacent to opponent
        if(!board.isAdjacentToPla(moveLoc,getOpp(pla)))
          moveLoc = PlayUtils::chooseRandomLegalMove(board,tmpHist,pla,rand,Board::NULL_LOC);
      }
      bool preventEncore = rand.nextBool(0.5);
      if(rand.nextBool(0.5)) {
        // if(!hist.isLegal(board,moveLoc,pla))
        //   cout << "AAAA" << endl;
        bool suc = hist.makeBoardMoveTolerant(board, moveLoc, pla, preventEncore);
        testAssert(suc);
      }
      else {
        if(hist.isLegal(board,moveLoc,pla))
          hist.makeBoardMoveAssumeLegal(board, moveLoc, pla, NULL, preventEncore);
        else
          continue;
      }
      pla = getOpp(pla);
      if(rand.nextBool(0.1))
        pla = getOpp(pla);
      if(rand.nextBool(0.025)) {
        anyKomiSet = true;
        hist.setKomi(hist.rules.komi + (float)(rand.nextBool(0.5) ? -0.5 : +0.5));
      }

      graphHash = GraphHash::getGraphHash(graphHash, hist, pla, repBound, drawEquivalentWinsForWhite);
    }

    BoardHistory histCopy = hist.copyToInitial();
    Board boardCopy = histCopy.getRecentBoard(0);
    for(int i = 0; i<hist.moveHistory.size(); i++) {
      histCopy.makeBoardMoveAssumeLegal(boardCopy, hist.moveHistory[i].loc, hist.moveHistory[i].pla, NULL, hist.preventEncoreHistory[i]);
    }
    if(rand.nextBool(0.05))
      histCopy = hist;
    //if(rep < 100)
    //  hist.printDebugInfo(cout,board);

    testAssert(boardCopy.isEqualForTesting(board, true, true));
    testAssert(boardCopy.isEqualForTesting(histCopy.getRecentBoard(0), true, true));
    testAssert(histCopy.getRecentBoard(0).isEqualForTesting(hist.getRecentBoard(0), true, true));
    testAssert(BoardHistory::getSituationRulesAndKoHash(boardCopy,histCopy,pla,drawEquivalentWinsForWhite) == hist.getSituationRulesAndKoHash(board,hist,pla,drawEquivalentWinsForWhite));
    testAssert(histCopy.currentSelfKomi(P_BLACK, drawEquivalentWinsForWhite) == hist.currentSelfKomi(P_BLACK, drawEquivalentWinsForWhite));
    testAssert(histCopy.currentSelfKomi(P_WHITE, drawEquivalentWinsForWhite) == hist.currentSelfKomi(P_WHITE, drawEquivalentWinsForWhite));
    testAssert(histCopy.initialTurnNumber == hist.initialTurnNumber);
    testAssert(histCopy.presumedNextMovePla == hist.presumedNextMovePla);
    testAssert(histCopy.assumeMultipleStartingBlackMovesAreHandicap == hist.assumeMultipleStartingBlackMovesAreHandicap);
    testAssert(GraphHash::getStateHash(histCopy, pla, drawEquivalentWinsForWhite) == GraphHash::getStateHash(hist, pla, drawEquivalentWinsForWhite));
    testAssert(GraphHash::getGraphHashFromScratch(histCopy, pla, repBound, drawEquivalentWinsForWhite) == GraphHash::getGraphHashFromScratch(hist, pla, repBound, drawEquivalentWinsForWhite));
    testAssert(anyKomiSet || graphHash == GraphHash::getGraphHashFromScratch(hist, pla, repBound, drawEquivalentWinsForWhite));
  }
}

