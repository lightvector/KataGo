#include "../tests/tests.h"

#include "../neuralnet/nninputs.h"

using namespace std;
using namespace TestCommon;

void Tests::runBasicSymmetryTests() {
  cout << "Running basic symmetries tests" << endl;
  ostringstream out;

  {
    const char* name = "Basic flipping of tensors and boards and locations";
    {
      Board board = Board::parseBoard(7,7,R"%%(
.......
.......
.xxx...
....o..
....o..
....o..
.......
)%%");
      float buf[49] = {
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
        0.0f,1.0f,1.0f,1.0f,0.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.0f,2.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.0f,2.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.0f,2.0f,0.0f,0.0f,
        0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
      };

      for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
        float dst[49];
        SymmetryHelpers::copyInputsWithSymmetry(buf,dst,1,7,7,1,true,symmetry);
        Board symBoard = SymmetryHelpers::getSymBoard(board,symmetry);
        Loc symLocWithX = SymmetryHelpers::getSymLoc(1,2,board,symmetry);
        out << "SYMMETRY " << symmetry << endl;
        out << symBoard << endl;
        out << Global::boolToString(symBoard.colors[symLocWithX] == P_BLACK) << endl;
        for(int y = 0; y<7; y++) {
          for(int x = 0; x<7; x++)
            out << dst[x+y*7] << " ";
          out << endl;
        }
      }
    }

    {
      Board board = Board::parseBoard(7,5,R"%%(
.......
.x.....
.x.....
.oo....
.......
)%%");
      board.setSimpleKoLoc(Location::getLoc(5,0,board.x_size));

      for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
        Board symBoard = SymmetryHelpers::getSymBoard(board,symmetry);
        out << "SYMMETRY " << symmetry << endl;
        out << symBoard << endl;
        out << symBoard.getSitHashWithSimpleKo(P_BLACK) << endl;
        Loc symLocWithX = SymmetryHelpers::getSymLoc(1,1,board,symmetry);
        Loc symLocWithX2 = SymmetryHelpers::getSymLoc(1,2,board,symmetry);
        Loc symLocWithO = SymmetryHelpers::getSymLoc(1,3,board,symmetry);
        Loc symLocWithO2 = SymmetryHelpers::getSymLoc(2,3,board,symmetry);
        out << Global::boolToString(symBoard.colors[symLocWithX] == P_BLACK) << endl;
        out << Global::boolToString(symBoard.colors[symLocWithX2] == P_BLACK) << endl;
        out << Global::boolToString(symBoard.colors[symLocWithO] == P_WHITE) << endl;
        out << Global::boolToString(symBoard.colors[symLocWithO2] == P_WHITE) << endl;
        out << Global::boolToString(symBoard.ko_loc == SymmetryHelpers::getSymLoc(5,0,board,symmetry)) << endl;
      }
    }

    string expected = R"%%(
SYMMETRY 0
HASH: 10536BA4F5B60C6BAEF367A2D60E8EDA
   A B C D E F G
 7 . . . . . . .
 6 . . . . . . .
 5 . X X X . . .
 4 . . . . O . .
 3 . . . . O . .
 2 . . . . O . .
 1 . . . . . . .


true
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 1 1 1 0 0 0
0 0 0 0 2 0 0
0 0 0 0 2 0 0
0 0 0 0 2 0 0
0 0 0 0 0 0 0
SYMMETRY 1
HASH: 4A73C1FBB96574A1EBBB05C4FBC6128B
   A B C D E F G
 7 . . . . . . .
 6 . . . . O . .
 5 . . . . O . .
 4 . . . . O . .
 3 . X X X . . .
 2 . . . . . . .
 1 . . . . . . .


true
0 0 0 0 0 0 0
0 0 0 0 2 0 0
0 0 0 0 2 0 0
0 0 0 0 2 0 0
0 1 1 1 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
SYMMETRY 2
HASH: B595E477F1130F9C23C544E9362BBB81
   A B C D E F G
 7 . . . . . . .
 6 . . . . . . .
 5 . . . X X X .
 4 . . O . . . .
 3 . . O . . . .
 2 . . O . . . .
 1 . . . . . . .


true
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 1 1 1 0
0 0 2 0 0 0 0
0 0 2 0 0 0 0
0 0 2 0 0 0 0
0 0 0 0 0 0 0
SYMMETRY 3
HASH: E837B93887D00B87590FE32DF7EACF34
   A B C D E F G
 7 . . . . . . .
 6 . . O . . . .
 5 . . O . . . .
 4 . . O . . . .
 3 . . . X X X .
 2 . . . . . . .
 1 . . . . . . .


true
0 0 0 0 0 0 0
0 0 2 0 0 0 0
0 0 2 0 0 0 0
0 0 2 0 0 0 0
0 0 0 1 1 1 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
SYMMETRY 4
HASH: 2C091C7336C956E4190AEB8C971593BE
   A B C D E F G
 7 . . . . . . .
 6 . . X . . . .
 5 . . X . . . .
 4 . . X . . . .
 3 . . . O O O .
 2 . . . . . . .
 1 . . . . . . .


true
0 0 0 0 0 0 0
0 0 1 0 0 0 0
0 0 1 0 0 0 0
0 0 1 0 0 0 0
0 0 0 2 2 2 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
SYMMETRY 5
HASH: 34A1589420527491B8CEA75DC1B6F48E
   A B C D E F G
 7 . . . . . . .
 6 . . . . X . .
 5 . . . . X . .
 4 . . . . X . .
 3 . O O O . . .
 2 . . . . . . .
 1 . . . . . . .


true
0 0 0 0 0 0 0
0 0 0 0 1 0 0
0 0 0 0 1 0 0
0 0 0 0 1 0 0
0 2 2 2 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
SYMMETRY 6
HASH: 075AEB9CF818D878D220B3FB43D8BCBE
   A B C D E F G
 7 . . . . . . .
 6 . . . . . . .
 5 . . . O O O .
 4 . . X . . . .
 3 . . X . . . .
 2 . . X . . . .
 1 . . . . . . .


true
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 2 2 2 0
0 0 1 0 0 0 0
0 0 1 0 0 0 0
0 0 1 0 0 0 0
0 0 0 0 0 0 0
SYMMETRY 7
HASH: 2398D7EE75668E65C5AAA3C650096162
   A B C D E F G
 7 . . . . . . .
 6 . . . . . . .
 5 . O O O . . .
 4 . . . . X . .
 3 . . . . X . .
 2 . . . . X . .
 1 . . . . . . .


true
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 2 2 2 0 0 0
0 0 0 0 1 0 0
0 0 0 0 1 0 0
0 0 0 0 1 0 0
0 0 0 0 0 0 0
SYMMETRY 0
HASH: A5352E294DBE4F6B2B4599F80D25B366
   A B C D E F G
 5 . . . . . . .
 4 . X . . . . .
 3 . X . . . . .
 2 . O O . . . .
 1 . . . . . . .


A45A5C8D2DE8346394278209E73A926D
true
true
true
true
true
SYMMETRY 1
HASH: A8FACF75A64B6A8DD514DF4D7712A63A
   A B C D E F G
 5 . . . . . . .
 4 . O O . . . .
 3 . X . . . . .
 2 . X . . . . .
 1 . . . . . . .


BFCD2A6D21BF86A61F34AE48DCB8AEC6
true
true
true
true
true
SYMMETRY 2
HASH: AD610645C81C5509EAF0A92125D6C782
   A B C D E F G
 5 . . . . . . .
 4 . . . . . X .
 3 . . . . . X .
 2 . . . . O O .
 1 . . . . . . .


973F598CDF7EC20C5F33A53CD7924EAA
true
true
true
true
true
SYMMETRY 3
HASH: EEE84256E41FD211A4B352EEF5C7CC7B
   A B C D E F G
 5 . . . . . . .
 4 . . . . O O .
 3 . . . . . X .
 2 . . . . . X .
 1 . . . . . . .


625DFB8C14A84F5F6E6FA9A59DD301DA
true
true
true
true
true
SYMMETRY 4
HASH: 1EA67062B2A515A834C9974D7E15A768
   A B C D E
 7 . . . . .
 6 . X X O .
 5 . . . O .
 4 . . . . .
 3 . . . . .
 2 . . . . .
 1 . . . . .


5DDD312C75F3DD542351ED3B73563B91
true
true
true
true
true
SYMMETRY 5
HASH: F007BED32BD9CF351C98DA163AA05887
   A B C D E
 7 . . . . .
 6 . O X X .
 5 . O . . .
 4 . . . . .
 3 . . . . .
 2 . . . . .
 1 . . . . .


E640978A901DA2740E21EA68A4905384
true
true
true
true
true
SYMMETRY 6
HASH: ED57525FF5D510A172D3B280505FDF53
   A B C D E
 7 . . . . .
 6 . . . . .
 5 . . . . .
 4 . . . . .
 3 . . . O .
 2 . X X O .
 1 . . . . .


3FBCDFE3BC7356387800B6F220F9C2F9
true
true
true
true
true
SYMMETRY 7
HASH: 0FBBCE35DF09E50C13EB9BF65F27A7C3
   A B C D E
 7 . . . . .
 6 . . . . .
 5 . . . . .
 4 . . . . .
 3 . O . . .
 2 . O X X .
 1 . . . . .


3D4B8867677FB2EBC1C1608C14BD9D5F
true
true
true
true
true
)%%";
    expect(name,out,expected);
  }

  {
    Board board = Board::parseBoard(12,7,R"%%(
............
.x..........
.x..........
.oo.........
............
............
............
)%%");
    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      testAssert(SymmetryHelpers::getSymLoc(1,2,board,symmetry) == SymmetryHelpers::getSymLoc(Location::getLoc(1,2,board.x_size),board,symmetry));
      testAssert(SymmetryHelpers::getSymLoc(1,2,board,symmetry) == SymmetryHelpers::getSymLoc(Location::getLoc(1,2,board.x_size),12,7,symmetry));
      testAssert(SymmetryHelpers::getSymLoc(1,2,board,symmetry) == SymmetryHelpers::getSymLoc(1,2,12,7,symmetry));
    }
  }

  {
    const char* name = "Symmetry composition tests";
    Board board = Board::parseBoard(9,7,R"%%(
x.......o
x.xxo....
..ox....x
..o..x...
.o.......
...xooo..
.x....x.o
)%%");

    Loc loc = Location::getLoc(1,2,board.x_size);
    for(int symmetry1 = 0; symmetry1 < SymmetryHelpers::NUM_SYMMETRIES; symmetry1++) {
      for(int symmetry2 = 0; symmetry2 < SymmetryHelpers::NUM_SYMMETRIES; symmetry2++) {
        int symmetryComposed = SymmetryHelpers::compose(symmetry1,symmetry2);
        Board symBoardComb = SymmetryHelpers::getSymBoard(board,symmetryComposed);
        Board symBoardCombManual = SymmetryHelpers::getSymBoard(SymmetryHelpers::getSymBoard(board,symmetry1),symmetry2);
        Loc symLocComb = SymmetryHelpers::getSymLoc(loc,board,symmetryComposed);
        Loc symLocCombManual = SymmetryHelpers::getSymLoc(SymmetryHelpers::getSymLoc(loc,board,symmetry1),SymmetryHelpers::getSymBoard(board,symmetry1),symmetry2);
        out << "Symmetry " << symmetry1 << " + " << symmetry2 << " = " << symmetryComposed << endl;
        testAssert(symBoardCombManual.isEqualForTesting(symBoardComb,true,true));
        testAssert(symLocComb == symLocCombManual);
      }
    }

    string expected = R"%%(
Symmetry 0 + 0 = 0
Symmetry 0 + 1 = 1
Symmetry 0 + 2 = 2
Symmetry 0 + 3 = 3
Symmetry 0 + 4 = 4
Symmetry 0 + 5 = 5
Symmetry 0 + 6 = 6
Symmetry 0 + 7 = 7
Symmetry 1 + 0 = 1
Symmetry 1 + 1 = 0
Symmetry 1 + 2 = 3
Symmetry 1 + 3 = 2
Symmetry 1 + 4 = 5
Symmetry 1 + 5 = 4
Symmetry 1 + 6 = 7
Symmetry 1 + 7 = 6
Symmetry 2 + 0 = 2
Symmetry 2 + 1 = 3
Symmetry 2 + 2 = 0
Symmetry 2 + 3 = 1
Symmetry 2 + 4 = 6
Symmetry 2 + 5 = 7
Symmetry 2 + 6 = 4
Symmetry 2 + 7 = 5
Symmetry 3 + 0 = 3
Symmetry 3 + 1 = 2
Symmetry 3 + 2 = 1
Symmetry 3 + 3 = 0
Symmetry 3 + 4 = 7
Symmetry 3 + 5 = 6
Symmetry 3 + 6 = 5
Symmetry 3 + 7 = 4
Symmetry 4 + 0 = 4
Symmetry 4 + 1 = 6
Symmetry 4 + 2 = 5
Symmetry 4 + 3 = 7
Symmetry 4 + 4 = 0
Symmetry 4 + 5 = 2
Symmetry 4 + 6 = 1
Symmetry 4 + 7 = 3
Symmetry 5 + 0 = 5
Symmetry 5 + 1 = 7
Symmetry 5 + 2 = 4
Symmetry 5 + 3 = 6
Symmetry 5 + 4 = 1
Symmetry 5 + 5 = 3
Symmetry 5 + 6 = 0
Symmetry 5 + 7 = 2
Symmetry 6 + 0 = 6
Symmetry 6 + 1 = 4
Symmetry 6 + 2 = 7
Symmetry 6 + 3 = 5
Symmetry 6 + 4 = 2
Symmetry 6 + 5 = 0
Symmetry 6 + 6 = 3
Symmetry 6 + 7 = 1
Symmetry 7 + 0 = 7
Symmetry 7 + 1 = 5
Symmetry 7 + 2 = 6
Symmetry 7 + 3 = 4
Symmetry 7 + 4 = 3
Symmetry 7 + 5 = 1
Symmetry 7 + 6 = 2
Symmetry 7 + 7 = 0
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Symmetry inverse tests";
    for(int symmetry1 = 0; symmetry1 < SymmetryHelpers::NUM_SYMMETRIES; symmetry1++) {
      for(int symmetry2 = 0; symmetry2 < SymmetryHelpers::NUM_SYMMETRIES; symmetry2++) {
        out << "Symmetry " << symmetry1 << " : " << symmetry2 << " inverses " << Global::boolToString(SymmetryHelpers::invert(symmetry1) == symmetry2) << endl;
        testAssert((SymmetryHelpers::compose(symmetry1,symmetry2) == 0) == (SymmetryHelpers::invert(symmetry1) == symmetry2));
      }
    }
    string expected = R"%%(
Symmetry 0 : 0 inverses true
Symmetry 0 : 1 inverses false
Symmetry 0 : 2 inverses false
Symmetry 0 : 3 inverses false
Symmetry 0 : 4 inverses false
Symmetry 0 : 5 inverses false
Symmetry 0 : 6 inverses false
Symmetry 0 : 7 inverses false
Symmetry 1 : 0 inverses false
Symmetry 1 : 1 inverses true
Symmetry 1 : 2 inverses false
Symmetry 1 : 3 inverses false
Symmetry 1 : 4 inverses false
Symmetry 1 : 5 inverses false
Symmetry 1 : 6 inverses false
Symmetry 1 : 7 inverses false
Symmetry 2 : 0 inverses false
Symmetry 2 : 1 inverses false
Symmetry 2 : 2 inverses true
Symmetry 2 : 3 inverses false
Symmetry 2 : 4 inverses false
Symmetry 2 : 5 inverses false
Symmetry 2 : 6 inverses false
Symmetry 2 : 7 inverses false
Symmetry 3 : 0 inverses false
Symmetry 3 : 1 inverses false
Symmetry 3 : 2 inverses false
Symmetry 3 : 3 inverses true
Symmetry 3 : 4 inverses false
Symmetry 3 : 5 inverses false
Symmetry 3 : 6 inverses false
Symmetry 3 : 7 inverses false
Symmetry 4 : 0 inverses false
Symmetry 4 : 1 inverses false
Symmetry 4 : 2 inverses false
Symmetry 4 : 3 inverses false
Symmetry 4 : 4 inverses true
Symmetry 4 : 5 inverses false
Symmetry 4 : 6 inverses false
Symmetry 4 : 7 inverses false
Symmetry 5 : 0 inverses false
Symmetry 5 : 1 inverses false
Symmetry 5 : 2 inverses false
Symmetry 5 : 3 inverses false
Symmetry 5 : 4 inverses false
Symmetry 5 : 5 inverses false
Symmetry 5 : 6 inverses true
Symmetry 5 : 7 inverses false
Symmetry 6 : 0 inverses false
Symmetry 6 : 1 inverses false
Symmetry 6 : 2 inverses false
Symmetry 6 : 3 inverses false
Symmetry 6 : 4 inverses false
Symmetry 6 : 5 inverses true
Symmetry 6 : 6 inverses false
Symmetry 6 : 7 inverses false
Symmetry 7 : 0 inverses false
Symmetry 7 : 1 inverses false
Symmetry 7 : 2 inverses false
Symmetry 7 : 3 inverses false
Symmetry 7 : 4 inverses false
Symmetry 7 : 5 inverses false
Symmetry 7 : 6 inverses false
Symmetry 7 : 7 inverses true
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Symmetry and move commute";
    Board board = Board::parseBoard(9,7,R"%%(
x.......o
x.xxo....
..ox....x
..o..x...
.o.......
...xooo..
.x....x.o
)%%");

    Loc loc = Location::getLoc(8,4,board.x_size);
    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      Board boardA = SymmetryHelpers::getSymBoard(board,symmetry);
      boardA.playMove(SymmetryHelpers::getSymLoc(loc,board,symmetry), P_BLACK, true);

      Board boardB = board;
      boardB.playMove(loc, P_BLACK, true);
      boardB = SymmetryHelpers::getSymBoard(boardB,symmetry);

      out << "SYMMETRY " << symmetry << endl;
      out << boardA << endl;
      out << boardB << endl;
      testAssert(boardA.isEqualForTesting(boardB,true,true));
    }
    string expected = R"%%(
SYMMETRY 0
HASH: 56798EFB369AAD4E774C248FD652F95E
   A B C D E F G H J
 7 X . . . . . . . O
 6 X . X X O . . . .
 5 . . O X . . . . X
 4 . . O . . X . . .
 3 . O . . . . . . X
 2 . . . X O O O . .
 1 . X . . . . X . O


HASH: 56798EFB369AAD4E774C248FD652F95E
   A B C D E F G H J
 7 X . . . . . . . O
 6 X . X X O . . . .
 5 . . O X . . . . X
 4 . . O . . X . . .
 3 . O . . . . . . X
 2 . . . X O O O . .
 1 . X . . . . X . O


SYMMETRY 1
HASH: 28D390F7DAAD7ACC2DC345898255355C
   A B C D E F G H J
 7 . X . . . . X . O
 6 . . . X O O O . .
 5 . O . . . . . . X
 4 . . O . . X . . .
 3 . . O X . . . . X
 2 X . X X O . . . .
 1 X . . . . . . . O


HASH: 28D390F7DAAD7ACC2DC345898255355C
   A B C D E F G H J
 7 . X . . . . X . O
 6 . . . X O O O . .
 5 . O . . . . . . X
 4 . . O . . X . . .
 3 . . O X . . . . X
 2 X . X X O . . . .
 1 X . . . . . . . O


SYMMETRY 2
HASH: 4A4757278AA22029B01A83FE170F9D65
   A B C D E F G H J
 7 O . . . . . . . X
 6 . . . . O X X . X
 5 X . . . . X O . .
 4 . . . X . . O . .
 3 X . . . . . . O .
 2 . . O O O X . . .
 1 O . X . . . . X .


HASH: 4A4757278AA22029B01A83FE170F9D65
   A B C D E F G H J
 7 O . . . . . . . X
 6 . . . . O X X . X
 5 X . . . . X O . .
 4 . . . X . . O . .
 3 X . . . . . . O .
 2 . . O O O X . . .
 1 O . X . . . . X .


SYMMETRY 3
HASH: EE186B4F08581E0E604B679A44526705
   A B C D E F G H J
 7 O . X . . . . X .
 6 . . O O O X . . .
 5 X . . . . . . O .
 4 . . . X . . O . .
 3 X . . . . X O . .
 2 . . . . O X X . X
 1 O . . . . . . . X


HASH: EE186B4F08581E0E604B679A44526705
   A B C D E F G H J
 7 O . X . . . . X .
 6 . . O O O X . . .
 5 X . . . . . . O .
 4 . . . X . . O . .
 3 X . . . . X O . .
 2 . . . . O X X . X
 1 O . . . . . . . X


SYMMETRY 4
HASH: EA1BA453DFA10B913CCF8E53D4550EC3
   A B C D E F G
 9 X X . . . . .
 8 . . . . O . X
 7 . X O O . . .
 6 . X X . . X .
 5 . O . . . O .
 4 . . . X . O .
 3 . . . . . O X
 2 . . . . . . .
 1 O . X . X . O


HASH: EA1BA453DFA10B913CCF8E53D4550EC3
   A B C D E F G
 9 X X . . . . .
 8 . . . . O . X
 7 . X O O . . .
 6 . X X . . X .
 5 . O . . . O .
 4 . . . X . O .
 3 . . . . . O X
 2 . . . . . . .
 1 O . X . X . O


SYMMETRY 5
HASH: BB69F761E815818D4A4A508377B7F0A9
   A B C D E F G
 9 . . . . . X X
 8 X . O . . . .
 7 . . . O O X .
 6 . X . . X X .
 5 . O . . . O .
 4 . O . X . . .
 3 X O . . . . .
 2 . . . . . . .
 1 O . X . X . O


HASH: BB69F761E815818D4A4A508377B7F0A9
   A B C D E F G
 9 . . . . . X X
 8 X . O . . . .
 7 . . . O O X .
 6 . X . . X X .
 5 . O . . . O .
 4 . O . X . . .
 3 X O . . . . .
 2 . . . . . . .
 1 O . X . X . O


SYMMETRY 6
HASH: 4618413C04A4D83A49F924616E3821B7
   A B C D E F G
 9 O . X . X . O
 8 . . . . . . .
 7 . . . . . O X
 6 . . . X . O .
 5 . O . . . O .
 4 . X X . . X .
 3 . X O O . . .
 2 . . . . O . X
 1 X X . . . . .


HASH: 4618413C04A4D83A49F924616E3821B7
   A B C D E F G
 9 O . X . X . O
 8 . . . . . . .
 7 . . . . . O X
 6 . . . X . O .
 5 . O . . . O .
 4 . X X . . X .
 3 . X O O . . .
 2 . . . . O . X
 1 X X . . . . .


SYMMETRY 7
HASH: 8E86BD1078F74D5BA0A8DCAD3D8E6BCC
   A B C D E F G
 9 O . X . X . O
 8 . . . . . . .
 7 X O . . . . .
 6 . O . X . . .
 5 . O . . . O .
 4 . X . . X X .
 3 . . . O O X .
 2 X . O . . . .
 1 . . . . . X X


HASH: 8E86BD1078F74D5BA0A8DCAD3D8E6BCC
   A B C D E F G
 9 O . X . X . O
 8 . . . . . . .
 7 X O . . . . .
 6 . O . X . . .
 5 . O . . . O .
 4 . X . . X X .
 3 . . . O O X .
 2 X . O . . . .
 1 . . . . . X X

)%%";
    expect(name,out,expected);
  }

}

void Tests::runBoardSymmetryTests() {
  cout << "Running board symmetry tests" << endl;

  ostringstream out;
  auto printMarkedSymDupArea = [&out](const Board& board, const bool* isSymDupLoc, const vector<int>& validSymmetries){
    out << "Symmetries: ";
    for(int symmetry: validSymmetries)
      out << symmetry << " ";
    out << endl;
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(isSymDupLoc[loc])
          out << 'x';
        else
          out << '.';
      }
      out << endl;
    }
    out << endl;
  };

  auto computeAndPrintMarkedSymDupArea = [&out,&printMarkedSymDupArea](const Board& board, Player pla, const std::vector<int>* onlySymmetries) {
    BoardHistory hist(board,pla,Rules::getTrompTaylorish(),0);
    bool isSymDupLoc[Board::MAX_ARR_SIZE];
    vector<int> validSymmetries;
    vector<int> avoidMoves;
    SymmetryHelpers::markDuplicateMoveLocs(board,hist,onlySymmetries,avoidMoves,isSymDupLoc,validSymmetries);
    out << board << endl;
    printMarkedSymDupArea(board,isSymDupLoc,validSymmetries);
  };

  const char* name = "Testing SymmetryHelpers::markDuplicateMoveLocs";
  {
    out << "Fully symmetric board" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
.........
.........
.........
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Fully symmetric board white next" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
.........
.........
.........
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_WHITE,NULL);
  }

  {
    out << "X-flip symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
.........
.........
..X...X..
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Y-flip symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
..X......
.........
.........
.........
..X......
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Diagonal-flips symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
......X..
.........
.........
.........
..X......
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }


  {
    out << "2 fold-rotational symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.....X...
.........
.........
.........
...X.....
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Square-flips symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
....X....
..O.O.O..
....X....
.........
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Diagonal-flips symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
..X......
.........
.........
.........
......X..
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "2 fold-rotational symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.O.......
.....X...
.........
.........
.........
...X.....
.......O.
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "4 fold-rotational symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.....X...
..X......
.........
......X..
...X.....
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "4 fold-rotational symmetry white next" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.....X...
..X......
.........
......X..
...X.....
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_WHITE,NULL);
  }

  {
    out << "UR-only diagonal flip symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
....O.X..
.........
......O..
...X.....
.........
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "UL-only diagonal flip symmetry" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
..O......
.........
....X....
.....OX..
.....X...
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Not symmetric" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.....O...
.........
.........
.........
...X.....
.........
.........
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Empty rectangular board" << endl;
    Board board = Board::parseBoard(5,7,R"%%(
.....
.....
.....
.....
.....
.....
.....
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Fully symmetric rectangular board" << endl;
    Board board = Board::parseBoard(5,7,R"%%(
..O..
.....
.....
O.X.O
.....
.....
..O..
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "X-flip rectangular board" << endl;
    Board board = Board::parseBoard(5,7,R"%%(
.....
.....
.....
.....
.....
.X.X.
.....
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Y-flip rectangular board" << endl;
    Board board = Board::parseBoard(5,7,R"%%(
.....
.X...
.....
.....
.....
.X...
.....
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Both-flip rectangular board" << endl;
    Board board = Board::parseBoard(5,7,R"%%(
.....
...X.
.....
.....
.....
.X...
.....
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "Both-flip rectangular board white next" << endl;
    Board board = Board::parseBoard(5,7,R"%%(
.....
...X.
.....
.....
.....
.X...
.....
)%%");
    computeAndPrintMarkedSymDupArea(board,P_WHITE,NULL);
  }

  {
    out << "Not symmetric" << endl;
    Board board = Board::parseBoard(5,7,R"%%(
.....
.....
.....
.....
.....
.X...
.....
)%%");
    computeAndPrintMarkedSymDupArea(board,P_BLACK,NULL);
  }

  {
    out << "4 fold-rotational symmetry, no 5 6" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.....X...
..X......
.........
......X..
...X.....
.........
.........
)%%");
    vector<int> onlySymmetries = {0,1,2,3,4,7};
    computeAndPrintMarkedSymDupArea(board,P_BLACK,&onlySymmetries);
  }

  {
    out << "Empty board, no 1,2,4,7" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
.........
.........
.........
.........
.........
)%%");
    vector<int> onlySymmetries = {0,3,5,6};
    computeAndPrintMarkedSymDupArea(board,P_BLACK,&onlySymmetries);
  }

  {
    out << "Empty board, only hflip" << endl;
    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
.........
.........
.........
.........
.........
)%%");
    vector<int> onlySymmetries = {0,2};
    computeAndPrintMarkedSymDupArea(board,P_BLACK,&onlySymmetries);
  }

  string expected = R"%%(
Fully symmetric board
HASH: 9F0B2D702FC8448C75410E097F089AEB
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . . . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 1 2 3 4 5 6 7
xxxxxxxx.
xxxxxxx..
xxxxxx...
xxxxx....
xxxx.....
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx

Fully symmetric board white next
HASH: 9F0B2D702FC8448C75410E097F089AEB
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . . . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 1 2 3 4 5 6 7
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx
.....xxxx
....xxxxx
...xxxxxx
..xxxxxxx
.xxxxxxxx

X-flip symmetry
HASH: AC45122339406741350EF7164F7537BA
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . . . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . X . . . X . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 2
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....

Y-flip symmetry
HASH: 5E99A27FD0F9F1F5CE83F28467EC0A20
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . X . . . . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . X . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 1
.........
.........
.........
.........
.........
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx

Diagonal-flips symmetry
HASH: B7760937ABCCDDEB7806FB3350688799
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . . X . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . X . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 3 4 7
xxxxxxxx.
xxxxxxx..
xxxxxx...
xxxxx....
xxxx.....
xxxxx....
xxxxxx...
xxxxxxx..
xxxxxxxx.

2 fold-rotational symmetry
HASH: 6D5019C791BAB0D7E9D8E41D4F90487D
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . X . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . X . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 3
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxxx....
xxxxx....
xxxxx....
xxxxx....

Square-flips symmetry
HASH: 800DA33B7624879BE7EA5FDB80A3BCA0
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . . . . .
 6 . . . . X . . . .
 5 . . O . O . O . .
 4 . . . . X . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 1 2 3
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx

Diagonal-flips symmetry
HASH: 6DD79D2CC671D2388ECC0B9B5791A771
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . X . . . . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . . . . X . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 3 4 7
xxxxxxxx.
xxxxxxx..
xxxxxx...
xxxxx....
xxxx.....
xxxxx....
xxxxxx...
xxxxxxx..
xxxxxxxx.

2 fold-rotational symmetry
HASH: C187B4A7C45ADFEC2010F47B431E6ABC
   A B C D E F G H J
 9 . . . . . . . . .
 8 . O . . . . . . .
 7 . . . . . X . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . X . . . . .
 2 . . . . . . . O .
 1 . . . . . . . . .


Symmetries: 0 3
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxxx....
xxxxx....
xxxxx....
xxxxx....

4 fold-rotational symmetry
HASH: 9A99D63A6BCF4A80259685FCC27D968C
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . X . . .
 6 . . X . . . . . .
 5 . . . . . . . . .
 4 . . . . . . X . .
 3 . . . X . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 3 5 6
xxxxxxxx.
xxxxxxx..
xxxxxx...
xxxxx....
xxxx.....
xxxxxx...
xxxxxxx..
xxxxxxxx.
xxxxxxxxx

4 fold-rotational symmetry white next
HASH: 9A99D63A6BCF4A80259685FCC27D968C
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . X . . .
 6 . . X . . . . . .
 5 . . . . . . . . .
 4 . . . . . . X . .
 3 . . . X . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 3 5 6
xxxxxxxxx
.xxxxxxxx
..xxxxxxx
...xxxxxx
.....xxxx
....xxxxx
...xxxxxx
..xxxxxxx
.xxxxxxxx

UR-only diagonal flip symmetry
HASH: B58EB39FCBE4BB347271756E0404D59E
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . O . X . .
 6 . . . . . . . . .
 5 . . . . . . O . .
 4 . . . X . . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 7
xxxxxxxx.
xxxxxxx..
xxxxxx...
xxxxx....
xxxx.....
xxx......
xx.......
x........
.........

UL-only diagonal flip symmetry
HASH: 599EAB2B3EFFA546F09C5912E678BF76
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . O . . . . . .
 6 . . . . . . . . .
 5 . . . . X . . . .
 4 . . . . . O X . .
 3 . . . . . X . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 4
.........
x........
xx.......
xxx......
xxxx.....
xxxxx....
xxxxxx...
xxxxxxx..
xxxxxxxx.

Not symmetric
HASH: 3ABD04808F9C1524634579C78A198436
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . O . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . X . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0
.........
.........
.........
.........
.........
.........
.........
.........
.........

Empty rectangular board
HASH: 1F1DB4700ED476F6F9783C1E9C3A563A
   A B C D E
 7 . . . . .
 6 . . . . .
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . . . . .
 1 . . . . .


Symmetries: 0 1 2 3
xx...
xx...
xx...
xx...
xxxxx
xxxxx
xxxxx

Fully symmetric rectangular board
HASH: 53E2C35DAB4280F5F678CBC9207C87EC
   A B C D E
 7 . . O . .
 6 . . . . .
 5 . . . . .
 4 O . X . O
 3 . . . . .
 2 . . . . .
 1 . . O . .


Symmetries: 0 1 2 3
xx...
xx...
xx...
xx...
xxxxx
xxxxx
xxxxx

X-flip rectangular board
HASH: 3A7FF58A0936DC8EF710C7A589001C63
   A B C D E
 7 . . . . .
 6 . . . . .
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . X . X .
 1 . . . . .


Symmetries: 0 2
xx...
xx...
xx...
xx...
xx...
xx...
xx...

Y-flip rectangular board
HASH: 7D78242056062482634520C44021A007
   A B C D E
 7 . . . . .
 6 . X . . .
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . X . . .
 1 . . . . .


Symmetries: 0 1
.....
.....
.....
.....
xxxxx
xxxxx
xxxxx

Both-flip rectangular board
HASH: 7A1C4D8F92C366AB477354585AAE5710
   A B C D E
 7 . . . . .
 6 . . . X .
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . X . . .
 1 . . . . .


Symmetries: 0 3
xx...
xx...
xx...
xx...
xxx..
xxx..
xxx..

Both-flip rectangular board white next
HASH: 7A1C4D8F92C366AB477354585AAE5710
   A B C D E
 7 . . . . .
 6 . . . X .
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . X . . .
 1 . . . . .


Symmetries: 0 3
..xxx
..xxx
..xxx
...xx
...xx
...xx
...xx

Not symmetric
HASH: 35099DA736AFB3AE345C49821CE3F188
   A B C D E
 7 . . . . .
 6 . . . . .
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . X . . .
 1 . . . . .


Symmetries: 0
.....
.....
.....
.....
.....
.....
.....

4 fold-rotational symmetry, no 5 6
HASH: 9A99D63A6BCF4A80259685FCC27D968C
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . X . . .
 6 . . X . . . . . .
 5 . . . . . . . . .
 4 . . . . . . X . .
 3 . . . X . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 3
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxxx....
xxxxx....
xxxxx....
xxxxx....

Empty board, no 1,2,4,7
HASH: 9F0B2D702FC8448C75410E097F089AEB
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . . . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 3 5 6
xxxxxxxx.
xxxxxxx..
xxxxxx...
xxxxx....
xxxx.....
xxxxxx...
xxxxxxx..
xxxxxxxx.
xxxxxxxxx

Empty board, only hflip
HASH: 9F0B2D702FC8448C75410E097F089AEB
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 . . . . . . . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Symmetries: 0 2
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....
xxxx.....

)%%";

  expect(name,out,expected);

}


void Tests::runSymmetryDifferenceTests() {
  cout << "Running symmetry difference tests" << endl;
  ostringstream out;

  auto testSymmetryDifferences = [&out](const Board& board, const Board& other, double maxDifferenceToReport) {
    double diffs[SymmetryHelpers::NUM_SYMMETRIES];
    SymmetryHelpers::getSymmetryDifferences(
      board, other, maxDifferenceToReport, diffs
    );
    out << Global::strprintf("%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f", diffs[0], diffs[1], diffs[2], diffs[3], diffs[4], diffs[5], diffs[6], diffs[7]) << endl;
  };

  {
    const char* name = "Simple difference tests";

    Board board = Board::parseBoard(5,5,R"%%(
.....
..o..
.xxx.
..o..
...x.
)%%");
    Board board2 = Board::parseBoard(5,5,R"%%(
.....
..x..
.oxo.
x.x..
.....
)%%");
    Board board3 = Board::parseBoard(5,5,R"%%(
.....
x.x..
.o.o.
..x..
.....
)%%");
    Board board4 = Board::parseBoard(5,5,R"%%(
.x...
..x..
.ooo.
..x..
.....
)%%");

    testSymmetryDifferences(board,board,20.0);
    testSymmetryDifferences(board,board2,20.0);
    testSymmetryDifferences(board,board3,20.0);
    testSymmetryDifferences(board,board4,20.0);
    testSymmetryDifferences(board,board4,7.0);

    string expected = R"%%(
0.0 2.0 2.0 2.0 14.0 14.0 14.0 14.0
14.0 14.0 14.0 14.0 2.0 0.0 2.0 2.0
15.0 15.0 15.0 15.0 3.0 3.0 3.0 1.0
17.0 17.0 17.0 15.0 5.0 5.0 5.0 5.0
7.0 7.0 7.0 7.0 5.0 5.0 5.0 5.0
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "All 8 symmetries difference tests";

    Board board5 = Board::parseBoard(5,5,R"%%(
xo.oo
.x.o.
...o.
xo.x.
x.x.o
)%%");

    for(int i = 0; i<8; i++) {
      Board b = SymmetryHelpers::getSymBoard(board5,i);
      testSymmetryDifferences(board5,b,99.0);
    }
    string expected = R"%%(
0.0 20.0 28.0 22.0 18.0 29.0 29.0 16.0
20.0 0.0 22.0 28.0 29.0 16.0 18.0 29.0
28.0 22.0 0.0 20.0 29.0 18.0 16.0 29.0
22.0 28.0 20.0 0.0 16.0 29.0 29.0 18.0
18.0 29.0 29.0 16.0 0.0 20.0 28.0 22.0
29.0 16.0 18.0 29.0 20.0 0.0 22.0 28.0
29.0 18.0 16.0 29.0 28.0 22.0 0.0 20.0
16.0 29.0 29.0 18.0 22.0 28.0 20.0 0.0
)%%";
    expect(name,out,expected);
  }
}
