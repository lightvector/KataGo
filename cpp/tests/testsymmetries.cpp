#include "../tests/tests.h"

#include "../neuralnet/nninputs.h"

using namespace std;
using namespace TestCommon;

void Tests::runBasicSymmetryTests() {
  cout << "Running basic symmetries tests" << endl;

  const char* name = "Basic flipping of tensors and boards and locations";
  ostringstream out;
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

    for(int symmetry = 0; symmetry<8; symmetry++) {
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

    for(int symmetry = 0; symmetry<8; symmetry++) {
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

void Tests::runBoardSymmetryTests() {
  cout << "Running board symmetry tests" << endl;

  ostringstream out;
  auto printMarkedSymDupArea = [&out](const Board& board, const bool* isSymDupLoc){
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

  auto computeAndPrintMarkedSymDupArea = [&out,&printMarkedSymDupArea](const Board& board) {
    BoardHistory hist(board,P_BLACK,Rules::getTrompTaylorish(),0);
    bool isSymDupLoc[Board::MAX_ARR_SIZE];
    SymmetryHelpers::markDuplicateMoveLocs(board,hist,isSymDupLoc);
    out << board << endl;
    printMarkedSymDupArea(board,isSymDupLoc);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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
    computeAndPrintMarkedSymDupArea(board);
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


xxxxxxxx.
xxxxxxx..
xxxxxx...
xxxxx....
xxxx.....
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx
xxxxxxxxx

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


xxxxxxxx.
xxxxxxx..
xxxxxx...
xxxxx....
xxxx.....
xxxxxx...
xxxxxxx..
xxxxxxxx.
xxxxxxxxx

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


xx...
xx...
xx...
xx...
xxx..
xxx..
xxx..

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


.....
.....
.....
.....
.....
.....
.....
)%%";

  expect(name,out,expected);
}
