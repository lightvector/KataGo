#include "../tests/tests.h"

#include <algorithm>
#include <iterator>

#include "../dataio/sgf.h"
#include "../search/asyncbot.h"

using namespace std;
using namespace TestCommon;

void Tests::runSgfTests() {
  cout << "Running sgf tests" << endl;
  ostringstream out;

  auto parseAndPrintSgf = [&out](const string& sgfStr) {
    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    out << "xSize " << sgf->xSize << endl;
    out << "ySize " << sgf->ySize << endl;
    out << "depth " << sgf->depth << endl;
    out << "komi " << sgf->komi << endl;

    Board board;
    BoardHistory hist;
    Rules rules;
    Player pla;
    rules = sgf->getRulesOrFailAllowUnspecified(rules);
    sgf->setupInitialBoardAndHist(rules,board,pla,hist);

    out << "placements" << endl;
    for(int i = 0; i < sgf->placements.size(); i++) {
      Move move = sgf->placements[i];
      out << PlayerIO::colorToChar(move.pla) << " " << Location::toString(move.loc,board) << endl;
    }
    out << "moves" << endl;
    for(int i = 0; i < sgf->moves.size(); i++) {
      Move move = sgf->moves[i];
      out << PlayerIO::colorToChar(move.pla) << " " << Location::toString(move.loc,board) << endl;
    }

    out << "Initial board hist " << endl;
    out << "pla " << PlayerIO::playerToString(pla) << endl;
    hist.printDebugInfo(out,board);

    sgf->setupBoardAndHistAssumeLegal(rules,board,pla,hist,sgf->moves.size());
    out << "Final board hist " << endl;
    out << "pla " << PlayerIO::playerToString(pla) << endl;
    hist.printDebugInfo(out,board);

    {
      //Test SGF writing roundtrip.
      //This is not exactly holding if there is pass for ko, but should be good in all other cases
      ostringstream out2;
      WriteSgf::writeSgf(out2,"foo","bar",hist,NULL,false);
      CompactSgf* sgf2 = CompactSgf::parse(out2.str());
      Board board2;
      BoardHistory hist2;
      Rules rules2;
      Player pla2;
      rules2 = sgf2->getRulesOrFail();
      sgf->setupBoardAndHistAssumeLegal(rules2,board2,pla2,hist2,sgf2->moves.size());
      testAssert(rules2 == rules);
      testAssert(board2.pos_hash == board.pos_hash);
      testAssert(hist2.moveHistory.size() == hist.moveHistory.size());
      delete sgf2;
    }

    delete sgf;
  };

  //============================================================================
  {
    const char* name = "Basic Sgf parse test";
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Tromp-Taylor]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
    parseAndPrintSgf(sgfStr);
    string expected = R"%%(
xSize 19
ySize 19
depth 8
komi 5
placements
X D16
X Q16
X D4
X Q4
moves
O R14
O N16
X Q14
O Q13
X P14
O pass
X pass
Initial board hist
pla White
HASH: B7F8C756D3C44C031B6A7CDF9164EDA7
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
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
HASH: DD088CF25D937776F4CC6E2CBC169CD4
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . O . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . X X O . .
13 . . . . . . . . . . . . . . . O . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 7
Rules koPOSITIONALscoreAREAtaxNONEsui1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 1 White 2 1 0 0
Last moves R14 N16 Q14 Q13 P14 pass pass

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Japanese Sgf parse test";
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
    parseAndPrintSgf(sgfStr);
    string expected = R"%%(
xSize 19
ySize 19
depth 8
komi 5
placements
X D16
X Q16
X D4
X Q4
moves
O R14
O N16
X Q14
O Q13
X P14
O pass
X pass
Initial board hist
pla White
HASH: B7F8C756D3C44C031B6A7CDF9164EDA7
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
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Rules koSIMPLEscoreTERRITORYtaxSEKIsui0komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 4
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
HASH: DD088CF25D937776F4CC6E2CBC169CD4
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . O . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . X X O . .
13 . . . . . . . . . . . . . . . O . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 1
Turns this phase 0
Rules koSIMPLEscoreTERRITORYtaxSEKIsui0komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 3
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves R14 N16 Q14 Q13 P14 pass pass

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Chinese Sgf parse test";
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
    parseAndPrintSgf(sgfStr);
    string expected = R"%%(
xSize 19
ySize 19
depth 8
komi 5
placements
X D16
X Q16
X D4
X Q4
moves
O R14
O N16
X Q14
O Q13
X P14
O pass
X pass
Initial board hist
pla White
HASH: B7F8C756D3C44C031B6A7CDF9164EDA7
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
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Rules koSIMPLEscoreAREAtaxNONEsui0whbNkomi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 4
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
HASH: DD088CF25D937776F4CC6E2CBC169CD4
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . O . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . X X O . .
13 . . . . . . . . . . . . . . . O . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 7
Rules koSIMPLEscoreAREAtaxNONEsui0whbNkomi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 4
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 1 White 6 1 0 0
Last moves R14 N16 Q14 Q13 P14 pass pass

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "AGA Sgf parse test";
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[AGA]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
    parseAndPrintSgf(sgfStr);
    string expected = R"%%(
xSize 19
ySize 19
depth 8
komi 5
placements
X D16
X Q16
X D4
X Q4
moves
O R14
O N16
X Q14
O Q13
X P14
O pass
X pass
Initial board hist
pla White
HASH: B7F8C756D3C44C031B6A7CDF9164EDA7
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
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Rules koSITUATIONALscoreAREAtaxNONEsui0whbN-1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 3
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
HASH: DD088CF25D937776F4CC6E2CBC169CD4
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . .
16 . . . X . . . . . . . . O . . X . . .
15 . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . X X O . .
13 . . . . . . . . . . . . . . . O . . .
12 . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . .
 4 . . . X . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 7
Rules koSITUATIONALscoreAREAtaxNONEsui0whbN-1komi5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 3
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 1 White 5 1 0 0
Last moves R14 N16 Q14 Q13 P14 pass pass

)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Rectangle Sgf parse test";
    string sgfStr = "(;GM[1]FF[4]SZ[17:3]KM[-6.5];B[fc];W[cc];B[la];)";
    parseAndPrintSgf(sgfStr);
    string expected = R"%%(
xSize 17
ySize 3
depth 5
komi -6.5
placements
moves
X F1
O C1
X M3
Initial board hist
pla Black
HASH: A8A8A5ADA4E1BFB3AEB041A4513D7405
   A B C D E F G H J K L M N O P Q R
 3 . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi-6.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
HASH: B39271113663636A4AF3B55AFB01F6AB
   A B C D E F G H J K L M N O P Q R
 3 . . . . . . . . . . . X . . . . .
 2 . . . . . . . . . . . . . . . . .
 1 . . O . . X . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 3
Rules koPOSITIONALscoreAREAtaxNONEsui1komi-6.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves F1 C1 M3
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Sgf parsing with whitespace and placements and comments";
    string sgfStr = R"%%((;GM[1]FF[4]SZ[9]
GN[]
C[Diagram

]
PL[W]

AB[bc][dc][fc]
AW[ac][cc]AW[ec]


;
))%%";
    parseAndPrintSgf(sgfStr);
    string expected = R"%%(
xSize 9
ySize 9
depth 2
komi 7.5
placements
X B7
X D7
X F7
O A7
O C7
O E7
moves
Initial board hist
pla White
HASH: B9DEED0632FD395A12CA242D89060D3B
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 O X O X O X . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi7.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
HASH: B9DEED0632FD395A12CA242D89060D3B
   A B C D E F G H J
 9 . . . . . . . . .
 8 . . . . . . . . .
 7 O X O X O X . . .
 6 . . . . . . . . .
 5 . . . . . . . . .
 4 . . . . . . . . .
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .


Initial pla White
Encore phase 0
Turns this phase 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi7.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  if(Board::MAX_LEN >= 37)
  {
    const char* name = "Giant Sgf parse test";
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Chinese]SZ[37]KM[0.00];B[dd];W[Hd];B[HH];W[dH];B[dG];W[eG];B[eF];W[Gd];B[Ge];W[He];B[ee];W[GG];B[ss])";

    parseAndPrintSgf(sgfStr);
    string expected = R"%%(
xSize 37
ySize 37
depth 14
komi 0
placements
moves
X (3,3)
O (33,3)
X (33,33)
O (3,33)
X (3,32)
O (4,32)
X (4,31)
O (32,3)
X (32,4)
O (33,4)
X (4,4)
O (32,32)
X (18,18)
Initial board hist
pla Black
HASH: B7D1534B9B7F9D0424902AFED43500FC
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Rules koSIMPLEscoreAREAtaxNONEsui0komi0
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves
Final board hist
pla White
HASH: 04E936BD3026457C27E5E849DED6AE6A
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . X . . . . . . . . . . . . . . . . . . . . . . . . . . . . O O . . .
. . . . X . . . . . . . . . . . . . . . . . . . . . . . . . . . X O . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . X . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . X . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . X O . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . .
. . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . X . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 13
Rules koSIMPLEscoreAREAtaxNONEsui0komi0
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves (3,3) (33,3) (33,33) (3,33) (3,32) (4,32) (4,31) (32,3) (32,4) (33,4) (4,4) (32,32) (18,18)
)%%";
    expect(name,out,expected);
  }

}
