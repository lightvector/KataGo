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
      WriteSgf::writeSgf(out2,"foo","bar",hist,NULL,false,false);
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
HASH: 0483785A1D3D994549631CC1DE3E1CCE
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
HASH: D2679AE98871290D03776F89C5C34607
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
HASH: 0483785A1D3D994549631CC1DE3E1CCE
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
HASH: D2679AE98871290D03776F89C5C34607
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
HASH: 0483785A1D3D994549631CC1DE3E1CCE
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
HASH: D2679AE98871290D03776F89C5C34607
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
HASH: 0483785A1D3D994549631CC1DE3E1CCE
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
HASH: D2679AE98871290D03776F89C5C34607
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
HASH: 07791CF7FCA8A7E67538A413A98483AB
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
HASH: 52376CECFC43AC1A05655FF3DAF26336
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
HASH: AB9059CC6FB63D2E54306382320C0E8D
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
HASH: AB9059CC6FB63D2E54306382320C0E8D
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
  {
    const char* name = "Sgf parsing with moveless and multimove nodes";
    string sgfStr = "(;GM[1]FF[4]SZ[5]KM[24];B[cc]W[cb];;B[bb];C[test];C[test2];W[dc];B[db];W[cd];;;B[bc];C[test3])";
    parseAndPrintSgf(sgfStr);
    {
      Sgf* sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      vector<Sgf::PositionSample> samples;
      sgf->loadAllUniquePositions(uniqueHashes, false, samples);
      for(int i = 0; i<samples.size(); i++) {
        out << Sgf::PositionSample::toJsonLine(samples[i]) << endl;
      }
      delete sgf;
    }

    string expected = R"%%(
xSize 5
ySize 5
depth 13
komi 24
placements
moves
X C3
O C4
X B4
O D3
X D4
O C2
X B3
Initial board hist
pla Black
HASH: FBE16917FFBF22C1CD6D3A1EEB1FC363
   A B C D E
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . . . . .
 1 . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
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
HASH: F319D79992F6E6C1ABF52DB6631D470B
   A B C D E
 5 . . . . .
 4 . X O X .
 3 . X X O .
 2 . . O . .
 1 . . . . .


Initial pla Black
Encore phase 0
Turns this phase 7
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves C3 C4 B4 D3 D4 C2 B3
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4"],"movePlas":["B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4","D3"],"movePlas":["B","W","B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4","D3","D4"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["C4","B4","D3","D4","C2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4","C2","B3"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Sgf parsing with more multimove nodes, katago doesn't handle ordering but that's okay since this is not actually valid sgf";
    string sgfStr = "(;GM[1]FF[4]SZ[5]KM[24];B[cc]W[cb]B[bb];W[dc]B[db];W[cd];;;B[bc])";
    parseAndPrintSgf(sgfStr);
    {
      Sgf* sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      vector<Sgf::PositionSample> samples;
      sgf->loadAllUniquePositions(uniqueHashes, false, samples);
      for(int i = 0; i<samples.size(); i++) {
        out << Sgf::PositionSample::toJsonLine(samples[i]) << endl;
      }
      delete sgf;
    }

    string expected = R"%%(
xSize 5
ySize 5
depth 7
komi 24
placements
moves
X C3
X B4
O C4
X D4
O D3
O C2
X B3
Initial board hist
pla Black
HASH: FBE16917FFBF22C1CD6D3A1EEB1FC363
   A B C D E
 5 . . . . .
 4 . . . . .
 3 . . . . .
 2 . . . . .
 1 . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
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
HASH: F319D79992F6E6C1ABF52DB6631D470B
   A B C D E
 5 . . . . .
 4 . X O X .
 3 . X X O .
 2 . . O . .
 1 . . . . .


Initial pla Black
Encore phase 0
Turns this phase 7
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves C3 B4 C4 D4 D3 C2 B3
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["B4"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["B4","C4"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["B4","C4","D4"],"movePlas":["B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["B4","C4","D4","D3"],"movePlas":["B","W","B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XOX./..XO./...../...../","hintLoc":"null","initialTurnNumber":5,"moveLocs":["C2"],"movePlas":["W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XOX./..XO./...../...../","hintLoc":"null","initialTurnNumber":5,"moveLocs":["C2","B3"],"movePlas":["W","B"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5})%%";
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
X D34
O AJ34
X AJ4
O D4
X D5
O E5
X E6
O AH34
X AH33
O AJ33
X E33
O AH5
X T19
Initial board hist
pla Black
HASH: 09D2EA49BF2F78E5B210AEA0C9838C85
   A B C D E F G H J K L M N O P Q R S T U V W X Y ZAAABACADAEAFAGAHAJAKALAM
37 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
36 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
35 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
34 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
33 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
32 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
31 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 5 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 0
Rules koSIMPLEscoreAREAtaxNONEsui0whbNkomi0
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
HASH: BEECF0ED5AC11C8DBA6C50AAF871843D
   A B C D E F G H J K L M N O P Q R S T U V W X Y ZAAABACADAEAFAGAHAJAKALAM
37 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
36 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
35 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
34 . . . X . . . . . . . . . . . . . . . . . . . . . . . . . . . . O O . . .
33 . . . . X . . . . . . . . . . . . . . . . . . . . . . . . . . . X O . . .
32 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
31 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
30 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
29 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
28 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
27 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
26 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
23 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
22 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
21 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
20 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
19 . . . . . . . . . . . . . . . . . . X . . . . . . . . . . . . . . . . . .
18 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
17 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
16 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
15 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
13 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
12 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
11 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
10 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 9 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 8 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 6 . . . . X . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 5 . . . X O . . . . . . . . . . . . . . . . . . . . . . . . . . . O . . . .
 4 . . . O . . . . . . . . . . . . . . . . . . . . . . . . . . . . . X . . .
 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


Initial pla Black
Encore phase 0
Turns this phase 13
Rules koSIMPLEscoreAREAtaxNONEsui0whbNkomi0
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves D34 AJ34 AJ4 D4 D5 E5 E6 AH34 AH33 AJ33 E33 AH5 T19

)%%";
    expect(name,out,expected);
  }

}
