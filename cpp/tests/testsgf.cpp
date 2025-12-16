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

  auto parseAndPrintSgfLinear = [&out](const string& sgfStr) {
    std::unique_ptr<CompactSgf> sgf = CompactSgf::parse(sgfStr);

    out << "xSize " << sgf->xSize << endl;
    out << "ySize " << sgf->ySize << endl;
    out << "depth " << sgf->depth << endl;
    out << "komi " << sgf->getRulesOrFailAllowUnspecified(Rules()).komi << endl;

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
      std::unique_ptr<CompactSgf> sgf2 = CompactSgf::parse(out2.str());
      Board board2;
      BoardHistory hist2;
      Rules rules2;
      Player pla2;
      rules2 = sgf2->getRulesOrFail();
      sgf->setupBoardAndHistAssumeLegal(rules2,board2,pla2,hist2,sgf2->moves.size());
      testAssert(rules2 == rules);
      testAssert(board2.pos_hash == board.pos_hash);
      testAssert(hist2.moveHistory.size() == hist.moveHistory.size());
    }
  };

  auto parseAndPrintSgf = [&out](const string& sgfStr) {
    std::unique_ptr<Sgf> sgf = Sgf::parse(sgfStr);

    // Print SGF metadata
    XYSize xySize = sgf->getXYSize();
    out << "xSize " << xySize.x << endl;
    out << "ySize " << xySize.y << endl;

    out << "komi " << sgf->getKomiOrDefault(NAN) << endl;
    out << "hasRules " << Global::boolToString(sgf->hasRules()) << endl;
    out << "rules ";
    try {
      out << sgf->getRulesOrFail().toString() << endl;
    }
    catch(...) {
      out << "not found" << endl;
    }

    out << "handicapValue " << sgf->getHandicapValue() << endl;
    out << "sgfWinner " << PlayerIO::playerToString(sgf->getSgfWinner()) << endl;
    out << "firstPlayerColor " << PlayerIO::colorToChar(sgf->getFirstPlayerColor()) << endl;

    out << "black rank " << sgf->getRank(P_BLACK) << endl;
    out << "white rank " << sgf->getRank(P_WHITE) << endl;
    out << "black name " << sgf->getPlayerName(P_BLACK) << endl;
    out << "white name " << sgf->getPlayerName(P_WHITE) << endl;

    out << "hasRootProperty(GN) " << Global::boolToString(sgf->hasRootProperty("GN")) << endl;
    out << "root property (GN) " << sgf->getRootPropertyWithDefault("GN", "") << endl;
    out << "hasRootProperty(SZ) " << Global::boolToString(sgf->hasRootProperty("SZ")) << endl;
    out << "root property (SZ) " << sgf->getRootPropertyWithDefault("SZ", "") << endl;

    if(sgf->hasRootProperty("AW")) {
      std::vector<std::string> props = sgf->getRootProperties("AW");
      out << "getRootProperties(AW) size=" << props.size();
      for(const std::string& prop : props) {
        out << " [" << prop << "]";
      }
    }
    out << endl;
    if(sgf->hasRootProperty("AB")) {
      std::vector<std::string> props = sgf->getRootProperties("AB");
      out << "getRootProperties(AB) size=" << props.size();
      for(const std::string& prop : props) {
        out << " [" << prop << "]";
      }
      out << endl;
    }

    std::vector<Move> placements;
    sgf->getPlacements(placements, xySize.x, xySize.y);
    out << "placements " << placements.size() << endl;
    for(const Move& move: placements) {
      out << PlayerIO::playerToString(move.pla) << " " << Location::toString(move.loc, xySize.x, xySize.y) << " ";
    }
    out << endl;

    std::vector<Move> moves;
    sgf->getMoves(moves, xySize.x, xySize.y);
    out << "moves " << moves.size() << endl;
    for(const Move& move: moves) {
      out << PlayerIO::playerToString(move.pla) << " " << Location::toString(move.loc, xySize.x, xySize.y) << " ";
    }
    out << endl;

    out << "depth " << sgf->depth() << endl;
    out << "nodeCount " << sgf->nodeCount() << endl;
    out << "branchCount " << sgf->branchCount() << endl;

    std::set<Hash128> uniqueHashes;
    sgf->iterAllUniquePositions(
      uniqueHashes,
      true,
      false,
      false,
      false,
      NULL,
      [&out](Sgf::PositionSample& sample, const BoardHistory& hist, const std::string& comments) {
        out << sample.board.pos_hash << endl;
        out << "Comments: " << comments << endl;
        Board::printBoard(out, hist.getRecentBoard(0), Board::NULL_LOC, &(hist.moveHistory));
      }
    );
  };

  //============================================================================
  {
    const char* name = "Basic Sgf parse test";
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Tromp-Taylor]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
    parseAndPrintSgfLinear(sgfStr);
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
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
Approx valid turns this phase 7
Approx consec valid turns this game 5
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
    parseAndPrintSgfLinear(sgfStr);
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
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
Approx valid turns this phase 0
Approx consec valid turns this game 5
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
    parseAndPrintSgfLinear(sgfStr);
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koSIMPLEscoreAREAtaxNONEsui0whbNfpok1komi5
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
Approx valid turns this phase 7
Approx consec valid turns this game 5
Rules koSIMPLEscoreAREAtaxNONEsui0whbNfpok1komi5
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
    parseAndPrintSgfLinear(sgfStr);
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
Rules koSITUATIONALscoreAREAtaxNONEsui0whbN-1fpok1komi5
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
Approx valid turns this phase 7
Approx consec valid turns this game 5
Rules koSITUATIONALscoreAREAtaxNONEsui0whbN-1fpok1komi5
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
    parseAndPrintSgfLinear(sgfStr);
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
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
Approx valid turns this phase 3
Approx consec valid turns this game 3
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
    parseAndPrintSgfLinear(sgfStr);
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
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
    parseAndPrintSgfLinear(sgfStr);
    {
      std::unique_ptr<Sgf> sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      vector<Sgf::PositionSample> samples;
      sgf->loadAllUniquePositions(uniqueHashes, false, false, false, false, NULL, samples);
      for(int i = 0; i<samples.size(); i++) {
        out << Sgf::PositionSample::toJsonLine(samples[i]) << endl;
        Sgf::PositionSample reloaded = Sgf::PositionSample::ofJsonLine(Sgf::PositionSample::toJsonLine(samples[i]));
        bool checkNumCaptures = false;
        bool checkSimpleKo = false;
        testAssert(samples[i].isEqualForTesting(reloaded,checkNumCaptures,checkSimpleKo));
      }
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
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
Approx valid turns this phase 7
Approx consec valid turns this game 7
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves C3 C4 B4 D3 D4 C2 B3
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
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
    parseAndPrintSgfLinear(sgfStr);
    {
      std::unique_ptr<Sgf> sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      vector<Sgf::PositionSample> samples;
      sgf->loadAllUniquePositions(uniqueHashes, false, false, false, false, NULL, samples);
      for(int i = 0; i<samples.size(); i++) {
        out << Sgf::PositionSample::toJsonLine(samples[i]) << endl;
        Sgf::PositionSample reloaded = Sgf::PositionSample::ofJsonLine(Sgf::PositionSample::toJsonLine(samples[i]));
        bool checkNumCaptures = false;
        bool checkSimpleKo = false;
        testAssert(samples[i].isEqualForTesting(reloaded,checkNumCaptures,checkSimpleKo));
      }
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
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
Approx valid turns this phase 7
Approx consec valid turns this game 1
Rules koPOSITIONALscoreAREAtaxNONEsui1komi24
Ko recap block hash 00000000000000000000000000000000
White bonus score 0
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves C3 B4 C4 D4 D3 C2 B3
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
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

    parseAndPrintSgfLinear(sgfStr);
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
Approx valid turns this phase 0
Approx consec valid turns this game 0
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
Approx valid turns this phase 13
Approx consec valid turns this game 13
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

  {
    const char* name = "Branching sgf parse test";
    string sgfStr = R"%%(
(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3] ST[2]
RU[Japanese]SZ[5]KM[12.50]
PW[WhitePlayer]PB[BlackPlayer]HA[5]RE [B+1]AW[aa][ab]PL[B]
(;B[cc]C[Center move!]
(;W[dc]
;B[dd]
((( ;W[cb])))
(;W[db]
;B[cb]) )
(;W[cd])
(;W[dd] ))
(; B[cd])
(;B [dd]
(;W[cc]
(;B[dc]
;W[cb]C[Other branch])))
(;W[cc] C[White first]
;B[cd]
;B[dc]C[Black twice in a row.]))

)%%";
    parseAndPrintSgf(sgfStr);
    string expected = R"%%(
xSize 5
ySize 5
komi 12.5
hasRules true
rules koSIMPLEscoreTERRITORYtaxSEKIsui0komi12.5
handicapValue 5
sgfWinner Black
firstPlayerColor X
black rank -100000
white rank -100000
black name BlackPlayer
white name WhitePlayer
hasRootProperty(GN) false
root property (GN)
hasRootProperty(SZ) true
root property (SZ) 5
getRootProperties(AW) size=2 [aa] [ab]
placements 2
White A5 White A4
moves 5
Black C3 White D3 Black D2 White D4 Black C4
depth 6
nodeCount 17
branchCount 7
8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 0 HASH: 8FF4DE5FEC9AA30E0B7F337FA8338312
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . . . .
 2 . . . . .
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments: Center move!
MoveNum: 1 HASH: 00561C44119943C7F15A3C0CD562853B
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . X1. .
 2 . . . . .
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 2 HASH: 7BCDC59BC772FCC1C5AB0862270835D7
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . X1O2.
 2 . . . . .
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 3 HASH: E70BD738FD88F770A568E8748B76B5FD
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . X1O2.
 2 . . . X3.
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 4 HASH: 371F469509A1716324722FE39C903FBB
   A B C D E
 5 O . . . .
 4 O . O3. .
 3 . . X O1.
 2 . . . X2.
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 4 HASH: 00B639564F40F72FB4ABE14152825E76
   A B C D E
 5 O . . . .
 4 O . . O3.
 3 . . X O1.
 2 . . . X2.
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 5 HASH: D55A7372F7BBBC040B311E0FC7F1A5CC
   A B C D E
 5 O . . . .
 4 O . X3O2.
 3 . . X O .
 2 . . . X1.
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 2 HASH: 386B5D6A7B4B96619EFE2C5D2C34E0B8
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . X1. .
 2 . . O2. .
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 2 HASH: E258247ABC08E3A7312A3BFDF7AE8D4C
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . X1. .
 2 . . . O2.
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 1 HASH: D22F053328FE381595D4608CE97B2763
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . . . .
 2 . . X1. .
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 1 HASH: 1332CCFCD660A8BF6BBCD369044D0338
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . . . .
 2 . . . X1.
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 2 HASH: 63F5B4E07ED10D48C9079BCE9915632E
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . O2. .
 2 . . . X1.
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 3 HASH: AC22251861209D892A038B0A22B88F36
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . O2X3.
 2 . . . X1.
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments: Other branch
MoveNum: 4 HASH: 7C36B4B595091B9AAB194C9D355E0570
   A B C D E
 5 O . . . .
 4 O . O3. .
 3 . . O1X2.
 2 . . . X .
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments: White first
MoveNum: 1 HASH: FF33A643442B06F9A9C47BD8356BE304
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . O1. .
 2 . . . . .
 1 . . . . .

8FF4DE5FEC9AA30E0B7F337FA8338312
Comments:
MoveNum: 2 HASH: A2E87D2F804F9DE2376F282B74234775
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . O1. .
 2 . . X2. .
 1 . . . . .

A2E87D2F804F9DE2376F282B74234775
Comments: Black twice in a row.
MoveNum: 3 HASH: 6D3FECD79FBE0D23D46B38EFCF8EAB6D
   A B C D E
 5 O . . . .
 4 O . . . .
 3 . . O1X3.
 2 . . X2. .
 1 . . . . .

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "More rigorous test of positionsample parsing";
    (void)name;
    string sgfStr = "(;FF[4]GM[1]SZ[9:17]HA[0]KM[7.5]RU[koSIMPLEscoreAREAtaxSEKIsui0]RE[W+32.5];B[fo];W[dd];B[fc];W[fd];B[ec];W[ed];B[dc];W[cd];B[gd];W[ge];B[hd];W[dn];B[fm];W[eo];B[fp];W[el];B[ck];W[di];B[bn];W[fl];B[gm];W[gl];B[ci];W[ch];B[dh];W[ei];B[em];W[dm];B[dl];W[bm];B[ek];W[hm];B[hn];W[cj];B[bk];W[hl];B[cm];W[cn];B[bo];W[cl];B[dk];W[ep];B[hp];W[cp];B[fj];W[gi];B[am];W[al];B[fi];W[gh];B[bl];W[an];B[cm];W[in];B[ho];W[cl];B[gj];W[hj];B[cm];W[gc];B[fh];W[gg];B[gb];W[cl];B[hk];W[gk];B[cm];W[fn];B[gn];W[cl];B[ik];W[ij];B[cm];W[hc];B[hb];W[cl];B[hi];W[il];B[cm];W[bp];B[bi];W[eh];B[am];W[bh];B[dg];W[bm];B[fg];W[he];B[am];W[ap];B[cf];W[bj];B[eg];W[id];B[cc];W[bm];B[bc];W[bd];B[bf];W[cl];B[ak];W[cm];B[ff];W[gf];B[ac];W[ae];B[af];W[ai];B[ib];W[df];B[ef];W[cg];B[de];W[ee];B[df];W[be];B[aj];W[fe];B[ej];W[ci];B[ah];W[ag];B[bg];W[ce];B[am];W[fk];B[ah];W[fq];B[bi];W[gp];B[go];W[ai];B[en];W[ag];B[co];W[al];B[ah];W[am];B[bi];W[hq];B[gq];W[ai];B[do];W[ao];B[bi];W[gp];B[ii];W[gq];B[ic];W[ad];B[dj];W[hh];B[bh];W[dp];B[fb];W[ih];B[];W[hd];B[];W[ip];B[];W[io];B[];W[fn];B[];W[])";
    {
      std::unique_ptr<Sgf> sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      vector<Sgf::PositionSample> samples;
      sgf->loadAllUniquePositions(uniqueHashes, false, false, false, false, NULL, samples);
      for(int i = 0; i<samples.size(); i++) {
        samples[i].weight = i * 0.5;
        Sgf::PositionSample reloaded = Sgf::PositionSample::ofJsonLine(Sgf::PositionSample::toJsonLine(samples[i]));
        bool checkNumCaptures = false;
        bool checkSimpleKo = false;
        testAssert(samples[i].isEqualForTesting(reloaded,checkNumCaptures,checkSimpleKo));
      }
    }
  }


  //============================================================================
  {
    const char* name = "Sgf parsing with white first, and flip";
    string sgfStr = "(;GM[1]FF[4]SZ[5]KM[24];W[cc];B[cb];W[bb];B[dc];W[db];B[cd];W[bc];B[dd];W[bd])";
    {
      std::unique_ptr<Sgf> sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      vector<Sgf::PositionSample> samples;
      sgf->loadAllUniquePositions(uniqueHashes, false, false, true, false, NULL, samples);
      for(int i = 0; i<samples.size(); i++) {
        out << Sgf::PositionSample::toJsonLine(samples[i]) << endl;
        Sgf::PositionSample reloaded = Sgf::PositionSample::ofJsonLine(Sgf::PositionSample::toJsonLine(samples[i]));
        bool checkNumCaptures = false;
        bool checkSimpleKo = false;
        testAssert(samples[i].isEqualForTesting(reloaded,checkNumCaptures,checkSimpleKo));
      }
    }

    string expected = R"%%(
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4"],"movePlas":["B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4","D3"],"movePlas":["B","W","B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C3","C4","B4","D3","D4"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../..X../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["C4","B4","D3","D4","C2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4","C2","B3"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..X../...../...../","hintLoc":"null","initialTurnNumber":3,"moveLocs":["D3","D4","C2","B3","D2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..XO./...../...../","hintLoc":"null","initialTurnNumber":4,"moveLocs":["D4","C2","B3","D2","B2"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Sgf parsing with black pass, and flip";
    string sgfStr = "(;GM[1]FF[4]SZ[5]KM[24];B[cb];W[cc];B[];W[bb];B[dc];W[db];B[cd];W[bc];B[dd];W[bd];B[])";
    {
      std::unique_ptr<Sgf> sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      vector<Sgf::PositionSample> samples;
      sgf->loadAllUniquePositions(uniqueHashes, false, false, true, false, NULL, samples);
      for(int i = 0; i<samples.size(); i++) {
        out << Sgf::PositionSample::toJsonLine(samples[i]) << endl;
        Sgf::PositionSample reloaded = Sgf::PositionSample::ofJsonLine(Sgf::PositionSample::toJsonLine(samples[i]));
        bool checkNumCaptures = false;
        bool checkSimpleKo = false;
        testAssert(samples[i].isEqualForTesting(reloaded,checkNumCaptures,checkSimpleKo));
      }
    }

    string expected = R"%%(
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3","pass"],"movePlas":["W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3","pass","B4"],"movePlas":["W","B","W","B"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3","pass","B4","D3"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../...../...../...../","hintLoc":"null","initialTurnNumber":1,"moveLocs":["C3","pass","B4","D3","D4"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["pass","B4","D3","D4","C2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":3,"moveLocs":["B4","D3","D4","C2","B3"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..X../...../...../","hintLoc":"null","initialTurnNumber":4,"moveLocs":["D3","D4","C2","B3","D2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..XO./...../...../","hintLoc":"null","initialTurnNumber":5,"moveLocs":["D4","C2","B3","D2","B2"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.OXO./..OX./...../...../","hintLoc":"null","initialTurnNumber":6,"moveLocs":["C2","B3","D2","B2","pass"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
)%%";
    expect(name,out,expected);
  }

  //============================================================================
  {
    const char* name = "Sgf parsing with black white double move, and flip";
    string sgfStr = "(;GM[1]FF[4]SZ[5]KM[24];B[cb];W[cc];W[bb];B[dc];W[db];B[cd];W[bc];B[dd];W[bd];B[])";
    {
      std::unique_ptr<Sgf> sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      vector<Sgf::PositionSample> samples;
      sgf->loadAllUniquePositions(uniqueHashes, false, false, true, false, NULL, samples);
      for(int i = 0; i<samples.size(); i++) {
        out << Sgf::PositionSample::toJsonLine(samples[i]) << endl;
        Sgf::PositionSample reloaded = Sgf::PositionSample::ofJsonLine(Sgf::PositionSample::toJsonLine(samples[i]));
        bool checkNumCaptures = false;
        bool checkSimpleKo = false;
        testAssert(samples[i].isEqualForTesting(reloaded,checkNumCaptures,checkSimpleKo));
      }
    }

    string expected = R"%%(
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":[],"movePlas":[],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../...../...../...../...../","hintLoc":"null","initialTurnNumber":0,"moveLocs":["C4","C3"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4"],"movePlas":["B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3"],"movePlas":["B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4"],"movePlas":["B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4","C2"],"movePlas":["B","W","B","W"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../..O../..X../...../...../","hintLoc":"null","initialTurnNumber":2,"moveLocs":["B4","D3","D4","C2","B3"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..X../...../...../","hintLoc":"null","initialTurnNumber":3,"moveLocs":["D3","D4","C2","B3","D2"],"movePlas":["W","B","W","B","W"],"nextPla":"W","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.XO../..XO./...../...../","hintLoc":"null","initialTurnNumber":4,"moveLocs":["D4","C2","B3","D2","B2"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
{"board":"...../.OXO./..OX./...../...../","hintLoc":"null","initialTurnNumber":5,"moveLocs":["C2","B3","D2","B2","pass"],"movePlas":["B","W","B","W","B"],"nextPla":"B","weight":1.0,"xSize":5,"ySize":5}
)%%";
    expect(name,out,expected);
  }

}


// Some tests that depend on files on disk in the repo.
void Tests::runSgfFileTests() {
  std::unique_ptr<Sgf> sgf = Sgf::loadFile("tests/data/foxlike.sgf");
  testAssert(sgf->getXYSize().x == 19);
  testAssert(sgf->getXYSize().y == 19);
  testAssert(sgf->getKomiOrFail() == 6.5f);
  testAssert(sgf->hasRules() == true);
  testAssert(sgf->getRulesOrFail().equalsIgnoringKomi(Rules::parseRules("chinese")));
  testAssert(sgf->getHandicapValue() == 2);
  testAssert(sgf->getSgfWinner() == C_EMPTY);
  testAssert(sgf->getPlayerName(P_BLACK) == "testname1");
  testAssert(sgf->getPlayerName(P_WHITE) == "testname2");
  testAssert(sgf->getRank(P_BLACK) == 2);
  testAssert(sgf->getRank(P_WHITE) == 4);
  cout << "SgfFileTests ok" << endl;
}

