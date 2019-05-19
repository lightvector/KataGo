#include "../tests/tests.h"
#include "../search/asyncbot.h"
#include "../dataio/sgf.h"
#include <algorithm>
#include <iterator>
using namespace TestCommon;

void Tests::runSgfTests() {
//   cout << "Running sgf tests" << endl;
//   ostringstream out;

//   auto parseAndPrintSgf = [&out](const string& sgfStr) {
//     CompactSgf* sgf = CompactSgf::parse(sgfStr);

//     out << "xSize " << sgf->xSize << endl;
//     out << "ySize " << sgf->ySize << endl;
//     out << "depth " << sgf->depth << endl;
//     out << "komi " << sgf->komi << endl;

//     Board board;
//     BoardHistory hist;
//     Rules rules;
//     Player pla;
//     sgf->setupInitialBoardAndHist(rules,board,
    
//     out << "placements" << endl;
//     for(int i = 0; i < sgf->placements.size(); i++) {
//       Move move = sgf->placements[i];
//       out << colorToChar(move.pla) << " " << Location::toString(move.loc,) << endl;
//     }
//     out << "moves" << endl;
//     for(int i = 0; i < sgf->moves.size(); i++) {
//       Move move = sgf->moves[i];
//       out << colorToChar(move.pla) << " " << Location::toString(move.loc) << endl;
//     }

    
//     void setupInitialBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist);
//     void setupBoardAndHist(const Rules& initialRules, Board& board, Player& nextPla, BoardHistory& hist, int turnNumber);

//   };
  
//   //============================================================================
//   {
//     const char* name = "Basic Sgf parse test";
//     string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[19]KM[5.00]PW[White]PB[Black]AB[dd][pd][dp][pp]PL[W];W[qf];W[md];B[pf];W[pg];B[of];W[];B[tt])";
//     parseAndPrintSgf(sgfStr);
//     string expected = R"%%(

// )%%";
//     expect(name,out,expected);
//   }

  
}
