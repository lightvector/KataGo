#include "../tests/tests.h"
using namespace TestCommon;

void Tests::runRulesTests() {
  cout << "Running rules tests" << endl;
  ostringstream out;

  {
    const char* name = "Basic area rules";
    Board board = parseBoard(4,4,R"%%(
....
....
....
....
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = true;
    BoardHistory hist(board,P_BLACK,rules);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,1,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,2,board.x_size), P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,2,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,1,board.x_size), P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,3,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,3,board.x_size), P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,0,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,0,board.x_size), P_WHITE, NULL);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    //Resurrecting the board after game over with another pass
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    //And then some real moves followed by more passes
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,2,board.x_size), P_WHITE, NULL);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    out << board << endl;
    string expected = R"%%(
HASH: 551911C639136FD87CFD8C126ABC2737
. X O .
. X O .
. X O O
. X O .
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Basic territory rules";
    Board board = parseBoard(4,4,R"%%(
....
....
....
....
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = true;
    BoardHistory hist(board,P_BLACK,rules);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,1,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,2,board.x_size), P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,2,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,1,board.x_size), P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,3,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,3,board.x_size), P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,0,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,0,board.x_size), P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,2,board.x_size), P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
    testAssert(hist.encorePhase == 0);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    testAssert(hist.encorePhase == 1);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
    testAssert(hist.encorePhase == 1);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 3.5f);
    out << board << endl;

    //Resurrecting the board after pass to have black throw in a dead stone, since second encore, should make no difference
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,1,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 3.5f);
    out << board << endl;

    //Resurrecting again to have black solidfy his group and prove it pass-alive
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,0,board.x_size), P_WHITE, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(0,1,board.x_size), P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
    //White claimed 3 points pre-second-encore, while black waited until second encore, so black gets 4 points and wins by 0.5.
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_BLACK);
    testAssert(hist.finalWhiteMinusBlackScore == -0.5f);
    out << board << endl;

    string expected = R"%%(
HASH: 551911C639136FD87CFD8C126ABC2737
. X O .
. X O .
. X O O
. X O .


HASH: 4234472D4CF6700889EE541B518C2FF9
. X O .
. X O X
. X O O
. X O .


HASH: FAE9F3EAAF790C5CF1EC62AFDD264F77
. X O O
X X O .
. X O O
. X O .
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

//   {
//     const char* name = "Basic ko rules";
//     Board baseBoard = parseBoard(6,5,R"%%(
// .o.xxo
// oxxxo.
// o.x.oo
// xxxoo.
// oooo.o
// )%%");

//     Rules baseRules;
//     baseRules.koRule = Rules::KO_POSITIONAL;
//     baseRules.scoringRule = Rules::SCORING_TERRITORY;
//     baseRules.komi = 0.5f;
//     baseRules.multiStoneSuicideLegal = false;

//     auto printIllegalMoves = [](ostream& o, const Board& board, const BoardHistory& hist, Player pla) {
//       for(int y = 0; y<board.y_size; y++) {
//         for(int x = 0; y<board.x_size; x++) {
//           Loc loc = Location::getLoc(x,y,board.x_size);
//           if(board.colors[loc] == C_EMPTY && !board.isIllegalSuicide(loc,pla,hist.rules.multiStoneSuicideLegal) && !hist.isLegal(board,loc,pla)) {
//             o << "Illegal: " << Location::toString(loc,board.x_size) << " " << getCharOfColor(pla) << endl;
//           }
//         }
//       }
//     };

//     {
//       Board board(baseBoard);
//       Rules rules(baseRules);
//       rules.koRule = Rules::KO_SIMPLE;
//       BoardHistory hist(board,P_BLACK,rules);

//       hist.makeBoardMoveAssumeLegal(board, Location::getLoc(5,1,board.x_size), P_BLACK, NULL);
//       out << "After black ko capture:" << endl;
//       printIllegalMoves(out,board,hist,P_WHITE);

//       hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
//       out << "After black ko capture and one pass:" << endl;
//       printIllegalMoves(out,board,hist,P_BLACK);

//       hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
//       testAssert(hist.encorePhase == 0);
//       testAssert(hist.isGameOver() == false);
//       out << "After black ko capture and two passes:" << endl;
//       printIllegalMoves(out,board,hist,P_WHITE);

//       hist.makeBoardMoveAssumeLegal(board, Location::getLoc(5,0,board.x_size), P_WHITE, NULL);
//       out << "White recapture:" << endl;
//       printIllegalMoves(out,board,hist,P_BLACK);

//       hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,2,board.x_size), P_BLACK, NULL);

//       out << "Beginning sending two returning one cycle" << endl;
//       hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,0,board.x_size), P_WHITE, NULL);
//       printIllegalMoves(out,board,hist,P_BLACK);
//       hist.makeBoardMoveAssumeLegal(board, Location::getLoc(0,0,board.x_size), P_BLACK, NULL);
//       printIllegalMoves(out,board,hist,P_WHITE);
//       hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,1,board.x_size), P_WHITE, NULL);
//       printIllegalMoves(out,board,hist,P_BLACK);
//       hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
//       printIllegalMoves(out,board,hist,P_WHITE);
//       hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,0,board.x_size), P_WHITE, NULL);
//       printIllegalMoves(out,board,hist,P_BLACK);
//       hist.makeBoardMoveAssumeLegal(board, Location::getLoc(0,0,board.x_size), P_BLACK, NULL);
//       printIllegalMoves(out,board,hist,P_WHITE);
//       hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,1,board.x_size), P_WHITE, NULL);
//       printIllegalMoves(out,board,hist,P_BLACK);
//       hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
//       printIllegalMoves(out,board,hist,P_WHITE);
//       testAssert(hist.encorePhase == 0);
//       testAssert(hist.isGameOver() == false);

//       string expected = R"%%(
// HASH: 551911C639136FD87CFD8C126ABC2737
// . X O .
// . X O .
// . X O O
// . X O .


// HASH: 4234472D4CF6700889EE541B518C2FF9
// . X O .
// . X O X
// . X O O
// . X O .


// HASH: FAE9F3EAAF790C5CF1EC62AFDD264F77
// . X O O
// X X O .
// . X O O
// . X O .
// )%%";
//       expect(name,out,expected);
//       out.str("");
//       out.clear();
//     }

//   }

}
