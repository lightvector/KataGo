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
    board.setMultiStoneSuicideLegal(rules.multiStoneSuicideLegal);
    BoardHistory hist(board,P_BLACK,rules);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,1,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,2,board.x_size), P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,2,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,1,board.x_size), P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,3,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,3,board.x_size), P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,0,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,0,board.x_size), P_WHITE, rules, NULL);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, rules, NULL);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, rules, NULL);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    //Resurrecting the board after game over with another pass
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, rules, NULL);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    //And then some real moves followed by more passes
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,2,board.x_size), P_WHITE, rules, NULL);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, rules, NULL);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, rules, NULL);
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
    board.setMultiStoneSuicideLegal(rules.multiStoneSuicideLegal);
    BoardHistory hist(board,P_BLACK,rules);

    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,1,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,2,board.x_size), P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,2,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,1,board.x_size), P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,3,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,3,board.x_size), P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,0,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,0,board.x_size), P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,2,board.x_size), P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, rules, NULL);
    testAssert(hist.encorePhase == 0);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, rules, NULL);
    testAssert(hist.encorePhase == 1);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, rules, NULL);
    testAssert(hist.encorePhase == 1);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, rules, NULL);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, rules, NULL);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == false);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, rules, NULL);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 3.5f);
    out << board << endl;

    //Resurrecting the board after pass to have black throw in a dead stone, since second encore, should make no difference
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,1,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, rules, NULL);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 3.5f);
    out << board << endl;

    //Resurrecting again to have black solidfy his group and prove it pass-alive
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,0,board.x_size), P_WHITE, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(0,1,board.x_size), P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, rules, NULL);
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, rules, NULL);
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
}
