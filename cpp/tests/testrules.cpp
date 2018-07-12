#include "../tests/tests.h"
using namespace TestCommon;

static void makeMoveAssertLegal(BoardHistory& hist, Board& board, Loc loc, Player pla, int line) {
  if(!hist.isLegal(board, loc, pla)) 
    throw StringError("Illegal move on line " + Global::intToString(line));
  hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
}

static double finalScoreIfGameEndedNow(const BoardHistory& baseHist, const Board& baseBoard) {
  Player pla = P_BLACK;
  Board board(baseBoard);
  BoardHistory hist(baseHist);
  if(hist.moveHistory.size() > 0)
    pla = getOpp(hist.moveHistory[hist.moveHistory.size()-1].pla);
  while(!hist.isGameOver()) {
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, pla, NULL);
    pla = getOpp(pla);
  }
  return hist.finalWhiteMinusBlackScore;
}

void Tests::runRulesTests() {
  cout << "Running rules tests" << endl;
  ostringstream out;

  //Some helpers
  auto printIllegalMoves = [](ostream& o, const Board& board, const BoardHistory& hist, Player pla) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] == C_EMPTY && !board.isIllegalSuicide(loc,pla,hist.rules.multiStoneSuicideLegal) && !hist.isLegal(board,loc,pla)) {
          o << "Illegal: " << Location::toString(loc,board.x_size) << " " << getCharOfColor(pla) << endl;
        }
        if((pla == P_BLACK && hist.blackKoProhibited[loc]) || (pla == P_WHITE && hist.whiteKoProhibited[loc])) {
          o << "Ko-prohibited: " << Location::toString(loc,board.x_size) << " " << getCharOfColor(pla) << endl;
        }
      }
    }
  };

  auto printGameResult = [](ostream& o, const BoardHistory& hist) {
    if(!hist.isGameOver())
      o << "Game is not over";
    else {
      o << "Winner: " << getCharOfColor(hist.winner) << endl;
      o << "W-B Score: " << hist.finalWhiteMinusBlackScore << endl;
      o << "isNoResult: " << hist.isNoResult << endl;
    }
  };
    
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

    makeMoveAssertLegal(hist, board, Location::getLoc(1,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,2,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,1,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,3,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    //Resurrecting the board after game over with another pass
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    //And then some real moves followed by more passes
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_WHITE, __LINE__);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
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

    makeMoveAssertLegal(hist, board, Location::getLoc(1,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,2,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,1,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,3,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.encorePhase == 0);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.encorePhase == 1);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.encorePhase == 1);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 3.5f);
    out << board << endl;

    //Resurrecting the board after pass to have black throw in a dead stone, since second encore, should make no difference
    makeMoveAssertLegal(hist, board, Location::getLoc(3,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameOver() == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 3.5f);
    out << board << endl;

    //Resurrecting again to have black solidfy his group and prove it pass-alive
    makeMoveAssertLegal(hist, board, Location::getLoc(3,0,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(0,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
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


  //Basic ko rule testing with a regular ko and a sending two returning 1
  {
    Board baseBoard = parseBoard(6,5,R"%%(
.o.xxo
oxxxo.
o.x.oo
xxxoo.
oooo.o
)%%");

    Rules baseRules;
    baseRules.koRule = Rules::KO_POSITIONAL;
    baseRules.scoringRule = Rules::SCORING_TERRITORY;
    baseRules.komi = 0.5f;
    baseRules.multiStoneSuicideLegal = false;

    {
      const char* name = "Simple ko rules";
      Board board(baseBoard);
      Rules rules(baseRules);
      rules.koRule = Rules::KO_SIMPLE;
      BoardHistory hist(board,P_BLACK,rules);

      makeMoveAssertLegal(hist, board, Location::getLoc(5,1,board.x_size), P_BLACK, __LINE__);
      out << "After black ko capture:" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      out << "After black ko capture and one pass:" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      testAssert(hist.encorePhase == 0);
      testAssert(hist.isGameOver() == false);
      out << "After black ko capture and two passes:" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
      out << "White recapture:" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);

      out << "Beginning sending two returning one cycle" << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
      printIllegalMoves(out,board,hist,P_BLACK);
      makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_BLACK, __LINE__);
      printIllegalMoves(out,board,hist,P_WHITE);
      makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
      printIllegalMoves(out,board,hist,P_BLACK);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      printIllegalMoves(out,board,hist,P_WHITE);
      makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
      printIllegalMoves(out,board,hist,P_BLACK);
      makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_BLACK, __LINE__);
      printIllegalMoves(out,board,hist,P_WHITE);
      makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
      printIllegalMoves(out,board,hist,P_BLACK);
      testAssert(hist.encorePhase == 0);
      testAssert(hist.isGameOver() == false);
      //Spight ending condition cuts this cycle a bit shorter
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      printIllegalMoves(out,board,hist,P_WHITE);
      testAssert(hist.encorePhase == 1);
      testAssert(hist.isGameOver() == false);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      testAssert(hist.encorePhase == 2);
      printGameResult(out,hist);
      
      string expected = R"%%(
After black ko capture:
Illegal: (5,0) O
After black ko capture and one pass:
After black ko capture and two passes:
White recapture:
Illegal: (5,1) X
Beginning sending two returning one cycle
Winner: O
W-B Score: 0.5
isNoResult: 0
)%%";
      expect(name,out,expected);
      out.str("");
      out.clear();
    }

    {
      const char* name = "Positional ko rules";
      Board board(baseBoard);
      Rules rules(baseRules);
      rules.koRule = Rules::KO_POSITIONAL;
      BoardHistory hist(board,P_BLACK,rules);

      makeMoveAssertLegal(hist, board, Location::getLoc(5,1,board.x_size), P_BLACK, __LINE__);
      out << "After black ko capture:" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      out << "After black ko capture and one pass:" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      //On tmp board and hist, verify that the main phase ends if black passes now
      Board tmpboard(board);
      BoardHistory tmphist(hist);
      makeMoveAssertLegal(tmphist, tmpboard, Board::PASS_LOC, P_BLACK, __LINE__);
      testAssert(tmphist.encorePhase == 1);
      testAssert(tmphist.isGameOver() == false);

      makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);
      out << "Beginning sending two returning one cycle" << endl;

      makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
      out << "After white sends two?" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_BLACK, __LINE__);
      out << "Can white recapture?" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
      out << "After white recaptures the other ko instead" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "After white recaptures the other ko instead and black passes" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
      out << "After white now returns 1" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "After white now returns 1 and black passes" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
      out << "After white sends 2 again" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);
      testAssert(hist.encorePhase == 0);
      testAssert(hist.isGameOver() == false);

      string expected = R"%%(
After black ko capture:
Illegal: (5,0) O
After black ko capture and one pass:
Beginning sending two returning one cycle
After white sends two?
Can white recapture?
Illegal: (1,0) O
After white recaptures the other ko instead
Illegal: (5,1) X
After white recaptures the other ko instead and black passes
After white now returns 1
Illegal: (5,1) X
After white now returns 1 and black passes
After white sends 2 again
Illegal: (0,0) X
Illegal: (5,1) X
)%%";
      expect(name,out,expected);
      out.str("");
      out.clear();
    }

    {
      const char* name = "Situational ko rules";
      Board board(baseBoard);
      Rules rules(baseRules);
      rules.koRule = Rules::KO_SITUATIONAL;
      BoardHistory hist(board,P_BLACK,rules);

      makeMoveAssertLegal(hist, board, Location::getLoc(5,1,board.x_size), P_BLACK, __LINE__);
      out << "After black ko capture:" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      out << "After black ko capture and one pass:" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      //On tmp board and hist, verify that the main phase ends if black passes now
      Board tmpboard(board);
      BoardHistory tmphist(hist);
      makeMoveAssertLegal(tmphist, tmpboard, Board::PASS_LOC, P_BLACK, __LINE__);
      testAssert(tmphist.encorePhase == 1);
      testAssert(tmphist.isGameOver() == false);

      makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);
      out << "Beginning sending two returning one cycle" << endl;

      makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
      out << "After white sends two?" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_BLACK, __LINE__);
      out << "Can white recapture?" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
      out << "After white recaptures the other ko instead" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "After white recaptures the other ko instead and black passes" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
      out << "After white now returns 1" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "After white now returns 1 and black passes" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
      out << "After white sends 2 again" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);
      testAssert(hist.encorePhase == 0);
      testAssert(hist.isGameOver() == false);

      string expected = R"%%(
After black ko capture:
Illegal: (5,0) O
After black ko capture and one pass:
Beginning sending two returning one cycle
After white sends two?
Can white recapture?
After white recaptures the other ko instead
Illegal: (5,1) X
After white recaptures the other ko instead and black passes
After white now returns 1
Illegal: (5,1) X
After white now returns 1 and black passes
After white sends 2 again
Illegal: (0,0) X
)%%";
      expect(name,out,expected);
      out.str("");
      out.clear();
    }

    {
      const char* name = "Spight ko rules";
      Board board(baseBoard);
      Rules rules(baseRules);
      rules.koRule = Rules::KO_SPIGHT;
      BoardHistory hist(board,P_BLACK,rules);

      makeMoveAssertLegal(hist, board, Location::getLoc(5,1,board.x_size), P_BLACK, __LINE__);
      out << "After black ko capture:" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      out << "After black ko capture and one pass:" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      //On tmp board and hist, verify that the main phase does not end if black passes now
      Board tmpboard(board);
      BoardHistory tmphist(hist);
      makeMoveAssertLegal(tmphist, tmpboard, Board::PASS_LOC, P_BLACK, __LINE__);
      testAssert(tmphist.encorePhase == 0);
      testAssert(tmphist.isGameOver() == false);
      out << "If black were to pass as well??" << endl;
      printIllegalMoves(out,tmpboard,tmphist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);
      out << "Beginning sending two returning one cycle" << endl;

      makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
      out << "After white sends two?" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_BLACK, __LINE__);
      out << "Can white recapture?" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
      out << "After white recaptures the other ko instead" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "After white recaptures the other ko instead and black passes" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
      out << "After white now returns 1" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "After white now returns 1 and black passes" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_WHITE, __LINE__);
      out << "After white sends 2 again" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_BLACK, __LINE__);
      out << "Can white recapture?" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      out << "After pass" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);
      testAssert(hist.encorePhase == 0);
      testAssert(hist.isGameOver() == false);

      //This is actually black's second pass in this position!
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "After pass" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);
      testAssert(hist.encorePhase == 1);
      testAssert(hist.isGameOver() == false);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      testAssert(hist.encorePhase == 2);
      printGameResult(out,hist);

      string expected = R"%%(
After black ko capture:
Illegal: (5,0) O
After black ko capture and one pass:
If black were to pass as well??
Beginning sending two returning one cycle
After white sends two?
Can white recapture?
Illegal: (1,0) O
After white recaptures the other ko instead
Illegal: (5,1) X
After white recaptures the other ko instead and black passes
After white now returns 1
After white now returns 1 and black passes
After white sends 2 again
Can white recapture?
Illegal: (1,0) O
After pass
After pass
Winner: X
W-B Score: -0.5
isNoResult: 0
)%%";
      expect(name,out,expected);
      out.str("");
      out.clear();
    }
  }
  
  //Testing superko with suicide
  {
    Board baseBoard = parseBoard(6,5,R"%%(
.oxo.x
oxxooo
xx....
......
......
)%%");

    Rules baseRules;
    baseRules.koRule = Rules::KO_POSITIONAL;
    baseRules.scoringRule = Rules::SCORING_AREA;
    baseRules.komi = 0.5f;
    baseRules.multiStoneSuicideLegal = true;

    int koRulesToTest[3] = { Rules::KO_POSITIONAL, Rules::KO_SITUATIONAL, Rules::KO_SPIGHT };
    const char* name = "Suicide ko testing";
    for(int i = 0; i<3; i++)
    {
      out << "------------------------------" << endl;
      Board board(baseBoard);
      Rules rules(baseRules);
      rules.koRule = koRulesToTest[i];
      BoardHistory hist(board,P_BLACK,rules);

      makeMoveAssertLegal(hist, board, Location::getLoc(4,0,board.x_size), P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      out << "After black suicide and white pass" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      makeMoveAssertLegal(hist, board, Location::getLoc(4,0,board.x_size), P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "After a little looping" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Location::getLoc(4,0,board.x_size), P_BLACK, __LINE__);
      out << "Filling in a bit more" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);
      
      //Illegal under non-spight superkos, but still should be handled gracefully
      hist.makeBoardMoveAssumeLegal(board, Location::getLoc(0,1,board.x_size), P_WHITE, NULL);
      hist.makeBoardMoveAssumeLegal(board, Location::getLoc(5,0,board.x_size), P_BLACK, NULL);
      hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,0,board.x_size), P_WHITE, NULL);
      hist.makeBoardMoveAssumeLegal(board, Location::getLoc(4,0,board.x_size), P_BLACK, NULL);
      out << "Looped some more" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);
      out << board << endl;
      
    }
    string expected = R"%%(
------------------------------
After black suicide and white pass
Illegal: (5,0) X
After a little looping
Illegal: (0,1) O
Filling in a bit more
Illegal: (0,1) O
Looped some more
Illegal: (0,0) O
Illegal: (0,1) O
HASH: D369FBF88E276F2F6D21FF1A9FA349F4
. O X O X .
. X X O O O
X X . . . .
. . . . . .
. . . . . .


------------------------------
After black suicide and white pass
After a little looping
Illegal: (0,1) O
Filling in a bit more
Illegal: (0,1) O
Looped some more
HASH: D369FBF88E276F2F6D21FF1A9FA349F4
. O X O X .
. X X O O O
X X . . . .
. . . . . .
. . . . . .


------------------------------
After black suicide and white pass
After a little looping
Filling in a bit more
Looped some more
Illegal: (0,0) O
HASH: D369FBF88E276F2F6D21FF1A9FA349F4
. O X O X .
. X X O O O
X X . . . .
. . . . . .
. . . . . .
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Eternal life";
    Board board = parseBoard(8,5,R"%%(
........
oooooo..
xxxxxo..
xoooxxoo
.o.x.ox.
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_BLACK,rules);

    makeMoveAssertLegal(hist, board, Location::getLoc(2,4,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,4,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,4,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,4,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,4,board.x_size), P_BLACK, __LINE__);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,4,board.x_size), P_WHITE, __LINE__);
    testAssert(hist.isGameOver() == true);
    printGameResult(out,hist);
    
    string expected = R"%%(
Winner: .
W-B Score: 0
isNoResult: 1
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Triple ko simple";
    Board board = parseBoard(7,6,R"%%(
ooooooo
oxo.o.o
x.xoxox
xxxxxxx
ooooooo
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_BLACK,rules);

    makeMoveAssertLegal(hist, board, Location::getLoc(3,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,1,board.x_size), P_BLACK, __LINE__);
    testAssert(hist.isGameOver() == false);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,2,board.x_size), P_WHITE, __LINE__);
    testAssert(hist.isGameOver() == true);
    printGameResult(out,hist);
    
    string expected = R"%%(
Winner: .
W-B Score: 0
isNoResult: 1
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }
  
  {
    const char* name = "Triple ko superko";
    Board board = parseBoard(7,6,R"%%(
ooooooo
oxo.o.o
x.xoxox
xxxxxxx
ooooooo
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_BLACK,rules);

    makeMoveAssertLegal(hist, board, Location::getLoc(3,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,1,board.x_size), P_BLACK, __LINE__);
    printIllegalMoves(out,board,hist,P_WHITE);
    string expected = R"%%(
Illegal: (1,2) O
Illegal: (5,2) O
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Territory scoring in the main phase";
    Board board = parseBoard(7,7,R"%%(
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_WHITE,rules);

    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(5,3,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,3,board.x_size), P_WHITE, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,4,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(5,4,board.x_size), P_WHITE, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(0,3,board.x_size), P_WHITE, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,6,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    string expected = R"%%(
Score: 0.5
Score: 0.5
Score: 0.5
Score: -4.5
Score: -5.5
Score: -4.5
Score: -3.5
Score: -2.5
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }
  {
    const char* name = "Territory scoring in encore 1";
    Board board = parseBoard(7,7,R"%%(
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_WHITE,rules);

    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(5,3,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,3,board.x_size), P_WHITE, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,4,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,4,board.x_size), P_WHITE, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(0,3,board.x_size), P_WHITE, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,6,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    string expected = R"%%(
Score: 0.5
Score: 0.5
Score: 0.5
Score: -4.5
Score: -5.5
Score: -4.5
Score: -3.5
Score: -2.5
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }
  {
    const char* name = "Territory scoring in encore 2";
    Board board = parseBoard(7,7,R"%%(
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_WHITE,rules);

    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(5,3,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,3,board.x_size), P_WHITE, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,4,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,4,board.x_size), P_WHITE, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(0,3,board.x_size), P_WHITE, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,6,board.x_size), P_BLACK, __LINE__);
    out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
    string expected = R"%%(
Score: 0.5
Score: 0.5
Score: 0.5
Score: -4.5
Score: -4.5
Score: -4.5
Score: -3.5
Score: -3.5
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }
  
  {
    const char* name = "Pass for ko";
    Board board = parseBoard(7,7,R"%%(
..ox.oo
..oxxxo
...oox.
....oxx
..o.oo.
.......
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_BLACK,rules);
    Hash128 hasha;
    Hash128 hashb;
    Hash128 hashc;
    Hash128 hashd;
    Hash128 hashe;
    Hash128 hashf;
    
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.encorePhase == 1);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(6,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,0,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(6,1,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
    out << "Black can't retake" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,2,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,2,board.x_size), P_WHITE, __LINE__);
    out << "Ko threat shouldn't work in the encore" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(0,6,board.x_size), P_WHITE, __LINE__);
    out << "Regular pass shouldn't work in the encore" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);
    out << "Pass for ko! (Should not affect the board stones)" << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(0,5,board.x_size), P_WHITE, __LINE__);
    hashd = hist.koHashHistory[hist.koHashHistory.size()-1];
    out << "Now black can retake, and white's retake isn't legal" << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
    printIllegalMoves(out,board,hist,P_WHITE);
    hasha = hist.koHashHistory[hist.koHashHistory.size()-1];
    makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
    hashb = hist.koHashHistory[hist.koHashHistory.size()-1];
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    hashc = hist.koHashHistory[hist.koHashHistory.size()-1];
    testAssert(hasha != hashb);
    testAssert(hasha != hashc);
    testAssert(hashb != hashc);    
    out << "White's retake is legal after passing for ko" << endl;
    printIllegalMoves(out,board,hist,P_WHITE);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
    out << "Black's retake is illegal again" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);
    makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    assert(hashd == hist.koHashHistory[hist.koHashHistory.size()-1]);
    out << "And is still illegal due to only-once" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,3,board.x_size), P_WHITE, __LINE__);
    out << "But a ko threat fixes that" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);
    makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
    out << "White illegal now" << endl;    
    printIllegalMoves(out,board,hist,P_WHITE);
    testAssert(hist.encorePhase == 1);
    hasha = hist.koHashHistory[hist.koHashHistory.size()-1];
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    hashc = hist.koHashHistory[hist.koHashHistory.size()-1];
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    hashc = hist.koHashHistory[hist.koHashHistory.size()-1];
    testAssert(hist.encorePhase == 2);
    testAssert(hasha != hashb);
    testAssert(hasha != hashc);
    testAssert(hashb != hashc);    
    out << "Legal again in second encore" << endl;
    printIllegalMoves(out,board,hist,P_WHITE);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
    out << "Lastly, try black ko threat one more time" << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,2,board.x_size), P_WHITE, __LINE__);
    printIllegalMoves(out,board,hist,P_BLACK);
    out << "And a pass for ko" << endl;
    hashd = hist.koHashHistory[hist.koHashHistory.size()-1];
    makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
    hashe = hist.koHashHistory[hist.koHashHistory.size()-1];
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    hashf = hist.koHashHistory[hist.koHashHistory.size()-1];
    printIllegalMoves(out,board,hist,P_BLACK);
    out << "And repeat with white" << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,0,board.x_size), P_WHITE, __LINE__);
    assert(hashd == hist.koHashHistory[hist.koHashHistory.size()-1]);
    makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
    assert(hashe == hist.koHashHistory[hist.koHashHistory.size()-1]);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    assert(hashf == hist.koHashHistory[hist.koHashHistory.size()-1]);
    out << "And see the only-once for black" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);

    string expected = R"%%(
Black can't retake
Ko-prohibited: (6,0) X
Ko threat shouldn't work in the encore
Ko-prohibited: (6,0) X
Regular pass shouldn't work in the encore
Ko-prohibited: (6,0) X
Pass for ko! (Should not affect the board stones)
HASH: F1139C17E04FC2DF5ABA49348EF744D1
. . O X X O .
. . O X X X O
. O X O O X .
. . . . O X X
. . O . O O .
. . . . . . .
O . . . . . .


Now black can retake, and white's retake isn't legal
Ko-prohibited: (5,0) O
White's retake is legal after passing for ko
Black's retake is illegal again
Ko-prohibited: (6,0) X
And is still illegal due to only-once
Illegal: (6,0) X
But a ko threat fixes that
White illegal now
Ko-prohibited: (5,0) O
Legal again in second encore
Lastly, try black ko threat one more time
Ko-prohibited: (6,0) X
And a pass for ko
And repeat with white
And see the only-once for black
Illegal: (6,0) X
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Two step ko mark clearing";
    Board board = parseBoard(7,5,R"%%(
x.x....
.xx....
xox....
ooo....
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = true;
    BoardHistory hist(board,P_WHITE,rules);
    
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.encorePhase == 1);
    makeMoveAssertLegal(hist, board, Location::getLoc(0,1,board.x_size), P_WHITE, __LINE__);
    out << "After first cap" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
    out << "After second cap" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);
    makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_BLACK, __LINE__);
    out << "Just after black pass for ko" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);
    out << board << endl;

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_BLACK, __LINE__);
    out <<"After first cap" << endl;
    printIllegalMoves(out,board,hist,P_WHITE);
    out << board << endl;
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(0,2,board.x_size), P_BLACK, __LINE__);
    out << "After second cap" << endl;
    printIllegalMoves(out,board,hist,P_WHITE);
    out << board << endl;
    makeMoveAssertLegal(hist, board, Location::getLoc(0,1,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    out << "After pass for ko" << endl;
    printIllegalMoves(out,board,hist,P_WHITE);
    out << board << endl;

    string expected = R"%%(
After first cap
Ko-prohibited: (0,2) X
After second cap
Ko-prohibited: (0,0) X
Just after black pass for ko
Illegal: (0,0) X
HASH: C31D698C43AF8F4719AF350A52362800
. O X . . . .
O X X . . . .
. O X . . . .
O O O . . . .
. . . . . . .


After first cap
Ko-prohibited: (1,0) O
HASH: D9AD14AFEF3A6FB208691C6EABB3ACC0
X . X . . . .
O X X . . . .
. O X . . . .
O O O . . . .
. . . . . . .


After second cap
Ko-prohibited: (0,1) O
HASH: 06E53C9488F4A69B52F544FC1C6E1D40
X . X . . . .
. X X . . . .
X O X . . . .
O O O . . . .
. . . . . . .


After pass for ko
Illegal: (0,1) O
HASH: 06E53C9488F4A69B52F544FC1C6E1D40
X . X . . . .
. X X . . . .
X O X . . . .
O O O . . . .
. . . . . . .
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Throw in that destroys the ko momentarily does not clear ko prohibition";
    Board board = parseBoard(7,5,R"%%(
x......
oxx....
.o.....
oo.....
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = true;
    BoardHistory hist(board,P_BLACK,rules);
    
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.encorePhase == 2);
    makeMoveAssertLegal(hist, board, Location::getLoc(0,2,board.x_size), P_BLACK, __LINE__);
    printIllegalMoves(out,board,hist,P_WHITE);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,0,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printIllegalMoves(out,board,hist,P_WHITE);

    string expected = R"%%(
Ko-prohibited: (0,1) O
HASH: 7549DEF1D74769D79E9729028FF1A1D5
X . X . . . .
. X X . . . .
X O . . . . .
O O . . . . .
. . . . . . .


Ko-prohibited: (0,1) O
)%%";
    expect(name,out,expected);
    out.str("");
    out.clear();
  }

}
