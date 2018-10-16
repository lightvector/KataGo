#include "../tests/tests.h"
using namespace TestCommon;

static void checkKoHashConsistency(BoardHistory& hist, Board& board, Player nextPla) {
  testAssert(hist.koHashHistory.size() > 0);
  Hash128 expected = board.pos_hash;
  if(hist.encorePhase > 0) {
    expected ^= Board::ZOBRIST_PLAYER_HASH[nextPla];
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(hist.blackKoProhibited[loc])
          expected ^= Board::ZOBRIST_KO_MARK_HASH[loc][P_BLACK];
        if(hist.whiteKoProhibited[loc])
          expected ^= Board::ZOBRIST_KO_MARK_HASH[loc][P_WHITE];
      }
    }
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL) {
    expected ^= Board::ZOBRIST_PLAYER_HASH[nextPla];
  }
  testAssert(expected == hist.koHashHistory[hist.koHashHistory.size()-1]);
}

static void makeMoveAssertLegal(BoardHistory& hist, Board& board, Loc loc, Player pla, int line) {
  bool phaseWouldEnd = hist.passWouldEndPhase(board,pla);
  int oldPhase = hist.encorePhase;
  
  if(!hist.isLegal(board, loc, pla))
    throw StringError("Illegal move on line " + Global::intToString(line));
  hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
  checkKoHashConsistency(hist,board,getOpp(pla));

  if(loc == Board::PASS_LOC) {
    int newPhase = hist.encorePhase;
    if(phaseWouldEnd != (newPhase != oldPhase || hist.isGameFinished))
      throw StringError("hist.passWouldEndPhase returned different answer than what actually happened after a pass");
  }
}

static double finalScoreIfGameEndedNow(const BoardHistory& baseHist, const Board& baseBoard) {
  Player pla = P_BLACK;
  Board board(baseBoard);
  BoardHistory hist(baseHist);
  if(hist.moveHistory.size() > 0)
    pla = getOpp(hist.moveHistory[hist.moveHistory.size()-1].pla);
  while(!hist.isGameFinished) {
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, pla, NULL);
    pla = getOpp(pla);
  }

  double score = hist.finalWhiteMinusBlackScore;

  hist.endAndScoreGameNow(board);
  assert(hist.finalWhiteMinusBlackScore == score);

  BoardHistory hist2(baseHist);
  hist2.endAndScoreGameNow(baseBoard);
  assert(hist2.finalWhiteMinusBlackScore == score);

  return score;
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
          o << "Illegal: " << Location::toStringMach(loc,board.x_size) << " " << colorToChar(pla) << endl;
        }
        if((pla == P_BLACK && hist.blackKoProhibited[loc]) || (pla == P_WHITE && hist.whiteKoProhibited[loc])) {
          o << "Ko-prohibited: " << Location::toStringMach(loc,board.x_size) << " " << colorToChar(pla) << endl;
        }
      }
    }
  };

  auto printEncoreKoProhibition = [](ostream& o, const Board& board, const BoardHistory& hist) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(hist.blackKoProhibited[loc])
          o << "Ko prohibited black at " << Location::toString(loc,board) << endl;
        if(hist.whiteKoProhibited[loc])
          o << "Ko prohibited white at " << Location::toString(loc,board) << endl;
      }
    }
  };

  auto printGameResult = [](ostream& o, const BoardHistory& hist) {
    if(!hist.isGameFinished)
      o << "Game is not over";
    else {
      o << "Winner: " << playerToString(hist.winner) << endl;
      o << "W-B Score: " << hist.finalWhiteMinusBlackScore << endl;
      o << "isNoResult: " << hist.isNoResult << endl;
    }
  };

  {
    const char* name = "Area rules";
    Board board = Board::parseBoard(4,4,R"%%(
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
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.isGameFinished == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    //Resurrecting the board after game over with another pass
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.isGameFinished == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    //And then some real moves followed by more passes
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_WHITE, __LINE__);
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.isGameFinished == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 0.5f);
    out << board << endl;
    string expected = R"%%(
HASH: 551911C639136FD87CFD8C126ABC2737
   A B C D
 4 . X O .
 3 . X O .
 2 . X O O
 1 . X O .
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Territory rules";
    Board board = Board::parseBoard(4,4,R"%%(
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
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.encorePhase == 1);
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.encorePhase == 1);
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameFinished == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 3.5f);
    out << board << endl;

    //Resurrecting the board after pass to have black throw in a dead stone, since second encore, should make no difference
    makeMoveAssertLegal(hist, board, Location::getLoc(3,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameFinished == true);
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
    testAssert(hist.isGameFinished == true);
    testAssert(hist.winner == P_BLACK);
    testAssert(hist.finalWhiteMinusBlackScore == -0.5f);
    out << board << endl;

    string expected = R"%%(
HASH: 551911C639136FD87CFD8C126ABC2737
   A B C D
 4 . X O .
 3 . X O .
 2 . X O O
 1 . X O .


HASH: 4234472D4CF6700889EE541B518C2FF9
   A B C D
 4 . X O .
 3 . X O X
 2 . X O O
 1 . X O .


HASH: FAE9F3EAAF790C5CF1EC62AFDD264F77
   A B C D
 4 . X O O
 3 X X O .
 2 . X O O
 1 . X O .
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }


  //Ko rule testing with a regular ko and a sending two returning 1
  {
    Board baseBoard = Board::parseBoard(6,5,R"%%(
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
      testAssert(hist.isGameFinished == false);
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
      testAssert(hist.isGameFinished == false);
      //Spight ending condition cuts this cycle a bit shorter
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      printIllegalMoves(out,board,hist,P_WHITE);
      testAssert(hist.encorePhase == 1);
      testAssert(hist.isGameFinished == false);

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
Winner: White
W-B Score: 0.5
isNoResult: 0
)%%";
      expect(name,out.str(),expected);
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
      testAssert(tmphist.isGameFinished == false);

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
      testAssert(hist.isGameFinished == false);

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
      expect(name,out.str(),expected);
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
      testAssert(tmphist.isGameFinished == false);

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
      testAssert(hist.isGameFinished == false);

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
      expect(name,out.str(),expected);
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
      testAssert(tmphist.isGameFinished == false);
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
      testAssert(hist.isGameFinished == false);

      //This is actually black's second pass in this position!
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "After pass" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);
      testAssert(hist.encorePhase == 1);
      testAssert(hist.isGameFinished == false);

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
Winner: Black
W-B Score: -0.5
isNoResult: 0
)%%";
      expect(name,out.str(),expected);
      out.str("");
      out.clear();
    }
  }

  //Testing superko with suicide
  {
    Board baseBoard = Board::parseBoard(6,5,R"%%(
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
   A B C D E F
 5 . O X O X .
 4 . X X O O O
 3 X X . . . .
 2 . . . . . .
 1 . . . . . .


------------------------------
After black suicide and white pass
After a little looping
Illegal: (0,1) O
Filling in a bit more
Illegal: (0,1) O
Looped some more
HASH: D369FBF88E276F2F6D21FF1A9FA349F4
   A B C D E F
 5 . O X O X .
 4 . X X O O O
 3 X X . . . .
 2 . . . . . .
 1 . . . . . .


------------------------------
After black suicide and white pass
After a little looping
Filling in a bit more
Looped some more
Illegal: (0,0) O
HASH: D369FBF88E276F2F6D21FF1A9FA349F4
   A B C D E F
 5 . O X O X .
 4 . X X O O O
 3 X X . . . .
 2 . . . . . .
 1 . . . . . .

)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Eternal life";
    Board board = Board::parseBoard(8,5,R"%%(
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
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,4,board.x_size), P_WHITE, __LINE__);
    testAssert(hist.isGameFinished == true);
    printGameResult(out,hist);

    string expected = R"%%(
Winner: Empty
W-B Score: 0
isNoResult: 1
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Triple ko simple";
    Board board = Board::parseBoard(7,6,R"%%(
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
    testAssert(hist.isGameFinished == false);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,2,board.x_size), P_WHITE, __LINE__);
    testAssert(hist.isGameFinished == true);
    printGameResult(out,hist);

    string expected = R"%%(
Winner: Empty
W-B Score: 0
isNoResult: 1
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Triple ko superko";
    Board board = Board::parseBoard(7,6,R"%%(
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
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Triple ko encore";
    Board board = Board::parseBoard(7,6,R"%%(
ooooooo
oxo.o.o
x.xoxox
xxxxxxx
ooooooo
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_BLACK,rules);

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,2,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,1,board.x_size), P_BLACK, __LINE__);
    //Pass for ko
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_WHITE, __LINE__);
    //Should be a complete capture
    makeMoveAssertLegal(hist, board, Location::getLoc(1,1,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    //There should be no ko marks on the board at this point.
    printEncoreKoProhibition(out,board,hist);

    string expected = R"%%(
HASH: EA1DB3D0A1A4D729AFE423A3B6425B29
   A B C D E F G
 6 . . . . . . .
 5 . X . X . X .
 4 X . X . X . X
 3 X X X X X X X
 2 O O O O O O O
 1 . . . . . . .
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Encore - own throwin that temporarily breaks the ko shape should not clear the ko prohibition";
    Board board = Board::parseBoard(7,6,R"%%(
..o....
...o...
.xoxo..
..x.x..
...x...
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_WHITE,rules);

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,3,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,1,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,1,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);

    string expected = R"%%(
HASH: 211F4559FB155DA94DC2C5CB753077E7
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko prohibited black at D4
HASH: EE414000D22F0F8E999B58DDA897C6BD
   A B C D E F G
 6 . . O . . . .
 5 . . X O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko prohibited black at D4
HASH: 26526703BA88855735804E2D4B6CE3C7
   A B C D E F G
 6 . . O . . . .
 5 . O . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko prohibited black at D4
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Encore - ko prohibition clears if opponent moves without restoring the ko shape";
    Board board = Board::parseBoard(7,6,R"%%(
..o....
...o...
.xoxo..
..x.x..
...x...
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_WHITE,rules);

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,3,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,1,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);

    string expected = R"%%(
HASH: 211F4559FB155DA94DC2C5CB753077E7
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko prohibited black at D4
HASH: EE414000D22F0F8E999B58DDA897C6BD
   A B C D E F G
 6 . . O . . . .
 5 . . X O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko prohibited black at D4
HASH: 1CD632B0C14F8233EEF065D2BF0BCA6C
   A B C D E F G
 6 O . O . . . .
 5 . . X O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


HASH: 1FD443E8C77DDC9908A5E4EC94AD28F3
   A B C D E F G
 6 O . O . . . .
 5 . . X O . . .
 4 . X . X O . .
 3 . . X . X . .
 2 . . . X . . .
 1 . . . . . . .

)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }


  {
    const char* name = "Encore - once only rule doesn't prevent the opponent moving there (filling ko)";
    Board board = Board::parseBoard(7,6,R"%%(
..o....
...o...
.xoxo..
..x.x..
...x...
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_WHITE,rules);

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,3,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    //Pass for ko
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    //Pass
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    //Take ko
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    //Pass
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);
    //Fill ko
    makeMoveAssertLegal(hist, board, Location::getLoc(3,3,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoProhibition(out,board,hist);

    string expected = R"%%(
HASH: 211F4559FB155DA94DC2C5CB753077E7
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko prohibited black at D4
HASH: 211F4559FB155DA94DC2C5CB753077E7
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


HASH: 211F4559FB155DA94DC2C5CB753077E7
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


HASH: 1565EDCF73D8956419E2D5E305F15BEC
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . . X . X . .
 2 . . . X . . .
 1 . . . . . . .


Ko prohibited white at D3
HASH: 1565EDCF73D8956419E2D5E305F15BEC
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . . X . X . .
 2 . . . X . . .
 1 . . . . . . .


Ko prohibited white at D3
HASH: C1B32AE0968F96F2D47DA5E16C8C1C82
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . . X X X . .
 2 . . . X . . .
 1 . . . . . . .

)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Territory scoring in the main phase";
    Board board = Board::parseBoard(7,7,R"%%(
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
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }
  {
    const char* name = "Territory scoring in encore 1";
    Board board = Board::parseBoard(7,7,R"%%(
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
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }
  {
    const char* name = "Territory scoring in encore 2";
    Board board = Board::parseBoard(7,7,R"%%(
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
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Pass for ko";
    Board board = Board::parseBoard(7,7,R"%%(
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
   A B C D E F G
 7 . . O X X O .
 6 . . O X X X O
 5 . O X O O X .
 4 . . . . O X X
 3 . . O . O O .
 2 . . . . . . .
 1 O . . . . . .


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
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Two step ko mark clearing";
    Board board = Board::parseBoard(7,5,R"%%(
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
   A B C D E F G
 5 . O X . . . .
 4 O X X . . . .
 3 . O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After first cap
Ko-prohibited: (1,0) O
HASH: D9AD14AFEF3A6FB208691C6EABB3ACC0
   A B C D E F G
 5 X . X . . . .
 4 O X X . . . .
 3 . O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After second cap
Ko-prohibited: (0,1) O
HASH: 06E53C9488F4A69B52F544FC1C6E1D40
   A B C D E F G
 5 X . X . . . .
 4 . X X . . . .
 3 X O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After pass for ko
Illegal: (0,1) O
HASH: 06E53C9488F4A69B52F544FC1C6E1D40
   A B C D E F G
 5 X . X . . . .
 4 . X X . . . .
 3 X O X . . . .
 2 O O O . . . .
 1 . . . . . . .
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Throw in that destroys the ko momentarily does not clear ko prohibition";
    Board board = Board::parseBoard(7,5,R"%%(
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
   A B C D E F G
 5 X . X . . . .
 4 . X X . . . .
 3 X O . . . . .
 2 O O . . . . .
 1 . . . . . . .


Ko-prohibited: (0,1) O
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Various komis";
    Board board = Board::parseBoard(7,6,R"%%(
.......
.......
ooooooo
xxxxxxx
.......
.......
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    BoardHistory hist(board,P_BLACK,rules);

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.isGameFinished == true);
    printGameResult(out,hist);

    hist.setKomi(0.0f);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.isGameFinished == true);
    printGameResult(out,hist);

    hist.setKomi(-0.5f);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.isGameFinished == true);
    printGameResult(out,hist);

    string expected = R"%%(
Winner: White
W-B Score: 0.5
isNoResult: 0
Winner: Empty
W-B Score: 0
isNoResult: 0
Winner: Black
W-B Score: -0.5
isNoResult: 0
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Stress test on tiny boards";

    Rand baseRand("Tiny board stress test");
    auto stressTest = [&](Board board, BoardHistory hist, Player nextPla, bool prolongGame) {
      Rand rand(baseRand.nextUInt64());
      for(int i = 0; i<1000; i++) {
        int numLegal = 0;
        Loc legalMoves[board.x_size*board.y_size + 1];
        Loc move;

        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            move = Location::getLoc(x,y,board.x_size);
            if(hist.isLegal(board, move, nextPla)) legalMoves[numLegal++] = move;
          }
        }
        move = Board::PASS_LOC;
        if(hist.isLegal(board, move, nextPla)) legalMoves[numLegal++] = move;

        out << numLegal;
        out << " ";
        for(int y = 0; y<board.y_size; y++)
          for(int x = 0; x<board.x_size; x++)
            out << colorToChar(board.colors[Location::getLoc(x,y,board.x_size)]);
        out << " NP" << colorToChar(nextPla);
        out << " PS" << hist.consecutiveEndingPasses;
        out << " E" << hist.encorePhase;
        out << " ";
        for(int y = 0; y<board.y_size; y++)
          for(int x = 0; x<board.x_size; x++)
            out << (int)(hist.blackKoProhibited[Location::getLoc(x,y,board.x_size)]);
        out << " ";
        for(int y = 0; y<board.y_size; y++)
          for(int x = 0; x<board.x_size; x++)
            out << (int)(hist.whiteKoProhibited[Location::getLoc(x,y,board.x_size)]);
        out << " ";
        for(int y = 0; y<board.y_size; y++)
          for(int x = 0; x<board.x_size; x++)
            out << (int)(hist.secondEncoreStartColors[Location::getLoc(x,y,board.x_size)]);

        out << endl;

        if(hist.isGameFinished)
          break;

        assert(numLegal > 0);
        move = legalMoves[rand.nextUInt(numLegal)];
        if(prolongGame && move == Board::PASS_LOC)
          move = legalMoves[rand.nextUInt(numLegal)];
        makeMoveAssertLegal(hist, board, move, nextPla, __LINE__);
        nextPla = getOpp(nextPla);
      }
      printGameResult(out,hist);
    };

    Board emptyBoard22 = Board::parseBoard(2,2,R"%%(
..
..
)%%");

    Rules rules;
    string expected;

    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules),P_BLACK,true);
    rules.multiStoneSuicideLegal = true;
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules),P_BLACK,true);
    expected = R"%%(
5 .... NPX PS0 E0 0000 0000 0000
4 .X.. NPO PS0 E0 0000 0000 0000
3 .X.O NPX PS0 E0 0000 0000 0000
1 .XX. NPO PS0 E0 0000 0000 0000
3 .XX. NPX PS1 E0 0000 0000 0000
2 XXX. NPO PS0 E0 0000 0000 0000
4 ...O NPX PS0 E0 0000 0000 0000
4 ...O NPO PS1 E0 0000 0000 0000
1 O..O NPX PS0 E0 0000 0000 0000
3 O..O NPO PS1 E0 0000 0000 0000
2 O.OO NPX PS0 E0 0000 0000 0000
1 O.OO NPO PS1 E0 0000 0000 0000
2 O.OO NPX PS0 E1 0000 0000 0000
1 O.OO NPO PS1 E1 0000 0000 0000
2 O.OO NPX PS0 E2 0000 0000 2022
1 O.OO NPO PS1 E2 0000 0000 2022
2 O.OO NPX PS2 E2 0000 0000 2022
Winner: White
W-B Score: 2.5
isNoResult: 0
5 .... NPX PS0 E0 0000 0000 0000
4 ..X. NPO PS0 E0 0000 0000 0000
3 .OX. NPX PS0 E0 0000 0000 0000
2 XOX. NPO PS0 E0 0000 0000 0000
2 XOX. NPX PS1 E0 0000 0000 0000
2 X.XX NPO PS0 E0 0000 0000 0000
4 .O.. NPX PS0 E0 0000 0000 0000
3 XO.. NPO PS0 E0 0000 0000 0000
1 .OO. NPX PS0 E0 0000 0000 0000
3 .OO. NPO PS1 E0 0000 0000 0000
2 .OOO NPX PS0 E0 0000 0000 0000
4 X... NPO PS0 E0 0000 0000 0000
3 X..O NPX PS0 E0 0000 0000 0000
2 XX.O NPO PS0 E0 0000 0000 0000
3 ..OO NPX PS0 E0 0000 0000 0000
2 .XOO NPO PS0 E0 0000 0000 0000
2 .XOO NPX PS1 E0 0000 0000 0000
3 XX.. NPO PS0 E0 0000 0000 0000
2 XX.O NPX PS0 E0 0000 0000 0000
2 XXX. NPO PS0 E0 0000 0000 0000
4 ...O NPX PS0 E0 0000 0000 0000
3 .X.O NPO PS0 E0 0000 0000 0000
1 O..O NPX PS0 E0 0000 0000 0000
3 O..O NPO PS1 E0 0000 0000 0000
2 OO.O NPX PS0 E0 0000 0000 0000
4 ..X. NPO PS0 E0 0000 0000 0000
3 ..XO NPX PS0 E0 0000 0000 0000
3 ..XO NPO PS1 E0 0000 0000 0000
2 .OXO NPX PS0 E0 0000 0000 0000
2 .OXO NPO PS1 E0 0000 0000 0000
2 OO.O NPX PS0 E0 0000 0000 0000
4 ..X. NPO PS0 E0 0000 0000 0000
3 .OX. NPX PS0 E0 0000 0000 0000
2 .OXX NPO PS0 E0 0000 0000 0000
3 OO.. NPX PS0 E0 0000 0000 0000
2 OOX. NPO PS0 E0 0000 0000 0000
2 OO.O NPX PS0 E0 0000 0000 0000
4 ..X. NPO PS0 E0 0000 0000 0000
3 O.X. NPX PS0 E0 0000 0000 0000
2 O.XX NPO PS0 E0 0000 0000 0000
3 OO.. NPX PS0 E0 0000 0000 0000
2 OO.X NPO PS0 E0 0000 0000 0000
2 OOO. NPX PS0 E0 0000 0000 0000
4 ...X NPO PS0 E0 0000 0000 0000
3 O..X NPX PS0 E0 0000 0000 0000
2 O.XX NPO PS0 E0 0000 0000 0000
3 OO.. NPX PS0 E0 0000 0000 0000
Winner: Empty
W-B Score: 0
isNoResult: 1
)%%";

    expect(name,out.str(),expected);
    out.str("");
    out.clear();

    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules),P_BLACK,false);
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules),P_BLACK,false);
    rules.multiStoneSuicideLegal = true;
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules),P_BLACK,false);
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules),P_BLACK,false);
    expected = R"%%(
5 .... NPX PS0 E0 0000 0000 0000
5 .... NPO PS1 E0 0000 0000 0000
5 .... NPX PS2 E0 0000 0000 0000
Winner: White
W-B Score: 0.5
isNoResult: 0
5 .... NPX PS0 E0 0000 0000 0000
5 .... NPO PS1 E0 0000 0000 0000
4 O... NPX PS0 E0 0000 0000 0000
3 O.X. NPO PS0 E0 0000 0000 0000
2 OOX. NPX PS0 E0 0000 0000 0000
2 OOX. NPO PS1 E0 0000 0000 0000
2 OOX. NPX PS2 E0 0000 0000 0000
Winner: White
W-B Score: 1.5
isNoResult: 0
5 .... NPX PS0 E0 0000 0000 0000
4 .X.. NPO PS0 E0 0000 0000 0000
3 .XO. NPX PS0 E0 0000 0000 0000
2 .XOX NPO PS0 E0 0000 0000 0000
2 .XOX NPX PS1 E0 0000 0000 0000
2 .XOX NPO PS2 E0 0000 0000 0000
Winner: Black
W-B Score: -0.5
isNoResult: 0
5 .... NPX PS0 E0 0000 0000 0000
4 ...X NPO PS0 E0 0000 0000 0000
3 ..OX NPX PS0 E0 0000 0000 0000
3 ..OX NPO PS1 E0 0000 0000 0000
3 ..OX NPX PS2 E0 0000 0000 0000
Winner: White
W-B Score: 0.5
isNoResult: 0
)%%";

    expect(name,out.str(),expected);
    out.str("");
    out.clear();

    Board koBoard71 = Board::parseBoard(7,1,R"%%(
.o.ox.o
)%%");
    Board koBoard41 = Board::parseBoard(4,1,R"%%(
....
)%%");

    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    stressTest(koBoard71,BoardHistory(koBoard71,P_BLACK,rules),P_BLACK,true);

    expected = R"%%(
3 .O.OX.O NPX PS0 E0 0000000 0000000 0000000
1 .OX.X.O NPO PS0 E0 0000000 0000000 0000000
4 .OX.X.O NPX PS0 E0 0000000 0000000 0000000
2 .OXXX.O NPO PS0 E0 0000000 0000000 0000000
4 .O...OO NPX PS0 E0 0000000 0000000 0000000
6 .O..X.. NPO PS0 E0 0000000 0000000 0000000
4 .O..XO. NPX PS0 E0 0000000 0000000 0000000
3 .O.XXO. NPO PS0 E0 0000000 0000000 0000000
2 .O.XXO. NPX PS1 E0 0000000 0000000 0000000
3 .O.XX.X NPO PS0 E0 0000000 0000000 0000000
3 OO.XX.X NPX PS0 E0 0000000 0000000 0000000
4 ..XXX.X NPO PS0 E0 0000000 0000000 0000000
3 ..XXXO. NPX PS0 E0 0000000 0000000 0000000
2 .XXXXO. NPO PS0 E0 0000000 0000000 0000000
2 .XXXXO. NPX PS1 E0 0000000 0000000 0000000
1 .XXXX.X NPO PS0 E0 0000000 0000000 0000000
3 .XXXX.X NPX PS0 E0 0000000 0000000 0000000
2 .XXXXXX NPO PS0 E0 0000000 0000000 0000000
7 O...... NPX PS0 E0 0000000 0000000 0000000
5 O....X. NPO PS0 E0 0000000 0000000 0000000
5 O..O.X. NPX PS0 E0 0000000 0000000 0000000
3 .X.O.X. NPO PS0 E0 0000000 0000000 0000000
3 .X.OOX. NPX PS0 E0 0000000 0000000 0000000
3 XX.OOX. NPO PS0 E0 0000000 0000000 0000000
3 ..OOOX. NPX PS0 E0 0000000 0000000 0000000
3 X.OOOX. NPO PS0 E0 0000000 0000000 0000000
1 X.OOO.O NPX PS0 E0 0000000 0000000 0000000
3 X.OOO.O NPO PS0 E0 0000000 0000000 0000000
2 X.OOOOO NPX PS0 E0 0000000 0000000 0000000
2 X.OOOOO NPO PS1 E0 0000000 0000000 0000000
2 .OOOOOO NPX PS0 E0 0000000 0000000 0000000
1 .OOOOOO NPO PS1 E0 0000000 0000000 0000000
2 .OOOOOO NPX PS0 E1 0000000 0000000 0000000
7 X...... NPO PS0 E1 0000000 0000000 0000000
6 X..O... NPX PS0 E1 0000000 0000000 0000000
5 XX.O... NPO PS0 E1 0000000 0000000 0000000
1 XX.O.O. NPX PS0 E1 0000000 0000000 0000000
4 XX.O.O. NPO PS1 E1 0000000 0000000 0000000
3 ..OO.O. NPX PS0 E1 0000000 0000000 0000000
4 .XOO.O. NPO PS0 E1 0000000 0000000 0000000
2 O.OO.O. NPX PS0 E1 0100000 0000000 0000000
3 O.OO.O. NPO PS0 E1 0000000 0000000 0000000
2 O.OO.O. NPX PS1 E1 0000000 0000000 0000000
4 .XOO.O. NPO PS0 E1 0000000 1000000 0000000
2 .XOO.OO NPX PS0 E1 0000000 1000000 0000000
5 .X..X.. NPO PS0 E1 0000000 0000000 0000000
5 .X.OX.. NPX PS0 E1 0000000 0000000 0000000
4 XX.OX.. NPO PS0 E1 0000000 0000000 0000000
4 XX.OX.. NPX PS1 E1 0000000 0000000 0000000
3 XX.OXX. NPO PS0 E1 0000000 0000000 0000000
3 XX.O..O NPX PS0 E1 0000000 0000000 0000000
4 XX.O..O NPO PS1 E1 0000000 0000000 0000000
2 XX.OO.O NPX PS0 E1 0000000 0000000 0000000
3 XX.OOX. NPO PS0 E1 0000000 0000001 0000000
2 XX.OOX. NPX PS0 E1 0000000 0000000 0000000
3 XXX..X. NPO PS0 E1 0000000 0000000 0000000
2 XXX.OX. NPX PS0 E1 0000000 0000000 0000000
2 XXXX.X. NPO PS0 E1 0000000 0000000 0000000
5 ....OX. NPX PS0 E1 0000000 0000000 0000000
4 ..X.OX. NPO PS0 E1 0000000 0000000 0000000
3 O.X.OX. NPX PS0 E1 0000000 0000000 0000000
1 O.XX.X. NPO PS0 E1 0000000 0000000 0000000
4 O.XX.X. NPX PS1 E1 0000000 0000000 0000000
1 .XXX.X. NPO PS0 E1 0000000 0000000 0000000
4 .XXX.X. NPX PS1 E1 0000000 0000000 0000000
2 .XXX.XX NPO PS0 E1 0000000 0000000 0000000
3 .XXXO.. NPX PS0 E1 0000000 0000000 0000000
1 .XXX.X. NPO PS0 E1 0000000 0000000 0000000
4 .XXX.X. NPX PS0 E2 0000000 0000000 0111010
1 .XXXXX. NPO PS0 E2 0000000 0000000 0111010
3 .XXXXX. NPX PS1 E2 0000000 0000000 0111010
2 XXXXXX. NPO PS0 E2 0000000 0000000 0111010
1 XXXXXX. NPX PS1 E2 0000000 0000000 0111010
2 XXXXXX. NPO PS2 E2 0000000 0000000 0111010
Winner: Black
W-B Score: -0.5
isNoResult: 0

)%%";

    expect(name,out.str(),expected);
    out.str("");
    out.clear();

    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules),P_BLACK,true);
    expected = R"%%(
5 .... NPX PS0 E0 0000 0000 0000
4 X... NPO PS0 E0 0000 0000 0000
1 X.O. NPX PS0 E0 0000 0000 0000
3 X.O. NPO PS1 E0 0000 0000 0000
2 X.OO NPX PS0 E0 0000 0000 0000
3 XX.. NPO PS0 E0 0000 0000 0000
3 XX.. NPX PS1 E0 0000 0000 0000
3 XX.. NPO PS0 E1 0000 0000 0000
2 XX.O NPX PS0 E1 0000 0000 0000
2 XXX. NPO PS0 E1 0000 0000 0000
4 ...O NPX PS0 E1 0000 0000 0000
3 X..O NPO PS0 E1 0000 0000 0000
2 X.OO NPX PS0 E1 0000 0000 0000
3 XX.. NPO PS0 E1 0000 0000 0000
3 ..O. NPX PS0 E1 0000 0000 0000
2 .XO. NPO PS0 E1 0000 0000 0000
2 O.O. NPX PS0 E1 0100 0000 0000
2 O.O. NPO PS0 E1 0000 0000 0000
2 O.O. NPX PS1 E1 0000 0000 0000
2 .XO. NPO PS0 E1 0000 1000 0000
2 .XO. NPX PS0 E1 0000 0000 0000
1 .XO. NPO PS1 E1 0000 0000 0000
2 .XO. NPX PS0 E2 0000 0000 0120
2 .X.X NPO PS0 E2 0000 0010 0120
3 .X.X NPX PS1 E2 0000 0010 0120
2 XX.X NPO PS0 E2 0000 0000 0120
3 ..O. NPX PS0 E2 0000 0000 0120
4 ..O. NPO PS1 E2 0000 0000 0120
1 .OO. NPX PS0 E2 0000 0000 0120
3 .OO. NPO PS1 E2 0000 0000 0120
2 OOO. NPX PS0 E2 0000 0000 0120
4 ...X NPO PS0 E2 0000 0000 0120
1 .O.X NPX PS0 E2 0000 0000 0120
3 .O.X NPO PS1 E2 0000 0000 0120
2 OO.X NPX PS0 E2 0000 0000 0120
3 ..XX NPO PS0 E2 0000 0000 0120
3 .O.. NPX PS0 E2 0000 0000 0120
3 .O.X NPO PS0 E2 0000 0000 0120
2 OO.X NPX PS0 E2 0000 0000 0120
2 OO.X NPO PS1 E2 0000 0000 0120
2 OO.X NPX PS2 E2 0000 0000 0120
Winner: White
W-B Score: 0.5
isNoResult: 0

)%%";

    expect(name,out.str(),expected);
    out.str("");
    out.clear();

    rules.koRule = Rules::KO_SITUATIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules),P_BLACK,true);
    expected = R"%%(

5 .... NPX PS0 E0 0000 0000 0000
4 X... NPO PS0 E0 0000 0000 0000
3 .O.. NPX PS0 E0 0000 0000 0000
3 .O.X NPO PS0 E0 0000 0000 0000
2 OO.X NPX PS0 E0 0000 0000 0000
2 ..XX NPO PS0 E0 0000 0000 0000
2 O.XX NPX PS0 E0 0000 0000 0000
2 O.XX NPO PS1 E0 0000 0000 0000
3 OO.. NPX PS0 E0 0000 0000 0000
3 ..X. NPO PS0 E0 0000 0000 0000
2 O.X. NPX PS0 E0 0000 0000 0000
1 .XX. NPO PS0 E0 0000 0000 0000
3 .XX. NPX PS1 E0 0000 0000 0000
2 .XXX NPO PS0 E0 0000 0000 0000
4 O... NPX PS0 E0 0000 0000 0000
2 O..X NPO PS0 E0 0000 0000 0000
2 O.O. NPX PS0 E0 0000 0000 0000
1 .XO. NPO PS0 E0 0000 0000 0000
2 .XO. NPX PS1 E0 0000 0000 0000
1 .X.X NPO PS0 E0 0000 0000 0000
2 .X.X NPX PS1 E0 0000 0000 0000
2 .X.X NPO PS0 E1 0000 0000 0000
2 .XO. NPX PS0 E1 0001 0000 0000
2 .XO. NPO PS0 E1 0000 0000 0000
2 O.O. NPX PS0 E1 0100 0000 0000
3 O.O. NPO PS1 E1 0100 0000 0000
2 O.OO NPX PS0 E1 0000 0000 0000
3 .X.. NPO PS0 E1 0000 0000 0000
3 .X.O NPX PS0 E1 0000 0000 0000
2 XX.O NPO PS0 E1 0000 0000 0000
3 ..OO NPX PS0 E1 0000 0000 0000
2 X.OO NPO PS0 E1 0000 0000 0000
2 .OOO NPX PS0 E1 0000 0000 0000
4 X... NPO PS0 E1 0000 0000 0000
1 X.O. NPX PS0 E1 0000 0000 0000
3 X.O. NPO PS1 E1 0000 0000 0000
1 .OO. NPX PS0 E1 0000 0000 0000
3 .OO. NPO PS1 E1 0000 0000 0000
2 .OOO NPX PS0 E1 0000 0000 0000
4 X... NPO PS0 E1 0000 0000 0000
3 X..O NPX PS0 E1 0000 0000 0000
3 X..O NPO PS1 E1 0000 0000 0000
2 X.OO NPX PS0 E1 0000 0000 0000
3 XX.. NPO PS0 E1 0000 0000 0000
3 ..O. NPX PS0 E1 0000 0000 0000
1 .XO. NPO PS0 E1 0000 0000 0000
2 .XO. NPX PS1 E1 0000 0000 0000
2 .X.X NPO PS0 E1 0000 0010 0000
2 .X.X NPX PS0 E1 0000 0000 0000
1 .X.X NPO PS1 E1 0000 0000 0000
3 .X.X NPX PS0 E2 0000 0000 0101
2 XX.X NPO PS0 E2 0000 0000 0101
1 XX.X NPX PS1 E2 0000 0000 0101
2 XX.X NPO PS2 E2 0000 0000 0101
Winner: Black
W-B Score: -3.5
isNoResult: 0

)%%";

    expect(name,out.str(),expected);
    out.str("");
    out.clear();


    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules),P_BLACK,true);

    expected = R"%%(
5 .... NPX PS0 E0 0000 0000 0000
4 ...X NPO PS0 E0 0000 0000 0000
3 ..O. NPX PS0 E0 0000 0000 0000
2 .XO. NPO PS0 E0 0000 0000 0000
1 O.O. NPX PS0 E0 0000 0000 0000
3 O.O. NPO PS0 E0 0000 0000 0000
2 O.OO NPX PS0 E0 0000 0000 0000
3 .X.. NPO PS0 E0 0000 0000 0000
2 .XO. NPX PS0 E0 0000 0000 0000
1 .X.X NPO PS0 E0 0000 0000 0000
3 .X.X NPX PS0 E0 0000 0000 0000
2 .XXX NPO PS0 E0 0000 0000 0000
4 O... NPX PS0 E0 0000 0000 0000
4 O... NPO PS1 E0 0000 0000 0000
3 OO.. NPX PS0 E0 0000 0000 0000
3 ..X. NPO PS0 E0 0000 0000 0000
2 .OX. NPX PS0 E0 0000 0000 0000
1 X.X. NPO PS0 E0 0000 0000 0000
3 X.X. NPX PS0 E0 0000 0000 0000
2 X.XX NPO PS0 E0 0000 0000 0000
3 .O.. NPX PS0 E0 0000 0000 0000
3 .O.X NPO PS0 E0 0000 0000 0000
1 .OO. NPX PS0 E0 0000 0000 0000
3 .OO. NPO PS1 E0 0000 0000 0000
2 OOO. NPX PS0 E0 0000 0000 0000
4 ...X NPO PS0 E0 0000 0000 0000
3 O..X NPX PS0 E0 0000 0000 0000
2 O.XX NPO PS0 E0 0000 0000 0000
3 OO.. NPX PS0 E0 0000 0000 0000
2 OO.X NPO PS0 E0 0000 0000 0000
2 OO.X NPX PS1 E0 0000 0000 0000
3 ..XX NPO PS0 E0 0000 0000 0000
2 O.XX NPX PS0 E0 0000 0000 0000
2 O.XX NPO PS1 E0 0000 0000 0000
3 OO.. NPX PS0 E0 0000 0000 0000
2 OO.X NPO PS0 E0 0000 0000 0000
2 OO.X NPX PS1 E0 0000 0000 0000
Winner: White
W-B Score: 1.5
isNoResult: 0

)%%";

    expect(name,out.str(),expected);
    out.str("");
    out.clear();

    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules),P_BLACK,true);
    expected = R"%%(
5 .... NPX PS0 E0 0000 0000 0000
4 ...X NPO PS0 E0 0000 0000 0000
1 .O.X NPX PS0 E0 0000 0000 0000
3 .O.X NPO PS1 E0 0000 0000 0000
2 OO.X NPX PS0 E0 0000 0000 0000
3 ..XX NPO PS0 E0 0000 0000 0000
2 .O.. NPX PS0 E0 0000 0000 0000
4 .O.. NPO PS1 E0 0000 0000 0000
2 OO.. NPX PS0 E0 0000 0000 0000
3 OO.. NPO PS1 E0 0000 0000 0000
1 OOO. NPX PS0 E0 0000 0000 0000
1 OOO. NPO PS1 E0 0000 0000 0000
1 OOO. NPX PS2 E0 0000 0000 0000
Winner: White
W-B Score: 4.5
isNoResult: 0

)%%";

    expect(name,out.str(),expected);
    out.str("");
    out.clear();

    rules.koRule = Rules::KO_SITUATIONAL;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules),P_BLACK,true);
    expected = R"%%(
5 .... NPX PS0 E0 0000 0000 0000
4 X... NPO PS0 E0 0000 0000 0000
3 .O.. NPX PS0 E0 0000 0000 0000
3 .O.X NPO PS0 E0 0000 0000 0000
1 .OO. NPX PS0 E0 0000 0000 0000
3 .OO. NPO PS1 E0 0000 0000 0000
1 .OOO NPX PS0 E0 0000 0000 0000
1 .OOO NPO PS1 E0 0000 0000 0000
1 .OOO NPX PS2 E0 0000 0000 0000
Winner: White
W-B Score: 4.5
isNoResult: 0

)%%";

    expect(name,out.str(),expected);
    out.str("");
    out.clear();


  }

}
