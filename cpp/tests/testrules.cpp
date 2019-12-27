#include "../tests/tests.h"
#include "../dataio/sgf.h"

using namespace std;
using namespace TestCommon;

static void checkKoHashConsistency(BoardHistory& hist, Board& board, Player nextPla) {
  testAssert(hist.koHashHistory.size() > 0);
  Hash128 expected = board.pos_hash;
  if(hist.encorePhase > 0) {
    expected ^= Board::ZOBRIST_PLAYER_HASH[nextPla];
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(hist.koRecapBlocked[loc])
          expected ^= Board::ZOBRIST_KO_MARK_HASH[loc][P_BLACK] ^ Board::ZOBRIST_KO_MARK_HASH[loc][P_WHITE];
      }
    }
  }
  else if(hist.rules.koRule == Rules::KO_SITUATIONAL || hist.rules.koRule == Rules::KO_SIMPLE) {
    expected ^= Board::ZOBRIST_PLAYER_HASH[nextPla];
  }
  testAssert(expected == hist.koHashHistory[hist.koHashHistory.size()-1]);
}

static void makeMoveAssertLegal(BoardHistory& hist, Board& board, Loc loc, Player pla, int line, bool preventEncore, const KoHashTable* table) {
  bool phaseWouldEnd = hist.passWouldEndPhase(board,pla);
  int oldPhase = hist.encorePhase;

  if(!hist.isLegal(board, loc, pla))
    throw StringError("Illegal move on line " + Global::intToString(line));
  hist.makeBoardMoveAssumeLegal(board, loc, pla, table, preventEncore);
  checkKoHashConsistency(hist,board,getOpp(pla));

  if(loc == Board::PASS_LOC) {
    int newPhase = hist.encorePhase;
    if((phaseWouldEnd && !preventEncore) != (newPhase != oldPhase || hist.isGameFinished))
      throw StringError("hist.passWouldEndPhase returned different answer than what actually happened after a pass");
  }
}

static void makeMoveAssertLegal(BoardHistory& hist, Board& board, Loc loc, Player pla, int line, bool preventEncore) {
  makeMoveAssertLegal(hist,board,loc,pla,line,preventEncore,NULL);
}

static void makeMoveAssertLegal(BoardHistory& hist, Board& board, Loc loc, Player pla, int line, const KoHashTable* table) {
  makeMoveAssertLegal(hist,board,loc,pla,line,false,table);
}

static void makeMoveAssertLegal(BoardHistory& hist, Board& board, Loc loc, Player pla, int line) {
  makeMoveAssertLegal(hist,board,loc,pla,line,false,NULL);
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
  testAssert(hist.finalWhiteMinusBlackScore == score);

  BoardHistory hist2(baseHist);
  hist2.endAndScoreGameNow(baseBoard);
  testAssert(hist2.finalWhiteMinusBlackScore == score);

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
          o << "Illegal: " << Location::toStringMach(loc,board.x_size) << " " << PlayerIO::colorToChar(pla) << endl;
        }
        if(hist.koRecapBlocked[loc]) {
          o << "Ko-recap-blocked: " << Location::toStringMach(loc,board.x_size) << endl;
        }
      }
    }
  };

  auto printEncoreKoBlock = [](ostream& o, const Board& board, const BoardHistory& hist) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(hist.koRecapBlocked[loc])
          o << "Ko recap blocked at " << Location::toString(loc,board) << endl;
      }
    }
  };

  auto printGameResult = [](ostream& o, const BoardHistory& hist) {
    if(!hist.isGameFinished)
      o << "Game is not over" << endl;
    else {
      o << "Winner: " << PlayerIO::playerToString(hist.winner) << endl;
      o << "W-B Score: " << hist.finalWhiteMinusBlackScore << endl;
      o << "isNoResult: " << hist.isNoResult << endl;
      o << "isResignation: " << hist.isResignation << endl;
      testAssert((int)hist.isNoResult + (int)hist.isResignation + (int)hist.isScored == (int)hist.isGameFinished);
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
    rules.taxRule = Rules::TAX_NONE;
    BoardHistory hist(board,P_BLACK,rules,0);

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
HASH: 5FA73DC4EC4D5C8EF52ECF27BFF1754C
   A B C D
 4 . X O .
 3 . X O .
 2 . X O O
 1 . X O .
)%%";
    expect(name,out,expected);
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
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_BLACK,rules,0);

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
    testAssert(hist.winner == P_BLACK);
    testAssert(hist.finalWhiteMinusBlackScore == -0.5f);
    out << board << endl;

    //Resurrecting the board after pass to have black throw in a dead stone, since second encore, should make no difference
    makeMoveAssertLegal(hist, board, Location::getLoc(3,1,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameFinished == true);
    testAssert(hist.winner == P_BLACK);
    testAssert(hist.finalWhiteMinusBlackScore == -0.5f);
    out << board << endl;

    //Resurrecting again to have white throw in a junk stone that makes it unclear if black has anything
    //White gets a point for playing, but it's not there second encore, so again no difference
    makeMoveAssertLegal(hist, board, Location::getLoc(0,1,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameFinished == true);
    testAssert(hist.winner == P_WHITE);
    testAssert(hist.finalWhiteMinusBlackScore == 3.5f);
    out << board << endl;

    //Resurrecting again to have black solidfy his group and prove it pass-alive
    makeMoveAssertLegal(hist, board, Location::getLoc(0,2,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,0,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    //Back to the original result
    testAssert(hist.encorePhase == 2);
    testAssert(hist.isGameFinished == true);
    testAssert(hist.winner == P_BLACK);
    testAssert(hist.finalWhiteMinusBlackScore == -0.5f);
    out << board << endl;

    string expected = R"%%(
HASH: 5FA73DC4EC4D5C8EF52ECF27BFF1754C
   A B C D
 4 . X O .
 3 . X O .
 2 . X O O
 1 . X O .


HASH: D7D56E29425FCBAE79353E413C56BE86
   A B C D
 4 . X O .
 3 . X O X
 2 . X O O
 1 . X O .


HASH: 6B8B39058225C530338D4A62EEB12785
   A B C D
 4 . X O .
 3 O X O X
 2 . X O O
 1 . X O .


HASH: 820C9E3B06A638D02CBDE734B72F2777
   A B C D
 4 . X O O
 3 O X O .
 2 X X O O
 1 . X O .

)%%";
    expect(name,out,expected);
  }


  //Ko rule testing with a regular ko and a sending two returning 1
  {
    Board baseBoard = Board::parseBoard(6,5,R"%%(
.o.xxo
oxxxo.
o.x.oo
xx.oo.
oooo.o
)%%");

    Rules baseRules;
    baseRules.koRule = Rules::KO_POSITIONAL;
    baseRules.scoringRule = Rules::SCORING_TERRITORY;
    baseRules.komi = 0.5f;
    baseRules.multiStoneSuicideLegal = false;
    baseRules.taxRule = Rules::TAX_SEKI;

    {
      const char* name = "Simple ko rules";
      Board board(baseBoard);
      Rules rules(baseRules);
      rules.koRule = Rules::KO_SIMPLE;
      BoardHistory hist(board,P_BLACK,rules,0);

      makeMoveAssertLegal(hist, board, Location::getLoc(5,1,board.x_size), P_BLACK, __LINE__);
      out << "After black ko capture:" << endl;
      printIllegalMoves(out,board,hist,P_WHITE);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      out << "After black ko capture and one pass:" << endl;
      printIllegalMoves(out,board,hist,P_BLACK);

      testAssert(hist.passWouldEndPhase(board,P_BLACK));
      testAssert(!hist.passWouldEndGame(board,P_BLACK));
      makeMoveAssertLegal(hist, board, Location::getLoc(2,3,board.x_size), P_BLACK, __LINE__);
      testAssert(hist.encorePhase == 0);
      testAssert(hist.isGameFinished == false);

      out << "After black ko capture and one pass and black other move:" << endl;
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
After black ko capture and one pass and black other move:
White recapture:
Illegal: (5,1) X
Beginning sending two returning one cycle
Winner: Black
W-B Score: -1.5
isNoResult: 0
isResignation: 0
)%%";
      expect(name,out,expected);
    }

    {
      const char* name = "Positional ko rules";
      Board board(baseBoard);
      Rules rules(baseRules);
      rules.koRule = Rules::KO_POSITIONAL;
      BoardHistory hist(board,P_BLACK,rules,0);

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
      expect(name,out,expected);
    }

    {
      const char* name = "Situational ko rules";
      Board board(baseBoard);
      Rules rules(baseRules);
      rules.koRule = Rules::KO_SITUATIONAL;
      BoardHistory hist(board,P_BLACK,rules,0);

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
      expect(name,out,expected);
    }

    {
      const char* name = "Spight ko rules";
      Board board(baseBoard);
      board.setStone(Location::getLoc(2,3,board.x_size),C_BLACK);
      Rules rules(baseRules);
      rules.koRule = Rules::KO_SPIGHT;
      BoardHistory hist(board,P_BLACK,rules,0);

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
W-B Score: -2.5
isNoResult: 0
isResignation: 0
)%%";
      expect(name,out,expected);
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
    baseRules.taxRule = Rules::TAX_NONE;

    int koRulesToTest[3] = { Rules::KO_POSITIONAL, Rules::KO_SITUATIONAL, Rules::KO_SPIGHT };
    const char* name = "Suicide ko testing";
    for(int i = 0; i<3; i++)
    {
      out << "------------------------------" << endl;
      Board board(baseBoard);
      Rules rules(baseRules);
      rules.koRule = koRulesToTest[i];
      BoardHistory hist(board,P_BLACK,rules,0);

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
HASH: D9EA171850FEC7E00195801AB2AC1575
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
HASH: D9EA171850FEC7E00195801AB2AC1575
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
HASH: D9EA171850FEC7E00195801AB2AC1575
   A B C D E F
 5 . O X O X .
 4 . X X O O O
 3 X X . . . .
 2 . . . . . .
 1 . . . . . .

)%%";
    expect(name,out,expected);
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
    rules.taxRule = Rules::TAX_NONE;
    BoardHistory hist(board,P_BLACK,rules,0);

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
isResignation: 0
)%%";
    expect(name,out,expected);
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
    rules.taxRule = Rules::TAX_NONE;
    BoardHistory hist(board,P_BLACK,rules,0);

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
isResignation: 0
)%%";
    expect(name,out,expected);
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
    rules.taxRule = Rules::TAX_NONE;
    BoardHistory hist(board,P_BLACK,rules,0);

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
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_BLACK,rules,0);

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
    printEncoreKoBlock(out,board,hist);

    string expected = R"%%(
HASH: 2FA527ADE62EF25B530B64733AFFDBF6
   A B C D E F G
 6 . . . . . . .
 5 . X . X . X .
 4 X . X . X . X
 3 X X X X X X X
 2 O O O O O O O
 1 . . . . . . .


Ko recap blocked at F5

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Encore - own throwin that temporarily breaks the ko shape should not clear the ko recap block";
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
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_WHITE,rules,0);

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,3,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,1,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(1,1,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);

    string expected = R"%%(
HASH: 7232311C746B8B9CD09B4B5E78F36FDB
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
HASH: 51A42639B1FD03594FC9F5DCAF16D642
   A B C D E F G
 6 . . O . . . .
 5 . . X O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
HASH: C28F759972CFA74DCA869C1EE08828C2
   A B C D E F G
 6 . . O . . . .
 5 . O . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Encore - ko recap block does not stop non-ko-capture";
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
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_WHITE,rules,0);

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,3,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,1,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(0,0,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);

    string expected = R"%%(
HASH: 7232311C746B8B9CD09B4B5E78F36FDB
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
HASH: 51A42639B1FD03594FC9F5DCAF16D642
   A B C D E F G
 6 . . O . . . .
 5 . . X O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
HASH: 3BA8E71777E554D6E368DCEC26777E08
   A B C D E F G
 6 O . O . . . .
 5 . . X O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
HASH: FEA42DE99C790CB13056CF1C1DE10E7C
   A B C D E F G
 6 O . O . . . .
 5 . . X O . . .
 4 . X . X O . .
 3 . . X . X . .
 2 . . . X . . .
 1 . . . . . . .

)%%";
    expect(name,out,expected);
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
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_WHITE,rules,0);

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,3,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    //Pass for ko
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    //Pass
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    //Take ko
    makeMoveAssertLegal(hist, board, Location::getLoc(3,2,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    //Pass
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    //Fill ko
    makeMoveAssertLegal(hist, board, Location::getLoc(3,3,board.x_size), P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);

    makeMoveAssertLegal(hist, board, Location::getLoc(1,3,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(2,4,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(3,5,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);
    makeMoveAssertLegal(hist, board, Location::getLoc(5,3,board.x_size), P_WHITE, __LINE__);
    out << board << endl;
    printEncoreKoBlock(out,board,hist);

    string expected = R"%%(
HASH: 7232311C746B8B9CD09B4B5E78F36FDB
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D3
HASH: 7232311C746B8B9CD09B4B5E78F36FDB
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


HASH: 7232311C746B8B9CD09B4B5E78F36FDB
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . . X O X . .
 2 . . . X . . .
 1 . . . . . . .


HASH: A191A543B756FCD6B78EF314F5CEBE65
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . . X . X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D4
HASH: A191A543B756FCD6B78EF314F5CEBE65
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . . X . X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D4
HASH: 83A43A9FDE43E4E9601FF8E2CB94D35A
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . . X X X . .
 2 . . . X . . .
 1 . . . . . . .


Ko recap blocked at D4
HASH: 48CD0E7B0968186D3722ABA96E12600B
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O X O . .
 3 . O X X X . .
 2 . . O X O . .
 1 . . . O . . .


Ko recap blocked at D4
HASH: E91FF9270E4CC0CCC9D560FA95C81D59
   A B C D E F G
 6 . . O . . . .
 5 . . . O . . .
 4 . X O . O . .
 3 . O . . . O .
 2 . . O . O . .
 1 . . . O . . .

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Area scoring in the main phase";

    int taxRules[3] = {Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL};
    for(int whichTaxRule = 0; whichTaxRule < 3; whichTaxRule++) {
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
      rules.scoringRule = Rules::SCORING_AREA;
      rules.komi = 0.5f;
      rules.multiStoneSuicideLegal = false;
      rules.taxRule = taxRules[whichTaxRule];
      BoardHistory hist(board,P_WHITE,rules,0);

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
      out << endl;
    }
    string expected = R"%%(
Score: -3.5
Score: -4.5
Score: -3.5
Score: -6.5
Score: -6.5
Score: -6.5
Score: -3.5
Score: -3.5

Score: 0.5
Score: -0.5
Score: 0.5
Score: -5.5
Score: -5.5
Score: -5.5
Score: -3.5
Score: -3.5

Score: 0.5
Score: -0.5
Score: 0.5
Score: -3.5
Score: -3.5
Score: -3.5
Score: -1.5
Score: -1.5
)%%";
    expect(name,out,expected);
  }
  {
    const char* name = "Territory scoring in the main phase";

    int taxRules[3] = {Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL};
    for(int whichTaxRule = 0; whichTaxRule < 3; whichTaxRule++) {
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
      rules.taxRule = taxRules[whichTaxRule];
      BoardHistory hist(board,P_WHITE,rules,0);

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
      out << endl;
    }
    string expected = R"%%(
Score: -3.5
Score: -3.5
Score: -3.5
Score: -5.5
Score: -6.5
Score: -5.5
Score: -3.5
Score: -2.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -4.5
Score: -5.5
Score: -4.5
Score: -3.5
Score: -2.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -2.5
Score: -3.5
Score: -2.5
Score: -1.5
Score: -0.5
)%%";
    expect(name,out,expected);
  }
  {
    const char* name = "Territory scoring in encore 1";

    int taxRules[3] = {Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL};
    for(int whichTaxRule = 0; whichTaxRule < 3; whichTaxRule++) {
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
      rules.taxRule = taxRules[whichTaxRule];
      BoardHistory hist(board,P_WHITE,rules,0);

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
      out << endl;
    }
    string expected = R"%%(
Score: -3.5
Score: -3.5
Score: -3.5
Score: -5.5
Score: -6.5
Score: -5.5
Score: -3.5
Score: -2.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -4.5
Score: -5.5
Score: -4.5
Score: -3.5
Score: -2.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -2.5
Score: -3.5
Score: -2.5
Score: -1.5
Score: -0.5
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Territory scoring in encore 2";

    int taxRules[3] = {Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL};
    for(int whichTaxRule = 0; whichTaxRule < 3; whichTaxRule++) {
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
      rules.taxRule = taxRules[whichTaxRule];
      BoardHistory hist(board,P_WHITE,rules,0);

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
      out << endl;
    }
    string expected = R"%%(
Score: -3.5
Score: -3.5
Score: -3.5
Score: -5.5
Score: -5.5
Score: -5.5
Score: -3.5
Score: -3.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -4.5
Score: -4.5
Score: -4.5
Score: -3.5
Score: -3.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -1.5
Score: -1.5
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Fill seki liberties in main phase";

    int taxRules[3] = {Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL};
    for(int whichTaxRule = 0; whichTaxRule < 3; whichTaxRule++) {
      Board board = Board::parseBoard(7,7,R"%%(
...oxx.
oooox.x
xxxxoxx
o.xoooo
.oxox.o
oxxo.x.
o.xoo.x
)%%");
      Rules rules;
      rules.koRule = Rules::KO_POSITIONAL;
      rules.scoringRule = Rules::SCORING_TERRITORY;
      rules.komi = -0.5f;
      rules.multiStoneSuicideLegal = false;
      rules.taxRule = taxRules[whichTaxRule];
      BoardHistory hist(board,P_WHITE,rules,0);

      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(6,5,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(5,6,board.x_size), P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(0,4,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(4,5,board.x_size), P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(5,4,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      out << endl;
    }
    string expected = R"%%(
Score: 1.5
Score: 1.5
Score: 1.5
Score: 0.5
Score: 1.5
Score: 0.5
Score: 0.5
Score: 10.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 2.5
Score: 1.5
Score: 1.5
Score: 11.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: -0.5
Score: -0.5
Score: 7.5
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Fill seki liberties in encore 2";

    int taxRules[3] = {Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL};
    for(int whichTaxRule = 0; whichTaxRule < 3; whichTaxRule++) {
      Board board = Board::parseBoard(7,7,R"%%(
...oxx.
oooox.x
xxxxoxx
o.xoooo
.oxox.o
oxxo.x.
o.xoo.x
)%%");
      Rules rules;
      rules.koRule = Rules::KO_POSITIONAL;
      rules.scoringRule = Rules::SCORING_TERRITORY;
      rules.komi = -0.5f;
      rules.multiStoneSuicideLegal = false;
      rules.taxRule = taxRules[whichTaxRule];
      BoardHistory hist(board,P_WHITE,rules,0);

      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(6,5,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(5,6,board.x_size), P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(0,4,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(1,0,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(4,5,board.x_size), P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(5,4,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      out << endl;
    }
    string expected = R"%%(
Score: 1.5
Score: 1.5
Score: 1.5
Score: 0.5
Score: 1.5
Score: 1.5
Score: 1.5
Score: 11.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 2.5
Score: 2.5
Score: 2.5
Score: 12.5

Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 0.5
Score: 8.5
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Area scoring with button";

    bool buttonRule[2] = {false, true};
    for(int whichRule = 0; whichRule < 2; whichRule++) {
      Board board = Board::parseBoard(7,7,R"%%(
..x.xo.
..xxoo.
...xo..
..xxo..
..x.o..
..xxo..
...xo..
)%%");
      Rules rules;
      rules.koRule = Rules::KO_SIMPLE;
      rules.scoringRule = Rules::SCORING_AREA;
      rules.taxRule = Rules::TAX_NONE;
      rules.multiStoneSuicideLegal = false;
      rules.komi = 2.5f;
      rules.hasButton = buttonRule[whichRule];
      BoardHistory hist(board,P_BLACK,rules,0);

      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(3,4,board.x_size), P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(3,0,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(1,2,board.x_size), P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(4,0,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Location::getLoc(6,2,board.x_size), P_WHITE, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished);
      out << "Score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      out << endl;
    }
    string expected = R"%%(
Score: -5.5
Score: -6.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -2.5
Score: -2.5

Score: -6
Score: -6
Score: -3
Score: -2
Score: -3
Score: -3
Score: -3
Score: -3
Score: -3
)%%";
    expect(name,out,expected);
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
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_BLACK,rules,0);
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
    testAssert(hashd == hist.koHashHistory[hist.koHashHistory.size()-1]);
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
    testAssert(hashd == hist.koHashHistory[hist.koHashHistory.size()-1]);
    makeMoveAssertLegal(hist, board, Location::getLoc(6,0,board.x_size), P_BLACK, __LINE__);
    testAssert(hashe == hist.koHashHistory[hist.koHashHistory.size()-1]);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hashf == hist.koHashHistory[hist.koHashHistory.size()-1]);
    out << "And see the only-once for black" << endl;
    printIllegalMoves(out,board,hist,P_BLACK);

    string expected = R"%%(
Black can't retake
Ko-recap-blocked: (5,0)
Ko threat shouldn't work in the encore
Ko-recap-blocked: (5,0)
Regular pass shouldn't work in the encore
Ko-recap-blocked: (5,0)
Pass for ko! (Should not affect the board stones)
HASH: 42FE4FEAAF27B840EA45877C528FEE84
   A B C D E F G
 7 . . O X X O .
 6 . . O X X X O
 5 . O X O O X .
 4 . . . . O X X
 3 . . O . O O .
 2 . . . . . . .
 1 O . . . . . .


Now black can retake, and white's retake isn't legal
Ko-recap-blocked: (6,0)
White's retake is legal after passing for ko
Black's retake is illegal again
Ko-recap-blocked: (5,0)
And is still illegal due to only-once
Illegal: (6,0) X
But a ko threat fixes that
White illegal now
Ko-recap-blocked: (6,0)
Legal again in second encore
Lastly, try black ko threat one more time
Ko-recap-blocked: (5,0)
And a pass for ko
And repeat with white
And see the only-once for black
Illegal: (6,0) X
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Two step ko in encore";
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
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_WHITE,rules,0);

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
    out << "After second pass for ko" << endl;
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
Ko-recap-blocked: (0,1)
After second cap
Ko-recap-blocked: (1,0)
Ko-recap-blocked: (0,1)
Just after black pass for ko
Ko-recap-blocked: (0,1)
HASH: 3E2C923D4675E38712F67207D0B3D21B
   A B C D E F G
 5 . O X . . . .
 4 O X X . . . .
 3 . O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After first cap
Ko-recap-blocked: (0,0)
Ko-recap-blocked: (0,1)
HASH: E51C9D5AE43BA59520B8877210F8CBED
   A B C D E F G
 5 X . X . . . .
 4 O X X . . . .
 3 . O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After second pass for ko
Ko-recap-blocked: (0,0)
HASH: E51C9D5AE43BA59520B8877210F8CBED
   A B C D E F G
 5 X . X . . . .
 4 O X X . . . .
 3 . O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After second cap
Ko-recap-blocked: (0,0)
Ko-recap-blocked: (0,2)
HASH: 8E15AD0AFD434346B3E4F2ED554621B7
   A B C D E F G
 5 X . X . . . .
 4 . X X . . . .
 3 X O X . . . .
 2 O O O . . . .
 1 . . . . . . .


After pass for ko
Ko-recap-blocked: (0,0)
Illegal: (0,1) O
HASH: 8E15AD0AFD434346B3E4F2ED554621B7
   A B C D E F G
 5 X . X . . . .
 4 . X X . . . .
 3 X O X . . . .
 2 O O O . . . .
 1 . . . . . . .
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Throw in that destroys the ko momentarily does not clear ko recap block";
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
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_BLACK,rules,0);

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
Ko-recap-blocked: (0,2)
HASH: 6CA50E111B93619273B4EEE5AC396990
   A B C D E F G
 5 X . X . . . .
 4 . X X . . . .
 3 X O . . . . .
 2 O O . . . . .
 1 . . . . . . .


Ko-recap-blocked: (0,2)
)%%";
    expect(name,out,expected);
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
    rules.taxRule = Rules::TAX_NONE;
    BoardHistory hist(board,P_BLACK,rules,0);

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
isResignation: 0
Winner: Empty
W-B Score: 0
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -0.5
isNoResult: 0
isResignation: 0
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "GroupTaxSekiScoring";
    Board board = Board::parseBoard(9,9,R"%%(
.x.xo.o.x
...xooox.
.xxxxxxoo
xoooooxo.
xo.o.oxoo
xoooooxxx
xxxo...oo
.xxxoooo.
.x.xo.o.o
)%%");
    Rules rules;
    rules.komi = 0.5f;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.multiStoneSuicideLegal = false;

    {
      rules.scoringRule = Rules::SCORING_AREA;
      rules.taxRule = Rules::TAX_NONE;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_AREA;
      rules.taxRule = Rules::TAX_SEKI;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_AREA;
      rules.taxRule = Rules::TAX_ALL;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_TERRITORY;
      rules.taxRule = Rules::TAX_NONE;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_TERRITORY;
      rules.taxRule = Rules::TAX_SEKI;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_TERRITORY;
      rules.taxRule = Rules::TAX_ALL;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }

    string expected = R"%%(
Winner: White
W-B Score: 4.5
isNoResult: 0
isResignation: 0
Winner: White
W-B Score: 6.5
isNoResult: 0
isResignation: 0
Winner: White
W-B Score: 6.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -1.5
isNoResult: 0
isResignation: 0
Winner: White
W-B Score: 0.5
isNoResult: 0
isResignation: 0
Winner: White
W-B Score: 0.5
isNoResult: 0
isResignation: 0
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "GroupTaxSekiScoring2";
    Board board = Board::parseBoard(9,9,R"%%(
.x.xo.o.x
...xooox.
.xxxxxxoo
xoooooxo.
xo.o.oxoo
xoooooxxx
xxxoxxxoo
.xxxoooo.
.x.xo.o.o
)%%");
    Rules rules;
    rules.komi = 0.5f;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.multiStoneSuicideLegal = false;

    {
      rules.scoringRule = Rules::SCORING_AREA;
      rules.taxRule = Rules::TAX_NONE;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_AREA;
      rules.taxRule = Rules::TAX_SEKI;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_AREA;
      rules.taxRule = Rules::TAX_ALL;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_TERRITORY;
      rules.taxRule = Rules::TAX_NONE;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_TERRITORY;
      rules.taxRule = Rules::TAX_SEKI;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }
    {
      rules.scoringRule = Rules::SCORING_TERRITORY;
      rules.taxRule = Rules::TAX_ALL;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      testAssert(hist.isGameFinished == true);
      printGameResult(out,hist);
    }

    string expected = R"%%(
Winner: White
W-B Score: 1.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -0.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -2.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -1.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -3.5
isNoResult: 0
isResignation: 0
Winner: Black
W-B Score: -5.5
isNoResult: 0
isResignation: 0
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "PreventEncore";
    Board board = Board::parseBoard(3,3,R"%%(
.x.
xxo
.o.
)%%");
    Rules rules;
    rules.komi = 0.5f;
    rules.koRule = Rules::KO_POSITIONAL;
    rules.multiStoneSuicideLegal = false;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.taxRule = Rules::TAX_NONE;

    {
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      hist.printDebugInfo(out,board);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      hist.printDebugInfo(out,board);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      hist.printDebugInfo(out,board);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
      hist.printDebugInfo(out,board);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
      hist.printDebugInfo(out,board);
      out << endl;
    }
    {
      out << "-----------------------" << endl;
      out << "Preventing encore" << endl;
      BoardHistory hist(board,P_BLACK,rules,0);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__, true);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__, true);
      hist.printDebugInfo(out,board);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__, true);
      hist.printDebugInfo(out,board);
      makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__, true);
      hist.printDebugInfo(out,board);
    }

    string expected = R"%%(
HASH: B7CA8ABCFA02190C5AA69D0C2777E4DC
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 1
Turns this phase 0
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass
HASH: B7CA8ABCFA02190C5AA69D0C2777E4DC
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 1
Turns this phase 1
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass
HASH: B7CA8ABCFA02190C5AA69D0C2777E4DC
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 2
Turns this phase 0
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass pass
HASH: B7CA8ABCFA02190C5AA69D0C2777E4DC
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 2
Turns this phase 1
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 0
Game result 0 Empty 0 0 0 0
Last moves pass pass pass pass pass
HASH: B7CA8ABCFA02190C5AA69D0C2777E4DC
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 2
Turns this phase 2
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 1 White 0.5 1 0 0
Last moves pass pass pass pass pass pass

-----------------------
Preventing encore
HASH: B7CA8ABCFA02190C5AA69D0C2777E4DC
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 0
Turns this phase 2
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 1
Game result 0 Empty 0 0 0 0
Last moves pass pass
HASH: B7CA8ABCFA02190C5AA69D0C2777E4DC
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 0
Turns this phase 3
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla White
Past normal phase end 1
Game result 0 Empty 0 0 0 0
Last moves pass pass pass
HASH: B7CA8ABCFA02190C5AA69D0C2777E4DC
   A B C
 3 . X .
 2 X X O
 1 . O .


Initial pla Black
Encore phase 0
Turns this phase 4
Rules koPOSITIONALscoreTERRITORYtaxNONEsui0komi0.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 1
Game result 0 Empty 0 0 0 0
Last moves pass pass pass pass
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Stress test on tiny boards";

    Rand baseRand("Tiny board stress test");
    auto stressTest = [&](Board board, BoardHistory hist, Player nextPla, bool prolongGame) {
      Rand rand(baseRand.nextUInt64());
      for(int i = 0; i<1000; i++) {
        int numLegal = 0;
        static constexpr int MAX_LEGAL_MOVES = Board::MAX_PLAY_SIZE + 1;
        Loc legalMoves[MAX_LEGAL_MOVES];
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
            out << PlayerIO::colorToChar(board.colors[Location::getLoc(x,y,board.x_size)]);
        out << " NP" << PlayerIO::colorToChar(nextPla);
        out << " PS" << hist.consecutiveEndingPasses;
        out << " E" << hist.encorePhase;
        out << " ";
        out << " ";
        for(int y = 0; y<board.y_size; y++)
          for(int x = 0; x<board.x_size; x++)
            out << (int)(hist.koRecapBlocked[Location::getLoc(x,y,board.x_size)]);
        out << " ";
        for(int y = 0; y<board.y_size; y++)
          for(int x = 0; x<board.x_size; x++)
            out << (int)(hist.secondEncoreStartColors[Location::getLoc(x,y,board.x_size)]);

        out << endl;

        if(hist.isGameFinished)
          break;

        testAssert(numLegal > 0);
        move = legalMoves[rand.nextUInt(numLegal)];
        if(prolongGame && move == Board::PASS_LOC)
          move = legalMoves[rand.nextUInt(numLegal)];
        makeMoveAssertLegal(hist, board, move, nextPla, __LINE__);
        nextPla = getOpp(nextPla);
      }
      out << "White bonus score " << hist.whiteBonusScore << endl;
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
    rules.taxRule = Rules::TAX_SEKI;
    rules.multiStoneSuicideLegal = false;
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules,0),P_BLACK,true);
    rules.multiStoneSuicideLegal = true;
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules,0),P_BLACK,true);
    expected = R"%%(
5 .... NPX PS0 E0  0000 0000
4 .X.. NPO PS0 E0  0000 0000
3 .X.O NPX PS0 E0  0000 0000
1 .XX. NPO PS0 E0  0000 0000
3 .XX. NPX PS1 E0  0000 0000
2 XXX. NPO PS0 E0  0000 0000
4 ...O NPX PS0 E0  0000 0000
4 ...O NPO PS1 E0  0000 0000
1 O..O NPX PS0 E0  0000 0000
3 O..O NPO PS1 E0  0000 0000
2 O.OO NPX PS0 E0  0000 0000
1 O.OO NPO PS1 E0  0000 0000
2 O.OO NPX PS0 E1  0000 0000
1 O.OO NPO PS1 E1  0000 0000
2 O.OO NPX PS0 E2  0000 2022
1 O.OO NPO PS1 E2  0000 2022
2 O.OO NPX PS2 E2  0000 2022
White bonus score -1
Winner: White
W-B Score: 2.5
isNoResult: 0
isResignation: 0
5 .... NPX PS0 E0  0000 0000
4 ..X. NPO PS0 E0  0000 0000
3 .OX. NPX PS0 E0  0000 0000
2 XOX. NPO PS0 E0  0000 0000
2 XOX. NPX PS1 E0  0000 0000
2 X.XX NPO PS0 E0  0000 0000
4 .O.. NPX PS0 E0  0000 0000
3 XO.. NPO PS0 E0  0000 0000
1 .OO. NPX PS0 E0  0000 0000
3 .OO. NPO PS1 E0  0000 0000
2 .OOO NPX PS0 E0  0000 0000
4 X... NPO PS0 E0  0000 0000
3 X..O NPX PS0 E0  0000 0000
2 XX.O NPO PS0 E0  0000 0000
3 ..OO NPX PS0 E0  0000 0000
2 .XOO NPO PS0 E0  0000 0000
2 .XOO NPX PS1 E0  0000 0000
3 XX.. NPO PS0 E0  0000 0000
2 XX.O NPX PS0 E0  0000 0000
2 XXX. NPO PS0 E0  0000 0000
4 ...O NPX PS0 E0  0000 0000
3 .X.O NPO PS0 E0  0000 0000
1 O..O NPX PS0 E0  0000 0000
3 O..O NPO PS1 E0  0000 0000
2 OO.O NPX PS0 E0  0000 0000
4 ..X. NPO PS0 E0  0000 0000
3 ..XO NPX PS0 E0  0000 0000
3 ..XO NPO PS1 E0  0000 0000
2 .OXO NPX PS0 E0  0000 0000
2 .OXO NPO PS1 E0  0000 0000
2 OO.O NPX PS0 E0  0000 0000
4 ..X. NPO PS0 E0  0000 0000
3 .OX. NPX PS0 E0  0000 0000
2 .OXX NPO PS0 E0  0000 0000
3 OO.. NPX PS0 E0  0000 0000
2 OOX. NPO PS0 E0  0000 0000
2 OO.O NPX PS0 E0  0000 0000
4 ..X. NPO PS0 E0  0000 0000
3 O.X. NPX PS0 E0  0000 0000
2 O.XX NPO PS0 E0  0000 0000
3 OO.. NPX PS0 E0  0000 0000
2 OO.X NPO PS0 E0  0000 0000
2 OOO. NPX PS0 E0  0000 0000
4 ...X NPO PS0 E0  0000 0000
3 O..X NPX PS0 E0  0000 0000
2 O.XX NPO PS0 E0  0000 0000
3 OO.. NPX PS0 E0  0000 0000
White bonus score -2
Winner: Empty
W-B Score: 0
isNoResult: 1
isResignation: 0
)%%";

    expect(name,out,expected);

    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    rules.taxRule = Rules::TAX_NONE;
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules,0),P_BLACK,false);
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules,0),P_BLACK,false);
    rules.multiStoneSuicideLegal = true;
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules,0),P_BLACK,false);
    stressTest(emptyBoard22,BoardHistory(emptyBoard22,P_BLACK,rules,0),P_BLACK,false);
    expected = R"%%(
5 .... NPX PS0 E0  0000 0000
5 .... NPO PS1 E0  0000 0000
5 .... NPX PS2 E0  0000 0000
White bonus score 0
Winner: White
W-B Score: 0.5
isNoResult: 0
isResignation: 0
5 .... NPX PS0 E0  0000 0000
5 .... NPO PS1 E0  0000 0000
4 O... NPX PS0 E0  0000 0000
3 O.X. NPO PS0 E0  0000 0000
2 OOX. NPX PS0 E0  0000 0000
2 OOX. NPO PS1 E0  0000 0000
2 OOX. NPX PS2 E0  0000 0000
White bonus score 0
Winner: White
W-B Score: 1.5
isNoResult: 0
isResignation: 0
5 .... NPX PS0 E0  0000 0000
4 .X.. NPO PS0 E0  0000 0000
3 .XO. NPX PS0 E0  0000 0000
2 .XOX NPO PS0 E0  0000 0000
2 .XOX NPX PS1 E0  0000 0000
2 .XOX NPO PS2 E0  0000 0000
White bonus score 0
Winner: Black
W-B Score: -0.5
isNoResult: 0
isResignation: 0
5 .... NPX PS0 E0  0000 0000
4 ...X NPO PS0 E0  0000 0000
3 ..OX NPX PS0 E0  0000 0000
3 ..OX NPO PS1 E0  0000 0000
3 ..OX NPX PS2 E0  0000 0000
White bonus score 0
Winner: White
W-B Score: 0.5
isNoResult: 0
isResignation: 0
)%%";

    expect(name,out,expected);

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
    rules.taxRule = Rules::TAX_SEKI;
    stressTest(koBoard71,BoardHistory(koBoard71,P_BLACK,rules,0),P_BLACK,true);

    expected = R"%%(
3 .O.OX.O NPX PS0 E0  0000000 0000000
1 .OX.X.O NPO PS0 E0  0000000 0000000
4 .OX.X.O NPX PS1 E0  0000000 0000000
2 .OXXX.O NPO PS0 E0  0000000 0000000
4 .O...OO NPX PS0 E0  0000000 0000000
6 .O..X.. NPO PS0 E0  0000000 0000000
4 .O..XO. NPX PS0 E0  0000000 0000000
3 .O.XXO. NPO PS0 E0  0000000 0000000
2 .O.XXO. NPX PS1 E0  0000000 0000000
3 .O.XX.X NPO PS0 E0  0000000 0000000
3 OO.XX.X NPX PS0 E0  0000000 0000000
4 ..XXX.X NPO PS0 E0  0000000 0000000
3 ..XXXO. NPX PS0 E0  0000000 0000000
2 .XXXXO. NPO PS0 E0  0000000 0000000
2 .XXXXO. NPX PS1 E0  0000000 0000000
1 .XXXX.X NPO PS0 E0  0000000 0000000
3 .XXXX.X NPX PS1 E0  0000000 0000000
2 .XXXXXX NPO PS0 E0  0000000 0000000
7 O...... NPX PS0 E0  0000000 0000000
5 O....X. NPO PS0 E0  0000000 0000000
5 O..O.X. NPX PS0 E0  0000000 0000000
3 .X.O.X. NPO PS0 E0  0000000 0000000
3 .X.OOX. NPX PS0 E0  0000000 0000000
3 XX.OOX. NPO PS0 E0  0000000 0000000
3 ..OOOX. NPX PS0 E0  0000000 0000000
3 X.OOOX. NPO PS0 E0  0000000 0000000
1 X.OOO.O NPX PS0 E0  0000000 0000000
3 X.OOO.O NPO PS1 E0  0000000 0000000
2 X.OOOOO NPX PS0 E0  0000000 0000000
2 X.OOOOO NPO PS1 E0  0000000 0000000
2 .OOOOOO NPX PS0 E0  0000000 0000000
1 .OOOOOO NPO PS1 E0  0000000 0000000
2 .OOOOOO NPX PS0 E1  0000000 0000000
7 X...... NPO PS0 E1  0000000 0000000
6 X..O... NPX PS0 E1  0000000 0000000
5 XX.O... NPO PS0 E1  0000000 0000000
1 XX.O.O. NPX PS0 E1  0000000 0000000
4 XX.O.O. NPO PS1 E1  0000000 0000000
3 ..OO.O. NPX PS0 E1  0000000 0000000
4 .XOO.O. NPO PS0 E1  0000000 0000000
2 O.OO.O. NPX PS0 E1  1000000 0000000
4 O.OO.O. NPO PS0 E1  0000000 0000000
2 OOOO.O. NPX PS0 E1  0000000 0000000
5 ....XO. NPO PS0 E1  0000000 0000000
4 ..O.XO. NPX PS0 E1  0000000 0000000
5 ..O.X.X NPO PS0 E1  0000001 0000000
4 O.O.X.X NPX PS0 E1  0000001 0000000
3 O.O.XXX NPO PS0 E1  0000001 0000000
2 O.O.XXX NPX PS1 E1  0000001 0000000
3 O.O.XXX NPO PS0 E2  0000000 2020111
2 OOO.XXX NPX PS0 E2  0000000 2020111
2 OOO.XXX NPO PS1 E2  0000000 2020111
4 OOOO... NPX PS0 E2  0000000 2020111
3 OOOO..X NPO PS0 E2  0000000 2020111
2 OOOO.O. NPX PS0 E2  0000000 2020111
5 ....XO. NPO PS0 E2  0000000 2020111
4 ...O.O. NPX PS0 E2  0000000 2020111
5 X..O.O. NPO PS0 E2  0000000 2020111
4 X..O.OO NPX PS0 E2  0000000 2020111
5 X..OX.. NPO PS0 E2  0000000 2020111
4 X..OX.O NPX PS0 E2  0000000 2020111
3 XX.OX.O NPO PS0 E2  0000000 2020111
4 ..OOX.O NPX PS0 E2  0000000 2020111
4 ..OOXX. NPO PS0 E2  0000000 2020111
5 ..OO..O NPX PS0 E2  0000000 2020111
4 ..OOX.O NPO PS0 E2  0000000 2020111
3 O.OOX.O NPX PS0 E2  0000000 2020111
2 O.OOXX. NPO PS0 E2  0000000 2020111
4 O.OO..O NPX PS0 E2  0000000 2020111
2 O.OOX.O NPO PS0 E2  0000000 2020111
3 O.OO.OO NPX PS0 E2  0000000 2020111
2 .XOO.OO NPO PS0 E2  0100000 2020111
2 .XOO.OO NPX PS0 E2  0000000 2020111
2 .XOO.OO NPO PS1 E2  0000000 2020111
3 O.OO.OO NPX PS0 E2  1000000 2020111
3 O.OO.OO NPO PS0 E2  0000000 2020111
2 OOOO.OO NPX PS0 E2  0000000 2020111
7 ....X.. NPO PS0 E2  0000000 2020111
6 ....X.O NPX PS0 E2  0000000 2020111
4 ...XX.O NPO PS0 E2  0000000 2020111
4 O..XX.O NPX PS0 E2  0000000 2020111
3 O..XXX. NPO PS0 E2  0000000 2020111
3 OO.XXX. NPX PS0 E2  0000000 2020111
2 OO.XXXX NPO PS0 E2  0000000 2020111
5 OOO.... NPX PS0 E2  0000000 2020111
3 OOO.X.. NPO PS0 E2  0000000 2020111
3 OOO.X.O NPX PS0 E2  0000000 2020111
1 OOO.XX. NPO PS0 E2  0000000 2020111
3 OOO.XX. NPX PS1 E2  0000000 2020111
2 OOO.XXX NPO PS0 E2  0000000 2020111
2 OOO.XXX NPX PS1 E2  0000000 2020111
4 ...XXXX NPO PS0 E2  0000000 2020111
1 .O.XXXX NPX PS0 E2  0000000 2020111
3 .O.XXXX NPO PS1 E2  0000000 2020111
5 .OO.... NPX PS0 E2  0000000 2020111
4 .OOX... NPO PS0 E2  0000000 2020111
3 .OO.O.. NPX PS0 E2  0000000 2020111
4 .OO.O.X NPO PS0 E2  0000000 2020111
1 .OOOO.X NPX PS0 E2  0000000 2020111
3 .OOOO.X NPO PS1 E2  0000000 2020111
1 .OOOOO. NPX PS0 E2  0000000 2020111
3 .OOOOO. NPO PS1 E2  0000000 2020111
2 OOOOOO. NPX PS0 E2  0000000 2020111
1 OOOOOO. NPO PS1 E2  0000000 2020111
2 OOOOOO. NPX PS2 E2  0000000 2020111
White bonus score -1
Winner: White
W-B Score: 1.5
isNoResult: 0
isResignation: 0

)%%";

    expect(name,out,expected);

    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    rules.taxRule = Rules::TAX_SEKI;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules,0),P_BLACK,true);
    expected = R"%%(
5 .... NPX PS0 E0  0000 0000
4 X... NPO PS0 E0  0000 0000
1 X.O. NPX PS0 E0  0000 0000
3 X.O. NPO PS1 E0  0000 0000
2 X.OO NPX PS0 E0  0000 0000
3 XX.. NPO PS0 E0  0000 0000
3 XX.. NPX PS1 E0  0000 0000
3 XX.. NPO PS0 E1  0000 0000
2 XX.O NPX PS0 E1  0000 0000
2 XXX. NPO PS0 E1  0000 0000
4 ...O NPX PS0 E1  0000 0000
3 X..O NPO PS0 E1  0000 0000
2 X.OO NPX PS0 E1  0000 0000
3 XX.. NPO PS0 E1  0000 0000
3 ..O. NPX PS0 E1  0000 0000
2 .XO. NPO PS0 E1  0000 0000
2 O.O. NPX PS0 E1  1000 0000
3 O.O. NPO PS0 E1  0000 0000
2 OOO. NPX PS0 E1  0000 0000
4 ...X NPO PS0 E1  0000 0000
3 ..O. NPX PS0 E1  0000 0000
3 X.O. NPO PS0 E1  0000 0000
1 .OO. NPX PS0 E1  0000 0000
3 .OO. NPO PS1 E1  0000 0000
2 OOO. NPX PS0 E1  0000 0000
1 OOO. NPO PS1 E1  0000 0000
2 OOO. NPX PS0 E2  0000 2220
1 OOO. NPO PS1 E2  0000 2220
2 OOO. NPX PS2 E2  0000 2220
White bonus score -3
Winner: White
W-B Score: 0.5
isNoResult: 0
isResignation: 0

)%%";

    expect(name,out,expected);

    rules.koRule = Rules::KO_SITUATIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    rules.taxRule = Rules::TAX_SEKI;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules,0),P_BLACK,true);
    expected = R"%%(

5 .... NPX PS0 E0  0000 0000
4 X... NPO PS0 E0  0000 0000
3 .O.. NPX PS0 E0  0000 0000
3 .O.X NPO PS0 E0  0000 0000
2 OO.X NPX PS0 E0  0000 0000
2 ..XX NPO PS0 E0  0000 0000
2 O.XX NPX PS0 E0  0000 0000
2 O.XX NPO PS1 E0  0000 0000
3 OO.. NPX PS0 E0  0000 0000
3 ..X. NPO PS0 E0  0000 0000
2 O.X. NPX PS0 E0  0000 0000
1 .XX. NPO PS0 E0  0000 0000
3 .XX. NPX PS1 E0  0000 0000
2 .XXX NPO PS0 E0  0000 0000
4 O... NPX PS0 E0  0000 0000
2 O..X NPO PS0 E0  0000 0000
2 O.O. NPX PS0 E0  0000 0000
1 .XO. NPO PS0 E0  0000 0000
2 .XO. NPX PS1 E0  0000 0000
1 .X.X NPO PS0 E0  0000 0000
2 .X.X NPX PS1 E0  0000 0000
2 .X.X NPO PS0 E1  0000 0000
2 .XO. NPX PS0 E1  0010 0000
2 .XO. NPO PS0 E1  0000 0000
2 O.O. NPX PS0 E1  1000 0000
3 O.O. NPO PS1 E1  1000 0000
2 O.OO NPX PS0 E1  1000 0000
3 .X.. NPO PS0 E1  0000 0000
3 .X.O NPX PS0 E1  0000 0000
2 XX.O NPO PS0 E1  0000 0000
3 ..OO NPX PS0 E1  0000 0000
2 X.OO NPO PS0 E1  0000 0000
2 .OOO NPX PS0 E1  0000 0000
4 X... NPO PS0 E1  0000 0000
1 X.O. NPX PS0 E1  0000 0000
3 X.O. NPO PS1 E1  0000 0000
1 .OO. NPX PS0 E1  0000 0000
3 .OO. NPO PS1 E1  0000 0000
2 .OOO NPX PS0 E1  0000 0000
4 X... NPO PS0 E1  0000 0000
3 X..O NPX PS0 E1  0000 0000
3 X..O NPO PS1 E1  0000 0000
2 X.OO NPX PS0 E1  0000 0000
3 XX.. NPO PS0 E1  0000 0000
3 ..O. NPX PS0 E1  0000 0000
1 .XO. NPO PS0 E1  0000 0000
2 .XO. NPX PS1 E1  0000 0000
2 .X.X NPO PS0 E1  0001 0000
3 .X.X NPX PS0 E1  0000 0000
2 XX.X NPO PS0 E1  0000 0000
1 XX.X NPX PS1 E1  0000 0000
2 XX.X NPO PS0 E2  0000 1101
1 XX.X NPX PS1 E2  0000 1101
2 XX.X NPO PS2 E2  0000 1101
White bonus score -1
Winner: Black
W-B Score: -3.5
isNoResult: 0
isResignation: 0

)%%";

    expect(name,out,expected);


    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    rules.taxRule = Rules::TAX_NONE;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules,0),P_BLACK,true);

    expected = R"%%(
5 .... NPX PS0 E0  0000 0000
4 ...X NPO PS0 E0  0000 0000
3 ..O. NPX PS0 E0  0000 0000
2 .XO. NPO PS0 E0  0000 0000
1 O.O. NPX PS0 E0  0000 0000
3 O.O. NPO PS1 E0  0000 0000
2 O.OO NPX PS0 E0  0000 0000
3 .X.. NPO PS0 E0  0000 0000
2 .XO. NPX PS0 E0  0000 0000
1 .X.X NPO PS0 E0  0000 0000
3 .X.X NPX PS1 E0  0000 0000
2 .XXX NPO PS0 E0  0000 0000
4 O... NPX PS0 E0  0000 0000
4 O... NPO PS1 E0  0000 0000
3 OO.. NPX PS0 E0  0000 0000
3 ..X. NPO PS0 E0  0000 0000
2 .OX. NPX PS0 E0  0000 0000
1 X.X. NPO PS0 E0  0000 0000
3 X.X. NPX PS1 E0  0000 0000
2 X.XX NPO PS0 E0  0000 0000
3 .O.. NPX PS0 E0  0000 0000
3 .O.X NPO PS0 E0  0000 0000
1 .OO. NPX PS0 E0  0000 0000
3 .OO. NPO PS1 E0  0000 0000
2 OOO. NPX PS0 E0  0000 0000
4 ...X NPO PS0 E0  0000 0000
3 O..X NPX PS0 E0  0000 0000
2 O.XX NPO PS0 E0  0000 0000
3 OO.. NPX PS0 E0  0000 0000
2 OO.X NPO PS0 E0  0000 0000
2 OO.X NPX PS1 E0  0000 0000
3 ..XX NPO PS0 E0  0000 0000
2 O.XX NPX PS0 E0  0000 0000
2 O.XX NPO PS1 E0  0000 0000
3 OO.. NPX PS0 E0  0000 0000
2 OO.X NPO PS0 E0  0000 0000
2 OO.X NPX PS1 E0  0000 0000
White bonus score 0
Winner: White
W-B Score: 1.5
isNoResult: 0
isResignation: 0

)%%";

    expect(name,out,expected);

    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    rules.taxRule = Rules::TAX_NONE;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules,0),P_BLACK,true);
    expected = R"%%(
5 .... NPX PS0 E0  0000 0000
4 ...X NPO PS0 E0  0000 0000
1 .O.X NPX PS0 E0  0000 0000
3 .O.X NPO PS1 E0  0000 0000
2 OO.X NPX PS0 E0  0000 0000
3 ..XX NPO PS0 E0  0000 0000
2 .O.. NPX PS0 E0  0000 0000
4 .O.. NPO PS1 E0  0000 0000
2 OO.. NPX PS0 E0  0000 0000
3 OO.. NPO PS1 E0  0000 0000
1 OOO. NPX PS0 E0  0000 0000
1 OOO. NPO PS1 E0  0000 0000
1 OOO. NPX PS2 E0  0000 0000
White bonus score 0
Winner: White
W-B Score: 4.5
isNoResult: 0
isResignation: 0

)%%";

    expect(name,out,expected);

    rules.koRule = Rules::KO_SITUATIONAL;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    rules.taxRule = Rules::TAX_NONE;
    stressTest(koBoard41,BoardHistory(koBoard41,P_BLACK,rules,0),P_BLACK,true);
    expected = R"%%(
5 .... NPX PS0 E0  0000 0000
4 X... NPO PS0 E0  0000 0000
3 .O.. NPX PS0 E0  0000 0000
3 .O.X NPO PS0 E0  0000 0000
1 .OO. NPX PS0 E0  0000 0000
3 .OO. NPO PS1 E0  0000 0000
1 .OOO NPX PS0 E0  0000 0000
1 .OOO NPO PS1 E0  0000 0000
1 .OOO NPX PS2 E0  0000 0000
White bonus score 0
Winner: White
W-B Score: 4.5
isNoResult: 0
isResignation: 0

)%%";

    expect(name,out,expected);


    rules.koRule = Rules::KO_POSITIONAL;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    rules.taxRule = Rules::TAX_SEKI;
    baseRand.init("123");
    stressTest(koBoard71,BoardHistory(koBoard71,P_BLACK,rules,0),P_BLACK,false);
    expected = R"%%(
3 .O.OX.O NPX PS0 E0  0000000 0000000
1 .OX.X.O NPO PS0 E0  0000000 0000000
4 .OX.X.O NPX PS1 E0  0000000 0000000
2 .OX.X.O NPO PS0 E1  0000000 0000000
3 .O.OX.O NPX PS0 E1  0001000 0000000
4 .O.OX.O NPO PS1 E1  0001000 0000000
3 .OOOX.O NPX PS0 E1  0001000 0000000
2 .OOOXX. NPO PS0 E1  0001000 0000000
2 .OOOXX. NPX PS1 E1  0001000 0000000
2 .OOOXX. NPO PS0 E2  0000000 0222110
2 .OOOXX. NPX PS1 E2  0000000 0222110
4 X...XX. NPO PS0 E2  0000000 0222110
3 X.O.XX. NPX PS0 E2  0000000 0222110
3 X.O.XXX NPO PS0 E2  0000000 0222110
1 X.O.XXX NPX PS1 E2  0000000 0222110
3 X.O.XXX NPO PS2 E2  0000000 0222110
White bonus score -2
Winner: Black
W-B Score: -2.5
isNoResult: 0
isResignation: 0

)%%";

    expect(name,out,expected);
  }


  {
    const char* name = "Board history clearing directly to the encore";
    Board board = Board::parseBoard(4,4,R"%%(
..o.
.o.o
.xox
..xx
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = true;
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_BLACK,rules,0);
    BoardHistory hist2(board,P_BLACK,rules,0);

    auto compareHists = [&]() {
      out << hist.moveHistory.size() << " " << hist2.moveHistory.size() << endl;
      out << hist.koHashHistory.size() << " " << hist2.koHashHistory.size() << endl;
      out << hist.koHashHistory[0] << " " << hist2.koHashHistory[0] << endl;
      out << hist.firstTurnIdxWithKoHistory << " " << hist2.firstTurnIdxWithKoHistory << endl;
      out << hist.getRecentBoard(0).pos_hash <<  " " << hist2.getRecentBoard(0).pos_hash << endl;
      out << hist.getRecentBoard(1).pos_hash <<  " " << hist2.getRecentBoard(1).pos_hash << endl;
      out << hist.getRecentBoard(2).pos_hash <<  " " << hist2.getRecentBoard(2).pos_hash << endl;
      out << hist.getRecentBoard(3).pos_hash <<  " " << hist2.getRecentBoard(3).pos_hash << endl;
      out << hist.getRecentBoard(4).pos_hash <<  " " << hist2.getRecentBoard(4).pos_hash << endl;
      out << hist.getRecentBoard(5).pos_hash <<  " " << hist2.getRecentBoard(5).pos_hash << endl;

      for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
        testAssert(hist.wasEverOccupiedOrPlayed[i] == hist2.wasEverOccupiedOrPlayed[i]);
      for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
        testAssert(hist.superKoBanned[i] == false);
      for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
        testAssert(hist2.superKoBanned[i] == false);

      out << hist.consecutiveEndingPasses << " " << hist2.consecutiveEndingPasses << endl;
      out << hist.hashesBeforeBlackPass.size() << " " << hist2.hashesBeforeBlackPass.size() << endl;
      out << hist.hashesBeforeWhitePass.size() << " " << hist2.hashesBeforeWhitePass.size() << endl;
      out << hist.encorePhase << " " << hist2.encorePhase << endl;

      for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
        testAssert(hist.koRecapBlocked[i] == false);
      for(int i = 0; i<Board::MAX_ARR_SIZE; i++)
        testAssert(hist2.koRecapBlocked[i] == false);

      out << hist.koRecapBlockHash << " " << hist2.koRecapBlockHash << endl;
      out << hist.koCapturesInEncore.size() << " " << hist2.koCapturesInEncore.size() << endl;

      for(int y = 0; y<board.y_size; y++)
        for(int x = 0; x<board.x_size; x++)
          out << (int)(hist.secondEncoreStartColors[Location::getLoc(x,y,board.x_size)]);
      out << endl;
      for(int y = 0; y<board.y_size; y++)
        for(int x = 0; x<board.x_size; x++)
          out << (int)(hist2.secondEncoreStartColors[Location::getLoc(x,y,board.x_size)]);
      out << endl;

      out << hist.whiteBonusScore << " " << hist2.whiteBonusScore << endl;
      out << hist.isGameFinished << " " << hist2.isGameFinished << endl;
      out << (int)hist.winner << " " << (int)hist2.winner << endl;
      out << hist.finalWhiteMinusBlackScore << " " << hist2.finalWhiteMinusBlackScore << endl;
      out << hist.isNoResult << " " << hist2.isNoResult << endl;

    };

    Board copy = board;
    makeMoveAssertLegal(hist, copy, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, copy, Board::PASS_LOC, P_WHITE, __LINE__);

    hist2.clear(board, P_BLACK, hist2.rules, 1);

    compareHists();
    string expected = R"%%(

2 0
1 1
F43A55D89EAFC93CA62848648DA051CF F43A55D89EAFC93CA62848648DA051CF
2 0
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
0 0
0 0
0 0
1 1
00000000000000000000000000000000 00000000000000000000000000000000
0 0
0000000000000000
0000000000000000
0 0
0 0
0 0
0 0
0 0

)%%";
    expect(name,out,expected);

    makeMoveAssertLegal(hist, copy, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, copy, Board::PASS_LOC, P_WHITE, __LINE__);

    hist2.clear(board, P_BLACK, hist2.rules, 2);

    compareHists();
    expected = R"%%(

4 0
1 1
F43A55D89EAFC93CA62848648DA051CF F43A55D89EAFC93CA62848648DA051CF
4 0
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
D314459C37E7C630DCB23301AE1B492C D314459C37E7C630DCB23301AE1B492C
0 0
0 0
0 0
2 2
00000000000000000000000000000000 00000000000000000000000000000000
0 0
0020020201210011
0020020201210011
0 0
0 0
0 0
0 0
0 0

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Test case failing in search before";
    Board board = Board::parseBoard(9,9,R"%%(
XXXXXXXXX
X.OXXXXXX
XXXXOXXXX
XXX.OOXX.
OXXXOOXXX
.OXXXXXXO
XXXX.XOOO
XXXOXOOOO
XXXOO.OOO
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f;
    rules.multiStoneSuicideLegal = false;
    rules.taxRule = Rules::TAX_SEKI;
    BoardHistory hist(board,P_BLACK,rules,0);

    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);

    testAssert(hist.encorePhase == 1);
    makeMoveAssertLegal(hist, board, Location::getLoc(8,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,6,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,7,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,7,board.x_size), P_WHITE, __LINE__);
    out << board << endl;

    string expected = R"%%(
HASH: C377EB251DBAB5E2F6C1BABE18EEE392
   A B C D E F G H J
 9 X X X X X X X X X
 8 X . O X X X X X X
 7 X X X X O X X X X
 6 X X X . O O X X X
 5 O X X X O O X X X
 4 . O X X X X X X O
 3 X X X X O X O O O
 2 X X X O O O O O O
 1 X X X O O . O O O
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Test basic game";

    string sgfStr = "(;FF[4]GM[1]SZ[12]PB[b6c96-s49543680-d12165287]PW[b6c96-s50529536-d12424600]HA[0]KM[7.5]RU[koSIMPLEscoreTERRITORYtaxSEKIsui1]RE[B+1.5];B[di];W[ii];B[dd];W[id];B[gj];W[fc];B[jg];W[hh];B[jj];W[ji];B[ij];W[if];B[ec];W[fd];B[cf];W[cj];B[ci];W[dj];B[ej];W[ek];B[fk];W[ei];B[fj];W[bi];B[bh];W[bj];B[dk];W[cc];B[fb];W[ck];B[cd];W[gb];B[el];W[eb];B[db];W[fa];B[ki];W[kh];B[kj];W[kg];B[jf];W[je];B[eg];W[cb];B[dc];W[da];B[bc];W[bb];B[bd];W[ef];B[fg];W[dg];B[cg];W[df];B[dh];W[ff];B[gg];W[eh];B[ch];W[gf];B[gh];W[gi];B[fi];W[hi];B[jh];W[kf];B[hg];W[ig];B[ab];W[hf];B[fh];W[ca];B[de];W[li];B[lj];W[lh];B[ee];W[fe];B[hj];W[ih];B[aa];W[ed];B[ac];W[];B[ba];W[ea];B[];W[];B[];W[];B[ai];W[];B[cl];W[bl];B[ak];W[];B[aj];W[];B[bk];W[];B[cj];W[jb];B[];W[])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    BoardHistory hist;
    Player nextPla = P_BLACK;
    int turnNumberToSetup = sgf->moves.size();
    Rules initialRules = sgf->getRulesOrFailAllowUnspecified(Rules());

    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnNumberToSetup);
    string expected = R"%%(
HASH: 328F99331CC19E0AF700EB536A83D1F4
   A B C D E F G H J K L M
12 X X O O O O . . . . . .
11 X O O X O . O . . O . .
10 X X O X X O . . . . . .
 9 . X X X O O . . O . . .
 8 . . . X X O . . . O . .
 7 . . X O O O O O O . O .
 6 . . X O X X X X O . O .
 5 . X X X . X X O O . O O
 4 X . X X . X O O O O X O
 3 X . X . X X X X X X X X
 2 X X . X . X . . . . . .
 1 . O X . X . . . . . . .


Initial pla Black
Encore phase 2
Turns this phase 14
Rules koSIMPLEscoreTERRITORYtaxSEKIsui1komi7.5
Ko recap block hash 00000000000000000000000000000000
White bonus score 1
White handicap bonus score 0
Has button 0
Presumed next pla Black
Past normal phase end 0
Game result 1 Black -1.5 1 0 0
Last moves D4 J4 D9 J9 G3 F10 K6 H5 K3 K4 J3 J7 E10 F9 C7 C3 C4 D3 E3 E2 F2 E4 F3 B4 B5 B3 D2 C10 F11 C2 C9 G11 E1 E11 D11 F12 L4 L5 L3 L6 K7 K8 E6 C11 D10 D12 B10 B11 B9 E7 F6 D6 C6 D7 D5 F7 G6 E5 C5 G7 G5 G4 F4 H4 K5 L7 H6 J6 A11 H7 F5 C12 D8 M4 M3 M5 E8 F8 H3 J5 A12 E9 A10 pass B12 E12 pass pass pass pass A4 pass C1 B1 A2 pass A3 pass B2 pass C3 K11 pass pass
XXOOOOOOOOOO
XOOXOOOOOOOO
XXOXXOOOOOOO
XXXXOOOOOOOO
XXXXXOOOOOOO
XXXOOOOOOOOO
XXXOXXXXOOOO
XXXXXXXOOOOO
XXXXXXOOOOXO
XXXXXXXXXXXX
XXXXXXXXXXXX
XXXXXXXXXXXX
)%%";
    hist.printDebugInfo(out,board);

    Color area[Board::MAX_ARR_SIZE];
    hist.endAndScoreGameNow(board,area);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        out << PlayerIO::colorToChar(area[loc]);
      }
      out << endl;
    }
    out << endl;

    expect(name,out,expected);
  }

  {
    const char* name = "Sending two returning one initial pass WITHOUT button, simple";

    Board board = Board::parseBoard(5,6,R"%%(
.....
..xxx
xx.oo
xooo.
xo.ox
xoxx.
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.taxRule = Rules::TAX_SEKI;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 0.0f;
    rules.hasButton = false;
    BoardHistory hist(board,P_BLACK,rules,0);

    makeMoveAssertLegal(hist, board, Location::getLoc(2,2,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,5,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    testAssert(!hist.isGameFinished);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.isGameFinished);
    printGameResult(out,hist);
    out << endl;
    string expected = R"%%(
Winner: Black
W-B Score: -11
isNoResult: 0
isResignation: 0
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Sending two returning one initial pass WITH button, simple";

    Board board = Board::parseBoard(5,6,R"%%(
.....
..xxx
xx.oo
xooo.
xo.ox
xoxx.
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SIMPLE;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.taxRule = Rules::TAX_SEKI;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 0.5f;
    rules.hasButton = true;
    BoardHistory hist(board,P_BLACK,rules,0);

    makeMoveAssertLegal(hist, board, Location::getLoc(2,2,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,5,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    testAssert(!hist.isGameFinished);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(!hist.isGameFinished);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,5,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    testAssert(!hist.isGameFinished);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    testAssert(hist.isGameFinished);
    printGameResult(out,hist);
    out << endl;
    string expected = R"%%(
Winner: Black
W-B Score: -10
isNoResult: 0
isResignation: 0
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Sending two returning one NO initial pass WITHOUT button, SSK";

    Board board = Board::parseBoard(5,6,R"%%(
.....
..xxx
xxxoo
xooo.
xo.ox
xoxx.
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.taxRule = Rules::TAX_SEKI;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 0.0f;
    rules.hasButton = false;
    BoardHistory hist(board,P_BLACK,rules,0);

    makeMoveAssertLegal(hist, board, Location::getLoc(4,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,5,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    printIllegalMoves(out,board,hist,P_BLACK);
    testAssert(!hist.isGameFinished);
    out << endl;
    string expected = R"%%(
Illegal: (4,3) X
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Sending two returning one NO initial pass WITH button, SSK";

    Board board = Board::parseBoard(5,6,R"%%(
.....
..xxx
xxxoo
xooo.
xo.ox
xoxx.
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.taxRule = Rules::TAX_SEKI;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 0.5f;
    rules.hasButton = true;
    BoardHistory hist(board,P_BLACK,rules,0);

    makeMoveAssertLegal(hist, board, Location::getLoc(4,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,5,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    printIllegalMoves(out,board,hist,P_BLACK);
    testAssert(!hist.isGameFinished);
    out << "--" << endl;
    testAssert(!hist.isGameFinished);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,5,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    printIllegalMoves(out,board,hist,P_BLACK);
    out << endl;
    string expected = R"%%(
--
Illegal: (4,3) X
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Sending two returning one initial pass WITH button, SSK";

    Board board = Board::parseBoard(5,6,R"%%(
.....
..xxx
xx.oo
xooo.
xo.ox
xoxx.
)%%");
    Rules rules;
    rules.koRule = Rules::KO_SITUATIONAL;
    rules.scoringRule = Rules::SCORING_AREA;
    rules.taxRule = Rules::TAX_SEKI;
    rules.multiStoneSuicideLegal = false;
    rules.komi = 0.5f;
    rules.hasButton = true;
    BoardHistory hist(board,P_BLACK,rules,0);

    makeMoveAssertLegal(hist, board, Location::getLoc(2,2,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,3,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,5,board.x_size), P_WHITE, __LINE__);
    makeMoveAssertLegal(hist, board, Location::getLoc(4,4,board.x_size), P_BLACK, __LINE__);
    makeMoveAssertLegal(hist, board, Board::PASS_LOC, P_WHITE, __LINE__);
    printIllegalMoves(out,board,hist,P_BLACK);
    testAssert(!hist.isGameFinished);
    out << endl;
    string expected = R"%%(
Illegal: (4,3) X
)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "Brute force testing of ko hash table";
    vector<Rules> rules = {
      Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, 1.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, 3.0f),
      Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_N, 5.5f),
      Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, 1.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, true, false, Rules::WHB_N_MINUS_ONE, 4.5f),
      Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, true, Rules::WHB_ZERO, 6.5f),
      Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, 2.0f),
      Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, true, Rules::WHB_N_MINUS_ONE, 5.5f),
      Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, true, false, Rules::WHB_N, 2.5f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, 3.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, 4.0f),
      Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, false, Rules::WHB_N, 6.5f),
    };
    Rand baseRand(name);

    constexpr int numBoards = 6;
    Board boards[numBoards] = {
      Board(2,2),
      Board(5,1),
      Board(6,1),
      Board(2,3),
      Board(4,2),
    };
    for(int i = 0; i<numBoards; i++) {
      for(int j = 0; j<rules.size(); j++) {
        Player nextPla = P_BLACK;
        Board board = boards[i];
        BoardHistory hist(board,nextPla,rules[j],0);
        Board board2 = boards[i];
        BoardHistory hist2(board2,nextPla,rules[j],0);
        KoHashTable* table = new KoHashTable();

        Rand rand(baseRand.nextUInt64());
        int k;
        for(k = 0; k<300; k++) {
          int numLegal = 0;
          static constexpr int MAX_LEGAL_MOVES = Board::MAX_PLAY_SIZE + 1;
          Loc legalMoves[MAX_LEGAL_MOVES];
          for(int y = 0; y<board.y_size; y++) {
            for(int x = 0; x<board.x_size; x++) {
              Loc move = Location::getLoc(x,y,board.x_size);
              bool isLegal = hist.isLegal(board, move, nextPla);
              bool isLegal2 = hist2.isLegal(board2, move, nextPla);
              testAssert(isLegal == isLegal2);
              if(isLegal) legalMoves[numLegal++] = move;
            }
          }
          {
            Loc move = Board::PASS_LOC;
            bool isLegal = hist.isLegal(board, move, nextPla);
            bool isLegal2 = hist2.isLegal(board2, move, nextPla);
            testAssert(isLegal == isLegal2);
            if(isLegal) legalMoves[numLegal++] = move;
          }
          if(hist.isGameFinished)
            break;

          testAssert(numLegal > 0);
          Loc move = legalMoves[rand.nextUInt(numLegal)];
          if(move == Board::PASS_LOC)
            move = legalMoves[rand.nextUInt(numLegal)];
          if(move == Board::PASS_LOC)
            move = legalMoves[rand.nextUInt(numLegal)];
          makeMoveAssertLegal(hist, board, move, nextPla, __LINE__);
          makeMoveAssertLegal(hist2, board2, move, nextPla, __LINE__, table);
          nextPla = getOpp(nextPla);

          bool justRecomputed = false;
          if(rand.nextBool(0.10)) {
            table->recompute(hist2);
            justRecomputed = true;
          }

          for(int m = 0; m < hist.koHashHistory.size(); m++) {
            Hash128 hash = hist.koHashHistory[m];
            testAssert(
              hist.numberOfKoHashOccurrencesInHistory(hash,NULL) ==
              hist2.numberOfKoHashOccurrencesInHistory(hash,table)
            );
            if(justRecomputed) {
              testAssert(
                hist.numberOfKoHashOccurrencesInHistory(hash,NULL) ==
                table->numberOfOccurrencesOfHash(hash)
              );
            }
          }
        }
        delete table;
      }
    }
  }

  {
    //const char* name = "Test some roundtripping of rules strings";
    vector<Rules> rules = {
      Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, 1.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, 3.0f),
      Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, 5.5f),
      Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, 1.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, true, false, Rules::WHB_N, 4.5f),
      Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, true, Rules::WHB_N, 6.5f),
      Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, 2.0f),
      Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, true, Rules::WHB_ZERO, 5.5f),
      Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, true, false, Rules::WHB_N_MINUS_ONE, 2.5f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, 3.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, true, Rules::WHB_N_MINUS_ONE, 4.0f),
      Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, false, Rules::WHB_N, 6.5f),
    };

    for(int i = 0; i<rules.size(); i++) {
      bool suc;

      Rules parsed;
      suc = Rules::tryParseRules(rules[i].toString(),parsed);
      testAssert(suc);
      testAssert(rules[i] == parsed);

      Rules parsed2;
      suc = Rules::tryParseRulesWithoutKomi(rules[i].toStringNoKomi(),parsed2,rules[i].komi);
      testAssert(suc);
      testAssert(rules[i] == parsed2);

      Rules parsed3;
      suc = Rules::tryParseRules(rules[i].toJsonString(),parsed3);
      testAssert(suc);
      testAssert(rules[i] == parsed3);

      Rules parsed4;
      suc = Rules::tryParseRulesWithoutKomi(rules[i].toJsonStringNoKomi(),parsed4,rules[i].komi);
      testAssert(suc);
      testAssert(rules[i] == parsed4);
    }
  }

}
