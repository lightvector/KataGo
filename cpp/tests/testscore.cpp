
#include "../neuralnet/nninputs.h"

#include "../tests/tests.h"
using namespace TestCommon;

void Tests::runScoreTests() {
  cout << "Running score and utility tests" << endl;
  ostringstream out;

  {
    const char* name = "On-board even 9x9, komi 7.5";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    BoardHistory hist(board,P_BLACK,rules);
    hist.endAndScoreGameNow(board);

    out << "Black self komi wdu 0: " << hist.currentSelfKomi(P_BLACK, 0.0) << endl;
    out << "White self komi wdu 0: " << hist.currentSelfKomi(P_WHITE, 0.0) << endl;
    out << "Black self komi wdu -0.5: " << hist.currentSelfKomi(P_BLACK, -0.5) << endl;
    out << "White self komi wdu -0.5: " << hist.currentSelfKomi(P_WHITE, -0.5) << endl;
    out << "Black self komi wdu 0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wdu 0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;

    out << "Winner: " << colorToChar(hist.winner) << endl;
    out << "Final score: " << hist.finalWhiteMinusBlackScore << endl;
    out << "WL Util wdu 0: " << NNOutput::whiteValueOfWinner(hist.winner, 0.0) << endl;
    out << "Score Util wdu 0: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, 0.0, board, hist) << endl;
    out << "WL Util wdu -0.4: " << NNOutput::whiteValueOfWinner(hist.winner, -0.4) << endl;
    out << "Score Util wdu -0.4: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, -0.4, board, hist) << endl;
    out << "WL Util wdu 0.4: " << NNOutput::whiteValueOfWinner(hist.winner, 0.4) << endl;
    out << "Score Util wdu 0.4: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, 0.4, board, hist) << endl;

    string expected = R"%%(
Black self komi wdu 0: -7.5
White self komi wdu 0: 7.5
Black self komi wdu -0.5: -7.5
White self komi wdu -0.5: 7.5
Black self komi wdu 0.5: -7.5
White self komi wdu 0.5: 7.5
Winner: O
Final score: 7.5
WL Util wdu 0: 1
Score Util wdu 0: 0.394119
WL Util wdu -0.4: 1
Score Util wdu -0.4: 0.394119
WL Util wdu 0.4: 1
Score Util wdu 0.4: 0.394119
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "On-board even 9x9, komi 7";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules);
    hist.endAndScoreGameNow(board);

    out << "Black self komi wdu 0: " << hist.currentSelfKomi(P_BLACK, 0.0) << endl;
    out << "White self komi wdu 0: " << hist.currentSelfKomi(P_WHITE, 0.0) << endl;
    out << "Black self komi wdu -0.5: " << hist.currentSelfKomi(P_BLACK, -0.5) << endl;
    out << "White self komi wdu -0.5: " << hist.currentSelfKomi(P_WHITE, -0.5) << endl;
    out << "Black self komi wdu 0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wdu 0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;

    out << "Winner: " << colorToChar(hist.winner) << endl;
    out << "Final score: " << hist.finalWhiteMinusBlackScore << endl;
    out << "WL Util wdu 0: " << NNOutput::whiteValueOfWinner(hist.winner, 0.0) << endl;
    out << "Score Util wdu 0: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, 0.0, board, hist) << endl;
    out << "WL Util wdu -0.4: " << NNOutput::whiteValueOfWinner(hist.winner, -0.4) << endl;
    out << "Score Util wdu -0.4: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, -0.4, board, hist) << endl;
    out << "WL Util wdu 0.4: " << NNOutput::whiteValueOfWinner(hist.winner, 0.4) << endl;
    out << "Score Util wdu 0.4: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, 0.4, board, hist) << endl;

    string expected = R"%%(
Black self komi wdu 0: -7
White self komi wdu 0: 7
Black self komi wdu -0.5: -6.75
White self komi wdu -0.5: 6.75
Black self komi wdu 0.5: -7.25
White self komi wdu 0.5: 7.25
Winner: O
Final score: 7
WL Util wdu 0: 1
Score Util wdu 0: 0.370402
WL Util wdu -0.4: 1
Score Util wdu -0.4: 0.360776
WL Util wdu 0.4: 1
Score Util wdu 0.4: 0.379949
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "On-board black ahead 7 9x9, komi 7";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
xxxxxxx..
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules);
    hist.endAndScoreGameNow(board);

    out << "Black self komi wdu 0: " << hist.currentSelfKomi(P_BLACK, 0.0) << endl;
    out << "White self komi wdu 0: " << hist.currentSelfKomi(P_WHITE, 0.0) << endl;
    out << "Black self komi wdu -0.5: " << hist.currentSelfKomi(P_BLACK, -0.5) << endl;
    out << "White self komi wdu -0.5: " << hist.currentSelfKomi(P_WHITE, -0.5) << endl;
    out << "Black self komi wdu 0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wdu 0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;

    out << "Winner: " << colorToChar(hist.winner) << endl;
    out << "Final score: " << hist.finalWhiteMinusBlackScore << endl;
    out << "WL Util wdu 0: " << NNOutput::whiteValueOfWinner(hist.winner, 0.0) << endl;
    out << "Score Util wdu 0: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, 0.0, board, hist) << endl;
    out << "WL Util wdu -0.4: " << NNOutput::whiteValueOfWinner(hist.winner, -0.4) << endl;
    out << "Score Util wdu -0.4: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, -0.4, board, hist) << endl;
    out << "WL Util wdu 0.4: " << NNOutput::whiteValueOfWinner(hist.winner, 0.4) << endl;
    out << "Score Util wdu 0.4: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, 0.4, board, hist) << endl;

    string expected = R"%%(
Black self komi wdu 0: -7
White self komi wdu 0: 7
Black self komi wdu -0.5: -6.75
White self komi wdu -0.5: 6.75
Black self komi wdu 0.5: -7.25
White self komi wdu 0.5: 7.25
Winner: .
Final score: 0
WL Util wdu 0: 0
Score Util wdu 0: 0
WL Util wdu -0.4: -0.4
Score Util wdu -0.4: -0.0111107
WL Util wdu 0.4: 0.4
Score Util wdu 0.4: 0.0111107
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }


  {
    const char* name = "On-board even 5x5, komi 7";

    Board board = Board::parseBoard(5,5,R"%%(
.....
ooooo
.....
xxxxx
.....
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules);
    hist.endAndScoreGameNow(board);

    out << "Black self komi wdu 0: " << hist.currentSelfKomi(P_BLACK, 0.0) << endl;
    out << "White self komi wdu 0: " << hist.currentSelfKomi(P_WHITE, 0.0) << endl;
    out << "Black self komi wdu -0.5: " << hist.currentSelfKomi(P_BLACK, -0.5) << endl;
    out << "White self komi wdu -0.5: " << hist.currentSelfKomi(P_WHITE, -0.5) << endl;
    out << "Black self komi wdu 0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wdu 0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;

    out << "Winner: " << colorToChar(hist.winner) << endl;
    out << "Final score: " << hist.finalWhiteMinusBlackScore << endl;
    out << "WL Util wdu 0: " << NNOutput::whiteValueOfWinner(hist.winner, 0.0) << endl;
    out << "Score Util wdu 0: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, 0.0, board, hist) << endl;
    out << "WL Util wdu -0.4: " << NNOutput::whiteValueOfWinner(hist.winner, -0.4) << endl;
    out << "Score Util wdu -0.4: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, -0.4, board, hist) << endl;
    out << "WL Util wdu 0.4: " << NNOutput::whiteValueOfWinner(hist.winner, 0.4) << endl;
    out << "Score Util wdu 0.4: " << NNOutput::whiteValueOfScore(hist.finalWhiteMinusBlackScore, 0.4, board, hist) << endl;

    string expected = R"%%(
Black self komi wdu 0: -7
White self komi wdu 0: 7
Black self komi wdu -0.5: -6.75
White self komi wdu -0.5: 6.75
Black self komi wdu 0.5: -7.25
White self komi wdu 0.5: 7.25
Winner: O
Final score: 7
WL Util wdu 0: 1
Score Util wdu 0: 0.604368
WL Util wdu -0.4: 1
Score Util wdu -0.4: 0.591519
WL Util wdu 0.4: 1
Score Util wdu 0.4: 0.616909
)%%";
    expect(name,out.str(),expected);
    out.str("");
    out.clear();
  }

}
