
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
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    out << "Black self komi wins/draw=0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wins/draw=0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;
    out << "Black self komi wins/draw=0.25: " << hist.currentSelfKomi(P_BLACK, 0.25) << endl;
    out << "White self komi wins/draw=0.25: " << hist.currentSelfKomi(P_WHITE, 0.25) << endl;
    out << "Black self komi wins/draw=0.75: " << hist.currentSelfKomi(P_BLACK, 0.75) << endl;
    out << "White self komi wins/draw=0.75: " << hist.currentSelfKomi(P_WHITE, 0.75) << endl;

    out << "Winner: " << colorToChar(hist.winner) << endl;
    out << "Final score: " << hist.finalWhiteMinusBlackScore << endl;
    out << "WL Wins wins/draw=0.5: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.5) << endl;
    out << "Score Util wins/draw=0.5: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.5, board, hist) << endl;
    out << "WL Wins wins/draw=0.3: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.3) << endl;
    out << "Score Util wins/draw=0.3: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.3, board, hist) << endl;
    out << "WL Wins wins/draw=0.7: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.7) << endl;
    out << "Score Util wins/draw=0.7: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.7, board, hist) << endl;

    string expected = R"%%(
Black self komi wins/draw=0.5: -7.5
White self komi wins/draw=0.5: 7.5
Black self komi wins/draw=0.25: -7.5
White self komi wins/draw=0.25: 7.5
Black self komi wins/draw=0.75: -7.5
White self komi wins/draw=0.75: 7.5
Winner: O
Final score: 7.5
WL Wins wins/draw=0.5: 1
Score Util wins/draw=0.5: 0.394119
WL Wins wins/draw=0.3: 1
Score Util wins/draw=0.3: 0.394119
WL Wins wins/draw=0.7: 1
Score Util wins/draw=0.7: 0.394119
)%%";
    expect(name,out,expected);
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
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    out << "Black self komi wins/draw=0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wins/draw=0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;
    out << "Black self komi wins/draw=0.25: " << hist.currentSelfKomi(P_BLACK, 0.25) << endl;
    out << "White self komi wins/draw=0.25: " << hist.currentSelfKomi(P_WHITE, 0.25) << endl;
    out << "Black self komi wins/draw=0.75: " << hist.currentSelfKomi(P_BLACK, 0.75) << endl;
    out << "White self komi wins/draw=0.75: " << hist.currentSelfKomi(P_WHITE, 0.75) << endl;

    out << "Winner: " << colorToChar(hist.winner) << endl;
    out << "Final score: " << hist.finalWhiteMinusBlackScore << endl;
    out << "WL Wins wins/draw=0.5: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.5) << endl;
    out << "Score Util wins/draw=0.5: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.5, board, hist) << endl;
    out << "WL Wins wins/draw=0.3: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.3) << endl;
    out << "Score Util wins/draw=0.3: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.3, board, hist) << endl;
    out << "WL Wins wins/draw=0.7: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.7) << endl;
    out << "Score Util wins/draw=0.7: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.7, board, hist) << endl;

    string expected = R"%%(
Black self komi wins/draw=0.5: -7
White self komi wins/draw=0.5: 7
Black self komi wins/draw=0.25: -6.75
White self komi wins/draw=0.25: 6.75
Black self komi wins/draw=0.75: -7.25
White self komi wins/draw=0.75: 7.25
Winner: O
Final score: 7
WL Wins wins/draw=0.5: 1
Score Util wins/draw=0.5: 0.370402
WL Wins wins/draw=0.3: 1
Score Util wins/draw=0.3: 0.360776
WL Wins wins/draw=0.7: 1
Score Util wins/draw=0.7: 0.379949
)%%";
    expect(name,out,expected);
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
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    out << "Black self komi wins/draw=0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wins/draw=0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;
    out << "Black self komi wins/draw=0.25: " << hist.currentSelfKomi(P_BLACK, 0.25) << endl;
    out << "White self komi wins/draw=0.25: " << hist.currentSelfKomi(P_WHITE, 0.25) << endl;
    out << "Black self komi wins/draw=0.75: " << hist.currentSelfKomi(P_BLACK, 0.75) << endl;
    out << "White self komi wins/draw=0.75: " << hist.currentSelfKomi(P_WHITE, 0.75) << endl;

    out << "Winner: " << colorToChar(hist.winner) << endl;
    out << "Final score: " << hist.finalWhiteMinusBlackScore << endl;
    out << "WL Wins wins/draw=0.5: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.5) << endl;
    out << "Score Util wins/draw=0.5: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.5, board, hist) << endl;
    out << "WL Wins wins/draw=0.3: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.3) << endl;
    out << "Score Util wins/draw=0.3: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.3, board, hist) << endl;
    out << "WL Wins wins/draw=0.7: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.7) << endl;
    out << "Score Util wins/draw=0.7: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.7, board, hist) << endl;

    string expected = R"%%(
Black self komi wins/draw=0.5: -7
White self komi wins/draw=0.5: 7
Black self komi wins/draw=0.25: -6.75
White self komi wins/draw=0.25: 6.75
Black self komi wins/draw=0.75: -7.25
White self komi wins/draw=0.75: 7.25
Winner: .
Final score: 0
WL Wins wins/draw=0.5: 0.5
Score Util wins/draw=0.5: 0
WL Wins wins/draw=0.3: 0.3
Score Util wins/draw=0.3: -0.0111107
WL Wins wins/draw=0.7: 0.7
Score Util wins/draw=0.7: 0.0111107
)%%";
    expect(name,out,expected);
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
    BoardHistory hist(board,P_BLACK,rules,0);
    hist.endAndScoreGameNow(board);

    out << "Black self komi wins/draw=0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wins/draw=0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;
    out << "Black self komi wins/draw=0.25: " << hist.currentSelfKomi(P_BLACK, 0.25) << endl;
    out << "White self komi wins/draw=0.25: " << hist.currentSelfKomi(P_WHITE, 0.25) << endl;
    out << "Black self komi wins/draw=0.75: " << hist.currentSelfKomi(P_BLACK, 0.75) << endl;
    out << "White self komi wins/draw=0.75: " << hist.currentSelfKomi(P_WHITE, 0.75) << endl;

    out << "Winner: " << colorToChar(hist.winner) << endl;
    out << "Final score: " << hist.finalWhiteMinusBlackScore << endl;
    out << "WL Wins wins/draw=0.5: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.5) << endl;
    out << "Score Util wins/draw=0.5: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.5, board, hist) << endl;
    out << "WL Wins wins/draw=0.3: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.3) << endl;
    out << "Score Util wins/draw=0.3: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.3, board, hist) << endl;
    out << "WL Wins wins/draw=0.7: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.7) << endl;
    out << "Score Util wins/draw=0.7: " << NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, 0.7, board, hist) << endl;

    string expected = R"%%(
Black self komi wins/draw=0.5: -7
White self komi wins/draw=0.5: 7
Black self komi wins/draw=0.25: -6.75
White self komi wins/draw=0.25: 6.75
Black self komi wins/draw=0.75: -7.25
White self komi wins/draw=0.75: 7.25
Winner: O
Final score: 7
WL Wins wins/draw=0.5: 1
Score Util wins/draw=0.5: 0.604368
WL Wins wins/draw=0.3: 1
Score Util wins/draw=0.3: 0.591519
WL Wins wins/draw=0.7: 1
Score Util wins/draw=0.7: 0.616909
)%%";
    expect(name,out,expected);
  }

}
