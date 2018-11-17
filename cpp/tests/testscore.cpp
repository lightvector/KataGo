
#include "../neuralnet/nninputs.h"

#include "../tests/tests.h"
using namespace TestCommon;

void Tests::runScoreTests() {
  cout << "Running score and utility tests" << endl;
  ostringstream out;

  auto printScoreStats = [&out](const Board& board, const BoardHistory& hist) {
    out << "Black self komi wins/draw=0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wins/draw=0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;
    out << "Black self komi wins/draw=0.25: " << hist.currentSelfKomi(P_BLACK, 0.25) << endl;
    out << "White self komi wins/draw=0.25: " << hist.currentSelfKomi(P_WHITE, 0.25) << endl;
    out << "Black self komi wins/draw=0.75: " << hist.currentSelfKomi(P_BLACK, 0.75) << endl;
    out << "White self komi wins/draw=0.75: " << hist.currentSelfKomi(P_WHITE, 0.75) << endl;

    out << "Winner: " << colorToChar(hist.winner) << endl;
    double score = hist.finalWhiteMinusBlackScore;
    out << "Final score: " << score << endl;
    out << "WL Wins wins/draw=0.5: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.5) << endl;
    out << "Score Util Smooth  wins/draw=0.5: " << NNOutput::whiteScoreValueOfScoreSmooth(score, 0.5, board, hist) << endl;
    out << "Score Util SmootND wins/draw=0.5: " << NNOutput::whiteScoreValueOfScoreSmoothNoDrawAdjust(score, board) << endl;
    out << "Score Util Gridded wins/draw=0.5: " << NNOutput::whiteScoreValueOfScoreGridded(score, 0.5, board, hist) << endl;
    out << "Score Util GridInv wins/draw=0.5: " << NNOutput::approxWhiteScoreOfScoreValueSmooth(NNOutput::whiteScoreValueOfScoreGridded(score, 0.5, board, hist),board) << endl;
    out << "WL Wins wins/draw=0.3: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.3) << endl;
    out << "Score Util Smooth  wins/draw=0.3: " << NNOutput::whiteScoreValueOfScoreSmooth(score, 0.3, board, hist) << endl;
    out << "Score Util SmootND wins/draw=0.3: " << NNOutput::whiteScoreValueOfScoreSmoothNoDrawAdjust(score, board) << endl;
    out << "Score Util Gridded wins/draw=0.3: " << NNOutput::whiteScoreValueOfScoreGridded(score, 0.3, board, hist) << endl;
    out << "Score Util GridInv wins/draw=0.3: " << NNOutput::approxWhiteScoreOfScoreValueSmooth(NNOutput::whiteScoreValueOfScoreGridded(score, 0.3, board, hist),board) << endl;
    out << "WL Wins wins/draw=0.7: " << NNOutput::whiteWinsOfWinner(hist.winner, 0.7) << endl;
    out << "Score Util Smooth  wins/draw=0.7: " << NNOutput::whiteScoreValueOfScoreSmooth(score, 0.7, board, hist) << endl;
    out << "Score Util SmootND wins/draw=0.7: " << NNOutput::whiteScoreValueOfScoreSmoothNoDrawAdjust(score, board) << endl;
    out << "Score Util Gridded wins/draw=0.7: " << NNOutput::whiteScoreValueOfScoreGridded(score, 0.7, board, hist) << endl;
    out << "Score Util GridInv wins/draw=0.7: " << NNOutput::approxWhiteScoreOfScoreValueSmooth(NNOutput::whiteScoreValueOfScoreGridded(score, 0.7, board, hist),board) << endl;
    out << "WL Wins wins/draw=1.0: " << NNOutput::whiteWinsOfWinner(hist.winner, 1.0) << endl;
    out << "Score Util Smooth  wins/draw=1.0: " << NNOutput::whiteScoreValueOfScoreSmooth(score, 1.0, board, hist) << endl;
    out << "Score Util SmootND wins/draw=1.0: " << NNOutput::whiteScoreValueOfScoreSmoothNoDrawAdjust(score, board) << endl;
    out << "Score Util Gridded wins/draw=1.0: " << NNOutput::whiteScoreValueOfScoreGridded(score, 1.0, board, hist) << endl;
    out << "Score Util GridInv wins/draw=1.0: " << NNOutput::approxWhiteScoreOfScoreValueSmooth(NNOutput::whiteScoreValueOfScoreGridded(score, 1.0, board, hist),board) << endl;
  };

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

    printScoreStats(board,hist);

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
Score Util Smooth  wins/draw=0.5: 0.251332
Score Util SmootND wins/draw=0.5: 0.251332
Score Util Gridded wins/draw=0.5: 0.251332
Score Util GridInv wins/draw=0.5: 7.5
WL Wins wins/draw=0.3: 1
Score Util Smooth  wins/draw=0.3: 0.251332
Score Util SmootND wins/draw=0.3: 0.251332
Score Util Gridded wins/draw=0.3: 0.251332
Score Util GridInv wins/draw=0.3: 7.5
WL Wins wins/draw=0.7: 1
Score Util Smooth  wins/draw=0.7: 0.251332
Score Util SmootND wins/draw=0.7: 0.251332
Score Util Gridded wins/draw=0.7: 0.251332
Score Util GridInv wins/draw=0.7: 7.5
WL Wins wins/draw=1.0: 1
Score Util Smooth  wins/draw=1.0: 0.251332
Score Util SmootND wins/draw=1.0: 0.251332
Score Util Gridded wins/draw=1.0: 0.251332
Score Util GridInv wins/draw=1.0: 7.5
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

    printScoreStats(board,hist);

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
Score Util Smooth  wins/draw=0.5: 0.236117
Score Util SmootND wins/draw=0.5: 0.236117
Score Util Gridded wins/draw=0.5: 0.235973
Score Util GridInv wins/draw=0.5: 6.99531
WL Wins wins/draw=0.3: 1
Score Util Smooth  wins/draw=0.3: 0.229949
Score Util SmootND wins/draw=0.3: 0.236117
Score Util Gridded wins/draw=0.3: 0.229829
Score Util GridInv wins/draw=0.3: 6.79611
WL Wins wins/draw=0.7: 1
Score Util Smooth  wins/draw=0.7: 0.242238
Score Util SmootND wins/draw=0.7: 0.236117
Score Util Gridded wins/draw=0.7: 0.242116
Score Util GridInv wins/draw=0.7: 7.19601
WL Wins wins/draw=1.0: 1
Score Util Smooth  wins/draw=1.0: 0.251332
Score Util SmootND wins/draw=1.0: 0.236117
Score Util Gridded wins/draw=1.0: 0.251332
Score Util GridInv wins/draw=1.0: 7.5
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

    printScoreStats(board,hist);

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
Score Util Smooth  wins/draw=0.5: 0
Score Util SmootND wins/draw=0.5: 0
Score Util Gridded wins/draw=0.5: 0
Score Util GridInv wins/draw=0.5: 0
WL Wins wins/draw=0.3: 0.3
Score Util Smooth  wins/draw=0.3: -0.00707326
Score Util SmootND wins/draw=0.3: 0
Score Util Gridded wins/draw=0.3: -0.00707173
Score Util GridInv wins/draw=0.3: -0.199957
WL Wins wins/draw=0.7: 0.7
Score Util Smooth  wins/draw=0.7: 0.00707326
Score Util SmootND wins/draw=0.7: 0
Score Util Gridded wins/draw=0.7: 0.00707173
Score Util GridInv wins/draw=0.7: 0.199957
WL Wins wins/draw=1.0: 1
Score Util Smooth  wins/draw=1.0: 0.0176793
Score Util SmootND wins/draw=1.0: 0
Score Util Gridded wins/draw=1.0: 0.0176793
Score Util GridInv wins/draw=1.0: 0.5
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

    printScoreStats(board,hist);

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
Score Util Smooth  wins/draw=0.5: 0.3888
Score Util SmootND wins/draw=0.5: 0.3888
Score Util Gridded wins/draw=0.5: 0.388299
Score Util GridInv wins/draw=0.5: 6.98827
WL Wins wins/draw=0.3: 1
Score Util Smooth  wins/draw=0.3: 0.380174
Score Util SmootND wins/draw=0.3: 0.3888
Score Util Gridded wins/draw=0.3: 0.379752
Score Util GridInv wins/draw=0.3: 6.7903
WL Wins wins/draw=0.7: 1
Score Util Smooth  wins/draw=0.7: 0.397265
Score Util SmootND wins/draw=0.7: 0.3888
Score Util Gridded wins/draw=0.7: 0.396845
Score Util GridInv wins/draw=0.7: 7.18999
WL Wins wins/draw=1.0: 1
Score Util Smooth  wins/draw=1.0: 0.409666
Score Util SmootND wins/draw=1.0: 0.3888
Score Util Gridded wins/draw=1.0: 0.409666
Score Util GridInv wins/draw=1.0: 7.5
)%%";
    expect(name,out,expected);
  }

}
