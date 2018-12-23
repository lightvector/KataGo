
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

    double drawEquivsToTry[4] = {0.5, 0.3, 0.7, 1.0};
    for(int i = 0; i<4; i++) {
      double drawEquiv = drawEquivsToTry[i];
      string s = Global::strprintf("%.1f", drawEquiv);
      double scoreAdjusted = ScoreValue::whiteScoreDrawAdjust(score, drawEquiv, hist);
      double stdev = sqrt(std::max(0.0,ScoreValue::whiteScoreMeanSqOfScoreGridded(score,drawEquiv,hist) - scoreAdjusted * scoreAdjusted));
      double expectedScoreValue = ScoreValue::expectedWhiteScoreValue(scoreAdjusted, stdev, 0.0, 2.0, board);
      out << "WL Wins wins/draw=" << s << ": " << ScoreValue::whiteWinsOfWinner(hist.winner, drawEquiv) << endl;
      out << "Score wins/draw=" << s << ": " << scoreAdjusted << endl;
      out << "Score Stdev wins/draw=" << s << ": " << stdev << endl;
      out << "Score Util Smooth  wins/draw=" << s << ": " << ScoreValue::whiteScoreValueOfScoreSmooth(score, 0.0, 2.0, drawEquiv, board, hist) << endl;
      out << "Score Util SmootND wins/draw=" << s << ": " << ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust(score, 0.0, 2.0, board) << endl;
      out << "Score Util Gridded wins/draw=" << s << ": " << expectedScoreValue << endl;
      out << "Score Util GridInv wins/draw=" << s << ": " << ScoreValue::approxWhiteScoreOfScoreValueSmooth(expectedScoreValue,0.0,2.0,board) << endl;
    }
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
Score wins/draw=0.5: 7.5
Score Stdev wins/draw=0.5: 0
Score Util Smooth  wins/draw=0.5: 0.251332
Score Util SmootND wins/draw=0.5: 0.251332
Score Util Gridded wins/draw=0.5: 0.251202
Score Util GridInv wins/draw=0.5: 7.49569
WL Wins wins/draw=0.3: 1
Score wins/draw=0.3: 7.5
Score Stdev wins/draw=0.3: 0
Score Util Smooth  wins/draw=0.3: 0.251332
Score Util SmootND wins/draw=0.3: 0.251332
Score Util Gridded wins/draw=0.3: 0.251202
Score Util GridInv wins/draw=0.3: 7.49569
WL Wins wins/draw=0.7: 1
Score wins/draw=0.7: 7.5
Score Stdev wins/draw=0.7: 0
Score Util Smooth  wins/draw=0.7: 0.251332
Score Util SmootND wins/draw=0.7: 0.251332
Score Util Gridded wins/draw=0.7: 0.251202
Score Util GridInv wins/draw=0.7: 7.49569
WL Wins wins/draw=1.0: 1
Score wins/draw=1.0: 7.5
Score Stdev wins/draw=1.0: 0
Score Util Smooth  wins/draw=1.0: 0.251332
Score Util SmootND wins/draw=1.0: 0.251332
Score Util Gridded wins/draw=1.0: 0.251202
Score Util GridInv wins/draw=1.0: 7.49569
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
Score wins/draw=0.5: 7
Score Stdev wins/draw=0.5: 0.5
Score Util Smooth  wins/draw=0.5: 0.236117
Score Util SmootND wins/draw=0.5: 0.236117
Score Util Gridded wins/draw=0.5: 0.235795
Score Util GridInv wins/draw=0.5: 6.98954
WL Wins wins/draw=0.3: 1
Score wins/draw=0.3: 6.8
Score Stdev wins/draw=0.3: 0.458258
Score Util Smooth  wins/draw=0.3: 0.229949
Score Util SmootND wins/draw=0.3: 0.236117
Score Util Gridded wins/draw=0.3: 0.229594
Score Util GridInv wins/draw=0.3: 6.78852
WL Wins wins/draw=0.7: 1
Score wins/draw=0.7: 7.2
Score Stdev wins/draw=0.7: 0.458258
Score Util Smooth  wins/draw=0.7: 0.242238
Score Util SmootND wins/draw=0.7: 0.236117
Score Util Gridded wins/draw=0.7: 0.241938
Score Util GridInv wins/draw=0.7: 7.19017
WL Wins wins/draw=1.0: 1
Score wins/draw=1.0: 7.5
Score Stdev wins/draw=1.0: 0
Score Util Smooth  wins/draw=1.0: 0.251332
Score Util SmootND wins/draw=1.0: 0.236117
Score Util Gridded wins/draw=1.0: 0.251202
Score Util GridInv wins/draw=1.0: 7.49569
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
Score wins/draw=0.5: 0
Score Stdev wins/draw=0.5: 0.5
Score Util Smooth  wins/draw=0.5: 0
Score Util SmootND wins/draw=0.5: 0
Score Util Gridded wins/draw=0.5: 3.46945e-18
Score Util GridInv wins/draw=0.5: 9.80963e-17
WL Wins wins/draw=0.3: 0.3
Score wins/draw=0.3: -0.2
Score Stdev wins/draw=0.3: 0.458258
Score Util Smooth  wins/draw=0.3: -0.00707326
Score Util SmootND wins/draw=0.3: 0
Score Util Gridded wins/draw=0.3: -0.00706253
Score Util GridInv wins/draw=0.3: -0.199697
WL Wins wins/draw=0.7: 0.7
Score wins/draw=0.7: 0.2
Score Stdev wins/draw=0.7: 0.458258
Score Util Smooth  wins/draw=0.7: 0.00707326
Score Util SmootND wins/draw=0.7: 0
Score Util Gridded wins/draw=0.7: 0.00706253
Score Util GridInv wins/draw=0.7: 0.199697
WL Wins wins/draw=1.0: 1
Score wins/draw=1.0: 0.5
Score Stdev wins/draw=1.0: 0
Score Util Smooth  wins/draw=1.0: 0.0176793
Score Util SmootND wins/draw=1.0: 0
Score Util Gridded wins/draw=1.0: 0.0176772
Score Util GridInv wins/draw=1.0: 0.499941

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
Score wins/draw=0.5: 7
Score Stdev wins/draw=0.5: 0.5
Score Util Smooth  wins/draw=0.5: 0.3888
Score Util SmootND wins/draw=0.5: 0.3888
Score Util Gridded wins/draw=0.5: 0.388184
Score Util GridInv wins/draw=0.5: 6.98559
WL Wins wins/draw=0.3: 1
Score wins/draw=0.3: 6.8
Score Stdev wins/draw=0.3: 0.458258
Score Util Smooth  wins/draw=0.3: 0.380174
Score Util SmootND wins/draw=0.3: 0.3888
Score Util Gridded wins/draw=0.3: 0.379551
Score Util GridInv wins/draw=0.3: 6.7857
WL Wins wins/draw=0.7: 1
Score wins/draw=0.7: 7.2
Score Stdev wins/draw=0.7: 0.458258
Score Util Smooth  wins/draw=0.7: 0.397265
Score Util SmootND wins/draw=0.7: 0.3888
Score Util Gridded wins/draw=0.7: 0.396706
Score Util GridInv wins/draw=0.7: 7.18667
WL Wins wins/draw=1.0: 1
Score wins/draw=1.0: 7.5
Score Stdev wins/draw=1.0: 0
Score Util Smooth  wins/draw=1.0: 0.409666
Score Util SmootND wins/draw=1.0: 0.3888
Score Util Gridded wins/draw=1.0: 0.409563
Score Util GridInv wins/draw=1.0: 7.49749

)%%";
    expect(name,out,expected);
  }

}
