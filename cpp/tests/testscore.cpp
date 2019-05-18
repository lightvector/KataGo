#include "../tests/tests.h"

#include "../neuralnet/nninputs.h"

using namespace std;
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

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
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

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
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
    
    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
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
    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }

}
