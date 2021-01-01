#include "../tests/tests.h"

#include "../search/timecontrols.h"

using namespace std;
using namespace TestCommon;

void Tests::runTimeControlsTests() {
  Board board9Early = Board::parseBoard(9,9,R"%%(
.........
.........
.........
.........
.........
.........
.........
.........
.........
)%%");
  BoardHistory hist9Early(board9Early,P_BLACK,Rules(),0);

  Board board9Late = Board::parseBoard(9,9,R"%%(
..xoo..x.
.x.x.ox.x
..xoxo.x.
xx.oooo..
oxx..oxo.
oox.ox...
..o.ooxx.
.o..ox.x.
...oxxx..
)%%");
  BoardHistory hist9Late(board9Late,P_BLACK,Rules(),0);


  Board board19Early = Board::parseBoard(19,19,R"%%(
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
...................
)%%");
  BoardHistory hist19Early(board19Early,P_BLACK,Rules(),0);

  Board board19Late = Board::parseBoard(19,19,R"%%(
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . .
18 . . O . O . . X X . . . . . . . X O .
17 . O . X O . O O . X . X . . . . X X O
16 O X X . O X X O . . . . . X . X X O .
15 . . . . X O O O X . . . . X . . . O .
14 . . X X X O . O . . . . O X O X X O .
13 . . . . X X O O . . . . O . O O O O .
12 . . O . O O X . . . X . . O . O X . .
11 X X X . . X X . X X O O . O X . X . .
10 O O X . X . . . X O . . . . X . . . .
 9 O X . X . . . . . O . X . O O X . X .
 8 O O O O O O O . O . O X . . O X X O .
 7 . X . . . X O . . . O O O . . X O O .
 6 X . X X . X X . X . . O X X X . X O .
 5 . X O O X X X X . . O X O . . X . . .
 4 . O . O X O O O X X X X . . X . O O .
 3 . O . O O X O O O O . X . . X O O X .
 2 . O . O X . X O O X1X . . . . X X O O
 1 . . O . . X .2X3. O . . . . . . . X .
)%%");
  BoardHistory hist19Late(board19Late,P_BLACK,Rules(),0);

  auto tryTimeControlsOnBoard = [](const string& s, const TimeControls& timeControls, const Board& board, const BoardHistory& hist, double lagBuffer) {
    double minTime;
    double recommendedTime;
    double maxTime;
    timeControls.getTime(board,hist,lagBuffer,minTime,recommendedTime,maxTime);
    //Rounded time limit recommendation at the start of search
    double rrec0 = timeControls.roundUpTimeLimitIfNeeded(lagBuffer,0,recommendedTime);
    //Rounded time limit recommendation as we're just about to hit limit
    double rreclimit = timeControls.roundUpTimeLimitIfNeeded(lagBuffer,recommendedTime-0.000001,recommendedTime);
    //Rounded time limit recommendation as we're just about to hit rounded limit
    double rreclimit2 = timeControls.roundUpTimeLimitIfNeeded(lagBuffer,rreclimit-0.000001,rreclimit);

    cout << s << " min rec max = " << minTime << " " << recommendedTime << " " << maxTime
    << " roundedrec(used0) " << rrec0
    << " roundedrec(usedlimit) " << rreclimit
    << " roundedrec(usedlimit2) " << rreclimit2
    << endl;
  };

  auto tryTimeControlsOnBoards = [&](const TimeControls& timeControls, double lagBuffer) {
    tryTimeControlsOnBoard("board9Early",timeControls,board9Early,hist9Early,lagBuffer);
    tryTimeControlsOnBoard("board9Late",timeControls,board9Late,hist9Late,lagBuffer);
    tryTimeControlsOnBoard("board19Early",timeControls,board19Early,hist19Early,lagBuffer);
    tryTimeControlsOnBoard("board19Late",timeControls,board19Late,hist19Late,lagBuffer);
  };

  {
    cout << "===================================================================" << endl;
    cout << "Unlimited time controls" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    tryTimeControlsOnBoards(timeControls,0.0);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h absolute time controls, all time left" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 0;
    timeControls.numStonesPerPeriod = 0;
    timeControls.perPeriodTime = 0.0;
    timeControls.mainTimeLeft = 3600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 0;
    timeControls.numStonesLeftInPeriod = 0;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h absolute time controls, 10m left" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 0;
    timeControls.numStonesPerPeriod = 0;
    timeControls.perPeriodTime = 0.0;
    timeControls.mainTimeLeft = 600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 0;
    timeControls.numStonesLeftInPeriod = 0;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h fischer time controls, 10m left, 10s increment" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 10.0;
    timeControls.originalNumPeriods = 0;
    timeControls.numStonesPerPeriod = 0;
    timeControls.perPeriodTime = 0.0;
    timeControls.mainTimeLeft = 600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 0;
    timeControls.numStonesLeftInPeriod = 0;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h fischer time controls, 10m left, 10s increment, larger lag buffer" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 10.0;
    timeControls.originalNumPeriods = 0;
    timeControls.numStonesPerPeriod = 0;
    timeControls.perPeriodTime = 0.0;
    timeControls.mainTimeLeft = 600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 0;
    timeControls.numStonesLeftInPeriod = 0;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 5.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h fischer time controls, 15s left, 10s increment" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 10.0;
    timeControls.originalNumPeriods = 0;
    timeControls.numStonesPerPeriod = 0;
    timeControls.perPeriodTime = 0.0;
    timeControls.mainTimeLeft = 15.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 0;
    timeControls.numStonesLeftInPeriod = 0;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h fischer time controls, 5s left, 10s increment" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 10.0;
    timeControls.originalNumPeriods = 0;
    timeControls.numStonesPerPeriod = 0;
    timeControls.perPeriodTime = 0.0;
    timeControls.mainTimeLeft = 5.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 0;
    timeControls.numStonesLeftInPeriod = 0;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h fischer time controls, -1s left, 10s increment" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 10.0;
    timeControls.originalNumPeriods = 0;
    timeControls.numStonesPerPeriod = 0;
    timeControls.perPeriodTime = 0.0;
    timeControls.mainTimeLeft = -1.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 0;
    timeControls.numStonesLeftInPeriod = 0;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, all time left, 1 period of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 3600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, all time left, 3 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 3;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 3600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 3;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, all time left, 5 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 5;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 3600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 5;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, all time left, 6 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 6;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 3600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 6;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, all time left, 7 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 7;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 3600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 7;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, all time left, 3 moves canadian in 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 3600.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 3;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, no time left, 1 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, no time left, 5 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 5;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 5;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, no time left, 6 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 6;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 6;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, no time left, 7 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 7;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 7;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, no time left, 3 moves canadian in 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 3;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 15s left, 1 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 15.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 15s left, 2 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 2;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 15.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 2;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 15s left, 3 moves canadian in 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 15.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 3;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 45s left, 1 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 45.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 45s left, 2 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 2;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 45.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 2;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 45s left, 3 moves canadian in 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 45.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 3;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 61s left, 1 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 61.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 61s left, 2 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 2;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 61.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 2;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 61s left, 3 moves canadian in 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 61.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 3;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 70s left, 1 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 70.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 70s left, 2 periods of 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 2;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 70.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 2;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 70s left, 3 moves canadian in 30s" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 70.0;
    timeControls.inOvertime = false;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 3;
    timeControls.timeLeftInPeriod = 0.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 1 periods of 30s, just entered overtime" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 30.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 3 periods of 30s, just entered overtime" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 3;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 3;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 30.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 5 periods of 30s, just entered overtime" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 5;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 5;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 30.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 6 periods of 30s, just entered overtime" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 6;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 6;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 30.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 7 periods of 30s, just entered overtime" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 7;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 7;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 30.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 3 moves canadian in 30s, just entered overtime" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 3;
    timeControls.timeLeftInPeriod = 30.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 1 periods of 30s, entered overtime 15s used" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 15.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 5 periods of 30s, entered overtime 15s used" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 5;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 5;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 15.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 6 periods of 30s, entered overtime 15s used" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 6;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 6;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 15.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 7 periods of 30s, entered overtime 15s used" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 7;
    timeControls.numStonesPerPeriod = 1;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 7;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 15.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 3;
    timeControls.timeLeftInPeriod = 15.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used, 2 moves left" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 2;
    timeControls.timeLeftInPeriod = 15.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }

  {
    cout << "===================================================================" << endl;
    cout << "Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used, 1 moves left" << endl;
    cout << "===================================================================" << endl;

    TimeControls timeControls;
    timeControls.originalMainTime = 3600.0;
    timeControls.increment = 0.0;
    timeControls.originalNumPeriods = 1;
    timeControls.numStonesPerPeriod = 3;
    timeControls.perPeriodTime = 30.0;
    timeControls.mainTimeLeft = 0.0;
    timeControls.inOvertime = true;
    timeControls.numPeriodsLeftIncludingCurrent = 1;
    timeControls.numStonesLeftInPeriod = 1;
    timeControls.timeLeftInPeriod = 15.0;

    double lagBuffer = 1.0;
    tryTimeControlsOnBoards(timeControls,lagBuffer);
  }



}
