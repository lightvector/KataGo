#ifndef TIMECONTROLS_H
#define TIMECONTROLS_H

#include "../core/global.h"
#include "../game/board.h"
#include "../game/boardhistory.h"

struct TimeControls {
  /*
    Supported time controls are Fischer or generalized byo-yomi.
    Nonzero increment together with byo yomi is not supported.

    Fisher: Always in main time. After every move, increment is added.

    Byoyomi: Either in main time, or in overtime. In overtime, we have numPeriodsLeft many periods,
    each one of perPeriodTime long, and does not get used up if we play numStonesPerPeriod stones during
    that period. numPeriodsLeft
  */
  double originalMainTime;
  double increment;
  int originalNumPeriods;
  int numStonesPerPeriod;
  double perPeriodTime;
  
  double mainTimeLeft;
  bool inOvertime;
  int numPeriodsLeftIncludingCurrent;
  int numStonesLeftInPeriod;
  double timeLeftInPeriod;

  //Construct a TimeControls with unlimited main time and otherwise zero initialized.
  TimeControls();
  ~TimeControls();

  //minTime - if you use less than this, you are wasting time that will not be reclaimed
  //recommendedTime - recommended mean time to search
  //maxTime - very bad to go over this time, possibly immediately losing
  void getTime(const Board& board, const BoardHistory& hist, double lagBuffer, double& minTime, double& recommendedTime, double& maxTime) const;
};


#endif
