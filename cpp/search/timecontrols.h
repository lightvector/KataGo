#ifndef SEARCH_TIMECONTROLS_H
#define SEARCH_TIMECONTROLS_H

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
  double mainTimeLimit;
  double maxTimePerMove;
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

  //The threshold at which we consider time allowed to be unlimited
  static constexpr double UNLIMITED_TIME_THRESHOLD = 1e20;
  //The max time we tolerate a user inputting
  static constexpr double MAX_USER_INPUT_TIME = 1e25;
  //The value that fields default to when unset and need to be unlimited by default
  static constexpr double UNLIMITED_TIME_DEFAULT = 1e30;
  //The value that fields default to when unset and need to be unlimited by default and larger than other things
  static constexpr double UNLIMITED_TIME_DEFAULT_LARGE = 1e40;

  static TimeControls absoluteTime(double mainTime);
  static TimeControls fischerTime(double mainTime, double increment);
  static TimeControls fischerCappedTime(double mainTime, double increment, double mainTimeLimit, double maxTimePerMove);
  static TimeControls canadianOrByoYomiTime(
    double mainTime,
    double perPeriodTime,
    int numPeriods,
    int numStonesPerPeriod
  );

  bool isEffectivelyUnlimitedTime() const;

  //minTime - if you use less than this, you are wasting time that will not be reclaimed
  //recommendedTime - recommended mean time to search
  //maxTime - very bad to go over this time, possibly immediately losing
  void getTime(const Board& board, const BoardHistory& hist, double lagBuffer, double& minTime, double& recommendedTime, double& maxTime) const;

  //If we'd think for a given time limit and actually it would lose time to stop at this limit, then bump the limit up
  //This is used for not partial-wasting byo yomi periods.
  double roundUpTimeLimitIfNeeded(double lagBuffer, double timeUsed, double timeLimit) const;

  std::string toDebugString() const;
  std::string toDebugString(const Board& board, const BoardHistory& hist, double lagBuffer) const;
};

#endif  // SEARCH_TIMECONTROLS_H_
