#include "../search/timecontrols.h"

#include <sstream>
#include <cmath>

TimeControls::TimeControls()
  :originalMainTime(1.0e30),
   increment(0.0),
   originalNumPeriods(0),
   numStonesPerPeriod(0),
   perPeriodTime(0.0),

   mainTimeLeft(1.0e30),
   inOvertime(false),
   numPeriodsLeftIncludingCurrent(0),
   numStonesLeftInPeriod(0),
   timeLeftInPeriod(0.0)
{}

TimeControls::~TimeControls()
{}


TimeControls TimeControls::absoluteTime(double mainTime) {
  TimeControls tc;
  tc.originalMainTime = mainTime;
  tc.increment = 0.0;
  tc.originalNumPeriods = 0;
  tc.numStonesPerPeriod = 0;
  tc.perPeriodTime = 0.0;
  tc.mainTimeLeft = mainTime;
  tc.inOvertime = false;
  tc.numPeriodsLeftIncludingCurrent = 0;
  tc.numStonesLeftInPeriod = 0;
  tc.timeLeftInPeriod = 0;
  return tc;
}

TimeControls TimeControls::canadianOrByoYomiTime(
  double mainTime,
  double perPeriodTime,
  int numPeriods,
  int numStonesPerPeriod
) {
  TimeControls tc;
  tc.originalMainTime = mainTime;
  tc.increment = 0.0;
  tc.originalNumPeriods = numPeriods;
  tc.numStonesPerPeriod = numStonesPerPeriod;
  tc.perPeriodTime = perPeriodTime;
  tc.mainTimeLeft = mainTime;
  tc.inOvertime = false;
  tc.numPeriodsLeftIncludingCurrent = numPeriods;
  tc.numStonesLeftInPeriod = 0;
  tc.timeLeftInPeriod = 0;
  return tc;
}

std::string TimeControls::toDebugString(const Board& board, const BoardHistory& hist, double lagBuffer) const {
  std::ostringstream out;
  out << "originalMainTime " << originalMainTime;
  if(increment != 0)
    out << "increment " << increment;
  if(originalNumPeriods != 0)
    out << " originalNumPeriods " << originalNumPeriods;
  if(numStonesPerPeriod != 0)
    out << " numStonesPerPeriod " << numStonesPerPeriod;
  if(perPeriodTime != 0)
    out << " perPeriodTime " << perPeriodTime;
  out << " mainTimeLeft " << mainTimeLeft;
  out << " inOvertime " << inOvertime;
  if(numPeriodsLeftIncludingCurrent != 0)
    out << " numPeriodsLeftIncludingCurrent " << numPeriodsLeftIncludingCurrent;
  if(numStonesLeftInPeriod != 0)
    out << " numStonesLeftInPeriod " << numStonesLeftInPeriod;
  if(timeLeftInPeriod != 0)
    out << " timeLeftInPeriod " << timeLeftInPeriod;

  double minTime;
  double recommendedTime;
  double maxTime;
  getTime(board,hist,lagBuffer,minTime,recommendedTime,maxTime);
  out << " minRecMax " << minTime << " " << recommendedTime << " " << maxTime;
  return out.str();
}

std::string TimeControls::toDebugString() const {
  std::ostringstream out;
  out << "originalMainTime " << originalMainTime;
  if(increment != 0)
    out << "increment " << increment;
  if(originalNumPeriods != 0)
    out << "originalNumPeriods " << originalNumPeriods;
  if(numStonesPerPeriod != 0)
    out << "numStonesPerPeriod " << numStonesPerPeriod;
  if(perPeriodTime != 0)
    out << "perPeriodTime " << perPeriodTime;
  out << "mainTimeLeft " << mainTimeLeft;
  out << "inOvertime " << inOvertime;
  if(numPeriodsLeftIncludingCurrent != 0)
    out << "numPeriodsLeftIncludingCurrent " << numPeriodsLeftIncludingCurrent;
  if(numStonesLeftInPeriod != 0)
    out << "numStonesLeftInPeriod " << numStonesLeftInPeriod;
  if(timeLeftInPeriod != 0)
    out << "timeLeftInPeriod " << timeLeftInPeriod;
  return out.str();
}


static double applyLagBuffer(double time, double lagBuffer) {
  if(time < 0)
    return time;
  else if(time < 2.0 * lagBuffer)
    return time * 0.5;
  else
    return time - lagBuffer;
}

void TimeControls::getTime(const Board& board, const BoardHistory& hist, double lagBuffer, double& minTime, double& recommendedTime, double& maxTime) const {
  (void)hist;

  int boardArea = board.x_size * board.y_size;
  int numStonesOnBoard = 0;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(board.colors[loc] == C_BLACK || board.colors[loc] == C_WHITE)
        numStonesOnBoard++;
    }
  }

  //Very crude way to estimate game progress
  double approxTurnsLeftAbsolute;
  double approxTurnsLeftIncrement; //Turns left in which we plan to spend our main time
  double approxTurnsLeftByoYomi;   //Turns left in which we plan to spend our main time
  {
    double typicalGameLengthToAllowForAbsolute = 0.95 * boardArea + 20.0;
    double typicalGameLengthToAllowForIncrement = 0.75 * boardArea + 15.0;
    double typicalGameLengthToAllowForByoYomi = 0.50 * boardArea + 10.0;

    double minApproxTurnsLeftAbsolute = 0.15 * boardArea + 30.0;
    double minApproxTurnsLeftIncrement = 0.10 * boardArea + 20.0;
    double minApproxTurnsLeftByoYomi = 0.02 * boardArea + 4.0;

    approxTurnsLeftAbsolute = std::max(typicalGameLengthToAllowForAbsolute - numStonesOnBoard, minApproxTurnsLeftAbsolute);
    approxTurnsLeftIncrement = std::max(typicalGameLengthToAllowForIncrement - numStonesOnBoard, minApproxTurnsLeftIncrement);
    approxTurnsLeftByoYomi = std::max(typicalGameLengthToAllowForByoYomi - numStonesOnBoard, minApproxTurnsLeftByoYomi);

    //Multiply by 0.5 since we only make half the moves
    approxTurnsLeftAbsolute *= 0.5;
    approxTurnsLeftIncrement *= 0.5;
    approxTurnsLeftByoYomi *= 0.5;
  }

  auto divideTimeEvenlyForGame = [approxTurnsLeftAbsolute,approxTurnsLeftIncrement,approxTurnsLeftByoYomi,this](double time, bool isIncrementOrAbs, bool isByoYomi) {
    double mainTimeToUseIfAbsolute = time / approxTurnsLeftAbsolute;

    if(isIncrementOrAbs) {
      double mainTimeToUse;
      if(time <= 0)
        mainTimeToUse = time;
      else {
        mainTimeToUse = time / approxTurnsLeftIncrement;
        //Make sure that if the increment is really very small, we don't choose a policy that is all that much more extreme than absolute time.
        mainTimeToUse = std::min(mainTimeToUse, mainTimeToUseIfAbsolute + 2.0 * increment);
      }
      return mainTimeToUse;
    }

    else if(isByoYomi) {
      double mainTimeToUse;
      if(perPeriodTime <= 0 || numStonesPerPeriod <= 0)
        mainTimeToUse = mainTimeToUseIfAbsolute;
      else {
        double byoYomiTimePerMove = perPeriodTime / numStonesPerPeriod;

        //Under the assumption that we spend a fixed amount of time per move and then when we run out of main time, we use our byo yomi time, and
        //strength is proportional to log(time spent), then the optimal policy is to use e * byoYomi time per move and running out in 1/e proportion
        //of the turns that we would if we spent only the byo yomi time per move.
        double theoreticalOptimalTurnsToSpendOurTime = (time / byoYomiTimePerMove) * exp(-1.0);
        double approxTurnsLeftToUse = theoreticalOptimalTurnsToSpendOurTime;

        //If our desired time is longer than optimal (because in reality saving time for deep enough in the midgame is more important)
        //then attempt to stretch it out to some degree.
        if(approxTurnsLeftByoYomi > theoreticalOptimalTurnsToSpendOurTime)
          approxTurnsLeftToUse = std::min(approxTurnsLeftByoYomi, theoreticalOptimalTurnsToSpendOurTime * 1.75);

        //If we'd be even slower than absolute time, then of course move as if absolute time.
        if(approxTurnsLeftToUse > approxTurnsLeftAbsolute)
          approxTurnsLeftToUse = approxTurnsLeftAbsolute;
        //Make sure that at the very end of our main time, we don't do silly things
        if(approxTurnsLeftToUse < 1)
          approxTurnsLeftToUse = 1;

        mainTimeToUse = time / approxTurnsLeftToUse;
        //Make sure that if the byo yomi is really very small, we don't choose a policy that is all that much more extreme than absolute time.
        mainTimeToUse = std::min(mainTimeToUse, mainTimeToUseIfAbsolute + 3.0 * byoYomiTimePerMove);
        //If we would have less than half the per-stone byo yomi main time remaining, then just go ahead and use it all.
        if(time - mainTimeToUse < byoYomiTimePerMove * 0.5)
          mainTimeToUse = time + byoYomiTimePerMove;
      }
      return mainTimeToUse;
    }

    return mainTimeToUseIfAbsolute;
  };

  //Initialize
  minTime = 0.0;
  recommendedTime = 0.0;
  maxTime = 0.0;

  double lagBufferToUse = lagBuffer;

  //Fischer or absolute time handling
  if(increment > 0 || numPeriodsLeftIncludingCurrent <= 0) {
    if(inOvertime)
      throw StringError("TimeControls: inOvertime with Fischer or absolute time, inconsistent time control?");
    if(numPeriodsLeftIncludingCurrent != 0)
      throw StringError("TimeControls: numPeriodsLeftIncludingCurrent != 0 with Fischer or absolute time, inconsistent time control?");

    //Note that some GTP controllers might give us a negative mainTimeLeft in weird cases. We tolerate this and do the best we can.
    if(mainTimeLeft <= increment) {
      minTime = 0.0;
      //Apply lagbuffer an extra time to the mainTimeLeft, ensuring we get extra buffering
      recommendedTime = applyLagBuffer(mainTimeLeft, lagBuffer);
      maxTime = mainTimeLeft;
    }
    else {
      //Apply lagbuffer an extra time to the excessMainTime, ensuring we get extra buffering
      double excessMainTime = applyLagBuffer(mainTimeLeft - increment, lagBuffer);
      minTime = 0.0;
      recommendedTime = increment + divideTimeEvenlyForGame(excessMainTime,true,false);
      maxTime = std::min(mainTimeLeft, increment + excessMainTime / 5.0);
    }
  }
  //Byo yomi or canadian time handling
  else {
    if(numStonesPerPeriod <= 0)
      throw StringError("TimeControls: numStonesPerPeriod <= 0 with byo-yomiish periods, inconsistent time control?");
    if(!inOvertime && numPeriodsLeftIncludingCurrent != originalNumPeriods)
      throw StringError("TimeControls: not in overtime, but numPeriodsLeftIncludingCurrent != originalNumPeriods");

    //Crudely treat all but the last 3 periods as main time.
    double effectiveMainTimeLeft = mainTimeLeft;
    bool effectivelyInOvertime = inOvertime;
    if(numPeriodsLeftIncludingCurrent > 3) {
      effectivelyInOvertime = false;
      if(!inOvertime) {
        effectiveMainTimeLeft += perPeriodTime * (numPeriodsLeftIncludingCurrent - 3);
      }
      else {
        effectiveMainTimeLeft += timeLeftInPeriod + perPeriodTime * (numPeriodsLeftIncludingCurrent - 4);
      }
    }

    if(!effectivelyInOvertime) {
      //The upper limit of what we'll tolerate for spending on a move in byo yomi
      double largeByoYomiTimePerMove = perPeriodTime / (0.75 * numStonesPerPeriod + 0.25);

      minTime = 0.0;
      recommendedTime = divideTimeEvenlyForGame(effectiveMainTimeLeft,false,true);
      maxTime = largeByoYomiTimePerMove + std::max(std::min(largeByoYomiTimePerMove * 1.75, effectiveMainTimeLeft), effectiveMainTimeLeft / 5.0);

      //If we're going into byo yomi, we might as well allow using the whole period
      if(maxTime > effectiveMainTimeLeft && maxTime < effectiveMainTimeLeft + largeByoYomiTimePerMove)
        maxTime = effectiveMainTimeLeft + largeByoYomiTimePerMove;

      //Increase the lagbuffer a little if upon entering byo yomi we're actually on the last byo yomi (i.e. running out actually kills us)
      if(maxTime > effectiveMainTimeLeft && numPeriodsLeftIncludingCurrent <= 1 && numStonesPerPeriod <= 1)
        lagBufferToUse *= 2.0;
    }
    else {
      if(numStonesLeftInPeriod < 1)
        throw StringError("TimeControls: numStonesLeftInPeriod < 1 while in overtime, inconsistent time control?");

      minTime = (numStonesLeftInPeriod <= 1) ? timeLeftInPeriod : 0.0;
      recommendedTime = timeLeftInPeriod / numStonesLeftInPeriod;
      maxTime = timeLeftInPeriod / (0.75 * numStonesLeftInPeriod + 0.25);

      //Increase the lagbuffer a little if we're actually on the last stone of the last byo yomi (i.e. running out actually kills us)
      if(numPeriodsLeftIncludingCurrent <= 1 && numStonesLeftInPeriod <= 1)
        lagBufferToUse *= 2.0;
    }
  }

  //Lag buffer
  minTime = applyLagBuffer(minTime,lagBufferToUse);
  recommendedTime = applyLagBuffer(recommendedTime,lagBufferToUse);
  maxTime = applyLagBuffer(maxTime,lagBufferToUse);

  //Just in case
  if(maxTime < 0)
    maxTime = 0;
  if(minTime < 0)
    minTime = 0;
  if(recommendedTime < 0)
    recommendedTime = 0;
  if(minTime > maxTime)
    minTime = maxTime;
  if(recommendedTime > maxTime)
    recommendedTime = maxTime;
}
