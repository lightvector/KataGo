#include "../program/playutils.h"
#include "../core/timer.h"

#include <sstream>

using namespace std;

Loc PlayUtils::chooseRandomLegalMove(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc banMove) {
  int numLegalMoves = 0;
  Loc locs[Board::MAX_ARR_SIZE];
  for(Loc loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
    if(hist.isLegal(board,loc,pla) && loc != banMove) {
      locs[numLegalMoves] = loc;
      numLegalMoves += 1;
    }
  }
  if(numLegalMoves > 0) {
    int n = gameRand.nextUInt(numLegalMoves);
    return locs[n];
  }
  return Board::NULL_LOC;
}

int PlayUtils::chooseRandomLegalMoves(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc* buf, int len) {
  int numLegalMoves = 0;
  Loc locs[Board::MAX_ARR_SIZE];
  for(Loc loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
    if(hist.isLegal(board,loc,pla)) {
      locs[numLegalMoves] = loc;
      numLegalMoves += 1;
    }
  }
  if(numLegalMoves > 0) {
    for(int i = 0; i<len; i++) {
      int n = gameRand.nextUInt(numLegalMoves);
      buf[i] = locs[n];
    }
    return len;
  }
  return 0;
}


Loc PlayUtils::chooseRandomPolicyMove(
  const NNOutput* nnOutput, const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, double temperature, bool allowPass, Loc banMove
) {
  const float* policyProbs = nnOutput->policyProbs;
  int nnXLen = nnOutput->nnXLen;
  int nnYLen = nnOutput->nnYLen;
  int numLegalMoves = 0;
  double relProbs[NNPos::MAX_NN_POLICY_SIZE];
  int locs[NNPos::MAX_NN_POLICY_SIZE];
  for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
    Loc loc = NNPos::posToLoc(pos,board.x_size,board.y_size,nnXLen,nnYLen);
    if((loc == Board::PASS_LOC && !allowPass) || loc == banMove)
      continue;
    if(policyProbs[pos] > 0.0 && hist.isLegal(board,loc,pla)) {
      double relProb = policyProbs[pos];
      relProbs[numLegalMoves] = relProb;
      locs[numLegalMoves] = loc;
      numLegalMoves += 1;
    }
  }

  //Just in case the policy map is somehow not consistent with the board position
  if(numLegalMoves > 0) {
    uint32_t n = Search::chooseIndexWithTemperature(gameRand, relProbs, numLegalMoves, temperature);
    return locs[n];
  }
  return Board::NULL_LOC;
}

//Place black handicap stones, free placement
//Does NOT switch the initial player of the board history to white
void PlayUtils::playExtraBlack(
  Search* bot,
  int numExtraBlack,
  Board& board,
  BoardHistory& hist,
  double temperature,
  Rand& gameRand
) {
  Player pla = P_BLACK;

  NNResultBuf buf;
  for(int i = 0; i<numExtraBlack; i++) {
    MiscNNInputParams nnInputParams;
    nnInputParams.drawEquivalentWinsForWhite = bot->searchParams.drawEquivalentWinsForWhite;
    bot->nnEvaluator->evaluate(board,hist,pla,nnInputParams,buf,false,false);
    std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);

    bool allowPass = false;
    Loc banMove = Board::NULL_LOC;
    Loc loc = chooseRandomPolicyMove(nnOutput.get(), board, hist, pla, gameRand, temperature, allowPass, banMove);
    if(loc == Board::NULL_LOC)
      break;

    assert(hist.isLegal(board,loc,pla));
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
    hist.clear(board,pla,hist.rules,0);
  }

  bot->setPosition(pla,board,hist);
}

void PlayUtils::placeFixedHandicap(Board& board, int n) {
  int xSize = board.x_size;
  int ySize = board.y_size;
  if(xSize < 7 || ySize < 7)
    throw StringError("Board is too small for fixed handicap");
  if((xSize % 2 == 0 || ySize % 2 == 0) && n > 4)
    throw StringError("Fixed handicap > 4 is not allowed on boards with even dimensions");
  if((xSize <= 7 || ySize <= 7) && n > 4)
    throw StringError("Fixed handicap > 4 is not allowed on boards with size 7");
  if(n < 2)
    throw StringError("Fixed handicap < 2 is not allowed");
  if(n > 9)
    throw StringError("Fixed handicap > 9 is not allowed");

  board = Board(xSize,ySize);

  int xCoords[3]; //Corner, corner, side
  int yCoords[3]; //Corner, corner, side
  if(xSize <= 12) { xCoords[0] = 2; xCoords[1] = xSize-3; xCoords[2] = xSize/2; }
  else            { xCoords[0] = 3; xCoords[1] = xSize-4; xCoords[2] = xSize/2; }
  if(ySize <= 12) { yCoords[0] = 2; yCoords[1] = ySize-3; yCoords[2] = ySize/2; }
  else            { yCoords[0] = 3; yCoords[1] = ySize-4; yCoords[2] = ySize/2; }

  auto s = [&](int xi, int yi) {
    board.setStone(Location::getLoc(xCoords[xi],yCoords[yi],board.x_size),P_BLACK);
  };
  if(n == 2) { s(0,1); s(1,0); }
  else if(n == 3) { s(0,1); s(1,0); s(1,1); }
  else if(n == 4) { s(0,1); s(1,0); s(1,1); s(0,0); }
  else if(n == 5) { s(0,1); s(1,0); s(1,1); s(0,0); s(2,2); }
  else if(n == 6) { s(0,1); s(1,0); s(1,1); s(0,0); s(0,2); s(1,2); }
  else if(n == 7) { s(0,1); s(1,0); s(1,1); s(0,0); s(0,2); s(1,2); s(2,2); }
  else if(n == 8) { s(0,1); s(1,0); s(1,1); s(0,0); s(0,2); s(1,2); s(2,0); s(2,1); }
  else if(n == 9) { s(0,1); s(1,0); s(1,1); s(0,0); s(0,2); s(1,2); s(2,0); s(2,1); s(2,2); }
  else { ASSERT_UNREACHABLE; }
}

double PlayUtils::getHackedLCBForWinrate(const Search* search, const AnalysisData& data, Player pla) {
  double winrate = 0.5 * (1.0 + data.winLossValue);
  //Super hacky - in KataGo, lcb is on utility (i.e. also weighting score), not winrate, but if you're using
  //lz-analyze you probably don't know about utility and expect LCB to be about winrate. So we apply the LCB
  //radius to the winrate in order to get something reasonable to display, and also scale it proportionally
  //by how much winrate is supposed to matter relative to score.
  double radiusScaleHackFactor = search->searchParams.winLossUtilityFactor / (
    search->searchParams.winLossUtilityFactor +
    search->searchParams.staticScoreUtilityFactor +
    search->searchParams.dynamicScoreUtilityFactor +
    1.0e-20 //avoid divide by 0
  );
  //Also another factor of 0.5 because winrate goes from only 0 to 1 instead of -1 to 1 when it's part of utility
  radiusScaleHackFactor *= 0.5;
  double lcb = pla == P_WHITE ? winrate - data.radius * radiusScaleHackFactor : winrate + data.radius * radiusScaleHackFactor;
  return lcb;
}

float PlayUtils::roundAndClipKomi(double unrounded, const Board& board, bool looseClipping) {
  //Just in case, make sure komi is reasonable
  float range = looseClipping ? 40.0f + board.x_size * board.y_size : 40.0f + 0.5f * board.x_size * board.y_size;
  if(unrounded < -range)
    unrounded = -range;
  if(unrounded > range)
    unrounded = range;
  return (float)(0.5 * round(2.0 * unrounded));
}

ReportedSearchValues PlayUtils::getWhiteScoreValues(
  Search* bot,
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  int64_t numVisits,
  Logger& logger,
  const OtherGameProperties& otherGameProps
) {
  assert(numVisits > 0);
  SearchParams oldParams = bot->searchParams;
  SearchParams newParams = oldParams;
  newParams.maxVisits = numVisits;
  newParams.maxPlayouts = numVisits;
  newParams.rootNoiseEnabled = false;
  newParams.rootPolicyTemperature = 1.0;
  newParams.rootPolicyTemperatureEarly = 1.0;
  newParams.rootFpuReductionMax = newParams.fpuReductionMax;
  newParams.rootFpuLossProp = newParams.fpuLossProp;
  newParams.rootDesiredPerChildVisitsCoeff = 0.0;
  newParams.rootNumSymmetriesToSample = 1;

  if(otherGameProps.playoutDoublingAdvantage != 0.0 && otherGameProps.playoutDoublingAdvantagePla != C_EMPTY) {
    //Don't actually adjust playouts, but DO tell the bot what it's up against, so that it gives estimates
    //appropriate to the asymmetric game about to be played
    newParams.playoutDoublingAdvantagePla = otherGameProps.playoutDoublingAdvantagePla;
    newParams.playoutDoublingAdvantage = otherGameProps.playoutDoublingAdvantage;
  }

  bot->setParams(newParams);
  bot->setPosition(pla,board,hist);
  bot->runWholeSearch(pla,logger);

  ReportedSearchValues values = bot->getRootValuesRequireSuccess();
  bot->setParams(oldParams);
  return values;
}

static std::pair<double,double> evalKomi(
  map<float,std::pair<double,double>>& scoreWLCache,
  Search* botB,
  Search* botW,
  const Board& board,
  BoardHistory& hist,
  Player pla,
  int64_t numVisits,
  Logger& logger,
  const OtherGameProperties& otherGameProps,
  float roundedClippedKomi
) {
  auto iter = scoreWLCache.find(roundedClippedKomi);
  if(iter != scoreWLCache.end())
    return iter->second;

  float oldKomi = hist.rules.komi;
  hist.setKomi(roundedClippedKomi);

  ReportedSearchValues values0 = PlayUtils::getWhiteScoreValues(botB, board, hist, pla, numVisits, logger, otherGameProps);
  double lead = values0.lead;
  double winLoss = values0.winLossValue;

  //If we have a second bot, average the two
  if(botW != NULL && botW != botB) {
    ReportedSearchValues values1 = PlayUtils::getWhiteScoreValues(botW, board, hist, pla, numVisits, logger, otherGameProps);
    lead = 0.5 * (values0.lead + values1.lead);
    winLoss = 0.5 * (values0.winLossValue + values1.winLossValue);
  }
  std::pair<double,double> result = std::make_pair(lead,winLoss);
  scoreWLCache[roundedClippedKomi] = result;

  hist.setKomi(oldKomi);
  return result;
}

static double getNaiveEvenKomiHelper(
  map<float,std::pair<double,double>>& scoreWLCache,
  Search* botB,
  Search* botW,
  const Board& board,
  BoardHistory& hist,
  Player pla,
  int64_t numVisits,
  Logger& logger,
  const OtherGameProperties& otherGameProps,
  bool looseClipping
) {
  float oldKomi = hist.rules.komi;

  //A few times iterate based on expected score a few times to hopefully get a value close to fair
  double lastShift = 0.0;
  double lastWinLoss = 0.0;
  double lastLead = 0.0;
  for(int i = 0; i<3; i++) {
    std::pair<double,double> result = evalKomi(scoreWLCache,botB,botW,board,hist,pla,numVisits,logger,otherGameProps,hist.rules.komi);
    double lead = result.first;
    double winLoss = result.second;

    //If the last shift made stats go the WRONG way, and by a nontrivial amount, then revert half of it and stop immediately.
    if(i > 0) {
      if((lastLead > 0 && lead > lastLead + 5 && winLoss < 0.75) ||
         (lastLead < 0 && lead < lastLead - 5 && winLoss > -0.75) ||
         (lastWinLoss > 0 && winLoss > lastWinLoss + 0.1) ||
         (lastWinLoss < 0 && winLoss < lastWinLoss - 0.1)
      ) {
        float fairKomi = PlayUtils::roundAndClipKomi(hist.rules.komi - lastShift * 0.5f, board, looseClipping);
        hist.setKomi(fairKomi);
        // cout << "STOP" << endl;
        // cout << lastLead << " " << lead << " " << lastWinLoss << " " << winLoss << endl;
        break;
      }
    }
    lastLead = lead;
    lastWinLoss = winLoss;

    // cout << hist.rules.komi << " " << lead << " " << winLoss << endl;

    //Shift by the predicted lead
    double shift = -lead;
    //Under no situations should the shift be bigger in absolute value than the last shift
    if(i > 0 && abs(shift) > abs(lastShift)) {
      if(shift < 0) shift = -abs(lastShift);
      else if(shift > 0) shift = abs(lastShift);
    }
    lastShift = shift;

    //If the score and winrate would like to move in opposite directions, quit immediately.
    if((shift > 0 && winLoss > 0) || (shift < 0 && lead < 0))
      break;

    // cout << "Shifting by " << shift << endl;
    float fairKomi = PlayUtils::roundAndClipKomi(hist.rules.komi + shift, board, looseClipping);
    hist.setKomi(fairKomi);

    //After a small shift, break out to the binary search.
    if(abs(shift) < 16.0)
      break;
  }

  //Try a small window and do a binary search
  auto evalWinLoss = [&](double delta) {
    double newKomi = hist.rules.komi + delta;
    double winLoss = evalKomi(scoreWLCache,botB,botW,board,hist,pla,numVisits,logger,otherGameProps,PlayUtils::roundAndClipKomi(newKomi,board,looseClipping)).second;
    // cout << "Delta " << delta << " wr " << winLoss << endl;
    return winLoss;
  };

  double lowerDelta;
  double upperDelta;
  double lowerWinLoss;
  double upperWinLoss;

  //Grow window outward
  {
    double winLossZero = evalWinLoss(0);
    if(winLossZero < 0) {
      //Losing, so this is the lower bound
      lowerDelta = 0.0;
      lowerWinLoss = winLossZero;
      for(int i = 0; i<=5; i++) {
        upperDelta = round(pow(2.0,i));
        upperWinLoss = evalWinLoss(upperDelta);
        if(upperWinLoss >= 0)
          break;
      }
    }
    else {
      //Winning, so this is the upper bound
      upperDelta = 0.0;
      upperWinLoss = winLossZero;
      for(int i = 0; i<=5; i++) {
        lowerDelta = -round(pow(2.0,i));
        lowerWinLoss = evalWinLoss(lowerDelta);
        if(lowerWinLoss <= 0)
          break;
      }
    }
  }

  while(upperDelta - lowerDelta > 0.50001) {
    double midDelta = 0.5 * (lowerDelta + upperDelta);
    double midWinLoss = evalWinLoss(midDelta);
    if(midWinLoss < 0) {
      lowerDelta = midDelta;
      lowerWinLoss = midWinLoss;
    }
    else {
      upperDelta = midDelta;
      upperWinLoss = midWinLoss;
    }
  }
  //Floating point math should be exact to multiples of 0.5 so this should hold *exactly*.
  assert(upperDelta - lowerDelta == 0.5);

  double finalDelta;
  //If the winLoss are crossed, potentially due to noise, then just pick the average
  if(lowerWinLoss >= upperWinLoss - 1e-30)
    finalDelta = 0.5 * (lowerDelta + upperDelta);
  //If 0 is outside of the range, then choose the endpoint of the range.
  else if(upperWinLoss <= 0)
    finalDelta = upperDelta;
  else if(lowerWinLoss >= 0)
    finalDelta = lowerDelta;
  //Interpolate
  else
    finalDelta = lowerDelta + (upperDelta - lowerDelta) * (0-lowerWinLoss) / (upperWinLoss-lowerWinLoss);

  double newKomi = hist.rules.komi + finalDelta;
  // cout << "Final " << finalDelta << " " << newKomi << endl;

  hist.setKomi(oldKomi);
  return newKomi;
}

void PlayUtils::adjustKomiToEven(
  Search* botB,
  Search* botW,
  const Board& board,
  BoardHistory& hist,
  Player pla,
  int64_t numVisits,
  Logger& logger,
  const OtherGameProperties& otherGameProps,
  Rand& rand
) {
  map<float,std::pair<double,double>> scoreWLCache;
  bool looseClipping = false;
  double newKomi = getNaiveEvenKomiHelper(scoreWLCache,botB,botW,board,hist,pla,numVisits,logger,otherGameProps,looseClipping);
  double lower = floor(newKomi * 2.0) * 0.5;
  double upper = lower + 0.5;
  if(rand.nextBool((newKomi - lower) / (upper - lower)))
    newKomi = upper;
  else
    newKomi = lower;
  hist.setKomi(PlayUtils::roundAndClipKomi(newKomi,board,looseClipping));
}

float PlayUtils::computeLead(
  Search* botB,
  Search* botW,
  const Board& board,
  BoardHistory& hist,
  Player pla,
  int64_t numVisits,
  Logger& logger,
  const OtherGameProperties& otherGameProps
) {
  map<float,std::pair<double,double>> scoreWLCache;
  bool looseClipping = true;
  float oldKomi = hist.rules.komi;
  double naiveKomi = getNaiveEvenKomiHelper(scoreWLCache,botB,botW,board,hist,pla,numVisits,logger,otherGameProps,looseClipping);

  bool granularityIsCoarse = hist.rules.scoringRule == Rules::SCORING_AREA && !hist.rules.hasButton;
  if(!granularityIsCoarse) {
    assert(hist.rules.komi == oldKomi);
    return (float)(oldKomi - naiveKomi);
  }

  auto evalWinLoss = [&](double newKomi) {
    double winLoss = evalKomi(scoreWLCache,botB,botW,board,hist,pla,numVisits,logger,otherGameProps,PlayUtils::roundAndClipKomi(newKomi,board,looseClipping)).second;
    // cout << "Delta " << delta << " wr " << winLoss << endl;
    return winLoss;
  };

  //Smooth over area scoring 2-point granularity

  //If komi is exactly an integer, then we're good.
  if(naiveKomi == round(naiveKomi)) {
    assert(hist.rules.komi == oldKomi);
    return (float)(oldKomi - naiveKomi);
  }

  double lower = floor(naiveKomi * 2.0) * 0.5;
  double upper = lower + 0.5;

  //Average out the oscillation
  double lowerWinLoss = 0.5 * (evalWinLoss(upper) + evalWinLoss(lower-0.5));
  double upperWinLoss = 0.5 * (evalWinLoss(upper + 0.5) + evalWinLoss(lower));

  //If the winLoss are crossed, potentially due to noise, then just pick the average
  double result;
  if(lowerWinLoss >= upperWinLoss - 1e-30)
    result = 0.5 * (lower + upper);
  else {
    //Interpolate
    result = lower + (upper - lower) * (0-lowerWinLoss) / (upperWinLoss-lowerWinLoss);
    //Bound the result to be within lower-0.5 and upper+0.5
    if(result < lower-0.5) result = lower-0.5;
    if(result > upper+0.5) result = upper+0.5;
  }
  assert(hist.rules.komi == oldKomi);
  return (float)(oldKomi - result);
}


double PlayUtils::getSearchFactor(
  double searchFactorWhenWinningThreshold,
  double searchFactorWhenWinning,
  const SearchParams& params,
  const vector<double>& recentWinLossValues,
  Player pla
) {
  double searchFactor = 1.0;
  if(recentWinLossValues.size() >= 3 && params.winLossUtilityFactor - searchFactorWhenWinningThreshold > 1e-10) {
    double recentLeastWinning = pla == P_BLACK ? -params.winLossUtilityFactor : params.winLossUtilityFactor;
    for(int i = recentWinLossValues.size()-3; i < recentWinLossValues.size(); i++) {
      if(pla == P_BLACK && recentWinLossValues[i] > recentLeastWinning)
        recentLeastWinning = recentWinLossValues[i];
      if(pla == P_WHITE && recentWinLossValues[i] < recentLeastWinning)
        recentLeastWinning = recentWinLossValues[i];
    }
    double excessWinning = pla == P_BLACK ? -searchFactorWhenWinningThreshold - recentLeastWinning : recentLeastWinning - searchFactorWhenWinningThreshold;
    if(excessWinning > 0) {
      double lambda = excessWinning / (params.winLossUtilityFactor - searchFactorWhenWinningThreshold);
      searchFactor = 1.0 + lambda * (searchFactorWhenWinning - 1.0);
    }
  }
  return searchFactor;
}

vector<double> PlayUtils::computeOwnership(
  Search* bot,
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  int64_t numVisits,
  Logger& logger
) {
  assert(numVisits > 0);
  bool oldAlwaysIncludeOwnerMap = bot->alwaysIncludeOwnerMap;
  bot->setAlwaysIncludeOwnerMap(true);

  SearchParams oldParams = bot->searchParams;
  SearchParams newParams = oldParams;
  newParams.maxVisits = numVisits;
  newParams.maxPlayouts = numVisits;
  newParams.rootNoiseEnabled = false;
  newParams.rootPolicyTemperature = 1.0;
  newParams.rootPolicyTemperatureEarly = 1.0;
  newParams.rootFpuReductionMax = newParams.fpuReductionMax;
  newParams.rootFpuLossProp = newParams.fpuLossProp;
  newParams.rootDesiredPerChildVisitsCoeff = 0.0;
  newParams.rootNumSymmetriesToSample = 1;
  newParams.playoutDoublingAdvantagePla = C_EMPTY;
  newParams.playoutDoublingAdvantage = 0.0;
  if(newParams.numThreads > (numVisits+7)/8)
    newParams.numThreads = (numVisits+7)/8;

  bot->setParams(newParams);
  bot->setPosition(pla,board,hist);
  bot->runWholeSearch(pla,logger);

  int64_t minVisitsForOwnership = 2;
  vector<double> ownerships = bot->getAverageTreeOwnership(minVisitsForOwnership);

  bot->setParams(oldParams);
  bot->setAlwaysIncludeOwnerMap(oldAlwaysIncludeOwnerMap);
  bot->clearSearch();

  return ownerships;
}

vector<bool> PlayUtils::computeAnticipatedStatusesSimple(
  const Board& board,
  const BoardHistory& hist
) {
  vector<bool> isAlive(Board::MAX_ARR_SIZE,false);

  //Treat all stones as alive under a no result
  if(hist.isGameFinished && hist.isNoResult) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY)
          isAlive[loc] = true;
      }
    }
  }
  //Else use Tromp-taylorlike scoring, except recognizing pass-dead stones.
  else {
    Color area[Board::MAX_ARR_SIZE];
    BoardHistory histCopy = hist;
    histCopy.endAndScoreGameNow(board,area);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY) {
          isAlive[loc] = board.colors[loc] == area[loc];
        }
      }
    }
  }
  return isAlive;
}

vector<bool> PlayUtils::computeAnticipatedStatusesWithOwnership(
  Search* bot,
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  int64_t numVisits,
  Logger& logger
) {
  if(hist.isGameFinished)
    return computeAnticipatedStatusesSimple(board,hist);

  vector<bool> isAlive(Board::MAX_ARR_SIZE,false);
  bool solved[Board::MAX_ARR_SIZE];
  for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
    isAlive[i] = false;
    solved[i] = false;
  }

  vector<double> ownerships = computeOwnership(bot,board,hist,pla,numVisits,logger);
  int nnXLen = bot->nnXLen;
  int nnYLen = bot->nnYLen;

  //Heuristic:
  //Stones are considered dead if their average ownership is less than 0.2 equity in their own color,
  //or if the worst equity in the chain is less than -0.6 equity in their color.
  double avgThresholdForLife = 0.2;
  double worstThresholdForLife = -0.6;

  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(solved[loc])
        continue;

      if(board.colors[loc] == P_WHITE || board.colors[loc] == P_BLACK) {
        int pos = NNPos::locToPos(loc,board.x_size,nnXLen,nnYLen);
        double minOwnership = ownerships[pos];
        double maxOwnership = ownerships[pos];
        double ownershipSum = 0.0;
        double count = 0;

        //Run through the whole chain
        Loc cur = loc;
        do {
          pos = NNPos::locToPos(cur,board.x_size,nnXLen,nnYLen);
          minOwnership = std::min(ownerships[pos],minOwnership);
          maxOwnership = std::max(ownerships[pos],maxOwnership);
          ownershipSum += ownerships[pos];
          count += 1.0;
          cur = board.next_in_chain[cur];
        } while (cur != loc);

        double avgOwnership = ownershipSum / count;
        bool alive;
        if(board.colors[loc] == P_WHITE)
          alive = avgOwnership > avgThresholdForLife && minOwnership > worstThresholdForLife;
        else
          alive = avgOwnership < -avgThresholdForLife && maxOwnership < -worstThresholdForLife;

        //Run through the whole chain again, recording the result
        cur = loc;
        do {
          isAlive[cur] = alive;
          solved[cur] = true;
          cur = board.next_in_chain[cur];
        } while (cur != loc);
      }
    }
  }
  return isAlive;

}

string PlayUtils::BenchmarkResults::toStringNotDone() const {
  ostringstream out;
  out << "numSearchThreads = " << Global::strprintf("%2d",numThreads) << ":"
      << " " << totalPositionsSearched << " / " << totalPositions << " positions,"
      << " visits/s = " << Global::strprintf("%.2f",totalVisits / totalSeconds)
      << " (" << Global::strprintf("%.1f", totalSeconds) << " secs)";
  return out.str();
}
string PlayUtils::BenchmarkResults::toString() const {
  ostringstream out;
  out << "numSearchThreads = " << Global::strprintf("%2d",numThreads) << ":"
      << " " << totalPositionsSearched << " / " << totalPositions << " positions,"
      << " visits/s = " << Global::strprintf("%.2f",totalVisits / totalSeconds)
      << " nnEvals/s = " << Global::strprintf("%.2f",numNNEvals / totalSeconds)
      << " nnBatches/s = " << Global::strprintf("%.2f",numNNBatches / totalSeconds)
      << " avgBatchSize = " << Global::strprintf("%.2f",avgBatchSize)
      << " (" << Global::strprintf("%.1f", totalSeconds) << " secs)";
  return out.str();
}
string PlayUtils::BenchmarkResults::toStringWithElo(const BenchmarkResults* baseline, double secondsPerGameMove) const {
  ostringstream out;
  out << "numSearchThreads = " << Global::strprintf("%2d",numThreads) << ":"
      << " " << totalPositionsSearched << " / " << totalPositions << " positions,"
      << " visits/s = " << Global::strprintf("%.2f",totalVisits / totalSeconds)
      << " nnEvals/s = " << Global::strprintf("%.2f",numNNEvals / totalSeconds)
      << " nnBatches/s = " << Global::strprintf("%.2f",numNNBatches / totalSeconds)
      << " avgBatchSize = " << Global::strprintf("%.2f",avgBatchSize)
      << " (" << Global::strprintf("%.1f", totalSeconds) << " secs)";

  if(baseline == NULL)
    out << " (EloDiff baseline)";
  else {
    double diff = computeEloEffect(secondsPerGameMove) - baseline->computeEloEffect(secondsPerGameMove);
    out << " (EloDiff " << Global::strprintf("%+.0f",diff) << ")";
  }
  return out.str();
}

//From some test matches by lightvector using g170
static constexpr double eloGainPerDoubling = 250;

double PlayUtils::BenchmarkResults::computeEloEffect(double secondsPerGameMove) const {
  auto computeEloCost = [&](double baseVisits) {
    //Completely ad-hoc formula that approximately fits noisy tests. Probably not very good
    //but then again the recommendation of this benchmark program is very rough anyways, it
    //doesn't need to be all that great.
    return numThreads * 7.0 * pow(1600.0 / (800.0 + baseVisits),0.85);
  };

  double visitsPerSecond = totalVisits / totalSeconds;
  double gain = eloGainPerDoubling * log(visitsPerSecond) / log(2);
  double visitsPerMove = visitsPerSecond * secondsPerGameMove;
  double cost = computeEloCost(visitsPerMove);
  return gain - cost;
}

void PlayUtils::BenchmarkResults::printEloComparison(const vector<BenchmarkResults>& results, double secondsPerGameMove) {
  int bestIdx = 0;
  for(int i = 1; i<results.size(); i++) {
    if(results[i].computeEloEffect(secondsPerGameMove) > results[bestIdx].computeEloEffect(secondsPerGameMove))
      bestIdx = i;
  }

  cout << endl;
  cout << "Based on some test data, each speed doubling gains perhaps ~" << eloGainPerDoubling << " Elo by searching deeper." << endl;
  cout << "Based on some test data, each thread costs perhaps 7 Elo if using 800 visits, and 2 Elo if using 5000 visits (by making MCTS worse)." << endl;
  cout << "So APPROXIMATELY based on this benchmark, if you intend to do a " << secondsPerGameMove << " second search: " << endl;
  for(int i = 0; i<results.size(); i++) {
    int numThreads = results[i].numThreads;
    double eloEffect = results[i].computeEloEffect(secondsPerGameMove) - results[0].computeEloEffect(secondsPerGameMove);
    cout << "numSearchThreads = " << Global::strprintf("%2d",numThreads) << ": ";
    if(i == 0)
      cout << "(baseline)" << (i == bestIdx ? " (recommended)" : "") << endl;
    else
      cout << Global::strprintf("%+5.0f",eloEffect) << " Elo" << (i == bestIdx ? " (recommended)" : "") << endl;
  }
  cout << endl;
}


PlayUtils::BenchmarkResults PlayUtils::benchmarkSearchOnPositionsAndPrint(
  const SearchParams& params,
  const CompactSgf* sgf,
  int numPositionsToUse,
  NNEvaluator* nnEval,
  Logger& logger,
  const BenchmarkResults* baseline,
  double secondsPerGameMove,
  bool printElo
) {
  //Pick random positions from the SGF file, but deterministically
  vector<Move> moves = sgf->moves;
  string posSeed = "benchmarkPosSeed|";
  for(int i = 0; i<moves.size(); i++) {
    posSeed += Global::intToString((int)moves[i].loc);
    posSeed += "|";
  }

  vector<int> possiblePositionIdxs;
  {
    Rand posRand(posSeed);
    for(int i = 0; i<moves.size(); i++) {
      possiblePositionIdxs.push_back(i);
    }
    for(int i = possiblePositionIdxs.size()-1; i > 1; i--) {
      int r = posRand.nextUInt(i);
      int tmp = possiblePositionIdxs[i];
      possiblePositionIdxs[i] = possiblePositionIdxs[r];
      possiblePositionIdxs[r] = tmp;
    }
    if(possiblePositionIdxs.size() > numPositionsToUse)
      possiblePositionIdxs.resize(numPositionsToUse);
  }

  std::sort(possiblePositionIdxs.begin(),possiblePositionIdxs.end());

  BenchmarkResults results;
  results.numThreads = params.numThreads;
  results.totalPositions = possiblePositionIdxs.size();

  nnEval->clearCache();
  nnEval->clearStats();

  Rand seedRand;
  Search* bot = new Search(params,nnEval,Global::uint64ToString(seedRand.nextUInt64()));

  //Ignore the SGF rules, except for komi. Just use Tromp-taylor.
  Rules initialRules = Rules::getTrompTaylorish();
  //Take the komi from the sgf, otherwise ignore the rules in the sgf
  initialRules.komi = sgf->komi;

  Board board;
  Player nextPla;
  BoardHistory hist;
  sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);

  int moveNum = 0;

  for(int i = 0; i<possiblePositionIdxs.size(); i++) {
    cout << "\r" << results.toStringNotDone() << "      " << std::flush;

    int nextIdx = possiblePositionIdxs[i];
    while(moveNum < moves.size() && moveNum < nextIdx) {
      bool suc = hist.makeBoardMoveTolerant(board,moves[moveNum].loc,moves[moveNum].pla);
      if(!suc) {
        cerr << endl;
        cerr << board << endl;
        cerr << "SGF Illegal move " << (moveNum+1) << " for " << PlayerIO::colorToChar(moves[moveNum].pla) << ": " << Location::toString(moves[moveNum].loc,board) << endl;
        throw StringError("Illegal move in SGF");
      }
      nextPla = getOpp(moves[moveNum].pla);
      moveNum += 1;
    }

    bot->clearSearch();
    bot->setPosition(nextPla,board,hist);
    nnEval->clearCache();

    ClockTimer timer;
    bot->runWholeSearch(nextPla,logger);
    double seconds = timer.getSeconds();

    results.totalPositionsSearched += 1;
    results.totalSeconds += seconds;
    results.totalVisits += bot->getRootVisits();
  }

  results.numNNEvals = nnEval->numRowsProcessed();
  results.numNNBatches = nnEval->numBatchesProcessed();
  results.avgBatchSize = nnEval->averageProcessedBatchSize();

  if(printElo)
    cout << "\r" << results.toStringWithElo(baseline,secondsPerGameMove) << std::endl;
  else
    cout << "\r" << results.toString() << std::endl;

  delete bot;

  return results;
}
