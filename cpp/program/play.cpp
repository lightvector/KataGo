#include "../program/play.h"

#include "../core/global.h"
#include "../program/setup.h"
#include "../search/asyncbot.h"

using namespace std;

static double nextGaussianTruncated(Rand& rand, double bound) {
  double d = rand.nextGaussian();
  //Truncated refers to the probability distribution, not the sample
  //So on falling outside the range, we redraw, rather than capping.
  while(d < -bound || d > bound)
    d = rand.nextGaussian();
  return d;
}

static int getMaxExtraBlack(int bSize) {
  if(bSize <= 10)
    return 0;
  if(bSize <= 14)
    return 1;
  if(bSize <= 18)
    return 2;
  return 3;
}

static ExtraBlackAndKomi chooseExtraBlackAndKomi(
  float base, float stdev, double allowIntegerProb, double handicapProb, double bigStdevProb, float bigStdev, int bSize, Rand& rand
) {
  int extraBlack = 0;
  float komi = base;

  if(stdev > 0.0f)
    komi += stdev * (float)nextGaussianTruncated(rand,3.0);
  if(bigStdev > 0.0f && rand.nextDouble() < bigStdevProb)
    komi += bigStdev * (float)nextGaussianTruncated(rand,3.0);

  //Adjust for bSize, so that we don't give the same massive komis on smaller boards
  komi = base + (komi - base) * (float)bSize / 19.0f;

  //Add handicap stones compensated with komi
  int maxExtraBlack = getMaxExtraBlack(bSize);
  if(maxExtraBlack > 0 && rand.nextDouble() < handicapProb) {
    extraBlack += 1+rand.nextUInt(maxExtraBlack);
  }

  //Discretize komi
  float lower;
  float upper;
  if(rand.nextDouble() < allowIntegerProb) {
    lower = floor(komi*2.0f) / 2.0f;
    upper = ceil(komi*2.0f) / 2.0f;
  }
  else {
    lower = floor(komi+ 0.5f)-0.5f;
    upper = ceil(komi+0.5f)-0.5f;
  }

  if(lower == upper)
    komi = lower;
  else {
    assert(upper > lower);
    if(rand.nextDouble() < (komi - lower) / (upper - lower))
      komi = upper;
    else
      komi = lower;
  }

  assert(Rules::komiIsIntOrHalfInt(komi));
  return ExtraBlackAndKomi(extraBlack,komi,base);
}

int Play::numHandicapStones(const Board& initialBoard, const vector<Move>& moveHistory, bool assumeMultipleStartingBlackMovesAreHandicap) {
  //Make the longest possible contiguous sequence of black moves - treat a string of consecutive black
  //moves at the start of the game as "handicap"
  //This is necessary because when loading sgfs or on some servers, (particularly with free placement)
  //handicap is implemented by having black just make a bunch of moves in a row.
  //But if white makes multiple moves in a row after that, then the plays are probably not handicap, someone's setting
  //up a problem position by having black play all moves in a row then white play all moves in a row.
  Board board = initialBoard;

  if(assumeMultipleStartingBlackMovesAreHandicap) {
    for(int i = 0; i<moveHistory.size(); i++) {
      Loc moveLoc = moveHistory[i].loc;
      Player movePla = moveHistory[i].pla;
      if(movePla != P_BLACK) {
        //Two white moves in a row?
        if(i+1 < moveHistory.size() && moveHistory[i+1].pla != P_BLACK) {
          //Re-set board, don't play these moves
          board = initialBoard;
        }
        break;
      }
      bool isMultiStoneSuicideLegal = true;
      bool suc = board.playMove(moveLoc,movePla,isMultiStoneSuicideLegal);
      if(!suc)
        break;
    }
  }

  int startBoardNumBlackStones = 0;
  int startBoardNumWhiteStones = 0;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(board.colors[loc] == C_BLACK)
        startBoardNumBlackStones += 1;
      else if(board.colors[loc] == C_WHITE)
        startBoardNumWhiteStones += 1;
    }
  }
  //If we set up in a nontrivial position, then consider it a non-handicap game.
  if(startBoardNumWhiteStones != 0)
    return 0;
  //If there was only one "handicap" stone, then it was a regular game
  if(startBoardNumBlackStones <= 1)
    return 0;
  return startBoardNumBlackStones;
}

double Play::getHackedLCBForWinrate(const Search* search, const AnalysisData& data, Player pla) {
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


//----------------------------------------------------------------------------------------------------------

InitialPosition::InitialPosition()
  :board(),hist(),pla(C_EMPTY)
{}
InitialPosition::InitialPosition(const Board& b, const BoardHistory& h, Player p)
  :board(b),hist(h),pla(p)
{}
InitialPosition::~InitialPosition()
{}

//------------------------------------------------------------------------------------------------

GameInitializer::GameInitializer(ConfigParser& cfg)
  :createGameMutex(),rand()
{
  initShared(cfg);
  noResultStdev = cfg.contains("noResultStdev") ? cfg.getDouble("noResultStdev",0.0,1.0) : 0.0;
  drawRandRadius = cfg.contains("drawRandRadius") ? cfg.getDouble("drawRandRadius",0.0,1.0) : 0.0;
}

void GameInitializer::initShared(ConfigParser& cfg) {

  allowedKoRuleStrs = cfg.getStrings("koRules", Rules::koRuleStrings());
  allowedScoringRuleStrs = cfg.getStrings("scoringRules", Rules::scoringRuleStrings());
  allowedTaxRuleStrs = cfg.getStrings("taxRules", Rules::taxRuleStrings());
  allowedMultiStoneSuicideLegals = cfg.getBools("multiStoneSuicideLegals");

  for(size_t i = 0; i < allowedKoRuleStrs.size(); i++)
    allowedKoRules.push_back(Rules::parseKoRule(allowedKoRuleStrs[i]));
  for(size_t i = 0; i < allowedScoringRuleStrs.size(); i++)
    allowedScoringRules.push_back(Rules::parseScoringRule(allowedScoringRuleStrs[i]));
  for(size_t i = 0; i < allowedTaxRuleStrs.size(); i++)
    allowedTaxRules.push_back(Rules::parseTaxRule(allowedTaxRuleStrs[i]));

  if(allowedKoRules.size() <= 0)
    throw IOError("koRules must have at least one value in " + cfg.getFileName());
  if(allowedScoringRules.size() <= 0)
    throw IOError("scoringRules must have at least one value in " + cfg.getFileName());
  if(allowedTaxRules.size() <= 0)
    throw IOError("taxRules must have at least one value in " + cfg.getFileName());
  if(allowedMultiStoneSuicideLegals.size() <= 0)
    throw IOError("multiStoneSuicideLegals must have at least one value in " + cfg.getFileName());

  allowedBSizes = cfg.getInts("bSizes", 2, Board::MAX_LEN);
  allowedBSizeRelProbs = cfg.getDoubles("bSizeRelProbs",0.0,1e100);

  komiMean = cfg.getFloat("komiMean",-60.0f,60.0f);
  komiStdev = cfg.getFloat("komiStdev",0.0f,60.0f);
  komiAllowIntegerProb = cfg.getDouble("komiAllowIntegerProb",0.0,1.0);
  handicapProb = cfg.getDouble("handicapProb",0.0,1.0);
  komiBigStdevProb = cfg.getDouble("komiBigStdevProb",0.0,1.0);
  komiBigStdev = cfg.getFloat("komiBigStdev",0.0f,60.0f);

  if(allowedBSizes.size() <= 0)
    throw IOError("bSizes must have at least one value in " + cfg.getFileName());
  if(allowedBSizes.size() != allowedBSizeRelProbs.size())
    throw IOError("bSizes and bSizeRelProbs must have same number of values in " + cfg.getFileName());
}

GameInitializer::~GameInitializer()
{}

void GameInitializer::createGame(Board& board, Player& pla, BoardHistory& hist, ExtraBlackAndKomi& extraBlackAndKomi, const InitialPosition* initialPosition) {
  //Multiple threads will be calling this, and we have some mutable state such as rand.
  lock_guard<std::mutex> lock(createGameMutex);
  createGameSharedUnsynchronized(board,pla,hist,extraBlackAndKomi,initialPosition);
  if(noResultStdev != 0.0 || drawRandRadius != 0.0)
    throw StringError("GameInitializer::createGame called in a mode that doesn't support specifying noResultStdev or drawRandRadius");
}

void GameInitializer::createGame(Board& board, Player& pla, BoardHistory& hist, ExtraBlackAndKomi& extraBlackAndKomi, SearchParams& params, const InitialPosition* initialPosition) {
  //Multiple threads will be calling this, and we have some mutable state such as rand.
  lock_guard<std::mutex> lock(createGameMutex);
  createGameSharedUnsynchronized(board,pla,hist,extraBlackAndKomi,initialPosition);

  if(noResultStdev > 1e-30) {
    double mean = params.noResultUtilityForWhite;
    params.noResultUtilityForWhite = mean + noResultStdev * nextGaussianTruncated(rand, 3.0);
    while(params.noResultUtilityForWhite < -1.0 || params.noResultUtilityForWhite > 1.0)
      params.noResultUtilityForWhite = mean + noResultStdev * nextGaussianTruncated(rand, 3.0);
  }
  if(drawRandRadius > 1e-30) {
    double mean = params.drawEquivalentWinsForWhite;
    if(mean < 0.0 || mean > 1.0)
      throw StringError("GameInitializer: params.drawEquivalentWinsForWhite not within [0,1]: " + Global::doubleToString(mean));
    params.drawEquivalentWinsForWhite = mean + drawRandRadius * (rand.nextDouble() * 2 - 1);
    while(params.drawEquivalentWinsForWhite < 0.0 || params.drawEquivalentWinsForWhite > 1.0)
      params.drawEquivalentWinsForWhite = mean + drawRandRadius * (rand.nextDouble() * 2 - 1);
  }
}

void GameInitializer::createGameSharedUnsynchronized(Board& board, Player& pla, BoardHistory& hist, ExtraBlackAndKomi& extraBlackAndKomi, const InitialPosition* initialPosition) {
  if(initialPosition != NULL) {
    board = initialPosition->board;
    hist = initialPosition->hist;
    pla = initialPosition->pla;

    extraBlackAndKomi = chooseExtraBlackAndKomi(
      hist.rules.komi, komiStdev, komiAllowIntegerProb, 0.0,
      komiBigStdevProb, komiBigStdev, std::min(board.x_size,board.y_size), rand
    );
    assert(extraBlackAndKomi.extraBlack == 0);
    hist.setKomi(extraBlackAndKomi.komi);
    return;
  }

  int bSize = allowedBSizes[rand.nextUInt(allowedBSizeRelProbs.data(),allowedBSizeRelProbs.size())];
  board = Board(bSize,bSize);

  Rules rules;
  rules.koRule = allowedKoRules[rand.nextUInt(allowedKoRules.size())];
  rules.scoringRule = allowedScoringRules[rand.nextUInt(allowedScoringRules.size())];
  rules.taxRule = allowedTaxRules[rand.nextUInt(allowedTaxRules.size())];
  rules.multiStoneSuicideLegal = allowedMultiStoneSuicideLegals[rand.nextUInt(allowedMultiStoneSuicideLegals.size())];

  extraBlackAndKomi = chooseExtraBlackAndKomi(
    komiMean, komiStdev, komiAllowIntegerProb, handicapProb,
    komiBigStdevProb, komiBigStdev, bSize, rand
  );
  rules.komi = extraBlackAndKomi.komi;

  pla = P_BLACK;
  hist.clear(board,pla,rules,0);
}

//----------------------------------------------------------------------------------------------------------

MatchPairer::MatchPairer(
  ConfigParser& cfg,
  int nBots,
  const vector<string>& bNames,
  const vector<NNEvaluator*>& nEvals,
  const vector<SearchParams>& bParamss,
  bool forSelfPlay,
  bool forGateKeeper
): MatchPairer(cfg,nBots,bNames,nEvals,bParamss,forSelfPlay,forGateKeeper,vector<bool>(nBots))
{}


MatchPairer::MatchPairer(
  ConfigParser& cfg,
  int nBots,
  const vector<string>& bNames,
  const vector<NNEvaluator*>& nEvals,
  const vector<SearchParams>& bParamss,
  bool forSelfPlay,
  bool forGateKeeper,
  const vector<bool>& exclude
)
  :numBots(nBots),
   botNames(bNames),
   nnEvals(nEvals),
   baseParamss(bParamss),
   excludeBot(exclude),
   secondaryBots(),
   nextMatchups(),
   nextMatchupsBuf(),
   rand(),
   matchRepFactor(1),
   repsOfLastMatchup(0),
   numGamesStartedSoFar(0),
   numGamesTotal(),
   logGamesEvery(),
   getMatchupMutex()
{
  assert(!(forSelfPlay && forGateKeeper));
  assert(botNames.size() == numBots);
  assert(nnEvals.size() == numBots);
  assert(baseParamss.size() == numBots);
  assert(exclude.size() == numBots);
  if(forSelfPlay) {
    assert(numBots == 1);
    numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
  }
  else if(forGateKeeper) {
    assert(numBots == 2);
    numGamesTotal = cfg.getInt64("numGamesPerGating",0,((int64_t)1) << 24);
  }
  else {
    if(cfg.contains("secondaryBots"))
      secondaryBots = cfg.getInts("secondaryBots",0,4096);
    for(int i = 0; i<secondaryBots.size(); i++)
      assert(secondaryBots[i] >= 0 && secondaryBots[i] < numBots);
    numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
  }

  if(cfg.contains("matchRepFactor"))
    matchRepFactor = cfg.getInt("matchRepFactor",1,100000);

  logGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);
}

MatchPairer::~MatchPairer()
{}

int MatchPairer::getNumGamesTotalToGenerate() const {
  return numGamesTotal;
}

bool MatchPairer::getMatchup(
  int64_t& gameIdx, BotSpec& botSpecB, BotSpec& botSpecW, Logger& logger
)
{
  std::lock_guard<std::mutex> lock(getMatchupMutex);

  if(numGamesStartedSoFar >= numGamesTotal)
    return false;

  gameIdx = numGamesStartedSoFar;
  numGamesStartedSoFar += 1;

  if(numGamesStartedSoFar % logGamesEvery == 0)
    logger.write("Started " + Global::int64ToString(numGamesStartedSoFar) + " games");
  int logNNEvery = logGamesEvery*100 > 1000 ? logGamesEvery*100 : 1000;
  if(numGamesStartedSoFar % logNNEvery == 0) {
    for(int i = 0; i<nnEvals.size(); i++) {
      if(nnEvals[i] != NULL) {
        logger.write(nnEvals[i]->getModelFileName());
        logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
        logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
        logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
      }
    }
  }

  pair<int,int> matchup = getMatchupPairUnsynchronized();
  botSpecB.botIdx = matchup.first;
  botSpecB.botName = botNames[matchup.first];
  botSpecB.nnEval = nnEvals[matchup.first];
  botSpecB.baseParams = baseParamss[matchup.first];

  botSpecW.botIdx = matchup.second;
  botSpecW.botName = botNames[matchup.second];
  botSpecW.nnEval = nnEvals[matchup.second];
  botSpecW.baseParams = baseParamss[matchup.second];

  return true;
}

pair<int,int> MatchPairer::getMatchupPairUnsynchronized() {
  if(nextMatchups.size() <= 0) {
    if(numBots == 1)
      return make_pair(0,0);

    nextMatchupsBuf.clear();
    //First generate the pairs only in a one-sided manner
    for(int i = 0; i<numBots; i++) {
      if(excludeBot[i])
        continue;
      for(int j = 0; j<numBots; j++) {
        if(excludeBot[j])
          continue;
        if(i < j && !(contains(secondaryBots,i) && contains(secondaryBots,j))) {
          nextMatchupsBuf.push_back(make_pair(i,j));
        }
      }
    }
    //Shuffle
    for(int i = nextMatchupsBuf.size()-1; i >= 1; i--) {
      int j = (int)rand.nextUInt(i+1);
      pair<int,int> tmp = nextMatchupsBuf[i];
      nextMatchupsBuf[i] = nextMatchupsBuf[j];
      nextMatchupsBuf[j] = tmp;
    }

    //Then expand each pair into each player starting first
    for(int i = 0; i<nextMatchupsBuf.size(); i++) {
      pair<int,int> p = nextMatchupsBuf[i];
      pair<int,int> swapped = make_pair(p.second,p.first);
      if(rand.nextBool(0.5)) {
        nextMatchups.push_back(p);
        nextMatchups.push_back(swapped);
      }
      else {
        nextMatchups.push_back(swapped);
        nextMatchups.push_back(p);
      }
    }
  }

  pair<int,int> matchup = nextMatchups.back();

  //Swap pair every other matchup if doing more than one rep
  if(repsOfLastMatchup % 2 == 1) {
    pair<int,int> tmp = make_pair(matchup.second,matchup.first);
    matchup = tmp;
  }

  if(repsOfLastMatchup >= matchRepFactor-1) {
    nextMatchups.pop_back();
    repsOfLastMatchup = 0;
  }
  else {
    repsOfLastMatchup++;
  }

  return matchup;
}

//----------------------------------------------------------------------------------------------------------

FancyModes::FancyModes()
  :initGamesWithPolicy(false),forkSidePositionProb(0.0),
   noCompensateKomiProb(0.0),compensateKomiVisits(1),
   earlyForkGameProb(0.0),earlyForkGameExpectedMoveProp(0.0),earlyForkGameMinChoices(1),earlyForkGameMaxChoices(1),
   sekiForkHack(false),
   cheapSearchProb(0),cheapSearchVisits(0),cheapSearchTargetWeight(0.0f),
   reduceVisits(false),reduceVisitsThreshold(100.0),reduceVisitsThresholdLookback(1),reducedVisitsMin(0),reducedVisitsWeight(1.0f),
   policySurpriseDataWeight(0.0),
   recordTreePositions(false),recordTreeThreshold(0),recordTreeTargetWeight(0.0f),
   allowResignation(false),resignThreshold(0.0),resignConsecTurns(1),
   forSelfPlay(false),dataXLen(-1),dataYLen(-1)
{}
FancyModes::~FancyModes()
{}

//----------------------------------------------------------------------------------------------------------

static void failIllegalMove(Search* bot, Logger& logger, Board board, Loc loc) {
  ostringstream sout;
  sout << "Bot returned null location or illegal move!?!" << "\n";
  sout << board << "\n";
  sout << bot->getRootBoard() << "\n";
  sout << "Pla: " << PlayerIO::playerToString(bot->getRootPla()) << "\n";
  sout << "Loc: " << Location::toString(loc,bot->getRootBoard()) << "\n";
  logger.write(sout.str());
  bot->getRootBoard().checkConsistency();
  ASSERT_UNREACHABLE;
}

static void logSearch(Search* bot, Logger& logger, Loc loc) {
  ostringstream sout;
  Board::printBoard(sout, bot->getRootBoard(), loc, &(bot->getRootHist().moveHistory));
  sout << "\n";
  sout << "Root visits: " << bot->numRootVisits() << "\n";
  sout << "Policy surprise " << bot->getPolicySurprise() << "\n";
  sout << "PV: ";
  bot->printPV(sout, bot->rootNode, 25);
  sout << "\n";
  sout << "Tree:\n";
  bot->printTree(sout, bot->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10),P_WHITE);

  logger.write(sout.str());
}


Loc Play::chooseRandomPolicyMove(const NNOutput* nnOutput, const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, double temperature, bool allowPass, Loc banMove) {
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

static float roundAndClipKomi(double unrounded, const Board& board) {
  //Just in case, make sure komi is reasonable
  float range = board.x_size * board.y_size;
  if(unrounded < -range)
    unrounded = -range;
  if(unrounded > range)
    unrounded = range;
  return (float)(0.5 * round(2.0 * unrounded));
}

static double getWhiteScoreEstimate(Search* bot, const Board& board, const BoardHistory& hist, Player pla, int numVisits, Logger& logger) {
  assert(numVisits > 0);
  SearchParams oldParams = bot->searchParams;
  SearchParams newParams = oldParams;
  newParams.maxVisits = numVisits;
  newParams.maxPlayouts = numVisits;
  newParams.rootNoiseEnabled = false;
  newParams.rootFpuReductionMax = newParams.fpuReductionMax;
  newParams.rootFpuLossProp = newParams.fpuLossProp;
  newParams.rootDesiredPerChildVisitsCoeff = 0.0;

  bot->setParams(newParams);
  bot->setPosition(pla,board,hist);
  bot->runWholeSearch(pla,logger,NULL);

  ReportedSearchValues values = bot->getRootValuesAssertSuccess();
  bot->setParams(oldParams);
  return values.expectedScore;
}

void Play::adjustKomiToEven(
  Search* bot,
  const Board& board,
  BoardHistory& hist,
  Player pla,
  int64_t numVisits,
  Logger& logger
) {
  //Iterate a few times in case the neural net knows the bot isn't perfectly score maximizing.
  for(int i = 0; i<3; i++) {
    double finalWhiteScore = getWhiteScoreEstimate(bot, board, hist, pla, numVisits, logger);
    double fairKomi = hist.rules.komi - finalWhiteScore;
    hist.setKomi(roundAndClipKomi(fairKomi,board));
  }
}

double Play::getSearchFactor(
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



//Place black handicap stones, free placement
//Does NOT switch the initial player of the board history to white
void Play::playExtraBlack(
  Search* bot,
  Logger& logger,
  ExtraBlackAndKomi extraBlackAndKomi,
  Board& board,
  BoardHistory& hist,
  double temperature,
  Rand& gameRand,
  bool adjustKomi,
  int numVisitsForKomi
) {
  Player pla = P_BLACK;

  //First, restore back to baseline komi
  hist.setKomi(roundAndClipKomi(extraBlackAndKomi.komiBase,board));

  NNResultBuf buf;
  for(int i = 0; i<extraBlackAndKomi.extraBlack; i++) {
    MiscNNInputParams nnInputParams;
    nnInputParams.drawEquivalentWinsForWhite = bot->searchParams.drawEquivalentWinsForWhite;
    bot->nnEvaluator->evaluate(board,hist,pla,nnInputParams,buf,NULL,false,false);
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

  if(adjustKomi) {
    //Adjust komi to be fair for the handicap according to what the bot thinks.
    adjustKomiToEven(bot,board,hist,pla,numVisitsForKomi,logger);
    //Then, reapply the komi offset from base that we should have had
    hist.setKomi(roundAndClipKomi(hist.rules.komi + extraBlackAndKomi.komi - extraBlackAndKomi.komiBase, board));
  }

  bot->setPosition(pla,board,hist);
}

static bool shouldStop(vector<std::atomic<bool>*>& stopConditions) {
  for(int j = 0; j<stopConditions.size(); j++) {
    if(stopConditions[j]->load())
      return true;
  }
  return false;
}

static Loc chooseRandomLegalMove(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc banMove) {
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

static int chooseRandomLegalMoves(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc* buf, int len) {
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

static Loc chooseRandomForkingMove(const NNOutput* nnOutput, const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc banMove) {
  double r = gameRand.nextDouble();
  bool allowPass = true;
  //70% of the time, do a random temperature 1 policy move
  if(r < 0.70)
    return Play::chooseRandomPolicyMove(nnOutput, board, hist, pla, gameRand, 1.0, allowPass, banMove);
  //25% of the time, do a random temperature 2 policy move
  else if(r < 0.95)
    return Play::chooseRandomPolicyMove(nnOutput, board, hist, pla, gameRand, 2.0, allowPass, banMove);
  //5% of the time, do a random legal move
  else
    return chooseRandomLegalMove(board, hist, pla, gameRand, banMove);
}

static void extractPolicyTarget(
  vector<PolicyTargetMove>& buf,
  const Search* toMoveBot,
  const SearchNode* node,
  vector<Loc>& locsBuf,
  vector<double>& playSelectionValuesBuf
) {
  double scaleMaxToAtLeast = 10.0;

  assert(node != NULL);
  bool allowDirectPolicyMoves = false;
  bool success = toMoveBot->getPlaySelectionValues(*node,locsBuf,playSelectionValuesBuf,scaleMaxToAtLeast,allowDirectPolicyMoves);
  assert(success);
  (void)success; //Avoid warning when asserts are disabled

  assert(locsBuf.size() == playSelectionValuesBuf.size());
  assert(locsBuf.size() <= toMoveBot->rootBoard.x_size * toMoveBot->rootBoard.y_size + 1);

  //Make sure we don't overflow int16
  double maxValue = 0.0;
  for(int moveIdx = 0; moveIdx<locsBuf.size(); moveIdx++) {
    double value = playSelectionValuesBuf[moveIdx];
    assert(value >= 0.0);
    if(value > maxValue)
      maxValue = value;
  }

  //TODO
  // if(maxValue > 1e9) {
  //   cout << toMoveBot->rootBoard << endl;
  //   cout << "LARGE PLAY SELECTION VALUE " << maxValue << endl;
  //   toMoveBot->printTree(cout, node, PrintTreeOptions(), P_WHITE);
  // }

  double factor = 1.0;
  if(maxValue > 30000.0)
    factor = 30000.0 / maxValue;

  for(int moveIdx = 0; moveIdx<locsBuf.size(); moveIdx++) {
    double value = playSelectionValuesBuf[moveIdx] * factor;
    assert(value <= 30001.0);
    buf.push_back(PolicyTargetMove(locsBuf[moveIdx],(int16_t)round(value)));
  }
}

static void extractValueTargets(ValueTargets& buf, const Search* toMoveBot, const SearchNode* node, vector<double>* recordUtilities) {
  ReportedSearchValues values;
  bool success = toMoveBot->getNodeValues(*node,values);
  assert(success);
  (void)success; //Avoid warning when asserts are disabled

  buf.win = (float)values.winValue;
  buf.loss = (float)values.lossValue;
  buf.noResult = (float)values.noResultValue;
  buf.score = (float)values.expectedScore;

  if(recordUtilities != NULL) {
    buf.hasMctsUtility = true;
    assert(recordUtilities->size() > 255);
    buf.mctsUtility1 = (float)((*recordUtilities)[0]);
    buf.mctsUtility4 = (float)((*recordUtilities)[3]);
    buf.mctsUtility16 = (float)((*recordUtilities)[15]);
    buf.mctsUtility64 = (float)((*recordUtilities)[63]);
    buf.mctsUtility256 = (float)((*recordUtilities)[255]);
  }
  else {
    buf.hasMctsUtility = false;
    buf.mctsUtility1 = 0.0f;
    buf.mctsUtility4 = 0.0f;
    buf.mctsUtility16 = 0.0f;
    buf.mctsUtility64 = 0.0f;
    buf.mctsUtility256 = 0.0f;
  }
}

//Recursively walk non-root-node subtree under node recording positions that have enough visits
//We also only record positions where the player to move made best moves along the tree so far.
//Does NOT walk down branches of excludeLoc0 and excludeLoc1 - these are used to avoid writing
//subtree positions for branches that we are about to actually play or do a forked sideposition search on.
static void recordTreePositionsRec(
  FinishedGameData* gameData,
  const Board& board, const BoardHistory& hist, Player pla,
  const Search* toMoveBot,
  const SearchNode* node, int depth, int maxDepth, bool plaAlwaysBest, bool oppAlwaysBest,
  int minVisitsAtNode, float recordTreeTargetWeight,
  int numNeuralNetChangesSoFar,
  vector<Loc>& locsBuf, vector<double>& playSelectionValuesBuf,
  Loc excludeLoc0, Loc excludeLoc1
) {
  if(node->numChildren <= 0)
    return;

  if(plaAlwaysBest && node != toMoveBot->rootNode) {
    SidePosition* sp = new SidePosition(board,hist,pla,numNeuralNetChangesSoFar);
    extractPolicyTarget(sp->policyTarget, toMoveBot, node, locsBuf, playSelectionValuesBuf);
    extractValueTargets(sp->whiteValueTargets, toMoveBot, node, NULL);
    sp->targetWeight = recordTreeTargetWeight;
    sp->unreducedNumVisits = toMoveBot->getRootVisits();
    gameData->sidePositions.push_back(sp);
  }

  if(depth >= maxDepth)
    return;

  //Best child is the one with the largest number of visits, find it
  int bestChildIdx = 0;
  int64_t bestChildVisits = 0;
  for(int i = 1; i<node->numChildren; i++) {
    const SearchNode* child = node->children[i];
    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t numVisits = child->stats.visits;
    child->statsLock.clear(std::memory_order_release);
    if(numVisits > bestChildVisits) {
      bestChildVisits = numVisits;
      bestChildIdx = i;
    }
  }

  for(int i = 0; i<node->numChildren; i++) {
    bool newPlaAlwaysBest = oppAlwaysBest;
    bool newOppAlwaysBest = plaAlwaysBest && i == bestChildIdx;

    if(!newPlaAlwaysBest && !newOppAlwaysBest)
      continue;

    const SearchNode* child = node->children[i];
    if(child->prevMoveLoc == excludeLoc0 || child->prevMoveLoc == excludeLoc1)
      continue;

    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t numVisits = child->stats.visits;
    child->statsLock.clear(std::memory_order_release);

    if(numVisits < minVisitsAtNode)
      continue;

    Board copy = board;
    BoardHistory histCopy = hist;
    histCopy.makeBoardMoveAssumeLegal(copy, child->prevMoveLoc, pla, NULL);
    Player nextPla = getOpp(pla);
    recordTreePositionsRec(
      gameData,
      copy,histCopy,nextPla,
      toMoveBot,
      child,depth+1,maxDepth,newPlaAlwaysBest,newOppAlwaysBest,
      minVisitsAtNode,recordTreeTargetWeight,
      numNeuralNetChangesSoFar,
      locsBuf,playSelectionValuesBuf,
      Board::NULL_LOC,Board::NULL_LOC
    );
  }
}

//Top-level caller for recursive func
static void recordTreePositions(
  FinishedGameData* gameData,
  const Board& board, const BoardHistory& hist, Player pla,
  const Search* toMoveBot,
  int minVisitsAtNode, float recordTreeTargetWeight,
  int numNeuralNetChangesSoFar,
  vector<Loc>& locsBuf, vector<double>& playSelectionValuesBuf,
  Loc excludeLoc0, Loc excludeLoc1
) {
  assert(toMoveBot->rootBoard.pos_hash == board.pos_hash);
  assert(toMoveBot->rootHistory.moveHistory.size() == hist.moveHistory.size());
  assert(toMoveBot->rootPla == pla);
  assert(toMoveBot->rootNode != NULL);
  //Don't go too deep recording extra positions
  int maxDepth = 5;
  recordTreePositionsRec(
    gameData,
    board,hist,pla,
    toMoveBot,
    toMoveBot->rootNode, 0, maxDepth, true, true,
    minVisitsAtNode, recordTreeTargetWeight,
    numNeuralNetChangesSoFar,
    locsBuf,playSelectionValuesBuf,
    excludeLoc0,excludeLoc1
  );
}


static Loc getGameInitializationMove(
  Search* botB, Search* botW, Board& board, const BoardHistory& hist, Player pla, NNResultBuf& buf,
  Rand& gameRand, double temperature
) {
  NNEvaluator* nnEval = (pla == P_BLACK ? botB : botW)->nnEvaluator;
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = (pla == P_BLACK ? botB : botW)->searchParams.drawEquivalentWinsForWhite;
  nnEval->evaluate(board,hist,pla,nnInputParams,buf,NULL,false,false);
  std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);

  vector<Loc> locs;
  vector<double> playSelectionValues;
  int nnXLen = nnOutput->nnXLen;
  int nnYLen = nnOutput->nnYLen;
  assert(nnXLen >= board.x_size);
  assert(nnYLen >= board.y_size);
  assert(nnXLen > 0 && nnXLen < 100); //Just a sanity check to make sure no other crazy values have snuck in
  assert(nnYLen > 0 && nnYLen < 100); //Just a sanity check to make sure no other crazy values have snuck in
  int policySize = NNPos::getPolicySize(nnXLen,nnYLen);
  for(int movePos = 0; movePos<policySize; movePos++) {
    Loc moveLoc = NNPos::posToLoc(movePos,board.x_size,board.y_size,nnXLen,nnYLen);
    double policyProb = nnOutput->policyProbs[movePos];
    if(!hist.isLegal(board,moveLoc,pla) || policyProb <= 0)
      continue;
    locs.push_back(moveLoc);
    playSelectionValues.push_back(pow(policyProb,1.0/temperature));
  }

  //In practice, this should never happen, but in theory, a very badly-behaved net that rounds
  //all legal moves to zero could result in this. We still go ahead and fail, since this more likely some sort of bug.
  if(playSelectionValues.size() <= 0)
    throw StringError("getGameInitializationMove: playSelectionValues.size() <= 0");

  //With a tiny probability, choose a uniformly random move instead of a policy move, to also
  //add a bit more outlierish variety
  uint32_t idxChosen;
  if(gameRand.nextBool(0.0002))
    idxChosen = gameRand.nextUInt(playSelectionValues.size());
  else
    idxChosen = gameRand.nextUInt(playSelectionValues.data(),playSelectionValues.size());
  Loc loc = locs[idxChosen];
  return loc;
}

//Try playing a bunch of pure policy moves instead of playing from the start to initialize the board
//and add entropy
static void initializeGameUsingPolicy(
  Search* botB, Search* botW, Board& board, BoardHistory& hist, Player& pla,
  Rand& gameRand, bool doEndGameIfAllPassAlive,
  double proportionOfBoardArea, double temperature
) {
  NNResultBuf buf;

  //This gives us about 15 moves on average for 19x19.
  int numInitialMovesToPlay = (int)floor(gameRand.nextExponential() * (board.x_size * board.y_size * proportionOfBoardArea));
  assert(numInitialMovesToPlay >= 0);
  for(int i = 0; i<numInitialMovesToPlay; i++) {
    Loc loc = getGameInitializationMove(botB, botW, board, hist, pla, buf, gameRand, temperature);

    //Make the move!
    assert(hist.isLegal(board,loc,pla));
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
    pla = getOpp(pla);

    //Rarely, playing the random moves out this way will end the game
    if(doEndGameIfAllPassAlive)
      hist.endGameIfAllPassAlive(board);
    if(hist.isGameFinished)
      break;
  }
}

struct SearchLimitsThisMove {
  bool doCapVisitsPlayouts;
  int64_t numCapVisits;
  int64_t numCapPlayouts;
  bool clearBotBeforeSearchThisMove;
  bool removeRootNoise;
  float targetWeight;

  bool wasReduced;
};

static SearchLimitsThisMove getSearchLimitsThisMove(
  const Search* toMoveBot, const FancyModes& fancyModes, Rand& gameRand,
  const vector<double>& historicalMctsWinLossValues,
  bool clearBotBeforeSearch,
  bool allowReduce
) {
  bool doCapVisitsPlayouts = false;
  int64_t numCapVisits = toMoveBot->searchParams.maxVisits;
  int64_t numCapPlayouts = toMoveBot->searchParams.maxPlayouts;
  bool clearBotBeforeSearchThisMove = clearBotBeforeSearch;
  bool removeRootNoise = false;
  float targetWeight = 1.0f;
  bool wasReduced = false;

  if(allowReduce && fancyModes.cheapSearchProb > 0.0 && gameRand.nextBool(fancyModes.cheapSearchProb)) {
    if(fancyModes.cheapSearchVisits <= 0)
      throw StringError("fancyModes.cheapSearchVisits <= 0");
    doCapVisitsPlayouts = true;
    numCapVisits = fancyModes.cheapSearchVisits;
    numCapPlayouts = fancyModes.cheapSearchVisits;
    targetWeight *= fancyModes.cheapSearchTargetWeight;
    wasReduced = true;

    //If not recording cheap searches, do a few more things
    if(fancyModes.cheapSearchTargetWeight <= 0.0) {
      clearBotBeforeSearchThisMove = false;
      removeRootNoise = true;
    }
  }
  else if(allowReduce && fancyModes.reduceVisits) {
    if(fancyModes.reducedVisitsMin <= 0)
      throw StringError("fancyModes.reducedVisitsMin <= 0");
    if(historicalMctsWinLossValues.size() >= fancyModes.reduceVisitsThresholdLookback) {
      double minWinLossValue = 1e20;
      double maxWinLossValue = -1e20;
      for(int j = 0; j<fancyModes.reduceVisitsThresholdLookback; j++) {
        double winLossValue = historicalMctsWinLossValues[historicalMctsWinLossValues.size()-1-j];
        if(winLossValue < minWinLossValue)
          minWinLossValue = winLossValue;
        if(winLossValue > maxWinLossValue)
          maxWinLossValue = winLossValue;
      }
      assert(fancyModes.reduceVisitsThreshold >= 0.0);
      double signedMostExtreme = std::max(minWinLossValue,-maxWinLossValue);
      assert(signedMostExtreme <= 1.000001);
      if(signedMostExtreme > 1.0)
        signedMostExtreme = 1.0;
      double amountThrough = signedMostExtreme - fancyModes.reduceVisitsThreshold;
      if(amountThrough > 0) {
        double proportionThrough = amountThrough / (1.0 - fancyModes.reduceVisitsThreshold);
        assert(proportionThrough >= 0.0 && proportionThrough <= 1.0);
        double visitReductionProp = proportionThrough * proportionThrough;
        doCapVisitsPlayouts = true;
        numCapVisits = (int64_t)round(numCapVisits + visitReductionProp * ((double)fancyModes.reducedVisitsMin - (double)numCapVisits));
        numCapPlayouts = (int64_t)round(numCapPlayouts + visitReductionProp * ((double)fancyModes.reducedVisitsMin - (double)numCapPlayouts));
        targetWeight = (float)(targetWeight + visitReductionProp * (fancyModes.reducedVisitsWeight - targetWeight));
        numCapVisits = std::max(numCapVisits,(int64_t)fancyModes.reducedVisitsMin);
        numCapPlayouts = std::max(numCapPlayouts,(int64_t)fancyModes.reducedVisitsMin);
        wasReduced = true;
      }
    }
  }

  SearchLimitsThisMove limits;
  limits.doCapVisitsPlayouts = doCapVisitsPlayouts;
  limits.numCapVisits = numCapVisits;
  limits.numCapPlayouts = numCapPlayouts;
  limits.clearBotBeforeSearchThisMove = clearBotBeforeSearchThisMove;
  limits.removeRootNoise = removeRootNoise;
  limits.targetWeight = targetWeight;
  limits.wasReduced = wasReduced;
  return limits;
}

//Returns the move chosen
static Loc runBotWithLimits(
  Search* toMoveBot, Player pla, const FancyModes& fancyModes,
  const SearchLimitsThisMove& limits,
  Logger& logger,
  vector<double>* recordUtilities
) {
  if(limits.clearBotBeforeSearchThisMove)
    toMoveBot->clearSearch();

  Loc loc;

  //HACK - Disable LCB for making the move (it will still affect the policy target gen)
  bool lcb = toMoveBot->searchParams.useLcbForSelection;
  if(fancyModes.forSelfPlay) {
    toMoveBot->searchParams.useLcbForSelection = false;
  }

  if(limits.doCapVisitsPlayouts) {
    assert(limits.numCapVisits > 0);
    assert(limits.numCapPlayouts > 0);
    SearchParams oldParams = toMoveBot->searchParams;

    toMoveBot->searchParams.maxVisits = std::min(toMoveBot->searchParams.maxVisits, limits.numCapVisits);
    toMoveBot->searchParams.maxPlayouts = std::min(toMoveBot->searchParams.maxPlayouts, limits.numCapPlayouts);
    if(limits.removeRootNoise) {
      //Note - this is slightly sketchy to set the params directly. This works because
      //some of the parameters like FPU are basically stateless and will just affect future playouts
      //and because even stateful effects like rootNoiseEnabled and rootPolicyTemperature only affect
      //the root so when we step down in the tree we get a fresh start.
      toMoveBot->searchParams.rootNoiseEnabled = false;
      toMoveBot->searchParams.rootPolicyTemperature = 1.0;
      toMoveBot->searchParams.rootFpuLossProp = toMoveBot->searchParams.fpuLossProp;
      toMoveBot->searchParams.rootFpuReductionMax = toMoveBot->searchParams.fpuReductionMax;
      toMoveBot->searchParams.rootDesiredPerChildVisitsCoeff = 0.0;
    }

    //If we cleared the search, do a very short search first to get a good dynamic score utility center
    if(limits.clearBotBeforeSearchThisMove && toMoveBot->searchParams.maxVisits > 10 && toMoveBot->searchParams.maxPlayouts > 10) {
      int64_t oldMaxVisits = toMoveBot->searchParams.maxVisits;
      toMoveBot->searchParams.maxVisits = 10;
      toMoveBot->runWholeSearchAndGetMove(pla,logger,recordUtilities);
      toMoveBot->searchParams.maxVisits = oldMaxVisits;
    }
    loc = toMoveBot->runWholeSearchAndGetMove(pla,logger,recordUtilities);

    toMoveBot->searchParams = oldParams;
  }
  else {
    assert(!limits.removeRootNoise);
    loc = toMoveBot->runWholeSearchAndGetMove(pla,logger,recordUtilities);
  }

  //HACK - restore LCB so that it affects policy target gen
  if(fancyModes.forSelfPlay) {
    toMoveBot->searchParams.useLcbForSelection = lcb;
  }

  return loc;
}


//Run a game between two bots. It is OK if both bots are the same bot.
FinishedGameData* Play::runGame(
  const Board& startBoard, Player pla, const BoardHistory& startHist, ExtraBlackAndKomi extraBlackAndKomi,
  MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
  const string& searchRandSeed,
  bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
  Logger& logger, bool logSearchInfo, bool logMoves,
  int maxMovesPerGame, vector<std::atomic<bool>*>& stopConditions,
  const FancyModes& fancyModes, bool allowPolicyInit,
  Rand& gameRand,
  std::function<NNEvaluator*()>* checkForNewNNEval
) {
  Search* botB;
  Search* botW;
  if(botSpecB.botIdx == botSpecW.botIdx) {
    botB = new Search(botSpecB.baseParams, botSpecB.nnEval, searchRandSeed);
    botW = botB;
  }
  else {
    botB = new Search(botSpecB.baseParams, botSpecB.nnEval, searchRandSeed + "@B");
    botW = new Search(botSpecW.baseParams, botSpecW.nnEval, searchRandSeed + "@W");
  }

  FinishedGameData* gameData = runGame(
    startBoard, pla, startHist, extraBlackAndKomi,
    botSpecB, botSpecW,
    botB, botW,
    doEndGameIfAllPassAlive, clearBotBeforeSearch,
    logger, logSearchInfo, logMoves,
    maxMovesPerGame, stopConditions,
    fancyModes, allowPolicyInit,
    gameRand,
    checkForNewNNEval
  );

  if(botW != botB)
    delete botW;
  delete botB;

  return gameData;
}

FinishedGameData* Play::runGame(
  const Board& startBoard, Player pla, const BoardHistory& startHist, ExtraBlackAndKomi extraBlackAndKomi,
  MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
  Search* botB, Search* botW,
  bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
  Logger& logger, bool logSearchInfo, bool logMoves,
  int maxMovesPerGame, vector<std::atomic<bool>*>& stopConditions,
  const FancyModes& fancyModes, bool allowPolicyInit,
  Rand& gameRand,
  std::function<NNEvaluator*()>* checkForNewNNEval
) {
  FinishedGameData* gameData = new FinishedGameData();

  Board board(startBoard);
  BoardHistory hist(startHist);
  if(extraBlackAndKomi.extraBlack > 0) {
    double extraBlackTemperature = 1.0;
    bool adjustKomi = !gameRand.nextBool(fancyModes.noCompensateKomiProb);
    playExtraBlack(botB,logger,extraBlackAndKomi,board,hist,extraBlackTemperature,gameRand,adjustKomi,fancyModes.compensateKomiVisits);
    assert(hist.moveHistory.size() == 0);
  }

  vector<double>* recordUtilities = NULL;

  gameData->bName = botSpecB.botName;
  gameData->wName = botSpecW.botName;
  gameData->bIdx = botSpecB.botIdx;
  gameData->wIdx = botSpecW.botIdx;

  gameData->gameHash.hash0 = gameRand.nextUInt64();
  gameData->gameHash.hash1 = gameRand.nextUInt64();

  gameData->drawEquivalentWinsForWhite = botSpecB.baseParams.drawEquivalentWinsForWhite;

  gameData->numExtraBlack = extraBlackAndKomi.extraBlack;
  gameData->mode = 0;
  gameData->modeMeta1 = 0;
  gameData->modeMeta2 = 0;

  //In selfplay, record all the policy maps and evals and such as well for training data
  bool recordFullData = fancyModes.forSelfPlay;

  //Also record mcts utilities... DISABLED.
  //Originally accidentally disabled in a change, but it's probably not worth bringing back either.
  //TODO clean up utilityvar outputs
  //if(recordFullData) {
  //  recordUtilities = new vector<double>(256);
  //}

  //NOTE: that checkForNewNNEval might also cause the old nnEval to be invalidated and freed. This is okay since the only
  //references we both hold on to and use are the ones inside the bots here, and we replace the ones in the botSpecs.
  //We should NOT ever store an nnEval separately from these.
  auto maybeCheckForNewNNEval = [&botB,&botW,&botSpecB,&botSpecW,&checkForNewNNEval,&gameRand,&gameData](int nextTurnNumber) {
    //Check if we got a new nnEval, with some probability.
    //Randomized and low-probability so as to reduce contention in checking, while still probably happening in a timely manner.
    if(checkForNewNNEval != NULL && gameRand.nextBool(0.1)) {
      NNEvaluator* newNNEval = (*checkForNewNNEval)();
      if(newNNEval != NULL) {
        botB->setNNEval(newNNEval);
        if(botW != botB)
          botW->setNNEval(newNNEval);
        botSpecB.nnEval = newNNEval;
        botSpecW.nnEval = newNNEval;
        gameData->changedNeuralNets.push_back(new ChangedNeuralNet(newNNEval->getModelName(),nextTurnNumber));
      }
    }
  };

  if(fancyModes.initGamesWithPolicy && allowPolicyInit) {
    double proportionOfBoardArea = 1.0 / 25.0;
    double temperature = 1.0;
    initializeGameUsingPolicy(botB, botW, board, hist, pla, gameRand, doEndGameIfAllPassAlive, proportionOfBoardArea, temperature);
  }

  //Make sure there's some minimum tiny amount of data about how the encore phases work
  if(fancyModes.forSelfPlay && hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.encorePhase == 0 && gameRand.nextBool(0.04)) {
    //Play out to go a quite a bit later in the game.
    double proportionOfBoardArea = 0.25;
    double temperature = 2.0/3.0;
    initializeGameUsingPolicy(botB, botW, board, hist, pla, gameRand, doEndGameIfAllPassAlive, proportionOfBoardArea, temperature);

    if(!hist.isGameFinished) {
      //Even out the game
      adjustKomiToEven(botB, board, hist, pla, fancyModes.compensateKomiVisits, logger);

      //Randomly set to one of the encore phases
      //Since we played out the game a bunch we should get a good mix of stones that were present or not present at the start
      //of the second encore phase if we're going into the second.
      int encorePhase = gameRand.nextInt(1,2);
      hist.clear(board,pla,hist.rules,encorePhase);

      gameData->mode = 1;
      gameData->modeMeta1 = encorePhase;
      gameData->modeMeta2 = 0;
    }
  }

  //Set in the starting board and history to gameData and both bots
  gameData->startBoard = board;
  gameData->startHist = hist;
  gameData->startPla = pla;

  botB->setPosition(pla,board,hist);
  if(botB != botW)
    botW->setPosition(pla,board,hist);

  vector<Loc> locsBuf;
  vector<double> playSelectionValuesBuf;

  vector<SidePosition*> sidePositionsToSearch;

  vector<double> historicalMctsWinLossValues;
  vector<double> policySurpriseByTurn;

  //Main play loop
  for(int i = 0; i<maxMovesPerGame; i++) {
    if(doEndGameIfAllPassAlive)
      hist.endGameIfAllPassAlive(board);
    if(hist.isGameFinished)
      break;
    if(shouldStop(stopConditions))
      break;

    Search* toMoveBot = pla == P_BLACK ? botB : botW;

    SearchLimitsThisMove limits = getSearchLimitsThisMove(
      toMoveBot, fancyModes, gameRand, historicalMctsWinLossValues, clearBotBeforeSearch, true
    );
    Loc loc = runBotWithLimits(toMoveBot, pla, fancyModes, limits, logger, recordUtilities);

    if(loc == Board::NULL_LOC || !toMoveBot->isLegalStrict(loc,pla))
      failIllegalMove(toMoveBot,logger,board,loc);
    if(logSearchInfo)
      logSearch(toMoveBot,logger,loc);
    if(logMoves)
      logger.write("Move " + Global::intToString(hist.moveHistory.size()) + " made: " + Location::toString(loc,board));

    if(recordFullData) {
      vector<PolicyTargetMove>* policyTarget = new vector<PolicyTargetMove>();
      int64_t unreducedNumVisits = toMoveBot->getRootVisits();
      extractPolicyTarget(*policyTarget, toMoveBot, toMoveBot->rootNode, locsBuf, playSelectionValuesBuf);
      gameData->policyTargetsByTurn.push_back(PolicyTarget(policyTarget,unreducedNumVisits));
      gameData->targetWeightByTurn.push_back(limits.targetWeight);
      policySurpriseByTurn.push_back(toMoveBot->getPolicySurprise());

      ValueTargets whiteValueTargets;
      extractValueTargets(whiteValueTargets, toMoveBot, toMoveBot->rootNode, recordUtilities);
      gameData->whiteValueTargetsByTurn.push_back(whiteValueTargets);


      //Occasionally fork off some positions to evaluate
      Loc sidePositionForkLoc = Board::NULL_LOC;
      if(fancyModes.forkSidePositionProb > 0.0 && gameRand.nextBool(fancyModes.forkSidePositionProb)) {
        assert(toMoveBot->rootNode != NULL);
        assert(toMoveBot->rootNode->nnOutput != nullptr);
        Loc banMove = loc;
        sidePositionForkLoc = chooseRandomForkingMove(toMoveBot->rootNode->nnOutput.get(), board, hist, pla, gameRand, banMove);
        if(sidePositionForkLoc != Board::NULL_LOC) {
          SidePosition* sp = new SidePosition(board,hist,pla,gameData->changedNeuralNets.size());
          sp->hist.makeBoardMoveAssumeLegal(sp->board,sidePositionForkLoc,sp->pla,NULL);
          sp->pla = getOpp(sp->pla);
          if(sp->hist.isGameFinished) delete sp;
          else sidePositionsToSearch.push_back(sp);
        }
      }

      //If enabled, also record subtree positions from the search as training positions
      if(fancyModes.recordTreePositions && fancyModes.recordTreeTargetWeight > 0.0f) {
        if(fancyModes.recordTreeTargetWeight > 1.0f)
          throw StringError("fancyModes.recordTreeTargetWeight > 1.0f");

        recordTreePositions(
          gameData,
          board,hist,pla,
          toMoveBot,
          fancyModes.recordTreeThreshold,fancyModes.recordTreeTargetWeight,
          gameData->changedNeuralNets.size(),
          locsBuf,playSelectionValuesBuf,
          loc,sidePositionForkLoc
        );
      }
    }

    if(fancyModes.allowResignation || fancyModes.reduceVisits) {
      ReportedSearchValues values = toMoveBot->getRootValuesAssertSuccess();
      historicalMctsWinLossValues.push_back(values.winLossValue);
    }

    //Finally, make the move on the bots
    bool suc;
    suc = botB->makeMove(loc,pla);
    assert(suc);
    if(botB != botW) {
      suc = botW->makeMove(loc,pla);
      assert(suc);
    }
    (void)suc; //Avoid warning when asserts disabled

    //And make the move on our copy of the board
    assert(hist.isLegal(board,loc,pla));
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);

    //Check for resignation
    if(fancyModes.allowResignation && historicalMctsWinLossValues.size() >= fancyModes.resignConsecTurns) {
      if(fancyModes.resignThreshold > 0 || std::isnan(fancyModes.resignThreshold))
        throw StringError("fancyModes.resignThreshold > 0 || std::isnan(fancyModes.resignThreshold)");

      bool shouldResign = true;
      for(int j = 0; j<fancyModes.resignConsecTurns; j++) {
        double winLossValue = historicalMctsWinLossValues[historicalMctsWinLossValues.size()-j-1];
        Player resignPlayerThisTurn = C_EMPTY;
        if(winLossValue < fancyModes.resignThreshold)
          resignPlayerThisTurn = P_WHITE;
        else if(winLossValue > -fancyModes.resignThreshold)
          resignPlayerThisTurn = P_BLACK;

        if(resignPlayerThisTurn != pla) {
          shouldResign = false;
          break;
        }
      }

      if(shouldResign)
        hist.setWinnerByResignation(getOpp(pla));
    }

    int nextTurnNumber = hist.moveHistory.size();
    maybeCheckForNewNNEval(nextTurnNumber);

    pla = getOpp(pla);
  }

  gameData->endHist = hist;
  if(hist.isGameFinished)
    gameData->hitTurnLimit = false;
  else
    gameData->hitTurnLimit = true;

  if(recordFullData) {
    if(hist.isResignation)
      throw StringError("Recording full data currently incompatible with resignation");

    ValueTargets finalValueTargets;

    assert(gameData->finalFullArea == NULL);
    assert(gameData->finalOwnership == NULL);
    assert(gameData->finalSekiAreas == NULL);
    gameData->finalFullArea = new Color[Board::MAX_ARR_SIZE];
    gameData->finalOwnership = new Color[Board::MAX_ARR_SIZE];
    gameData->finalSekiAreas = new bool[Board::MAX_ARR_SIZE];

    if(hist.isGameFinished && hist.isNoResult) {
      finalValueTargets.win = 0.0f;
      finalValueTargets.loss = 0.0f;
      finalValueTargets.noResult = 1.0f;
      finalValueTargets.score = 0.0f;

      //Fill with empty so that we use "nobody owns anything" as the training target.
      //Although in practice actually the training normally weights by having a result or not, so it doesn't matter what we fill.
      std::fill(gameData->finalFullArea,gameData->finalFullArea+Board::MAX_ARR_SIZE,C_EMPTY);
      std::fill(gameData->finalOwnership,gameData->finalOwnership+Board::MAX_ARR_SIZE,C_EMPTY);
      std::fill(gameData->finalSekiAreas,gameData->finalSekiAreas+Board::MAX_ARR_SIZE,false);
    }
    else {
      //Relying on this to be idempotent, so that we can get the final territory map
      //We also do want to call this here to force-end the game if we crossed a move limit.
      hist.endAndScoreGameNow(board,gameData->finalOwnership);

      finalValueTargets.win = (float)ScoreValue::whiteWinsOfWinner(hist.winner, gameData->drawEquivalentWinsForWhite);
      finalValueTargets.loss = 1.0f - finalValueTargets.win;
      finalValueTargets.noResult = 0.0f;
      finalValueTargets.score = (float)ScoreValue::whiteScoreDrawAdjust(hist.finalWhiteMinusBlackScore,gameData->drawEquivalentWinsForWhite,hist);

      //Dummy values, doesn't matter since we didn't do a search for the final values
      finalValueTargets.mctsUtility1 = 0.0f;
      finalValueTargets.mctsUtility4 = 0.0f;
      finalValueTargets.mctsUtility16 = 0.0f;
      finalValueTargets.mctsUtility64 = 0.0f;
      finalValueTargets.mctsUtility256 = 0.0f;

      //Fill full and seki areas
      {
        board.calculateArea(gameData->finalFullArea, true, true, true, hist.rules.multiStoneSuicideLegal);

        Color* nonDameArea = new Color[Board::MAX_ARR_SIZE];
        int whiteMinusBlackNonDameTouchingRegionCount;
        board.calculateNonDameTouchingArea(nonDameArea,whiteMinusBlackNonDameTouchingRegionCount, false, false, hist.rules.multiStoneSuicideLegal);
        for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
          if(nonDameArea[i] == C_EMPTY && (gameData->finalFullArea[i] == C_BLACK || gameData->finalFullArea[i] == C_WHITE))
            gameData->finalSekiAreas[i] = true;
          else
            gameData->finalSekiAreas[i] = false;
        }
        delete nonDameArea;
      }
    }
    gameData->whiteValueTargetsByTurn.push_back(finalValueTargets);

    int dataXLen = fancyModes.dataXLen;
    int dataYLen = fancyModes.dataYLen;
    assert(dataXLen > 0);
    assert(dataYLen > 0);
    assert(gameData->finalWhiteScoring == NULL);

    gameData->finalWhiteScoring = new float[dataXLen*dataYLen];
    NNInputs::fillOwnership(board,gameData->finalOwnership,hist.rules.taxRule == Rules::TAX_ALL,dataXLen,dataYLen,gameData->finalWhiteScoring);

    gameData->hasFullData = true;
    gameData->dataXLen = dataXLen;
    gameData->dataYLen = dataYLen;

    //Compute desired expectation with which to write main game rows
    if(fancyModes.policySurpriseDataWeight > 0) {
      int numWeights = gameData->targetWeightByTurn.size();
      assert(numWeights == policySurpriseByTurn.size());

      double sumWeights = 0.0;
      double sumPolicySurpriseWeighted = 0.0;
      for(int i = 0; i<numWeights; i++) {
        float targetWeight = gameData->targetWeightByTurn[i];
        assert(targetWeight >= 0.0 && targetWeight <= 1.0);
        sumWeights += targetWeight;
        double policySurprise = policySurpriseByTurn[i];
        assert(policySurprise >= 0.0);
        sumPolicySurpriseWeighted += policySurprise * targetWeight;
      }

      if(sumWeights >= 1) {
        double averagePolicySurpriseWeighted = sumPolicySurpriseWeighted / sumWeights;

        //We also include some rows from non-full searches, if despite the shallow search
        //they were quite surprising to the policy.
        double thresholdToIncludeReduced = averagePolicySurpriseWeighted * 1.5;

        //Part of the weight will be proportional to surprisePropValue which is just policySurprise on normal rows
        //and the excess policySurprise beyond threshold on shallow searches.
        //First pass - we sum up the surpriseValue.
        double sumSurprisePropValue = 0.0;
        for(int i = 0; i<numWeights; i++) {
          float targetWeight = gameData->targetWeightByTurn[i];
          double policySurprise = policySurpriseByTurn[i];
          double surprisePropValue =
            targetWeight * policySurprise + (1-targetWeight) * std::max(0.0,policySurprise-thresholdToIncludeReduced);
          sumSurprisePropValue += surprisePropValue;
        }

        //Just in case, avoid div by 0
        if(sumSurprisePropValue > 1e-10) {
          for(int i = 0; i<numWeights; i++) {
            float targetWeight = gameData->targetWeightByTurn[i];
            double policySurprise = policySurpriseByTurn[i];
            double surprisePropValue =
              targetWeight * policySurprise + (1-targetWeight) * std::max(0.0,policySurprise-thresholdToIncludeReduced);
            double newValue =
              (1.0-fancyModes.policySurpriseDataWeight) * targetWeight
              + fancyModes.policySurpriseDataWeight * surprisePropValue * sumWeights / sumSurprisePropValue;
            gameData->targetWeightByTurn[i] = (float)(newValue);
          }
        }
      }
    }

    //Also evaluate all the side positions as well that we queued up to be searched
    NNResultBuf nnResultBuf;
    for(int i = 0; i<sidePositionsToSearch.size(); i++) {
      SidePosition* sp = sidePositionsToSearch[i];

      if(shouldStop(stopConditions)) {
        delete sp;
        continue;
      }

      Search* toMoveBot = sp->pla == P_BLACK ? botB : botW;
      toMoveBot->setPosition(sp->pla,sp->board,sp->hist);
      Loc responseLoc = toMoveBot->runWholeSearchAndGetMove(sp->pla,logger,recordUtilities);

      extractPolicyTarget(sp->policyTarget, toMoveBot, toMoveBot->rootNode, locsBuf, playSelectionValuesBuf);
      extractValueTargets(sp->whiteValueTargets, toMoveBot, toMoveBot->rootNode, recordUtilities);
      sp->targetWeight = 1.0;
      sp->unreducedNumVisits = toMoveBot->getRootVisits();
      sp->numNeuralNetChangesSoFar = gameData->changedNeuralNets.size();

      gameData->sidePositions.push_back(sp);

      //If enabled, also record subtree positions from the search as training positions
      if(fancyModes.recordTreePositions && fancyModes.recordTreeTargetWeight > 0.0f) {
        if(fancyModes.recordTreeTargetWeight > 1.0f)
          throw StringError("fancyModes.recordTreeTargetWeight > 1.0f");
        recordTreePositions(
          gameData,
          sp->board,sp->hist,sp->pla,
          toMoveBot,
          fancyModes.recordTreeThreshold,fancyModes.recordTreeTargetWeight,
          gameData->changedNeuralNets.size(),
          locsBuf,playSelectionValuesBuf,
          Board::NULL_LOC, Board::NULL_LOC
        );
      }

      //Occasionally continue the fork a second move or more, to provide some situations where the opponent has played "weird" moves not
      //only on the most immediate turn, but rather the turns before.
      if(gameRand.nextBool(0.25)) {
        if(responseLoc == Board::NULL_LOC || !sp->hist.isLegal(sp->board,responseLoc,sp->pla))
          failIllegalMove(toMoveBot,logger,sp->board,responseLoc);

        SidePosition* sp2 = new SidePosition(sp->board,sp->hist,sp->pla,gameData->changedNeuralNets.size());
        sp2->hist.makeBoardMoveAssumeLegal(sp2->board,responseLoc,sp2->pla,NULL);
        sp2->pla = getOpp(sp2->pla);
        if(sp2->hist.isGameFinished)
          delete sp2;
        else {
          Search* toMoveBot2 = sp2->pla == P_BLACK ? botB : botW;
          MiscNNInputParams nnInputParams;
          nnInputParams.drawEquivalentWinsForWhite = toMoveBot2->searchParams.drawEquivalentWinsForWhite;
          toMoveBot2->nnEvaluator->evaluate(
            sp2->board,sp2->hist,sp2->pla,nnInputParams,
            nnResultBuf,NULL,false,false
          );
          Loc banMove = Board::NULL_LOC;
          Loc forkLoc = chooseRandomForkingMove(nnResultBuf.result.get(), sp2->board, sp2->hist, sp2->pla, gameRand, banMove);
          if(forkLoc != Board::NULL_LOC) {
            sp2->hist.makeBoardMoveAssumeLegal(sp2->board,forkLoc,sp2->pla,NULL);
            sp2->pla = getOpp(sp2->pla);
            if(sp2->hist.isGameFinished) delete sp2;
            else sidePositionsToSearch.push_back(sp2);
          }
        }
      }

      maybeCheckForNewNNEval(gameData->endHist.moveHistory.size());
    }
  }

  if(recordUtilities != NULL) {
    delete recordUtilities;
  }

  return gameData;
}


void Play::maybeForkGame(
  const FinishedGameData* finishedGameData,
  const InitialPosition** nextInitialPosition,
  const FancyModes& fancyModes,
  Rand& gameRand,
  Search* bot,
  Logger& logger
) {
  if(nextInitialPosition == NULL)
    return;
  *nextInitialPosition = NULL;

  assert(finishedGameData->startHist.initialBoard.pos_hash == finishedGameData->endHist.initialBoard.pos_hash);
  assert(finishedGameData->startHist.initialPla == finishedGameData->endHist.initialPla);

  //TODO
  // if(fancyModes.sekiForkHack && gameRand.nextBool(0.3)) {

  // }

  //Just for conceptual simplicity, don't early fork games that started in the encore
  if(finishedGameData->startHist.encorePhase != 0)
    return;
  if(!gameRand.nextBool(fancyModes.earlyForkGameProb))
    return;

  Board board = finishedGameData->startHist.initialBoard;
  Player pla = finishedGameData->startHist.initialPla;
  BoardHistory hist(board,pla,finishedGameData->startHist.rules,finishedGameData->startHist.initialEncorePhase);

  //Pick a random move to fork from near the start
  int moveIdx = (int)floor(gameRand.nextExponential() * (fancyModes.earlyForkGameExpectedMoveProp * board.x_size * board.y_size));
  //Make sure it's prior to the last move, so we have a real place to fork from
  moveIdx = std::min(moveIdx,(int)(finishedGameData->endHist.moveHistory.size()-1));
  //Replay all those moves
  for(int i = 0; i<moveIdx; i++) {
    Loc loc = finishedGameData->endHist.moveHistory[i].loc;
    if(!hist.isLegal(board,loc,pla)) {
      cout << board << endl;
      cout << PlayerIO::colorToChar(pla) << endl;
      cout << Location::toString(loc,board) << endl;
      hist.printDebugInfo(cout,board);
      cout << endl;
      throw StringError("Illegal move when replaying to fork game?");
    }
    assert(finishedGameData->endHist.moveHistory[i].pla == pla);
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
    pla = getOpp(pla);

    //Just in case if somehow the game is over now, don't actually do anything
    if(hist.isGameFinished)
      return;
  }

  //Pick a move!

  if(fancyModes.earlyForkGameMaxChoices > NNPos::MAX_NN_POLICY_SIZE)
    throw StringError("fancyModes.earlyForkGameMaxChoices > NNPos::MAX_NN_POLICY_SIZE");

  //Generate a selection of a small random number of choices
  int numChoices = gameRand.nextInt(fancyModes.earlyForkGameMinChoices,fancyModes.earlyForkGameMaxChoices);
  assert(numChoices <= NNPos::MAX_NN_POLICY_SIZE);
  Loc possibleMoves[NNPos::MAX_NN_POLICY_SIZE];
  int numPossible = chooseRandomLegalMoves(board,hist,pla,gameRand,possibleMoves,numChoices);
  if(numPossible <= 0)
    return;

  //Try the one the value net thinks is best
  Loc bestMove = Board::NULL_LOC;
  double bestScore = 0.0;

  NNResultBuf buf;
  double drawEquivalentWinsForWhite = 0.5;
  for(int i = 0; i<numChoices; i++) {
    Loc loc = possibleMoves[i];
    Board copy = board;
    BoardHistory copyHist = hist;
    copyHist.makeBoardMoveAssumeLegal(copy,loc,pla,NULL);
    MiscNNInputParams nnInputParams;
    nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
    bot->nnEvaluator->evaluate(copy,copyHist,getOpp(pla),nnInputParams,buf,NULL,false,false);
    std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);
    double whiteScore = nnOutput->whiteScoreMean;
    if(bestMove == Board::NULL_LOC || (pla == P_WHITE && whiteScore > bestScore) || (pla == P_BLACK && whiteScore < bestScore)) {
      bestMove = loc;
      bestScore = whiteScore;
    }
  }

  //Make that move
  assert(hist.isLegal(board,bestMove,pla));
  hist.makeBoardMoveAssumeLegal(board,bestMove,pla,NULL);
  pla = getOpp(pla);

  //If the game is over now, don't actually do anything
  if(hist.isGameFinished)
    return;

  //Adjust komi to be fair for the new unusual move according to what the net thinks
  if(!gameRand.nextBool(fancyModes.noCompensateKomiProb)) {
    //Adjust komi to be fair for the handicap according to what the bot thinks. Iterate a few times in case
    //the neural net knows the bot isn't perfectly score maximizing.
    for(int i = 0; i<3; i++) {
      double finalWhiteScore = getWhiteScoreEstimate(bot, board, hist, pla, fancyModes.compensateKomiVisits, logger);
      double fairKomi = hist.rules.komi - finalWhiteScore;
      hist.setKomi(roundAndClipKomi(fairKomi,board));
    }
  }
  *nextInitialPosition = new InitialPosition(board,hist,pla);
}


GameRunner::GameRunner(ConfigParser& cfg, const string& sRandSeedBase, FancyModes fModes)
  :logSearchInfo(),logMoves(),maxMovesPerGame(),clearBotBeforeSearch(),
   searchRandSeedBase(sRandSeedBase),
   fancyModes(fModes),
   gameInit(NULL)
{
  logSearchInfo = cfg.getBool("logSearchInfo");
  logMoves = cfg.getBool("logMoves");
  maxMovesPerGame = cfg.getInt("maxMovesPerGame",1,1 << 30);
  clearBotBeforeSearch = cfg.contains("clearBotBeforeSearch") ? cfg.getBool("clearBotBeforeSearch") : false;

  //Initialize object for randomizing game settings
  gameInit = new GameInitializer(cfg);
}

GameRunner::~GameRunner() {
  delete gameInit;
}

FinishedGameData* GameRunner::runGame(
  int64_t gameIdx,
  const MatchPairer::BotSpec& bSpecB,
  const MatchPairer::BotSpec& bSpecW,
  const InitialPosition* initialPosition,
  const InitialPosition** nextInitialPosition,
  Logger& logger,
  vector<std::atomic<bool>*>& stopConditions,
  std::function<NNEvaluator*()>* checkForNewNNEval
) {
  MatchPairer::BotSpec botSpecB = bSpecB;
  MatchPairer::BotSpec botSpecW = bSpecW;

  if(nextInitialPosition != NULL)
    *nextInitialPosition = NULL;

  Board board; Player pla; BoardHistory hist; ExtraBlackAndKomi extraBlackAndKomi;
  if(fancyModes.forSelfPlay) {
    assert(botSpecB.botIdx == botSpecW.botIdx);
    SearchParams params = botSpecB.baseParams;
    gameInit->createGame(board,pla,hist,extraBlackAndKomi,params,initialPosition);
    botSpecB.baseParams = params;
    botSpecW.baseParams = params;
  }
  else {
    gameInit->createGame(board,pla,hist,extraBlackAndKomi,initialPosition);

    bool rulesWereSupported;
    if(botSpecB.nnEval != NULL) {
      botSpecB.nnEval->getSupportedRules(hist.rules,rulesWereSupported);
      if(!rulesWereSupported)
        logger.write("WARNING: Match is running bot on rules that it does not support: " + botSpecB.botName);
    }
    if(botSpecW.nnEval != NULL) {
      botSpecW.nnEval->getSupportedRules(hist.rules,rulesWereSupported);
      if(!rulesWereSupported)
        logger.write("WARNING: Match is running bot on rules that it does not support: " + botSpecW.botName);
    }
  }

  bool clearBotBeforeSearchThisGame = clearBotBeforeSearch;
  if(botSpecB.botIdx == botSpecW.botIdx) {
    //Avoid interactions between the two bots since they're the same.
    //Also in self-play this makes sure root noise is effective on each new search
    clearBotBeforeSearchThisGame = true;
  }

  string searchRandSeed = searchRandSeedBase + ":" + Global::int64ToString(gameIdx);
  Rand gameRand(searchRandSeed + ":" + "forGameRand");

  //In 2% of games, don't autoterminate the game upon all pass alive, to just provide a tiny bit of training data on positions that occur
  //as both players must wrap things up manually, because within the search we don't autoterminate games, meaning that the NN will get
  //called on positions that occur after the game would have been autoterminated.
  bool doEndGameIfAllPassAlive = fancyModes.forSelfPlay ? gameRand.nextBool(0.98) : true;
  //Allow initial moves via direct policy if we're not specially specifying the initial position for this game
  bool allowPolicyInit = initialPosition == NULL;

  Search* botB;
  Search* botW;
  if(botSpecB.botIdx == botSpecW.botIdx) {
    botB = new Search(botSpecB.baseParams, botSpecB.nnEval, searchRandSeed);
    botW = botB;
  }
  else {
    botB = new Search(botSpecB.baseParams, botSpecB.nnEval, searchRandSeed + "@B");
    botW = new Search(botSpecW.baseParams, botSpecW.nnEval, searchRandSeed + "@W");
  }

  FinishedGameData* finishedGameData = Play::runGame(
    board,pla,hist,extraBlackAndKomi,
    botSpecB,botSpecW,
    botB,botW,
    doEndGameIfAllPassAlive,clearBotBeforeSearchThisGame,
    logger,logSearchInfo,logMoves,
    maxMovesPerGame,stopConditions,
    fancyModes,allowPolicyInit,
    gameRand,
    checkForNewNNEval //Note that if this triggers, botSpecB and botSpecW will get updated, for use in maybeForkGame
  );

  if(initialPosition != NULL)
    finishedGameData->modeMeta2 = 1;

  //Make sure not to write the game if we terminated in the middle of this game!
  if(shouldStop(stopConditions)) {
    if(botW != botB)
      delete botW;
    delete botB;
    delete finishedGameData;
    return NULL;
  }

  assert(finishedGameData != NULL);

  Play::maybeForkGame(finishedGameData, nextInitialPosition, fancyModes, gameRand, botB, logger);

  if(botW != botB)
    delete botW;
  delete botB;

  return finishedGameData;
}
