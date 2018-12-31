#include "../core/global.h"
#include "../search/asyncbot.h"
#include "../program/play.h"
#include "../program/setup.h"

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

static pair<int,float> chooseExtraBlackAndKomi(
  float base, float stdev, double allowIntegerProb, double handicapProb, float handicapStoneValue, double bigStdevProb, float bigStdev, int bSize, Rand& rand
) {
  int extraBlack = 0;
  float komi = base;

  if(stdev > 0.0f)
    komi += stdev * (float)nextGaussianTruncated(rand,2.0);
  if(bigStdev > 0.0f && rand.nextDouble() < bigStdevProb)
    komi += bigStdev * (float)nextGaussianTruncated(rand,2.0);

  //Adjust for bSize, so that we don't give the same massive komis on smaller boards
  komi = base + (komi - base) * (float)bSize / 19.0f;

  //Add handicap stones compensated with komi
  int maxExtraBlack = getMaxExtraBlack(bSize);
  if(maxExtraBlack > 0 && rand.nextDouble() < handicapProb) {
    extraBlack += 1+rand.nextUInt(maxExtraBlack);
    komi += extraBlack * handicapStoneValue;
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

  assert((float)((int)(komi * 2)) == komi * 2);
  return make_pair(extraBlack,komi);
}

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
  allowedMultiStoneSuicideLegals = cfg.getBools("multiStoneSuicideLegals");

  for(size_t i = 0; i < allowedKoRuleStrs.size(); i++)
    allowedKoRules.push_back(Rules::parseKoRule(allowedKoRuleStrs[i]));
  for(size_t i = 0; i < allowedScoringRuleStrs.size(); i++)
    allowedScoringRules.push_back(Rules::parseScoringRule(allowedScoringRuleStrs[i]));

  if(allowedKoRules.size() <= 0)
    throw IOError("koRules must have at least one value in " + cfg.getFileName());
  if(allowedScoringRules.size() <= 0)
    throw IOError("scoringRules must have at least one value in " + cfg.getFileName());
  if(allowedMultiStoneSuicideLegals.size() <= 0)
    throw IOError("multiStoneSuicideLegals must have at least one value in " + cfg.getFileName());

  allowedBSizes = cfg.getInts("bSizes", 9, 19);
  allowedBSizeRelProbs = cfg.getDoubles("bSizeRelProbs",0.0,1e100);

  komiMean = cfg.getFloat("komiMean",-60.0f,60.0f);
  komiStdev = cfg.getFloat("komiStdev",0.0f,60.0f);
  komiAllowIntegerProb = cfg.getDouble("komiAllowIntegerProb",0.0,1.0);
  handicapProb = cfg.getDouble("handicapProb",0.0,1.0);
  handicapStoneValue = cfg.getFloat("handicapStoneValue",0.0f,30.0f);
  komiBigStdevProb = cfg.getDouble("komiBigStdevProb",0.0,1.0);
  komiBigStdev = cfg.getFloat("komiBigStdev",0.0f,60.0f);

  if(allowedBSizes.size() <= 0)
    throw IOError("bSizes must have at least one value in " + cfg.getFileName());
  if(allowedBSizes.size() != allowedBSizeRelProbs.size())
    throw IOError("bSizes and bSizeRelProbs must have same number of values in " + cfg.getFileName());
}

GameInitializer::~GameInitializer()
{}

void GameInitializer::createGame(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack) {
  //Multiple threads will be calling this, and we have some mutable state such as rand.
  lock_guard<std::mutex> lock(createGameMutex);
  createGameSharedUnsynchronized(board,pla,hist,numExtraBlack);
  if(noResultStdev != 0.0 || drawRandRadius != 0.0)
    throw StringError("GameInitializer::createGame called in a mode that doesn't support specifying noResultStdev or drawRandRadius");
}

void GameInitializer::createGame(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack, SearchParams& params) {
  //Multiple threads will be calling this, and we have some mutable state such as rand.
  lock_guard<std::mutex> lock(createGameMutex);
  createGameSharedUnsynchronized(board,pla,hist,numExtraBlack);

  if(noResultStdev > 1e-30) {
    double mean = params.noResultUtilityForWhite;
    params.noResultUtilityForWhite = mean + noResultStdev * nextGaussianTruncated(rand, 2.0);
    while(params.noResultUtilityForWhite < -1.0 || params.noResultUtilityForWhite > 1.0)
      params.noResultUtilityForWhite = mean + noResultStdev * nextGaussianTruncated(rand, 2.0);
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

void GameInitializer::createGameSharedUnsynchronized(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack) {
  int bSize = allowedBSizes[rand.nextUInt(allowedBSizeRelProbs.data(),allowedBSizeRelProbs.size())];
  board = Board(bSize,bSize);

  Rules rules;
  rules.koRule = allowedKoRules[rand.nextUInt(allowedKoRules.size())];
  rules.scoringRule = allowedScoringRules[rand.nextUInt(allowedScoringRules.size())];
  rules.multiStoneSuicideLegal = allowedMultiStoneSuicideLegals[rand.nextUInt(allowedMultiStoneSuicideLegals.size())];

  pair<int,float> extraBlackAndKomi = chooseExtraBlackAndKomi(
    komiMean, komiStdev, komiAllowIntegerProb, handicapProb, handicapStoneValue,
    komiBigStdevProb, komiBigStdev, bSize, rand
  );
  rules.komi = extraBlackAndKomi.second;

  pla = P_BLACK;
  hist.clear(board,pla,rules,0);
  numExtraBlack = extraBlackAndKomi.first;
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
)
  :numBots(nBots),
   botNames(bNames),
   nnEvals(nEvals),
   baseParamss(bParamss),
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
      logger.write(nnEvals[i]->getModelFileName());
      logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
      logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
      logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
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
      for(int j = 0; j<numBots; j++) {
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
   cheapSearchProb(0),cheapSearchVisits(0),cheapSearchTargetWeight(0.0f),
   reduceVisits(false),reduceVisitsThreshold(100.0),reduceVisitsThresholdLookback(1),reducedVisitsMin(0),reducedVisitsWeight(1.0f),
   recordTreePositions(false),recordTreeThreshold(0),recordTreeTargetWeight(0.0f),
   allowResignation(false),resignThreshold(0.0),resignConsecTurns(1)
{}
FancyModes::~FancyModes()
{}

//----------------------------------------------------------------------------------------------------------

static void failIllegalMove(Search* bot, Logger& logger, Board board, Loc loc) {
  ostringstream sout;
  sout << "Bot returned null location or illegal move!?!" << "\n";
  sout << board << "\n";
  sout << bot->getRootBoard() << "\n";
  sout << "Pla: " << playerToString(bot->getRootPla()) << "\n";
  sout << "Loc: " << Location::toString(loc,bot->getRootBoard()) << "\n";
  logger.write(sout.str());
  bot->getRootBoard().checkConsistency();
  assert(false);
}

static void logSearch(Search* bot, Logger& logger, Loc loc) {
  ostringstream sout;
  Board::printBoard(sout, bot->getRootBoard(), loc, &(bot->getRootHist().moveHistory));
  sout << "\n";
  sout << "Root visits: " << bot->numRootVisits() << "\n";
  sout << "PV: ";
  bot->printPV(sout, bot->rootNode, 25);
  sout << "\n";
  sout << "Tree:\n";
  bot->printTree(sout, bot->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10));
  logger.write(sout.str());
}


//Place black handicap stones, free placement
void Play::playExtraBlack(Search* bot, Logger& logger, int numExtraBlack, Board& board, BoardHistory& hist, double temperature) {
  SearchParams oldParams = bot->searchParams;
  bool oldRootPassLegal = bot->rootPassLegal;
  SearchParams tempParams = oldParams;
  tempParams.rootNoiseEnabled = false;
  tempParams.chosenMoveSubtract = 0.0;
  tempParams.chosenMoveTemperature = temperature;
  tempParams.numThreads = 1;
  tempParams.maxVisits = 1;

  //Toggle this since we cant have this set simultaneously with rootRootLegal false.
  tempParams.rootPruneUselessSuicides = false;

  Player pla = P_BLACK;
  bot->setPosition(pla,board,hist);
  bot->setParams(tempParams);
  bot->setRootPassLegal(false);

  for(int i = 0; i<numExtraBlack; i++) {
    Loc loc = bot->runWholeSearchAndGetMove(pla,logger,NULL);
    if(loc == Board::NULL_LOC || !bot->isLegal(loc,pla))
      failIllegalMove(bot,logger,board,loc);
    assert(hist.isLegal(board,loc,pla));
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
    hist.clear(board,pla,hist.rules,0);
    bot->setPosition(pla,board,hist);
  }

  bot->setParams(oldParams);
  bot->setRootPassLegal(oldRootPassLegal);
}

static bool shouldStop(vector<std::atomic<bool>*>& stopConditions) {
  for(int j = 0; j<stopConditions.size(); j++) {
    if(stopConditions[j]->load())
      return true;
  }
  return false;
}

static Loc chooseRandomLegalMove(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand) {
  int numLegalMoves = 0;
  Loc locs[Board::MAX_ARR_SIZE];
  for(Loc loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
    if(hist.isLegal(board,loc,pla)) {
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

static Loc chooseRandomPolicyMove(const NNOutput* nnOutput, const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, double temperature) {
  const float* policyProbs = nnOutput->policyProbs;
  int posLen = nnOutput->posLen;
  int numLegalMoves = 0;
  double relProbs[NNPos::MAX_NN_POLICY_SIZE];
  int locs[NNPos::MAX_NN_POLICY_SIZE];
  double probSum = 0.0;
  for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
    Loc loc = NNPos::posToLoc(pos,board.x_size,board.y_size,posLen);
    if(policyProbs[pos] > 0.0 && hist.isLegal(board,loc,pla)) {
      double relProb = (temperature == 1.0) ? policyProbs[pos] : pow(policyProbs[pos],1.0/temperature);
      relProbs[numLegalMoves] = relProb;
      locs[numLegalMoves] = loc;
      numLegalMoves += 1;
      probSum += relProb;
    }
  }

  //Just in case the policy map is somehow not consistent with the board position
  if(numLegalMoves > 0 && probSum > 0.01) {
    int n = gameRand.nextUInt(relProbs,numLegalMoves);
    return locs[n];
  }
  return Board::NULL_LOC;
}

static Loc chooseRandomForkingMove(const NNOutput* nnOutput, const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand) {
  double r = gameRand.nextDouble();
  //80% of the time, do a random temperature 1 policy move
  if(r < 0.8)
    return chooseRandomPolicyMove(nnOutput, board, hist, pla, gameRand, 1.0);
  //15% of the time, do a random temperature 2 policy move
  else if(r < 0.95)
    return chooseRandomPolicyMove(nnOutput, board, hist, pla, gameRand, 2.0);
  //5% of the time, do a random legal move
  else
    return chooseRandomLegalMove(board, hist, pla, gameRand);
}

static void extractPolicyTarget(vector<PolicyTargetMove>& buf, const Search* toMoveBot, const SearchNode* node, vector<Loc>& locsBuf, vector<double>& playSelectionValuesBuf) {
  double scaleMaxToAtLeast = 10.0;

  assert(node != NULL);
  bool success = toMoveBot->getPlaySelectionValues(*node,locsBuf,playSelectionValuesBuf,scaleMaxToAtLeast);
  assert(success);

  assert(locsBuf.size() == playSelectionValuesBuf.size());
  assert(locsBuf.size() <= toMoveBot->rootBoard.x_size * toMoveBot->rootBoard.y_size + 1);
  for(int moveIdx = 0; moveIdx<locsBuf.size(); moveIdx++) {
    double value = playSelectionValuesBuf[moveIdx];
    assert(value >= 0.0 && value < 30000.0); //Make sure we don't oveflow int16
    buf.push_back(PolicyTargetMove(locsBuf[moveIdx],(int16_t)round(value)));
  }
}

static void extractValueTargets(ValueTargets& buf, const Search* toMoveBot, const SearchNode* node, vector<double>* recordUtilities) {
  double winValue;
  double lossValue;
  double noResultValue;
  double staticScoreValue;
  double dynamicScoreValue;
  double expectedScore;
  bool success = toMoveBot->getNodeValues(*node,winValue,lossValue,noResultValue,staticScoreValue,dynamicScoreValue,expectedScore);
  assert(success);

  buf.win = winValue;
  buf.loss = lossValue;
  buf.noResult = noResultValue;
  buf.score = expectedScore;

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
    gameData->sidePositions.push_back(sp);
  }

  if(depth >= maxDepth)
    return;

  //Best child is the one with the largest number of visits, find it
  int bestChildIdx = 0;
  uint64_t bestChildVisits = 0;
  for(int i = 1; i<node->numChildren; i++) {
    const SearchNode* child = node->children[i];
    while(child->statsLock.test_and_set(std::memory_order_acquire));
    uint64_t numVisits = child->stats.visits;
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
    uint64_t numVisits = child->stats.visits;
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

//Run a game between two bots. It is OK if both bots are the same bot.
FinishedGameData* Play::runGame(
  const Board& initialBoard, Player pla, const BoardHistory& initialHist, int numExtraBlack,
  const MatchPairer::BotSpec& botSpecB, const MatchPairer::BotSpec& botSpecW,
  const string& searchRandSeed,
  bool doEndGameIfAllPassAlive, bool clearBotAfterSearch,
  Logger& logger, bool logSearchInfo, bool logMoves,
  int maxMovesPerGame, vector<std::atomic<bool>*>& stopConditions,
  FancyModes fancyModes, bool recordFullData, int dataPosLen,
  Rand& gameRand,
  std::function<NNEvaluator*()>* checkForNewNNEval
) {
  FinishedGameData* gameData = new FinishedGameData();

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

  Board board(initialBoard);
  BoardHistory hist(initialHist);
  if(numExtraBlack > 0) {
    double extraBlackTemperature = 1.0;
    playExtraBlack(botB,logger,numExtraBlack,board,hist,extraBlackTemperature);
  }

  vector<double>* recordUtilities = NULL;

  gameData->bName = botSpecB.botName;
  gameData->wName = botSpecW.botName;
  gameData->bIdx = botSpecB.botIdx;
  gameData->wIdx = botSpecW.botIdx;

  gameData->preStartBoard = board;
  gameData->gameHash.hash0 = gameRand.nextUInt64();
  gameData->gameHash.hash1 = gameRand.nextUInt64();

  gameData->drawEquivalentWinsForWhite = botSpecB.baseParams.drawEquivalentWinsForWhite;

  gameData->numExtraBlack = numExtraBlack;
  gameData->mode = 0;
  gameData->modeMeta1 = 0;
  gameData->modeMeta2 = 0;

  if(recordFullData)
    recordUtilities = new vector<double>(256);

  //NOTE: that checkForNewNNEval might also cause the old nnEval to be invalidated and freed. This is okay since the only
  //references we both hold on to and use are the ones inside the bots here.
  //We should NOT ever refer to botSpecB.nnEval or botSpecW.nnEval past this point, or store an nnEval separately from the bot.
  auto maybeCheckForNewNNEval = [&botB,&botW,&checkForNewNNEval,&gameRand,&gameData](int nextTurnNumber) {
    //Check if we got a new nnEval, with some probability.
    //Randomized and low-probability so as to reduce contention in checking, while still probably happening in a timely manner.
    if(checkForNewNNEval != NULL && gameRand.nextBool(0.1)) {
      NNEvaluator* newNNEval = (*checkForNewNNEval)();
      if(newNNEval != NULL) {
        botB->setNNEval(newNNEval);
        if(botW != botB)
          botW->setNNEval(newNNEval);
        gameData->changedNeuralNets.push_back(new ChangedNeuralNet(newNNEval->getModelName(),nextTurnNumber));
      }
    }
  };

  if(fancyModes.initGamesWithPolicy) {
    //Try playing a bunch of pure policy moves instead of playing from the start to initialize the board
    //and add entropy
    {
      NNResultBuf buf;
      NNEvaluator* nnEval = botB->nnEvaluator;

      double r = 0;
      while(r < 0.00000001)
        r = gameRand.nextDouble();
      r = -log(r);
      //This gives us about 15 moves on average for 19x19.
      int numInitialMovesToPlay = floor(r * board.x_size * board.y_size / 24.0);
      assert(numInitialMovesToPlay >= 0);
      for(int i = 0; i<numInitialMovesToPlay; i++) {
        double drawEquivalentWinsForWhite = (pla == P_BLACK ? botB : botW)->searchParams.drawEquivalentWinsForWhite;
        nnEval->evaluate(board,hist,pla,drawEquivalentWinsForWhite,buf,NULL,false,false);
        std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);

        //TODO maybe check the win chance after all these policy moves and try again if too lopsided
        //Or adjust the komi?
        vector<Loc> locs;
        vector<double> playSelectionValues;
        int posLen = nnOutput->posLen;
        assert(posLen >= board.x_size);
        assert(posLen >= board.y_size);
        assert(posLen > 0 && posLen < 100);
        int policySize = NNPos::getPolicySize(posLen);
        for(int movePos = 0; movePos<policySize; movePos++) {
          Loc moveLoc = NNPos::posToLoc(movePos,board.x_size,board.y_size,posLen);
          double policyProb = nnOutput->policyProbs[movePos];
          if(!hist.isLegal(board,moveLoc,pla) || policyProb <= 0)
            continue;
          locs.push_back(moveLoc);
          playSelectionValues.push_back(policyProb);
        }

        assert(playSelectionValues.size() > 0);

        //With a tiny probability, choose a uniformly random move instead of a policy move, to also
        //add a bit more outlierish variety
        uint32_t idxChosen;
        if(gameRand.nextBool(0.0002))
          idxChosen = gameRand.nextUInt(playSelectionValues.size());
        else
          idxChosen = gameRand.nextUInt(playSelectionValues.data(),playSelectionValues.size());
        Loc loc = locs[idxChosen];

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

    //Make sure there's some minimum tiny amount of data about how the encore phases work
    if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.encorePhase == 0 && gameRand.nextBool(0.02)) {
      int encorePhase = gameRand.nextInt(1,2);
      hist.clear(board,pla,hist.rules,encorePhase);

      gameData->preStartBoard = board;
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

  //Main play loop
  for(int i = 0; i<maxMovesPerGame; i++) {
    if(doEndGameIfAllPassAlive)
      hist.endGameIfAllPassAlive(board);
    if(hist.isGameFinished)
      break;
    if(shouldStop(stopConditions))
      break;

    Search* toMoveBot = pla == P_BLACK ? botB : botW;
    float targetWeight = 1.0;

    bool doCapVisitsPlayouts = false;
    uint64_t numCapVisits = toMoveBot->searchParams.maxVisits;
    uint64_t numCapPlayouts = toMoveBot->searchParams.maxPlayouts;
    if(fancyModes.cheapSearchProb > 0.0 && gameRand.nextBool(fancyModes.cheapSearchProb)) {
      if(fancyModes.cheapSearchVisits <= 0)
        throw StringError("fancyModes.cheapSearchVisits <= 0");
      doCapVisitsPlayouts = true;
      numCapVisits = fancyModes.cheapSearchVisits;
      numCapPlayouts = fancyModes.cheapSearchVisits;
      targetWeight *= fancyModes.cheapSearchTargetWeight;
    }
    else if(fancyModes.reduceVisits) {
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
          numCapVisits = (uint64_t)round(numCapVisits + visitReductionProp * ((double)fancyModes.reducedVisitsMin - (double)numCapVisits));
          numCapPlayouts = (uint64_t)round(numCapPlayouts + visitReductionProp * ((double)fancyModes.reducedVisitsMin - (double)numCapPlayouts));
          targetWeight = (float)(targetWeight + visitReductionProp * (fancyModes.reducedVisitsWeight - targetWeight));
          numCapVisits = std::max(numCapVisits,(uint64_t)fancyModes.reducedVisitsMin);
          numCapPlayouts = std::max(numCapPlayouts,(uint64_t)fancyModes.reducedVisitsMin);
        }
      }
    }

    Loc loc;

    if(doCapVisitsPlayouts) {
      assert(numCapVisits > 0);
      assert(numCapPlayouts > 0);
      uint64_t oldMaxVisits = toMoveBot->searchParams.maxVisits;
      uint64_t oldMaxPlayouts = toMoveBot->searchParams.maxPlayouts;
      toMoveBot->searchParams.maxVisits = std::min(oldMaxVisits, numCapVisits);
      toMoveBot->searchParams.maxPlayouts = std::min(oldMaxPlayouts, numCapPlayouts);
      loc = toMoveBot->runWholeSearchAndGetMove(pla,logger,recordUtilities);
      toMoveBot->searchParams.maxVisits = oldMaxVisits;
      toMoveBot->searchParams.maxPlayouts = oldMaxPlayouts;
    }
    else
      loc = toMoveBot->runWholeSearchAndGetMove(pla,logger,recordUtilities);

    if(loc == Board::NULL_LOC || !toMoveBot->isLegal(loc,pla))
      failIllegalMove(toMoveBot,logger,board,loc);
    if(logSearchInfo)
      logSearch(toMoveBot,logger,loc);
    if(logMoves)
      logger.write("Move " + Global::intToString(hist.moveHistory.size()) + " made: " + Location::toString(loc,board));

    if(recordFullData) {
      vector<PolicyTargetMove>* policyTarget = new vector<PolicyTargetMove>();
      extractPolicyTarget(*policyTarget, toMoveBot, toMoveBot->rootNode, locsBuf, playSelectionValuesBuf);
      gameData->policyTargetsByTurn.push_back(policyTarget);
      gameData->targetWeightByTurn.push_back(targetWeight);

      ValueTargets whiteValueTargets;
      extractValueTargets(whiteValueTargets, toMoveBot, toMoveBot->rootNode, recordUtilities);
      gameData->whiteValueTargetsByTurn.push_back(whiteValueTargets);


      //Occasionally fork off some positions to evaluate
      Loc sidePositionForkLoc = Board::NULL_LOC;
      if(fancyModes.forkSidePositionProb > 0.0 && gameRand.nextBool(fancyModes.forkSidePositionProb)) {
        assert(toMoveBot->rootNode != NULL);
        assert(toMoveBot->rootNode->nnOutput != nullptr);
        sidePositionForkLoc = chooseRandomForkingMove(toMoveBot->rootNode->nnOutput.get(), board, hist, pla, gameRand);
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
        assert(fancyModes.recordTreeTargetWeight <= 1.0f);
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
      double winValue;
      double lossValue;
      double noResultValue;
      double staticScoreValue;
      double dynamicScoreValue;
      double expectedScore;
      bool success = toMoveBot->getRootValues(winValue,lossValue,noResultValue,staticScoreValue,dynamicScoreValue,expectedScore);
      assert(success);

      double winLossValue = winValue - lossValue;
      assert(winLossValue > -1.01 && winLossValue < 1.01); //Sanity check, but allow generously for float imprecision
      if(winLossValue > 1.0) winLossValue = 1.0;
      if(winLossValue < -1.0) winLossValue = -1.0;
      historicalMctsWinLossValues.push_back(winLossValue);
    }

    //In many cases, we are using root-level noise, so we want to clear the search each time so that we don't
    //bias the next search with the result of the previous... and also to make each color's search independent of the other's.
    if(clearBotAfterSearch)
      toMoveBot->clearSearch();

    //Finally, make the move on the bots
    bool suc;
    suc = botB->makeMove(loc,pla);
    assert(suc);
    if(botB != botW) {
      suc = botW->makeMove(loc,pla);
      assert(suc);
    }

    //And make the move on our copy of the board
    assert(hist.isLegal(board,loc,pla));
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);

    //Check for resignation
    if(fancyModes.allowResignation && historicalMctsWinLossValues.size() >= fancyModes.resignConsecTurns) {
      assert(fancyModes.resignThreshold <= 0);
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
    assert(!hist.isResignation); //Recording full data currently incompatible with resignation

    ValueTargets finalValueTargets;
    Color area[Board::MAX_ARR_SIZE];
    if(hist.isGameFinished && hist.isNoResult) {
      finalValueTargets.win = 0.0f;
      finalValueTargets.loss = 0.0f;
      finalValueTargets.noResult = 1.0f;
      finalValueTargets.score = 0.0f;

      //Fill with empty so that we use "nobody owns anything" as the training target.
      //Although in practice actually the training normally weights by having a result or not, so it doesn't matter what we fill.
      std::fill(area,area+Board::MAX_ARR_SIZE,C_EMPTY);
    }
    else {
      //Relying on this to be idempotent, so that we can get the final territory map
      //We also do want to call this here to force-end the game if we crossed a move limit.
      hist.endAndScoreGameNow(board,area);

      finalValueTargets.win = (float)ScoreValue::whiteWinsOfWinner(hist.winner, gameData->drawEquivalentWinsForWhite);
      finalValueTargets.loss = 1.0f - finalValueTargets.win;
      finalValueTargets.noResult = 0.0f;
      finalValueTargets.score = ScoreValue::whiteScoreDrawAdjust(hist.finalWhiteMinusBlackScore,gameData->drawEquivalentWinsForWhite,hist);

      //Dummy values, doesn't matter since we didn't do a search for the final values
      finalValueTargets.mctsUtility1 = 0.0f;
      finalValueTargets.mctsUtility4 = 0.0f;
      finalValueTargets.mctsUtility16 = 0.0f;
      finalValueTargets.mctsUtility64 = 0.0f;
      finalValueTargets.mctsUtility256 = 0.0f;
    }
    gameData->whiteValueTargetsByTurn.push_back(finalValueTargets);

    assert(dataPosLen > 0);
    assert(gameData->finalWhiteOwnership == NULL);
    gameData->finalWhiteOwnership = new int8_t[dataPosLen*dataPosLen];
    std::fill(gameData->finalWhiteOwnership, gameData->finalWhiteOwnership + dataPosLen*dataPosLen, 0);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,dataPosLen);
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(area[loc] == P_BLACK)
          gameData->finalWhiteOwnership[pos] = -1;
        else if(area[loc] == P_WHITE)
          gameData->finalWhiteOwnership[pos] = 1;
        else if(area[loc] == C_EMPTY)
          gameData->finalWhiteOwnership[pos] = 0;
        else
          assert(false);
      }
    }

    gameData->hasFullData = true;
    gameData->posLen = dataPosLen;

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
      sp->numNeuralNetChangesSoFar = gameData->changedNeuralNets.size();

      gameData->sidePositions.push_back(sp);

      //If enabled, also record subtree positions from the search as training positions
      if(fancyModes.recordTreePositions && fancyModes.recordTreeTargetWeight > 0.0f) {
        assert(fancyModes.recordTreeTargetWeight <= 1.0f);
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
          toMoveBot2->nnEvaluator->evaluate(
            sp2->board,sp2->hist,sp2->pla,toMoveBot2->searchParams.drawEquivalentWinsForWhite,
            nnResultBuf,NULL,false,false
          );
          Loc forkLoc = chooseRandomForkingMove(nnResultBuf.result.get(), sp2->board, sp2->hist, sp2->pla, gameRand);
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

  if(recordUtilities != NULL)
    delete recordUtilities;

  if(botW != botB)
    delete botW;
  delete botB;

  return gameData;
}



GameRunner::GameRunner(ConfigParser& cfg, const string& sRandSeedBase, bool forSelfP, FancyModes fModes)
  :logSearchInfo(),logMoves(),forSelfPlay(forSelfP),maxMovesPerGame(),clearBotAfterSearch(),
   searchRandSeedBase(sRandSeedBase),
   fancyModes(fModes),
   gameInit(NULL)
{
  logSearchInfo = cfg.getBool("logSearchInfo");
  logMoves = cfg.getBool("logMoves");
  maxMovesPerGame = cfg.getInt("maxMovesPerGame",1,1 << 30);
  clearBotAfterSearch = cfg.contains("clearBotAfterSearch") ? cfg.getBool("clearBotAfterSearch") : false;

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
  Logger& logger,
  int dataPosLen,
  vector<std::atomic<bool>*>& stopConditions,
  std::function<NNEvaluator*()>* checkForNewNNEval
) {
  MatchPairer::BotSpec botSpecB = bSpecB;
  MatchPairer::BotSpec botSpecW = bSpecW;

  Board board; Player pla; BoardHistory hist; int numExtraBlack;
  if(forSelfPlay) {
    assert(botSpecB.botIdx == botSpecW.botIdx);
    SearchParams params = botSpecB.baseParams;
    gameInit->createGame(board,pla,hist,numExtraBlack,params);
    botSpecB.baseParams = params;
    botSpecW.baseParams = params;
  }
  else {
    gameInit->createGame(board,pla,hist,numExtraBlack);
  }

  bool clearBotAfterSearchThisGame = clearBotAfterSearch;
  if(botSpecB.botIdx == botSpecW.botIdx) {
    //Avoid interactions between the two bots since they're the same.
    //Also in self-play this makes sure root noise is effective on each new search
    clearBotAfterSearchThisGame = true;
  }

  string searchRandSeed = searchRandSeedBase + ":" + Global::int64ToString(gameIdx);
  Rand gameRand(searchRandSeed + ":" + "forGameRand");

  //In 2% of games, don't autoterminate the game upon all pass alive, to just provide a tiny bit of training data on positions that occur
  //as both players must wrap things up manually, because within the search we don't autoterminate games, meaning that the NN will get
  //called on positions that occur after the game would have been autoterminated.
  bool doEndGameIfAllPassAlive = forSelfPlay ? gameRand.nextBool(0.98) : true;
  //In selfplay, record all the policy maps and evals and such as well for training data
  bool recordFullData = forSelfPlay;
  FinishedGameData* finishedGameData = Play::runGame(
    board,pla,hist,numExtraBlack,
    botSpecB,botSpecW,
    searchRandSeed,
    doEndGameIfAllPassAlive,clearBotAfterSearchThisGame,
    logger,logSearchInfo,logMoves,
    maxMovesPerGame,stopConditions,
    fancyModes,recordFullData,dataPosLen,
    gameRand,
    checkForNewNNEval
  );

  //Make sure not to write the game if we terminated in the middle of this game!
  if(shouldStop(stopConditions)) {
    delete finishedGameData;
    return NULL;
  }

  assert(finishedGameData != NULL);
  return finishedGameData;
}

