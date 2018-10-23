#include "../core/global.h"
#include "../search/asyncbot.h"
#include "../program/play.h"

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
  :createGameMutex(),rand(),hasParams(false),baseParams()
{
  initShared(cfg);
  noResultStdev = 0.0;
  drawStdev = 0.0;
}

GameInitializer::GameInitializer(ConfigParser& cfg, const SearchParams& params)
  :createGameMutex(),rand(),hasParams(true),baseParams(params)
{
  initShared(cfg);
  noResultStdev = cfg.getDouble("noResultStdev",0.0,1.0);
  drawStdev = cfg.getDouble("drawStdev",0.0,1.0);
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
  assert(!hasParams);
}

void GameInitializer::createGame(Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack, SearchParams& params) {
  //Multiple threads will be calling this, and we have some mutable state such as rand.
  lock_guard<std::mutex> lock(createGameMutex);
  createGameSharedUnsynchronized(board,pla,hist,numExtraBlack);
  assert(hasParams);

  params = baseParams;
  if(noResultStdev > 1e-30) {
    params.noResultUtilityForWhite = baseParams.noResultUtilityForWhite + noResultStdev * nextGaussianTruncated(rand, 2.0);
    while(params.noResultUtilityForWhite < -1.0 || params.noResultUtilityForWhite > 1.0)
      params.noResultUtilityForWhite = baseParams.noResultUtilityForWhite + noResultStdev * nextGaussianTruncated(rand, 2.0);
  }
  if(drawStdev > 1e-30) {
    params.drawEquivalentWinsForWhite = baseParams.drawEquivalentWinsForWhite + drawStdev * nextGaussianTruncated(rand, 2.0);
    while(params.drawEquivalentWinsForWhite < 0.0 || params.drawEquivalentWinsForWhite > 1.0)
      params.drawEquivalentWinsForWhite = baseParams.drawEquivalentWinsForWhite + drawStdev * nextGaussianTruncated(rand, 2.0);
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

MatchPairer::MatchPairer(ConfigParser& cfg, bool forSelfPlay)
  :numBots(),secondaryBots(),nextMatchups(),rand(),numGamesStartedSoFar(0),numGamesTotal(),logGamesEvery(),getMatchupMutex()
{
  if(forSelfPlay) {
    numBots = 1;
    numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
  }
  else {
    numBots = cfg.getInt("numBots",1,1024);
    if(cfg.contains("secondaryBots"))
      secondaryBots = cfg.getInts("secondaryBots",0,4096);
    numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
  }

  logGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);
}

MatchPairer::~MatchPairer()
{}

bool MatchPairer::getMatchup(
  int64_t& gameIdx, Logger& logger,
  const NNEvaluator* nnEvalToLog, const vector<NNEvaluator*>* nnEvalsToLog
)
{
  int botIdxB;
  int botIdxW;
  assert(numBots == 1);
  return getMatchup(gameIdx,botIdxB,botIdxW,logger,nnEvalToLog,nnEvalsToLog);
}

bool MatchPairer::getMatchup(
  int64_t& gameIdx, int& botIdxB, int& botIdxW, Logger& logger,
  const NNEvaluator* nnEvalToLog, const vector<NNEvaluator*>* nnEvalsToLog
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
    if(nnEvalsToLog != NULL) {
      const vector<NNEvaluator*>& nnEvals = *nnEvalsToLog;
      for(int i = 0; i<nnEvals.size(); i++) {
        logger.write(nnEvals[i]->getModelFileName());
        logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
        logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
        logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
      }
    }
    if(nnEvalToLog != NULL) {
      logger.write(nnEvalToLog->getModelFileName());
      logger.write("NN rows: " + Global::int64ToString(nnEvalToLog->numRowsProcessed()));
      logger.write("NN batches: " + Global::int64ToString(nnEvalToLog->numBatchesProcessed()));
      logger.write("NN avg batch size: " + Global::doubleToString(nnEvalToLog->averageProcessedBatchSize()));
    }
  }

  pair<int,int> matchup = getMatchupPair();
  botIdxB = matchup.first;
  botIdxW = matchup.second;

  return true;
}

pair<int,int> MatchPairer::getMatchupPair() {
  if(nextMatchups.size() <= 0) {
    if(numBots == 1)
      return make_pair(0,0);
    for(int i = 0; i<numBots; i++) {
      for(int j = 0; j<numBots; j++) {
        if(i != j && !(contains(secondaryBots,i) && contains(secondaryBots,j))) {
          nextMatchups.push_back(make_pair(i,j));
        }
      }
    }
    //Shuffle
    for(int i = nextMatchups.size()-1; i >= 1; i--) {
      int j = (int)rand.nextUInt(i+1);
      pair<int,int> tmp = nextMatchups[i];
      nextMatchups[i] = nextMatchups[j];
      nextMatchups[j] = tmp;
    }
  }
  pair<int,int> matchup = nextMatchups.back();
  nextMatchups.pop_back();
  return matchup;
}

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
static void playExtraBlack(Search* bot, Logger& logger, int numExtraBlack, Board& board, BoardHistory& hist) {
  SearchParams oldParams = bot->searchParams;
  SearchParams tempParams = oldParams;
  tempParams.rootNoiseEnabled = false;
  tempParams.chosenMoveSubtract = 0.0;
  tempParams.chosenMoveTemperature = 1.0;
  tempParams.numThreads = 1;
  tempParams.maxVisits = 1;

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
  bot->setRootPassLegal(true);
}

//Run a game between two bots. It is OK if both bots are the same bot.
//Mutates the given board and history
void Play::runGame(
  Board& board, Player pla, BoardHistory& hist, int numExtraBlack, Search* botB, Search* botW,
  bool doEndGameIfAllPassAlive, bool clearBotAfterSearch,
  Logger& logger, bool logSearchInfo, bool logMoves,
  int maxMovesPerGame, std::atomic<bool>& stopSignalReceived,
  bool fancyModes,
  FinishedGameData* gameData, Rand* gameRand
) {
  if(numExtraBlack > 0)
    playExtraBlack(botB,logger,numExtraBlack,board,hist);

  vector<double>* recordUtilities = NULL;

  if(gameData != NULL) {
    gameData->gameHash.hash0 = gameRand->nextUInt64();
    gameData->gameHash.hash1 = gameRand->nextUInt64();

    gameData->mode = 0;
    gameData->modeMeta1 = 0;
    gameData->modeMeta2 = 0;

    assert(gameData->moves.size() == 0);
    recordUtilities = new vector<double>(256);
  }

  if(fancyModes) {
    //Make sure there's some minimum tiny amount of data about how the encore phases work
    if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.encorePhase == 0 && gameRand->nextBool(0.02)) {
      int encorePhase = gameRand->nextInt(1,2);
      hist.clear(board,pla,hist.rules,encorePhase);

      if(gameData != NULL) {
        gameData->mode = 1;
        gameData->modeMeta1 = 0;
        gameData->modeMeta2 = encorePhase;
      }
    }
  }

  //Set in the starting board and history to gameData and both bots
  if(gameData != NULL) {
    gameData->startBoard = board;
    gameData->startHist = hist;
    gameData->startPla = pla;
  }
  botB->setPosition(pla,board,hist);
  if(botB != botW)
    botW->setPosition(pla,board,hist);

  vector<Loc> locsBuf;
  vector<double> playSelectionValuesBuf;

  //Main play loop
  for(int i = 0; i<maxMovesPerGame; i++) {
    if(doEndGameIfAllPassAlive)
      hist.endGameIfAllPassAlive(board);
    if(hist.isGameFinished)
      break;
    if(stopSignalReceived.load())
      break;

    Search* toMoveBot = pla == P_BLACK ? botB : botW;
    Loc loc = toMoveBot->runWholeSearchAndGetMove(pla,logger,recordUtilities);

    if(loc == Board::NULL_LOC || !toMoveBot->isLegal(loc,pla))
      failIllegalMove(toMoveBot,logger,board,loc);
    if(logSearchInfo)
      logSearch(toMoveBot,logger,loc);
    if(logMoves)
      logger.write("Move " + Global::intToString(hist.moveHistory.size()) + " made: " + Location::toString(loc,board));

    if(gameData != NULL) {
      gameData->moves.push_back(Move(loc,pla));

      vector<PolicyTargetMove>* policyTargets = new vector<PolicyTargetMove>();
      locsBuf.clear();
      playSelectionValuesBuf.clear();
      double scaleMaxToAtLeast = 10.0;

      bool success = toMoveBot->getPlaySelectionValues(
        locsBuf,playSelectionValuesBuf,scaleMaxToAtLeast
      );
      assert(success);

      double winValue;
      double lossValue;
      double noResultValue;
      double scoreValue;
      success = toMoveBot->getRootValues(
        winValue,lossValue,noResultValue,scoreValue
      );
      assert(success);

      for(int moveIdx = 0; moveIdx<locsBuf.size(); moveIdx++) {
        double value = playSelectionValuesBuf[moveIdx];
        assert(value >= 0.0 && value < 30000.0); //Make sure we don't oveflow int16
        (*policyTargets).push_back(PolicyTargetMove(locsBuf[moveIdx],(int16_t)round(value)));
      }
      gameData->policyTargetsByTurn.push_back(policyTargets);

      ValueTargets valueTargets;
      valueTargets.win = winValue;
      valueTargets.loss = lossValue;
      valueTargets.noResult = noResultValue;
      valueTargets.scoreValue = scoreValue;

      //Not defined, only matters for the final value targets for the game result
      valueTargets.score = 0.0f;

      (void)gameRand; //TODO use this for sampling some conditional positions and forking?

      assert(recordUtilities != NULL);
      assert(recordUtilities->size() > 255);
      valueTargets.mctsUtility1 = (float)((*recordUtilities)[0]);
      valueTargets.mctsUtility4 = (float)((*recordUtilities)[3]);
      valueTargets.mctsUtility16 = (float)((*recordUtilities)[15]);
      valueTargets.mctsUtility64 = (float)((*recordUtilities)[63]);
      valueTargets.mctsUtility256 = (float)((*recordUtilities)[255]);

      gameData->whiteValueTargetsByTurn.push_back(valueTargets);
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
    pla = getOpp(pla);

  }


  if(gameData != NULL) {
    gameData->endHist = hist;

    if(hist.isGameFinished)
      gameData->hitTurnLimit = false;
    else
      gameData->hitTurnLimit = true;

    ValueTargets finalValueTargets;
    Color area[Board::MAX_ARR_SIZE];
    if(hist.isGameFinished && hist.isNoResult) {
      finalValueTargets.win = 0.0f;
      finalValueTargets.loss = 0.0f;
      finalValueTargets.noResult = 1.0f;
      finalValueTargets.scoreValue = 0.0f;
      finalValueTargets.score = 0.0f;
      std::fill(area,area+Board::MAX_ARR_SIZE,C_EMPTY);
    }
    else {
      //Relying on this to be idempotent, so that we can get the final territory map
      //We do want to call this here to force-end the game if we crossed a move limit.
      hist.endAndScoreGameNow(board,area);

      finalValueTargets.win = (float)NNOutput::whiteWinsOfWinner(hist.winner, gameData->drawEquivalentWinsForWhite);
      finalValueTargets.loss = 1.0f - finalValueTargets.win;
      finalValueTargets.noResult = 0.0f;
      finalValueTargets.scoreValue = NNOutput::whiteScoreValueOfScore(hist.finalWhiteMinusBlackScore, gameData->drawEquivalentWinsForWhite, board, hist);
      finalValueTargets.score = hist.finalWhiteMinusBlackScore + hist.whiteKomiAdjustmentForDraws(gameData->drawEquivalentWinsForWhite);

      //Dummy values, doesn't matter since we didn't do a search for the final values
      finalValueTargets.mctsUtility1 = 0.0f;
      finalValueTargets.mctsUtility4 = 0.0f;
      finalValueTargets.mctsUtility16 = 0.0f;
      finalValueTargets.mctsUtility64 = 0.0f;
      finalValueTargets.mctsUtility256 = 0.0f;
    }
    gameData->whiteValueTargetsByTurn.push_back(finalValueTargets);

    int posLen = gameData->posLen;
    std::fill(gameData->finalOwnership, gameData->finalOwnership + posLen*posLen, 0);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,posLen);
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(area[loc] == P_BLACK)
          gameData->finalOwnership[pos] = -1;
        else if(area[loc] == P_WHITE)
          gameData->finalOwnership[pos] = 1;
        else if(area[loc] == C_EMPTY)
          gameData->finalOwnership[pos] = 0;
        else
          assert(false);
      }
    }
  }

  if(recordUtilities != NULL)
    delete recordUtilities;

}

