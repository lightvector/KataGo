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

static int getMaxExtraBlack(double sqrtBoardArea) {
  if(sqrtBoardArea <= 10.00001)
    return 0;
  if(sqrtBoardArea <= 14.00001)
    return 1;
  if(sqrtBoardArea <= 16.00001)
    return 2;
  if(sqrtBoardArea <= 17.00001)
    return 3;
  if(sqrtBoardArea <= 18.00001)
    return 4;
  return 5;
}

static ExtraBlackAndKomi chooseExtraBlackAndKomi(
  float base, float stdev, double allowIntegerProb,
  double handicapProb,
  double bigStdevProb, float bigStdev, double sqrtBoardArea, Rand& rand
) {
  int extraBlack = 0;
  float komi = base;

  if(stdev > 0.0f)
    komi += stdev * (float)nextGaussianTruncated(rand,3.0);
  if(bigStdev > 0.0f && rand.nextBool(bigStdevProb))
    komi += bigStdev * (float)nextGaussianTruncated(rand,3.0);

  //Adjust for board size, so that we don't give the same massive komis on smaller boards
  komi = base + (komi - base) * (float)(sqrtBoardArea / 19.0);

  //Add handicap stones
  int maxExtraBlack = getMaxExtraBlack(sqrtBoardArea);
  if(maxExtraBlack > 0 && rand.nextBool(handicapProb)) {
    extraBlack += 1+rand.nextUInt(maxExtraBlack);
  }

  //Discretize komi
  float lower;
  float upper;
  if(rand.nextBool(allowIntegerProb)) {
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
  ExtraBlackAndKomi ret;
  ret.extraBlack = extraBlack;
  ret.komi = komi;
  ret.komiBase = base;
  //These two are set later
  ret.makeGameFair = false;
  ret.makeGameFairForEmptyBoard = false;
  return ret;
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


ForkData::~ForkData() {
  for(int i = 0; i<forks.size(); i++)
    delete forks[i];
  forks.clear();
  for(int i = 0; i<sekiForks.size(); i++)
    delete sekiForks[i];
  sekiForks.clear();
}

void ForkData::add(const InitialPosition* pos) {
  std::lock_guard<std::mutex> lock(mutex);
  forks.push_back(pos);
}
const InitialPosition* ForkData::get(Rand& rand) {
  std::lock_guard<std::mutex> lock(mutex);
  if(forks.size() <= 0)
    return NULL;
  int r = rand.nextUInt(forks.size());
  int last = forks.size()-1;
  const InitialPosition* pos = forks[r];
  forks[r] = forks[last];
  forks.resize(forks.size()-1);
  return pos;
}

void ForkData::addSeki(const InitialPosition* pos, Rand& rand) {
  std::unique_lock<std::mutex> lock(mutex);
  if(sekiForks.size() >= 1000) {
    int r = rand.nextUInt(sekiForks.size());
    const InitialPosition* oldPos = sekiForks[r];
    sekiForks[r] = pos;
    lock.unlock();
    delete oldPos;
  }
  else {
    sekiForks.push_back(pos);
  }
}
const InitialPosition* ForkData::getSeki(Rand& rand) {
  std::lock_guard<std::mutex> lock(mutex);
  if(sekiForks.size() <= 0)
    return NULL;
  int r = rand.nextUInt(sekiForks.size());
  int last = sekiForks.size()-1;
  const InitialPosition* pos = sekiForks[r];
  sekiForks[r] = sekiForks[last];
  sekiForks.resize(sekiForks.size()-1);
  return pos;
}

//------------------------------------------------------------------------------------------------

GameInitializer::GameInitializer(ConfigParser& cfg, Logger& logger)
  :createGameMutex(),rand()
{
  initShared(cfg,logger);
  noResultStdev = cfg.contains("noResultStdev") ? cfg.getDouble("noResultStdev",0.0,1.0) : 0.0;
  numExtraBlackFixed = cfg.contains("numExtraBlackFixed") ? cfg.getInt("numExtraBlackFixed",1,18) : 0;
  drawRandRadius = cfg.contains("drawRandRadius") ? cfg.getDouble("drawRandRadius",0.0,1.0) : 0.0;
}

void GameInitializer::initShared(ConfigParser& cfg, Logger& logger) {

  allowedKoRuleStrs = cfg.getStrings("koRules", Rules::koRuleStrings());
  allowedScoringRuleStrs = cfg.getStrings("scoringRules", Rules::scoringRuleStrings());
  allowedTaxRuleStrs = cfg.getStrings("taxRules", Rules::taxRuleStrings());
  allowedMultiStoneSuicideLegals = cfg.getBools("multiStoneSuicideLegals");
  allowedButtons = cfg.getBools("hasButtons");

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
  if(allowedButtons.size() <= 0)
    throw IOError("hasButtons must have at least one value in " + cfg.getFileName());

  {
    bool hasAreaScoring = false;
    for(int i = 0; i<allowedScoringRules.size(); i++)
      if(allowedScoringRules[i] == Rules::SCORING_AREA)
        hasAreaScoring = true;
    bool hasTrueButton = false;
    for(int i = 0; i<allowedButtons.size(); i++)
      if(allowedButtons[i])
        hasTrueButton = true;
    if(!hasAreaScoring && hasTrueButton)
      throw IOError("If scoringRules does not include AREA, hasButtons must be false in " + cfg.getFileName());
  }

  allowedBSizes = cfg.getInts("bSizes", 2, Board::MAX_LEN);
  allowedBSizeRelProbs = cfg.getDoubles("bSizeRelProbs",0.0,1e100);

  allowRectangleProb = cfg.contains("allowRectangleProb") ? cfg.getDouble("allowRectangleProb",0.0,1.0) : 0.0;

  if(!cfg.contains("komiMean") && !(cfg.contains("komiAuto") && cfg.getBool("komiAuto")))
    throw IOError("Must specify either komiMean=<komi value> or komiAuto=True in config");
  if(cfg.contains("komiMean") && (cfg.contains("komiAuto") && cfg.getBool("komiAuto")))
    throw IOError("Must specify only one of komiMean=<komi value> or komiAuto=True in config");

  komiMean = cfg.contains("komiMean") ? cfg.getFloat("komiMean",Rules::MIN_USER_KOMI,Rules::MAX_USER_KOMI) : 7.5f;
  komiStdev = cfg.contains("komiStdev") ? cfg.getFloat("komiStdev",0.0f,60.0f) : 0.0f;
  handicapProb = cfg.contains("handicapProb") ? cfg.getDouble("handicapProb",0.0,1.0) : 0.0;
  handicapCompensateKomiProb = cfg.contains("handicapCompensateKomiProb") ? cfg.getDouble("handicapCompensateKomiProb",0.0,1.0) : 0.0;
  komiBigStdevProb = cfg.contains("komiBigStdevProb") ? cfg.getDouble("komiBigStdevProb",0.0,1.0) : 0.0;
  komiBigStdev = cfg.contains("komiBigStdev") ? cfg.getFloat("komiBigStdev",0.0f,60.0f) : 10.0f;
  komiAuto = cfg.contains("komiAuto") ? cfg.getBool("komiAuto") : false;

  forkCompensateKomiProb = cfg.contains("forkCompensateKomiProb") ? cfg.getDouble("forkCompensateKomiProb",0.0,1.0) : handicapCompensateKomiProb;

  //Disabled because there are enough other things that can result in integer komi
  //such as komiAuto that this is very confusing
  //komiAllowIntegerProb = cfg.getDouble("komiAllowIntegerProb",0.0,1.0);
  komiAllowIntegerProb = 1.0;

  startPosesProb = 0.0;
  if(cfg.contains("startPosesFromSgfDir")) {
    startPoses.clear();
    startPosCumProbs.clear();
    startPosesProb = cfg.getDouble("startPosesProb",0.0,1.0);

    string dir = cfg.getString("startPosesFromSgfDir");
    double startPosesLoadProb = cfg.getDouble("startPosesLoadProb",0.0,1.0);
    double startPosesTurnWeightLambda = cfg.getDouble("startPosesTurnWeightLambda",-10,10);

    vector<string> files;
    std::function<bool(const string&)> fileFilter = [](const string& fileName) {
      return Global::isSuffix(fileName,".sgf");
    };
    Global::collectFiles(dir, fileFilter, files);
    logger.write("Found " + Global::uint64ToString(files.size()) + " sgf files in " + dir);
    std::set<Hash128> uniqueHashes;
    std::function<void(Sgf::PositionSample&)> posHandler = [startPosesLoadProb,this](Sgf::PositionSample& posSample) {
      if(rand.nextBool(startPosesLoadProb))
        startPoses.push_back(posSample);
    };
    for(size_t i = 0; i<files.size(); i++) {
      Sgf* sgf = Sgf::loadFile(files[i]);
      sgf->iterAllUniquePositions(uniqueHashes, posHandler);
      delete sgf;
    }
    logger.write("Loaded " + Global::uint64ToString(startPoses.size()) + " start positions from " + dir);

    int minInitialTurnNumber = 0;
    for(size_t i = 0; i<startPoses.size(); i++)
      minInitialTurnNumber = std::min(minInitialTurnNumber, startPoses[i].initialTurnNumber);

    startPosCumProbs.resize(startPoses.size());
    for(size_t i = 0; i<startPoses.size(); i++) {
      int64_t startTurn = startPoses[i].initialTurnNumber + (int64_t)startPoses[i].moves.size() - minInitialTurnNumber;
      startPosCumProbs[i] = exp(-startTurn * startPosesTurnWeightLambda);
    }
    for(size_t i = 0; i<startPoses.size(); i++) {
      if(!(startPosCumProbs[i] > -1e200 && startPosCumProbs[i] < 1e200)) {
        throw StringError("startPos found bad unnormalized probability: " + Global::doubleToString(startPosCumProbs[i]));
      }
    }
    for(size_t i = 1; i<startPoses.size(); i++)
      startPosCumProbs[i] += startPosCumProbs[i-1];

    if(startPoses.size() <= 0) {
      logger.write("No start positions loaded, disabling start position logic");
      startPosesProb = 0;
    }
    else {
      logger.write("Cumulative unnormalized probability for start poses: " + Global::doubleToString(startPosCumProbs[startPoses.size()-1]));
    }
  }

  if(allowedBSizes.size() <= 0)
    throw IOError("bSizes must have at least one value in " + cfg.getFileName());
  if(allowedBSizes.size() != allowedBSizeRelProbs.size())
    throw IOError("bSizes and bSizeRelProbs must have same number of values in " + cfg.getFileName());
}

GameInitializer::~GameInitializer()
{}

void GameInitializer::createGame(
  Board& board, Player& pla, BoardHistory& hist,
  ExtraBlackAndKomi& extraBlackAndKomi,
  const InitialPosition* initialPosition,
  const FancyModes& fancyModes,
  OtherGameProperties& otherGameProps
) {
  //Multiple threads will be calling this, and we have some mutable state such as rand.
  lock_guard<std::mutex> lock(createGameMutex);
  createGameSharedUnsynchronized(board,pla,hist,extraBlackAndKomi,initialPosition,fancyModes,otherGameProps);
  if(noResultStdev != 0.0 || drawRandRadius != 0.0)
    throw StringError("GameInitializer::createGame called in a mode that doesn't support specifying noResultStdev or drawRandRadius");
}

void GameInitializer::createGame(
  Board& board, Player& pla, BoardHistory& hist,
  ExtraBlackAndKomi& extraBlackAndKomi,
  SearchParams& params,
  const InitialPosition* initialPosition,
  const FancyModes& fancyModes,
  OtherGameProperties& otherGameProps
) {
  //Multiple threads will be calling this, and we have some mutable state such as rand.
  lock_guard<std::mutex> lock(createGameMutex);
  createGameSharedUnsynchronized(board,pla,hist,extraBlackAndKomi,initialPosition,fancyModes,otherGameProps);

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

Rules GameInitializer::randomizeScoringAndTaxRules(Rules rules, Rand& randToUse) const {
  rules.scoringRule = allowedScoringRules[randToUse.nextUInt(allowedScoringRules.size())];
  rules.taxRule = allowedTaxRules[randToUse.nextUInt(allowedTaxRules.size())];

  if(rules.scoringRule == Rules::SCORING_AREA)
    rules.hasButton = allowedButtons[randToUse.nextUInt(allowedButtons.size())];
  else
    rules.hasButton = false;

  return rules;
}

void GameInitializer::createGameSharedUnsynchronized(
  Board& board, Player& pla, BoardHistory& hist,
  ExtraBlackAndKomi& extraBlackAndKomi,
  const InitialPosition* initialPosition,
  const FancyModes& fancyModes,
  OtherGameProperties& otherGameProps
) {
  if(initialPosition != NULL) {
    board = initialPosition->board;
    hist = initialPosition->hist;
    pla = initialPosition->pla;

    //No handicap when starting from an initial position.
    double thisHandicapProb = 0.0;
    extraBlackAndKomi = chooseExtraBlackAndKomi(
      hist.rules.komi, komiStdev, komiAllowIntegerProb,
      thisHandicapProb,
      komiBigStdevProb, komiBigStdev, sqrt(board.x_size*board.y_size), rand
    );
    assert(extraBlackAndKomi.extraBlack == 0);
    hist.setKomi(extraBlackAndKomi.komi);
    otherGameProps.isSgfPos = false;
    otherGameProps.allowPolicyInit = false; //On initial positions, don't play extra moves at start
    otherGameProps.isFork = true;
    extraBlackAndKomi.makeGameFair = rand.nextBool(forkCompensateKomiProb);
    extraBlackAndKomi.makeGameFairForEmptyBoard = false;
    return;
  }

  double makeGameFairProb = 0.0;

  int xSizeIdx = rand.nextUInt(allowedBSizeRelProbs.data(),allowedBSizeRelProbs.size());
  int ySizeIdx = xSizeIdx;
  if(allowRectangleProb > 0 && rand.nextBool(allowRectangleProb))
    ySizeIdx = rand.nextUInt(allowedBSizeRelProbs.data(),allowedBSizeRelProbs.size());

  Rules rules;
  rules.koRule = allowedKoRules[rand.nextUInt(allowedKoRules.size())];
  rules.scoringRule = allowedScoringRules[rand.nextUInt(allowedScoringRules.size())];
  rules.taxRule = allowedTaxRules[rand.nextUInt(allowedTaxRules.size())];
  rules.multiStoneSuicideLegal = allowedMultiStoneSuicideLegals[rand.nextUInt(allowedMultiStoneSuicideLegals.size())];

  if(rules.scoringRule == Rules::SCORING_AREA)
    rules.hasButton = allowedButtons[rand.nextUInt(allowedButtons.size())];
  else
    rules.hasButton = false;

  if(startPosesProb > 0 && rand.nextBool(startPosesProb)) {
    assert(startPoses.size() > 0);
    size_t r = rand.nextIndexCumulative(startPosCumProbs.data(),startPosCumProbs.size());
    assert(r < startPosCumProbs.size());
    const Sgf::PositionSample& startPos = startPoses[r];
    board = startPos.board;
    pla = startPos.nextPla;
    hist.clear(board,pla,rules,0);
    hist.setInitialTurnNumber(startPos.initialTurnNumber);
    for(size_t i = 0; i<startPos.moves.size(); i++) {
      hist.makeBoardMoveAssumeLegal(board,startPos.moves[i].loc,startPos.moves[i].pla,NULL);
      pla = getOpp(startPos.moves[i].pla);
    }
    //No handicap when starting from a sampled position.
    double thisHandicapProb = 0.0;
    extraBlackAndKomi = chooseExtraBlackAndKomi(
      komiMean, komiStdev, komiAllowIntegerProb,
      thisHandicapProb,
      komiBigStdevProb, komiBigStdev, sqrt(board.x_size*board.y_size), rand
    );
    otherGameProps.isSgfPos = true;
    otherGameProps.allowPolicyInit = false; //On sgf positions, don't play extra moves at start
    otherGameProps.isFork = false;
    makeGameFairProb = forkCompensateKomiProb;
  }
  else {
    int xSize = allowedBSizes[xSizeIdx];
    int ySize = allowedBSizes[ySizeIdx];
    board = Board(xSize,ySize);

    extraBlackAndKomi = chooseExtraBlackAndKomi(
      komiMean, komiStdev, komiAllowIntegerProb,
      handicapProb,
      komiBigStdevProb, komiBigStdev, sqrt(board.x_size*board.y_size), rand
    );
    rules.komi = extraBlackAndKomi.komi;
    if(extraBlackAndKomi.extraBlack > 0 && numExtraBlackFixed != 0)
      extraBlackAndKomi.extraBlack = numExtraBlackFixed;

    pla = P_BLACK;
    hist.clear(board,pla,rules,0);
    otherGameProps.isSgfPos = false;
    otherGameProps.allowPolicyInit = true;
    otherGameProps.isFork = false;
    makeGameFairProb = extraBlackAndKomi.extraBlack > 0 ? handicapCompensateKomiProb : 0.0;
  }

  double asymmetricProb = (extraBlackAndKomi.extraBlack > 0) ? fancyModes.handicapAsymmetricPlayoutProb : fancyModes.normalAsymmetricPlayoutProb;
  if(asymmetricProb > 0 && rand.nextBool(asymmetricProb)) {
    assert(fancyModes.maxAsymmetricRatio >= 1.0);
    double maxNumDoublings = log(fancyModes.maxAsymmetricRatio) / log(2.0);
    double numDoublings = rand.nextDouble(maxNumDoublings);
    if(extraBlackAndKomi.extraBlack > 0 || rand.nextBool(0.5)) {
      otherGameProps.playoutDoublingAdvantagePla = C_WHITE;
      otherGameProps.playoutDoublingAdvantage = numDoublings;
    }
    else {
      otherGameProps.playoutDoublingAdvantagePla = C_BLACK;
      otherGameProps.playoutDoublingAdvantage = numDoublings;
    }
    makeGameFairProb = std::max(makeGameFairProb,fancyModes.minAsymmetricCompensateKomiProb);
  }

  if(komiAuto) {
    if(makeGameFairProb > 0.0)
      extraBlackAndKomi.makeGameFair = rand.nextBool(makeGameFairProb);
    extraBlackAndKomi.makeGameFairForEmptyBoard = !extraBlackAndKomi.makeGameFair;
  }
  else {
    if(makeGameFairProb > 0.0)
      extraBlackAndKomi.makeGameFair = rand.nextBool(makeGameFairProb);
    extraBlackAndKomi.makeGameFairForEmptyBoard = false;
  }
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
   compensateKomiVisits(20),estimateLeadVisits(10),estimateLeadProb(0.0),
   earlyForkGameProb(0.0),earlyForkGameExpectedMoveProp(0.0),forkGameProb(0.0),forkGameMinChoices(1),earlyForkGameMaxChoices(1),forkGameMaxChoices(1),
   sekiForkHack(false),fancyKomiVarying(false),
   cheapSearchProb(0),cheapSearchVisits(0),cheapSearchTargetWeight(0.0f),
   reduceVisits(false),reduceVisitsThreshold(100.0),reduceVisitsThresholdLookback(1),reducedVisitsMin(0),reducedVisitsWeight(1.0f),
   policySurpriseDataWeight(0.0),
   recordTreePositions(false),recordTreeThreshold(0),recordTreeTargetWeight(0.0f),
   allowResignation(false),resignThreshold(0.0),resignConsecTurns(1),
   forSelfPlay(false),dataXLen(-1),dataYLen(-1),
   handicapAsymmetricPlayoutProb(0.0),normalAsymmetricPlayoutProb(0.0),maxAsymmetricRatio(2.0)
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

static float roundAndClipKomi(double unrounded, const Board& board, bool looseClipping) {
  //Just in case, make sure komi is reasonable
  float range = looseClipping ? 40.0f + board.x_size * board.y_size : 40.0f + 0.5f * board.x_size * board.y_size;
  if(unrounded < -range)
    unrounded = -range;
  if(unrounded > range)
    unrounded = range;
  return (float)(0.5 * round(2.0 * unrounded));
}

static ReportedSearchValues getWhiteScoreValues(
  Search* bot,
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  int numVisits,
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
  bot->runWholeSearch(pla,logger,NULL);

  ReportedSearchValues values = bot->getRootValuesAssertSuccess();
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

  ReportedSearchValues values0 = getWhiteScoreValues(botB, board, hist, pla, numVisits, logger, otherGameProps);
  double finalLead = values0.lead;
  double finalWinLoss = values0.winLossValue;

  //If we have a second bot, average the two
  if(botW != NULL && botW != botB) {
    ReportedSearchValues values1 = getWhiteScoreValues(botW, board, hist, pla, numVisits, logger, otherGameProps);
    finalLead = 0.5 * (values0.lead + values1.lead);
    finalWinLoss = 0.5 * (values0.winLossValue + values1.winLossValue);
  }
  std::pair<double,double> result = std::make_pair(finalLead,finalWinLoss);
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
  for(int i = 0; i<3; i++) {
    std::pair<float,float> result = evalKomi(scoreWLCache,botB,botW,board,hist,pla,numVisits,logger,otherGameProps,hist.rules.komi);
    double finalLead = result.first;
    double finalWinLoss = result.second;
    //Shift by the predicted lead
    double shift = -finalLead;
    //Under no situations should the shift be bigger in absolute value than the last shift
    if(i > 0 && abs(shift) > abs(lastShift)) {
      if(shift < 0) shift = -abs(lastShift);
      else if(shift > 0) shift = abs(lastShift);
    }
    lastShift = shift;

    //If the score and winrate would like to move in opposite directions, quit immediately.
    if((shift > 0 && finalWinLoss > 0) || (shift < 0 && finalLead < 0))
      break;

    // cout << "Shifting by " << shift << endl;
    float fairKomi = roundAndClipKomi(hist.rules.komi + shift, board, looseClipping);
    hist.setKomi(fairKomi);

    //After a small shift, break out to the binary search.
    if(abs(shift) < 16.0)
      break;
  }

  //Try a small window and do a binary search
  auto evalWinLoss = [&](double delta) {
    double newKomi = hist.rules.komi + delta;
    double winLoss = evalKomi(scoreWLCache,botB,botW,board,hist,pla,numVisits,logger,otherGameProps,roundAndClipKomi(newKomi,board,looseClipping)).second;
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

void Play::adjustKomiToEven(
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
  hist.setKomi(roundAndClipKomi(newKomi,board,looseClipping));
}

float Play::computeLead(
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
    double winLoss = evalKomi(scoreWLCache,botB,botW,board,hist,pla,numVisits,logger,otherGameProps,roundAndClipKomi(newKomi,board,looseClipping)).second;
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

static void extractValueTargets(ValueTargets& buf, const Search* toMoveBot, const SearchNode* node) {
  ReportedSearchValues values;
  bool success = toMoveBot->getNodeValues(*node,values);
  assert(success);
  (void)success; //Avoid warning when asserts are disabled

  buf.win = (float)values.winValue;
  buf.loss = (float)values.lossValue;
  buf.noResult = (float)values.noResultValue;
  buf.score = (float)values.expectedScore;
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
    extractValueTargets(sp->whiteValueTargets, toMoveBot, node);
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
  bool doAlterVisitsPlayouts;
  int64_t numAlterVisits;
  int64_t numAlterPlayouts;
  bool clearBotBeforeSearchThisMove;
  bool removeRootNoise;
  float targetWeight;

  //Note: these two behave slightly differently than the ones in searchParams - derived from OtherGameProperties
  //game, they make the playouts *actually* vary instead of only making the neural net think they do.
  double playoutDoublingAdvantage;
  Player playoutDoublingAdvantagePla;
};

static SearchLimitsThisMove getSearchLimitsThisMove(
  const Search* toMoveBot, Player pla, const FancyModes& fancyModes, Rand& gameRand,
  const vector<double>& historicalMctsWinLossValues,
  bool clearBotBeforeSearch,
  OtherGameProperties otherGameProps
) {
  bool doAlterVisitsPlayouts = false;
  int64_t numAlterVisits = toMoveBot->searchParams.maxVisits;
  int64_t numAlterPlayouts = toMoveBot->searchParams.maxPlayouts;
  bool clearBotBeforeSearchThisMove = clearBotBeforeSearch;
  bool removeRootNoise = false;
  float targetWeight = 1.0f;
  double playoutDoublingAdvantage = 0.0;
  Player playoutDoublingAdvantagePla = C_EMPTY;

  if(fancyModes.cheapSearchProb > 0.0 && gameRand.nextBool(fancyModes.cheapSearchProb)) {
    if(fancyModes.cheapSearchVisits <= 0)
      throw StringError("fancyModes.cheapSearchVisits <= 0");
    if(fancyModes.cheapSearchVisits > toMoveBot->searchParams.maxVisits ||
       fancyModes.cheapSearchVisits > toMoveBot->searchParams.maxPlayouts)
      throw StringError("fancyModes.cheapSearchVisits > maxVisits and/or maxPlayouts");

    doAlterVisitsPlayouts = true;
    numAlterVisits = std::min(numAlterVisits,(int64_t)fancyModes.cheapSearchVisits);
    numAlterPlayouts = std::min(numAlterPlayouts,(int64_t)fancyModes.cheapSearchVisits);
    targetWeight *= fancyModes.cheapSearchTargetWeight;

    //If not recording cheap searches, do a few more things
    if(fancyModes.cheapSearchTargetWeight <= 0.0) {
      clearBotBeforeSearchThisMove = false;
      removeRootNoise = true;
    }
  }
  else if(fancyModes.reduceVisits) {
    if(fancyModes.reducedVisitsMin <= 0)
      throw StringError("fancyModes.reducedVisitsMin <= 0");
    if(fancyModes.reducedVisitsMin > toMoveBot->searchParams.maxVisits ||
       fancyModes.reducedVisitsMin > toMoveBot->searchParams.maxPlayouts)
      throw StringError("fancyModes.reducedVisitsMin > maxVisits and/or maxPlayouts");

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
        doAlterVisitsPlayouts = true;
        numAlterVisits = (int64_t)round(numAlterVisits + visitReductionProp * ((double)fancyModes.reducedVisitsMin - (double)numAlterVisits));
        numAlterPlayouts = (int64_t)round(numAlterPlayouts + visitReductionProp * ((double)fancyModes.reducedVisitsMin - (double)numAlterPlayouts));
        targetWeight = (float)(targetWeight + visitReductionProp * (fancyModes.reducedVisitsWeight - targetWeight));
        numAlterVisits = std::max(numAlterVisits,(int64_t)fancyModes.reducedVisitsMin);
        numAlterPlayouts = std::max(numAlterPlayouts,(int64_t)fancyModes.reducedVisitsMin);
      }
    }
  }

  if(otherGameProps.playoutDoublingAdvantage != 0.0 && otherGameProps.playoutDoublingAdvantagePla != C_EMPTY) {
    assert(pla == otherGameProps.playoutDoublingAdvantagePla || getOpp(pla) == otherGameProps.playoutDoublingAdvantagePla);

    playoutDoublingAdvantage = otherGameProps.playoutDoublingAdvantage;
    playoutDoublingAdvantagePla = otherGameProps.playoutDoublingAdvantagePla;

    double factor = pow(2.0, otherGameProps.playoutDoublingAdvantage);
    if(pla == otherGameProps.playoutDoublingAdvantagePla)
      factor = 2.0 * (factor / (factor + 1.0));
    else
      factor = 2.0 * (1.0 / (factor + 1.0));


    doAlterVisitsPlayouts = true;
    //Set this back to true - we need to always clear the search if we are doing asymmetric playouts
    clearBotBeforeSearchThisMove = true;
    numAlterVisits = (int64_t)round(numAlterVisits * factor);
    numAlterPlayouts = (int64_t)round(numAlterPlayouts * factor);

    //Hardcoded limit here to ensure sanity
    if(numAlterVisits < 5)
      throw StringError("ERROR: asymmetric playout doubling resulted in fewer than 5 visits");
    if(numAlterPlayouts < 5)
      throw StringError("ERROR: asymmetric playout doubling resulted in fewer than 5 playouts");
  }

  SearchLimitsThisMove limits;
  limits.doAlterVisitsPlayouts = doAlterVisitsPlayouts;
  limits.numAlterVisits = numAlterVisits;
  limits.numAlterPlayouts = numAlterPlayouts;
  limits.clearBotBeforeSearchThisMove = clearBotBeforeSearchThisMove;
  limits.removeRootNoise = removeRootNoise;
  limits.targetWeight = targetWeight;
  limits.playoutDoublingAdvantage = playoutDoublingAdvantage;
  limits.playoutDoublingAdvantagePla = playoutDoublingAdvantagePla;
  return limits;
}

//Returns the move chosen
static Loc runBotWithLimits(
  Search* toMoveBot, Player pla, const FancyModes& fancyModes,
  const SearchLimitsThisMove& limits,
  Logger& logger
) {
  if(limits.clearBotBeforeSearchThisMove)
    toMoveBot->clearSearch();

  Loc loc;

  //HACK - Disable LCB for making the move (it will still affect the policy target gen)
  bool lcb = toMoveBot->searchParams.useLcbForSelection;
  if(fancyModes.forSelfPlay) {
    toMoveBot->searchParams.useLcbForSelection = false;
  }

  if(limits.doAlterVisitsPlayouts) {
    assert(limits.numAlterVisits > 0);
    assert(limits.numAlterPlayouts > 0);
    SearchParams oldParams = toMoveBot->searchParams;

    toMoveBot->searchParams.maxVisits = limits.numAlterVisits;
    toMoveBot->searchParams.maxPlayouts = limits.numAlterPlayouts;
    if(limits.removeRootNoise) {
      //Note - this is slightly sketchy to set the params directly. This works because
      //some of the parameters like FPU are basically stateless and will just affect future playouts
      //and because even stateful effects like rootNoiseEnabled and rootPolicyTemperature only affect
      //the root so when we step down in the tree we get a fresh start.
      toMoveBot->searchParams.rootNoiseEnabled = false;
      toMoveBot->searchParams.rootPolicyTemperature = 1.0;
      toMoveBot->searchParams.rootPolicyTemperatureEarly = 1.0;
      toMoveBot->searchParams.rootFpuLossProp = toMoveBot->searchParams.fpuLossProp;
      toMoveBot->searchParams.rootFpuReductionMax = toMoveBot->searchParams.fpuReductionMax;
      toMoveBot->searchParams.rootDesiredPerChildVisitsCoeff = 0.0;
      toMoveBot->searchParams.rootNumSymmetriesToSample = 1;
    }
    if(limits.playoutDoublingAdvantagePla != C_EMPTY) {
      toMoveBot->searchParams.playoutDoublingAdvantagePla = limits.playoutDoublingAdvantagePla;
      toMoveBot->searchParams.playoutDoublingAdvantage = limits.playoutDoublingAdvantage;
    }

    //If we cleared the search, do a very short search first to get a good dynamic score utility center
    if(limits.clearBotBeforeSearchThisMove && toMoveBot->searchParams.maxVisits > 10 && toMoveBot->searchParams.maxPlayouts > 10) {
      int64_t oldMaxVisits = toMoveBot->searchParams.maxVisits;
      toMoveBot->searchParams.maxVisits = 10;
      toMoveBot->runWholeSearchAndGetMove(pla,logger,NULL);
      toMoveBot->searchParams.maxVisits = oldMaxVisits;
    }
    loc = toMoveBot->runWholeSearchAndGetMove(pla,logger,NULL);

    toMoveBot->searchParams = oldParams;
  }
  else {
    assert(!limits.removeRootNoise);
    loc = toMoveBot->runWholeSearchAndGetMove(pla,logger,NULL);
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
  const FancyModes& fancyModes, const OtherGameProperties& otherGameProps,
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
    fancyModes, otherGameProps,
    gameRand,
    checkForNewNNEval
  );

  if(botW != botB)
    delete botW;
  delete botB;

  return gameData;
}

FinishedGameData* Play::runGame(
  const Board& startBoard, Player startPla, const BoardHistory& startHist, ExtraBlackAndKomi extraBlackAndKomi,
  MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
  Search* botB, Search* botW,
  bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
  Logger& logger, bool logSearchInfo, bool logMoves,
  int maxMovesPerGame, vector<std::atomic<bool>*>& stopConditions,
  const FancyModes& fancyModes, const OtherGameProperties& otherGameProps,
  Rand& gameRand,
  std::function<NNEvaluator*()>* checkForNewNNEval
) {
  FinishedGameData* gameData = new FinishedGameData();

  Board board(startBoard);
  BoardHistory hist(startHist);
  Player pla = startPla;
  assert(!(extraBlackAndKomi.makeGameFair && extraBlackAndKomi.makeGameFairForEmptyBoard));

  if(extraBlackAndKomi.makeGameFairForEmptyBoard) {
    Board b(startBoard.x_size,startBoard.y_size);
    BoardHistory h(b,pla,startHist.rules,startHist.encorePhase);
    h.setKomi(roundAndClipKomi(extraBlackAndKomi.komiBase,board,false));
    adjustKomiToEven(botB,botW,b,h,pla,fancyModes.compensateKomiVisits,logger,otherGameProps,gameRand);
    hist.setKomi(roundAndClipKomi(h.rules.komi + extraBlackAndKomi.komi - extraBlackAndKomi.komiBase, board, false));
  }
  if(extraBlackAndKomi.extraBlack > 0) {
    double extraBlackTemperature = 1.0;
    playExtraBlack(botB,extraBlackAndKomi.extraBlack,board,hist,extraBlackTemperature,gameRand);
    assert(hist.moveHistory.size() == 0);
  }
  if(extraBlackAndKomi.makeGameFair) {
    //First, restore back to baseline komi
    hist.setKomi(roundAndClipKomi(extraBlackAndKomi.komiBase,board,false));
    //Adjust komi to be fair for the handicap according to what the bot thinks.
    adjustKomiToEven(botB,botW,board,hist,pla,fancyModes.compensateKomiVisits,logger,otherGameProps,gameRand);
    //Then, reapply the komi offset from base that we should have had
    hist.setKomi(roundAndClipKomi(hist.rules.komi + extraBlackAndKomi.komi - extraBlackAndKomi.komiBase, board, false));
  }
  else if((extraBlackAndKomi.extraBlack > 0 || otherGameProps.isFork) && fancyModes.fancyKomiVarying && gameRand.nextBool(0.5)) {
    double origKomi = hist.rules.komi;
    //First, restore back to baseline komi
    hist.setKomi(roundAndClipKomi(extraBlackAndKomi.komiBase,board,false));
    //Adjust komi to be fair for the handicap according to what the bot thinks.
    adjustKomiToEven(botB,botW,board,hist,pla,fancyModes.compensateKomiVisits,logger,otherGameProps,gameRand);
    //Then, reapply the komi offset from base that we should have had
    hist.setKomi(roundAndClipKomi(hist.rules.komi + extraBlackAndKomi.komi - extraBlackAndKomi.komiBase, board, false));
    double newKomi = hist.rules.komi;
    //Now, randomize between the old and new komi, with extra noise
    double randKomi = gameRand.nextDouble(min(origKomi,newKomi),max(origKomi,newKomi));
    randKomi += 0.75 * sqrt(board.x_size * board.y_size) * nextGaussianTruncated(gameRand,2.5);
    hist.setKomi(roundAndClipKomi(randKomi, board, false));
  }

  gameData->bName = botSpecB.botName;
  gameData->wName = botSpecW.botName;
  gameData->bIdx = botSpecB.botIdx;
  gameData->wIdx = botSpecW.botIdx;

  gameData->gameHash.hash0 = gameRand.nextUInt64();
  gameData->gameHash.hash1 = gameRand.nextUInt64();

  gameData->drawEquivalentWinsForWhite = botSpecB.baseParams.drawEquivalentWinsForWhite;
  gameData->playoutDoublingAdvantagePla = otherGameProps.playoutDoublingAdvantagePla;
  gameData->playoutDoublingAdvantage = otherGameProps.playoutDoublingAdvantage;

  gameData->numExtraBlack = extraBlackAndKomi.extraBlack;
  gameData->mode = 0;
  gameData->modeMeta1 = 0;
  gameData->modeMeta2 = 0;

  //In selfplay, record all the policy maps and evals and such as well for training data
  bool recordFullData = fancyModes.forSelfPlay;

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

  if(fancyModes.initGamesWithPolicy && otherGameProps.allowPolicyInit) {
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
      adjustKomiToEven(botB, botW, board, hist, pla, fancyModes.compensateKomiVisits, logger, otherGameProps, gameRand);

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
      toMoveBot, pla, fancyModes, gameRand, historicalMctsWinLossValues, clearBotBeforeSearch, otherGameProps
    );
    Loc loc = runBotWithLimits(toMoveBot, pla, fancyModes, limits, logger);

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
      extractValueTargets(whiteValueTargets, toMoveBot, toMoveBot->rootNode);
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
      finalValueTargets.hasLead = true;
      finalValueTargets.lead = finalValueTargets.score;

      //Fill full and seki areas
      {
        board.calculateArea(gameData->finalFullArea, true, true, true, hist.rules.multiStoneSuicideLegal);

        Color* independentLifeArea = new Color[Board::MAX_ARR_SIZE];
        int whiteMinusBlackIndependentLifeRegionCount;
        board.calculateIndependentLifeArea(independentLifeArea,whiteMinusBlackIndependentLifeRegionCount, false, false, hist.rules.multiStoneSuicideLegal);
        for(int i = 0; i<Board::MAX_ARR_SIZE; i++) {
          if(independentLifeArea[i] == C_EMPTY && (gameData->finalFullArea[i] == C_BLACK || gameData->finalFullArea[i] == C_WHITE))
            gameData->finalSekiAreas[i] = true;
          else
            gameData->finalSekiAreas[i] = false;
        }
        delete independentLifeArea;
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
      //We do NOT apply playoutDoublingAdvantage here. If changing this, note that it is coordinated with train data writing
      //not using playoutDoublingAdvantage for these rows too.
      Loc responseLoc = toMoveBot->runWholeSearchAndGetMove(sp->pla,logger,NULL);

      extractPolicyTarget(sp->policyTarget, toMoveBot, toMoveBot->rootNode, locsBuf, playSelectionValuesBuf);
      extractValueTargets(sp->whiteValueTargets, toMoveBot, toMoveBot->rootNode);
      sp->targetWeight = 1.0f;
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

    //Resolve probabilistic weights of things
    {
      auto resolveWeight = [&gameRand](float weight){
        if(weight <= 0) weight = 0;
        float floored = floor(weight);
        float excess = weight - floored;
        weight = gameRand.nextBool(excess) ? floored+1 : floored;
        return weight;
      };

      for(int i = 0; i<gameData->targetWeightByTurn.size(); i++)
        gameData->targetWeightByTurn[i] = resolveWeight(gameData->targetWeightByTurn[i]);
      for(int i = 0; i<gameData->sidePositions.size(); i++)
        gameData->sidePositions[i]->targetWeight = resolveWeight(gameData->sidePositions[i]->targetWeight);
    }

    //Fill in lead estimation on full-search positions
    if(fancyModes.estimateLeadProb > 0.0) {
      assert(gameData->targetWeightByTurn.size() + 1 == gameData->whiteValueTargetsByTurn.size());
      board = gameData->startBoard;
      hist = gameData->startHist;
      pla = gameData->startPla;

      int startTurnNumber = gameData->startHist.moveHistory.size();
      int numMoves = gameData->endHist.moveHistory.size() - gameData->startHist.moveHistory.size();
      for(int turnNumberAfterStart = 0; turnNumberAfterStart<numMoves; turnNumberAfterStart++) {
        int absoluteTurnNumber = turnNumberAfterStart + startTurnNumber;
        if(gameData->targetWeightByTurn[turnNumberAfterStart] > 0 &&
           //Avoid computing lead when no result was considered to be very likely, since in such cases
           //the relationship between komi and the result can somewhat break.
           gameData->whiteValueTargetsByTurn[turnNumberAfterStart].noResult < 0.3 &&
           gameRand.nextBool(fancyModes.estimateLeadProb)
        ) {
          gameData->whiteValueTargetsByTurn[turnNumberAfterStart].lead =
            computeLead(botB,botW,board,hist,pla,fancyModes.estimateLeadVisits,logger,otherGameProps);
          gameData->whiteValueTargetsByTurn[turnNumberAfterStart].hasLead = true;
        }
        Move move = gameData->endHist.moveHistory[absoluteTurnNumber];
        assert(move.pla == pla);
        hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
        pla = getOpp(pla);
      }

      for(int i = 0; i<gameData->sidePositions.size(); i++) {
        SidePosition* sp = gameData->sidePositions[i];
        if(sp->targetWeight > 0 &&
           sp->whiteValueTargets.noResult < 0.3 &&
           gameRand.nextBool(fancyModes.estimateLeadProb)
        ) {
          sp->whiteValueTargets.lead =
            computeLead(botB,botW,sp->board,sp->hist,sp->pla,fancyModes.estimateLeadVisits,logger,otherGameProps);
          sp->whiteValueTargets.hasLead = true;
        }
      }
    }
  }

  return gameData;
}

static void replayGameUpToMove(const FinishedGameData* finishedGameData, int moveIdx, Rules rules, Board& board, BoardHistory& hist, Player& pla) {
  board = finishedGameData->startHist.initialBoard;
  pla = finishedGameData->startHist.initialPla;

  if(rules.scoringRule == Rules::SCORING_AREA)
    hist.clear(board,pla,rules,0);
  else
    hist.clear(board,pla,rules,finishedGameData->startHist.initialEncorePhase);

  //Make sure it's prior to the last move
  moveIdx = std::min(moveIdx,(int)(finishedGameData->endHist.moveHistory.size()-1));

  //Replay all those moves
  for(int i = 0; i<moveIdx; i++) {
    Loc loc = finishedGameData->endHist.moveHistory[i].loc;
    if(!hist.isLegal(board,loc,pla)) {
      //We have a bug of some sort if we got an illegal move on replay, unless
      //we are in encore phase (pass for ko may change) or the rules are different
      if(rules == finishedGameData->startHist.rules && hist.encorePhase == 0) {
        cout << board << endl;
        cout << PlayerIO::colorToChar(pla) << endl;
        cout << Location::toString(loc,board) << endl;
        hist.printDebugInfo(cout,board);
        cout << endl;
        throw StringError("Illegal move when replaying to fork game?");
      }
      //Just break out due to the illegal move and stop the replay here
      return;
    }
    assert(finishedGameData->endHist.moveHistory[i].pla == pla);
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
    pla = getOpp(pla);

    if(hist.isGameFinished)
      return;
  }
}

static bool hasUnownedSpot(const FinishedGameData* finishedGameData) {
  assert(finishedGameData->finalOwnership != NULL);
  const Board& board = finishedGameData->startBoard;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(finishedGameData->finalOwnership[loc] == C_EMPTY)
        return true;
    }
  }
  return false;
}

void Play::maybeForkGame(
  const FinishedGameData* finishedGameData,
  ForkData* forkData,
  const FancyModes& fancyModes,
  Rand& gameRand,
  Search* bot
) {
  if(forkData == NULL)
    return;
  assert(finishedGameData->startHist.initialBoard.pos_hash == finishedGameData->endHist.initialBoard.pos_hash);
  assert(finishedGameData->startHist.initialPla == finishedGameData->endHist.initialPla);

  //Just for conceptual simplicity, don't early fork games that started in the encore
  if(finishedGameData->startHist.encorePhase != 0)
    return;
  bool earlyFork = gameRand.nextBool(fancyModes.earlyForkGameProb);
  bool lateFork = !earlyFork && fancyModes.forkGameProb > 0 ? gameRand.nextBool(fancyModes.forkGameProb) : false;
  if(!earlyFork && !lateFork)
    return;

  //Pick a random move to fork from near the start
  int moveIdx;
  if(earlyFork) {
    moveIdx = (int)floor(
      gameRand.nextExponential() * (
        fancyModes.earlyForkGameExpectedMoveProp * finishedGameData->startBoard.x_size * finishedGameData->startBoard.y_size
      )
    );
  }
  else if(lateFork) {
    moveIdx = finishedGameData->endHist.moveHistory.size() <= 0 ? 0 : (int)gameRand.nextUInt(finishedGameData->endHist.moveHistory.size());
  }
  else {
    ASSERT_UNREACHABLE;
  }

  Board board;
  Player pla;
  BoardHistory hist;
  replayGameUpToMove(finishedGameData, moveIdx, finishedGameData->startHist.rules, board, hist, pla);
  //Just in case if somehow the game is over now, don't actually do anything
  if(hist.isGameFinished)
    return;

  //Pick a move!
  if(fancyModes.forkGameMaxChoices > NNPos::MAX_NN_POLICY_SIZE)
    throw StringError("fancyModes.forkGameMaxChoices > NNPos::MAX_NN_POLICY_SIZE");
  if(fancyModes.earlyForkGameMaxChoices > NNPos::MAX_NN_POLICY_SIZE)
    throw StringError("fancyModes.earlyForkGameMaxChoices > NNPos::MAX_NN_POLICY_SIZE");
  int maxChoices = earlyFork ? fancyModes.earlyForkGameMaxChoices : fancyModes.forkGameMaxChoices;
  if(maxChoices < fancyModes.forkGameMinChoices)
    throw StringError("fancyModes fork game max choices < fancyModes.forkGameMinChoices");

  //Generate a selection of a small random number of choices
  int numChoices = gameRand.nextInt(fancyModes.forkGameMinChoices, maxChoices);
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
  forkData->add(new InitialPosition(board,hist,pla));
}


void Play::maybeSekiForkGame(
  const FinishedGameData* finishedGameData,
  ForkData* forkData,
  const FancyModes& fancyModes,
  const GameInitializer* gameInit,
  Rand& gameRand
) {
  if(forkData == NULL)
    return;
  if(!fancyModes.sekiForkHack)
    return;

  //If there are any unowned spots, consider forking the last bit of the game, with random rules and even score
  //Don't fork games starting in second encore though
  const BoardHistory& endHist = finishedGameData->endHist;
  if(endHist.isGameFinished && endHist.isScored && finishedGameData->startHist.encorePhase < 2 && hasUnownedSpot(finishedGameData)) {

    for(int i = 0; i<2; i++) {
      //Pick a random move to fork from near the end of the game
      int moveIdx = (int)floor(endHist.moveHistory.size() * (1.0 - 0.10 * gameRand.nextExponential()) - 1.0);
      if(moveIdx < 0)
        moveIdx = 0;
      if(moveIdx > endHist.moveHistory.size())
        moveIdx = endHist.moveHistory.size();

      //Randomly permute the rules
      Rules rules = finishedGameData->startHist.rules;
      rules = gameInit->randomizeScoringAndTaxRules(rules,gameRand);

      Board board;
      Player pla;
      BoardHistory hist;
      replayGameUpToMove(finishedGameData, moveIdx, rules, board, hist, pla);
      //Just in case if somehow the game is over now, don't actually do anything
      if(hist.isGameFinished)
        continue;
      forkData->addSeki(new InitialPosition(board,hist,pla),gameRand);
    }
  }
}

GameRunner::GameRunner(ConfigParser& cfg, const string& sRandSeedBase, FancyModes fModes, Logger& logger)
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
  gameInit = new GameInitializer(cfg,logger);
}

GameRunner::~GameRunner() {
  delete gameInit;
}

FinishedGameData* GameRunner::runGame(
  int64_t gameIdx,
  const MatchPairer::BotSpec& bSpecB,
  const MatchPairer::BotSpec& bSpecW,
  ForkData* forkData,
  Logger& logger,
  vector<std::atomic<bool>*>& stopConditions,
  std::function<NNEvaluator*()>* checkForNewNNEval
) {
  MatchPairer::BotSpec botSpecB = bSpecB;
  MatchPairer::BotSpec botSpecW = bSpecW;

  string searchRandSeed = searchRandSeedBase + ":" + Global::int64ToString(gameIdx);
  Rand gameRand(searchRandSeed + ":" + "forGameRand");

  const InitialPosition* initialPosition = NULL;
  bool usedSekiForkHackPosition = false;
  if(forkData != NULL) {
    initialPosition = forkData->get(gameRand);

    if(initialPosition == NULL && fancyModes.sekiForkHack && gameRand.nextBool(0.04)) {
      initialPosition = forkData->getSeki(gameRand);
      if(initialPosition != NULL)
        usedSekiForkHackPosition = true;
    }
  }

  Board board;
  Player pla;
  BoardHistory hist;
  ExtraBlackAndKomi extraBlackAndKomi;
  OtherGameProperties otherGameProps;
  if(fancyModes.forSelfPlay) {
    assert(botSpecB.botIdx == botSpecW.botIdx);
    SearchParams params = botSpecB.baseParams;
    gameInit->createGame(board,pla,hist,extraBlackAndKomi,params,initialPosition,fancyModes,otherGameProps);
    botSpecB.baseParams = params;
    botSpecW.baseParams = params;
  }
  else {
    gameInit->createGame(board,pla,hist,extraBlackAndKomi,initialPosition,fancyModes,otherGameProps);

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

  //In 2% of games, don't autoterminate the game upon all pass alive, to just provide a tiny bit of training data on positions that occur
  //as both players must wrap things up manually, because within the search we don't autoterminate games, meaning that the NN will get
  //called on positions that occur after the game would have been autoterminated.
  bool doEndGameIfAllPassAlive = fancyModes.forSelfPlay ? gameRand.nextBool(0.98) : true;

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
    fancyModes,otherGameProps,
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

  Play::maybeForkGame(finishedGameData, forkData, fancyModes, gameRand, botB);
  if(!usedSekiForkHackPosition) {
    Play::maybeSekiForkGame(finishedGameData, forkData, fancyModes, gameInit, gameRand);
  }

  if(botW != botB)
    delete botW;
  delete botB;

  if(initialPosition != NULL)
    delete initialPosition;

  return finishedGameData;
}
