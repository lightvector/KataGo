#include "core/global.h"
#include "core/makedir.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"

using namespace std;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

#include <csignal>
static std::atomic<bool> sigTermReceived(false);
void signalHandler(int signal);
void signalHandler(int signal)
{
  if(signal == SIGTERM)
    sigTermReceived.store(true);
}

int main(int argc, const char* argv[]) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  vector<string> nnModelFiles;
  vector<string> nnModelNames;
  string sgfOutputDir;
  try {
    TCLAP::CmdLine cmd("Sgf->HDF5 data writer", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/match_example.cfg)",true,string(),"FILE");
    TCLAP::MultiArg<string> nnModelFileArg("","nn-model-file","Neural net model .pb graph file to use",true,"FILE");
    TCLAP::MultiArg<string> nnModelNameArg("","nn-model-name","Neural net model name for logs and data",true,"STR");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",false,string(),"DIR");
    cmd.add(configFileArg);
    cmd.add(nnModelFileArg);
    cmd.add(nnModelNameArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    nnModelFiles = nnModelFileArg.getValue();
    nnModelNames = nnModelNameArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  if(nnModelFiles.size() <= 0) {
    cerr << "Must specify at least one -nn-model-file" << endl;
    return 1;
  }
  if(nnModelFiles.size() >= 100) {
    cerr << "Too many model files" << endl;
    return 1;
  }

  ConfigParser cfg(configFile);

  Logger logger;
  logger.addFile(cfg.getString("logFile"));
  logger.setLogToStdout(true);
  bool logSearchInfo = cfg.getBool("logSearchInfo");

  logger.write("Match Engine starting...");

  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(nnModelFiles,cfg,logger,seedRand);
  logger.write("Loaded neural net");


  vector<string> allowedKoRuleStrs = cfg.getStrings("koRules", Rules::koRuleStrings());
  vector<string> allowedScoringRuleStrs = cfg.getStrings("scoringRules", Rules::scoringRuleStrings());
  vector<bool> allowedMultiStoneSuicideLegals = cfg.getBools("multiStoneSuicideLegals");

  vector<int> allowedKoRules;
  vector<int> allowedScoringRules;
  for(size_t i = 0; i < allowedKoRuleStrs.size(); i++)
    allowedKoRules.push_back(Rules::parseKoRule(allowedKoRuleStrs[i]));
  for(size_t i = 0; i < allowedScoringRuleStrs.size(); i++)
    allowedScoringRules.push_back(Rules::parseScoringRule(allowedScoringRuleStrs[i]));

  if(allowedKoRules.size() <= 0)
    throw IOError("koRules must have at least one value in " + configFile);
  if(allowedScoringRules.size() <= 0)
    throw IOError("scoringRules must have at least one value in " + configFile);
  if(allowedMultiStoneSuicideLegals.size() <= 0)
    throw IOError("multiStoneSuicideLegals must have at least one value in " + configFile);


  //TODO we need a mechanism to make the komi range shrink for smaller boards!
  vector<int> allowedBSizes = cfg.getInts("bSizes", 9, 19);
  vector<double> allowedBSizeRelProbs = cfg.getDoubles("bSizeRelProbs",0.0,1.0);

  float baseKomi = cfg.getFloat("komiMean",-60.0f,60.0f);
  float komiStdev = cfg.getFloat("komiStdev",-60.0f,60.0f);
  double komiAllowIntegerProb = cfg.getDouble("komiAllowIntegerProb",0.0,1.0);
  double handicapProb = cfg.getDouble("handicapProb",0.0,1.0);
  float handicapStoneValue = cfg.getFloat("handicapStoneValue",0.0f,30.0f);
  double komiBigStdevProb = cfg.getDouble("komiBigStdevProb",0.0,1.0);
  double komiBigStdev = cfg.getFloat("komiBigStdev",-60.0f,60.0f);
  
  if(allowedBSizes.size() <= 0)
    throw IOError("bSizes must have at least one value in " + configFile);
  if(allowedKomis.size() <= 0)
    throw IOError("komis must have at least one value in " + configFile);
  if(allowedBSizes.size() != allowedBSizeRelProbs.size())
    throw IOError("bSizes and bSizeRelProbs must have same number of values in " + configFile);

  vector<SearchParams> paramss = Setup::loadParams(cfg);
  assert(paramss.size() > 0);

  string searchRandSeedBase = Global::uint64ToHexString(seedRand.nextUInt64);

  vector<string> botNames;
  if(cfg.contains("botNames"))
    botNames = cfg.getStrings("botNames");
  if(paramss.size() > 1 && botNames.size() != paramss.size())
    throw IOError("botNames must be specified and have a name for each bot if numBots > 1");
  if(paramss.size() == 1 && botNames.size() != paramss.size())
    botNames.push_back("bot");
  if(botNames.size() > 1000)
    throw IOError("botNames has too many values");
  size_t numBots = botNames.size();

  vector<int> whichNNModel;
  if(cfg.contains("whichNNModel")) {
    whichNNModel = cfg.getInts("whichNNModel", 0, 1000);
    if(whichNNModel.size() != numBots)
      throw IOError("whichNNModel must have exactly one entry for each bot");
  }
  else {
    if(nnEvals.size() > 1)
      throw IOError("whichNNModel must be specified in config if cmdline gives more than one model");
    for(size_t i = 0; i<numBots; i++)
      whichNNModel.push_back(0);
  }

  {
    int numModels = nnEvals.size();
    bool nnModelUsed[numModels];
    for(int i = 0; i<numModels; i++)
      nnModelUsed[i] = false;
    for(size_t i = 0; i<numBots; i++) {
      if(whichNNModel[i] >= numModels)
        throw IOError("whichNNModel specified using model " + Global::intToString(whichNNModel[i])
                      + " but only " + Global::intToString(numModels) + " were provided");
      nnModelUsed[whichNNModel[i]] = true;
    }
    for(int i = 0; i<numModels; i++) {
      if(!nnModelUsed[i])
        throw IOError("whichNNModel does not use model #" + Global::intToString(i)
                      + " from the command line: " + nnModelFiles[i]);
    }
  }

  int numMatchThreads = cfg.getInt("numMatchThreads",1,16384);
  int64_t numMatchGamesTotal = cfg.getInt64("numMatchGamesTotal",1,((int64_t)1) << 62);
  int maxMovesPerGame = cfg.getInt("maxMovesPerGame",1,1 << 30);

  {
    vector<string> unusedKeys = cfg.unusedKeys();
    for(size_t i = 0; i<unusedKeys.size(); i++) {
      string msg = "WARNING: Unused key '" + unusedKeys[i] + "' in " + configFile;
      logger.write(msg);
      cerr << msg << endl;
    }
  }

  if(sgfOutputDir != string())
    MakeDir::make(sgfOutputDir);

  //TODO terminate a game if ALL territory for both players is strictly pass-alive!

  mutex newGameMutex;
  auto initNewGame = [&](Board& board, Player& pla, BoardHistory& hist, int& numExtraBlack) {
    //Multiple threads will be calling this, and seedRand is shared, so we mutex to protect things
    unique_lock<std::mutex> lock(newGameMutex);

    int bSize = allowedBSizes[seedRand.nextUInt(allowedBSizeRelProbs.data(),allowedBSizeRelProbs.size())];
    board = Board(bSize,bSize);

    Rules rules;
    rules.koRule = allowedKoRules[seedRand.nextUInt(allowedKoRules.size())];
    rules.scoringRule = allowedScoringRules[seedRand.nextUInt(allowedScoringRules.size())];
    rules.multiStoneSuicideLegal = allowedMultiStoneSuicideLegals[seedRand.nextUInt(allowedMultiStoneSuicideLegals.size())];
  
    pair<int,float> extraBlackAndKomi = Setup::chooseExtraBlackAndKomi(
      komiBase, komiStdev, komiAllowIntegerProb, handicapProb, handicapStoneValue,
      komiBigStdevProb, komiBigStdev, bSize, seedRand
    );
    rules.komi = extraBlackAndKomi.second();
  
    pla = P_BLACK;
    hist.clear(board,pla,rules);
    numExtraBlack = extraBlackAndKomi.first;

    return true;
  };

  if(!std::atomic_is_lock_free(&sigTermReceived))
    throw StringError("sigTermReceived is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGTERM, signalHandler);

  auto failIllegalMove = [&logger](AsyncBot* bot, Board board, Loc loc) {
    ostringstream sout;
    sout << "Bot returned null location or illegal move!?!" << "\n";
    sout << board << "\n";
    sout << bot->getRootBoard() << "\n";
    sout << "Pla: " << playerToString(bot->getRootPla()) << "\n";
    sout << "Loc: " << Location::toString(loc,bot->getRootBoard()) << "\n";
    logger.write(sout.str());
    bot->getRootBoard().checkConsistency();
    assert(false);
  };

  auto logSearch = [&logger](AsyncBot* bot, Loc loc) {
    Search* search = bot->getSearch();
    ostringstream sout;
    Board::printBoard(sout, bot->getRootBoard(), loc);
    sout << "\n";
    sout << "Root visits: " << search->numRootVisits() << "\n";
    sout << "PV: ";
    search->printPV(sout, search->rootNode, 25);
    sout << "\n";
    sout << "Tree:\n";
    search->printTree(sout, search->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10));
    logger.write(sout.str());
  };

  auto playExtraBlack = [&failIllegalMove](AsyncBot* bot, int numExtraBlack, Board& board, BoardHistory& hist) {
    SearchParams oldParams = bot->getSearcher()->searchParams;
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
      Loc loc = bot->genMoveSynchronous(pla);      
      if(loc == Board::NULL_LOC || !bot->isLegal(loc,pla))
        failIllegalMove(bot,board,loc);
      assert(hist.isLegal(board,loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      hist.clear(board,pla,hist.rules);
      bot->setPosition(pla,board,hist);
    }

    bot->setParams(oldParams);
    bot->setRootPassLegal(true);
  };
  
  auto runSelfPlayGame = [&failIllegalMove,&playExtraBlack,&logSearch,&paramss,&nnEvals,&whichNNModel,&logger,logSearchInfo,maxMovesPerGame,&searchRandSeedBase] (
    int gameIdx, int botIdx, Board& board, Player pla, BoardHistory& hist, int numExtraBlack
  ) {
    string searchRandSeed = searchRandSeedBase + ":" + Global::int64ToString(gameIdx);
    AsyncBot* bot = new AsyncBot(paramss[botIdx], nnEvals[whichNNModel[botIdx]], &logger, searchRandSeed);
    if(numExtraBlack > 0)
      playExtraBlack(bot,numExtraBlack,board,hist);
    bot->setPosition(pla,board,hist);

    for(int i = 0; i<maxMovesPerGame; i++) {
      if(hist.isGameFinished)
        break;
      if(sigTermReceived.load())
        break;

      Loc loc = bot->genMoveSynchronous(pla);
      bot->clearSearch();

      if(loc == Board::NULL_LOC || !bot->isLegal(loc,pla))
        failIllegalMove(bot,board,loc);
      if(logSearchInfo)
        logSearch(bot,loc);

      bool suc = bot->makeMove(loc,pla);
      assert(suc);

      assert(hist.isLegal(board,loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      pla = getOpp(pla);
    }
    delete bot;
  };

  auto runMatchGame = [&failIllegalMove,&playExtraBlack,&logSearch,&paramss,&nnEvals,&whichNNModel,&logger,logSearchInfo,maxMovesPerGame,&searchRandSeedBase](
    int64_t gameIdx, int botIdxB, int botIdxW, Board& board, Player pla, BoardHistory& hist, int numExtraBlack
  ) {
    string searchRandSeed = searchRandSeedBase + ":" + Global::int64ToString(gameIdx);
    AsyncBot* botB = new AsyncBot(paramss[botIdxB], nnEvals[whichNNModel[botIdxB]], &logger, searchRandSeed+"B");
    AsyncBot* botW = new AsyncBot(paramss[botIdxW], nnEvals[whichNNModel[botIdxW]], &logger, searchRandSeed+"W");
    if(numExtraBlack > 0)
      playExtraBlack(botB,numExtraBlack,board,hist);
    botB->setPosition(pla,board,hist);
    botW->setPosition(pla,board,hist);

    for(int i = 0; i<maxMovesPerGame; i++) {
      if(hist.isGameFinished)
        break;
      if(sigTermReceived.load())
        break;

      AsyncBot* toMoveBot = pla == P_BLACK ? botB : botW;
      Loc loc = toMoveBot->genMoveSynchronous(pla);

      if(loc == Board::NULL_LOC || !toMoveBot->isLegal(loc,pla))
        failIllegalMove(toMoveBot,board,loc);
      if(logSearchInfo)
        logSearch(toMoveBot,loc);

      bool suc;
      suc = botB->makeMove(loc,pla);
      assert(suc);
      suc = botW->makeMove(loc,pla);
      assert(suc);

      assert(hist.isLegal(board,loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      pla = getOpp(pla);
    }
    delete botB;
    delete botW;
  };

  mutex matchSetupMutex;
  int64_t numMatchGamesStartedSoFar = 0;
  vector<pair<int,int>> nextMatchups;
  Rand matchRand;

  //Only call this if matchSetupMutex is already locked
  auto getMatchup = [numBots,&nextMatchups,&matchRand]() {
    if(nextMatchups.size() <= 0) {
      if(numBots == 1)
        return make_pair(0,0);
      for(int i = 0; i<numBots; i++) {
        for(int j = 0; j<numBots; j++) {
          if(i != j) {
            nextMatchups.push_back(make_pair(i,j));
          }
        }
      }
      //Shuffle
      for(int i = nextMatchups.size()-1; i >= 1; i--) {
        int j = (int)matchRand.nextUInt(i+1);
        pair<int,int> tmp = nextMatchups[i];
        nextMatchups[i] = nextMatchups[j];
        nextMatchups[j] = tmp;
      }
    }
    pair<int,int> matchup = nextMatchups.back();
    nextMatchups.pop_back();
    return matchup;
  };

  auto runMatchLoop = [
    &botNames,&initNewGame,&runSelfPlayGame,&runMatchGame,&matchSetupMutex,
    numMatchGamesTotal,&numMatchGamesStartedSoFar,&getMatchup,&sgfOutputDir,&logger
  ](
    uint64_t threadHash
  ) {
    unique_lock<std::mutex> lock(matchSetupMutex,std::defer_lock);
    ofstream* sgfOut = sgfOutputDir.length() > 0 ? (new ofstream(sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs")) : NULL;

    while(true) {
      lock.lock();
      if(numMatchGamesStartedSoFar >= numMatchGamesTotal)
        break;
      if(sigTermReceived.load())
        break;
      int64_t gameIdx = numMatchGamesStartedSoFar;
      numMatchGamesStartedSoFar += 1;

      if(numMatchGamesStartedSoFar % 500 == 0)
        logger.write("Started " + Global::int64ToString(numMatchGamesStartedSoFar) + "games");

      pair<int,int> matchup = getMatchup();
      int botIdxB = matchup.first;
      int botIdxW = matchup.second;

      lock.unlock();

      Board board; Player pla; BoardHistory hist; int numExtraBlack;
      initNewGame(board,pla,hist,numExtraBlack);
      Board initialBoard = board;
      Rules initialRules = hist.rules;

      if(botIdxB == botIdxW)
        runSelfPlayGame(gameIdx,botIdxB,board,pla,hist,numExtraBlack);
      else
        runMatchGame(gameIdx,botIdxB,botIdxW,board,pla,hist,numExtraBlack);

      if(sigTermReceived.load())
        break;

      string bName = botNames[botIdxB];
      string wName = botNames[botIdxW];
      if(sgfOut != NULL)
        WriteSgf::writeSgf(*sgfOut,bName,wName,initialRules,initialBoard,hist);
    }
  };

  Rand hashRand;
  vector<std::thread> threads;
  for(int i = 0; i<numMatchThreads; i++) {
    threads.push_back(std::thread(runMatchLoop, hashRand.nextUInt64()));
  }
  for(int i = 0; i<numMatchThreads; i++)
    threads[i].join();

  logger.write("All cleaned up, quitting");
  return 0;
}

