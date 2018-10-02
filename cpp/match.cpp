#include "core/global.h"
#include "core/makedir.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "program/gitinfo.h"
#include "main.h"

using namespace std;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

#include <csignal>
static std::atomic<bool> sigReceived(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM)
    sigReceived.store(true);
}

int MainCmds::match(int argc, const char* const* argv) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  string logFile;
  string sgfOutputDir;
  try {
    TCLAP::CmdLine cmd("Sgf->HDF5 data writer", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/match_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",true,string(),"FILE");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",false,string(),"DIR");
    cmd.add(configFileArg);
    cmd.add(logFileArg);
    cmd.add(sgfOutputDirArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    logFile = logFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  ConfigParser cfg(configFile);

  Logger logger;
  logger.addFile(logFile);
  bool logToStdout = cfg.getBool("logToStdout");
  logger.setLogToStdout(logToStdout);
  bool logSearchInfo = cfg.getBool("logSearchInfo");
  bool logMoves = cfg.getBool("logMoves");
  int64_t logGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);

  logger.write("Match Engine starting...");
  logger.write(string("Git revision: ") + GIT_REVISION);

  //Load per-bot search config, first, which also tells us how many bots we're running
  vector<SearchParams> paramss = Setup::loadParams(cfg);
  assert(paramss.size() > 0);
  int numBots = paramss.size();

  //Load the names of the bots and which model each bot is using
  vector<string> nnModelFilesByBot;
  vector<string> botNames;
  for(int i = 0; i<numBots; i++) {
    string idxStr = Global::intToString(i);

    if(cfg.contains("botName"+idxStr))
      botNames.push_back(cfg.getString("botName"+idxStr));
    else if(numBots == 1)
      botNames.push_back(cfg.getString("botName"));
    else
      throw StringError("If more than one bot, must specify botName0, botName1,... individually");

    if(cfg.contains("nnModelFile"+idxStr))
      nnModelFilesByBot.push_back(cfg.getString("nnModelFile"+idxStr));
    else
      nnModelFilesByBot.push_back(cfg.getString("nnModelFile"));
  }

  //Load bots that should not play one another
  vector<int> secondaryBots;
  if(cfg.contains("secondaryBots"))
    secondaryBots = cfg.getInts("secondaryBots",0,4096);

  //Dedup and load each necessary model exactly once
  vector<string> nnModelFiles;
  vector<int> whichNNModel;
  for(int i = 0; i<numBots; i++) {
    const string& desiredFile = nnModelFilesByBot[i];
    int alreadyFoundIdx = -1;
    for(int j = 0; j<nnModelFiles.size(); j++) {
      if(nnModelFiles[j] == desiredFile) {
        alreadyFoundIdx = j;
        break;
      }
    }
    if(alreadyFoundIdx != -1)
      whichNNModel.push_back(alreadyFoundIdx);
    else {
      whichNNModel.push_back(nnModelFiles.size());
      nnModelFiles.push_back(desiredFile);
    }
  }

  //Initialize neural net inference engine globals, and load models
  Setup::initializeSession(cfg);
  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(nnModelFiles,cfg,logger,seedRand);
  logger.write("Loaded neural net");

  //Get the configuration for rules
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

  vector<int> allowedBSizes = cfg.getInts("bSizes", 9, 19);
  vector<double> allowedBSizeRelProbs = cfg.getDoubles("bSizeRelProbs",0.0,1.0);

  float komiMean = cfg.getFloat("komiMean",-60.0f,60.0f);
  float komiStdev = cfg.getFloat("komiStdev",-60.0f,60.0f);
  double komiAllowIntegerProb = cfg.getDouble("komiAllowIntegerProb",0.0,1.0);
  double handicapProb = cfg.getDouble("handicapProb",0.0,1.0);
  float handicapStoneValue = cfg.getFloat("handicapStoneValue",0.0f,30.0f);
  double komiBigStdevProb = cfg.getDouble("komiBigStdevProb",0.0,1.0);
  double komiBigStdev = cfg.getFloat("komiBigStdev",-60.0f,60.0f);

  if(allowedBSizes.size() <= 0)
    throw IOError("bSizes must have at least one value in " + configFile);
  if(allowedBSizes.size() != allowedBSizeRelProbs.size())
    throw IOError("bSizes and bSizeRelProbs must have same number of values in " + configFile);

  //Load match runner settings
  int numMatchThreads = cfg.getInt("numMatchThreads",1,16384);
  int64_t numMatchGamesTotal = cfg.getInt64("numMatchGamesTotal",1,((int64_t)1) << 62);
  int maxMovesPerGame = cfg.getInt("maxMovesPerGame",1,1 << 30);

  string searchRandSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Check for unused config keys
  {
    vector<string> unusedKeys = cfg.unusedKeys();
    for(size_t i = 0; i<unusedKeys.size(); i++) {
      string msg = "WARNING: Unused key '" + unusedKeys[i] + "' in " + configFile;
      logger.write(msg);
      cerr << msg << endl;
    }
  }

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting matches");
  if(!logToStdout)
    cout << "Loaded all config stuff, starting matches" << endl;

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
      komiMean, komiStdev, komiAllowIntegerProb, handicapProb, handicapStoneValue,
      komiBigStdevProb, komiBigStdev, bSize, seedRand
    );
    rules.komi = extraBlackAndKomi.second;

    pla = P_BLACK;
    hist.clear(board,pla,rules);
    numExtraBlack = extraBlackAndKomi.first;

    return true;
  };

  if(!std::atomic_is_lock_free(&sigReceived))
    throw StringError("sigReceived is not lock free, signal-quitting mechanism for intinating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
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
    Board::printBoard(sout, bot->getRootBoard(), loc, &(bot->getRootHist().moveHistory));
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
    SearchParams oldParams = bot->getSearch()->searchParams;
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

  auto runSelfPlayGame = [&failIllegalMove,&playExtraBlack,&logSearch,&paramss,&nnEvals,&whichNNModel,&logger,logSearchInfo,logMoves,maxMovesPerGame,&searchRandSeedBase] (
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
      if(sigReceived.load())
        break;

      Loc loc = bot->genMoveSynchronous(pla);
      bot->clearSearch();

      if(loc == Board::NULL_LOC || !bot->isLegal(loc,pla))
        failIllegalMove(bot,board,loc);
      if(logSearchInfo)
        logSearch(bot,loc);
      if(logMoves)
        logger.write("Move " + Global::intToString(hist.moveHistory.size()) + " made: " + Location::toString(loc,board));

      bool suc = bot->makeMove(loc,pla);
      assert(suc);

      assert(hist.isLegal(board,loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      pla = getOpp(pla);
    }
    delete bot;
  };

  auto runMatchGame = [&failIllegalMove,&playExtraBlack,&logSearch,&paramss,&nnEvals,&whichNNModel,&logger,logSearchInfo,logMoves,maxMovesPerGame,&searchRandSeedBase](
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
      if(sigReceived.load())
        break;

      AsyncBot* toMoveBot = pla == P_BLACK ? botB : botW;
      Loc loc = toMoveBot->genMoveSynchronous(pla);

      if(loc == Board::NULL_LOC || !toMoveBot->isLegal(loc,pla))
        failIllegalMove(toMoveBot,board,loc);
      if(logSearchInfo)
        logSearch(toMoveBot,loc);
      if(logMoves)
        logger.write("Move " + Global::intToString(hist.moveHistory.size()) + " made: " + Location::toString(loc,board));

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
  auto getMatchup = [numBots,&nextMatchups,&secondaryBots,&matchRand]() {
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
    numMatchGamesTotal,&numMatchGamesStartedSoFar,&getMatchup,&sgfOutputDir,&logger,logGamesEvery,
    &nnModelFiles,&nnEvals
  ](
    uint64_t threadHash
  ) {
    unique_lock<std::mutex> lock(matchSetupMutex,std::defer_lock);
    ofstream* sgfOut = sgfOutputDir.length() > 0 ? (new ofstream(sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs")) : NULL;

    while(true) {
      lock.lock();
      if(numMatchGamesStartedSoFar >= numMatchGamesTotal)
        break;
      if(sigReceived.load())
        break;
      int64_t gameIdx = numMatchGamesStartedSoFar;
      numMatchGamesStartedSoFar += 1;

      if(numMatchGamesStartedSoFar % logGamesEvery == 0)
        logger.write("Started " + Global::int64ToString(numMatchGamesStartedSoFar) + " games");
      int logNNEvery = logGamesEvery > 100 ? logGamesEvery : 100;
      if(numMatchGamesStartedSoFar % logNNEvery == 0) {
        for(int i = 0; i<nnModelFiles.size(); i++) {
          logger.write(nnModelFiles[i]);
          logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
          logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
          logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
        }
      }

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

      if(sigReceived.load())
        break;

      string bName = botNames[botIdxB];
      string wName = botNames[botIdxW];
      if(sgfOut != NULL) {
        WriteSgf::writeSgf(*sgfOut,bName,wName,initialRules,initialBoard,hist);
        (*sgfOut) << endl;
      }
    }
    if(sgfOut != NULL)
      sgfOut->close();
  };

  Rand hashRand;
  vector<std::thread> threads;
  for(int i = 0; i<numMatchThreads; i++) {
    threads.push_back(std::thread(runMatchLoop, hashRand.nextUInt64()));
  }
  for(int i = 0; i<numMatchThreads; i++)
    threads[i].join();

  for(int i = 0; i<nnEvals.size(); i++) {
    delete nnEvals[i];
  }
  NeuralNet::globalCleanup();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}

