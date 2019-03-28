#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "core/test.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "main.h"

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

static void writeLine(
  Search* search, const BoardHistory& baseHist,
  const vector<double>& winLossHistory, const vector<double>& scoreHistory, const vector<double>& scoreStdevHistory
) {
  const Board board = search->getRootBoard();
  int posLen = search->posLen;

  cout << board.x_size << " ";
  cout << board.y_size << " ";
  cout << posLen << " "; //in the future we may have posLenX
  cout << posLen << " "; //in the future we may have posLenY
  cout << baseHist.rules.komi << " ";
  if(baseHist.isGameFinished) {
    cout << playerToString(baseHist.winner) << " ";
    cout << baseHist.isResignation << " ";
    cout << baseHist.finalWhiteMinusBlackScore << " ";
  }
  else {
    cout << "-" << " ";
    cout << "false" << " ";
    cout << "0" << " ";
  }

  //Last move
  Loc moveLoc = Board::NULL_LOC;
  if(baseHist.moveHistory.size() > 0)
    moveLoc = baseHist.moveHistory[baseHist.moveHistory.size()-1].loc;
  cout << NNPos::locToPos(moveLoc,board.x_size,posLen) << " ";
  
  cout << baseHist.moveHistory.size() << " ";
  cout << board.numBlackCaptures << " ";
  cout << board.numWhiteCaptures << " ";
  
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(board.colors[loc] == C_BLACK)
        cout << "x";
      else if(board.colors[loc] == C_WHITE)
        cout << "o";
      else
        cout << ".";
    }
  }
  cout << " ";

  vector<AnalysisData> buf;
  if(!baseHist.isGameFinished) {
    int minMovesToTryToGet = 0; //just get the default number
    search->getAnalysisData(buf,minMovesToTryToGet);
  }
  cout << buf.size() << " ";
  for(int i = 0; i<buf.size(); i++) {
    const AnalysisData& data = buf[i];
    cout << NNPos::locToPos(data.move,board.x_size,posLen) << " ";
    cout << data.numVisits << " ";
    cout << data.winLossValue << " ";
    cout << data.scoreMean << " ";
    cout << data.scoreStdev << " ";
    cout << data.policyPrior << " ";
  }
  
  int minVisits = 3;
  vector<double> ownership = search->getAverageTreeOwnership(minVisits);
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      int pos = NNPos::xyToPos(x,y,posLen);
      cout << ownership[pos] << " ";
    }
  }

  cout << winLossHistory.size() << " ";
  for(int i = 0; i<winLossHistory.size(); i++)
    cout << winLossHistory[i] << " ";
  cout << scoreHistory.size() << " ";
  assert(scoreStdevHistory.size() == scoreHistory.size());
  for(int i = 0; i<scoreHistory.size(); i++)
    cout << scoreHistory[i] << " " << scoreStdevHistory[i] << " ";
  
  cout << endl;
}


int MainCmds::demoplay(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  string configFile;
  string logFile;
  string modelFile;
  try {
    TCLAP::CmdLine cmd("Self-play demo dumping status to stdout", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config","Config file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> modelFileArg("","model","Neural net model file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",true,string(),"FILE");
    cmd.add(configFileArg);
    cmd.add(modelFileArg);
    cmd.add(logFileArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    modelFile = modelFileArg.getValue();
    logFile = logFileArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  ConfigParser cfg(configFile);

  Logger logger;
  logger.addFile(logFile);

  logger.write("Engine starting...");

  string searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

  SearchParams params;
  {
    vector<SearchParams> paramss = Setup::loadParams(cfg);
    assert(paramss.size() > 0);
    if(paramss.size() != 1)
      throw StringError("Config specifies more than one bot but demoplay supports only one");
    params = paramss[0];
  }
  
  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    bool alwaysIncludeOwnerMap = true;
    vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators({modelFile},{modelFile},cfg,logger,seedRand,maxConcurrentEvals,false,alwaysIncludeOwnerMap);
    assert(nnEvals.size() == 1);
    nnEval = nnEvals[0];
  }
  logger.write("Loaded neural net");

  const bool allowResignation = cfg.contains("allowResignation") ? cfg.getBool("allowResignation") : false;
  const double resignThreshold = cfg.contains("allowResignation") ? cfg.getDouble("resignThreshold",-1.0,0.0) : -1.0; //Threshold on [-1,1], regardless of winLossUtilityFactor
  const double resignScoreThreshold = cfg.contains("allowResignation") ? cfg.getDouble("resignScoreThreshold",-10000.0,0.0) : -10000.0;

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  AsyncBot* bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);
  
  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting demo");

  //Game loop
  while(true) {

    Player pla = P_BLACK;
    Board baseBoard;
    BoardHistory baseHist(baseBoard,pla,Rules::getTrompTaylorish(),0);
    TimeControls tc;

    bot->setPosition(pla,baseBoard,baseHist);
    
    vector<double> recentWinLossValues;
    vector<double> recentScores;
    vector<double> recentScoreStdevs;

    double callbackPeriod = 0.10;

    auto callback = [&baseHist,&recentWinLossValues,recentScores,recentScoreStdevs](Search* search) {
      writeLine(search,baseHist,recentWinLossValues,recentScores,recentScoreStdevs);
    };
    
    //Move loop
    int maxMovesPerGame = 1600;
    for(int i = 0; i<maxMovesPerGame; i++) {
      baseHist.endGameIfAllPassAlive(baseBoard);
      if(baseHist.isGameFinished)
        break;

      double searchFactor = 1.0;
      Loc moveLoc = bot->genMoveSynchronousAnalyze(pla,tc,searchFactor,callbackPeriod,callback);

      bool isLegal = bot->isLegal(moveLoc,pla);
      if(moveLoc == Board::NULL_LOC || !isLegal) {
        ostringstream sout;
        sout << "genmove null location or illegal move!?!" << "\n";
        sout << bot->getRootBoard() << "\n";
        sout << "Pla: " << playerToString(pla) << "\n";
        sout << "MoveLoc: " << Location::toString(moveLoc,bot->getRootBoard()) << "\n";
        logger.write(sout.str());
        cerr << sout.str() << endl;
        throw new StringError("illegal move");
      }

      double winLossValue;
      double expectedScore;
      double expectedScoreStdev;
      {
        ReportedSearchValues values;
        bool success = bot->getSearch()->getRootValues(values);
        assert(success);
        winLossValue = values.winLossValue;
        expectedScore = values.expectedScore;
        expectedScoreStdev = values.expectedScoreStdev;
      }

      recentWinLossValues.push_back(winLossValue);
      recentScores.push_back(expectedScore);
      recentScoreStdevs.push_back(expectedScoreStdev);
        
      bool resigned = false;
      if(allowResignation) {
        const BoardHistory hist = bot->getRootHist();
        const Board initialBoard = hist.initialBoard; 

        //Play at least some moves no matter what
        int minTurnForResignation = 1 + initialBoard.x_size * initialBoard.y_size / 6;
          
        Player resignPlayerThisTurn = C_EMPTY;
        if(winLossValue < resignThreshold && expectedScore < resignScoreThreshold)
          resignPlayerThisTurn = P_WHITE;
        else if(winLossValue > -resignThreshold && expectedScore > -resignScoreThreshold)
          resignPlayerThisTurn = P_BLACK;

        if(resignPlayerThisTurn == pla &&
           bot->getRootHist().moveHistory.size() >= minTurnForResignation)
          resigned = true;
      }

      if(resigned) {
        baseHist.setWinnerByResignation(getOpp(pla));
      }
      else {
        bool suc = bot->makeMove(moveLoc,pla);
        assert(suc);
        //And make the move on our copy of the board
        assert(baseHist.isLegal(baseBoard,moveLoc,pla));
        baseHist.makeBoardMoveAssumeLegal(baseBoard,moveLoc,pla,NULL);

        pla = getOpp(pla);
      }
    }

    //End of game display line
    writeLine(bot->getSearch(),baseHist,recentWinLossValues,recentScores,recentScoreStdevs);
    //Wait a bit before diving into the next game
    std::this_thread::sleep_for(std::chrono::seconds(10));

    bot->clearSearch();
  }

  delete bot;
  delete nnEval;
  NeuralNet::globalCleanup();

  logger.write("All cleaned up, quitting");
  return 0;

}

int MainCmds::writeSearchValueTimeseries(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  string configFile;
  string nnModelFile;
  vector<string> sgfsDirs;
  string outputFile;
  int numThreads;
  double usePosProb;
  string mode;
  try {
    TCLAP::CmdLine cmd("Write search value timeseries", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> nnModelFileArg("","nn-model-file","Neural net model .pb graph file to use",true,string(),"FILE");
    TCLAP::MultiArg<string> sgfsDirsArg("","sgfs-dir","Directory of sgfs files",true,"DIR");
    TCLAP::ValueArg<string> outputFileArg("","output-csv","Output csv file",true,string(),"FILE");
    TCLAP::ValueArg<int> numThreadsArg("","num-threads","Number of threads to use",true,1,"INT");
    TCLAP::ValueArg<double> usePosProbArg("","use-pos-prob","Probability to use a position",true,0.0,"PROB");
    TCLAP::ValueArg<string> modeArg("","mode","rootValue|policyTargetSurprise",true,string(),"MODE");

    cmd.add(configFileArg);
    cmd.add(nnModelFileArg);
    cmd.add(sgfsDirsArg);
    cmd.add(outputFileArg);
    cmd.add(numThreadsArg);
    cmd.add(usePosProbArg);
    cmd.add(modeArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    nnModelFile = nnModelFileArg.getValue();
    sgfsDirs = sgfsDirsArg.getValue();
    outputFile = outputFileArg.getValue();
    numThreads = numThreadsArg.getValue();
    usePosProb = usePosProbArg.getValue();
    mode = modeArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  if(mode != "rootValue" && mode != "policyTargetSurprise") {
    cout << "Error: mode must be rootValue or policyTargetSurprise" << endl;
    return 1;
  }

  ConfigParser cfg(configFile);

  Logger logger;
  logger.setLogToStdout(true);

  logger.write("Engine starting...");

  Rules initialRules;
  {
    string koRule = cfg.getString("koRule", Rules::koRuleStrings());
    string scoringRule = cfg.getString("scoringRule", Rules::scoringRuleStrings());
    bool multiStoneSuicideLegal = cfg.getBool("multiStoneSuicideLegal");
    float komi = 7.5f; //Default komi, sgf will generally override this

    initialRules.koRule = Rules::parseKoRule(koRule);
    initialRules.scoringRule = Rules::parseScoringRule(scoringRule);
    initialRules.multiStoneSuicideLegal = multiStoneSuicideLegal;
    initialRules.komi = komi;
  }

  SearchParams params;
  {
    vector<SearchParams> paramss = Setup::loadParams(cfg);
    if(paramss.size() != 1)
      throw StringError("Can only specify examply one search bot in sgf mode");
    params = paramss[0];
  }


  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators({nnModelFile},{nnModelFile},cfg,logger,seedRand,maxConcurrentEvals,false,false);
    assert(nnEvals.size() == 1);
    nnEval = nnEvals[0];
  }
  int posLen = nnEval->getPosLen();
  int policySize = NNPos::getPolicySize(posLen);
  logger.write("Loaded neural net");

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  const string sgfsSuffix = ".sgfs";
  auto sgfsFilter = [&](const string& name) {
    return Global::isSuffix(name,sgfsSuffix);
  };

  vector<string> sgfsFiles;
  for(int i = 0; i<sgfsDirs.size(); i++)
    Global::collectFiles(sgfsDirs[i], sgfsFilter, sgfsFiles);
  cout << "Found " << sgfsFiles.size() << " files!" << endl;

  vector<Sgf*> sgfs = Sgf::loadSgfsFiles(sgfsFiles);
  cout << "Read " << sgfs.size() << " sgfs!" << endl;
  {
    uint64_t numPoses = 0;
    vector<Move> movesBuf;
    for(size_t i = 0; i<sgfs.size(); i++) {
      sgfs[i]->getMoves(movesBuf,sgfs[i]->getBSize());
      numPoses += movesBuf.size();
      movesBuf.clear();
    }
    cout << "Num unique poses: " << numPoses << endl;
    cout << "(avg moves per game): " << ((double)numPoses / sgfs.size()) << endl;
  }

  ofstream out;
  out.open(outputFile);
  mutex outMutex;

  auto computeSurprise = [&](Search* search) {
    vector<Loc> locs;
    vector<double> playSelectionValues;
    int64_t unreducedNumVisitsBuf;
    bool suc = search->getPlaySelectionValues(locs,playSelectionValues,unreducedNumVisitsBuf,0.0);
    testAssert(suc);

    assert(search->rootNode != NULL);
    assert(search->rootNode->nnOutput != NULL);
    float* policyProbs = search->rootNode->nnOutput->policyProbs;

    assert(locs.size() == playSelectionValues.size());
    double sum = 0.0;
    for(int i = 0; i<locs.size(); i++) {
      sum += playSelectionValues[i];
      assert(playSelectionValues[i] >= 0.0);
    }
    assert(sum > 0.0);

    for(int i = 0; i<locs.size(); i++) {
      playSelectionValues[i] /= sum;
    }

    double surprise = 0.0;
    for(int i = 0; i<locs.size(); i++) {
      if(playSelectionValues[i] > 1e-50) {
        Loc loc = locs[i];
        int pos = NNPos::locToPos(loc,search->rootBoard.x_size,posLen);
        //surprise += playSelectionValues[i] * (log(playSelectionValues[i]) - log(policyProbs[pos]));
        surprise += playSelectionValues[i] * log(policyProbs[pos]);
      }
    }
    return surprise;
  };

  auto runThread = [&](int threadIdx, string randSeed) {
    Search* search = new Search(params,nnEval,randSeed);

    int maxVisits;
    if(mode == "rootValue")
      maxVisits = 80000;
    else if(mode == "policyTargetSurprise")
      maxVisits = 5000;
    else
      assert(false);

    double* utilities = new double[maxVisits];
    double* policySurpriseNats = new double[maxVisits];
    Rand rand("root variance estimate " + Global::intToString(threadIdx));
    for(size_t sgfIdx = threadIdx; sgfIdx<sgfs.size(); sgfIdx += numThreads) {
      const Sgf* sgf = sgfs[sgfIdx];
      int bSize = sgf->getBSize();
      float komi = sgf->getKomi();
      initialRules.komi = komi;

      vector<Move> placements;
      sgf->getPlacements(placements, bSize);
      vector<Move> moves;
      sgf->getMoves(moves, bSize);

      {
        Board board(bSize,bSize);
        Player pla = P_BLACK;
        for(int i = 0; i<placements.size(); i++)
          board.setStone(placements[i].loc,placements[i].pla);

        BoardHistory hist(board,pla,initialRules,0);
        search->setPosition(pla,board,hist);
      }

      for(size_t moveNum = 0; moveNum < moves.size(); moveNum++) {

        if(rand.nextDouble() < usePosProb) {
          SearchThread* stbuf = new SearchThread(0,*search,&logger);
          search->beginSearch(logger);
          for(int i = 0; i<maxVisits; i++) {
            search->runSinglePlayout(*stbuf);
            utilities[i] = search->getRootUtility();
            policySurpriseNats[i] = computeSurprise(search);
          }
          delete stbuf;

          {
            float* policy = search->rootNode->nnOutput->policyProbs;
            double entropy = 0.0;
            for(int i = 0; i<policySize; i++) {
              if(policy[i] < 1e-20)
                continue;
              entropy -= policy[i] * log(policy[i]);
            }

            std::lock_guard<std::mutex> guard(outMutex);
            out << moveNum << ",";
            out << entropy << ",";
            if(mode == "rootValue") {
              for(int i = 0; i<maxVisits; i++)
                out << utilities[i] << ",";
            }
            else if(mode == "policyTargetSurprise") {
              for(int i = 0; i<maxVisits; i++)
                out << policySurpriseNats[i] << ",";
            }
            else {
              assert(false);
            }

            out << endl;
          }
        }

        search->makeMove(moves[moveNum].loc,moves[moveNum].pla);
        search->clearSearch();
      }
    }
    delete search;
    delete[] utilities;
    delete[] policySurpriseNats;
  };

  std::thread threads[numThreads];
  for(int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
    threads[threadIdx] = std::thread(runThread, threadIdx, Global::uint64ToString(seedRand.nextUInt64()));
  }
  for(int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
    threads[threadIdx].join();
  }

  out.close();

  for(size_t i = 0; i<sgfs.size(); i++)
    delete sgfs[i];
  sgfs.clear();

  cout << "Done!" << endl;

  return 0;
}
