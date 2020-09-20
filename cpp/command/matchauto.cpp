#include "../core/global.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/elo.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include <boost/filesystem.hpp>
#include <csignal>

using namespace std;

static std::atomic<bool> sigReceived(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM)
    sigReceived.store(true);
}

namespace {
  struct NetAndStuff {
    NNEvaluator* nnEval;

    int gamesTotal;
    int gamesCompleted;
    int gamesActive;

  public:
    NetAndStuff(
      NNEvaluator* neval
    )
      :nnEval(neval),
       gamesTotal(0),
       gamesCompleted(0),
       gamesActive(0)
    {
    }

    ~NetAndStuff() {
      delete nnEval;
    }

    //NOT threadsafe - needs to be externally synchronized
    void addGamesTotal(int n) {
      gamesTotal += n;
    }

    //NOT threadsafe - needs to be externally synchronized
    void addGamesActive(int n) {
      gamesActive += n;
    }

    //NOT threadsafe - needs to be externally synchronized
    void addGamesCompleted(int n) {
      gamesCompleted += n;
    }
  };

  struct NetManager {
    ConfigParser* cfg;
    Rand seedRand;
    int maxConcurrentEvals;
    int expectedConcurrentEvals;

    map<string, NetAndStuff*> loadedNets;

    std::mutex managerLock;

  public:
    NetManager(
      ConfigParser* c,
      int maxConcurrentEvs,
      int expectedConcurrentEvs
    )
      :cfg(c),
       seedRand(),
       maxConcurrentEvals(maxConcurrentEvs),
       expectedConcurrentEvals(expectedConcurrentEvs),
       loadedNets()
    {
    }

    ~NetManager() {
      auto iter = loadedNets.begin();
      for(; iter != loadedNets.end(); ++iter) {
        delete iter->second;
      }
    }

    void preregisterGames(const string& nnModelFile, Logger& logger, int n) {
      std::lock_guard<std::mutex> lock(managerLock);

      auto iter = loadedNets.find(nnModelFile);
      NetAndStuff* netAndStuff;
      if(iter == loadedNets.end()) {
        int defaultMaxBatchSize = -1;
        NNEvaluator* nnEval = Setup::initializeNNEvaluator(
          nnModelFile,nnModelFile,*cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
          NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,
          Setup::SETUP_FOR_MATCH
        );
        netAndStuff = new NetAndStuff(nnEval);
        loadedNets[nnModelFile] = netAndStuff;

        //Check for unused config keys
        cfg->warnUnusedKeys(cerr,&logger);

      }
      else {
        netAndStuff = iter->second;
      }

      assert(n > 0);
      netAndStuff->addGamesTotal(n);
    }

    NNEvaluator* registerStarting(const string& nnModelFile) {
      std::lock_guard<std::mutex> lock(managerLock);
      auto iter = loadedNets.find(nnModelFile);
      assert(iter != loadedNets.end());
      NetAndStuff* netAndStuff = iter->second;
      netAndStuff->addGamesActive(1);
      return netAndStuff->nnEval;
    }

    void registerFinishing(const string& nnModelFile) {
      std::lock_guard<std::mutex> lock(managerLock);
      auto iter = loadedNets.find(nnModelFile);
      assert(iter != loadedNets.end());
      NetAndStuff* netAndStuff = iter->second;
      netAndStuff->addGamesActive(-1);
      netAndStuff->addGamesCompleted(1);
      if(netAndStuff->gamesCompleted == netAndStuff->gamesTotal) {
        assert(netAndStuff->gamesActive == 0);
        loadedNets.erase(iter);
        delete netAndStuff;
      }
    }

  };


  STRUCT_NAMED_TRIPLE(int,forBot,int,b0,int,b1,NextMatchup);

  struct AutoMatchPairer {
    string resultsDir;

    int numBots;
    vector<string> botNames;
    vector<string> nnModelFiles;
    vector<SearchParams> baseParamss;

    vector<NextMatchup> nextMatchups;
    Rand rand;

    int matchRepFactor;

    int64_t numGamesStartedSoFar;
    int64_t numGamesTotal;
    int64_t logGamesEvery;

    std::mutex getMatchupMutex;

    AutoMatchPairer(
      ConfigParser& cfg,
      const string& resDir,
      int nBots,
      const vector<string>& bNames,
      const vector<string>& nFiles,
      const vector<SearchParams>& bParamss
    )
      :resultsDir(resDir),
       numBots(nBots),
       botNames(bNames),
       nnModelFiles(nFiles),
       baseParamss(bParamss),
       nextMatchups(),
       rand(),
       matchRepFactor(1),
       numGamesStartedSoFar(0),
       numGamesTotal(),
       logGamesEvery(),
       getMatchupMutex()
    {
      assert(botNames.size() == numBots);
      assert(nnModelFiles.size() == numBots);
      assert(baseParamss.size() == numBots);
      numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
      logGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);

      if(cfg.contains("matchRepFactor"))
        matchRepFactor = cfg.getInt("matchRepFactor",1,100000);
    }

    ~AutoMatchPairer()
    {}

    bool getMatchup(
      NetManager* manager, string& forBot, MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW, Logger& logger
    )
    {
      std::lock_guard<std::mutex> lock(getMatchupMutex);

      numGamesStartedSoFar += 1;
      if(numGamesStartedSoFar % logGamesEvery == 0)
        logger.write("Started " + Global::int64ToString(numGamesStartedSoFar) + " games");

      NextMatchup matchup = getMatchupPairUnsynchronized(manager,logger);
      forBot = botNames[matchup.forBot];

      botSpecB.botIdx = matchup.b0;
      botSpecB.botName = botNames[matchup.b0];
      botSpecB.nnEval = manager->registerStarting(nnModelFiles[matchup.b0]);
      botSpecB.baseParams = baseParamss[matchup.b0];

      botSpecW.botIdx = matchup.b1;
      botSpecW.botName = botNames[matchup.b1];
      botSpecW.nnEval = manager->registerStarting(nnModelFiles[matchup.b1]);
      botSpecW.baseParams = baseParamss[matchup.b1];

      return true;
    }

    void generateNewMatchups(NetManager* manager, Logger& logger) {
      //Load all results so far for all players

      map<string,int> idxOfBotName;
      for(int b0 = 0; b0<numBots; b0++) {
        idxOfBotName[botNames[b0]] = b0;
      }

      int64_t* numGamesForBot = new int64_t[numBots];
      int64_t* numGamesByBot = new int64_t[numBots];
      for(int b0 = 0; b0<numBots; b0++) {
        numGamesForBot[b0] = 0;
        numGamesByBot[b0] = 0;
      }

      ComputeElos::WLRecord* winMatrix = new ComputeElos::WLRecord[numBots*numBots];
      for(int b0 = 0; b0<numBots; b0++) {
        for(int b1 = 0; b1<numBots; b1++) {
          winMatrix[b0*numBots+b1] = ComputeElos::WLRecord(0.0,0.0);
        }
      }

      namespace bfs = boost::filesystem;

      for(bfs::directory_iterator iter(resultsDir); iter != bfs::directory_iterator(); ++iter) {
        bfs::path dirPath = iter->path();
        if(bfs::is_directory(dirPath))
          continue;
        string file = dirPath.string();
        if(Global::isSuffix(file,".results.csv")) {
          vector<string> lines = Global::readFileLines(file,'\n');
          for(int i = 0; i<lines.size(); i++) {
            string s = Global::trim(lines[i]);
            if(s.length() == 0)
              continue;
            vector<string> pieces = Global::split(s,',');
            if(pieces.size() != 4)
              continue;

            if(!contains(idxOfBotName,pieces[0]) || !contains(idxOfBotName,pieces[1]) || !contains(idxOfBotName,pieces[2]))
              continue;
            if(pieces[3] != "0" && pieces[3] != "1" && pieces[3] != "=")
              continue;

            int b0 = map_get(idxOfBotName,pieces[0]);
            int b1 = map_get(idxOfBotName,pieces[1]);
            int b2 = map_get(idxOfBotName,pieces[2]);
            numGamesForBot[b0]++;
            numGamesByBot[b1]++;
            numGamesByBot[b2]++;
            if(pieces[3] == "0")
              winMatrix[b1*numBots+b2].firstWins += 1.0;
            else if(pieces[3] == "1")
              winMatrix[b1*numBots+b2].secondWins += 1.0;
            else {
              winMatrix[b1*numBots+b2].firstWins += 0.5;
              winMatrix[b1*numBots+b2].secondWins += 0.5;
            }
          }
        }
      }

      double priorWL = 0.01;
      int maxIters = 20000;
      double tolerance = 0.000001;

      vector<double> elos = ComputeElos::computeElos(winMatrix,numBots,priorWL,maxIters,tolerance,NULL);
      vector<double> eloStdevs = ComputeElos::computeApproxEloStdevs(elos,winMatrix,numBots,priorWL);

      {
        ostringstream out;
        out << "Computed elos!" << endl;
        for(int i = 0; i<numBots; i++) {
          out << botNames[i] << " elo " << elos[i] << " stdev " << eloStdevs[i] << " ngames " << numGamesByBot[i] << endl;
        }
        logger.write(out.str());
      }

      vector<int> botIdxsShuffled(numBots);
      for(int i = 0; i<numBots; i++)
        botIdxsShuffled[i] = i;
      for(int i = numBots-1; i>0; i--) {
        int r = rand.nextUInt(i+1);
        std::swap(botIdxsShuffled[r],botIdxsShuffled[i]);
      }

      //Several times in a row, find the bot with the least games played, and chooose a random other bot with probability proportional
      //to the variance of the game result based on a random sample of the predicted elo difference
      //We use the average of games "for" this bot in conjunction with games played
      for(int i = 0; i<10; i++) {
        int bestBot = -1;
        int64_t minVal = (int64_t)1 << 62;
        for(int j = 0; j<numBots; j++) {
          int b = botIdxsShuffled[j];
          int64_t val = numGamesForBot[b] * 2 + numGamesByBot[b];
          if(val < minVal) {
            bestBot = b;
            minVal = val;
          }
        }
        assert(bestBot >= 0);

        vector<double> relProbs(numBots);
        double probSum = 0.0;
        for(int b = 0; b<numBots; b++) {
          if(b == bestBot)
            relProbs[b] = 0.0;
          else {
            //Vary elo a bit based on stdev so that bots that are more uncertain get more variety
            //Not as much as the whole stdev though, to make sure matches are still informative.
            double g = rand.nextGaussian();
            g = std::min(g,10.0);
            g = std::max(g,-10.0);
            double eloDiff = elos[b] - elos[bestBot] + 0.5 * eloStdevs[bestBot] * g;
            double p = ComputeElos::probWin(eloDiff);
            relProbs[b] = p * (1.0-p) + 1e-30; //Add a tiny bit just in case to avoid zero
          }
          probSum += relProbs[b];
        }
        assert(numBots > 1);
        assert(!std::isnan(probSum));
        if(probSum <= 0)
          throw StringError("Negative relative probabilities for matchauto");

        int otherBot = rand.nextUInt(relProbs.data(),numBots);
        if(otherBot == bestBot) //Just in case
          continue;

        logger.write("Scheduling game " + botNames[bestBot] + " vs " + botNames[otherBot] + "elos " +
                     Global::doubleToString(elos[bestBot]) + " " + Global::doubleToString(elos[otherBot]));

        //And schedule the games!
        manager->preregisterGames(nnModelFiles[bestBot],logger,matchRepFactor);
        manager->preregisterGames(nnModelFiles[otherBot],logger,matchRepFactor);

        numGamesForBot[bestBot] += matchRepFactor;
        numGamesByBot[bestBot] += matchRepFactor;
        numGamesByBot[otherBot] += matchRepFactor;

        for(int j = 0; j < matchRepFactor; j++) {
          if(rand.nextBool(0.5))
            nextMatchups.push_back(NextMatchup(bestBot,bestBot,otherBot));
          else
            nextMatchups.push_back(NextMatchup(bestBot,otherBot,bestBot));
        }
      }

      delete[] winMatrix;
      delete[] numGamesForBot;
      delete[] numGamesByBot;
    }

    NextMatchup getMatchupPairUnsynchronized(NetManager* manager, Logger& logger) {
      if(nextMatchups.size() <= 0) {
        generateNewMatchups(manager,logger);
      }
      assert(nextMatchups.size() > 0);

      NextMatchup matchup = nextMatchups.back();
      nextMatchups.pop_back();
      return matchup;
    }
  };

}


int MainCmds::matchauto(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string logFile;
  string sgfOutputDir;
  string resultsDir;
  try {
    KataGoCommandLine cmd("Play different nets against each other with different search settings in a match or tournament, experimental.");
    cmd.addConfigFileArg("","");

    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",false,string(),"FILE");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",false,string(),"DIR");
    TCLAP::ValueArg<string> resultsDirArg("","results-dir","Dir to read/write win loss result files",true,string(),"DIR");
    cmd.add(logFileArg);
    cmd.add(sgfOutputDirArg);
    cmd.add(resultsDirArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parse(argc,argv);

    logFile = logFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();
    resultsDir = resultsDirArg.getValue();

    cmd.getConfig(cfg);
}
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger;
  logger.addFile(logFile);
  bool logToStdout = cfg.getBool("logToStdout");
  logger.setLogToStdout(logToStdout);

  logger.write("Auto Match Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

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

  //Load match runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Work out an upper bound on how many concurrent nneval requests we could end up making.
  int maxConcurrentEvals;
  int expectedConcurrentEvals;
  {
    //Work out the max threads any one bot uses
    int maxBotThreads = 0;
    for(int i = 0; i<numBots; i++)
      if(paramss[i].numThreads > maxBotThreads)
        maxBotThreads = paramss[i].numThreads;
    //Mutiply by the number of concurrent games we could have
    expectedConcurrentEvals = maxBotThreads * numGameThreads;
    //Multiply by 2 and add some buffer, just so we have plenty of headroom.
    maxConcurrentEvals = expectedConcurrentEvals * 2 + 16;
  }

  //Initialize neural net inference engine globals, and set up model manager
  Setup::initializeSession(cfg);

  NetManager* manager = new NetManager(&cfg,maxConcurrentEvals,expectedConcurrentEvals);

  //Initialize object for randomly pairing bots
  AutoMatchPairer * autoMatchPairer = new AutoMatchPairer(cfg,resultsDir,numBots,botNames,nnModelFilesByBot,paramss);

  //Initialize object for randomizing game settings and running games
  PlaySettings playSettings = PlaySettings::loadForMatch(cfg);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);


  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting matches");
  if(!logToStdout)
    cout << "Loaded all config stuff, starting matches" << endl;

  if(sgfOutputDir != string())
    MakeDir::make(sgfOutputDir);
  MakeDir::make(resultsDir);

  if(!std::atomic_is_lock_free(&sigReceived))
    throw StringError("sigReceived is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::mutex resultLock;
  ofstream* resultOut = new ofstream(resultsDir + "/" + Global::uint64ToHexString(seedRand.nextUInt64()) + ".results.csv");

  auto runMatchLoop = [
    &gameRunner,&autoMatchPairer,&sgfOutputDir,&logger,&resultLock,&resultOut,&manager,&gameSeedBase
  ](
    uint64_t threadHash
  ) {
    ofstream* sgfOut = sgfOutputDir.length() > 0 ? (new ofstream(sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs")) : NULL;
    vector<std::atomic<bool>*> stopConditions = {&sigReceived};

    Rand thisLoopSeedRand;
    while(true) {
      if(sigReceived.load())
        break;

      FinishedGameData* gameData = NULL;

      string forBot;
      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      if(autoMatchPairer->getMatchup(manager, forBot, botSpecB, botSpecW, logger)) {
        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, NULL, NULL, logger,
          stopConditions, NULL
        );
      }

      manager->registerFinishing(botSpecB.nnEval->getModelFileName());
      manager->registerFinishing(botSpecW.nnEval->getModelFileName());

      bool shouldContinue = gameData != NULL;
      if(gameData != NULL) {
        if(sgfOut != NULL) {
          WriteSgf::writeSgf(*sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false);
          (*sgfOut) << endl;
        }

        {
          ostringstream out;
          out << forBot << "," << botSpecB.botName << "," << botSpecW.botName << ",";
          if(gameData->endHist.winner == P_BLACK)
            out << "0";
          else if(gameData->endHist.winner == P_WHITE)
            out << "1";
          else
            out << "=";

          std::lock_guard<std::mutex> lock(resultLock);
          (*resultOut) << out.str() << endl;
        }

        delete gameData;
      }

      if(sigReceived.load())
        break;
      if(!shouldContinue)
        break;
    }
    if(sgfOut != NULL) {
      sgfOut->close();
      delete sgfOut;
    }
  };

  Rand hashRand;
  vector<std::thread> threads;
  for(int i = 0; i<numGameThreads; i++) {
    threads.push_back(std::thread(runMatchLoop, hashRand.nextUInt64()));
  }
  for(int i = 0; i<numGameThreads; i++)
    threads[i].join();

  delete autoMatchPairer;
  delete gameRunner;
  delete manager;

  NeuralNet::globalCleanup();
  ScoreValue::freeTables();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
