#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "search/asyncbot.h"

using namespace std;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

static bool tryParsePlayer(const string& s, Player& pla) {
  string str = Global::toLower(s);
  if(str == "black" || str == "b") {
    pla = P_BLACK;
    return true;
  }
  else if(str == "white" || str == "w") {
    pla = P_WHITE;
    return true;
  }
  return false;
}

static bool tryParseLoc(const string& s, const Board& b, Loc& loc) {
  return Location::tryOfString(s,b,loc);
}

int main(int argc, const char* argv[]) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  string nnModelFile;
  try {
    TCLAP::CmdLine cmd("Sgf->HDF5 data writer", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> nnModelFileArg("","nn-model-file","Neural net model .pb graph file to use",true,string(),"FILE");
    cmd.add(configFileArg);
    cmd.add(nnModelFileArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    nnModelFile = nnModelFileArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << std::endl;
    return 1;
  }

  ConfigParser cfg(configFile);

  Logger logger;
  logger.addFile(cfg.getString("logFile"));
  bool logAllGTPCommunication = cfg.getBool("logAllGTPCommunication");
  bool logSearchInfo = cfg.getBool("logSearchInfo");

  logger.write("GTP Engine starting...");

  NNEvaluator* nnEval;
  {
    bool debugSkipNeuralNet = false;
    nnEval = new NNEvaluator(
      nnModelFile,
      cfg.getInt("nnMaxBatchSize", 1, 65536),
      cfg.getInt("nnCacheSizePowerOfTwo", -1, 48),
      debugSkipNeuralNet
    );


    bool nnRandomize = cfg.getBool("nnRandomize");
    string nnRandSeed;
    if(cfg.contains("nnRandSeed"))
      nnRandSeed = cfg.getString("nnRandSeed");
    else
      nnRandSeed = Global::uint64ToString(seedRand.nextUInt64());
    logger.write("nnRandSeed = " + nnRandSeed);

    vector<string> gpuVisibleDeviceListByThread;
    if(cfg.contains("gpuVisibleDeviceList"))
      gpuVisibleDeviceListByThread.push_back(cfg.getString("gpuVisibleDeviceList"));
    double perProcessGPUMemoryFraction = -1;
    if(cfg.contains("perProcessGPUMemoryFraction"))
       perProcessGPUMemoryFraction = cfg.getDouble("perProcessGPUMemoryFraction",0.0,1.0);

    int numNNServerThreads = 1;
    int defaultSymmetry = 0;
    nnEval->spawnServerThreads(
      numNNServerThreads,
      nnRandomize,
      nnRandSeed,
      defaultSymmetry,
      logger,
      gpuVisibleDeviceListByThread,
      perProcessGPUMemoryFraction
    );
  }
  logger.write("Loaded neural net");

  Rules initialRules;
  {
    string koRule = cfg.getString("koRule", {"SIMPLE","POSITIONAL","SITUATIONAL","SPIGHT"});
    string scoringRule = cfg.getString("scoringRule", {"AREA","TERRITORY"});
    bool multiStoneSuicideLegal = cfg.getBool("multiStoneSuicideLegal");
    float komi = 7.5f; //Default komi, gtp will generally override this

    if(koRule == "SIMPLE") initialRules.koRule = Rules::KO_SIMPLE;
    else if(koRule == "POSITIONAL") initialRules.koRule = Rules::KO_POSITIONAL;
    else if(koRule == "SITUATIONAL") initialRules.koRule = Rules::KO_SITUATIONAL;
    else if(koRule == "SPIGHT") initialRules.koRule = Rules::KO_SPIGHT;
    else assert(false);

    if(scoringRule == "AREA") initialRules.scoringRule = Rules::SCORING_AREA;
    else if(scoringRule == "TERRITORY") initialRules.scoringRule = Rules::SCORING_TERRITORY;
    else assert(false);

    initialRules.multiStoneSuicideLegal = multiStoneSuicideLegal;
    initialRules.komi = komi;
  }


  SearchParams params;
  {
    if(cfg.contains("maxPlayouts"))
      params.maxPlayouts = cfg.getUInt64("maxPlayouts", (uint64_t)1, (uint64_t)1 << 62);
    if(cfg.contains("maxVisits"))
      params.maxVisits = cfg.getUInt64("maxVisits", (uint64_t)1, (uint64_t)1 << 62);
    if(cfg.contains("maxTime"))
      params.maxTime = cfg.getDouble("maxTime", 0.0, 1.0e20);
    params.numThreads = cfg.getInt("numSearchThreads", 1, 1024);

    params.winLossUtilityFactor = cfg.getDouble("winLossUtilityFactor", 0.0, 1.0);
    params.scoreUtilityFactor = cfg.getDouble("scoreUtilityFactor", 0.0, 1.0);
    params.noResultUtilityForWhite = cfg.getDouble("noResultUtilityForWhite", -2.0, 2.0);

    params.cpuctExploration = cfg.getDouble("cpuctExploration", 0.0, 10.0);
    params.fpuReductionMax = cfg.getDouble("fpuReductionMax", 0.0, 2.0);

    params.rootNoiseEnabled = cfg.getBool("rootNoiseEnabled");
    params.rootDirichletNoiseTotalConcentration = cfg.getDouble("rootDirichletNoiseTotalConcentration", 0.001, 10000.0);
    params.rootDirichletNoiseWeight = cfg.getDouble("rootDirichletNoiseWeight", 0.0, 1.0);

    if(cfg.contains("searchRandSeed"))
      params.randSeed = cfg.getString("searchRandSeed");
    else
      params.randSeed = Global::uint64ToString(seedRand.nextUInt64());

    params.chosenMoveTemperature = cfg.getDouble("chosenMoveTemperature", 0.0, 5.0);
    params.chosenMoveSubtract = cfg.getDouble("chosenMoveSubtract", 0.0, 1.0e10);

    params.mutexPoolSize = (uint32_t)cfg.getInt("mutexPoolSize", 1, 1 << 24);
    params.numVirtualLossesPerThread = (int32_t)cfg.getInt("numVirtualLossesPerThread", 1, 1000);
  }

  bool ponderingEnabled = cfg.getBool("ponderingEnabled");

  AsyncBot* bot = new AsyncBot(params, nnEval, &logger);
  {
    Board board(19,19);
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,initialRules);
    bot->setPosition(pla,board,hist);
  }

  vector<string> knownCommands = {
    "protocol_version",
    "name",
    "version",
    "known_command",
    "list_commands",
    "quit",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
  };

  logger.write("Beginning main protocol loop");

  string line;
  while(cin) {
    getline(cin,line);
    //Filter down to only "normal" ascii characters. Also excludes carrage returns.
    //Newlines are already handled by getline
    size_t newLen = 0;
    for(size_t i = 0; i < line.length(); i++)
      if(((int)line[i] >= 32 && (int)line[i] <= 126) || line[i] == '\t')
        line[newLen++] = line[i];

    line.erase(line.begin()+newLen, line.end());

    //Remove comments
    size_t commentPos = line.find("#");
    if(commentPos != string::npos)
      line = line.substr(0, commentPos);

    //Convert tabs to spaces
    for(size_t i = 0; i < line.length(); i++)
      if(line[i] == '\t')
        line[i] = ' ';

    line = Global::trim(line);
    if(line.length() == 0)
      continue;

    assert(line.length() > 0);

    string strippedLine = line;
    if(logAllGTPCommunication)
      logger.write("Controller: " + strippedLine);

    //Parse id number of command, if present
    bool hasId = false;
    int id = 0;
    {
      size_t digitPrefixLen = 0;
      while(digitPrefixLen < line.length() && Global::isDigit(line[digitPrefixLen]))
        digitPrefixLen++;
      if(digitPrefixLen > 0) {
        hasId = true;
        try {
          id = Global::parseDigits(line,0,digitPrefixLen);
        }
        catch(const IOError& e) {
          cout << "? GTP id '" << id << "' could not be parsed: " << e.what() << endl;
          continue;
        }
        line = line.substr(digitPrefixLen);
      }
    }

    line = Global::trim(line);
    if(line.length() <= 0) {
      cout << "? empty command" << endl;
      continue;
    }

    vector<string> pieces = Global::split(line,' ');
    for(size_t i = 0; i<pieces.size(); i++)
      pieces[i] = Global::trim(pieces[i]);
    assert(pieces.size() > 0);

    string command = pieces[0];
    pieces.erase(pieces.begin());

    bool responseIsError = false;
    bool shouldQuitAfterResponse = false;
    bool maybeStartPondering = false;
    string response;

    if(command == "protocol_version") {
      response = "2";
    }

    else if(command == "name") {
      response = "lightvector/GoNN GTP Bot";
    }

    else if(command == "version") {
      response = nnModelFile;
    }

    else if(command == "known_command") {
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected single argument for known_command but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        if(std::find(knownCommands.begin(), knownCommands.end(), pieces[0]) != knownCommands.end())
          response = "true";
        else
          response = "false";
      }
    }

    else if(command == "list_commands") {
      for(size_t i = 0; i<knownCommands.size(); i++)
        response += knownCommands[i] + "\n";
    }

    else if(command == "quit") {
      shouldQuitAfterResponse = true;
      logger.write("Quit requested by controller");
    }

    else if(command == "boardsize") {
      int newBSize = 0;
      if(pieces.size() != 1 || !Global::tryStringToInt(pieces[0],newBSize)) {
        responseIsError = true;
        response = "Expected single int argument for boardsize but got '" + Global::concat(pieces," ") + "'";
      }
      else if(newBSize < 9 || newBSize > 19) {
        responseIsError = true;
        response = "unacceptable size";
      }
      else {
        Board board(newBSize,newBSize);
        Player pla = P_BLACK;
        BoardHistory hist(board,pla,bot->getRootHist().rules);
        bot->setPosition(pla,board,hist);
      }
    }

    else if(command == "clear_board") {
      assert(bot->getRootBoard().x_size == bot->getRootBoard().y_size);
      int newBSize = bot->getRootBoard().x_size;
      Board board(newBSize,newBSize);
      Player pla = P_BLACK;
      BoardHistory hist(board,pla,bot->getRootHist().rules);
      bot->setPosition(pla,board,hist);
    }

    else if(command == "komi") {
      float newKomi = 0;
      if(pieces.size() != 1 || !Global::tryStringToFloat(pieces[0],newKomi)) {
        responseIsError = true;
        response = "Expected single float argument for komi but got '" + Global::concat(pieces," ") + "'";
      }
      //GTP spec says that we should accept any komi, but we're going to ignore that.
      else if(isnan(newKomi) || newKomi < -100.0 || newKomi > 100.0) {
        responseIsError = true;
        response = "unacceptable komi";
      }
      else if(newKomi * 2 != (int)(newKomi * 2)) {
        responseIsError = true;
        response = "komi must be an integer or half-integer";
      }
      else {
        bot->setKomi(newKomi);
        //In case the controller tells us komi every move, restart pondering afterward.
        maybeStartPondering = bot->getRootHist().moveHistory.size() > 0;
      }
    }

    else if(command == "play") {
      Player pla;
      Loc loc;
      if(pieces.size() != 2) {
        responseIsError = true;
        response = "Expected two arguments for play but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!tryParsePlayer(pieces[0],pla)) {
        responseIsError = true;
        response = "Could not parse color: '" + pieces[0] + "'";
      }
      else if(!tryParseLoc(pieces[1],bot->getRootBoard(),loc)) {
        responseIsError = true;
        response = "Could not parse vertex: '" + pieces[1] + "'";
      }
      else {
        bool suc = bot->makeMove(loc,pla);
        if(!suc) {
          responseIsError = true;
          response = "illegal move";
        }
        maybeStartPondering = true;
      }
    }

    else if(command == "genmove") {
      Player pla;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for genmove but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!tryParsePlayer(pieces[0],pla)) {
        responseIsError = true;
        response = "Could not parse color: '" + pieces[0] + "'";
      }
      else {
        ClockTimer timer;
        nnEval->clearStats();
        Loc loc = bot->genMoveSynchronous(pla);
        bool isLegal = bot->isLegal(loc,pla);
        if(loc == Board::NULL_LOC || !isLegal) {
          responseIsError = true;
          response = "genmove returned null location or illegal move";
          ostringstream sout;
          sout << "genmove null location or illegal move!?!" << "\n";
          sout << bot->getRootBoard() << "\n";
          sout << "Pla: " << playerToString(pla) << "\n";
          sout << "Loc: " << Location::toString(loc,bot->getRootBoard()) << "\n";
          logger.write(sout.str());
        }
        response = Location::toString(loc,bot->getRootBoard());

        if(logSearchInfo) {
          Search* search = bot->getSearch();
          ostringstream sout;
          sout << bot->getRootBoard() << "\n";
          sout << "Time taken: " << timer.getSeconds() << "\n";
          sout << "Root visits: " << search->numRootVisits() << "\n";
          sout << "NN rows: " << nnEval->numRowsProcessed() << endl;
          sout << "NN batches: " << nnEval->numBatchesProcessed() << endl;
          sout << "NN avg batch size: " << nnEval->averageProcessedBatchSize() << endl;
          sout << "PV: ";
          search->printPV(sout, search->rootNode, 25);
          sout << "\n";
          sout << "Tree:\n";
          search->printTree(sout, search->rootNode, PrintTreeOptions().maxDepth(1));
          logger.write(sout.str());
        }

        bool suc = bot->makeMove(loc,pla);
        assert(suc);

        maybeStartPondering = true;
      }
    }

    else {
      responseIsError = true;
      response = "unknown command";
    }


    //Postprocessing of response
    if(hasId)
      response = Global::intToString(id) + " " + response;
    if(responseIsError)
      response = "?" + response;
    else
      response = "=" + response;

    cout << response << endl;

    if(logAllGTPCommunication)
      logger.write(response);

    if(shouldQuitAfterResponse)
      break;

    if(maybeStartPondering && ponderingEnabled)
      bot->ponder();

  } //Close read loop


  delete bot;
  delete nnEval;

  logger.write("All cleaned up, quitting");
  return 0;
}

