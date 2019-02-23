#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "program/play.h"
#include "main.h"

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

int MainCmds::gtp(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  string configFile;
  string nnModelFile;
  try {
    TCLAP::CmdLine cmd("Run GTP engine", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> nnModelFileArg("","model","Neural net model file",true,string(),"FILE");
    cmd.add(configFileArg);
    cmd.add(nnModelFileArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    nnModelFile = nnModelFileArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  ConfigParser cfg(configFile);

  Logger logger;
  logger.addFile(cfg.getString("logFile"));
  bool logAllGTPCommunication = cfg.getBool("logAllGTPCommunication");
  bool logSearchInfo = cfg.getBool("logSearchInfo");

  logger.write("GTP Engine starting...");

  Rules initialRules;
  {
    string koRule = cfg.getString("koRule", Rules::koRuleStrings());
    string scoringRule = cfg.getString("scoringRule", Rules::scoringRuleStrings());
    bool multiStoneSuicideLegal = cfg.getBool("multiStoneSuicideLegal");
    float komi = 7.5f; //Default komi, gtp will generally override this

    initialRules.koRule = Rules::parseKoRule(koRule);
    initialRules.scoringRule = Rules::parseScoringRule(scoringRule);
    initialRules.multiStoneSuicideLegal = multiStoneSuicideLegal;
    initialRules.komi = komi;
  }

  SearchParams params;
  {
    vector<SearchParams> paramss = Setup::loadParams(cfg);
    if(paramss.size() != 1)
      throw StringError("Can only specify examply one search bot in gtp mode");
    params = paramss[0];
  }

  string searchRandSeed;
  if(cfg.contains("searchRandSeed"))
    searchRandSeed = cfg.getString("searchRandSeed");
  else
    searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

  bool ponderingEnabled = cfg.getBool("ponderingEnabled");
  bool cleanupBeforePass = cfg.contains("cleanupBeforePass") ? cfg.getBool("cleanupBeforePass") : false;
  bool allowResignation = cfg.contains("allowResignation") ? cfg.getBool("allowResignation") : false;
  double resignThreshold = cfg.contains("allowResignation") ? cfg.getDouble("resignThreshold",-1.0,0.0) : -1.0; //Threshold on [-1,1], regardless of winLossUtilityFactor

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators({nnModelFile},{nnModelFile},cfg,logger,seedRand,maxConcurrentEvals,false);
    assert(nnEvals.size() == 1);
    nnEval = nnEvals[0];
  }
  logger.write("Loaded neural net");


  AsyncBot* bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);
  {
    Board board(19,19);
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,initialRules,0);
    bot->setPosition(pla,board,hist);
  }


  {
    vector<string> unusedKeys = cfg.unusedKeys();
    for(size_t i = 0; i<unusedKeys.size(); i++) {
      string msg = "WARNING: Unused key '" + unusedKeys[i] + "' in " + configFile;
      logger.write(msg);
      cerr << msg << endl;
    }
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
    "showboard",
    "place_free_handicap",
    "set_free_handicap",
    "final_score",
    "final_status_list",
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
        BoardHistory hist(board,pla,bot->getRootHist().rules,0);
        bot->setPosition(pla,board,hist);
      }
    }

    else if(command == "clear_board") {
      assert(bot->getRootBoard().x_size == bot->getRootBoard().y_size);
      int newBSize = bot->getRootBoard().x_size;
      Board board(newBSize,newBSize);
      Player pla = P_BLACK;
      BoardHistory hist(board,pla,bot->getRootHist().rules,0);
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
        Loc moveLoc = bot->genMoveSynchronous(pla);
        bool isLegal = bot->isLegal(moveLoc,pla);
        if(moveLoc == Board::NULL_LOC || !isLegal) {
          responseIsError = true;
          response = "genmove returned null location or illegal move";
          ostringstream sout;
          sout << "genmove null location or illegal move!?!" << "\n";
          sout << bot->getRootBoard() << "\n";
          sout << "Pla: " << playerToString(pla) << "\n";
          sout << "MoveLoc: " << Location::toString(moveLoc,bot->getRootBoard()) << "\n";
          logger.write(sout.str());
        }

        //Implement cleanupBeforePass hack - the bot wants to pass, so instead cleanup if there is something to clean
        if(cleanupBeforePass && moveLoc == Board::PASS_LOC) {
          Board board = bot->getRootBoard();
          BoardHistory hist = bot->getRootHist();
          Color* safeArea = bot->getSearch()->rootSafeArea;
          assert(safeArea != NULL);
          //Scan the board for any spot that is adjacent to an opponent group that is part of our pass-alive territory.
          for(int y = 0; y<board.y_size; y++) {
            for(int x = 0; x<board.x_size; x++) {
              Loc otherLoc = Location::getLoc(x,y,board.x_size);
              if(moveLoc == Board::PASS_LOC &&
                 board.colors[otherLoc] == C_EMPTY &&
                 safeArea[otherLoc] == pla &&
                 board.isAdjacentToPla(otherLoc,getOpp(pla)) &&
                 hist.isLegal(board,otherLoc,pla)
              ) {
                moveLoc = otherLoc;
              }
            }
          }
        }

        bool resigned = false;
        if(allowResignation) {
          double winValue;
          double lossValue;
          double noResultValue;
          double staticScoreValue;
          double dynamicScoreValue;
          double expectedScore;
          bool success = bot->getSearch()->getRootValues(winValue,lossValue,noResultValue,staticScoreValue,dynamicScoreValue,expectedScore);
          assert(success);

          double winLossValue = winValue - lossValue;
          assert(winLossValue > -1.01 && winLossValue < 1.01); //Sanity check, but allow generously for float imprecision
          if(winLossValue > 1.0) winLossValue = 1.0;
          if(winLossValue < -1.0) winLossValue = -1.0;

          Player resignPlayerThisTurn = C_EMPTY;
          if(winLossValue < resignThreshold)
            resignPlayerThisTurn = P_WHITE;
          else if(winLossValue > -resignThreshold)
            resignPlayerThisTurn = P_BLACK;

          if(resignPlayerThisTurn == pla)
            resigned = true;
        }

        if(resigned)
          response = "resign";
        else
          response = Location::toString(moveLoc,bot->getRootBoard());

        if(logSearchInfo) {
          Search* search = bot->getSearch();
          ostringstream sout;
          Board::printBoard(sout, bot->getRootBoard(), moveLoc, &(bot->getRootHist().moveHistory));
          sout << "\n";
          sout << "Time taken: " << timer.getSeconds() << "\n";
          sout << "Root visits: " << search->numRootVisits() << "\n";
          sout << "NN rows: " << nnEval->numRowsProcessed() << endl;
          sout << "NN batches: " << nnEval->numBatchesProcessed() << endl;
          sout << "NN avg batch size: " << nnEval->averageProcessedBatchSize() << endl;
          sout << "PV: ";
          search->printPV(sout, search->rootNode, 25);
          sout << "\n";
          sout << "Tree:\n";
          search->printTree(sout, search->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10));
          logger.write(sout.str());
        }

        if(!resigned) {
          bool suc = bot->makeMove(moveLoc,pla);
          assert(suc);
          maybeStartPondering = true;
        }

      }
    }

    else if(command == "showboard") {
      ostringstream sout;
      Board::printBoard(sout, bot->getRootBoard(), Board::NULL_LOC, &(bot->getRootHist().moveHistory));
      response = Global::trim(sout.str());
    }

    else if(command == "place_free_handicap") {
      int n;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for genmove but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!Global::tryStringToInt(pieces[0],n)) {
        responseIsError = true;
        response = "Could not parse number of handicap stones: '" + pieces[0] + "'";
      }
      else if(n < 2) {
        responseIsError = true;
        response = "Number of handicap stones less than 2: '" + pieces[0] + "'";
      }
      else if(!bot->getRootBoard().isEmpty()) {
        responseIsError = true;
        response = "Board is not empty";
      }
      else {
        //If asked to place more, we just go ahead and only place up to 30, or a quarter of the board
        int xSize = bot->getRootBoard().x_size;
        int ySize = bot->getRootBoard().y_size;
        int maxHandicap = xSize*ySize / 4;
        if(maxHandicap > 30)
          maxHandicap = 30;
        if(n > maxHandicap)
          n = maxHandicap;

        Board board(xSize,ySize);
        Player pla = P_BLACK;
        BoardHistory hist(board,pla,bot->getRootHist().rules,0);
        double extraBlackTemperature = 0.25;
        bool adjustKomi = false;
        int numVisitsForKomi = 0;
        Rand rand;
        ExtraBlackAndKomi extraBlackAndKomi(n,hist.rules.komi,hist.rules.komi);
        Play::playExtraBlack(bot->getSearch(), logger, extraBlackAndKomi, board, hist, extraBlackTemperature, rand, adjustKomi, numVisitsForKomi);

        response = "";
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            Loc loc = Location::getLoc(x,y,board.x_size);
            if(board.colors[loc] != C_EMPTY) {
              response += " " + Location::toString(loc,board);
            }
          }
        }
        response = Global::trim(response);

        bot->setPosition(pla,board,hist);
      }
    }

    else if(command == "set_free_handicap") {
      if(!bot->getRootBoard().isEmpty()) {
        responseIsError = true;
        response = "Board is not empty";
      }
      else {
        vector<Loc> locs;
        int xSize = bot->getRootBoard().x_size;
        int ySize = bot->getRootBoard().y_size;
        Board board(xSize,ySize);
        for(int i = 0; i<pieces.size(); i++) {
          Loc loc;
          bool suc = tryParseLoc(pieces[i],board,loc);
          if(!suc || loc == Board::PASS_LOC) {
            responseIsError = true;
            response = "Invalid handicap location: " + pieces[i];
          }
          locs.push_back(loc);
        }
        for(int i = 0; i<locs.size(); i++)
          board.setStone(locs[i],P_BLACK);
        Player pla = P_BLACK;
        BoardHistory hist(board,pla,bot->getRootHist().rules,0);

        bot->setPosition(pla,board,hist);
      }
    }

    else if(command == "final_score") {
      //Returns the resulting score if this position were scored AS-IS (players repeatedly passing until the game ends),
      //rather than attempting to estimate what the score would be with further playouts
      Board board = bot->getRootBoard();
      BoardHistory hist = bot->getRootHist();

      //For GTP purposes, we treat noResult as a draw since there is no provision for anything else.
      if(!hist.isGameFinished)
        hist.endAndScoreGameNow(board);

      if(hist.winner == C_EMPTY)
        response = "0";
      else if(hist.winner == C_BLACK)
        response = "B+" + Global::strprintf("%.1f",-hist.finalWhiteMinusBlackScore);
      else if(hist.winner == C_WHITE)
        response = "W+" + Global::strprintf("%.1f",hist.finalWhiteMinusBlackScore);
      else
        assert(false);
    }

    else if(command == "final_status_list") {
      int statusMode = 0;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for final_status_list but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        if(pieces[0] == "alive")
          statusMode = 0;
        else if(pieces[0] == "seki")
          statusMode = 1;
        else if(pieces[0] == "dead")
          statusMode = 2;
        else {
          responseIsError = true;
          response = "Argument to final_status_list must be 'alive' or 'seki' or 'dead'";
          statusMode = 3;
        }

        if(statusMode < 3) {
          vector<Loc> locsToReport;
          Board board = bot->getRootBoard();
          BoardHistory hist = bot->getRootHist();

          if(hist.isGameFinished && hist.isNoResult) {
            //Treat all stones as alive under a no result
            if(statusMode == 0) {
              for(int y = 0; y<board.y_size; y++) {
                for(int x = 0; x<board.x_size; x++) {
                  Loc loc = Location::getLoc(x,y,board.x_size);
                  if(board.colors[loc] != C_EMPTY)
                    locsToReport.push_back(loc);
                }
              }
            }
          }
          else {
            Color area[Board::MAX_ARR_SIZE];
            hist.endAndScoreGameNow(board,area);
            for(int y = 0; y<board.y_size; y++) {
              for(int x = 0; x<board.x_size; x++) {
                Loc loc = Location::getLoc(x,y,board.x_size);
                if(board.colors[loc] != C_EMPTY) {
                  if(statusMode == 0 && board.colors[loc] == area[loc])
                    locsToReport.push_back(loc);
                  else if(statusMode == 2 && board.colors[loc] != area[loc])
                    locsToReport.push_back(loc);
                }
              }
            }
          }

          response = "";
          for(int i = 0; i<locsToReport.size(); i++) {
            Loc loc = locsToReport[i];
            if(i > 0)
              response += " ";
            response += Location::toString(loc,board);
          }
        }
      }
    }

    else {
      responseIsError = true;
      response = "unknown command";
    }


    //Postprocessing of response
    if(hasId)
      response = Global::intToString(id) + " " + response;
    else
      response = " " + response;

    if(responseIsError)
      response = "?" + response;
    else
      response = "=" + response;

    cout << response << endl;
    cout << endl; //GTP needs extra newline

    if(logAllGTPCommunication)
      logger.write(response);

    if(shouldQuitAfterResponse)
      break;

    if(maybeStartPondering && ponderingEnabled)
      bot->ponder();

  } //Close read loop


  delete bot;
  delete nnEval;
  NeuralNet::globalCleanup();

  logger.write("All cleaned up, quitting");
  return 0;
}

