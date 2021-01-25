#include "../core/global.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/test.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <csignal>

using namespace std;

static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

static void writeLine(
  const Search* search, const BoardHistory& baseHist,
  const vector<double>& winLossHistory, const vector<double>& scoreHistory, const vector<double>& scoreStdevHistory
) {
  const Board board = search->getRootBoard();
  int nnXLen = search->nnXLen;
  int nnYLen = search->nnYLen;

  cout << board.x_size << " ";
  cout << board.y_size << " ";
  cout << nnXLen << " ";
  cout << nnYLen << " ";
  cout << baseHist.rules.komi << " ";
  if(baseHist.isGameFinished) {
    cout << PlayerIO::playerToString(baseHist.winner) << " ";
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
  cout << NNPos::locToPos(moveLoc,board.x_size,nnXLen,nnYLen) << " ";

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
    search->getAnalysisData(buf,minMovesToTryToGet,false,9);
  }
  cout << buf.size() << " ";
  for(int i = 0; i<buf.size(); i++) {
    const AnalysisData& data = buf[i];
    cout << NNPos::locToPos(data.move,board.x_size,nnXLen,nnYLen) << " ";
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
      int pos = NNPos::xyToPos(x,y,nnXLen);
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

static void initializeDemoGame(Board& board, BoardHistory& hist, Player& pla, Rand& rand, AsyncBot* bot, Logger& logger) {
  static const int numSizes = 9;
  int sizes[numSizes] = {19,13,9,15,11,10,12,14,16};
  int sizeFreqs[numSizes] = {240,18,12,6,2,1,1,1,1};

  const int size = sizes[rand.nextUInt(sizeFreqs,numSizes)];

  board = Board(size,size);
  pla = P_BLACK;
  hist.clear(board,pla,Rules::getTrompTaylorish(),0);
  bot->setPosition(pla,board,hist);

  if(size == 19) {
    //Many games use a special opening
    if(rand.nextBool(0.6)) {
      auto g = [size](int x, int y) { return Location::getLoc(x,y,size); };
      const Move nb = Move(Board::NULL_LOC, P_BLACK);
      const Move nw = Move(Board::NULL_LOC, P_WHITE);
      Player b = P_BLACK;
      Player w = P_WHITE;
      vector<vector<Move>> specialOpenings = {
        //Sanrensei
        { Move(g(3,3), b), nw, Move(g(15,3), b), nw, Move(g(9,3), b) },
        //Low Chinese
        { Move(g(3,3), b), nw, Move(g(16,3), b), nw, Move(g(10,2), b) },
        //Low Chinese
        { Move(g(3,3), b), nw, Move(g(16,3), b), nw, Move(g(10,2), b) },
        //High chinese
        { Move(g(3,3), b), nw, Move(g(16,3), b), nw, Move(g(10,3), b) },
        //Low small chinese
        { Move(g(3,3), b), nw, Move(g(16,3), b), nw, Move(g(11,2), b) },
        //Kobayashi
        { Move(g(3,3), b), Move(g(15,15), w), Move(g(16,3), b), nw, Move(g(16,13), b), Move(g(13,16), w), Move(g(15,9), b) },
        //Kobayashi
        { Move(g(3,3), b), Move(g(15,15), w), Move(g(16,3), b), nw, Move(g(16,13), b), Move(g(13,16), w), Move(g(15,9), b) },
        //Mini chinese
        { Move(g(3,3), b), Move(g(15,15), w), Move(g(15,2), b), nw, Move(g(16,13), b), Move(g(13,16), w), Move(g(16,8), b) },
        //Mini chinese
        { Move(g(3,3), b), Move(g(15,15), w), Move(g(15,2), b), nw, Move(g(16,13), b), Move(g(13,16), w), Move(g(16,8), b) },
        //Micro chinese
        { Move(g(3,3), b), Move(g(15,15), w), Move(g(15,2), b), nw, Move(g(16,13), b), Move(g(13,16), w), Move(g(16,7), b) },
        //Micro chinese with variable other corner
        { Move(g(15,2), b), Move(g(15,15), w), nb, nw, Move(g(16,13), b), Move(g(13,16), w), Move(g(16,7), b) },
        //Boring star points
        { Move(g(15,3), b), Move(g(15,15), w), nb, nw, Move(g(16,13), b), Move(g(13,16), w), Move(g(15,9), b) },
        //High 3-4 counter approaches
        { Move(g(3,3), b), Move(g(15,16), w), Move(g(16,3), b), nw, Move(g(15,14), b), Move(g(14,3), w) },
        //Double 3-3
        { Move(g(2,2), b), nw, Move(g(16,2), b) },
        //Low enclosure
        { Move(g(2,3), b), nw, Move(g(4,2), b) },
        //High enclosure
        { Move(g(2,3), b), nw, Move(g(4,3), b) },
        //5-5 point
        { Move(g(4,4), b) },
        //5-3 point
        { Move(g(2,4), b) },
        //5-4 point
        { Move(g(3,4), b) },
        //3-3 point
        { Move(g(2,2), b) },
        //3-4 point far approach
        { Move(g(3,2), b), Move(g(2,5), w) },
        //Tengen
        { Move(g(9,9), b) },
        //2-2 point
        { Move(g(1,1), b) },
        //Shusaku fuseki
        { Move(g(16,15), b), Move(g(3,16), w), Move(g(15,2), b), Move(g(14,16), w), nb, Move(g(16,4), w), Move(g(15,14), b) },
        //Miyamoto fuseki
        { Move(g(16,13), b), Move(g(3,15), w), Move(g(13,2), b), nw, Move(g(9,16), b) },
        //4-4 1-space low pincer - shared side
        { Move(g(15,15), b), Move(g(3,15), w), nb, nw, Move(g(5,16), b), Move(g(7,16), w) },
        //4-4 2-space high pincer - shared side
        { Move(g(15,15), b), Move(g(3,15), w), nb, nw, Move(g(5,16), b), Move(g(8,15), w) },
        //4-4 1-space low pincer - opponent side
        { Move(g(15,15), b), Move(g(3,15), w), nb, nw, Move(g(2,13), b), Move(g(2,11), w) },
        //4-4 2-space high pincer - opponent side
        { Move(g(15,15), b), Move(g(3,15), w), nb, nw, Move(g(2,13), b), Move(g(3,10), w) },
        //3-4 1-space low approach - shusaku kosumi and long extend
        { Move(g(15,15), b), Move(g(3,16), w), nb, nw, Move(g(2,14), b), Move(g(4,15), w), Move(g(2,10), b) },
        //3-4 1-space low approach low pincer - opponent side
        { Move(g(15,15), b), Move(g(3,16), w), nb, nw, Move(g(2,14), b), Move(g(2,12), w) },
        //3-4 2-space low approach high pincer - opponent side
        { Move(g(15,15), b), Move(g(3,16), w), nb, nw, Move(g(2,14), b), Move(g(3,11), w) },
        //3-4 1-space high approach - opponent side
        { Move(g(15,15), b), Move(g(3,16), w), nb, nw, Move(g(3,14), b) },
        //3-4 1-space high approach low pincer - opponent side
        { Move(g(15,15), b), Move(g(3,16), w), nb, nw, Move(g(3,14), b), Move(g(2,12), w) },
        //3-4 2-space high approach high pincer - opponent side
        { Move(g(15,15), b), Move(g(3,16), w), nb, nw, Move(g(3,14), b), Move(g(3,11), w) },
        //Orthodox
        { Move(g(3,3), b), nw, Move(g(15,2), b), nw, Move(g(16,4), b), Move(g(9,2), w) },
        //Manchurian
        { Move(g(4,3), b), nw, Move(g(16,3), b), nw, Move(g(10,3), b) },
        //Upper Manchurian
        { Move(g(4,4), b), nw, Move(g(16,4), b), nw, Move(g(10,4), b) },
        //Great wall
        { Move(g(9,9), b), nw, Move(g(9,15), b), nw, Move(g(9,3), b), nw, Move(g(8,12), b), nw, Move(g(10,6), b) },
        //Small wall
        { Move(g(9,8), b), nw, Move(g(8,11), b), nw, Move(g(10,5), b) },
        //High approaches
        { Move(g(3,2), b), Move(g(3,4), w), Move(g(16,3), b), Move(g(14,3), w), Move(g(15,16), b), Move(g(15,14), w) },
        //Black hole
        { Move(g(12,14), b), nw, Move(g(14,6), b), nw, Move(g(4,12), b), nw, Move(g(6,4), b) },
        //Crosscut
        { Move(g(9,9), b), Move(g(9,10), w), Move(g(10,10), b), Move(g(10,9), w) },
        //One-point jump center
        { Move(g(9,8), b), nw, Move(g(9,10), b) },
      };

      vector<Move> chosenOpening = specialOpenings[rand.nextUInt((int)specialOpenings.size())];
      vector<vector<Move>> chosenOpenings;

      for(int j = 0; j<8; j++) {
        vector<Move> symmetric;
        for(int k = 0; k<chosenOpening.size(); k++) {
          Loc loc = chosenOpening[k].loc;
          Player movePla = chosenOpening[k].pla;
          if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
            symmetric.push_back(Move(loc,movePla));
          else {
            int x = Location::getX(loc,size);
            int y = Location::getY(loc,size);
            if(j & 1) x = size-1-x;
            if(j & 2) y = size-1-y;
            if(j & 4) std::swap(x,y);
            symmetric.push_back(Move(Location::getLoc(x,y,size),movePla));
          }
        }
        chosenOpenings.push_back(symmetric);
      }
      for(int j = (int)chosenOpenings.size()-1; j>=1; j--) {
        int r = rand.nextUInt(j+1);
        vector<Move> tmp = chosenOpenings[j];
        chosenOpenings[j] = chosenOpenings[r];
        chosenOpenings[r] = tmp;
      }

      vector<Move> movesPlayed;
      vector<Move> freeMovesPlayed;
      vector<Move> specifiedMovesPlayed;
      while(true) {
        auto withinRadius1 = [size](Loc l0, Loc l1) {
          if(l0 == Board::NULL_LOC || l1 == Board::NULL_LOC || l0 == Board::PASS_LOC || l1 == Board::PASS_LOC)
            return false;
          int x0 = Location::getX(l0,size);
          int y0 = Location::getY(l0,size);
          int x1 = Location::getX(l1,size);
          int y1 = Location::getY(l1,size);
          return std::abs(x0-x1) <= 1 && std::abs(y0-y1) <= 1;
        };
        auto symmetryIsGood = [&movesPlayed,&specifiedMovesPlayed,&freeMovesPlayed,&withinRadius1](const vector<Move>& moves) {
          assert(movesPlayed.size() <= moves.size());
          //Make sure the symmetry matches up to the desired point,
          //and that free moves are not within radius 1 of any specified move
          for(int j = 0; j<movesPlayed.size(); j++) {
            if(moves[j].loc == Board::NULL_LOC) {
              Loc actualLoc = movesPlayed[j].loc;
              for(int k = 0; k<specifiedMovesPlayed.size(); k++) {
                if(withinRadius1(specifiedMovesPlayed[k].loc,actualLoc))
                  return false;
              }
            }
            else if(movesPlayed[j].loc != moves[j].loc)
              return false;
          }

          //Make sure the next move will also not be within radius 1 of any free move.
          if(movesPlayed.size() < moves.size()) {
            Loc nextLoc = moves[movesPlayed.size()].loc;
            for(int k = 0; k<freeMovesPlayed.size(); k++) {
              if(withinRadius1(freeMovesPlayed[k].loc,nextLoc))
                return false;
            }
          }

          return true;
        };

        //Take the first good symmetry
        vector<Move> goodSymmetry;
        for(int i = 0; i<chosenOpenings.size(); i++) {
          if(symmetryIsGood(chosenOpenings[i])) {
            goodSymmetry = chosenOpenings[i];
            break;
          }
        }

        //If we have no further moves on that symmetry, we're done
        if(movesPlayed.size() >= goodSymmetry.size())
          break;

        Move nextMove = goodSymmetry[movesPlayed.size()];
        bool wasSpecified = true;

        if(nextMove.loc == Board::NULL_LOC) {
          wasSpecified = false;
          Search* search = bot->getSearchStopAndWait();
          NNResultBuf buf;
          MiscNNInputParams nnInputParams;
          nnInputParams.drawEquivalentWinsForWhite = search->searchParams.drawEquivalentWinsForWhite;
          search->nnEvaluator->evaluate(board,hist,pla,nnInputParams,buf,false,false);
          std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);

          double temperature = 0.8;
          bool allowPass = false;
          Loc banMove = Board::NULL_LOC;
          Loc loc = PlayUtils::chooseRandomPolicyMove(nnOutput.get(), board, hist, pla, rand, temperature, allowPass, banMove);
          nextMove.loc = loc;
        }

        //Make sure the next move is legal
        if(!hist.isLegal(board,nextMove.loc,nextMove.pla))
          break;

        //Make the move!
        hist.makeBoardMoveAssumeLegal(board,nextMove.loc,nextMove.pla,NULL);
        pla = getOpp(pla);

        hist.clear(board,pla,hist.rules,0);
        bot->setPosition(pla,board,hist);

        movesPlayed.push_back(nextMove);
        if(wasSpecified)
          specifiedMovesPlayed.push_back(nextMove);
        else
          freeMovesPlayed.push_back(nextMove);

        bot->clearSearch();
        writeLine(bot->getSearch(),hist,vector<double>(),vector<double>(),vector<double>());
        std::this_thread::sleep_for(std::chrono::duration<double>(1.0));

      } //Close while(true)

      int numVisits = 20;
      PlayUtils::adjustKomiToEven(bot->getSearchStopAndWait(),NULL,board,hist,pla,numVisits,logger,OtherGameProperties(),rand);
      double komi = hist.rules.komi + 0.3 * rand.nextGaussian();
      komi = 0.5 * round(2.0 * komi);
      hist.setKomi((float)komi);
      bot->setPosition(pla,board,hist);
    }
  }

  bot->clearSearch();
  writeLine(bot->getSearch(),hist,vector<double>(),vector<double>(),vector<double>());
  std::this_thread::sleep_for(std::chrono::duration<double>(2.0));

}


int MainCmds::demoplay(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string logFile;
  string modelFile;
  try {
    KataGoCommandLine cmd("Self-play demo dumping status to stdout");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",false,string(),"FILE");
    cmd.add(logFileArg);
    cmd.parse(argc,argv);

    modelFile = cmd.getModelFile();
    logFile = logFileArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger;
  logger.addFile(logFile);

  logger.write("Engine starting...");

  string searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int expectedConcurrentEvals = params.numThreads;
    int defaultMaxBatchSize = -1;
    string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,
      Setup::SETUP_FOR_OTHER
    );
  }
  logger.write("Loaded neural net");

  const bool allowResignation = cfg.contains("allowResignation") ? cfg.getBool("allowResignation") : false;
  const double resignThreshold = cfg.contains("allowResignation") ? cfg.getDouble("resignThreshold",-1.0,0.0) : -1.0; //Threshold on [-1,1], regardless of winLossUtilityFactor
  const double resignScoreThreshold = cfg.contains("allowResignation") ? cfg.getDouble("resignScoreThreshold",-10000.0,0.0) : -10000.0;

  const double searchFactorWhenWinning = cfg.contains("searchFactorWhenWinning") ? cfg.getDouble("searchFactorWhenWinning",0.01,1.0) : 1.0;
  const double searchFactorWhenWinningThreshold = cfg.contains("searchFactorWhenWinningThreshold") ? cfg.getDouble("searchFactorWhenWinningThreshold",0.0,1.0) : 1.0;

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  AsyncBot* bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);
  bot->setAlwaysIncludeOwnerMap(true);
  Rand gameRand;

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting demo");

  //Game loop
  while(true) {

    Player pla = P_BLACK;
    Board baseBoard;
    BoardHistory baseHist(baseBoard,pla,Rules::getTrompTaylorish(),0);
    TimeControls tc;

    initializeDemoGame(baseBoard, baseHist, pla, gameRand, bot, logger);

    bot->setPosition(pla,baseBoard,baseHist);

    vector<double> recentWinLossValues;
    vector<double> recentScores;
    vector<double> recentScoreStdevs;

    double callbackPeriod = 0.05;

    std::function<void(const Search*)> callback = [&baseHist,&recentWinLossValues,&recentScores,&recentScoreStdevs](const Search* search) {
      writeLine(search,baseHist,recentWinLossValues,recentScores,recentScoreStdevs);
    };

    //Move loop
    int maxMovesPerGame = 1600;
    for(int i = 0; i<maxMovesPerGame; i++) {
      baseHist.endGameIfAllPassAlive(baseBoard);
      if(baseHist.isGameFinished)
        break;

      callback(bot->getSearch());

      double searchFactor =
        //Speed up when either player is winning confidently, not just the winner only
        std::min(
          PlayUtils::getSearchFactor(searchFactorWhenWinningThreshold,searchFactorWhenWinning,params,recentWinLossValues,P_BLACK),
          PlayUtils::getSearchFactor(searchFactorWhenWinningThreshold,searchFactorWhenWinning,params,recentWinLossValues,P_WHITE)
        );
      Loc moveLoc = bot->genMoveSynchronousAnalyze(pla,tc,searchFactor,callbackPeriod,callback);

      bool isLegal = bot->isLegalStrict(moveLoc,pla);
      if(moveLoc == Board::NULL_LOC || !isLegal) {
        ostringstream sout;
        sout << "genmove null location or illegal move!?!" << "\n";
        sout << bot->getRootBoard() << "\n";
        sout << "Pla: " << PlayerIO::playerToString(pla) << "\n";
        sout << "MoveLoc: " << Location::toString(moveLoc,bot->getRootBoard()) << "\n";
        logger.write(sout.str());
        cerr << sout.str() << endl;
        throw StringError("illegal move");
      }

      double winLossValue;
      double expectedScore;
      double expectedScoreStdev;
      {
        ReportedSearchValues values = bot->getSearch()->getRootValuesRequireSuccess();
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
        break;
      }
      else {
        //And make the move on our copy of the board
        assert(baseHist.isLegal(baseBoard,moveLoc,pla));
        baseHist.makeBoardMoveAssumeLegal(baseBoard,moveLoc,pla,NULL);

        //If the game is over, skip making the move on the bot, to preserve
        //the last known value of the search tree for display purposes
        //Just immediately terminate the game loop
        if(baseHist.isGameFinished)
          break;

        bool suc = bot->makeMove(moveLoc,pla);
        assert(suc);
        (void)suc; //Avoid warning when asserts are off

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
  ScoreValue::freeTables();

  logger.write("All cleaned up, quitting");
  return 0;

}

int MainCmds::printclockinfo(int argc, const char* const* argv) {
  (void)argc;
  (void)argv;
#ifdef OS_IS_WINDOWS
  cout << "Does nothing on windows, disabled" << endl;
#endif
#ifdef OS_IS_UNIX_OR_APPLE
  cout << "Tick unit in seconds: " << std::chrono::steady_clock::period::num << " / " <<  std::chrono::steady_clock::period::den << endl;
  cout << "Ticks since epoch: " << std::chrono::steady_clock::now().time_since_epoch().count() << endl;
#endif
  return 0;
}


int MainCmds::samplesgfs(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  vector<string> sgfDirs;
  string outDir;
  vector<string> excludeHashesFiles;
  double sampleProb;
  double turnWeightLambda;
  int64_t maxDepth;
  int64_t maxNodeCount;
  int64_t maxBranchCount;
  try {
    KataGoCommandLine cmd("Search for suprising good moves in sgfs");

    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",true,"DIR");
    TCLAP::ValueArg<string> outDirArg("","outdir","Directory to write results",true,string(),"DIR");
    TCLAP::MultiArg<string> excludeHashesArg("","exclude-hashes","Specify a list of hashes to filter out, one per line in a txt file",false,"FILEOF(HASH,HASH)");
    TCLAP::ValueArg<double> sampleProbArg("","sample-prob","Probability to sample each position",true,0.0,"PROB");
    TCLAP::ValueArg<double> turnWeightLambdaArg("","turn-weight-lambda","Adjust weight for writing down each position",true,0.0,"LAMBDA");
    TCLAP::ValueArg<string> maxDepthArg("","max-depth","Max depth allowed for sgf",false,"100000000","INT");
    TCLAP::ValueArg<string> maxNodeCountArg("","max-node-count","Max node count allowed for sgf",false,"100000000","INT");
    TCLAP::ValueArg<string> maxBranchCountArg("","max-branch-count","Max branch count allowed for sgf",false,"100000000","INT");
    cmd.add(sgfDirArg);
    cmd.add(outDirArg);
    cmd.add(excludeHashesArg);
    cmd.add(sampleProbArg);
    cmd.add(turnWeightLambdaArg);
    cmd.add(maxDepthArg);
    cmd.add(maxNodeCountArg);
    cmd.add(maxBranchCountArg);
    cmd.parse(argc,argv);
    sgfDirs = sgfDirArg.getValue();
    outDir = outDirArg.getValue();
    excludeHashesFiles = excludeHashesArg.getValue();
    sampleProb = sampleProbArg.getValue();
    turnWeightLambda = turnWeightLambdaArg.getValue();
    maxDepth = Global::stringToInt64(maxDepthArg.getValue());
    maxNodeCount = Global::stringToInt64(maxNodeCountArg.getValue());
    maxBranchCount = Global::stringToInt64(maxBranchCountArg.getValue());
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger;
  logger.setLogToStdout(true);

  const string sgfSuffix = ".sgf";
  auto sgfFilter = [&sgfSuffix](const string& name) {
    return Global::isSuffix(name,sgfSuffix);
  };
  vector<string> sgfFiles;
  for(int i = 0; i<sgfDirs.size(); i++)
    Global::collectFiles(sgfDirs[i], sgfFilter, sgfFiles);
  logger.write("Found " + Global::int64ToString((int64_t)sgfFiles.size()) + " sgf files!");

  set<Hash128> excludeHashes = Sgf::readExcludes(excludeHashesFiles);
  logger.write("Loaded " + Global::uint64ToString(excludeHashes.size()) + " excludes");

  MakeDir::make(outDir);

  // ---------------------------------------------------------------------------------------------------
  ThreadSafeQueue<string*> toWriteQueue;
  auto writeLoop = [&toWriteQueue,&outDir]() {
    int fileCounter = 0;
    int numWrittenThisFile = 0;
    ofstream* out = NULL;
    while(true) {
      string* message;
      bool suc = toWriteQueue.waitPop(message);
      if(!suc)
        break;

      if(out == NULL || numWrittenThisFile > 100000) {
        if(out != NULL) {
          out->close();
          delete out;
        }
        out = new ofstream(outDir + "/" + Global::intToString(fileCounter) + ".startposes.txt");
        fileCounter += 1;
        numWrittenThisFile = 0;
      }
      (*out) << *message << endl;
      numWrittenThisFile += 1;
      delete message;
    }

    if(out != NULL) {
      out->close();
      delete out;
    }
  };

  // ---------------------------------------------------------------------------------------------------

  //Begin writing
  std::thread writeLoopThread(writeLoop);

  // ---------------------------------------------------------------------------------------------------

  int64_t numKept = 0;
  std::set<Hash128> uniqueHashes;
  std::function<void(Sgf::PositionSample&, const BoardHistory&, const string&)> posHandler =
    [sampleProb,&toWriteQueue,turnWeightLambda,&numKept,&seedRand](Sgf::PositionSample& posSample, const BoardHistory& hist, const string& comments) {
    (void)hist;
    (void)comments;
    if(seedRand.nextBool(sampleProb)) {
      Sgf::PositionSample posSampleToWrite = posSample;
      int64_t startTurn = posSampleToWrite.initialTurnNumber + (int64_t)posSampleToWrite.moves.size();
      posSampleToWrite.weight = exp(-startTurn * turnWeightLambda) * posSampleToWrite.weight;
      toWriteQueue.waitPush(new string(Sgf::PositionSample::toJsonLine(posSampleToWrite)));
      numKept += 1;
    }
  };
  int64_t numExcluded = 0;
  for(size_t i = 0; i<sgfFiles.size(); i++) {
    Sgf* sgf = NULL;
    try {
      sgf = Sgf::loadFile(sgfFiles[i]);
      if(contains(excludeHashes,sgf->hash))
        numExcluded += 1;
      else {
        int64_t depth = sgf->depth();
        int64_t nodeCount = sgf->nodeCount();
        int64_t branchCount = sgf->branchCount();
        if(depth > maxDepth || nodeCount > maxNodeCount || branchCount > maxBranchCount) {
          logger.write(
            "Skipping due to violating limits depth " + Global::int64ToString(depth) +
            " nodes " + Global::int64ToString(nodeCount) +
            " branches " + Global::int64ToString(branchCount) +
            " " + sgfFiles[i]
          );
        }
        else {
          bool hashComments = false;
          sgf->iterAllUniquePositions(uniqueHashes, hashComments, posHandler);
        }
      }
    }
    catch(const StringError& e) {
      logger.write("Invalid SGF " + sgfFiles[i] + ": " + e.what());
    }
    if(sgf != NULL)
      delete sgf;
  }
  logger.write("Kept " + Global::int64ToString(numKept) + " start positions");
  logger.write("Excluded " + Global::int64ToString(numExcluded) + "/" + Global::uint64ToString(sgfFiles.size()) + " sgf files");


  // ---------------------------------------------------------------------------------------------------

  toWriteQueue.setReadOnly();
  writeLoopThread.join();

  logger.write("All done");

  ScoreValue::freeTables();
  return 0;
}

static bool maybeGetValuesAfterMove(
  Search* search, Logger& logger, Loc moveLoc,
  Player nextPla, const Board& board, const BoardHistory& hist,
  double quickSearchFactor,
  ReportedSearchValues& values
) {
  Board newBoard = board;
  BoardHistory newHist = hist;
  Player newNextPla = nextPla;

  if(moveLoc != Board::NULL_LOC) {
    if(!hist.isLegal(newBoard,moveLoc,newNextPla))
      return false;
    newHist.makeBoardMoveAssumeLegal(newBoard,moveLoc,newNextPla,NULL);
    newNextPla = getOpp(newNextPla);
  }

  search->setPosition(newNextPla,newBoard,newHist);

  if(quickSearchFactor != 1.0) {
    SearchParams oldSearchParams = search->searchParams;
    SearchParams newSearchParams = oldSearchParams;
    newSearchParams.maxVisits = 1 + (int64_t)(oldSearchParams.maxVisits * quickSearchFactor);
    newSearchParams.maxPlayouts = 1 + (int64_t)(oldSearchParams.maxPlayouts * quickSearchFactor);
    search->setParamsNoClearing(newSearchParams);
    search->runWholeSearch(newNextPla,logger,shouldStop);
    search->setParamsNoClearing(oldSearchParams);
  }
  else {
    search->runWholeSearch(newNextPla,logger,shouldStop);
  }

  if(shouldStop.load(std::memory_order_acquire))
    return false;
  values = search->getRootValuesRequireSuccess();
  return true;
}



//We want surprising moves that turned out not poorly
//The more surprising, the more we will weight it
static double surpriseWeight(double policyProb, Rand& rand, bool markedAsHintPos) {
  if(policyProb < 0)
    return 0;
  double weight = 0.12 / (policyProb + 0.02) - 0.5;
  if(markedAsHintPos && weight < 0.5)
    weight = 0.5;

  if(weight <= 0)
    return 0;
  if(weight < 0.2) {
    if(rand.nextDouble() * 0.2 >= weight)
      return 0;
    return 0.2;
  }
  return weight;
}

struct PosQueueEntry {
  BoardHistory* hist;
  int initialTurnNumber;
  bool markedAsHintPos;
};

int MainCmds::dataminesgfs(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> sgfDirs;
  string outDir;
  int numProcessThreads;
  vector<string> excludeHashesFiles;
  bool gameMode;
  bool treeMode;
  bool autoKomi;
  int sgfSplitCount;
  int sgfSplitIdx;
  double turnWeightLambda;
  try {
    KataGoCommandLine cmd("Search for suprising good moves in sgfs");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",true,"DIR");
    TCLAP::ValueArg<string> outDirArg("","outdir","Directory to write results",true,string(),"DIR");
    TCLAP::ValueArg<int> numProcessThreadsArg("","threads","Number of threads",true,1,"THREADS");
    TCLAP::MultiArg<string> excludeHashesArg("","exclude-hashes","Specify a list of hashes to filter out, one per line in a txt file",false,"FILEOF(HASH,HASH)");
    TCLAP::SwitchArg gameModeArg("","game-mode","Game mode");
    TCLAP::SwitchArg treeModeArg("","tree-mode","Tree mode");
    TCLAP::SwitchArg autoKomiArg("","auto-komi","Auto komi");
    TCLAP::ValueArg<int> sgfSplitCountArg("","sgf-split-count","Number of splits",false,1,"N");
    TCLAP::ValueArg<int> sgfSplitIdxArg("","sgf-split-idx","Which split",false,0,"IDX");
    TCLAP::ValueArg<double> turnWeightLambdaArg("","turn-weight-lambda","Adjust weight for writing down each position",false,0.0,"LAMBDA");
    cmd.add(sgfDirArg);
    cmd.add(outDirArg);
    cmd.add(numProcessThreadsArg);
    cmd.add(excludeHashesArg);
    cmd.add(gameModeArg);
    cmd.add(treeModeArg);
    cmd.add(autoKomiArg);
    cmd.add(sgfSplitCountArg);
    cmd.add(sgfSplitIdxArg);
    cmd.add(turnWeightLambdaArg);
    cmd.parse(argc,argv);
    nnModelFile = cmd.getModelFile();
    sgfDirs = sgfDirArg.getValue();
    outDir = outDirArg.getValue();
    numProcessThreads = numProcessThreadsArg.getValue();
    excludeHashesFiles = excludeHashesArg.getValue();
    gameMode = gameModeArg.getValue();
    treeMode = treeModeArg.getValue();
    autoKomi = autoKomiArg.getValue();
    sgfSplitCount = sgfSplitCountArg.getValue();
    sgfSplitIdx = sgfSplitIdxArg.getValue();
    turnWeightLambda = turnWeightLambdaArg.getValue();

    if(gameMode == treeMode)
      throw StringError("Must specify either -game-mode or -tree-mode");

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger;
  logger.setLogToStdout(true);

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_ANALYSIS);
  //Ignore temperature, noise
  params.chosenMoveTemperature = 0;
  params.chosenMoveTemperatureEarly = 0;
  params.rootNoiseEnabled = false;
  params.rootDesiredPerChildVisitsCoeff = 0;
  params.rootPolicyTemperature = 1.0;
  params.rootPolicyTemperatureEarly = 1.0;
  params.rootFpuReductionMax = params.fpuReductionMax * 0.5;

  //Disable dynamic utility so that utilities are always comparable
  params.staticScoreUtilityFactor += params.dynamicScoreUtilityFactor;
  params.dynamicScoreUtilityFactor = 0;

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int expectedConcurrentEvals = params.numThreads;
    int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  GameInitializer* gameInit = new GameInitializer(cfg,logger);
  cfg.warnUnusedKeys(cerr,&logger);

  const string sgfSuffix = ".sgf";
  auto sgfFilter = [&sgfSuffix](const string& name) {
    return Global::isSuffix(name,sgfSuffix);
  };
  vector<string> sgfFiles;
  for(int i = 0; i<sgfDirs.size(); i++)
    Global::collectFiles(sgfDirs[i], sgfFilter, sgfFiles);
  logger.write("Found " + Global::int64ToString((int64_t)sgfFiles.size()) + " sgf files!");

  vector<size_t> permutation(sgfFiles.size());
  for(size_t i = 0; i<sgfFiles.size(); i++)
    permutation[i] = i;
  for(size_t i = 1; i<sgfFiles.size(); i++) {
    size_t r = (size_t)seedRand.nextUInt64(i+1);
    std::swap(permutation[i],permutation[r]);
  }

  set<Hash128> excludeHashes = Sgf::readExcludes(excludeHashesFiles);
  logger.write("Loaded " + Global::uint64ToString(excludeHashes.size()) + " excludes");

  MakeDir::make(outDir);

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // ---------------------------------------------------------------------------------------------------
  ThreadSafeQueue<string*> toWriteQueue;
  auto writeLoop = [&toWriteQueue,&outDir,&sgfSplitCount,&sgfSplitIdx]() {
    int fileCounter = 0;
    int numWrittenThisFile = 0;
    ofstream* out = NULL;
    while(true) {
      string* message;
      bool suc = toWriteQueue.waitPop(message);
      if(!suc)
        break;

      if(out == NULL || numWrittenThisFile > 100000) {
        if(out != NULL) {
          out->close();
          delete out;
        }
        if(sgfSplitCount > 1)
          out = new ofstream(outDir + "/" + Global::intToString(fileCounter) + "." + Global::intToString(sgfSplitIdx) + ".hintposes.txt");
        else
          out = new ofstream(outDir + "/" + Global::intToString(fileCounter) + ".hintposes.txt");
        fileCounter += 1;
        numWrittenThisFile = 0;
      }
      (*out) << *message << endl;
      numWrittenThisFile += 1;
      delete message;
    }

    if(out != NULL) {
      out->close();
      delete out;
    }
  };

  // ---------------------------------------------------------------------------------------------------

  auto expensiveEvaluateMove = [&toWriteQueue,&logger,&turnWeightLambda](
    Search* search, Loc missedLoc,
    Player nextPla, const Board& board, const BoardHistory& hist,
    const Sgf::PositionSample& sample, bool markedAsHintPos
  ) {
    if(shouldStop.load(std::memory_order_acquire))
      return;

    {
      int numStonesOnBoard = 0;
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          if(board.colors[loc] != C_EMPTY)
            numStonesOnBoard += 1;
        }
      }
      if(numStonesOnBoard < 6)
        return;
    }

    ReportedSearchValues veryQuickValues;
    {
      bool suc = maybeGetValuesAfterMove(search,logger,Board::NULL_LOC,nextPla,board,hist,1.0/25.0,veryQuickValues);
      if(!suc)
        return;
    }
    Loc veryQuickMoveLoc = search->getChosenMoveLoc();

    ReportedSearchValues quickValues;
    {
      bool suc = maybeGetValuesAfterMove(search,logger,Board::NULL_LOC,nextPla,board,hist,1.0/5.0,quickValues);
      if(!suc)
        return;
    }
    Loc quickMoveLoc = search->getChosenMoveLoc();

    ReportedSearchValues baseValues;
    {
      bool suc = maybeGetValuesAfterMove(search,logger,Board::NULL_LOC,nextPla,board,hist,1.0,baseValues);
      if(!suc)
        return;
    }
    Loc moveLoc = search->getChosenMoveLoc();

    // const Player perspective = P_WHITE;
    // {
    //   ostringstream preOut;
    //   Board::printBoard(preOut, search->getRootBoard(), moveLoc, &(search->getRootHist().moveHistory));
    //   search->printTree(preOut, search->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10),perspective);
    //   cout << preOut.str() << endl;
    //   cout << Location::toString(missedLoc,board) << endl;
    // }

    Sgf::PositionSample sampleToWrite = sample;
    sampleToWrite.weight += abs(baseValues.utility - quickValues.utility);
    sampleToWrite.weight += abs(baseValues.utility - veryQuickValues.utility);

    //Bot DOES see the move?
    if(moveLoc == missedLoc) {
      if(quickMoveLoc == moveLoc)
        sampleToWrite.weight = sampleToWrite.weight * 0.75 - 0.1;
      if(veryQuickMoveLoc == moveLoc)
        sampleToWrite.weight = sampleToWrite.weight * 0.75 - 0.1;

      sampleToWrite.weight *= exp(-sampleToWrite.initialTurnNumber * turnWeightLambda);
      if(sampleToWrite.weight > 0.1) {
        //Still good to learn from given that policy was really low
        toWriteQueue.waitPush(new string(Sgf::PositionSample::toJsonLine(sampleToWrite)));
      }
    }

    //Bot doesn't see the move?
    else if(moveLoc != missedLoc) {

      //If marked as a hint pos, always trust that it should be better and add it.
      bool shouldWriteMove = markedAsHintPos;

      if(!shouldWriteMove) {
        ReportedSearchValues moveValues;
        if(!maybeGetValuesAfterMove(search,logger,moveLoc,nextPla,board,hist,1.0,moveValues))
          return;
        // ostringstream out0;
        // out0 << "BOT MOVE " << Location::toString(moveLoc,board) << endl;
        // search->printTree(out0, search->rootNode, PrintTreeOptions().maxDepth(0),perspective);
        // cout << out0.str() << endl;

        ReportedSearchValues missedValues;
        if(!maybeGetValuesAfterMove(search,logger,missedLoc,nextPla,board,hist,1.0,missedValues))
          return;
        // ostringstream out0;
        // out0 << "SGF MOVE " << Location::toString(missedLoc,board) << endl;
        // search->printTree(out0, search->rootNode, PrintTreeOptions().maxDepth(0),perspective);
        // cout << out0.str() << endl;

        //If the move is this minimum amount better, then record this position as a hint
        //Otherwise the bot actually thinks the move isn't better, so we reject it as an invalid hint.
        const double utilityThreshold = 0.01;
        ReportedSearchValues postValues = search->getRootValuesRequireSuccess();
        if((nextPla == P_WHITE && missedValues.utility > moveValues.utility + utilityThreshold) ||
           (nextPla == P_BLACK && missedValues.utility < moveValues.utility - utilityThreshold)) {
          shouldWriteMove = true;
        }
      }

      if(shouldWriteMove) {
        //Moves that the bot didn't see get written out more
        sampleToWrite.weight = sampleToWrite.weight * 1.5 + 1.0;
        sampleToWrite.weight *= exp(-sampleToWrite.initialTurnNumber * turnWeightLambda);
        if(sampleToWrite.weight > 0.1) {
          toWriteQueue.waitPush(new string(Sgf::PositionSample::toJsonLine(sampleToWrite)));
        }
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------
  //SGF MODE

  auto processSgfGame = [&logger,&excludeHashes,&gameInit,&nnEval,&expensiveEvaluateMove,autoKomi](
    Search* search, Rand& rand, const string& fileName, CompactSgf* sgf
  ) {
    if(contains(excludeHashes,sgf->hash))
      return;

    //Don't use the SGF rules - randomize them for a bit more entropy
    Rules rules = gameInit->createRules();

    Board board;
    Player nextPla;
    BoardHistory hist;
    sgf->setupInitialBoardAndHist(rules, board, nextPla, hist);
    if(!gameInit->isAllowedBSize(board.x_size,board.y_size))
      return;
    if(board.x_size != 19 || board.y_size != 19)
      return;

    const bool preventEncore = true;
    const vector<Move>& sgfMoves = sgf->moves;

    vector<Board> boards;
    vector<BoardHistory> hists;
    vector<Player> nextPlas;
    vector<shared_ptr<NNOutput>> nnOutputs;
    vector<double> winLossValues;
    vector<double> scoreLeads;

    vector<Move> moves;
    vector<double> policyPriors;

    bool quitEarly = false;
    for(int m = 0; m<sgfMoves.size()+1; m++) {
      MiscNNInputParams nnInputParams;
      NNResultBuf buf;
      bool skipCache = true; //Always ignore cache so that we get more entropy on repeated board positions due to symmetries
      bool includeOwnerMap = false;
      nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

      boards.push_back(board);
      hists.push_back(hist);
      nextPlas.push_back(nextPla);
      nnOutputs.push_back(std::move(buf.result));

      shared_ptr<NNOutput>& nnOutput = nnOutputs[nnOutputs.size()-1];
      winLossValues.push_back(nnOutput->whiteWinProb - nnOutput->whiteLossProb);
      scoreLeads.push_back(nnOutput->whiteLead);

      if(m < sgfMoves.size()) {
        moves.push_back(sgfMoves[m]);
        int pos = NNPos::locToPos(sgfMoves[m].loc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
        policyPriors.push_back(nnOutput->policyProbs[pos]);
      }

      if(m >= sgfMoves.size())
        break;

      //Quit out if according to our rules, we already finished the game, or we're somehow in a cleanup phase
      if(hist.isGameFinished || hist.encorePhase > 0)
        break;

      //Quit out if consecutive moves by the same player, to keep the history clean and "normal"
      if(sgfMoves[m].pla != nextPla && m > 0) {
        logger.write("Ending SGF " + fileName + " early due to non-alternating players on turn " + Global::intToString(m));
        quitEarly = true;
        break;
      }

      bool suc = hist.isLegal(board,sgfMoves[m].loc,sgfMoves[m].pla);
      if(!suc) {
        //Only log on errors that aren't simply due to ko rules, but quit out regardless
        suc = hist.makeBoardMoveTolerant(board,sgfMoves[m].loc,sgfMoves[m].pla,preventEncore);
        if(!suc)
          logger.write("Illegal move in " + fileName + " turn " + Global::intToString(m) + " move " + Location::toString(sgfMoves[m].loc, board.x_size, board.y_size));
        quitEarly = true;
        break;
      }
      hist.makeBoardMoveAssumeLegal(board,sgfMoves[m].loc,sgfMoves[m].pla,NULL,preventEncore);
      nextPla = getOpp(sgfMoves[m].pla);
    }
    boards.push_back(board);
    hists.push_back(hist);
    nextPlas.push_back(nextPla);

    if(winLossValues.size() <= 0)
      return;
    if(shouldStop.load(std::memory_order_acquire))
      return;

    vector<double> futureValue(winLossValues.size()+1);
    vector<double> futureLead(winLossValues.size()+1);
    vector<double> pastValue(winLossValues.size());
    vector<double> pastLead(winLossValues.size());
    futureValue[winLossValues.size()] = (
      (!quitEarly && sgf->sgfWinner == P_WHITE) ? 1.0 :
      (!quitEarly && sgf->sgfWinner == P_BLACK) ? -1.0 :
      winLossValues[winLossValues.size()-1]
    );
    futureLead[winLossValues.size()] = scoreLeads[winLossValues.size()];
    for(int i = (int)winLossValues.size()-1; i >= 0; i--) {
      futureValue[i] = 0.05 * winLossValues[i] + 0.95 * futureValue[i+1];
      futureLead[i] = 0.05 * scoreLeads[i] + 0.95 * futureLead[i+1];
    }
    pastValue[0] = winLossValues[0];
    pastLead[0] = scoreLeads[0];
    for(int i = 1; i<(int)winLossValues.size(); i++) {
      pastValue[i] = 0.5 * winLossValues[i] + 0.5 * pastValue[i+1];
      pastLead[i] = 0.5 * scoreLeads[i] + 0.5 * pastLead[i+1];
    }

    const double scoreLeadWeight = 0.01;
    const double sumThreshold = 0.005;

    //cout << fileName << endl;
    for(int m = 0; m<moves.size(); m++) {

      if(shouldStop.load(std::memory_order_acquire))
        break;

      //cout << m << endl;
      //Look for surprising moves that turned out not poorly
      //The more surprising, the more times we will write it out.
      double weight = surpriseWeight(policyPriors[m],rand,false);
      if(weight <= 0)
        continue;

      double pastSum = pastValue[m] + pastLead[m]*scoreLeadWeight;
      double futureSum = futureValue[m] + futureLead[m]*scoreLeadWeight;
      if((nextPlas[m] == P_WHITE && futureSum > pastSum + sumThreshold) ||
         (nextPlas[m] == P_BLACK && futureSum < pastSum - sumThreshold)) {
        //Good
      }
      else
        continue;

      Sgf::PositionSample sample;
      const int numMovesToRecord = 7;
      int startIdx = std::max(0,m-numMovesToRecord);
      sample.board = boards[startIdx];
      sample.nextPla = nextPlas[startIdx];
      for(int j = startIdx; j<m; j++)
        sample.moves.push_back(moves[j]);
      sample.initialTurnNumber = startIdx;
      sample.hintLoc = moves[m].loc;
      sample.weight = weight;

      if(autoKomi) {
        const int64_t numVisits = 10;
        OtherGameProperties props;
        PlayUtils::adjustKomiToEven(search,NULL,boards[m],hists[m],nextPlas[m],numVisits,logger,props,rand);
      }

      expensiveEvaluateMove(
        search, moves[m].loc, nextPlas[m], boards[m], hists[m],
        sample, false
      );
    }
  };

  // ---------------------------------------------------------------------------------------------------
  //TREE MODE

  auto treePosHandler = [&logger,&gameInit,&nnEval,&expensiveEvaluateMove,autoKomi](
    Search* search, Rand& rand, const BoardHistory& treeHist, int initialTurnNumber, bool markedAsHintPos
  ) {
    if(shouldStop.load(std::memory_order_acquire))
      return;
    if(treeHist.moveHistory.size() > 0x3FFFFFFF)
      throw StringError("Too many moves in history");
    int moveHistorySize = (int)treeHist.moveHistory.size();
    if(moveHistorySize <= 0)
      return;

    //Snap the position 7 turns ago so as to include 7 moves of history.
    int turnsAgoToSnap = 0;
    while(turnsAgoToSnap < 7) {
      if(turnsAgoToSnap >= moveHistorySize)
        break;
      //If a player played twice in a row, then instead snap so as not to have a move history
      //with a double move by the same player.
      if(turnsAgoToSnap > 0 && treeHist.moveHistory[moveHistorySize - turnsAgoToSnap - 1].pla == treeHist.moveHistory[moveHistorySize - turnsAgoToSnap].pla)
        break;
      turnsAgoToSnap++;
    }
    int startTurn = moveHistorySize - turnsAgoToSnap;
    //If the start turn is past the end of the last move, we don't actually have a move we're judging if it's good, so we quit.
    if(startTurn >= moveHistorySize)
      return;

    //Play moves out until we get back to where we need to be.
    //This is hacky and makes everything quadratic, but whatever
    Board board = treeHist.initialBoard;
    for(int i = 0; i<startTurn; i++) {
      bool multiStoneSuicideLegal = true;
      //Just in case
      if(!board.isLegal(treeHist.moveHistory[i].loc,treeHist.moveHistory[i].pla,multiStoneSuicideLegal))
        return;
      board.playMoveAssumeLegal(treeHist.moveHistory[i].loc,treeHist.moveHistory[i].pla);
    }

    Sgf::PositionSample sample;
    sample.board = board;
    sample.nextPla = treeHist.moveHistory[startTurn].pla;
    for(int j = startTurn; j<moveHistorySize-1; j++)
      sample.moves.push_back(treeHist.moveHistory[j]);
    sample.initialTurnNumber = initialTurnNumber;
    sample.hintLoc = treeHist.moveHistory[moveHistorySize-1].loc;
    sample.weight = 0.0; //dummy, filled in below

    //Don't use the SGF rules - randomize them for a bit more entropy
    Rules rules = gameInit->createRules();

    //Now play the rest of the moves out, except the last, which we keep as the potential hintloc
    int encorePhase = 0;
    Player pla = sample.nextPla;
    BoardHistory hist(board,pla,rules,encorePhase);
    int numSampleMoves = (int)sample.moves.size();
    for(int i = 0; i<numSampleMoves; i++) {
      if(!hist.isLegal(board,sample.moves[i].loc,sample.moves[i].pla))
        return;
      assert(sample.moves[i].pla == pla);
      hist.makeBoardMoveAssumeLegal(board,sample.moves[i].loc,sample.moves[i].pla,NULL);
      pla = getOpp(pla);
    }

    //Make sure the hinted move is legal too
    int hintIdx = (int)treeHist.moveHistory.size()-1;
    if(!treeHist.isLegal(board,treeHist.moveHistory[hintIdx].loc,treeHist.moveHistory[hintIdx].pla))
      return;
    assert(treeHist.moveHistory[hintIdx].pla == pla);
    assert(treeHist.moveHistory[hintIdx].loc == sample.hintLoc);

    if(autoKomi) {
      const int64_t numVisits = 10;
      OtherGameProperties props;
      PlayUtils::adjustKomiToEven(search,NULL,board,hist,pla,numVisits,logger,props,rand);
    }

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true; //Always ignore cache so that we get more entropy on repeated board positions due to symmetries
    bool includeOwnerMap = false;
    nnEval->evaluate(board,hist,pla,nnInputParams,buf,skipCache,includeOwnerMap);

    shared_ptr<NNOutput>& nnOutput = buf.result;

    int pos = NNPos::locToPos(sample.hintLoc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
    double policyProb = nnOutput->policyProbs[pos];

    double weight = surpriseWeight(policyProb,rand,markedAsHintPos);
    if(weight <= 0)
      return;
    sample.weight = weight;

    expensiveEvaluateMove(
      search, sample.hintLoc, pla, board, hist,
      sample, markedAsHintPos
    );
  };


  // ---------------------------------------------------------------------------------------------------

  //Begin writing
  std::thread writeLoopThread(writeLoop);

  // ---------------------------------------------------------------------------------------------------

  //In game mode, iterate through sgf games, which are expected to be nonbranching, and see if there are unexpected good moves,
  //requiring the outcome in the game to have been good.
  if(gameMode) {
    const int64_t maxSgfQueueSize = 16384;
    ThreadSafeQueue<size_t> sgfQueue(maxSgfQueueSize);
    std::atomic<int64_t> numSgfsBegun(0);
    std::atomic<int64_t> numSgfsDone(0);
    std::atomic<int64_t> numSgfsSkipped(0);

    auto processSgfLoop = [&sgfFiles,&logger,&processSgfGame,&permutation,&sgfQueue,&params,&numSgfsBegun,&numSgfsDone,&nnEval,&sgfSplitIdx,&sgfSplitCount,&numSgfsSkipped]() {
      Rand rand;
      string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
      Search* search = new Search(params,nnEval,searchRandSeed);

      while(true) {
        if(shouldStop.load(std::memory_order_acquire))
          break;

        size_t idx;
        bool success = sgfQueue.waitPop(idx);
        if(!success)
          break;
        int64_t numBegun = 1+numSgfsBegun.fetch_add(1);
        if(numBegun % 20 == 0)
          logger.write("Begun " + Global::int64ToString(numBegun) + " sgfs");

        const string& fileName = sgfFiles[permutation[idx]];
        CompactSgf* sgf = NULL;
        try {
          sgf = CompactSgf::loadFile(fileName);
        }
        catch(const StringError& e) {
          logger.write("Invalid SGF " + fileName + ": " + e.what());
          continue;
        }

        logger.write("Starting " + fileName);
        if(sgfSplitCount <= 1 || ((int)(sgf->hash.hash0 & 0x7FFFFFFF) % sgfSplitCount) == sgfSplitIdx)
          processSgfGame(search,rand,fileName,sgf);
        else
          numSgfsSkipped.fetch_add(1);

        int64_t numDone = 1+numSgfsDone.fetch_add(1);
        int64_t numSkipped = numSgfsSkipped.load();
        if(numDone % 20 == 0)
          logger.write("Done " + Global::int64ToString(numDone) + " sgfs, " + Global::int64ToString(numSkipped) + " skipped");

        delete sgf;
      }

      delete search;
    };

    vector<std::thread> threads;
    for(int i = 0; i<numProcessThreads; i++) {
      threads.push_back(std::thread(processSgfLoop));
    }

    for(size_t i = 0; i<sgfFiles.size(); i++) {
      sgfQueue.forcePush(i);
    }
    sgfQueue.setReadOnly();

    for(size_t i = 0; i<threads.size(); i++)
      threads[i].join();
  }

  // ---------------------------------------------------------------------------------------------------

  //In tree mode, we just explore everything in the sgf
  else if(treeMode) {
    const int64_t maxPosQueueSize = 16384;
    ThreadSafeQueue<PosQueueEntry> posQueue(maxPosQueueSize);
    std::atomic<int64_t> numPosesBegun(0);
    std::atomic<int64_t> numPosesDone(0);
    std::atomic<int64_t> numPosesEnqueued(0);

    auto processPosLoop = [&logger,&posQueue,&params,&numPosesBegun,&numPosesDone,&nnEval,&treePosHandler]() {
      Rand rand;
      string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
      Search* search = new Search(params,nnEval,searchRandSeed);

      while(true) {
        if(shouldStop.load(std::memory_order_acquire))
          break;

        PosQueueEntry p;
        bool success = posQueue.waitPop(p);
        if(!success)
          break;
        BoardHistory* hist = p.hist;
        int initialTurnNumber = p.initialTurnNumber;
        bool markedAsHintPos = p.markedAsHintPos;

        int64_t numBegun = 1+numPosesBegun.fetch_add(1);
        if(numBegun % 20 == 0)
          logger.write("Begun " + Global::int64ToString(numBegun) + " poses");

        treePosHandler(search, rand, *hist, initialTurnNumber, markedAsHintPos);

        int64_t numDone = 1+numPosesDone.fetch_add(1);
        if(numDone % 20 == 0)
          logger.write("Done " + Global::int64ToString(numDone) + " poses");

        delete hist;
      }
      delete search;
      posQueue.setReadOnly();
    };

    vector<std::thread> threads;
    for(int i = 0; i<numProcessThreads; i++) {
      threads.push_back(std::thread(processPosLoop));
    }

    std::set<Hash128> uniqueHashes;
    for(int i = 0; i<sgfFiles.size(); i++) {
      const string& fileName = sgfFiles[i];
      Sgf* sgf = NULL;
      try {
        sgf = Sgf::loadFile(fileName);
      }
      catch(const StringError& e) {
        logger.write("Invalid SGF " + fileName + ": " + e.what());
        continue;
      }
      if(contains(excludeHashes,sgf->hash))
        continue;
      if(sgfSplitCount > 1 && ((int)(sgf->hash.hash0 & 0x7FFFFFFF) % sgfSplitCount) != sgfSplitIdx) {
        logger.write("Skipping " + fileName);
        continue;
      }

      logger.write("Starting " + fileName);
      bool hashComments = true; //Hash comments so that if we see a position without %HINT% and one with, we make sure to re-load it.
      sgf->iterAllUniquePositions(
        uniqueHashes, hashComments, [&](Sgf::PositionSample& unusedSample, const BoardHistory& hist, const string& comments) {
          if(comments.size() > 0 && Global::trim(comments) == "%NOHINT%") {
            return;
          }
          //unusedSample doesn't have enough history, doesn't have hintloc the way we want it
          int64_t numEnqueued = 1+numPosesEnqueued.fetch_add(1);
          if(numEnqueued % 500 == 0)
            logger.write("Enqueued " + Global::int64ToString(numEnqueued) + " poses");
          PosQueueEntry entry;
          entry.hist = new BoardHistory(hist);
          entry.initialTurnNumber = unusedSample.initialTurnNumber; //this is the only thing we keep
          entry.markedAsHintPos = (comments.size() > 0 && (Global::trim(comments) == "%HINT%"));
          posQueue.waitPush(entry);
        }
      );
      delete sgf;
      logger.write("Done enqueing " + Global::int64ToString(numPosesEnqueued.load()) + " poses");
    }

    posQueue.setReadOnly();
    logger.write("All sgfs processed, waiting for analysis");

    for(int i = 0; i<threads.size(); i++)
      threads[i].join();
  }

  logger.write("Waiting for final writing and cleanup");

  toWriteQueue.setReadOnly();
  writeLoopThread.join();

  logger.write(nnEval->getModelFileName());
  logger.write("NN rows: " + Global::int64ToString(nnEval->numRowsProcessed()));
  logger.write("NN batches: " + Global::int64ToString(nnEval->numBatchesProcessed()));
  logger.write("NN avg batch size: " + Global::doubleToString(nnEval->averageProcessedBatchSize()));

  logger.write("All done");

  delete gameInit;
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}




int MainCmds::trystartposes(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> startPosesFiles;
  double minWeight;
  try {
    KataGoCommandLine cmd("Try running searches starting from startposes");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> startPosesFileArg("","startposes","Startposes file",true,"DIR");
    TCLAP::ValueArg<double> minWeightArg("","min-weight","Minimum weight of startpos to try",false,0.0,"WEIGHT");
    cmd.add(startPosesFileArg);
    cmd.add(minWeightArg);
    cmd.parse(argc,argv);
    nnModelFile = cmd.getModelFile();
    startPosesFiles = startPosesFileArg.getValue();
    minWeight = minWeightArg.getValue();
    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger;
  logger.setLogToStdout(true);

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_ANALYSIS);
  //Ignore temperature, noise
  params.chosenMoveTemperature = 0;
  params.chosenMoveTemperatureEarly = 0;
  params.rootNoiseEnabled = false;
  params.rootDesiredPerChildVisitsCoeff = 0;
  params.rootPolicyTemperature = 1.0;
  params.rootPolicyTemperatureEarly = 1.0;
  params.rootFpuReductionMax = params.fpuReductionMax * 0.5;

  //Disable dynamic utility so that utilities are always comparable
  params.staticScoreUtilityFactor += params.dynamicScoreUtilityFactor;
  params.dynamicScoreUtilityFactor = 0;

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int expectedConcurrentEvals = params.numThreads;
    int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  vector<Sgf::PositionSample> startPoses;
  for(size_t i = 0; i<startPosesFiles.size(); i++) {
    const string& startPosesFile = startPosesFiles[i];
    vector<string> lines = Global::readFileLines(startPosesFile,'\n');
    for(size_t j = 0; j<lines.size(); j++) {
      string line = Global::trim(lines[j]);
      if(line.size() > 0) {
        try {
          Sgf::PositionSample posSample = Sgf::PositionSample::ofJsonLine(line);
          startPoses.push_back(posSample);
        }
        catch(const StringError& err) {
          logger.write(string("ERROR parsing startpos:") + err.what());
        }
      }
    }
  }
  string searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());
  Search* search = new Search(params,nnEval,searchRandSeed);

  // ---------------------------------------------------------------------------------------------------

  for(size_t s = 0; s<startPoses.size(); s++) {
    const Sgf::PositionSample& startPos = startPoses[s];
    if(startPos.weight < minWeight)
      continue;

    Rules rules = PlayUtils::genRandomRules(seedRand);
    Board board = startPos.board;
    Player pla = startPos.nextPla;
    BoardHistory hist;
    hist.clear(board,pla,rules,0);
    hist.setInitialTurnNumber(startPos.initialTurnNumber);
    bool allLegal = true;
    for(size_t i = 0; i<startPos.moves.size(); i++) {
      bool isLegal = hist.makeBoardMoveTolerant(board,startPos.moves[i].loc,startPos.moves[i].pla,false);
      if(!isLegal) {
        allLegal = false;
        break;
      }
      pla = getOpp(startPos.moves[i].pla);
    }
    if(!allLegal) {
      throw StringError("Illegal move in startpos: " + Sgf::PositionSample::toJsonLine(startPos));
    }

    {
      const int64_t numVisits = 10;
      OtherGameProperties props;
      PlayUtils::adjustKomiToEven(search,NULL,board,hist,pla,numVisits,logger,props,seedRand);
    }

    Loc hintLoc = startPos.hintLoc;

    {
      ReportedSearchValues values;
      bool suc = maybeGetValuesAfterMove(search,logger,Board::NULL_LOC,pla,board,hist,1.0,values);
      (void)suc;
      assert(suc);
      cout << "Searching startpos: " << "\n";
      cout << "Weight: " << startPos.weight << "\n";
      cout << search->getRootHist().rules.toString() << "\n";
      Board::printBoard(cout, search->getRootBoard(), search->getChosenMoveLoc(), &(search->getRootHist().moveHistory));
      search->printTree(cout, search->rootNode, PrintTreeOptions().maxDepth(1),P_WHITE);
      cout << endl;
    }

    if(hintLoc != Board::NULL_LOC) {
      if(search->getChosenMoveLoc() == hintLoc) {
        cout << "There was a hintpos " << Location::toString(hintLoc,board) << ", but it was the chosen move" << "\n";
        cout << endl;
      }
      else {
        ReportedSearchValues values;
        cout << "There was a hintpos " << Location::toString(hintLoc,board) << ", re-searching after playing it: " << "\n";
        bool suc = maybeGetValuesAfterMove(search,logger,hintLoc,pla,board,hist,1.0,values);
        (void)suc;
        assert(suc);
        Board::printBoard(cout, search->getRootBoard(), search->getChosenMoveLoc(), &(search->getRootHist().moveHistory));
        search->printTree(cout, search->rootNode, PrintTreeOptions().maxDepth(1),P_WHITE);
        cout << endl;
      }
    }
  }

  delete search;
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}


int MainCmds::viewstartposes(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();

  vector<string> startPosesFiles;
  try {
    KataGoCommandLine cmd("View startposes");

    TCLAP::MultiArg<string> startPosesFileArg("","startPosesFile","Startposes file",true,"DIR");
    cmd.add(startPosesFileArg);
    cmd.parse(argc,argv);
    startPosesFiles = startPosesFileArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  vector<Sgf::PositionSample> startPoses;
  for(size_t i = 0; i<startPosesFiles.size(); i++) {
    const string& startPosesFile = startPosesFiles[i];
    vector<string> lines = Global::readFileLines(startPosesFile,'\n');
    for(size_t j = 0; j<lines.size(); j++) {
      string line = Global::trim(lines[j]);
      if(line.size() > 0) {
        try {
          Sgf::PositionSample posSample = Sgf::PositionSample::ofJsonLine(line);
          startPoses.push_back(posSample);
        }
        catch(const StringError& err) {
          cout << (string("ERROR parsing startpos:") + err.what()) << endl;
        }
      }
    }
  }

  for(size_t s = 0; s<startPoses.size(); s++) {
    const Sgf::PositionSample& startPos = startPoses[s];

    Board board = startPos.board;
    Player pla = startPos.nextPla;
    BoardHistory hist;
    hist.clear(board,pla,Rules(),0);
    hist.setInitialTurnNumber(startPos.initialTurnNumber);

    bool allLegal = true;
    for(size_t i = 0; i<startPos.moves.size(); i++) {
      bool isLegal = hist.makeBoardMoveTolerant(board,startPos.moves[i].loc,startPos.moves[i].pla,false);
      if(!isLegal) {
        allLegal = false;
        break;
      }
      pla = getOpp(startPos.moves[i].pla);
    }
    if(!allLegal) {
      throw StringError("Illegal move in startpos: " + Sgf::PositionSample::toJsonLine(startPos));
    }

    Loc hintLoc = startPos.hintLoc;
    cout << "Weight: " << startPos.weight << "\n";
    cout << "There was a hintpos " << Location::toString(hintLoc,board) << "\n";
    Board::printBoard(cout, board, hintLoc, &(hist.moveHistory));
    cout << endl;
  }

  ScoreValue::freeTables();
  return 0;
}
