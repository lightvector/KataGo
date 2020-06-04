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

  // cout << baseHist.rules << endl;
  // cout << board << endl;
  // if(winLossHistory.size() > 0)
  //   cout << winLossHistory[winLossHistory.size()-1] << " ";
  // assert(scoreStdevHistory.size() == scoreHistory.size());
  // if(scoreHistory.size() > 0) {
  //   cout << scoreHistory[scoreHistory.size()-1] << " ";
  //   cout << scoreStdevHistory[scoreStdevHistory.size()-1] << " ";
  // }
  // cout << endl;


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

      vector<Move> chosenOpening = specialOpenings[rand.nextUInt(specialOpenings.size())];
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
      for(int j = chosenOpenings.size()-1; j>=1; j--) {
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

  SearchParams params = Setup::loadSingleParams(cfg);

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int defaultMaxBatchSize = -1;
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,cfg,logger,seedRand,maxConcurrentEvals,
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

    auto callback = [&baseHist,&recentWinLossValues,&recentScores,&recentScoreStdevs](const Search* search) {
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


static uint64_t parseHex64(const string& str) {
  assert(str.length() == 16);
  uint64_t x = 0;
  for(int i = 0; i<16; i++) {
    x *= 16;
    if(str[i] >= '0' && str[i] <= '9')
      x += str[i] - '0';
    else if(str[i] >= 'a' && str[i] <= 'f')
      x += str[i] - 'a' + 10;
    else if(str[i] >= 'A' && str[i] <= 'F')
      x += str[i] - 'A' + 10;
    else
      assert(false);
  }
  return x;
}

int MainCmds::dataminesgfs(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> sgfDirs;
  string outDir;
  int numSearchThreads;
  vector<string> excludeHashesFiles;
  bool gameMode;
  bool treeMode;
  try {
    KataGoCommandLine cmd("Search for suprising good moves in sgfs");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();

    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",true,"DIR");
    TCLAP::ValueArg<string> outDirArg("","outdir","Directory to write results",true,string(),"DIR");
    TCLAP::ValueArg<int> numSearchThreadsArg("","threads","Number of threads",true,1,"THREADS");
    TCLAP::MultiArg<string> excludeHashesArg("","exclude-hashes","Specify a list of hashes to filter out, one per line in a txt file",false,"FILEOF(HASH,HASH)");
    TCLAP::SwitchArg gameModeArg("","game-mode","Game mode");
    TCLAP::SwitchArg treeModeArg("","tree-mode","Tree mode");
    cmd.add(sgfDirArg);
    cmd.add(outDirArg);
    cmd.add(numSearchThreadsArg);
    cmd.add(excludeHashesArg);
    cmd.add(gameModeArg);
    cmd.add(treeModeArg);
    cmd.parse(argc,argv);
    nnModelFile = cmd.getModelFile();
    sgfDirs = sgfDirArg.getValue();
    outDir = outDirArg.getValue();
    numSearchThreads = numSearchThreadsArg.getValue();
    excludeHashesFiles = excludeHashesArg.getValue();
    gameMode = gameModeArg.getValue();
    treeMode = treeModeArg.getValue();

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

  SearchParams params = Setup::loadSingleParams(cfg);
  //Ignore temperature, noise
  params.chosenMoveTemperature = 0;
  params.chosenMoveTemperatureEarly = 0;
  params.rootNoiseEnabled = false;
  params.rootDesiredPerChildVisitsCoeff = 0;
  params.rootPolicyTemperature = 1.0;
  params.rootPolicyTemperatureEarly = 1.0;
  params.rootFpuReductionMax = params.fpuReductionMax * 0.5;
  params.rootNumSymmetriesToSample = 1;

  //Disable dynamic utility so that utilities are always comparable
  params.staticScoreUtilityFactor += params.dynamicScoreUtilityFactor;
  params.dynamicScoreUtilityFactor = 0;

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,cfg,logger,seedRand,maxConcurrentEvals,
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

  vector<int64_t> permutation(sgfFiles.size());
  for(int64_t i = 0; i<sgfFiles.size(); i++)
    permutation[i] = i;
  for(int64_t i = 1; i<sgfFiles.size(); i++) {
    int64_t r = (int64_t)seedRand.nextUInt64(i+1);
    std::swap(permutation[i],permutation[r]);
  }

  set<Hash128> excludeHashes;
  for(int i = 0; i<excludeHashesFiles.size(); i++) {
    const string& excludeHashesFile = excludeHashesFiles[i];
    vector<string> hashes = Global::readFileLines(excludeHashesFile,'\n');
    for(int64_t j = 0; j < hashes.size(); j++) {
      const string& hash128 = Global::trim(Global::stripComments(hashes[j]));
      if(hash128.length() <= 0)
        continue;
      if(hash128.length() != 32)
        throw IOError("Could not parse hashpair in exclude hashes file: " + hash128);

      uint64_t hash0 = parseHex64(hash128.substr(0,16));
      uint64_t hash1 = parseHex64(hash128.substr(16,16));
      excludeHashes.insert(Hash128(hash0,hash1));
    }
  }

  MakeDir::make(outDir);

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

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

  static const int minMoveIdx = 4;

  //We want surprising moves that turned out not poorly
  //The more surprising, the more times we will write it out.
  auto numTimesToWriteOfPolicy = [](double policyProb, Rand& rand) {
    if(policyProb < 0)
      return 0;
    double numTimesToWriteFloat = 0.15 / (policyProb + 0.03) - 0.5;
    if(numTimesToWriteFloat <= 0)
      return 0;
    int numTimesToWrite = (int)floor(numTimesToWriteFloat);
    if(rand.nextBool(numTimesToWriteFloat - numTimesToWrite))
      numTimesToWrite += 1;
    return numTimesToWrite;
  };


  auto expensiveEvaluateMove = [&toWriteQueue,&logger](
    Search* search, Loc missedLoc,
    Player nextPla, const Board& board, const BoardHistory& hist,
    const Sgf::PositionSample& sample,
    Rand& rand,
    int numTimesToWrite
  ) {
    //cout << "EXPENSIVE" << endl;
    //Do a more expensive search before and after
    search->setPosition(nextPla,board,hist);
    search->runWholeSearch(nextPla,logger,shouldStop);
    if(shouldStop.load(std::memory_order_acquire))
      return;
    Loc moveLoc = search->getChosenMoveLoc();

    // const Player perspective = P_WHITE;
    // {
    //   ostringstream preOut;
    //   Board::printBoard(preOut, search->getRootBoard(), moveLoc, &(search->getRootHist().moveHistory));
    //   search->printTree(preOut, search->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10),perspective);
    //   cout << preOut.str() << endl;
    //   cout << Location::toString(missedLoc,board) << endl;
    // }

    //Bot DOES see the move?
    if(moveLoc == missedLoc) {
      //Still good to learn from given that policy was really low
      for(int n = 0; n<numTimesToWrite; n++)
        toWriteQueue.waitPush(new string(Sgf::PositionSample::toJsonLine(sample)));
    }

    //Bot doesn't see the move?
    else if(moveLoc != missedLoc) {
      vector<Loc> locs;
      vector<double> playSelectionValues;
      search->getPlaySelectionValues(locs,playSelectionValues,1.0);
      //Did the move not get much of the play selection value?
      double psvSum = 0.0;
      double psvForMove = 0.0;
      for(int k = 0; k<playSelectionValues.size(); k++) {
        psvSum += playSelectionValues[k];
        if(locs[k] == missedLoc)
          psvForMove = playSelectionValues[k];
      }
      if(psvForMove < psvSum * (0.10 + rand.nextDouble(0.90))) {
        // cout << "SECOND EXPENSIVE" << endl;

        ReportedSearchValues preValues = search->getRootValuesRequireSuccess();

        // ostringstream preOut;
        // Board::printBoard(preOut, search->getRootBoard(), moveLoc, &(search->getRootHist().moveHistory));
        // search->printTree(preOut, search->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10),perspective);

        Board newBoard = board;
        BoardHistory newHist = hist;
        Player newNextPla = nextPla;
        if(!hist.isLegal(newBoard,missedLoc,newNextPla))
          return;
        newHist.makeBoardMoveAssumeLegal(newBoard,missedLoc,newNextPla,NULL);
        newNextPla = getOpp(newNextPla);

        search->setPosition(newNextPla,newBoard,newHist);
        search->runWholeSearch(newNextPla,logger,shouldStop);
        if(shouldStop.load(std::memory_order_acquire))
          return;

        // {
        //   ostringstream postOut;
        //   Board::printBoard(postOut, search->getRootBoard(), Board::NULL_LOC, &(search->getRootHist().moveHistory));
        //   search->printTree(postOut, search->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10),perspective);
        //   cout << postOut.str() << endl;
        // }

        const double utilityThreshold = 0.005;

        ReportedSearchValues postValues = search->getRootValuesRequireSuccess();
        if((nextPla == P_WHITE && postValues.utility > preValues.utility + utilityThreshold) ||
           (nextPla == P_BLACK && postValues.utility < preValues.utility - utilityThreshold)) {
          // ostringstream postOut;
          // Board::printBoard(postOut, search->getRootBoard(), Board::NULL_LOC, &(search->getRootHist().moveHistory));
          // search->printTree(postOut, search->rootNode, PrintTreeOptions().maxDepth(1).maxChildrenToShow(10),perspective);

          // cout << "YAAAAAAY" << endl;

          //Moves that the bot didn't see get written out more
          numTimesToWrite *= 2;

          for(int n = 0; n<numTimesToWrite; n++)
            toWriteQueue.waitPush(new string(Sgf::PositionSample::toJsonLine(sample)));

          // string s;
          // s += "=======================================================\n";
          // s += preOut.str();
          // s += "\nMOVE = " + Location::toString(missedLoc,board) + "\n";
          // s += postOut.str();
          // cout << s << endl;
        }
      }
    }
  };

  auto processSgfGame = [&logger,&excludeHashes,&gameInit,&nnEval,&expensiveEvaluateMove,&numTimesToWriteOfPolicy](
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
    for(int i = winLossValues.size()-1; i >= 0; i--) {
      futureValue[i] = 0.05 * winLossValues[i] + 0.95 * futureValue[i+1];
      futureLead[i] = 0.05 * scoreLeads[i] + 0.95 * futureLead[i+1];
    }
    pastValue[0] = winLossValues[0];
    pastLead[0] = scoreLeads[0];
    for(int i = 1; i<winLossValues.size(); i++) {
      pastValue[i] = 0.5 * winLossValues[i] + 0.5 * pastValue[i+1];
      pastLead[i] = 0.5 * scoreLeads[i] + 0.5 * pastLead[i+1];
    }

    const double scoreLeadWeight = 0.01;
    const double sumThreshold = 0.005;

    //cout << fileName << endl;
    for(int m = minMoveIdx; m<moves.size(); m++) {

      if(shouldStop.load(std::memory_order_acquire))
        break;

      //cout << m << endl;
      //Look for surprising moves that turned out not poorly
      //The more surprising, the more times we will write it out.
      int numTimesToWrite = numTimesToWriteOfPolicy(policyPriors[m],rand);
      if(numTimesToWrite <= 0)
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

      expensiveEvaluateMove(
        search, moves[m].loc, nextPlas[m], boards[m], hists[m],
        sample, rand, numTimesToWrite
      );
    }
  };

  auto treePosHandler = [&logger,&gameInit,&nnEval,&expensiveEvaluateMove,&numTimesToWriteOfPolicy](Search* search, Rand& rand, const BoardHistory& treeHist) {
    if(shouldStop.load(std::memory_order_acquire))
      return;
    int moveHistorySize = treeHist.moveHistory.size();
    if(moveHistorySize <= 0)
      return;
    if(treeHist.initialTurnNumber + treeHist.moveHistory.size() < minMoveIdx)
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
    sample.initialTurnNumber = startTurn;
    sample.hintLoc = treeHist.moveHistory[moveHistorySize-1].loc;

    //Don't use the SGF rules - randomize them for a bit more entropy
    Rules rules = gameInit->createRules();

    //Now play the rest of the moves out, except the last, which we keep as the potential hintloc
    int encorePhase = 0;
    Player pla = sample.nextPla;
    BoardHistory hist(board,pla,rules,encorePhase);
    int numSampleMoves = sample.moves.size();
    for(int i = 0; i<numSampleMoves; i++) {
      if(!hist.isLegal(board,sample.moves[i].loc,sample.moves[i].pla))
        return;
      assert(sample.moves[i].pla == pla);
      hist.makeBoardMoveAssumeLegal(board,sample.moves[i].loc,sample.moves[i].pla,NULL);
      pla = getOpp(pla);
    }

    //Make sure the hinted move is legal too
    int hintIdx = treeHist.moveHistory.size()-1;
    if(!treeHist.isLegal(board,treeHist.moveHistory[hintIdx].loc,treeHist.moveHistory[hintIdx].pla))
      return;
    assert(treeHist.moveHistory[hintIdx].pla == pla);
    assert(treeHist.moveHistory[hintIdx].loc == sample.hintLoc);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true; //Always ignore cache so that we get more entropy on repeated board positions due to symmetries
    bool includeOwnerMap = false;
    nnEval->evaluate(board,hist,pla,nnInputParams,buf,skipCache,includeOwnerMap);

    shared_ptr<NNOutput>& nnOutput = buf.result;

    int pos = NNPos::locToPos(sample.hintLoc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
    double policyProb = nnOutput->policyProbs[pos];

    int numTimesToWrite = numTimesToWriteOfPolicy(policyProb,rand);
    if(numTimesToWrite <= 0)
      return;

    expensiveEvaluateMove(
      search, sample.hintLoc, pla, board, hist,
      sample, rand, numTimesToWrite
    );
  };

  //Begin writing
  std::thread writeLoopThread(writeLoop);

  //In game mode, iterate through sgf games, which are expected to be nonbranching, and see if there are unexpected good moves,
  //requiring the outcome in the game to have been good.
  if(gameMode) {
    const int64_t maxSgfQueueSize = 1024;
    ThreadSafeQueue<int64_t> sgfQueue(maxSgfQueueSize);
    std::atomic<int64_t> numSgfsBegun(0);
    std::atomic<int64_t> numSgfsDone(0);

    auto processSgfLoop = [&sgfFiles,&logger,&processSgfGame,&permutation,&sgfQueue,&params,&numSgfsBegun,&numSgfsDone,&nnEval]() {
      Rand rand;
      string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
      Search* search = new Search(params,nnEval,searchRandSeed);

      while(true) {
        if(shouldStop.load(std::memory_order_acquire))
          break;

        int64_t idx;
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
        processSgfGame(search,rand,fileName,sgf);
        int64_t numDone = 1+numSgfsDone.fetch_add(1);
        if(numDone % 20 == 0)
          logger.write("Done " + Global::int64ToString(numDone) + " sgfs");

        delete sgf;
      }

      delete search;
    };

    vector<std::thread> threads;
    for(int i = 0; i<numSearchThreads; i++) {
      threads.push_back(std::thread(processSgfLoop));
    }

    for(int64_t i = 0; i<sgfFiles.size(); i++) {
      sgfQueue.forcePush(i);
    }
    sgfQueue.setReadOnly();

    for(int i = 0; i<threads.size(); i++)
      threads[i].join();
  }

  else if(treeMode) {
    const int64_t maxPosQueueSize = 1024;
    ThreadSafeQueue<BoardHistory*> posQueue(maxPosQueueSize);
    std::atomic<int64_t> numPosesBegun(0);
    std::atomic<int64_t> numPosesDone(0);

    auto processPosLoop = [&logger,&posQueue,&params,&numPosesBegun,&numPosesDone,&nnEval,&treePosHandler]() {
      Rand rand;
      string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
      Search* search = new Search(params,nnEval,searchRandSeed);

      while(true) {
        if(shouldStop.load(std::memory_order_acquire))
          break;

        BoardHistory* hist;
        bool success = posQueue.waitPop(hist);
        if(!success)
          break;
        int64_t numBegun = 1+numPosesBegun.fetch_add(1);
        if(numBegun % 20 == 0)
          logger.write("Begun " + Global::int64ToString(numBegun) + " poses");

        treePosHandler(search, rand, *hist);

        int64_t numDone = 1+numPosesDone.fetch_add(1);
        if(numDone % 20 == 0)
          logger.write("Done " + Global::int64ToString(numDone) + " poses");

        delete hist;
      }
      delete search;
      posQueue.setReadOnly();
    };

    vector<std::thread> threads;
    for(int i = 0; i<numSearchThreads; i++) {
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

      logger.write("Starting " + fileName);
      sgf->iterAllUniquePositions(
        uniqueHashes, [&](Sgf::PositionSample& unusedSample, const BoardHistory& hist) {
          //Doesn't have enough history, doesn't have hintloc the way we want it
          (void)unusedSample;
          posQueue.waitPush(new BoardHistory(hist));
        }
      );
      delete sgf;
    }

    posQueue.setReadOnly();

    for(int i = 0; i<threads.size(); i++)
      threads[i].join();
  }

  logger.write("All sgfs processed, waiting for writing");

  toWriteQueue.setReadOnly();
  writeLoopThread.join();

  logger.write("All done");

  delete gameInit;
  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}
