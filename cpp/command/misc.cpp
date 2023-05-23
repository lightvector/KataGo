#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/test.h"
#include "../dataio/sgf.h"
#include "../dataio/poswriter.h"
#include "../dataio/files.h"
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
    bool duplicateForSymmetries = true;
    search->getAnalysisData(buf,minMovesToTryToGet,false,9,duplicateForSymmetries);
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

  vector<double> ownership = search->getAverageTreeOwnership();
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

static void initializeDemoGame(Board& board, BoardHistory& hist, Player& pla, Rand& rand, AsyncBot* bot) {
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
      PlayUtils::adjustKomiToEven(bot->getSearchStopAndWait(),NULL,board,hist,pla,numVisits,OtherGameProperties(),rand);
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


int MainCmds::demoplay(const vector<string>& args) {
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
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    logFile = logFileArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(&cfg);
  logger.addFile(logFile);

  logger.write("Engine starting...");

  string searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    const int expectedConcurrentEvals = params.numThreads;
    const int defaultMaxBatchSize = -1;
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
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

    initializeDemoGame(baseBoard, baseHist, pla, gameRand, bot);

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
      Loc moveLoc = bot->genMoveSynchronousAnalyze(pla,tc,searchFactor,callbackPeriod,callbackPeriod,callback);

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

int MainCmds::printclockinfo(const vector<string>& args) {
  (void)args;
#ifdef OS_IS_WINDOWS
  cout << "Does nothing on windows, disabled" << endl;
#endif
#ifdef OS_IS_UNIX_OR_APPLE
  cout << "Tick unit in seconds: " << std::chrono::steady_clock::period::num << " / " <<  std::chrono::steady_clock::period::den << endl;
  cout << "Ticks since epoch: " << std::chrono::steady_clock::now().time_since_epoch().count() << endl;
#endif
  return 0;
}


int MainCmds::samplesgfs(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  vector<string> sgfDirs;
  vector<string> sgfsDirs;
  string outDir;
  vector<string> excludeHashesFiles;
  double sampleProb;
  double sampleWeight;
  double forceSampleWeight;
  double turnWeightLambda;
  double minWeight;
  int64_t maxDepth;
  int64_t maxNodeCount;
  int64_t maxBranchCount;
  double minTurnNumberBoardAreaProp;
  double maxTurnNumberBoardAreaProp;
  bool flipIfPassOrWFirst;
  double afterPassFactor;
  bool allowGameOver;
  bool hashComments;
  double trainingWeight;
  int verbosity;

  string valueFluctuationModelFile;
  double valueFluctuationTurnScale;
  double valueFluctuationMaxWeight;

  int minMinRank;
  int minMinRating;
  string requiredPlayerName;
  int maxHandicap;
  double maxKomi;

  try {
    KataGoCommandLine cmd("Search for suprising good moves in sgfs");

    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",false,"DIR");
    TCLAP::MultiArg<string> sgfsDirArg("","sgfsdir","Directory of sgfs files",false,"DIR");
    TCLAP::ValueArg<string> outDirArg("","outdir","Directory to write results",true,string(),"DIR");
    TCLAP::MultiArg<string> excludeHashesArg("","exclude-hashes","Specify a list of hashes to filter out, one per line in a txt file",false,"FILEOF(HASH,HASH)");
    TCLAP::ValueArg<double> sampleProbArg("","sample-prob","Probability to sample each position",true,0.0,"PROB");
    TCLAP::ValueArg<double> sampleWeightArg("","sample-weight","",false,1.0,"Weight");
    TCLAP::ValueArg<double> forceSampleWeightArg("","force-sample-weight","",false,5.0,"Weight");
    TCLAP::ValueArg<double> turnWeightLambdaArg("","turn-weight-lambda","Adjust weight for writing down each position",true,0.0,"LAMBDA");
    TCLAP::ValueArg<double> minWeightArg("","min-weight","",false,0.0,"Weight");
    TCLAP::ValueArg<string> maxDepthArg("","max-depth","Max depth allowed for sgf",false,"100000000","INT");
    TCLAP::ValueArg<string> maxNodeCountArg("","max-node-count","Max node count allowed for sgf",false,"100000000","INT");
    TCLAP::ValueArg<string> maxBranchCountArg("","max-branch-count","Max branch count allowed for sgf",false,"100000000","INT");
    TCLAP::ValueArg<double> minTurnNumberBoardAreaPropArg("","min-turn-number-board-area-prop","Only use turn number >= this board area",false,-1.0,"PROP");
    TCLAP::ValueArg<double> maxTurnNumberBoardAreaPropArg("","max-turn-number-board-area-prop","Only use turn number <= this board area",false,10000.0,"PROP");
    TCLAP::SwitchArg flipIfPassOrWFirstArg("","flip-if-pass","Try to heuristically find cases where an sgf passes to simulate white<->black");
    TCLAP::ValueArg<double> afterPassFactorArg("","after-pass-factor","Scale down weight of positions following a pass",false, 1.0, "FACTOR");
    TCLAP::SwitchArg allowGameOverArg("","allow-game-over","Allow sampling game over positions in sgf");
    TCLAP::SwitchArg hashCommentsArg("","hash-comments","Hash comments in sgf");
    TCLAP::ValueArg<double> trainingWeightArg("","training-weight","Scale the loss function weight from data from games that originate from this position",false,1.0,"WEIGHT");
    TCLAP::ValueArg<int> verbosityArg("","verbosity","Print more stuff",false,0,"INT");

    TCLAP::ValueArg<string> valueFluctuationModelFileArg("","value-fluctuation-model","Upweight positions prior to value fluctuations",false,string(),"MODELFILE");
    TCLAP::ValueArg<double> valueFluctuationTurnScaleArg("","value-fluctuation-turn-scale","How much prior on average",false,1.0,"AVGTURNS");
    TCLAP::ValueArg<double> valueFluctuationMaxWeightArg("","value-fluctuation-max-weight","",false,10.0,"MAXWEIGHT");
    TCLAP::ValueArg<int> minMinRankArg("","min-min-rank","Require both players in a game to have rank at least this",false,Sgf::RANK_UNKNOWN,"INT");
    TCLAP::ValueArg<int> minMinRatingArg("","min-min-rating","Require both players in a game to have rating at least this",false,-1000000000,"INT");
    TCLAP::ValueArg<string> requiredPlayerNameArg("","required-player-name","Require player making the move to have this name",false,string(),"NAME");
    TCLAP::ValueArg<int> maxHandicapArg("","max-handicap","Require no more than this big handicap in stones",false,100,"INT");
    TCLAP::ValueArg<double> maxKomiArg("","max-komi","Require absolute value of game komi to be at most this",false,1000,"KOMI");

    cmd.add(sgfDirArg);
    cmd.add(sgfsDirArg);
    cmd.add(outDirArg);
    cmd.add(excludeHashesArg);
    cmd.add(sampleProbArg);
    cmd.add(sampleWeightArg);
    cmd.add(forceSampleWeightArg);
    cmd.add(turnWeightLambdaArg);
    cmd.add(minWeightArg);
    cmd.add(maxDepthArg);
    cmd.add(maxNodeCountArg);
    cmd.add(maxBranchCountArg);
    cmd.add(minTurnNumberBoardAreaPropArg);
    cmd.add(maxTurnNumberBoardAreaPropArg);
    cmd.add(flipIfPassOrWFirstArg);
    cmd.add(afterPassFactorArg);
    cmd.add(allowGameOverArg);
    cmd.add(hashCommentsArg);
    cmd.add(trainingWeightArg);
    cmd.add(verbosityArg);
    cmd.add(valueFluctuationModelFileArg);
    cmd.add(valueFluctuationTurnScaleArg);
    cmd.add(valueFluctuationMaxWeightArg);
    cmd.add(minMinRankArg);
    cmd.add(minMinRatingArg);
    cmd.add(requiredPlayerNameArg);
    cmd.add(maxHandicapArg);
    cmd.add(maxKomiArg);
    cmd.parseArgs(args);
    sgfDirs = sgfDirArg.getValue();
    sgfsDirs = sgfsDirArg.getValue();
    outDir = outDirArg.getValue();
    excludeHashesFiles = excludeHashesArg.getValue();
    sampleProb = sampleProbArg.getValue();
    sampleWeight = sampleWeightArg.getValue();
    minWeight = minWeightArg.getValue();
    forceSampleWeight = forceSampleWeightArg.getValue();
    turnWeightLambda = turnWeightLambdaArg.getValue();
    maxDepth = Global::stringToInt64(maxDepthArg.getValue());
    maxNodeCount = Global::stringToInt64(maxNodeCountArg.getValue());
    maxBranchCount = Global::stringToInt64(maxBranchCountArg.getValue());
    minTurnNumberBoardAreaProp = minTurnNumberBoardAreaPropArg.getValue();
    maxTurnNumberBoardAreaProp = maxTurnNumberBoardAreaPropArg.getValue();
    flipIfPassOrWFirst = flipIfPassOrWFirstArg.getValue();
    afterPassFactor = afterPassFactorArg.getValue();
    allowGameOver = allowGameOverArg.getValue();
    hashComments = hashCommentsArg.getValue();
    trainingWeight = trainingWeightArg.getValue();
    verbosity = verbosityArg.getValue();
    valueFluctuationModelFile = valueFluctuationModelFileArg.getValue();
    valueFluctuationTurnScale = valueFluctuationTurnScaleArg.getValue();
    valueFluctuationMaxWeight = valueFluctuationMaxWeightArg.getValue();
    minMinRank = minMinRankArg.getValue();
    minMinRating = minMinRatingArg.getValue();
    requiredPlayerName = requiredPlayerNameArg.getValue();
    maxHandicap = maxHandicapArg.getValue();
    maxKomi = maxKomiArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  MakeDir::make(outDir);

  const bool logToStdout = true;
  Logger logger(nullptr, logToStdout);
  logger.addFile(outDir + "/" + "log.log");
  for(const string& arg: args)
    logger.write(string("Command: ") + arg);

  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirsOrFiles(sgfDirs,sgfFiles);
  logger.write("Found " + Global::int64ToString((int64_t)sgfFiles.size()) + " sgf files!");

  std::set<string> sgfsSet;
  {
    vector<string> sgfsFiles;
    FileHelpers::collectMultiSgfsFromDirsOrFiles(sgfsDirs,sgfsFiles);
    logger.write("Found " + Global::int64ToString((int64_t)sgfsFiles.size()) + " sgfs files!");
    for(const string& s: sgfsFiles) {
      sgfFiles.push_back(s);
      sgfsSet.insert(s);
    }
  }

  set<Hash128> excludeHashes = Sgf::readExcludes(excludeHashesFiles);
  logger.write("Loaded " + Global::uint64ToString(excludeHashes.size()) + " excludes");

  NNEvaluator* valueFluctuationNNEval = NULL;
  if(valueFluctuationModelFile != "") {
    if(valueFluctuationTurnScale <= 0.0 || valueFluctuationTurnScale > 100000000.0)
      throw StringError("Invalid valueFluctuationTurnScale");
    if(valueFluctuationMaxWeight <= 0.0 || valueFluctuationMaxWeight > 100000000.0)
      throw StringError("Invalid valueFluctuationMaxWeight");
    ConfigParser cfg;
    Setup::initializeSession(cfg);
    int numThreads = 1;
    const int maxConcurrentEvals = numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    const int expectedConcurrentEvals = numThreads;
    const int defaultMaxBatchSize = std::max(8,((numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    valueFluctuationNNEval = Setup::initializeNNEvaluator(
      valueFluctuationModelFile,valueFluctuationModelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
    logger.write("Loaded neural net");
  }

  // ---------------------------------------------------------------------------------------------------

  auto isPlayerOkay = [&](const Sgf* sgf, Player pla) {
    if(requiredPlayerName != "") {
      if(sgf->getPlayerName(pla) != requiredPlayerName)
        return false;
    }
    return true;
  };

  auto isSgfOkay = [&](const Sgf* sgf) {
    if(maxHandicap < 100 && sgf->getHandicapValue() > maxHandicap)
      return false;
    if(sgf->depth() > maxDepth)
      return false;
    if(std::fabs(sgf->getKomi()) > maxKomi)
      return false;
    if(minMinRank != Sgf::RANK_UNKNOWN) {
      if(sgf->getRank(P_BLACK) < minMinRank || sgf->getRank(P_WHITE) < minMinRank)
        return false;
    }
    if(minMinRating > -10000000) {
      if(sgf->getRating(P_BLACK) < minMinRating || sgf->getRating(P_WHITE) < minMinRating)
        return false;
    }
    if(!isPlayerOkay(sgf,P_BLACK) && !isPlayerOkay(sgf,P_WHITE))
      return false;
    return true;
  };

  // ---------------------------------------------------------------------------------------------------
  PosWriter posWriter("startposes.txt", outDir, 1, 0, 100000);
  posWriter.start();

  // ---------------------------------------------------------------------------------------------------

  int64_t numKept = 0;
  double weightKept = 0;
  std::set<Hash128> uniqueHashes;
  std::function<void(Sgf::PositionSample&, const BoardHistory&, const string&)> posHandler =
    [sampleProb,sampleWeight,forceSampleWeight,&posWriter,turnWeightLambda,&numKept,&weightKept,&seedRand,minTurnNumberBoardAreaProp,maxTurnNumberBoardAreaProp,afterPassFactor,trainingWeight,minWeight](
      Sgf::PositionSample& posSample, const BoardHistory& hist, const string& comments
    ) {
      double minTurnNumber = minTurnNumberBoardAreaProp * (hist.initialBoard.x_size * hist.initialBoard.y_size);
      double maxTurnNumber = maxTurnNumberBoardAreaProp * (hist.initialBoard.x_size * hist.initialBoard.y_size);
      if(hist.initialBoard.numStonesOnBoard() + hist.moveHistory.size() < minTurnNumber ||
         hist.initialBoard.numStonesOnBoard() + hist.moveHistory.size() > maxTurnNumber) {
        return;
      }
      if(comments.size() > 0 && comments.find("%NOSAMPLE%") != string::npos)
        return;

      if(seedRand.nextBool(sampleProb)) {
        Sgf::PositionSample posSampleToWrite = posSample;
        int64_t startTurn = posSampleToWrite.initialTurnNumber + (int64_t)posSampleToWrite.moves.size();
        posSampleToWrite.weight = sampleWeight * exp(-startTurn * turnWeightLambda) * posSampleToWrite.weight;
        if(posSampleToWrite.moves.size() > 0 && posSampleToWrite.moves[posSampleToWrite.moves.size()-1].loc == Board::PASS_LOC)
          posSampleToWrite.weight *= afterPassFactor;
        if(comments.size() > 0 && comments.find("%SAMPLE%") != string::npos)
          posSampleToWrite.weight = std::max(posSampleToWrite.weight,forceSampleWeight);
        if(posSampleToWrite.weight < minWeight)
          return;
        posSampleToWrite.trainingWeight = trainingWeight;
        posWriter.writePos(posSampleToWrite);
        numKept += 1;
        weightKept += posSampleToWrite.weight;
      }
    };

  std::map<string,int64_t> sgfCountUsedByPlayerName;
  std::map<string,int64_t> sgfCountUsedByResult;

  int64_t numExcluded = 0;
  int64_t numSgfsFilteredTopLevel = 0;
  auto trySgf = [&](Sgf* sgf) {
    if(contains(excludeHashes,sgf->hash)) {
      numExcluded += 1;
      return;
    }

    int64_t depth = sgf->depth();
    int64_t nodeCount = sgf->nodeCount();
    int64_t branchCount = sgf->branchCount();
    if(depth > maxDepth || nodeCount > maxNodeCount || branchCount > maxBranchCount) {
      logger.write(
        "Skipping due to violating limits depth " + Global::int64ToString(depth) +
        " nodes " + Global::int64ToString(nodeCount) +
        " branches " + Global::int64ToString(branchCount) +
        " " + sgf->fileName
      );
      numSgfsFilteredTopLevel += 1;
      return;
    }

    try {
      if(!isSgfOkay(sgf)) {
        if(verbosity >= 2)
          logger.write("Filtering due to not okay: " + sgf->fileName);
        numSgfsFilteredTopLevel += 1;
        return;
      }
    }
    catch(const StringError& e) {
      logger.write("Filtering due to error checking okay: " + sgf->fileName + ": " + e.what());
      numSgfsFilteredTopLevel += 1;
      return;
    }

    if(valueFluctuationNNEval == NULL) {
      bool hashParent = false;
      Rand iterRand;
      sgf->iterAllUniquePositions(uniqueHashes, hashComments, hashParent, flipIfPassOrWFirst, allowGameOver, &iterRand, posHandler);
      if(verbosity >= 2)
        logger.write("Handled " + sgf->fileName + " kept weight " + Global::doubleToString(weightKept));
      sgfCountUsedByPlayerName[sgf->getPlayerName(P_BLACK)] += 1;
      sgfCountUsedByPlayerName[sgf->getPlayerName(P_WHITE)] += 1;
      sgfCountUsedByResult[sgf->getRootPropertyWithDefault("RE","")] += 1;
    }
    else {
      string fileName = sgf->fileName;
      CompactSgf compactSgf(sgf);
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules rules = compactSgf.getRulesOrFailAllowUnspecified(Rules::getSimpleTerritory());
      compactSgf.setupInitialBoardAndHist(rules, board, nextPla, hist);

      const bool preventEncore = true;
      const vector<Move>& sgfMoves = compactSgf.moves;

      vector<Board> boards;
      vector<BoardHistory> hists;
      vector<Player> nextPlas;
      vector<shared_ptr<NNOutput>> nnOutputs;
      vector<double> winLossValues;
      vector<Move> moves;

      for(int m = 0; m<sgfMoves.size()+1; m++) {
        MiscNNInputParams nnInputParams;
        nnInputParams.conservativePassAndIsRoot = true;
        NNResultBuf buf;
        bool skipCache = true;
        bool includeOwnerMap = false;
        valueFluctuationNNEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

        boards.push_back(board);
        hists.push_back(hist);
        nextPlas.push_back(nextPla);
        nnOutputs.push_back(std::move(buf.result));
        shared_ptr<NNOutput>& nnOutput = nnOutputs[nnOutputs.size()-1];
        winLossValues.push_back(nnOutput->whiteWinProb - nnOutput->whiteLossProb);

        if(m >= sgfMoves.size())
          break;

        moves.push_back(sgfMoves[m]);

        //Quit out if according to our rules, we already finished the game, or we're somehow in a cleanup phase
        if(hist.isGameFinished || hist.encorePhase > 0)
          break;
        //Quit out if consecutive moves by the same player, to keep the history clean and "normal"
        if(sgfMoves[m].pla != nextPla && m > 0) {
          logger.write("Ending SGF " + fileName + " early due to non-alternating players on turn " + Global::intToString(m));
          break;
        }

        bool suc = hist.isLegal(board,sgfMoves[m].loc,sgfMoves[m].pla);
        if(!suc) {
          //Only log on errors that aren't simply due to ko rules, but quit out regardless
          suc = hist.makeBoardMoveTolerant(board,sgfMoves[m].loc,sgfMoves[m].pla,preventEncore);
          if(!suc)
            logger.write("Illegal move in " + fileName + " turn " + Global::intToString(m) + " move " + Location::toString(sgfMoves[m].loc, board.x_size, board.y_size));
          break;
        }
        hist.makeBoardMoveAssumeLegal(board,sgfMoves[m].loc,sgfMoves[m].pla,NULL,preventEncore);
        nextPla = getOpp(sgfMoves[m].pla);
      }
      boards.push_back(board);
      hists.push_back(hist);
      nextPlas.push_back(nextPla);

      if(winLossValues.size() <= 1)
        return;

      double minTurnNumber = minTurnNumberBoardAreaProp * (hist.initialBoard.x_size * hist.initialBoard.y_size);
      double maxTurnNumber = maxTurnNumberBoardAreaProp * (hist.initialBoard.x_size * hist.initialBoard.y_size);
      vector<double> winrateVariance(winLossValues.size()-1);
      for(int i = 0; i<(int)winrateVariance.size()-1; i++) {
        if(i >= minTurnNumber && i <= maxTurnNumber)
          winrateVariance[i] = (winLossValues[i+1]-winLossValues[i]) * (winLossValues[i+1]-winLossValues[i]);
        else
          winrateVariance[i] = 0.0;
      }
      //Apply exponential blur
      vector<double> winrateVarianceBlurred(winLossValues.size()-1);
      double blurSum = 0.0;
      double totalWeight = 0.0;
      int totalCount = 0;
      for(int i = (int)winrateVarianceBlurred.size()-1; i >= 0; i--) {
        blurSum *= 1.0 - 1.0 / valueFluctuationTurnScale;
        blurSum += winrateVariance[i];
        if(i >= minTurnNumber && i <= maxTurnNumber) {
          winrateVarianceBlurred[i] = blurSum;
          totalWeight += winrateVarianceBlurred[i];
          totalCount += 1;
        }
      }
      if(totalCount <= 0 || totalWeight <= 0)
        return;

      //Normalize
      vector<double> desiredWeight(winLossValues.size()-1);
      for(int i = 0; i<(int)desiredWeight.size(); i++) {
        desiredWeight[i] = winrateVarianceBlurred[i] / totalWeight * totalCount;
      }

      for(int m = 0; m<(int)desiredWeight.size(); m++) {
        assert(m < moves.size());

        Sgf::PositionSample sample;
        const int numMovesToRecord = 8;
        int startIdx = std::max(0,m-numMovesToRecord);
        sample.board = boards[startIdx];
        sample.nextPla = nextPlas[startIdx];
        for(int j = startIdx; j<m; j++)
          sample.moves.push_back(moves[j]);
        sample.initialTurnNumber = startIdx;
        sample.hintLoc = Board::NULL_LOC;
        sample.weight = std::min(valueFluctuationMaxWeight, desiredWeight[m]);
        sample.trainingWeight = trainingWeight;

        if(sample.weight < 0.1)
          continue;

        posHandler(sample, hists[m], "");
        // cout << fileName << " " << m << " " << desiredWeight[m] << endl;
      }

      // Block all the main line hashes and then iterate through the whole SGF to catch side variations, weight them
      // the same way as the same turn in the main line.
      std::set<Hash128> blockedSituationHashes;
      for(int m = 0; m<hists.size(); m++) {
        blockedSituationHashes.insert(
          BoardHistory::getSituationAndSimpleKoAndPrevPosHash(hists[m].getRecentBoard(0),hists[m],hists[m].presumedNextMovePla)
        );
      }

      std::function<void(Sgf::PositionSample&, const BoardHistory&, const string&)> posHandler2 =
        [&blockedSituationHashes, &desiredWeight, &posHandler, trainingWeight](
          Sgf::PositionSample& posSample, const BoardHistory& posHist, const string& comments
        ) {
          // cout << "AAAA " << (posHist.initialTurnNumber + (int)posHist.moveHistory.size()) << endl;
          if(contains(
               blockedSituationHashes,
               BoardHistory::getSituationAndSimpleKoAndPrevPosHash(posHist.getRecentBoard(0),posHist,posHist.presumedNextMovePla)
             ))
            return;
          // cout << "BBBB" << endl;
          Sgf::PositionSample posSampleWeighted = posSample;
          if(desiredWeight.size() > 0) {
            int turnIdx = std::min((int)(desiredWeight.size()-1), posHist.initialTurnNumber + (int)posHist.moveHistory.size());
            turnIdx = std::max(turnIdx,0);
            posSampleWeighted.weight = desiredWeight[turnIdx];
          }
          posSampleWeighted.trainingWeight = trainingWeight;
          posHandler(posSampleWeighted, posHist, comments);
        };

      bool hashParent = false;
      Rand iterRand;
      sgf->iterAllUniquePositions(uniqueHashes, hashComments, hashParent, flipIfPassOrWFirst, allowGameOver, &iterRand, posHandler2);

      if(verbosity >= 2)
        cout << "Handled " << fileName << " kept weight " << weightKept << endl;
      sgfCountUsedByPlayerName[sgf->getPlayerName(P_BLACK)] += 1;
      sgfCountUsedByPlayerName[sgf->getPlayerName(P_WHITE)] += 1;
      sgfCountUsedByResult[sgf->getRootPropertyWithDefault("RE","")] += 1;
    }
  };

  for(size_t i = 0; i<sgfFiles.size(); i++) {
    if(sgfsSet.find(sgfFiles[i]) != sgfsSet.end()) {
      std::vector<Sgf*> sgfs;
      try {
        sgfs = Sgf::loadSgfsFile(sgfFiles[i]);
        for(size_t j = 0; j<sgfs.size(); j++) {
          trySgf(sgfs[j]);
        }
      }
      catch(const StringError& e) {
        logger.write("Invalid SGFS " + sgfFiles[i] + ": " + e.what());
        continue;
      }
      for(size_t j = 0; j<sgfs.size(); j++) {
        delete sgfs[j];
      }
    }
    else {
      Sgf* sgf = NULL;
      try {
        sgf = Sgf::loadFile(sgfFiles[i]);
        trySgf(sgf);
      }
      catch(const StringError& e) {
        logger.write("Invalid SGF " + sgfFiles[i] + ": " + e.what());
      }
      if(sgf != NULL) {
        delete sgf;
      }
    }
  }
  logger.write("Kept " + Global::int64ToString(numKept) + " start positions");
  logger.write("Excluded " + Global::int64ToString(numExcluded) + "/" + Global::uint64ToString(sgfFiles.size()) + " sgf files");
  logger.write("Filtered " + Global::int64ToString(numSgfsFilteredTopLevel) + "/" + Global::uint64ToString(sgfFiles.size()) + " sgf files");
  if(verbosity >= 1) {
    logger.write("SGF count used by player name:");
    for(const auto &elt: sgfCountUsedByPlayerName) {
      logger.write(elt.first + " " + Global::int64ToString(elt.second));
    }
    logger.write("SGF count used by result:");
    for(const auto &elt: sgfCountUsedByResult) {
      logger.write(elt.first + " " + Global::int64ToString(elt.second));
    }
  }

  // ---------------------------------------------------------------------------------------------------

  posWriter.flushAndStop();

  if(valueFluctuationNNEval != NULL)
    delete valueFluctuationNNEval;

  logger.write("All done");

  ScoreValue::freeTables();
  return 0;
}

static bool maybeGetValuesAfterMove(
  Search* search, Loc moveLoc,
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
    search->runWholeSearch(newNextPla,shouldStop);
    search->setParamsNoClearing(oldSearchParams);
  }
  else {
    search->runWholeSearch(newNextPla,shouldStop);
  }

  if(shouldStop.load(std::memory_order_acquire))
    return false;
  values = search->getRootValuesRequireSuccess();
  return true;
}



//We want surprising moves that turned out not poorly
//The more surprising, the more we will weight it
static double surpriseWeight(double policyProb, Rand& rand, bool alwaysAddWeight) {
  if(policyProb < 0)
    return 0;
  double weight = 0.12 / (policyProb + 0.02) - 0.5;
  if(alwaysAddWeight && weight < 1.0)
    weight = 1.0;

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

int MainCmds::dataminesgfs(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  vector<string> sgfDirs;
  vector<string> sgfsDirs;
  string outDir;
  int numProcessThreads;
  vector<string> excludeHashesFiles;
  bool gameMode;
  bool treeMode;
  bool surpriseMode;
  bool autoKomi;
  bool tolerateIllegalMoves;
  int sgfSplitCount;
  int sgfSplitIdx;
  int64_t maxDepth;
  double turnWeightLambda;
  int maxPosesPerOutFile;
  double gameModeFastThreshold;
  bool flipIfPassOrWFirst;
  bool allowGameOver;
  double trainingWeight;

  int minRank;
  int minMinRank;
  string requiredPlayerName;
  int maxHandicap;
  double maxKomi;
  double maxAutoKomi;
  double maxPolicy;
  double minHintWeight;

  try {
    KataGoCommandLine cmd("Search for suprising good moves in sgfs");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> sgfDirArg("","sgfdir","Directory of sgf files",false,"DIR");
    TCLAP::MultiArg<string> sgfsDirArg("","sgfsdir","Directory of sgfs files",false,"DIR");
    TCLAP::ValueArg<string> outDirArg("","outdir","Directory to write results",true,string(),"DIR");
    TCLAP::ValueArg<int> numProcessThreadsArg("","threads","Number of threads",true,1,"THREADS");
    TCLAP::MultiArg<string> excludeHashesArg("","exclude-hashes","Specify a list of hashes to filter out, one per line in a txt file",false,"FILEOF(HASH,HASH)");
    TCLAP::SwitchArg gameModeArg("","game-mode","Game mode");
    TCLAP::SwitchArg treeModeArg("","tree-mode","Tree mode");
    TCLAP::SwitchArg surpriseModeArg("","surprise-mode","Surprise mode");
    TCLAP::SwitchArg autoKomiArg("","auto-komi","Auto komi");
    TCLAP::SwitchArg tolerateIllegalMovesArg("","tolerate-illegal-moves","Tolerate illegal moves");
    TCLAP::ValueArg<int> sgfSplitCountArg("","sgf-split-count","Number of splits",false,1,"N");
    TCLAP::ValueArg<int> sgfSplitIdxArg("","sgf-split-idx","Which split",false,0,"IDX");
    TCLAP::ValueArg<int> maxDepthArg("","max-depth","Max depth allowed for sgf",false,1000000,"INT");
    TCLAP::ValueArg<double> turnWeightLambdaArg("","turn-weight-lambda","Adjust weight for writing down each position",false,0.0,"LAMBDA");
    TCLAP::ValueArg<int> maxPosesPerOutFileArg("","max-poses-per-out-file","Number of hintposes per output file",false,100000,"INT");
    TCLAP::ValueArg<double> gameModeFastThresholdArg("","game-mode-fast-threshold","Utility threshold for game mode fast pass",false,0.005,"UTILS");
    TCLAP::SwitchArg flipIfPassOrWFirstArg("","flip-if-pass","Try to heuristically find cases where an sgf passes to simulate white<->black");
    TCLAP::SwitchArg allowGameOverArg("","allow-game-over","Allow sampling game over positions in sgf");
    TCLAP::ValueArg<double> trainingWeightArg("","training-weight","Scale the loss function weight from data from games that originate from this position",false,1.0,"WEIGHT");
    TCLAP::ValueArg<int> minRankArg("","min-rank","Require player making the move to have rank at least this",false,Sgf::RANK_UNKNOWN,"INT");
    TCLAP::ValueArg<int> minMinRankArg("","min-min-rank","Require both players in a game to have rank at least this",false,Sgf::RANK_UNKNOWN,"INT");
    TCLAP::ValueArg<string> requiredPlayerNameArg("","required-player-name","Require player making the move to have this name",false,string(),"NAME");
    TCLAP::ValueArg<int> maxHandicapArg("","max-handicap","Require no more than this big handicap in stones",false,100,"INT");
    TCLAP::ValueArg<double> maxKomiArg("","max-komi","Require absolute value of game komi to be at most this",false,1000,"KOMI");
    TCLAP::ValueArg<double> maxAutoKomiArg("","max-auto-komi","If absolute value of auto komi would exceed this, skip position",false,1000,"KOMI");
    TCLAP::ValueArg<double> maxPolicyArg("","max-policy","Chop off moves with raw policy more than this",false,1000,"POLICY");
    TCLAP::ValueArg<double> minHintWeightArg("","min-hint-weight","Hinted moves get at least this weight",false,0.0,"WEIGHT");
    cmd.add(sgfDirArg);
    cmd.add(sgfsDirArg);
    cmd.add(outDirArg);
    cmd.add(numProcessThreadsArg);
    cmd.add(excludeHashesArg);
    cmd.add(gameModeArg);
    cmd.add(treeModeArg);
    cmd.add(surpriseModeArg);
    cmd.add(autoKomiArg);
    cmd.add(tolerateIllegalMovesArg);
    cmd.add(sgfSplitCountArg);
    cmd.add(sgfSplitIdxArg);
    cmd.add(maxDepthArg);
    cmd.add(turnWeightLambdaArg);
    cmd.add(maxPosesPerOutFileArg);
    cmd.add(gameModeFastThresholdArg);
    cmd.add(flipIfPassOrWFirstArg);
    cmd.add(allowGameOverArg);
    cmd.add(trainingWeightArg);
    cmd.add(minRankArg);
    cmd.add(minMinRankArg);
    cmd.add(requiredPlayerNameArg);
    cmd.add(maxHandicapArg);
    cmd.add(maxKomiArg);
    cmd.add(maxAutoKomiArg);
    cmd.add(maxPolicyArg);
    cmd.add(minHintWeightArg);
    cmd.parseArgs(args);

    nnModelFile = cmd.getModelFile();
    sgfDirs = sgfDirArg.getValue();
    sgfsDirs = sgfsDirArg.getValue();
    outDir = outDirArg.getValue();
    numProcessThreads = numProcessThreadsArg.getValue();
    excludeHashesFiles = excludeHashesArg.getValue();
    gameMode = gameModeArg.getValue();
    treeMode = treeModeArg.getValue();
    surpriseMode = surpriseModeArg.getValue();
    autoKomi = autoKomiArg.getValue();
    tolerateIllegalMoves = tolerateIllegalMovesArg.getValue();
    sgfSplitCount = sgfSplitCountArg.getValue();
    sgfSplitIdx = sgfSplitIdxArg.getValue();
    maxDepth = maxDepthArg.getValue();
    turnWeightLambda = turnWeightLambdaArg.getValue();
    maxPosesPerOutFile = maxPosesPerOutFileArg.getValue();
    gameModeFastThreshold = gameModeFastThresholdArg.getValue();
    flipIfPassOrWFirst = flipIfPassOrWFirstArg.getValue();
    allowGameOver = allowGameOverArg.getValue();
    trainingWeight = trainingWeightArg.getValue();
    minRank = minRankArg.getValue();
    minMinRank = minMinRankArg.getValue();
    requiredPlayerName = requiredPlayerNameArg.getValue();
    maxHandicap = maxHandicapArg.getValue();
    maxKomi = maxKomiArg.getValue();
    maxAutoKomi = maxAutoKomiArg.getValue();
    maxPolicy = maxPolicyArg.getValue();
    minHintWeight = minHintWeightArg.getValue();

    if((int)gameMode + (int)treeMode + (int)surpriseMode != 1)
      throw StringError("Must specify either -game-mode or -tree-mode or -surprise-mode");

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  MakeDir::make(outDir);

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.addFile(outDir + "/" + "log.log");
  for(const string& arg: args)
    logger.write(string("Command: ") + arg);
  logger.write("Git revision " + Version::getGitRevision());

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
    const int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    const int expectedConcurrentEvals = params.numThreads;
    const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  GameInitializer* gameInit = new GameInitializer(cfg,logger);
  cfg.warnUnusedKeys(cerr,&logger);

  vector<string> sgfFiles;
  FileHelpers::collectSgfsFromDirsOrFiles(sgfDirs,sgfFiles);
  logger.write("Found " + Global::int64ToString((int64_t)sgfFiles.size()) + " sgf files!");

  std::set<string> sgfsSet;
  {
    vector<string> sgfsFiles;
    FileHelpers::collectMultiSgfsFromDirsOrFiles(sgfsDirs,sgfsFiles);
    logger.write("Found " + Global::int64ToString((int64_t)sgfsFiles.size()) + " sgfs files!");
    for(const string& s: sgfsFiles) {
      sgfFiles.push_back(s);
      sgfsSet.insert(s);
    }
  }

  vector<size_t> permutation(sgfFiles.size());
  for(size_t i = 0; i<sgfFiles.size(); i++)
    permutation[i] = i;
  for(size_t i = 1; i<sgfFiles.size(); i++) {
    size_t r = (size_t)seedRand.nextUInt64(i+1);
    std::swap(permutation[i],permutation[r]);
  }

  set<Hash128> excludeHashes = Sgf::readExcludes(excludeHashesFiles);
  logger.write("Loaded " + Global::uint64ToString(excludeHashes.size()) + " excludes");


  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  // ---------------------------------------------------------------------------------------------------
  PosWriter posWriter("hintposes.txt", outDir, sgfSplitCount, sgfSplitIdx, maxPosesPerOutFile);

  //COMMON ---------------------------------------------------------------------------------------------------
  std::atomic<int64_t> numSgfsDone(0);
  std::atomic<int64_t> numFilteredIndivdualPoses(0);
  std::atomic<int64_t> numFilteredSgfs(0);

  auto isPlayerOkay = [&](const Sgf* sgf, Player pla) {
    if(minRank != Sgf::RANK_UNKNOWN) {
      if(sgf->getRank(pla) < minRank)
        return false;
    }
    if(requiredPlayerName != "") {
      if(sgf->getPlayerName(pla) != requiredPlayerName)
        return false;
    }
    return true;
  };

  auto isSgfOkay = [&](const Sgf* sgf) {
    if(maxHandicap < 100 && sgf->getHandicapValue() > maxHandicap)
      return false;
    if(sgf->depth() > maxDepth)
      return false;
    if(std::fabs(sgf->getKomi()) > maxKomi)
      return false;
    if(minMinRank != Sgf::RANK_UNKNOWN) {
      if(sgf->getRank(P_BLACK) < minMinRank && sgf->getRank(P_WHITE) < minMinRank)
        return false;
    }
    if(!isPlayerOkay(sgf,P_BLACK) && !isPlayerOkay(sgf,P_WHITE))
      return false;
    return true;
  };

  auto expensiveEvaluateMove = [&posWriter,&turnWeightLambda,&maxAutoKomi,&maxHandicap,&numFilteredIndivdualPoses,&surpriseMode,&minHintWeight,&logger,trainingWeight](
    Search* search, Loc missedLoc,
    Player nextPla, const Board& board, const BoardHistory& hist,
    const Sgf::PositionSample& sample, bool markedAsHintPos
  ) {
    if(shouldStop.load(std::memory_order_acquire))
      return;

    if(std::fabs(hist.rules.komi) > maxAutoKomi) {
      numFilteredIndivdualPoses.fetch_add(1);
      return;
    }
    if(hist.computeNumHandicapStones() > maxHandicap) {
      numFilteredIndivdualPoses.fetch_add(1);
      return;
    }

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

    if(surpriseMode) {
      // TODO Very simple logic - If a full search gives a different move than a quick search and
      // judges the move to be way better than the quick search's move, then record as a hintpos.
      // If a full search gives a very worse value than a quick search, then record as a sample position.

      ReportedSearchValues veryQuickValues;
      {
        bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0/50.0,veryQuickValues);
        if(!suc)
          return;
      }
      Loc veryQuickMoveLoc = search->getChosenMoveLoc();
      ReportedSearchValues baseValues;
      {
        bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0,baseValues);
        if(!suc)
          return;
      }
      Loc moveLoc = search->getChosenMoveLoc();

      if(moveLoc != veryQuickMoveLoc) {
        ReportedSearchValues veryQuickAfterMoveValues;
        {
          bool suc = maybeGetValuesAfterMove(search,veryQuickMoveLoc,nextPla,board,hist,1.0/2.0,veryQuickAfterMoveValues);
          if(!suc)
            return;
        }
        ReportedSearchValues baseAfterMoveValues;
        {
          bool suc = maybeGetValuesAfterMove(search,moveLoc,nextPla,board,hist,1.0/2.0,baseAfterMoveValues);
          if(!suc)
            return;
        }
        if(
          (nextPla == P_WHITE && baseAfterMoveValues.utility - veryQuickAfterMoveValues.utility > 0.2) ||
          (nextPla == P_BLACK && baseAfterMoveValues.utility - veryQuickAfterMoveValues.utility < -0.2)
        ) {
          Sgf::PositionSample sampleToWrite = sample;
          sampleToWrite.weight += std::fabs(baseValues.utility - veryQuickValues.utility);
          sampleToWrite.hintLoc = moveLoc;
          sampleToWrite.trainingWeight = trainingWeight;
          posWriter.writePos(sampleToWrite);
          if(sampleToWrite.hasPreviousPositions(1))
            posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.5));
          if(sampleToWrite.hasPreviousPositions(2))
            posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.25).previousPosition(sampleToWrite.weight * 0.25));
          logger.write("Surprising good " + Global::doubleToString(sampleToWrite.weight));
          return;
        }
      }

      if(
        (nextPla == P_WHITE && baseValues.utility - veryQuickValues.utility < -0.2) ||
        (nextPla == P_BLACK && baseValues.utility - veryQuickValues.utility > 0.2)
      ) {
        Sgf::PositionSample sampleToWrite = sample;
        sampleToWrite.weight = 1.0 + std::fabs(baseValues.utility - veryQuickValues.utility);
        sampleToWrite.hintLoc = Board::NULL_LOC;
        sampleToWrite.trainingWeight = trainingWeight;
        posWriter.writePos(sampleToWrite);
        if(sampleToWrite.hasPreviousPositions(1))
          posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.5));
        if(sampleToWrite.hasPreviousPositions(2))
          posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.25).previousPosition(sampleToWrite.weight * 0.25));
        logger.write("Inevitable bad " + Global::doubleToString(sampleToWrite.weight));
        return;
      }
      if(
        (nextPla == P_WHITE && baseValues.utility - veryQuickValues.utility > 0.2) ||
        (nextPla == P_BLACK && baseValues.utility - veryQuickValues.utility < -0.2)
      ) {
        Sgf::PositionSample sampleToWrite = sample;
        sampleToWrite.weight = 1.0 + std::fabs(baseValues.utility - veryQuickValues.utility);
        sampleToWrite.hintLoc = Board::NULL_LOC;
        sampleToWrite.trainingWeight = trainingWeight;
        posWriter.writePos(sampleToWrite);
        if(sampleToWrite.hasPreviousPositions(1))
          posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.5));
        if(sampleToWrite.hasPreviousPositions(2))
          posWriter.writePos(sampleToWrite.previousPosition(sampleToWrite.weight * 0.25).previousPosition(sampleToWrite.weight * 0.25));
        logger.write("Inevitable good " + Global::doubleToString(sampleToWrite.weight));
        return;
      }
      return;
    }

    ReportedSearchValues veryQuickValues;
    {
      bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0/25.0,veryQuickValues);
      if(!suc)
        return;
    }
    Loc veryQuickMoveLoc = search->getChosenMoveLoc();

    ReportedSearchValues quickValues;
    {
      bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0/5.0,quickValues);
      if(!suc)
        return;
    }
    Loc quickMoveLoc = search->getChosenMoveLoc();

    ReportedSearchValues baseValues;
    {
      bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0,baseValues);
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
    sampleToWrite.trainingWeight = trainingWeight;
    sampleToWrite.weight += std::fabs(baseValues.utility - quickValues.utility);
    sampleToWrite.weight += std::fabs(baseValues.utility - veryQuickValues.utility);

    //Bot DOES see the move?
    if(moveLoc == missedLoc) {
      if(quickMoveLoc == moveLoc)
        sampleToWrite.weight = sampleToWrite.weight * 0.75 - 0.1;
      if(veryQuickMoveLoc == moveLoc)
        sampleToWrite.weight = sampleToWrite.weight * 0.75 - 0.1;

      sampleToWrite.weight *= exp(-sampleToWrite.initialTurnNumber * turnWeightLambda);

      if(sampleToWrite.weight < minHintWeight && markedAsHintPos)
        sampleToWrite.weight = minHintWeight;
      if(sampleToWrite.weight > 0.1) {
        //Still good to learn from given that policy was really low
        posWriter.writePos(sampleToWrite);
      }
    }

    //Bot doesn't see the move?
    else if(moveLoc != missedLoc) {

      //If marked as a hint pos, always trust that it should be better and add it.
      bool shouldWriteMove = markedAsHintPos;

      if(!shouldWriteMove) {
        ReportedSearchValues moveValues;
        if(!maybeGetValuesAfterMove(search,moveLoc,nextPla,board,hist,1.0,moveValues))
          return;
        // ostringstream out0;
        // out0 << "BOT MOVE " << Location::toString(moveLoc,board) << endl;
        // search->printTree(out0, search->rootNode, PrintTreeOptions().maxDepth(0),perspective);
        // cout << out0.str() << endl;

        ReportedSearchValues missedValues;
        if(!maybeGetValuesAfterMove(search,missedLoc,nextPla,board,hist,1.0,missedValues))
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
        if(sampleToWrite.weight < minHintWeight && markedAsHintPos)
          sampleToWrite.weight = minHintWeight;
        if(sampleToWrite.weight > 0.1) {
          posWriter.writePos(sampleToWrite);
        }
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------
  //SGF MODE

  auto processSgfGame = [&logger,&gameInit,&nnEval,&expensiveEvaluateMove,autoKomi,&gameModeFastThreshold,&maxDepth,&numFilteredSgfs,&maxHandicap,&maxPolicy,allowGameOver,trainingWeight](
    Search* search, Rand& rand, const string& fileName, CompactSgf* sgf, bool blackOkay, bool whiteOkay
  ) {
    //Don't use the SGF rules - randomize them for a bit more entropy
    Rules rules = gameInit->createRules();

    Board board;
    Player nextPla;
    BoardHistory hist;
    sgf->setupInitialBoardAndHist(rules, board, nextPla, hist);
    if(!gameInit->isAllowedBSize(board.x_size,board.y_size)) {
      numFilteredSgfs.fetch_add(1);
      return;
    }
    if(board.x_size != 19 || board.y_size != 19) {
      numFilteredSgfs.fetch_add(1);
      return;
    }

    const bool preventEncore = true;
    const vector<Move>& sgfMoves = sgf->moves;

    if((int64_t)sgfMoves.size() > maxDepth) {
      numFilteredSgfs.fetch_add(1);
      return;
    }
    if(hist.computeNumHandicapStones() > maxHandicap) {
      numFilteredSgfs.fetch_add(1);
      return;
    }

    vector<Board> boards;
    vector<BoardHistory> hists;
    vector<Player> nextPlas;
    vector<shared_ptr<NNOutput>> nnOutputs;
    vector<double> winLossValues;
    vector<double> scoreLeads;

    vector<Move> moves;
    vector<double> policyPriors;

    for(int m = 0; m<sgfMoves.size()+1; m++) {
      MiscNNInputParams nnInputParams;
      NNResultBuf buf;
      bool skipCache = true; //Always ignore cache so that we get more entropy on repeated board positions due to symmetries
      bool includeOwnerMap = false;
      nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

      ReportedSearchValues superQuickValues;
      {
        bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,nextPla,board,hist,1.0/80.0,superQuickValues);
        if(!suc)
          break;
      }

      boards.push_back(board);
      hists.push_back(hist);
      nextPlas.push_back(nextPla);
      nnOutputs.push_back(std::move(buf.result));

      shared_ptr<NNOutput>& nnOutput = nnOutputs[nnOutputs.size()-1];

      winLossValues.push_back(superQuickValues.winLossValue);
      scoreLeads.push_back(superQuickValues.lead);

      if(m < sgfMoves.size()) {
        moves.push_back(sgfMoves[m]);
        int pos = NNPos::locToPos(sgfMoves[m].loc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
        policyPriors.push_back(nnOutput->policyProbs[pos]);
      }

      if(m >= sgfMoves.size())
        break;

      //Quit out if according to our rules, we already finished the game, or we're somehow in a cleanup phase
      if(!allowGameOver && (hist.isGameFinished || hist.encorePhase > 0))
        break;

      //Quit out if consecutive moves by the same player, to keep the history clean and "normal"
      if(sgfMoves[m].pla != nextPla && m > 0) {
        logger.write("Ending SGF " + fileName + " early due to non-alternating players on turn " + Global::intToString(m));
        break;
      }

      bool suc = hist.isLegal(board,sgfMoves[m].loc,sgfMoves[m].pla);
      if(!suc) {
        //Only log on errors that aren't simply due to ko rules, but quit out regardless
        suc = hist.makeBoardMoveTolerant(board,sgfMoves[m].loc,sgfMoves[m].pla,preventEncore);
        if(!suc)
          logger.write("Illegal move in " + fileName + " turn " + Global::intToString(m) + " move " + Location::toString(sgfMoves[m].loc, board.x_size, board.y_size));
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
    futureValue[winLossValues.size()] = winLossValues[winLossValues.size()-1];
    futureLead[winLossValues.size()] = scoreLeads[winLossValues.size()];
    for(int i = (int)winLossValues.size()-1; i >= 0; i--) {
      futureValue[i] = 0.10 * winLossValues[i] + 0.90 * futureValue[i+1];
      futureLead[i] = 0.10 * scoreLeads[i] + 0.90 * futureLead[i+1];
    }
    pastValue[0] = winLossValues[0];
    pastLead[0] = scoreLeads[0];
    for(int i = 1; i<(int)winLossValues.size(); i++) {
      pastValue[i] = 0.5 * winLossValues[i] + 0.5 * pastValue[i+1];
      pastLead[i] = 0.5 * scoreLeads[i] + 0.5 * pastLead[i+1];
    }

    const double scoreLeadWeight = 0.01;
    const double sumThreshold = gameModeFastThreshold;

    //cout << fileName << endl;
    for(int m = 0; m<moves.size(); m++) {

      if(shouldStop.load(std::memory_order_acquire))
        break;

      if((nextPlas[m] == P_BLACK && !blackOkay) || (nextPlas[m] == P_WHITE && !whiteOkay))
        continue;

      //cout << m << endl;
      //Look for surprising moves that turned out not poorly
      //The more surprising, the more times we will write it out.
      if(policyPriors[m] > maxPolicy)
        continue;
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
      const int numMovesToRecord = 8;
      int startIdx = std::max(0,m-numMovesToRecord);
      sample.board = boards[startIdx];
      sample.nextPla = nextPlas[startIdx];
      for(int j = startIdx; j<m; j++)
        sample.moves.push_back(moves[j]);
      sample.initialTurnNumber = startIdx;
      sample.hintLoc = moves[m].loc;
      sample.weight = weight;
      sample.trainingWeight = trainingWeight;

      if(autoKomi) {
        const int64_t numVisits = 10;
        OtherGameProperties props;
        PlayUtils::adjustKomiToEven(search,NULL,boards[m],hists[m],nextPlas[m],numVisits,props,rand);
      }

      expensiveEvaluateMove(
        search, moves[m].loc, nextPlas[m], boards[m], hists[m],
        sample, false
      );
    }
  };

  const int maxSgfQueueSize = 1024;
  ThreadSafeQueue<Sgf*> sgfQueue(maxSgfQueueSize);
  auto processSgfLoop = [&logger,&processSgfGame,&sgfQueue,&params,&nnEval,&numSgfsDone,&isPlayerOkay,&tolerateIllegalMoves]() {
    Rand rand;
    string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
    Search* search = new Search(params,nnEval,&logger,searchRandSeed);

    while(true) {
      if(shouldStop.load(std::memory_order_acquire))
        break;

      Sgf* sgfRaw;
      bool success = sgfQueue.waitPop(sgfRaw);
      if(!success)
        break;

      bool blackOkay = isPlayerOkay(sgfRaw,P_BLACK);
      bool whiteOkay = isPlayerOkay(sgfRaw,P_WHITE);

      CompactSgf* sgf = NULL;
      try {
        sgf = new CompactSgf(sgfRaw);
      }
      catch(const StringError& e) {
        if(!tolerateIllegalMoves)
          throw;
        else {
          logger.write(e.what());
        }
      }
      if(sgf != NULL)
        processSgfGame(search,rand,sgf->fileName,sgf,blackOkay,whiteOkay);

      numSgfsDone.fetch_add(1);
      delete sgf;
      delete sgfRaw;
    }
    delete search;
  };



  // ---------------------------------------------------------------------------------------------------
  //TREE MODE

  auto treePosHandler = [&gameInit,&nnEval,&expensiveEvaluateMove,&autoKomi,&maxPolicy,&flipIfPassOrWFirst,&surpriseMode,trainingWeight](
    Search* search, Rand& rand, const BoardHistory& treeHist, int initialTurnNumber, bool markedAsHintPos
  ) {
    if(shouldStop.load(std::memory_order_acquire))
      return;
    if(treeHist.moveHistory.size() > 0x3FFFFFFF)
      throw StringError("Too many moves in history");
    int moveHistorySize = (int)treeHist.moveHistory.size();
    if(moveHistorySize <= 0)
      return;

    //Snap the position 8 turns ago so as to include 8 moves of history.
    int turnsAgoToSnap = 0;
    while(turnsAgoToSnap < 8) {
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
    sample.trainingWeight = trainingWeight;

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

    //And make sure it's legal under our randomized rules.
    if(!hist.isLegal(board,sample.hintLoc,pla))
      return;

    if(autoKomi) {
      const int64_t numVisits = 10;
      OtherGameProperties props;
      PlayUtils::adjustKomiToEven(search,NULL,board,hist,pla,numVisits,props,rand);
    }

    MiscNNInputParams nnInputParams;
    bool skipCache = true; //Always ignore cache so that we get more entropy on repeated board positions due to symmetries
    bool includeOwnerMap = false;

    double policyProb = 0.0;
    {
      // Take 1.1 * the geometric mean of a few samples, so as to greatly upweight the importance of anomalous low values.
      double acc = 0.0;
      int count = 0;
      for(int samples = 0; samples < 4; samples++) {
        NNResultBuf buf;
        nnEval->evaluate(board,hist,pla,nnInputParams,buf,skipCache,includeOwnerMap);
        shared_ptr<NNOutput>& nnOutput = buf.result;
        int pos = NNPos::locToPos(sample.hintLoc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
        double prob = nnOutput->policyProbs[pos];
        assert(prob >= 0.0);
        acc += log(prob + 1e-30);
        count += 1;
      }
      assert(count > 0);
      policyProb = 1.1 * exp(acc / count);
    }

    if(policyProb > maxPolicy)
      return;
    bool alwaysAddWeight = markedAsHintPos || surpriseMode;
    double weight = surpriseWeight(policyProb,rand,alwaysAddWeight);
    if(weight <= 0)
      return;
    sample.weight = weight;

    if(flipIfPassOrWFirst) {
      if(treeHist.hasBlackPassOrWhiteFirst())
        sample = sample.getColorFlipped();
    }

    expensiveEvaluateMove(
      search, sample.hintLoc, pla, board, hist,
      sample, markedAsHintPos
    );
  };


  const int64_t maxPosQueueSize = 16384;
  ThreadSafeQueue<PosQueueEntry> posQueue(maxPosQueueSize);
  std::atomic<int64_t> numPosesBegun(0);
  std::atomic<int64_t> numPosesDone(0);
  std::atomic<int64_t> numPosesEnqueued(0);

  auto processPosLoop = [&logger,&posQueue,&params,&numPosesBegun,&numPosesDone,&numPosesEnqueued,&nnEval,&treePosHandler]() {
    Rand rand;
    string searchRandSeed = Global::uint64ToString(rand.nextUInt64());
    Search* search = new Search(params,nnEval,&logger,searchRandSeed);

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

      int64_t numEnqueued = numPosesEnqueued.load();
      int64_t numBegun = 1+numPosesBegun.fetch_add(1);
      if(numBegun % 20 == 0)
        logger.write("Begun " + Global::int64ToString(numBegun) + "/" + Global::int64ToString(numEnqueued) + " poses");

      treePosHandler(search, rand, *hist, initialTurnNumber, markedAsHintPos);

      int64_t numDone = 1+numPosesDone.fetch_add(1);
      if(numDone % 20 == 0)
        logger.write("Done " + Global::int64ToString(numDone) + "/" + Global::int64ToString(numEnqueued) + " poses");

      delete hist;
    }
    delete search;
    posQueue.setReadOnly();
  };


  // ---------------------------------------------------------------------------------------------------

  //Begin writing
  posWriter.start();

  vector<std::thread> threads;
  for(int i = 0; i<numProcessThreads; i++) {
    if(gameMode)
      threads.push_back(std::thread(processSgfLoop));
    else if(treeMode)
      threads.push_back(std::thread(processPosLoop));
    else if(surpriseMode)
      threads.push_back(std::thread(processPosLoop));
  }

  // ---------------------------------------------------------------------------------------------------

  int64_t numSgfsBegun = 0;
  int64_t numSgfsSkipped = 0;
  int64_t numSgfsFilteredTopLevel = 0;

  std::set<Hash128> uniqueHashes;

  auto logSgfProgress = [&]() {
    logger.write(
      "Begun " + Global::int64ToString(numSgfsBegun) + " / " + Global::int64ToString(sgfFiles.size()) + " sgfs, " +
      string("done ") + Global::int64ToString(numSgfsDone.load()) + " sgfs, " +
      string("skipped ") + Global::int64ToString(numSgfsSkipped) + " sgfs, " +
      string("filtered ") + Global::int64ToString(numSgfsFilteredTopLevel + numFilteredSgfs.load()) + " sgfs, " +
      string("filtered ") + Global::int64ToString(numFilteredIndivdualPoses.load()) + " individual poses"
    );
  };

  for(size_t i = 0; i<sgfFiles.size(); i++) {
    numSgfsBegun += 1;
    if(numSgfsBegun % std::min((size_t)20, 1 + sgfFiles.size() / 60) == 0)
      logSgfProgress();

    const string& fileName = sgfFiles[permutation[i]];

    std::vector<Sgf*> sgfs;
    if(sgfsSet.find(fileName) != sgfsSet.end()) {
      try {
        sgfs = Sgf::loadSgfsFile(fileName);
      }
      catch(const StringError& e) {
        logger.write("Invalid SGFS " + fileName + ": " + e.what());
        continue;
      }
    }
    else {
      Sgf* sgf = NULL;
      try {
        sgf = Sgf::loadFile(fileName);
      }
      catch(const StringError& e) {
        logger.write("Invalid SGF " + fileName + ": " + e.what());
        continue;
      }
      sgfs.push_back(sgf);
    }

    vector<size_t> subPermutation(sgfs.size());
    for(size_t j = 0; j<sgfs.size(); j++)
      subPermutation[j] = j;
    for(size_t j = 1; j<sgfs.size(); j++) {
      size_t r = (size_t)seedRand.nextUInt64(j+1);
      std::swap(subPermutation[j],subPermutation[r]);
    }

    for(size_t j = 0; j<sgfs.size(); j++) {
      Sgf* sgf = sgfs[subPermutation[j]];

      if(contains(excludeHashes,sgf->hash)) {
        logger.write("Filtering due to exclude: " + fileName);
        numSgfsFilteredTopLevel += 1;
        delete sgf;
        continue;
      }
      try {
        if(!isSgfOkay(sgf)) {
          logger.write("Filtering due to not okay: " + fileName);
          numSgfsFilteredTopLevel += 1;
          delete sgf;
          continue;
        }
      }
      catch(const StringError& e) {
        logger.write("Filtering due to error checking okay: " + fileName + ": " + e.what());
        numSgfsFilteredTopLevel += 1;
        delete sgf;
        continue;
      }
      if(sgfSplitCount > 1 && ((int)(sgf->hash.hash0 & 0x7FFFFFFF) % sgfSplitCount) != sgfSplitIdx) {
        numSgfsSkipped += 1;
        delete sgf;
        continue;
      }

      logger.write("Starting " + fileName);

      if(gameMode) {
        sgfQueue.waitPush(sgf);
      }
      else {
        bool hashComments = true; //Hash comments so that if we see a position without %HINT% and one with, we make sure to re-load it.
        bool blackOkay = isPlayerOkay(sgf,P_BLACK);
        bool whiteOkay = isPlayerOkay(sgf,P_WHITE);
        try {
          bool hashParent = true; //Hash parent so that we distinguish hint moves that reach the same position but were different moves from different starting states.
          sgf->iterAllUniquePositions(
            uniqueHashes, hashComments, hashParent, flipIfPassOrWFirst, allowGameOver, &seedRand,
            [&](Sgf::PositionSample& unusedSample, const BoardHistory& hist, const string& comments) {
              if(comments.size() > 0 && comments.find("%NOHINT%") != string::npos)
                return;
              if(hist.moveHistory.size() <= 0)
                return;
              int hintIdx = (int)hist.moveHistory.size()-1;
              if((hist.moveHistory[hintIdx].pla == P_BLACK && !blackOkay) || (hist.moveHistory[hintIdx].pla == P_WHITE && !whiteOkay))
                return;

              //unusedSample doesn't have enough history, doesn't have hintloc the way we want it
              int64_t numEnqueued = 1+numPosesEnqueued.fetch_add(1);
              if(numEnqueued % 500 == 0)
                logger.write("Enqueued " + Global::int64ToString(numEnqueued) + " poses");
              PosQueueEntry entry;
              entry.hist = new BoardHistory(hist);
              entry.initialTurnNumber = unusedSample.initialTurnNumber; //this is the only thing we keep
              entry.markedAsHintPos = (comments.size() > 0 && comments.find("%HINT%") != string::npos);
              posQueue.waitPush(entry);
            }
          );
        }
        catch(const StringError& e) {
          if(!tolerateIllegalMoves)
            throw;
          else
            logger.write(e.what());
        }
        numSgfsDone.fetch_add(1);
        delete sgf;
      }
    }
  }
  logSgfProgress();
  logger.write("All sgfs loaded, waiting for finishing analysis");
  logger.write(Global::uint64ToString(sgfQueue.size()) + " sgfs still enqueued");
  logger.write(Global::uint64ToString(sgfQueue.size()) + " sgfs still enqueued");

  sgfQueue.setReadOnly();
  posQueue.setReadOnly();
  for(size_t i = 0; i<threads.size(); i++)
    threads[i].join();

  logSgfProgress();
  logger.write("Waiting for final writing and cleanup");

  posWriter.flushAndStop();

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






int MainCmds::trystartposes(const vector<string>& args) {
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
    cmd.parseArgs(args);
    nnModelFile = cmd.getModelFile();
    startPosesFiles = startPosesFileArg.getValue();
    minWeight = minWeightArg.getValue();
    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

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
    const int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    const int expectedConcurrentEvals = params.numThreads;
    const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }
  logger.write("Loaded neural net");

  vector<Sgf::PositionSample> startPoses;
  for(size_t i = 0; i<startPosesFiles.size(); i++) {
    const string& startPosesFile = startPosesFiles[i];
    vector<string> lines = FileUtils::readFileLines(startPosesFile,'\n');
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
  Search* search = new Search(params,nnEval,&logger,searchRandSeed);

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
      PlayUtils::adjustKomiToEven(search,NULL,board,hist,pla,numVisits,props,seedRand);
    }

    Loc hintLoc = startPos.hintLoc;

    {
      ReportedSearchValues values;
      bool suc = maybeGetValuesAfterMove(search,Board::NULL_LOC,pla,board,hist,1.0,values);
      (void)suc;
      assert(suc);
      cout << "Searching startpos: " << "\n";
      cout << "Weight: " << startPos.weight << "\n";
      cout << "Training Weight: " << startPos.trainingWeight << "\n";
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
        bool suc = maybeGetValuesAfterMove(search,hintLoc,pla,board,hist,1.0,values);
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


int MainCmds::viewstartposes(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  vector<string> startPosesFiles;
  double minWeight;
  int idxToView;
  try {
    KataGoCommandLine cmd("View startposes");
    cmd.addConfigFileArg("","",false);
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::MultiArg<string> startPosesFileArg("","start-poses-file","Startposes file",true,"DIR");
    TCLAP::ValueArg<double> minWeightArg("","min-weight","Min weight of startpos to view",false,0.0,"WEIGHT");
    TCLAP::ValueArg<int> idxArg("","idx","Index of startpos to view in file",false,-1,"IDX");
    cmd.add(startPosesFileArg);
    cmd.add(minWeightArg);
    cmd.add(idxArg);
    cmd.parseArgs(args);
    startPosesFiles = startPosesFileArg.getValue();
    minWeight = minWeightArg.getValue();
    idxToView = idxArg.getValue();

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  Rules rules;
  AsyncBot* bot = NULL;
  NNEvaluator* nnEval = NULL;
  if(cfg.getFileName() != "") {
    const bool loadKomiFromCfg = false;
    rules = Setup::loadSingleRules(cfg,loadKomiFromCfg);
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    {
      Setup::initializeSession(cfg);
      const int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
      const int expectedConcurrentEvals = params.numThreads;
      const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
      const bool defaultRequireExactNNLen = false;
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
        Board::MAX_LEN,Board::MAX_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
    logger.write("Loaded neural net");

    string searchRandSeed;
    if(cfg.contains("searchRandSeed"))
      searchRandSeed = cfg.getString("searchRandSeed");
    else
      searchRandSeed = Global::uint64ToString(rand.nextUInt64());

    bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);
  }

  vector<Sgf::PositionSample> startPoses;
  for(size_t i = 0; i<startPosesFiles.size(); i++) {
    const string& startPosesFile = startPosesFiles[i];
    vector<string> lines = FileUtils::readFileLines(startPosesFile,'\n');
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
    if(startPos.weight < minWeight)
      continue;
    if(idxToView >= 0 && s != idxToView)
      continue;

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

    Loc hintLoc = startPos.hintLoc;
    cout << "StartPos: " << s << "/" << startPoses.size() << "\n";
    cout << "Next pla: " << PlayerIO::playerToString(pla) << "\n";
    cout << "Weight: " << startPos.weight << "\n";
    cout << "TrainingWeight: " << startPos.trainingWeight << "\n";
    cout << "HintLoc: " << Location::toString(hintLoc,board) << "\n";
    Board::printBoard(cout, board, hintLoc, &(hist.moveHistory));
    cout << endl;

    bool autoKomi = true;
    if(autoKomi && bot != NULL) {
      const int64_t numVisits = 10;
      OtherGameProperties props;
      PlayUtils::adjustKomiToEven(bot->getSearchStopAndWait(),NULL,board,hist,pla,numVisits,props,rand);
    }

    if(bot != NULL) {
      bot->setPosition(pla,board,hist);
      if(hintLoc != Board::NULL_LOC)
        bot->setRootHintLoc(hintLoc);
      else
        bot->setRootHintLoc(Board::NULL_LOC);
      bot->genMoveSynchronous(bot->getSearch()->rootPla,TimeControls());
      const Search* search = bot->getSearchStopAndWait();
      PrintTreeOptions options;
      Player perspective = P_WHITE;
      search->printTree(cout, search->rootNode, options, perspective);
    }
  }

  if(bot != NULL)
    delete bot;
  if(nnEval != NULL)
    delete nnEval;

  ScoreValue::freeTables();
  return 0;
}


int MainCmds::sampleinitializations(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  int numToGen;
  bool evaluate;
  try {
    KataGoCommandLine cmd("View startposes");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numToGenArg("","num","Num to gen",false,1,"N");
    TCLAP::SwitchArg evaluateArg("","evaluate","Print out values and scores on the inited poses");
    cmd.add(numToGenArg);
    cmd.add(evaluateArg);
    cmd.parseArgs(args);
    numToGen = numToGenArg.getValue();
    evaluate = evaluateArg.getValue();

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  NNEvaluator* nnEval = NULL;
  if(cfg.getFileName() != "") {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    {
      Setup::initializeSession(cfg);
      const int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
      const int expectedConcurrentEvals = params.numThreads;
      const int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
      const bool defaultRequireExactNNLen = false;
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
        Board::MAX_LEN,Board::MAX_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
    logger.write("Loaded neural net");
  }

  AsyncBot* evalBot;
  {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);
    params.maxVisits = 20;
    params.numThreads = 1;
    string seed = Global::uint64ToString(rand.nextUInt64());
    evalBot = new AsyncBot(params, nnEval, &logger, seed);
  }

  //Play no moves in game, since we're sampling initializations
  cfg.overrideKey("maxMovesPerGame","0");

  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);

  for(int i = 0; i<numToGen; i++) {
    string seed = Global::uint64ToString(rand.nextUInt64());
    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = "";
    botSpec.nnEval = nnEval;
    botSpec.baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);

    FinishedGameData* data = gameRunner->runGame(
      seed,
      botSpec,
      botSpec,
      NULL,
      NULL,
      logger,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr
    );

    cout << data->startHist.rules.toString() << endl;
    Board::printBoard(cout, data->startBoard, Board::NULL_LOC, &(data->startHist.moveHistory));
    cout << endl;
    if(evaluate) {
      evalBot->setPosition(data->startPla, data->startBoard, data->startHist);
      evalBot->genMoveSynchronous(data->startPla,TimeControls());
      ReportedSearchValues values = evalBot->getSearchStopAndWait()->getRootValuesRequireSuccess();
      cout << "Winloss: " << values.winLossValue << endl;
      cout << "Lead: " << values.lead << endl;
    }

    delete data;
  }

  delete gameRunner;
  delete evalBot;
  if(nnEval != NULL)
    delete nnEval;

  ScoreValue::freeTables();
  return 0;
}
