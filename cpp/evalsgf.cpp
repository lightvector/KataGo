#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "program/playutils.h"
#include "program/play.h"
#include "main.h"

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

using namespace std;

int MainCmds::evalsgf(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  string configFile;
  string modelFile;
  string sgfFile;
  int moveNum;
  string printBranch;
  string extraMoves;
  int64_t maxVisits;
  int numThreads;
  float overrideKomi;
  bool printOwnership;
  bool printRootNNValues;
  bool printPolicy;
  bool printScoreNow;
  bool printRootEndingBonus;
  bool printLead;
  bool rawNN;
  try {
    TCLAP::CmdLine cmd("Run a search on a position from an sgf file", ' ', Version::getKataGoVersionForHelp(),true);
    TCLAP::ValueArg<string> configFileArg("","config","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> modelFileArg("","model","Neural net model file to use",true,string(),"FILE");
    TCLAP::UnlabeledValueArg<string> sgfFileArg("","Sgf file to analyze",true,string(),"FILE");
    TCLAP::ValueArg<int> moveNumArg("m","move-num","Sgf move num to analyze, 1-indexed",true,0,"MOVENUM");
    TCLAP::ValueArg<string> printBranchArg("","print-branch","Move branch in search tree to print",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> printArg("p","print","Alias for -print-branch",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraMovesArg("","extra-moves","Extra moves to force-play before doing search",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraArg("e","extra","Alias for -extra-moves",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<long> visitsArg("v","visits","Set the number of visits",false,-1,"VISITS");
    TCLAP::ValueArg<int> threadsArg("t","threads","Set the number of threads",false,-1,"THREADS");
    TCLAP::ValueArg<float> overrideKomiArg("","override-komi","Artificially set komi",false,std::numeric_limits<float>::quiet_NaN(),"KOMI");
    TCLAP::SwitchArg printOwnershipArg("","print-ownership","Print ownership");
    TCLAP::SwitchArg printRootNNValuesArg("","print-root-nn-values","Print root nn values");
    TCLAP::SwitchArg printPolicyArg("","print-policy","Print policy");
    TCLAP::SwitchArg printScoreNowArg("","print-score-now","Print score now");
    TCLAP::SwitchArg printRootEndingBonusArg("","print-root-ending-bonus","Print root ending bonus now");
    TCLAP::SwitchArg printLeadArg("","print-lead","Compute and print lead");
    TCLAP::SwitchArg rawNNArg("","raw-nn","Perform single raw neural net eval");
    cmd.add(configFileArg);
    cmd.add(modelFileArg);
    cmd.add(sgfFileArg);
    cmd.add(moveNumArg);
    cmd.add(printBranchArg);
    cmd.add(printArg);
    cmd.add(extraMovesArg);
    cmd.add(extraArg);
    cmd.add(visitsArg);
    cmd.add(threadsArg);
    cmd.add(overrideKomiArg);
    cmd.add(printOwnershipArg);
    cmd.add(printRootNNValuesArg);
    cmd.add(printPolicyArg);
    cmd.add(printScoreNowArg);
    cmd.add(printRootEndingBonusArg);
    cmd.add(printLeadArg);
    cmd.add(rawNNArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    modelFile = modelFileArg.getValue();
    sgfFile = sgfFileArg.getValue();
    moveNum = moveNumArg.getValue();
    printBranch = printBranchArg.getValue();
    string print = printArg.getValue();
    extraMoves = extraMovesArg.getValue();
    string extra = extraArg.getValue();
    maxVisits = (int64_t)visitsArg.getValue();
    numThreads = threadsArg.getValue();
    overrideKomi = overrideKomiArg.getValue();
    printOwnership = printOwnershipArg.getValue();
    printRootNNValues = printRootNNValuesArg.getValue();
    printPolicy = printPolicyArg.getValue();
    printScoreNow = printScoreNowArg.getValue();
    printRootEndingBonus = printRootEndingBonusArg.getValue();
    printLead = printLeadArg.getValue();
    rawNN = rawNNArg.getValue();

    if(printBranch.length() > 0 && print.length() > 0) {
      cerr << "Error: -print-branch and -print both specified" << endl;
      return 1;
    }
    if(printBranch.length() <= 0)
      printBranch = print;

    if(extraMoves.length() > 0 && extra.length() > 0) {
      cerr << "Error: -extra-moves and -extra both specified" << endl;
      return 1;
    }
    if(extraMoves.length() <= 0)
      extraMoves = extra;
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  //Parse config and rules -------------------------------------------------------------------

  ConfigParser cfg(configFile);
  Rules defaultRules = Rules::getTrompTaylorish();

  Player perspective = Setup::parseReportAnalysisWinrates(cfg,P_BLACK);

  //Parse sgf file and board ------------------------------------------------------------------

  CompactSgf* sgf = CompactSgf::loadFile(sgfFile);

  Board board;
  Player nextPla;
  BoardHistory hist;

  auto setUpBoardUsingRules = [&board,&nextPla,&hist,overrideKomi,moveNum,&sgf,&extraMoves](const Rules& initialRules) {
    sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
    vector<Move>& moves = sgf->moves;

    if(!isnan(overrideKomi)) {
      if(overrideKomi > board.x_size * board.y_size || overrideKomi < -board.x_size * board.y_size)
        throw StringError("Invalid komi, greater than the area of the board");
      hist.setKomi(overrideKomi);
    }

    if(moveNum < 0)
      throw StringError("Move num " + Global::intToString(moveNum) + " requested but must be non-negative");
    if(moveNum > moves.size())
      throw StringError("Move num " + Global::intToString(moveNum) + " requested but sgf has only " + Global::intToString(moves.size()));

    sgf->playMovesTolerant(board,nextPla,hist,moveNum,false);

    vector<Loc> extraMoveLocs = Location::parseSequence(extraMoves,board);
    for(size_t i = 0; i<extraMoveLocs.size(); i++) {
      Loc loc = extraMoveLocs[i];
      if(!board.isLegal(loc,nextPla,hist.rules.multiStoneSuicideLegal)) {
        cerr << board << endl;
        cerr << "Extra illegal move for " << PlayerIO::colorToChar(nextPla) << ": " << Location::toString(loc,board) << endl;
        throw StringError("Illegal extra move");
      }
      hist.makeBoardMoveAssumeLegal(board,loc,nextPla,NULL);
      nextPla = getOpp(nextPla);
    }
  };

  Rules initialRules = sgf->getRulesOrWarn(
    defaultRules,
    [](const string& msg) { cout << msg << endl; }
  );
  setUpBoardUsingRules(initialRules);

  //Parse move sequence arguments------------------------------------------

  PrintTreeOptions options;
  options = options.maxDepth(1);
  if(printBranch.length() > 0)
    options = options.onlyBranch(board,printBranch);

  //Load neural net and start bot------------------------------------------

  Logger logger;
  logger.setLogToStdout(true);
  logger.write("Engine starting...");

  SearchParams params = Setup::loadSingleParams(cfg);
  if(maxVisits < -1 || maxVisits == 0)
    throw StringError("maxVisits: invalid value");
  else if(maxVisits == -1)
    logger.write("No max visits specified on cmdline, using defaults in " + cfg.getFileName());
  else {
    params.maxVisits = maxVisits;
    params.maxPlayouts = maxVisits; //Also set this so it doesn't cap us either
  }
  if(numThreads < -1 || numThreads == 0)
    throw StringError("numThreads: invalid value");
  else if(numThreads == -1)
    logger.write("No num threads specified on cmdline, using defaults in " + cfg.getFileName());
  else {
    params.numThreads = numThreads;
  }

  string searchRandSeed;
  if(cfg.contains("searchRandSeed"))
    searchRandSeed = cfg.getString("searchRandSeed");
  else
    searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = params.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    int defaultMaxBatchSize = std::max(8,((params.numThreads+3)/4)*4);
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,cfg,logger,seedRand,maxConcurrentEvals,
      board.x_size,board.y_size,defaultMaxBatchSize,
      Setup::SETUP_FOR_GTP
    );
  }
  logger.write("Loaded neural net");

  {
    bool rulesWereSupported;
    Rules supportedRules = nnEval->getSupportedRules(initialRules,rulesWereSupported);
    if(!rulesWereSupported) {
      cout << "Warning: Rules " << initialRules << " from sgf not supported by neural net, using " << supportedRules << " instead" << endl;
      //Attempt to re-set-up the board using supported rules
      setUpBoardUsingRules(supportedRules);
    }
  }

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  if(rawNN) {
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    MiscNNInputParams nnInputParams;
    nnInputParams.drawEquivalentWinsForWhite = params.drawEquivalentWinsForWhite;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    cout << "Rules: " << hist.rules << endl;
    cout << "Encore phase " << hist.encorePhase << endl;
    Board::printBoard(cout, board, Board::NULL_LOC, &(hist.moveHistory));
    buf.result->debugPrint(cout,board);
    return 0;
  }

  AsyncBot* bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);

  bot->setPosition(nextPla,board,hist);

  //Print initial state----------------------------------------------------------------
  const Search* search = bot->getSearchStopAndWait();
  ostringstream sout;
  sout << "Rules: " << hist.rules << endl;
  sout << "Encore phase " << hist.encorePhase << endl;
  Board::printBoard(sout, board, Board::NULL_LOC, &(hist.moveHistory));

  if(options.branch_.size() > 0) {
    Board copy = board;
    BoardHistory copyHist = hist;
    Player pla = nextPla;
    for(int i = 0; i<options.branch_.size(); i++) {
      Loc loc = options.branch_[i];
      if(!copy.isLegal(loc,pla,copyHist.rules.multiStoneSuicideLegal)) {
        cerr << board << endl;
        cerr << "Branch Illegal move for " << PlayerIO::colorToChar(pla) << ": " << Location::toString(loc,board) << endl;
        return 1;
      }
      copyHist.makeBoardMoveAssumeLegal(copy,loc,pla,NULL);
      pla = getOpp(pla);
    }
    Board::printBoard(sout, copy, Board::NULL_LOC, &(copyHist.moveHistory));
  }

  sout << "\n";
  logger.write(sout.str());
  sout.clear();

  //Search!----------------------------------------------------------------

  ClockTimer timer;
  nnEval->clearStats();
  Loc loc = bot->genMoveSynchronous(bot->getSearch()->rootPla,TimeControls());
  (void)loc;

  //Postprocess------------------------------------------------------------

  if(printOwnership) {
    sout << "Ownership map (ROOT position):\n";
    search->printRootOwnershipMap(sout,perspective);
  }

  if(printRootNNValues) {
    if(search->rootNode->nnOutput != nullptr) {
      NNOutput* nnOutput = search->rootNode->nnOutput.get();
      cout << "White win: " << nnOutput->whiteWinProb << endl;
      cout << "White loss: " << nnOutput->whiteLossProb << endl;
      cout << "White noresult: " << nnOutput->whiteNoResultProb << endl;
      cout << "White score mean " << nnOutput->whiteScoreMean << endl;
      cout << "White score stdev " << sqrt(max(0.0,(double)nnOutput->whiteScoreMeanSq - nnOutput->whiteScoreMean*nnOutput->whiteScoreMean)) << endl;
    }
  }

  if(printPolicy) {
    if(search->rootNode->nnOutput != nullptr) {
      NNOutput* nnOutput = search->rootNode->nnOutput.get();
      float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
      cout << "Root policy: " << endl;
      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
          double prob = policyProbs[pos];
          if(prob < 0)
            cout << "  -  " << " ";
          else
            cout << Global::strprintf("%5.2f",prob*100) << " ";
        }
        cout << endl;
      }
      double prob = policyProbs[NNPos::locToPos(Board::PASS_LOC,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen)];
      cout << "Pass " << Global::strprintf("%5.2f",prob*100) << endl;
    }
  }

  if(printScoreNow) {
    sout << "Score now (ROOT position):\n";
    Board copy(board);
    BoardHistory copyHist(hist);
    Color area[Board::MAX_ARR_SIZE];
    copyHist.endAndScoreGameNow(copy,area);

    for(int y = 0; y<copy.y_size; y++) {
      for(int x = 0; x<copy.x_size; x++) {
        Loc l = Location::getLoc(x,y,copy.x_size);
        sout << PlayerIO::colorToChar(area[l]);
      }
      sout << endl;
    }
    sout << endl;

    sout << "Komi: " << copyHist.rules.komi << endl;
    sout << "WBonus: " << copyHist.whiteBonusScore << endl;
    sout << "Final: "; WriteSgf::printGameResult(sout, copyHist); sout << endl;
  }

  if(printRootEndingBonus) {
    sout << "Ending bonus (ROOT position)\n";
    search->printRootEndingScoreValueBonus(sout);
  }

  sout << "Time taken: " << timer.getSeconds() << "\n";
  sout << "Root visits: " << search->numRootVisits() << "\n";
  sout << "NN rows: " << nnEval->numRowsProcessed() << endl;
  sout << "NN batches: " << nnEval->numBatchesProcessed() << endl;
  sout << "NN avg batch size: " << nnEval->averageProcessedBatchSize() << endl;
  sout << "PV: ";
  search->printPV(sout, search->rootNode, 25);
  sout << "\n";
  sout << "Tree:\n";
  search->printTree(sout, search->rootNode, options, perspective);
  logger.write(sout.str());

  if(printLead) {
    BoardHistory hist2(hist);
    double lead = PlayUtils::computeLead(
      bot->getSearchStopAndWait(), NULL, board, hist2, nextPla,
      20, logger, OtherGameProperties()
    );
    cout << "LEAD: " << lead << endl;
  }

  delete bot;
  delete nnEval;
  NeuralNet::globalCleanup();
  delete sgf;
  ScoreValue::freeTables();

  return 0;
}
