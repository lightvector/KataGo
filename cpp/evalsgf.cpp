#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "main.h"

using namespace std;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

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
  int maxVisits;
  int numThreads;
  float overrideKomi;
  bool printOwnership;
  bool printRootNNValues;
  bool printScoreNow;
  bool printRootEndingBonus;
  try {
    TCLAP::CmdLine cmd("Run a search on a position from an sgf file", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> modelFileArg("","model","Neural net model file to use",true,string(),"FILE");
    TCLAP::UnlabeledValueArg<string> sgfFileArg("","Sgf file to analyze",true,string(),"FILE");
    TCLAP::ValueArg<int> moveNumArg("m","move-num","Sgf move num to analyze, 1-indexed",true,0,"MOVENUM");
    TCLAP::ValueArg<string> printBranchArg("","print-branch","Move branch in search tree to print",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> printArg("p","print","Alias for -print-branch",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraMovesArg("","extra-moves","Extra moves to force-play before doing search",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraArg("e","extra","Alias for -extra-moves",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<int> visitsArg("v","visits","Set the number of visits",false,-1,"VISITS");
    TCLAP::ValueArg<int> threadsArg("t","threads","Set the number of threads",false,-1,"THREADS");
    TCLAP::ValueArg<float> overrideKomiArg("","override-komi","Artificially set komi",false,std::numeric_limits<float>::quiet_NaN(),"KOMI");
    TCLAP::SwitchArg printOwnershipArg("","print-ownership","Print ownership");
    TCLAP::SwitchArg printRootNNValuesArg("","print-root-nn-values","Print root nn values");
    TCLAP::SwitchArg printScoreNowArg("","print-score-now","Print score now");
    TCLAP::SwitchArg printRootEndingBonusArg("","print-root-ending-bonus","Print root ending bonus now");
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
    cmd.add(printScoreNowArg);
    cmd.add(printRootEndingBonusArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    modelFile = modelFileArg.getValue();
    sgfFile = sgfFileArg.getValue();
    moveNum = moveNumArg.getValue();
    printBranch = printBranchArg.getValue();
    string print = printArg.getValue();
    extraMoves = extraMovesArg.getValue();
    string extra = extraArg.getValue();
    maxVisits = visitsArg.getValue();
    numThreads = threadsArg.getValue();
    overrideKomi = overrideKomiArg.getValue();
    printOwnership = printOwnershipArg.getValue();
    printRootNNValues = printRootNNValuesArg.getValue();
    printScoreNow = printScoreNowArg.getValue();
    printRootEndingBonus = printRootEndingBonusArg.getValue();

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
  Rules initialRules;
  {
    //All of these might get overwritten by the sgf
    string koRule = cfg.getString("koRule", Rules::koRuleStrings());
    string scoringRule = cfg.getString("scoringRule", Rules::scoringRuleStrings());
    bool multiStoneSuicideLegal = cfg.getBool("multiStoneSuicideLegal");
    float komi = 7.5f; //Default komi, sgf will generally override this

    initialRules.koRule = Rules::parseKoRule(koRule);
    initialRules.scoringRule = Rules::parseScoringRule(scoringRule);
    initialRules.multiStoneSuicideLegal = multiStoneSuicideLegal;
    initialRules.komi = komi;
  }

  //Parse sgf file and board ------------------------------------------------------------------

  CompactSgf* sgf = CompactSgf::loadFile(sgfFile);

  Board board;
  Player nextPla;
  BoardHistory hist;
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

  for(int i = 0; i<moveNum; i++) {
    if(!board.isLegal(moves[i].loc,moves[i].pla,hist.rules.multiStoneSuicideLegal)) {
      cerr << board << endl;
      cerr << "SGF Illegal move " << (i+1) << " for " << colorToChar(moves[i].pla) << ": " << Location::toString(moves[i].loc,board) << endl;
      return 1;
    }
    hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
    nextPla = getOpp(moves[i].pla);
  }

  //Parse move sequence arguments------------------------------------------

  PrintTreeOptions options;
  options = options.maxDepth(1);
  if(printBranch.length() > 0)
    options = options.onlyBranch(board,printBranch);

  vector<Loc> extraMoveLocs = Location::parseSequence(extraMoves,board);
  for(size_t i = 0; i<extraMoveLocs.size(); i++) {
    Loc loc = extraMoveLocs[i];
    if(!board.isLegal(loc,nextPla,hist.rules.multiStoneSuicideLegal)) {
      cerr << board << endl;
      cerr << "Extra illegal move for " << colorToChar(nextPla) << ": " << Location::toString(loc,board) << endl;
      return 1;
    }
    hist.makeBoardMoveAssumeLegal(board,loc,nextPla,NULL);
    nextPla = getOpp(nextPla);
  }

  //Load neural net and start bot------------------------------------------

  Logger logger;
  logger.setLogToStdout(true);
  logger.write("Engine starting...");

  SearchParams params;
  {
    vector<SearchParams> paramss = Setup::loadParams(cfg);
    if(paramss.size() != 1)
      throw StringError("Can only specify exactly one bot in for searching in an sgf");
    params = paramss[0];
  }
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
    vector<NNEvaluator*> nnEvals =
      Setup::initializeNNEvaluators(
        {modelFile},{modelFile},cfg,logger,seedRand,maxConcurrentEvals,
        false,false,board.x_size,board.y_size
      );
    assert(nnEvals.size() == 1);
    nnEval = nnEvals[0];
  }
  logger.write("Loaded neural net");

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  AsyncBot* bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);

  bot->setPosition(nextPla,board,hist);

  //Print initial state----------------------------------------------------------------
  Search* search = bot->getSearch();
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
        cerr << "Branch Illegal move for " << colorToChar(pla) << ": " << Location::toString(loc,board) << endl;
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
    search->printRootOwnershipMap(sout);
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

  if(printScoreNow) {
    sout << "Score now (ROOT position):\n";
    Board copy(board);
    BoardHistory copyHist(hist);
    Color area[Board::MAX_ARR_SIZE];
    copyHist.endAndScoreGameNow(copy,area);

    for(int y = 0; y<copy.y_size; y++) {
      for(int x = 0; x<copy.x_size; x++) {
        Loc l = Location::getLoc(x,y,copy.x_size);
        sout << colorToChar(area[l]);
      }
      sout << endl;
    }
    sout << endl;

    sout << "Komi: " << copyHist.rules.komi << endl;
    sout << "WBonus: " << copyHist.whiteBonusScore << endl;
    sout << "Final: " << copyHist.finalWhiteMinusBlackScore << endl;
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
  search->printTree(sout, search->rootNode, options);
  logger.write(sout.str());

  delete bot;
  delete nnEval;
  NeuralNet::globalCleanup();
  delete sgf;
  ScoreValue::freeTables();

  return 0;
}

