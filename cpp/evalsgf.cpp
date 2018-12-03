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
  Rand seedRand;

  string configFile;
  string modelFile;
  string sgfFile;
  int moveNum;
  string printBranch;
  string extraMoves;
  int maxVisits;
  int numThreads;
  bool printOwnership;
  try {
    TCLAP::CmdLine cmd("Run a search on a position from an sgf file", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> modelFileArg("","model-file","Neural net model file to use",true,string(),"FILE");
    TCLAP::UnlabeledValueArg<string> sgfFileArg("","Sgf file to analyze",true,string(),"FILE");
    TCLAP::ValueArg<int> moveNumArg("","move-num","Sgf move num to analyze, 1-indexed",true,0,"MOVENUM");
    TCLAP::ValueArg<string> printBranchArg("","print-branch","Move branch in search tree to print",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> printArg("p","print","Alias for -print-branch",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraMovesArg("","extra-moves","Extra moves to force-play before doing search",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraArg("e","extra","Alias for -extra-moves",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<int> visitsArg("v","visits","Set the number of visits",false,-1,"VISTIS");
    TCLAP::ValueArg<int> threadsArg("t","threads","Set the number of threads",false,-1,"THREADS");
    TCLAP::SwitchArg printOwnershipArg("o","print-ownership","Print ownership");
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
    cmd.add(printOwnershipArg);
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
    printOwnership = printOwnershipArg.getValue();

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
    vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators({modelFile},cfg,logger,seedRand,maxConcurrentEvals,false);
    assert(nnEvals.size() == 1);
    nnEval = nnEvals[0];
  }
  logger.write("Loaded neural net");

  //Check for unused config keys
  {
    vector<string> unusedKeys = cfg.unusedKeys();
    for(size_t i = 0; i<unusedKeys.size(); i++) {
      string msg = "WARNING: Unused key '" + unusedKeys[i] + "' in " + configFile;
      logger.write(msg);
      cerr << msg << endl;
    }
  }

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
  Loc loc = bot->genMoveSynchronous(bot->getSearch()->rootPla);
  (void)loc;

  //Postprocess------------------------------------------------------------

  if(printOwnership) {
    sout << "Ownership map (ORIG position):\n";
    search->printRootOwnershipMap(sout);
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

  return 0;
}

