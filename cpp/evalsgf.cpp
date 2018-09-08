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

int MainCmds::evalSgf(int argc, const char* const* argv) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  string nnModelFile;
  string sgfFile;
  int moveNum;
  string printBranch;
  string extraMoves;
  try {
    TCLAP::CmdLine cmd("Sgf->HDF5 data writer", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> nnModelFileArg("","nn-model-file","Neural net model .pb graph file to use",true,string(),"FILE");
    TCLAP::UnlabeledValueArg<string> sgfFileArg("","Sgf file to analyze",true,string(),"FILE");
    TCLAP::ValueArg<int> moveNumArg("","move-num","Sgf move num to analyze, 1-indexed",true,0,"MOVENUM");
    TCLAP::ValueArg<string> printBranchArg("","print-branch","Move branch in search tree to print",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> printArg("","print","Alias for -print-branch",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraMovesArg("","extra-moves","Alias for -extra-moves",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> movesArg("","moves","Alias for -extra-moves",false,string(),"MOVE MOVE ...");
    cmd.add(configFileArg);
    cmd.add(nnModelFileArg);
    cmd.add(sgfFileArg);
    cmd.add(moveNumArg);
    cmd.add(printBranchArg);
    cmd.add(printArg);
    cmd.add(extraMovesArg);
    cmd.add(movesArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    nnModelFile = nnModelFileArg.getValue();
    sgfFile = sgfFileArg.getValue();
    moveNum = moveNumArg.getValue();
    printBranch = printBranchArg.getValue();
    string print = printArg.getValue();
    extraMoves = extraMovesArg.getValue();
    string moves = movesArg.getValue();

    if(printBranch.length() > 0 && print.length() > 0) {
      cerr << "Error: -print-branch and -print both specified" << endl;
      return 1;
    }
    if(printBranch.length() <= 0)
      printBranch = print;

    if(extraMoves.length() > 0 && moves.length() > 0) {
      cerr << "Error: -extra-moves and -moves both specified" << endl;
      return 1;
    }
    if(extraMoves.length() <= 0)
      extraMoves = moves;
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  //Parse config and rules -------------------------------------------------------------------

  ConfigParser cfg(configFile);
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

  //Parse sgf file and board ------------------------------------------------------------------

  Sgf* sgf = Sgf::loadFile(sgfFile);
  int bSize = sgf->getBSize();
  float komi = sgf->getKomi();
  initialRules.komi = komi;

  vector<Move> placements;
  sgf->getPlacements(placements, bSize);
  vector<Move> moves;
  sgf->getMoves(moves, bSize);

  Board board(bSize,bSize);
  Player nextPla = P_BLACK;
  BoardHistory hist(board,nextPla,initialRules);
  {
    bool hasBlack = false;
    bool allBlack = true;
    for(int i = 0; i<placements.size(); i++) {
      board.setStone(placements[i].loc,placements[i].pla);
      if(placements[i].pla == P_BLACK)
        hasBlack = true;
      else
        allBlack = false;
    }

    if(hasBlack && !allBlack)
      nextPla = P_WHITE;
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

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators({nnModelFile},cfg,logger,seedRand);
    assert(nnEvals.size() == 1);
    nnEval = nnEvals[0];
  }
  logger.write("Loaded neural net");

  SearchParams params;
  {
    vector<SearchParams> paramss = Setup::loadParams(cfg);
    if(paramss.size() != 1)
      throw StringError("Can only specify examply one search bot in sgf mode");
    params = paramss[0];
  }

  string searchRandSeed;
  if(cfg.contains("searchRandSeed"))
    searchRandSeed = cfg.getString("searchRandSeed");
  else
    searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

  AsyncBot* bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);

  bot->setPosition(nextPla,board,hist);

  //Print initial state----------------------------------------------------------------
  Search* search = bot->getSearch();
  ostringstream sout;
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

