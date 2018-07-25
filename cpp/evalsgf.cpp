#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"

using namespace std;

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

int main(int argc, const char* argv[]) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  string nnModelFile;
  string sgfFile;
  int moveNum;
  string printBranch;
  try {
    TCLAP::CmdLine cmd("Sgf->HDF5 data writer", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> nnModelFileArg("","nn-model-file","Neural net model .pb graph file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> sgfFileArg("","sgf-file","Sgf file to analyze",true,string(),"FILE");
    TCLAP::ValueArg<int> moveNumArg("","move-num","Sgf move num to analyze, 0-indexed",true,0,"MOVENUM");
    TCLAP::ValueArg<string> printBranchArg("","print-branch","Move branch in search tree to print",false,string(),"MOVE MOVE ...");
    cmd.add(configFileArg);
    cmd.add(nnModelFileArg);
    cmd.add(sgfFileArg);
    cmd.add(moveNumArg);
    cmd.add(printBranchArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    nnModelFile = nnModelFileArg.getValue();
    sgfFile = sgfFileArg.getValue();
    moveNum = moveNumArg.getValue();
    printBranch = printBranchArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  ConfigParser cfg(configFile);

  Logger logger;
  logger.setLogToStdout(true);

  logger.write("Engine starting...");

  Session* session;
  NNEvaluator* nnEval;
  {
    session = Setup::initializeSession(cfg);
    vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(session,{nnModelFile},cfg,logger,seedRand);
    assert(nnEvals.size() == 1);
    nnEval = nnEvals[0];
  }
  logger.write("Loaded neural net");


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


  SearchParams params;
  {
    vector<SearchParams> paramss = Setup::loadParams(cfg);
    if(paramss.size() != 1)
      throw new StringError("Can only specify examply one search bot in sgf mode");
    params = paramss[0];
  }

  string searchRandSeed;
  if(cfg.contains("searchRandSeed"))
    searchRandSeed = cfg.getString("searchRandSeed");
  else
    searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

  AsyncBot* bot = new AsyncBot(params, nnEval, &logger, searchRandSeed);
  
  Sgf* sgf = Sgf::loadFile(sgfFile);
  int bSize = sgf->getBSize();
  float komi = sgf->getKomi();
  initialRules.komi = komi;

  vector<Move> placements;
  sgf->getPlacements(placements, bSize);
  vector<Move> moves;
  sgf->getMoves(moves, bSize);

  {
    Board board(bSize,bSize);
    Player pla = P_BLACK;
    for(int i = 0; i<placements.size(); i++)
      board.setStone(placements[i].loc,placements[i].pla);

    BoardHistory hist(board,pla,initialRules);
    bot->setPosition(pla,board,hist);
  }

  if(moveNum < 0)
    throw StringError("Move num " + Global::intToString(moveNum) + " requested but must be non-negative");
  if(moveNum > moves.size())
    throw StringError("Move num " + Global::intToString(moveNum) + " requested but sgf has only " + Global::intToString(moves.size()));
  for(int i = 0; i<moveNum; i++)
    bot->makeMove(moves[i].loc,moves[i].pla);
  
  ClockTimer timer;
  nnEval->clearStats();
  Loc loc = bot->genMoveSynchronous(bot->getSearch()->rootPla);
  (void)loc;
  

  Search* search = bot->getSearch();
  ostringstream sout;
  Board::printBoard(sout, bot->getRootBoard(),Board::NULL_LOC,&(bot->getRootHist().moveHistory));

  PrintTreeOptions options;
  options = options.maxDepth(1);
  if(printBranch.length() > 0)
    options = options.onlyBranch(bot->getRootBoard(),printBranch);

  if(options.branch_.size() > 0) {
    Board board = search->rootBoard;
    Player pla = search->rootPla;
    for(int i = 0; i<options.branch_.size(); i++) {
      board.playMoveAssumeLegal(options.branch_[i],pla);
      pla = getOpp(pla);
    }
    Board::printBoard(sout, board,Board::NULL_LOC,NULL);
  }
  
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
  search->printTree(sout, search->rootNode, options);
  logger.write(sout.str());

  delete bot;
  delete nnEval;
  session->Close();

  return 0;
}

