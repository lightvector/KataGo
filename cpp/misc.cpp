#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "main.h"

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

int MainCmds::writeRootValueTimeseries(int argc, const char* const* argv) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  string nnModelFile;
  vector<string> sgfsDirs;
  string outputFile;
  try {
    TCLAP::CmdLine cmd("Sgf->HDF5 data writer", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> nnModelFileArg("","nn-model-file","Neural net model .pb graph file to use",true,string(),"FILE");
    TCLAP::MultiArg<string> sgfsDirsArg("","sgfs-dir","Directory of sgfs files",true,"DIR");
    TCLAP::ValueArg<string> outputFileArg("","output-csv","Output csv file",true,string(),"FILE");

    cmd.add(configFileArg);
    cmd.add(nnModelFileArg);
    cmd.add(sgfsDirsArg);
    cmd.add(outputFileArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    nnModelFile = nnModelFileArg.getValue();
    sgfsDirs = sgfsDirsArg.getValue();
    outputFile = outputFileArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  ConfigParser cfg(configFile);

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
      throw StringError("Can only specify examply one search bot in sgf mode");
    params = paramss[0];
  }

  const string sgfsSuffix = ".sgfs";
  auto sgfsFilter = [&](const string& name) {
    return Global::isSuffix(name,sgfsSuffix);
  };

  vector<string> sgfsFiles;
  for(int i = 0; i<sgfsDirs.size(); i++)
    Global::collectFiles(sgfsDirs[i], sgfsFilter, sgfsFiles);
  cout << "Found " << sgfsFiles.size() << " files!" << endl;

  vector<Sgf*> sgfs = Sgf::loadSgfsFiles(sgfsFiles);
  cout << "Read " << sgfs.size() << " sgfs!" << endl;

  ofstream out;
  out.open(outputFile);
  mutex outMutex;

  auto runThread = [&](int threadIdx, int numThreads, string randSeed) {
    Search* search = new Search(params,nnEval,randSeed);
    int maxVisits = 50000;
    double* valueSums = new double[maxVisits];
    for(size_t sgfIdx = threadIdx; sgfIdx<sgfs.size(); sgfIdx += numThreads) {
      const Sgf* sgf = sgfs[sgfIdx];
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
        search->setPosition(pla,board,hist);
      }

      for(size_t moveNum = 0; moveNum < moves.size(); moveNum++) {
        SearchThread* stbuf = new SearchThread(0,*search,&logger);
        search->beginSearch();
        for(int i = 0; i<maxVisits; i++) {
          search->runSinglePlayout(*stbuf);
          valueSums[i] = search->rootNode->stats.getCombinedValueSum(search->searchParams);
        }
        delete stbuf;

        {
          float* policy = search->rootNode->nnOutput->policyProbs;
          double entropy = 0.0;
          for(int i = 0; i<NNPos::NN_POLICY_SIZE; i++) {
            if(policy[i] < 1e-20)
              continue;
            entropy -= policy[i] * log(policy[i]);
          }

          std::lock_guard<std::mutex> guard(outMutex);
          out << moveNum << ",";
          out << entropy << ",";
          for(int i = 0; i<maxVisits; i++)
            out << valueSums[i] << ",";
          out << endl;
        }

        search->makeMove(moves[moveNum].loc,moves[moveNum].pla);
        search->clearSearch();
      }
    }
    delete search;
    delete valueSums;
  };

  int numThreads = 8;
  std::thread threads[numThreads];
  for(int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
    threads[threadIdx] = std::thread(runThread, threadIdx, numThreads, Global::uint64ToString(seedRand.nextUInt64()));
  }
  for(int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
    threads[threadIdx].join();
  }

  out.close();

  return 0;
}
