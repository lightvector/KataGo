#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "core/test.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "main.h"

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

int MainCmds::writeSearchValueTimeseries(int argc, const char* const* argv) {
  Board::initHash();
  Rand seedRand;

  string configFile;
  string nnModelFile;
  vector<string> sgfsDirs;
  string outputFile;
  int numThreads;
  double usePosProb;
  string mode;
  try {
    TCLAP::CmdLine cmd("Write search value timeseries", ' ', "1.0",true);
    TCLAP::ValueArg<string> configFileArg("","config-file","Config file to use (see configs/gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> nnModelFileArg("","nn-model-file","Neural net model .pb graph file to use",true,string(),"FILE");
    TCLAP::MultiArg<string> sgfsDirsArg("","sgfs-dir","Directory of sgfs files",true,"DIR");
    TCLAP::ValueArg<string> outputFileArg("","output-csv","Output csv file",true,string(),"FILE");
    TCLAP::ValueArg<int> numThreadsArg("","num-threads","Number of threads to use",true,1,"INT");
    TCLAP::ValueArg<double> usePosProbArg("","use-pos-prob","Probability to use a position",true,0.0,"PROB");
    TCLAP::ValueArg<string> modeArg("","mode","rootValue|policyTargetSurprise",true,string(),"MODE");

    cmd.add(configFileArg);
    cmd.add(nnModelFileArg);
    cmd.add(sgfsDirsArg);
    cmd.add(outputFileArg);
    cmd.add(numThreadsArg);
    cmd.add(usePosProbArg);
    cmd.add(modeArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    nnModelFile = nnModelFileArg.getValue();
    sgfsDirs = sgfsDirsArg.getValue();
    outputFile = outputFileArg.getValue();
    numThreads = numThreadsArg.getValue();
    usePosProb = usePosProbArg.getValue();
    mode = modeArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  if(mode != "rootValue" && mode != "policyTargetSurprise") {
    cout << "Error: mode must be rootValue or policyTargetSurprise" << endl;
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
  int posLen = nnEval->getPosLen();
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

  //Check for unused config keys
  {
    vector<string> unusedKeys = cfg.unusedKeys();
    for(size_t i = 0; i<unusedKeys.size(); i++) {
      string msg = "WARNING: Unused key '" + unusedKeys[i] + "' in " + configFile;
      logger.write(msg);
      cerr << msg << endl;
    }
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
  {
    uint64_t numPoses = 0;
    vector<Move> movesBuf;
    for(size_t i = 0; i<sgfs.size(); i++) {
      sgfs[i]->getMoves(movesBuf,sgfs[i]->getBSize());
      numPoses += movesBuf.size();
      movesBuf.clear();
    }
    cout << "Num unique poses: " << numPoses << endl;
    cout << "(avg moves per game): " << ((double)numPoses / sgfs.size()) << endl;
  }

  ofstream out;
  out.open(outputFile);
  mutex outMutex;

  auto computeSurprise = [&](Search* search) {
    vector<Loc> locs;
    vector<double> playSelectionValues;
    bool suc = search->getPlaySelectionValues(locs,playSelectionValues);
    testAssert(suc);

    assert(search->rootNode != NULL);
    assert(search->rootNode->nnOutput != NULL);
    float* policyProbs = search->rootNode->nnOutput->policyProbs;

    assert(locs.size() == playSelectionValues.size());
    double sum = 0.0;
    for(int i = 0; i<locs.size(); i++) {
      sum += playSelectionValues[i];
      assert(playSelectionValues[i] >= 0.0);
    }
    assert(sum > 0.0);

    for(int i = 0; i<locs.size(); i++) {
      playSelectionValues[i] /= sum;
    }

    double surprise = 0.0;
    for(int i = 0; i<locs.size(); i++) {
      if(playSelectionValues[i] > 1e-50) {
        Loc loc = locs[i];
        int pos = NNPos::locToPos(loc,search->rootBoard.x_size,posLen);
        //surprise += playSelectionValues[i] * (log(playSelectionValues[i]) - log(policyProbs[pos]));
        surprise += playSelectionValues[i] * log(policyProbs[pos]);
      }
    }
    return surprise;
  };

  auto runThread = [&](int threadIdx, string randSeed) {
    Search* search = new Search(params,nnEval,randSeed);

    int maxVisits;
    if(mode == "rootValue")
      maxVisits = 80000;
    else if(mode == "policyTargetSurprise")
      maxVisits = 5000;
    else
      assert(false);

    double* values = new double[maxVisits];
    double* policySurpriseNats = new double[maxVisits];
    Rand rand("root variance estimate " + Global::intToString(threadIdx));
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

        if(rand.nextDouble() < usePosProb) {
          SearchThread* stbuf = new SearchThread(0,*search,&logger);
          search->beginSearch();
          for(int i = 0; i<maxVisits; i++) {
            search->runSinglePlayout(*stbuf);
            values[i] = search->rootNode->stats.getCombinedValueSum(search->searchParams) / search->rootNode->stats.valueSumWeight;
            policySurpriseNats[i] = computeSurprise(search);
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
            if(mode == "rootValue") {
              for(int i = 0; i<maxVisits; i++)
                out << values[i] << ",";
            }
            else if(mode == "policyTargetSurprise") {
              for(int i = 0; i<maxVisits; i++)
                out << policySurpriseNats[i] << ",";
            }
            else {
              assert(false);
            }

            out << endl;
          }
        }

        search->makeMove(moves[moveNum].loc,moves[moveNum].pla);
        search->clearSearch();
      }
    }
    delete search;
    delete[] values;
    delete[] policySurpriseNats;
  };

  std::thread threads[numThreads];
  for(int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
    threads[threadIdx] = std::thread(runThread, threadIdx, Global::uint64ToString(seedRand.nextUInt64()));
  }
  for(int threadIdx = 0; threadIdx < numThreads; threadIdx++) {
    threads[threadIdx].join();
  }

  out.close();

  for(size_t i = 0; i<sgfs.size(); i++)
    delete sgfs[i];
  sgfs.clear();

  cout << "Done!" << endl;

  return 0;
}
