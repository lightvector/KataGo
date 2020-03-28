#include "../core/global.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../dataio/lzparse.h"
#include "../command/commandline.h"
#include "../command/main.h"

#include <fstream>
#include <algorithm>

using namespace std;

//a must be smaller than b
static double approxGCD(double a, double b) {
  while(true) {
    double remainder = b - floor(b / a) * a;
    if(remainder < 0.00005)
      return a;
    b = a;
    a = remainder;
  }
}

static void printLZCost(const string& lzFile, ofstream& out) {
  Board board;
  BoardHistory hist;
  vector<Move> moves;
  double maxEstimatedVisits = 0.0;
  double sumEstimatedVisits = 0.0;
  double sumMaxProb = 0.0;
  int64_t rowCount = 0;
  std::function<void(const LZSample& sample, const string& fileName, int sampleCount)> f =
    [&board,&hist,&moves,&maxEstimatedVisits,&sumEstimatedVisits,&sumMaxProb,&rowCount]
    (const LZSample& sample, const string& fileName, int sampleCount) {

    int policyTargetLen = 362;
    float policyTarget[362];
    Player nextPlayer;
    Player winner;
    try {
      sample.parse(board,hist,moves,policyTarget,nextPlayer,winner);
    }
    catch(const IOError &e) {
      cout << "Error reading: " << fileName << " sample " << sampleCount << ": " << e.message << endl;
      return;
    }

    const double maxReasonableVisits = 20000.0;

    //Find the smallest several distinct reasonable values
    double prob0 = 1.0;
    double prob1 = 1.0;
    double prob2 = 1.0;
    double prob3 = 1.0;
    double prob4 = 1.0;
    double prob5 = 1.0;

    double maxProb = 0.0;
    for(int i = 0; i<policyTargetLen; i++) {
      double prob = policyTarget[i];
      if(prob <= 1.0000001 && prob > maxProb)
        maxProb = prob;
      if(prob >= 1/maxReasonableVisits) {
        if(prob < prob0)
          std::swap(prob,prob0);
        if(prob < prob1)
          std::swap(prob,prob1);
        if(prob < prob2)
          std::swap(prob,prob2);
        if(prob < prob3)
          std::swap(prob,prob3);
        if(prob < prob4)
          std::swap(prob,prob4);
        if(prob < prob5)
          std::swap(prob,prob5);
      }
    }

    //Find approximate GCDs
    double gcd = approxGCD(prob0,prob1);
    gcd = approxGCD(gcd,prob1);
    gcd = approxGCD(gcd,prob2);
    gcd = approxGCD(gcd,prob3);
    gcd = approxGCD(gcd,prob4);
    gcd = approxGCD(gcd,prob5);

    //Invert as the estimate of visits
    double estVisits = 1.0 / gcd;

    if(estVisits > maxEstimatedVisits)
      maxEstimatedVisits = estVisits;
    sumEstimatedVisits += estVisits;
    sumMaxProb += maxProb;
    rowCount++;
  };

  LZSample::iterSamples(lzFile,f);

  //Basically, lzfile, probable max visits, probable based on avg, number of rows, proportion of playouts that might be reusable
  cout << lzFile << "," << (maxEstimatedVisits) << "," << (sumEstimatedVisits/rowCount) << "," << rowCount << "," << (sumMaxProb / rowCount) << endl;
  out << lzFile << "," << (maxEstimatedVisits) << "," << (sumEstimatedVisits/rowCount) << "," << rowCount << "," << (sumMaxProb / rowCount) << endl;
}

int MainCmds::lzcost(int argc, const char* const* argv) {
  assert(sizeof(size_t) == 8);
  Board::initHash();
  ScoreValue::initTables();

  cerr << "Command: ";
  for(int i = 0; i<argc; i++)
    cerr << argv[i] << " ";
  cerr << endl;

  vector<string> lzDirs;
  double sampleProb;
  string outFile;
  try {
    KataGoCommandLine cmd("Internal sandbox tool");
    TCLAP::MultiArg<string> lzdirArg("","lzdir","Directory of leela zero gzipped data files",false,"DIR");
    TCLAP::ValueArg<double> sampleProbArg("","sampleprob","Probability to sample a file",true,0.0,"PROB");
    TCLAP::ValueArg<string> outFileArg("","out","File to write results",true,string(),"FILE");
    cmd.add(lzdirArg);
    cmd.add(sampleProbArg);
    cmd.add(outFileArg);
    cmd.parse(argc,argv);
    lzDirs = lzdirArg.getValue();
    sampleProb = sampleProbArg.getValue();
    outFile = outFileArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  //Collect LZ data files-----------------------------------------------------------------
  const string lzSuffix = ".gz";
  auto lzFilter = [&lzSuffix](const string& name) {
    return Global::isSuffix(name,lzSuffix);
  };
  vector<string> lzFiles;
  for(int i = 0; i<lzDirs.size(); i++)
    Global::collectFiles(lzDirs[i], lzFilter, lzFiles);
  cerr << "Found " << lzFiles.size() << " leela zero gz files!" << endl;

  std::sort(lzFiles.begin(),lzFiles.end());

  Rand rand;
  ofstream out(outFile);
  for(int i = 0; i<lzFiles.size(); i++) {
    if(rand.nextBool(sampleProb))
      printLZCost(lzFiles[i],out);
  }
  out.close();

  ScoreValue::freeTables();

  cerr << "Done" << endl;
  return 0;
}
