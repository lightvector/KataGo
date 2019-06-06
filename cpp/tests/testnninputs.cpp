#include "../tests/tests.h"

#include <iomanip>

#include "../neuralnet/nninputs.h"
#include "../dataio/sgf.h"

using namespace std;
using namespace TestCommon;

template <typename T>
static void printNNInputHWAndBoard(
  ostream& out, int inputsVersion, const Board& board, const BoardHistory& hist,
  int nnXLen, int nnYLen, bool inputsUseNHWC, T* row, int c
) {
  int numFeatures;
  if(inputsVersion == 3)
    numFeatures = NNInputs::NUM_FEATURES_SPATIAL_V3;
  else if(inputsVersion == 4)
    numFeatures = NNInputs::NUM_FEATURES_SPATIAL_V4;
  else if(inputsVersion == 5)
    numFeatures = NNInputs::NUM_FEATURES_SPATIAL_V5;
  else
    testAssert(false);

  out << "Channel: " << c << endl;

  for(int y = 0; y<nnYLen; y++) {
    for(int x = 0; x<nnXLen; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      if(x > 0)
        out << " ";
      if(inputsUseNHWC)
        out << row[pos * numFeatures + c];
      else
        out << row[c * nnXLen * nnYLen + pos];
    }
    if(y < board.y_size) {
      out << "  ";
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        char s = colorToChar(board.colors[loc]);
        out << s;

        bool histMarked = false;
        for(int i = (int)hist.moveHistory.size()-5; i<hist.moveHistory.size(); i++) {
          if(i >= 0 && hist.moveHistory[i].loc == loc) {
            out << i - (hist.moveHistory.size()-5) + 1;
            histMarked = true;
            break;
          }
        }
        if(x < board.x_size-1 && !histMarked)
          out << ' ';
      }
    }
    out << endl;
  }
  out << endl;
}

template <typename T>
static void printNNInputGlobal(ostream& out, int inputsVersion, T* row, int c) {
  int numFeatures;
  if(inputsVersion == 3)
    numFeatures = NNInputs::NUM_FEATURES_GLOBAL_V3;
  else if(inputsVersion == 4)
    numFeatures = NNInputs::NUM_FEATURES_GLOBAL_V4;
  else if(inputsVersion == 5)
    numFeatures = NNInputs::NUM_FEATURES_GLOBAL_V5;
  else
    testAssert(false);
  (void)numFeatures;

  out << "Channel: " << c;
  out << ": " << row[c] << endl;
}

static string getAndClear(ostringstream& out) {
  string result = out.str();
  out.str("");
  out.clear();
  return result;
}


//==================================================================================================================
//==================================================================================================================
//==================================================================================================================
//==================================================================================================================


void Tests::runNNInputsV3V4Tests() {
  cout << "Running NN inputs V3V4V5 tests" << endl;
  ostringstream out;
  out << std::setprecision(5);

  auto allocateRows = [](int version, int nnXLen, int nnYLen, int& numFeaturesBin, int& numFeaturesGlobal, float*& rowBin, float*& rowGlobal) {
    if(version == 3) {
      numFeaturesBin = NNInputs::NUM_FEATURES_SPATIAL_V3;
      numFeaturesGlobal = NNInputs::NUM_FEATURES_GLOBAL_V3;
      rowBin = new float[NNInputs::NUM_FEATURES_SPATIAL_V3 * nnXLen * nnYLen];
      rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V3];
    }
    else if(version == 4) {
      numFeaturesBin = NNInputs::NUM_FEATURES_SPATIAL_V4;
      numFeaturesGlobal = NNInputs::NUM_FEATURES_GLOBAL_V4;
      rowBin = new float[NNInputs::NUM_FEATURES_SPATIAL_V4 * nnXLen * nnYLen];
      rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V4];
    }
    else if(version == 5) {
      numFeaturesBin = NNInputs::NUM_FEATURES_SPATIAL_V5;
      numFeaturesGlobal = NNInputs::NUM_FEATURES_GLOBAL_V5;
      rowBin = new float[NNInputs::NUM_FEATURES_SPATIAL_V5 * nnXLen * nnYLen];
      rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V5];
    }
    else
      testAssert(false);
  };

  auto fillRows = [](int version, Hash128& hash,
                     Board& board, const BoardHistory& hist, Player nextPla, double drawEquivalentWinsForWhite, int nnXLen, int nnYLen, bool inputsUseNHWC,
                     float* rowBin, float* rowGlobal) {
    if(version == 3) {
      hash = NNInputs::getHashV3(board,hist,nextPla,drawEquivalentWinsForWhite);
      NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
    }
    else if(version == 4) {
      hash = NNInputs::getHashV4(board,hist,nextPla,drawEquivalentWinsForWhite);
      NNInputs::fillRowV4(board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
    }
    else if(version == 5) {
      hash = NNInputs::getHashV5(board,hist,nextPla,drawEquivalentWinsForWhite);
      NNInputs::fillRowV5(board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
    }
    else
      testAssert(false);
  };

  int minVersion = 3;
  int maxVersion = 5;

  {
    const char* name = "NN Inputs V3V4V5 Basic";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    const string sgfStr = "(;FF[4]KM[7.5];B[pd];W[pq];B[dq];W[dd];B[qo];W[pl];B[qq];W[qr];B[pp];W[rq];B[oq];W[qp];B[pr];W[qq];B[oo];W[ro];B[qn];W[do];B[dl];W[gp];B[eo];W[en];B[fo];W[dp];B[eq];W[cq];B[cr];W[br];B[dn];W[bp];B[cn];W[ep];B[fp];W[fq];B[gq];W[fr];B[gr];W[er];B[hp];W[go];B[fn];W[ho];B[ip];W[io];B[jp];W[jo];B[lp];W[kp];B[kq];W[ko];B[lq];W[ir];B[hq];W[jq];B[jr];W[em];B[gm];W[el];B[hl];W[kl];B[ek];W[fk];B[ej];W[fl];B[fj];W[gk];B[ik];W[gj];B[jj];W[dm];B[lk];W[mm];B[nl];W[nm];B[om];W[ol];B[nk];W[ll];B[kk];W[jl];B[im];W[jk];B[ij];W[kj];B[mk];W[ki];B[ih];W[jh];B[ig];W[jg];B[if];W[oi];B[mi];W[mh];B[lh];W[li];B[nh];W[mj];B[ni];W[nj];B[oj];W[lj];B[ok];W[oh];B[ng];W[pj];B[ji];W[kh];B[jf];W[lg];B[cm];W[cl];B[dk];W[bl];B[bk];W[bn];B[ck];W[bm];B[cc];W[cd];B[dc];W[ec];B[eb];W[fb];B[fc];W[ed];B[gb];W[bc];B[cb];W[cg];B[be];W[bd];B[bg];W[bh];B[cf];W[df];B[ch];W[dg];B[bi];W[qd];B[qc];W[rc];B[rd];W[qe];B[re];W[rb];B[pc];W[qb];B[qf];W[ff];B[sc];W[pb];B[bo];W[ob];B[nc];W[nb];B[mb];W[mc];B[lb])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    for(int version = minVersion; version <= maxVersion; version++) {
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = Rules::getTrompTaylorish();
      sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
      vector<Move>& moves = sgf->moves;

      for(size_t i = 0; i<moves.size(); i++) {
        hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
        nextPla = getOpp(moves[i].pla);
      }

      int nnXLen = 19;
      int nnYLen = 19;
      double drawEquivalentWinsForWhite = 0.2;

      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      auto run = [&](bool inputsUseNHWC) {
        Hash128 hash;
        fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        out << hash << endl;
        for(int c = 0; c<numFeaturesBin; c++)
          printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
        for(int c = 0; c<numFeaturesGlobal; c++)
          printNNInputGlobal(out,version,rowGlobal,c);
        return getAndClear(out);
      };

      string actualNHWC = run(true);
      string actualNCHW = run(false);

      expect(name,actualNHWC,actualNCHW);
      cout << actualNHWC << endl;

      delete[] rowBin;
      delete[] rowGlobal;
    }

    delete sgf;
  }

  {
    const char* name = "NN Inputs V3V4V5 Ko";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    const string sgfStr = "(;FF[4]KM[0.5];B[rj];W[ri];B[si];W[rh];B[sh];W[sg];B[rk];W[sk];B[sl];W[sj];B[eg];W[fg];B[ff];W[gf];B[fh];W[gh];B[gg];W[hg];B[si];W[fg];B[sh];W[sk];B[gg])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    for(int version = minVersion; version <= maxVersion; version++) {
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = Rules::getTrompTaylorish();
      sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
      vector<Move>& moves = sgf->moves;

      for(size_t i = 0; i<moves.size(); i++) {
        hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
        nextPla = getOpp(moves[i].pla);
      }

      int nnXLen = 19;
      int nnYLen = 19;
      double drawEquivalentWinsForWhite = 0.3;

      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      auto run = [&](bool inputsUseNHWC) {
        Hash128 hash;
        fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        out << hash << endl;
        int c = version <= 5 ? 6 : 3;
        printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
        for(c = 0; c<numFeaturesGlobal; c++)
          printNNInputGlobal(out,version,rowGlobal,c);
        return getAndClear(out);
      };

      string actualNHWC = run(true);
      string actualNCHW = run(true);

      expect(name,actualNHWC,actualNCHW);
      cout << actualNHWC << endl;

      delete[] rowBin;
      delete[] rowGlobal;
    }

    delete sgf;
  }


  {
    const char* name = "NN Inputs V3 7x7";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    const string sgfStr = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Japanese]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    for(int version = minVersion; version <= maxVersion; version++) {
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = Rules::getTrompTaylorish();
      sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
      vector<Move>& moves = sgf->moves;

      for(size_t i = 0; i<moves.size(); i++) {
        hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
        nextPla = getOpp(moves[i].pla);
      }

      int nnXLen = 7;
      int nnYLen = 7;
      double drawEquivalentWinsForWhite = 0.5;

      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      auto run = [&](bool inputsUseNHWC) {
        Hash128 hash;
        fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        out << hash << endl;
        for(int c = 0; c<numFeaturesBin; c++)
          printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
        for(int c = 0; c<numFeaturesGlobal; c++)
          printNNInputGlobal(out,version,rowGlobal,c);
        return getAndClear(out);
      };

      string actualNHWC = run(true);
      string actualNCHW = run(false);

      expect(name,actualNHWC,actualNCHW);
      cout << actualNHWC << endl;

      delete[] rowBin;
      delete[] rowGlobal;
    }

    delete sgf;
  }

  {
    const char* name = "NN Inputs V3 7x7 embedded in 9x9";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    const string sgfStr = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Japanese]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    for(int version = minVersion; version <= maxVersion; version++) {
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = Rules::getTrompTaylorish();
      sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
      vector<Move>& moves = sgf->moves;

      for(size_t i = 0; i<moves.size(); i++) {
        hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
        nextPla = getOpp(moves[i].pla);
      }

      int nnXLen = 9;
      int nnYLen = 9;
      double drawEquivalentWinsForWhite = 0.8;

      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      auto run = [&](bool inputsUseNHWC) {
        Hash128 hash;
        fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        out << hash << endl;
        for(int c = 0; c<numFeaturesBin; c++)
          printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
        for(int c = 0; c<numFeaturesGlobal; c++)
          printNNInputGlobal(out,version,rowGlobal,c);
        return getAndClear(out);
      };

      string actualNHWC = run(true);
      string actualNCHW = run(false);

      expect(name,actualNHWC,actualNCHW);
      cout << actualNHWC << endl;

      delete[] rowBin;
      delete[] rowGlobal;
    }

    delete sgf;
  }

  {
    const char* name = "NN Inputs V3 Area Komi";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    for(int version = minVersion; version <= 4; version++) {
      cout << "VERSION " << version << endl;
      Board board = Board::parseBoard(7,7,R"%%(
.xo.oo.
xxo.xox
ooooooo
xxx..xx
..xoox.
..xxxxx
..xo.ox
)%%");

      Player nextPla = P_BLACK;
      Rules initialRules = Rules::getTrompTaylorish();
      initialRules.komi = 2;
      BoardHistory hist(board,nextPla,initialRules,0);

      int nnXLen = 7;
      int nnYLen = 7;
      double drawEquivalentWinsForWhite = 0.3;

      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      bool inputsUseNHWC = true;
      Hash128 hash;
      fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);

      int c = 18;
      printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
      c = 19;
      printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
      for(c = 0; c<numFeaturesGlobal; c++)
        printNNInputGlobal(out,version,rowGlobal,c);

      nextPla = P_WHITE;
      hist.clear(board,nextPla,initialRules,0);
      fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
      for(c = 0; c<numFeaturesGlobal; c++)
        printNNInputGlobal(out,version,rowGlobal,c);

      nextPla = P_BLACK;
      initialRules.komi = 1;
      hist.clear(board,nextPla,initialRules,0);
      fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
      for(c = 0; c<numFeaturesGlobal; c++)
        printNNInputGlobal(out,version,rowGlobal,c);

      delete[] rowBin;
      delete[] rowGlobal;

      cout << getAndClear(out) << endl;
    }
  }

  {
    const char* name = "NN Inputs V3 Rules";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    for(int size = 7; size >= 6; size--) {
      Board board = Board(size,size);
      Player nextPla = P_BLACK;

      vector<Rules> rules = {
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, false, 1.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, true, 1.5f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, false, 2.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, true, 2.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, false, 3.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, true, 3.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, false, 4.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, true, 4.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, false, 5.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, true, 5.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, false, 6.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, true, 6.5f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, false, 1.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, true, 1.5f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, false, 2.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, true, 2.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, false, 3.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, true, 3.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, false, 4.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, true, 4.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, false, 5.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, true, 5.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, false, 6.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, true, 6.5f)
      };

      int nnXLen = size;
      int nnYLen = size;
      double drawEquivalentWinsForWhite = 0.47;

      for(int version = minVersion; version <= maxVersion; version++) {
        cout << "VERSION " << version << endl;

        int numFeaturesBin;
        int numFeaturesGlobal;
        float* rowBin;
        float* rowGlobal;
        allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

        for(int c = 0; c<numFeaturesGlobal; c++) {
          for(int i = 0; i<rules.size(); i++) {
            BoardHistory hist(board,nextPla,rules[i],0);
            bool inputsUseNHWC = true;
            Hash128 hash;
            fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
            out << rowGlobal[c] << " ";
          }
          out << endl;
        }

        delete[] rowBin;
        delete[] rowGlobal;

        cout << getAndClear(out) << endl;
      }
    }

  }

  {
    const char* name = "NN Inputs V3 Ko Prohib and pass hist and whitebonus and encorestart";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    //Immediately enters encore via b0 pass w1 pass. Through w19, sets up various ko shapes. Then starts ko captures. b26 pass b27 pass switches to second encore.
    const string sgfStr = "(;GM[1]FF[4]SZ[6]KM[0.00];B[];W[];B[ab];W[bb];B[ba];W[ca];B[ec];W[ed];B[fd];W[fe];B[fb];W[dc];B[db];W[ae];B[ea];W[bf];B[be];W[ad];B[cf];W[dd];B[af];W[aa];B[];W[fc];B[bd];W[eb];B[];W[];B[ec];W[bf];B[ac];W[eb];B[af];W[eb])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);
    vector<Move>& moves = sgf->moves;

    for(int version = minVersion; version <= 4; version++) {
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, false, 0.0f);
      sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);

      int nnXLen = 6;
      int nnYLen = 6;
      double drawEquivalentWinsForWhite = 0.0;

      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      for(size_t i = 0; i<moves.size(); i++) {
        testAssert(hist.isLegal(board,moves[i].loc,moves[i].pla));
        hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
        nextPla = getOpp(moves[i].pla);

        if(i == 24 || i == 25 || i == 26 || i == 30 || i == 31 || i == 32 || i == 33 || i == 34) {
          out << "Move " << i << endl;
          for(int y = 0; y<board.y_size; y++) {
            for(int x = 0; x<board.x_size; x++) {
              out << (int)hist.isLegal(board,Location::getLoc(x,y,board.x_size),nextPla);
            }
            out << " ";
          }
          out << endl;
          bool inputsUseNHWC = true;
          Hash128 hash;
          fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
          out << "Pass Hist Channels: ";
          for(int c = 0; c<5; c++)
            out << rowGlobal[c] << " ";
          out << endl;
          out << "Selfkomi channel times 15: " << rowGlobal[5]*15 << endl;
          out << "EncorePhase channel 10,11: " << rowGlobal[10] << " " << rowGlobal[11] << endl;
          out << "PassWouldEndPhase channel 12: " << rowGlobal[12] << endl;
          printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,7);
          printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,8);
          printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,20);
          printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,21);
        }
      }

      delete[] rowBin;
      delete[] rowGlobal;

      cout << getAndClear(out) << endl;
    }

    delete sgf;
  }

  {
    const char* name = "NN Inputs V3 some other test positions";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    const string sgfStr = "(;FF[4]GM[1]SZ[13]PB[s75411712-d5152283-b8c128]PW[s78621440-d5365731-b8c128]HA[0]KM[7.5]RU[koPOSITIONALscoreAREAsui0]RE[B+11.5];B[ck];W[lb];B[ke];W[ld];B[jd];W[kc];B[jc];W[jb];B[ib];W[kk];B[ki];W[kh];B[ja];W[le];B[ic];W[kf];B[lj];W[li];B[kj];W[lk];B[jk];W[jl];B[ik];W[mj];B[kb];W[jj];B[ji];W[ij];B[ii];W[hj];B[lh];W[mi];B[kg];W[jg];B[jh];W[lg];B[hk];W[hi];B[mh];W[gk];B[mk];W[il];B[jf];W[lf];B[ig];W[cc];B[dc];W[cd];B[ed];W[kd];B[dj];W[el];B[eg];W[de];B[ee];W[ec];B[je];W[db];B[fc];W[eb];B[bj];W[fd];B[gc];W[cl];B[df];W[dd];B[cf];W[dl];B[gh];W[fk];B[la];W[hh];B[hg];W[fi];B[gg];W[mc];B[bk];W[fb];B[gb];W[ei];B[gi];W[fe];B[ef];W[ej];B[gj];W[hl];B[bh];W[mg];B[be];W[bd];B[ad];W[bb];B[ae];W[di];B[me];W[ci];B[bi];W[bl];B[ab];W[ba];B[ac];W[ml];B[ga];W[fa];B[al];W[bc];B[bf];W[mj];B[mi];W[mb];B[ge];W[mk];B[dk];W[md];B[ek];W[fj];B[jb];W[fh];B[ff];W[bm];B[ka];W[ce];B[ak];W[cj];B[ch];W[];B[id];W[fl];B[hc];W[am];B[ik];W[jk];B[ma];W[];B[mm];W[gl];B[aa];W[ca];B[dh];W[fg];B[];W[lm];B[bg];W[];B[hd];W[];B[ag];W[];B[hf];W[];B[gd];W[];B[ih];W[];B[li];W[];B[hb];W[];B[af];W[];B[ia];W[];B[kl];W[];B[])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    for(int version = minVersion; version <= 4; version++) {
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules;
      sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
      vector<Move>& moves = sgf->moves;

      int nnXLen = 13;
      int nnYLen = 13;
      double drawEquivalentWinsForWhite = 0.0;

      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      for(size_t i = 0; i<moves.size(); i++) {
        testAssert(hist.isLegal(board,moves[i].loc,moves[i].pla));
        hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
        nextPla = getOpp(moves[i].pla);

        if(i == 163) {
          out << "Move " << i << endl;
          bool inputsUseNHWC = true;
          Hash128 hash;
          fillRows(version,hash,board,hist,nextPla,drawEquivalentWinsForWhite,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
          printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,18);
          printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,19);
        }
      }

      delete[] rowBin;
      delete[] rowGlobal;

      cout << getAndClear(out) << endl;
    }

    delete sgf;

  }
}
