#include "../tests/tests.h"

#include <iomanip>

#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
#include "../dataio/sgf.h"

using namespace std;
using namespace TestCommon;

template <typename T>
static void printNNInputHWAndBoard(
  ostream& out, int inputsVersion, const Board& board, const BoardHistory& hist,
  int nnXLen, int nnYLen, bool inputsUseNHWC, T* row, int c
) {
  int numFeatures;
  static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
  if(inputsVersion == 3)
    numFeatures = NNInputs::NUM_FEATURES_SPATIAL_V3;
  else if(inputsVersion == 4)
    numFeatures = NNInputs::NUM_FEATURES_SPATIAL_V4;
  else if(inputsVersion == 5)
    numFeatures = NNInputs::NUM_FEATURES_SPATIAL_V5;
  else if(inputsVersion == 6)
    numFeatures = NNInputs::NUM_FEATURES_SPATIAL_V6;
  else if(inputsVersion == 7)
    numFeatures = NNInputs::NUM_FEATURES_SPATIAL_V7;
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
        char s = PlayerIO::colorToChar(board.colors[loc]);
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
  static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
  if(inputsVersion == 3)
    numFeatures = NNInputs::NUM_FEATURES_GLOBAL_V3;
  else if(inputsVersion == 4)
    numFeatures = NNInputs::NUM_FEATURES_GLOBAL_V4;
  else if(inputsVersion == 5)
    numFeatures = NNInputs::NUM_FEATURES_GLOBAL_V5;
  else if(inputsVersion == 6)
    numFeatures = NNInputs::NUM_FEATURES_GLOBAL_V6;
  else if(inputsVersion == 7)
    numFeatures = NNInputs::NUM_FEATURES_GLOBAL_V7;
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

static double finalScoreIfGameEndedNow(const BoardHistory& baseHist, const Board& baseBoard) {
  Player pla = P_BLACK;
  Board board(baseBoard);
  BoardHistory hist(baseHist);
  if(hist.moveHistory.size() > 0)
    pla = getOpp(hist.moveHistory[hist.moveHistory.size()-1].pla);
  while(!hist.isGameFinished) {
    hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, pla, NULL);
    pla = getOpp(pla);
  }

  double score = hist.finalWhiteMinusBlackScore;
  hist.endAndScoreGameNow(board);
  testAssert(hist.finalWhiteMinusBlackScore == score);
  return score;
}


//==================================================================================================================
//==================================================================================================================
//==================================================================================================================
//==================================================================================================================


void Tests::runNNInputsV3V4Tests() {
  cout << "Running NN inputs V3V4V5V6 tests" << endl;
  ostringstream out;
  out << std::setprecision(5);

  auto allocateRows = [](int version, int nnXLen, int nnYLen, int& numFeaturesBin, int& numFeaturesGlobal, float*& rowBin, float*& rowGlobal) {
    static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
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
    else if(version == 6) {
      numFeaturesBin = NNInputs::NUM_FEATURES_SPATIAL_V6;
      numFeaturesGlobal = NNInputs::NUM_FEATURES_GLOBAL_V6;
      rowBin = new float[NNInputs::NUM_FEATURES_SPATIAL_V6 * nnXLen * nnYLen];
      rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V6];
    }
    else if(version == 7) {
      numFeaturesBin = NNInputs::NUM_FEATURES_SPATIAL_V7;
      numFeaturesGlobal = NNInputs::NUM_FEATURES_GLOBAL_V7;
      rowBin = new float[NNInputs::NUM_FEATURES_SPATIAL_V7 * nnXLen * nnYLen];
      rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V7];
    }
    else
      testAssert(false);
  };

  auto fillRows = [](int version, Hash128& hash,
                     Board& board, const BoardHistory& hist, Player nextPla, MiscNNInputParams nnInputParams, int nnXLen, int nnYLen, bool inputsUseNHWC,
                     float* rowBin, float* rowGlobal) {
    hash = NNInputs::getHash(board,hist,nextPla,nnInputParams);

    static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
    if(version == 3)
      NNInputs::fillRowV3(board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
    else if(version == 4)
      NNInputs::fillRowV4(board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
    else if(version == 5)
      NNInputs::fillRowV5(board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
    else if(version == 6)
      NNInputs::fillRowV6(board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
    else if(version == 7)
      NNInputs::fillRowV7(board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
    else
      testAssert(false);
  };

  static_assert(NNModelVersion::latestInputsVersionImplemented == 7, "");
  int minVersion = 3;
  int maxVersion = 7;

  {
    const char* name = "NN Inputs V3V4V5V6 Basic";
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
      initialRules = sgf->getRulesOrFailAllowUnspecified(initialRules);
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
        MiscNNInputParams nnInputParams;
        nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
        fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
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
    const char* name = "NN Inputs V3V4V5V6 Ko";
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
      initialRules = sgf->getRulesOrFailAllowUnspecified(initialRules);
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
        MiscNNInputParams nnInputParams;
        nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
        fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        out << hash << endl;
        int c = version != 5 ? 6 : 3;
        printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
        for(c = 0; c<numFeaturesGlobal; c++)
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
    const char* name = "NN Inputs V3V4V5V6 7x7";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    const string sgfStr = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Tromp-Taylor]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    for(int version = minVersion; version <= maxVersion; version++) {
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = Rules::getTrompTaylorish();
      initialRules = sgf->getRulesOrFailAllowUnspecified(initialRules);
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
        MiscNNInputParams nnInputParams;
        nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
        fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
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
    const char* name = "NN Inputs V3V4V5V6 7x7 embedded in 9x9";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    const string sgfStr = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Tromp-Taylor]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    for(int version = minVersion; version <= maxVersion; version++) {
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = Rules::getTrompTaylorish();
      initialRules = sgf->getRulesOrFailAllowUnspecified(initialRules);
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
        MiscNNInputParams nnInputParams;
        nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
        fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
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
    const char* name = "NN Inputs V3V4V6 Area Komi";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    for(int version = minVersion; version <= maxVersion; version++) {
      if(version == 5)
        continue;
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
      MiscNNInputParams nnInputParams;
      nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
      fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);

      int c = 18;
      printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
      c = 19;
      printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
      for(c = 0; c<numFeaturesGlobal; c++)
        printNNInputGlobal(out,version,rowGlobal,c);

      nextPla = P_WHITE;
      hist.clear(board,nextPla,initialRules,0);
      fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
      for(c = 0; c<numFeaturesGlobal; c++)
        printNNInputGlobal(out,version,rowGlobal,c);

      nextPla = P_BLACK;
      initialRules.komi = 1;
      hist.clear(board,nextPla,initialRules,0);
      fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
      for(c = 0; c<numFeaturesGlobal; c++)
        printNNInputGlobal(out,version,rowGlobal,c);

      delete[] rowBin;
      delete[] rowGlobal;

      cout << getAndClear(out) << endl;
    }
  }

  {
    const char* name = "NN Inputs V3V4V5V6 Rules";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    for(int size = 7; size >= 6; size--) {
      Board board = Board(size,size);
      Player nextPla = P_BLACK;

      vector<Rules> rules = {
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 1.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, false, 1.5f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, false, 2.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, true, true, Rules::WHB_ZERO, false, 2.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 3.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, false, 3.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, false, 4.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, true, Rules::WHB_ZERO, false, 4.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 5.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, false, 5.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, false, 6.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, true, Rules::WHB_ZERO, false, 6.5f),

        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 1.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, true, false, Rules::WHB_ZERO, false, 1.5f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, false, 2.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, true, true, Rules::WHB_ZERO, false, 2.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 3.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, true, false, Rules::WHB_ZERO, false, 3.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, false, 4.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, true, true, Rules::WHB_ZERO, false, 4.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 5.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, true, false, Rules::WHB_ZERO, false, 5.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, false, 6.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, true, true, Rules::WHB_ZERO, false, 6.5f),

        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 1.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, false, 1.5f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, false, 2.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, true, Rules::WHB_ZERO, false, 2.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 3.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, false, 3.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, false, 4.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, true, Rules::WHB_ZERO, false, 4.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 5.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, false, 5.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, false, 6.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, true, Rules::WHB_ZERO, false, 6.5f),

        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 1.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, false, Rules::WHB_ZERO, false, 1.5f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, false, 2.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, true, Rules::WHB_ZERO, false, 2.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 3.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, false, Rules::WHB_ZERO, false, 3.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, false, 4.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, true, Rules::WHB_ZERO, false, 4.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 5.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, false, Rules::WHB_ZERO, false, 5.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, false, 6.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, true, Rules::WHB_ZERO, false, 6.5f),

        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, false, 1.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, true, false, Rules::WHB_ZERO, false, 1.5f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, false, 2.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, true, true, Rules::WHB_ZERO, false, 2.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, false, 3.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, true, false, Rules::WHB_ZERO, false, 3.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, false, 4.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, true, true, Rules::WHB_ZERO, false, 4.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, false, 5.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, true, false, Rules::WHB_ZERO, false, 5.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, false, 6.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, true, true, Rules::WHB_ZERO, false, 6.5f),

        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, false, 1.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, false, Rules::WHB_ZERO, false, 1.5f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, false, 2.0f),
        Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, true, Rules::WHB_ZERO, false, 2.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, false, 3.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, false, Rules::WHB_ZERO, false, 3.5f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, false, 4.0f),
        Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, true, Rules::WHB_ZERO, false, 4.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, false, Rules::WHB_ZERO, false, 5.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, false, Rules::WHB_ZERO, false, 5.5f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, false, 6.0f),
        Rules(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, true, Rules::WHB_ZERO, false, 6.5f),
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

        for(int i = 0; i<rules.size(); i += 24) {
          for(int c = 0; c<numFeaturesGlobal; c++) {
            for(int j = 0; j<24; j++) {
              testAssert(i + j < rules.size());
              BoardHistory hist(board,nextPla,rules[i+j],0);
              bool inputsUseNHWC = true;
              Hash128 hash;
              MiscNNInputParams nnInputParams;
              nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
              fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
              out << rowGlobal[c] << " ";
            }
            out << endl;
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
    const char* name = "NN Inputs V3V4V6 Ko Prohib and pass hist and whitebonus and encorestart";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    //Immediately enters encore via b0 pass w1 pass. Through w19, sets up various ko shapes. Then starts ko captures. b26 pass b27 pass switches to second encore.
    const string sgfStr = "(;GM[1]FF[4]SZ[6]KM[0.00];B[];W[];B[ab];W[bb];B[ba];W[ca];B[ec];W[ed];B[fd];W[fe];B[fb];W[dc];B[db];W[ae];B[ea];W[bf];B[be];W[ad];B[cf];W[dd];B[af];W[aa];B[];W[fc];B[bd];W[eb];B[];W[];B[ec];W[bf];B[ac];W[eb];B[af];W[eb])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);
    vector<Move>& moves = sgf->moves;

    for(int version = minVersion; version <= maxVersion; version++) {
      if(version == 5)
        continue;
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules = Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 0.0f);
      initialRules = sgf->getRulesOrFailAllowUnspecified(initialRules);
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
          MiscNNInputParams nnInputParams;
          nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
          fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
          out << "Pass Hist Channels: ";
          for(int c = 0; c<5; c++)
            out << rowGlobal[c] << " ";
          out << endl;
          out << "Selfkomi channel times 15: " << rowGlobal[5]*15 << endl;
          if(version >= 6) {
            out << "EncorePhase channel 12,13: " << rowGlobal[12] << " " << rowGlobal[13] << endl;
            out << "PassWouldEndPhase channel 14: " << rowGlobal[14] << endl;
          }
          else {
            out << "EncorePhase channel 10,11: " << rowGlobal[10] << " " << rowGlobal[11] << endl;
            out << "PassWouldEndPhase channel 12: " << rowGlobal[12] << endl;
          }
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
    const char* name = "NN Inputs V3V4V6 some other test positions";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    const string sgfStr = "(;FF[4]GM[1]SZ[13]PB[s75411712-d5152283-b8c128]PW[s78621440-d5365731-b8c128]HA[0]KM[7.5]RU[koPOSITIONALscoreAREAsui0]RE[B+11.5];B[ck];W[lb];B[ke];W[ld];B[jd];W[kc];B[jc];W[jb];B[ib];W[kk];B[ki];W[kh];B[ja];W[le];B[ic];W[kf];B[lj];W[li];B[kj];W[lk];B[jk];W[jl];B[ik];W[mj];B[kb];W[jj];B[ji];W[ij];B[ii];W[hj];B[lh];W[mi];B[kg];W[jg];B[jh];W[lg];B[hk];W[hi];B[mh];W[gk];B[mk];W[il];B[jf];W[lf];B[ig];W[cc];B[dc];W[cd];B[ed];W[kd];B[dj];W[el];B[eg];W[de];B[ee];W[ec];B[je];W[db];B[fc];W[eb];B[bj];W[fd];B[gc];W[cl];B[df];W[dd];B[cf];W[dl];B[gh];W[fk];B[la];W[hh];B[hg];W[fi];B[gg];W[mc];B[bk];W[fb];B[gb];W[ei];B[gi];W[fe];B[ef];W[ej];B[gj];W[hl];B[bh];W[mg];B[be];W[bd];B[ad];W[bb];B[ae];W[di];B[me];W[ci];B[bi];W[bl];B[ab];W[ba];B[ac];W[ml];B[ga];W[fa];B[al];W[bc];B[bf];W[mj];B[mi];W[mb];B[ge];W[mk];B[dk];W[md];B[ek];W[fj];B[jb];W[fh];B[ff];W[bm];B[ka];W[ce];B[ak];W[cj];B[ch];W[];B[id];W[fl];B[hc];W[am];B[ik];W[jk];B[ma];W[];B[mm];W[gl];B[aa];W[ca];B[dh];W[fg];B[];W[lm];B[bg];W[];B[hd];W[];B[ag];W[];B[hf];W[];B[gd];W[];B[ih];W[];B[li];W[];B[hb];W[];B[af];W[];B[ia];W[];B[kl];W[];B[])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    for(int version = minVersion; version <= maxVersion; version++) {
      if(version == 5)
        continue;
      cout << "VERSION " << version << endl;
      Board board;
      Player nextPla;
      BoardHistory hist;
      Rules initialRules;
      initialRules = sgf->getRulesOrFailAllowUnspecified(initialRules);
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
          MiscNNInputParams nnInputParams;
          nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
          fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
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


  {
    const char* name = "NN Inputs V6 Area Feature and Komi";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    for(int version = 6; version <= maxVersion; version++) {
      cout << "VERSION " << version << endl;

      int nnXLen = 7;
      int nnYLen = 7;
      double drawEquivalentWinsForWhite = 0.5;
      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      for(int goToEncore2 = 0; goToEncore2 <= 1; goToEncore2++) {
        int scoringRules[4] = {Rules::SCORING_AREA, Rules::SCORING_AREA, Rules::SCORING_TERRITORY, Rules::SCORING_TERRITORY};
        int taxRules[4] = {Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_NONE, Rules::TAX_SEKI};
        for(int whichRules = 0; whichRules < 4; whichRules++) {
          Board board = Board::parseBoard(7,7,R"%%(
...oxx.
oooox.x
xxxxoxx
o.xoooo
.oxox.o
oxxo.x.
o.xoo.x
)%%");
          Rules rules;
          rules.koRule = Rules::KO_POSITIONAL;
          rules.scoringRule = scoringRules[whichRules];
          rules.komi = 6.5f;
          rules.multiStoneSuicideLegal = false;
          rules.taxRule = taxRules[whichRules];
          BoardHistory hist(board,P_WHITE,rules,0);

          auto run = [&](bool inputsUseNHWC) {
            Player nextPla = hist.moveHistory.size() > 0 ? getOpp(hist.moveHistory[hist.moveHistory.size()-1].pla) : hist.initialPla;
            Hash128 hash;
            MiscNNInputParams nnInputParams;
            nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
            fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
            out << hash << endl;
            printNNInputGlobal(out,version,rowGlobal,5);
            int c = 18;
            printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
            c = 19;
            printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,c);
            return getAndClear(out);
          };
          auto printScoring = [&]() {
            Board b(board);
            BoardHistory h(hist);
            Color area[Board::MAX_ARR_SIZE];
            float scoring[Board::MAX_ARR_SIZE];
            h.endAndScoreGameNow(b,area);
            NNInputs::fillScoring(b,area,false,scoring);
            for(int y = 0; y<board.y_size; y++) {
              for(int x = 0; x<board.x_size; x++) {
                Loc loc = Location::getLoc(x,y,board.x_size);
                cout << Global::strprintf("%4.0f",100*scoring[loc]) << " ";
              }
              cout << endl;
            }
            cout << endl;
          };
          auto runBoth = [&]() {
            string actualNHWC = run(true);
            string actualNCHW = run(false);
            expect(name,actualNHWC,actualNCHW);
            cout << "Rules: " << hist.rules << endl;
            cout << "Komi and Bonus: " << hist.rules.komi << " " << hist.whiteBonusScore << endl;
            cout << "Encore phase: " << hist.encorePhase << endl;
            for(int i = 0; i<hist.moveHistory.size(); i++)
              cout << Location::toString(hist.moveHistory[i].loc,board) << " ";
            cout << actualNHWC;
            printScoring();
          };

          cout << "=========================================== " << endl;
          cout << "goToEncore2 = " << goToEncore2 << endl;

          runBoth();
          if((bool)goToEncore2) {
            hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
            hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
            runBoth();
            hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_WHITE, NULL);
            hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, P_BLACK, NULL);
            runBoth();
          }
          hist.makeBoardMoveAssumeLegal(board, Location::getLoc(6,5,board.x_size), P_WHITE, NULL);
          runBoth();
          hist.makeBoardMoveAssumeLegal(board, Location::getLoc(5,6,board.x_size), P_BLACK, NULL);
          runBoth();
          hist.makeBoardMoveAssumeLegal(board, Location::getLoc(0,4,board.x_size), P_WHITE, NULL);
          runBoth();
          hist.makeBoardMoveAssumeLegal(board, Location::getLoc(6,0,board.x_size), P_BLACK, NULL);
          runBoth();
          hist.makeBoardMoveAssumeLegal(board, Location::getLoc(1,0,board.x_size), P_WHITE, NULL);
          runBoth();
          hist.makeBoardMoveAssumeLegal(board, Location::getLoc(4,5,board.x_size), P_BLACK, NULL);
          runBoth();
          hist.makeBoardMoveAssumeLegal(board, Location::getLoc(5,4,board.x_size), P_WHITE, NULL);
          runBoth();
          cout << endl;
        }
      }

      delete[] rowBin;
      delete[] rowGlobal;
    }
  }

  {
    const char* name = "NN Inputs V6 Pass history";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    for(int version = 6; version <= maxVersion; version++) {
      cout << "VERSION " << version << endl;

      int nnXLen = 9;
      int nnYLen = 1;
      double drawEquivalentWinsForWhite = 0.5;
      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      Board board = Board(9,1);
      Player nextPla = P_BLACK;
      Rules initialRules = Rules::getSimpleTerritory();
      BoardHistory hist(board,nextPla,initialRules,0);

      auto run = [&](bool inputsUseNHWC) {
        Hash128 hash;
        MiscNNInputParams nnInputParams;
        nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
        fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,9);
        printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,10);
        printNNInputHWAndBoard(out,version,board,hist,nnXLen,nnYLen,inputsUseNHWC,rowBin,11);
        printNNInputGlobal(out,version,rowGlobal,0);
        printNNInputGlobal(out,version,rowGlobal,1);
        printNNInputGlobal(out,version,rowGlobal,2);
        return getAndClear(out);
      };
      auto runBoth = [&]() {
        string actualNHWC = run(true);
        string actualNCHW = run(false);
        expect(name,actualNHWC,actualNCHW);
        for(int i = 0; i<hist.moveHistory.size(); i++)
          cout << Location::toString(hist.moveHistory[i].loc,board) << " ";
        cout << endl;
        cout << actualNHWC;
        cout << "-----" << endl;
      };

      vector<Loc> locs = Location::parseSequence("pass A1 D1 F1 H1 E1 G1 B1 J1 F1 J1 G1 C1 E1 B1 pass pass H1 J1 E1 G1 pass pass A1 pass D1 C1 B1", board);

      for(int i = 0; i<locs.size(); i++) {
        runBoth();
        hist.makeBoardMoveAssumeLegal(board,locs[i],nextPla,NULL);
        nextPla = getOpp(nextPla);
      }

      delete[] rowBin;
      delete[] rowGlobal;
    }
  }

  {
    const char* name = "NN Inputs SelfKomi Handicap White Bonus";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    int size = 7;

    vector<Rules> rules = {
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 3.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_N_MINUS_ONE, false, 3.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_N, false, 3.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, false, Rules::WHB_N_MINUS_ONE, false, 3.0f),
      Rules(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, true, Rules::WHB_N, false, 3.0f),
    };

    int nnXLen = size;
    int nnYLen = size;
    double drawEquivalentWinsForWhite = 0.5;

    for(int version = maxVersion; version <= maxVersion; version++) {
      cout << "VERSION " << version << endl;

      int numFeaturesBin;
      int numFeaturesGlobal;
      float* rowBin;
      float* rowGlobal;
      allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

      auto test = [&](const Board& board, const BoardHistory& hist, Player nextPla) {
        bool inputsUseNHWC = true;
        Hash128 hash;
        Board b = board;
        MiscNNInputParams nnInputParams;
        nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
        fillRows(version,hash,b,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        for(int c = 0; c<numFeaturesGlobal; c++)
          cout << rowGlobal[c] << " ";
        cout << endl;
      };

      for(int i = 0; i<rules.size(); i++) {
        Board board = Board(size,size);
        Player nextPla = P_BLACK;
        Rules initialRules = rules[i];
        BoardHistory hist(board,nextPla,initialRules,0);
        cout << "----------------------------------------" << endl;
        cout << "Black makes 3 moves in a row" << endl;
        if(i >= 3) {
          cout << "Set assumeMultipleStartingBlackMovesAreHandicap" << endl;
          hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
        }
        hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,3,board.x_size), P_BLACK, NULL);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;
        hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,2,board.x_size), P_BLACK, NULL);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;
        hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3,1,board.x_size), P_BLACK, NULL);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;

        cout << endl;
        cout << "Set assumeMultipleStartingBlackMovesAreHandicap" << endl;
        hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;

        cout << endl;
        cout << "One more move" << endl;
        hist.makeBoardMoveAssumeLegal(board, Location::getLoc(5,1,board.x_size), P_BLACK, NULL);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;

        cout << endl;
        cout << "Reclear history" << endl;
        hist.clear(board,nextPla,initialRules,0);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;

        cout << endl;
        cout << "One more move" << endl;
        hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2,1,board.x_size), P_BLACK, NULL);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;

        cout << endl;
        cout << "Set assumeMultipleStartingBlackMovesAreHandicap" << endl;
        hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;

        cout << endl;
        cout << "Unset assumeMultipleStartingBlackMovesAreHandicap" << endl;
        hist.setAssumeMultipleStartingBlackMovesAreHandicap(false);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;

        cout << endl;
        cout << "Play white move" << endl;
        hist.makeBoardMoveAssumeLegal(board, Location::getLoc(4,1,board.x_size), P_WHITE, NULL);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;

        cout << endl;
        cout << "Reclear history" << endl;
        hist.clear(board,nextPla,initialRules,0);
        hist.printDebugInfo(cout,board);
        test(board,hist,nextPla);
        cout << "Final score: " << finalScoreIfGameEndedNow(hist,board) << endl;
      }
      delete[] rowBin;
      delete[] rowGlobal;
    }
  }


  {
    cout << "Passing hack condition based on as-is scoring ==========================================================================" << endl;
    cout << endl;

    auto testScoring = [](const Board& board, const BoardHistory& hist, bool expectedFriendlyPassSuppress) {
      hist.printDebugInfo(cout,board);
      cout << "Finished or past normal end: " << (hist.isGameFinished || hist.isPastNormalPhaseEnd) << endl;
      cout << "Pass would end phase: " << hist.passWouldEndPhase(board,hist.presumedNextMovePla) << endl;
      cout << "Pass would end game: " << hist.passWouldEndGame(board,hist.presumedNextMovePla) << endl;
      cout << "Pass should suppress: " << hist.shouldSuppressEndGameFromFriendlyPass(board,hist.presumedNextMovePla) << endl;

      int nnXLen = 7;
      int nnYLen = 7;
      bool inputsUseNHWC = false;
      float* rowBin = new float[NNInputs::NUM_FEATURES_SPATIAL_V7 * nnXLen * nnYLen];
      float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V7];

      MiscNNInputParams nnInputParams;
      nnInputParams.drawEquivalentWinsForWhite = 0.5;

      // Currently should be true given that we only test game-end stuff, not illegal moves.
      testAssert((hist.numApproxValidTurnsThisPhase > 0) == (hist.numTurnsThisPhase > 0));

      {
        BoardHistory histCopy(hist);
        nnInputParams.enablePassingHacks = false;
        histCopy.rules.friendlyPassOk = false;
        NNInputs::fillRowV7(board,histCopy,histCopy.presumedNextMovePla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        Color area[Board::MAX_ARR_SIZE];
        histCopy.endAndScoreGameNow(board,area);
        cout << "SCORE NOW " << histCopy.finalWhiteMinusBlackScore << endl;

        float historyInput = 0.0f;
        for(int y = 0; y<nnYLen; y++) {
          for(int x = 0; x<nnXLen; x++) {
            Loc loc = Location::getLoc(x,y,board.x_size);
            if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.encorePhase < 2) {
              testAssert(rowBin[18 * nnXLen * nnYLen + y * nnXLen + x] == 0.0f);
              testAssert(rowBin[19 * nnXLen * nnYLen + y * nnXLen + x] == 0.0f);
            }
            else {
              testAssert((rowBin[18 * nnXLen * nnYLen + y * nnXLen + x] == 1.0f) == (area[loc] == histCopy.presumedNextMovePla));
              testAssert((rowBin[19 * nnXLen * nnYLen + y * nnXLen + x] == 1.0f) == (area[loc] == getOpp(histCopy.presumedNextMovePla)));
            }
            historyInput += rowBin[9 * nnXLen * nnYLen + y * nnXLen + x];
          }
        }
        historyInput += rowGlobal[0];
        testAssert((hist.moveHistory.size() > 0 && hist.numApproxValidTurnsThisPhase > 0) == (historyInput > 0.0f));
      }

      {
        BoardHistory histCopy(hist);
        nnInputParams.enablePassingHacks = false;
        histCopy.rules.friendlyPassOk = true;
        NNInputs::fillRowV7(board,histCopy,histCopy.presumedNextMovePla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        Color area[Board::MAX_ARR_SIZE];
        histCopy.endAndScoreGameNow(board,area);

        float historyInput = 0.0f;
        for(int y = 0; y<nnYLen; y++) {
          for(int x = 0; x<nnXLen; x++) {
            Loc loc = Location::getLoc(x,y,board.x_size);
            if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.encorePhase < 2) {
              testAssert(rowBin[18 * nnXLen * nnYLen + y * nnXLen + x] == 0.0f);
              testAssert(rowBin[19 * nnXLen * nnYLen + y * nnXLen + x] == 0.0f);
            }
            else {
              testAssert((rowBin[18 * nnXLen * nnYLen + y * nnXLen + x] == 1.0f) == (area[loc] == histCopy.presumedNextMovePla));
              testAssert((rowBin[19 * nnXLen * nnYLen + y * nnXLen + x] == 1.0f) == (area[loc] == getOpp(histCopy.presumedNextMovePla)));
            }
            historyInput += rowBin[9 * nnXLen * nnYLen + y * nnXLen + x];
          }
        }
        historyInput += rowGlobal[0];
        testAssert((hist.moveHistory.size() > 0 && hist.numApproxValidTurnsThisPhase > 0 && !expectedFriendlyPassSuppress) == (historyInput > 0.0f));
      }

      Color area[Board::MAX_ARR_SIZE];
      for(float komi = -100.0f; komi <= 100.0f; komi += 0.5f) {
        BoardHistory histCopy(hist);
        nnInputParams.enablePassingHacks = true;
        histCopy.rules.friendlyPassOk = false;
        histCopy.rules.komi = komi;
        NNInputs::fillRowV7(board,histCopy,histCopy.presumedNextMovePla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);
        histCopy.endAndScoreGameNow(board,area);

        float historyInput = 0.0f;
        for(int y = 0; y<nnYLen; y++) {
          for(int x = 0; x<nnXLen; x++) {
            Loc loc = Location::getLoc(x,y,board.x_size);
            if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.encorePhase < 2) {
              testAssert(rowBin[18 * nnXLen * nnYLen + y * nnXLen + x] == 0.0f);
              testAssert(rowBin[19 * nnXLen * nnYLen + y * nnXLen + x] == 0.0f);
            }
            else {
              testAssert((rowBin[18 * nnXLen * nnYLen + y * nnXLen + x] == 1.0f) == (area[loc] == histCopy.presumedNextMovePla));
              testAssert((rowBin[19 * nnXLen * nnYLen + y * nnXLen + x] == 1.0f) == (area[loc] == getOpp(histCopy.presumedNextMovePla)));
            }
            historyInput += rowBin[9 * nnXLen * nnYLen + y * nnXLen + x];
          }
        }
        historyInput += rowGlobal[0];
        if(std::abs(histCopy.finalWhiteMinusBlackScore) <= 1.0f) {
          cout << "Komi " << komi << " hasHistory " << (historyInput > 0.0f) << endl;
        }
        if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && hist.encorePhase < 2)
          testAssert((hist.moveHistory.size() > 0 && hist.numApproxValidTurnsThisPhase > 0) == (historyInput > 0.0f));
        else
          testAssert((hist.moveHistory.size() > 0 && hist.numApproxValidTurnsThisPhase > 0 &&
                      (!hist.passWouldEndGame(board,hist.presumedNextMovePla) || histCopy.winner == hist.presumedNextMovePla)) == (historyInput > 0.0f));
      }

      delete[] rowBin;
      delete[] rowGlobal;
    };

    Board origBoard = Board::parseBoard(7,7,R"%%(
.x.o.o.
x.xoooo
xxxxxxx
ooxxooo
.oxo.o.
ooxooxo
.oxo.x.
)%%");

    {
      Rules rules = Rules::parseRules("tromp-taylor");
      cout << "---------------------------------------------------" << endl;
      cout << rules.toString() << endl;
      Board board(origBoard);
      BoardHistory hist(board,P_WHITE,rules,0);

      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("C7",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,true);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
    }

    {
      Rules rules = Rules::parseRules("chinese");
      cout << "---------------------------------------------------" << endl;
      cout << rules.toString() << endl;
      Board board(origBoard);
      BoardHistory hist(board,P_WHITE,rules,0);

      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("C7",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,true);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
    }

    {
      Rules rules = Rules::parseRules("chinese");
      rules.taxRule = Rules::TAX_SEKI;
      cout << "---------------------------------------------------" << endl;
      cout << rules.toString() << endl;
      Board board(origBoard);
      BoardHistory hist(board,P_WHITE,rules,0);

      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("C7",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,true);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
    }

    {
      Rules rules = Rules::parseRules("stonescoring");
      cout << "---------------------------------------------------" << endl;
      cout << rules.toString() << endl;
      Board board(origBoard);
      BoardHistory hist(board,P_WHITE,rules,0);

      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("C7",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,true);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
    }

    {
      Rules rules = Rules::parseRules("japanese");
      rules.taxRule = Rules::TAX_NONE;
      cout << "---------------------------------------------------" << endl;
      cout << rules.toString() << endl;
      Board board(origBoard);
      BoardHistory hist(board,P_WHITE,rules,0);

      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("C7",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("E3",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G1",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G3",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G2",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
    }

    {
      Rules rules = Rules::parseRules("japanese");
      cout << "---------------------------------------------------" << endl;
      cout << rules.toString() << endl;
      Board board(origBoard);
      BoardHistory hist(board,P_WHITE,rules,0);

      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("C7",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("E3",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G1",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G3",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G2",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
    }


    {
      Rules rules = Rules::parseRules("japanese");
      rules.taxRule = Rules::TAX_ALL;
      cout << "---------------------------------------------------" << endl;
      cout << rules.toString() << endl;
      Board board(origBoard);
      BoardHistory hist(board,P_WHITE,rules,0);

      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("C7",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("E3",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G1",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G3",board),P_BLACK,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("G2",board),P_WHITE,NULL);
      testScoring(board,hist,false);
      hist.makeBoardMoveAssumeLegal(board,Location::ofString("pass",board),P_BLACK,NULL);
      testScoring(board,hist,false);
    }

    cout << "Ok" << endl;
  }

  {
    const char* name = "NN Inputs history scoring past end of game or end of phase";
    cout << "-----------------------------------------------------------------" <<  endl;
    cout << name << endl;
    cout << "-----------------------------------------------------------------" <<  endl;

    const vector<string> sgfStrs = {
      "(;GM[1]FF[4]SZ[9]KM[7];B[ff];W[ee];B[dd];W[];B[];W[];B[cc];W[bb];B[];W[])",
      "(;GM[1]FF[4]SZ[9]KM[-7];B[ff];W[ee];B[dd];W[];B[];W[];B[cc];W[bb];B[];W[])",
    };

    for(int whichRules = 0; whichRules <= 3; whichRules++) {
      Rules rulesToUse = Rules::parseRules("tromp-taylor");
      if(whichRules == 1) {
        rulesToUse = Rules::parseRules("japanese");
        rulesToUse.friendlyPassOk = false;
      }
      if(whichRules == 2) {
        rulesToUse = Rules::parseRules("tromp-taylor");
        rulesToUse.hasButton = true;
      }
      if(whichRules == 3) {
        rulesToUse = Rules::parseRules("tromp-taylor");
        rulesToUse.friendlyPassOk = true;
      }

      for(const string& sgfStr : sgfStrs) {
        cout << sgfStr << endl;
        CompactSgf* sgf = CompactSgf::parse(sgfStr);
        vector<Move>& moves = sgf->moves;

        for(int whichMode = 0; whichMode <= 2; whichMode++) {
          bool enablePassingHacks = false;
          bool conservativePassAndIsRoot = false;
          if(whichMode == 1)
            enablePassingHacks = true;
          if(whichMode == 2) {
            enablePassingHacks = true;
            conservativePassAndIsRoot = true;
          }
          for(int version = 6; version <= maxVersion; version++) {
            cout << "rules " << rulesToUse.toString() << endl;
            cout << "enablePassingHacks " << enablePassingHacks << " conservativePassAndIsRoot " << conservativePassAndIsRoot << endl;
            cout << "VERSION " << version << endl;
            Board board;
            Player nextPla;
            BoardHistory hist;
            Rules rules = sgf->getRulesOrFailAllowUnspecified(rulesToUse);
            sgf->setupInitialBoardAndHist(rules, board, nextPla, hist);

            int nnXLen = 9;
            int nnYLen = 9;
            double drawEquivalentWinsForWhite = 0.5;

            int numFeaturesBin;
            int numFeaturesGlobal;
            float* rowBin;
            float* rowGlobal;
            allocateRows(version,nnXLen,nnYLen,numFeaturesBin,numFeaturesGlobal,rowBin,rowGlobal);

            for(size_t i = 0; i<moves.size()+1; i++) {
              bool inputsUseNHWC = false;
              Hash128 hash;
              MiscNNInputParams nnInputParams;
              nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
              nnInputParams.enablePassingHacks = enablePassingHacks;
              nnInputParams.conservativePassAndIsRoot = conservativePassAndIsRoot;
              fillRows(version,hash,board,hist,nextPla,nnInputParams,nnXLen,nnYLen,inputsUseNHWC,rowBin,rowGlobal);

              auto histLocStr = [&](int lookback) {
                int numPosesFound = 0;
                int histPos = 0;
                for(int y = 0; y<board.y_size; y++) {
                  for(int x = 0; x<board.x_size; x++) {
                    if(rowBin[nnXLen*nnYLen*(9+lookback-1) + y*nnXLen + x] > 0) {
                      numPosesFound += 1;
                      histPos += y*nnXLen + x;
                    }
                  }
                }
                if(rowGlobal[lookback-1] > 0) {
                  numPosesFound += 1;
                  histPos += nnYLen*nnXLen;
                }
                testAssert(numPosesFound <= 1);
                if(numPosesFound == 1)
                  return Location::toString(NNPos::posToLoc(histPos,board.x_size,board.y_size,nnXLen,nnYLen), board);
                return Location::toString(Board::NULL_LOC, board);
              };

              out << "encorephase " << hist.encorePhase
                  << " finished " << hist.isGameFinished
                  << " numTurnsThisPhase " << hist.numTurnsThisPhase
                  << " numApproxValidTurnsThisPhase " << hist.numApproxValidTurnsThisPhase << endl;
              out << hash << endl;
              out << "History that net sees "
                  << histLocStr(1) << " " << histLocStr(2) << " " << histLocStr(3) << " " << histLocStr(4) << " " << histLocStr(5) << endl;

              if(i == (int)moves.size())
                break;

              out << "Move " << i << " " << Location::toString(moves[i].loc, board) << endl;
              testAssert(hist.isLegal(board,moves[i].loc,moves[i].pla));
              bool preventEncore = true;
              hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL,preventEncore);
              nextPla = getOpp(moves[i].pla);
            }

            delete[] rowBin;
            delete[] rowGlobal;

            cout << getAndClear(out) << endl;
          }
        }
        delete sgf;
      }
    }
  }
}

