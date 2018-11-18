#include "../tests/tests.h"
#include "../neuralnet/nninputs.h"
#include "../dataio/sgf.h"
using namespace TestCommon;

#include <iomanip>

template <typename T>
static void printNNInputHWAndBoard(ostream& out, int inputsVersion, const Board& board, const BoardHistory& hist, int posLen, bool inputsUseNHWC, T* row, int c) {
  int numFeatures;
  if(inputsVersion == 2)
    numFeatures = NNInputs::NUM_FEATURES_V2;
  else if(inputsVersion == 3)
    numFeatures = NNInputs::NUM_FEATURES_BIN_V3;
  else
    testAssert(false);

  out << "Channel: " << c << endl;

  for(int y = 0; y<posLen; y++) {
    for(int x = 0; x<posLen; x++) {
      int pos = NNPos::xyToPos(x,y,posLen);
      if(x > 0)
        out << " ";
      if(inputsUseNHWC)
        out << row[pos * numFeatures + c];
      else
        out << row[c * posLen * posLen + pos];
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

void Tests::runNNInputsV2Tests() {
  cout << "Running NN inputs V2 tests" << endl;
  ostringstream out;
  out << std::setprecision(3);

  {
    const char* name = "NN Inputs V2 Basic";

    const string sgfStr = "(;FF[4]KM[7.5];B[pd];W[pq];B[dq];W[dd];B[qo];W[pl];B[qq];W[qr];B[pp];W[rq];B[oq];W[qp];B[pr];W[qq];B[oo];W[ro];B[qn];W[do];B[dl];W[gp];B[eo];W[en];B[fo];W[dp];B[eq];W[cq];B[cr];W[br];B[dn];W[bp];B[cn];W[ep];B[fp];W[fq];B[gq];W[fr];B[gr];W[er];B[hp];W[go];B[fn];W[ho];B[ip];W[io];B[jp];W[jo];B[lp];W[kp];B[kq];W[ko];B[lq];W[ir];B[hq];W[jq];B[jr];W[em];B[gm];W[el];B[hl];W[kl];B[ek];W[fk];B[ej];W[fl];B[fj];W[gk];B[ik];W[gj];B[jj];W[dm];B[lk];W[mm];B[nl];W[nm];B[om];W[ol];B[nk];W[ll];B[kk];W[jl];B[im];W[jk];B[ij];W[kj];B[mk];W[ki];B[ih];W[jh];B[ig];W[jg];B[if];W[oi];B[mi];W[mh];B[lh];W[li];B[nh];W[mj];B[ni];W[nj];B[oj];W[lj];B[ok];W[oh];B[ng];W[pj];B[ji];W[kh];B[jf];W[lg];B[cm];W[cl];B[dk];W[bl];B[bk];W[bn];B[ck];W[bm];B[cc];W[cd];B[dc];W[ec];B[eb];W[fb];B[fc];W[ed];B[gb];W[bc];B[cb];W[cg];B[be];W[bd];B[bg];W[bh];B[cf];W[df];B[ch];W[dg];B[bi];W[qd];B[qc];W[rc];B[rd];W[qe];B[re];W[rb];B[pc];W[qb];B[qf];W[ff];B[sc];W[pb];B[bo];W[ob];B[nc];W[nb];B[mb];W[mc];B[lb])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

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

    int posLen = 19;
    Hash128 hash = NNInputs::getHashV2(board,hist,nextPla);
    float* row = new float[NNInputs::ROW_SIZE_V2];

    auto run = [&](bool inputsUseNHWC) {
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_V2; c++)
        printNNInputHWAndBoard(out,2,board,hist,posLen,inputsUseNHWC,row,c);
      return getAndClear(out);
    };

    string actualNHWC = run(true);
    string actualNCHW = run(false);

    delete[] row;
    delete sgf;

    string expected = R"%%(
F00BA14AAD5D59D0EEE70042C2474AF0
Channel: 0
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . . . . . . . . . . . . . . . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . X . X O X . . . . X5X3O2O O O O .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X X O X . . . . . . O4X1. X X O X
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O O O O . . . . . . . . . . X O X .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X . . . . . . . . . . . . . . O X .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . X O . O . . X X . . . . . . X . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X O O . . . . X O . O . X . . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X . . . . . X O O . O X O . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X . . . . . . . X O O X X O . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . . . X X O . X X O O O O X O . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X X X X O O . X O X X X X X . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O O X O O . X . O O O . X O O . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X O O . X . X . . . O O X . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X X O X . . . . . . . . . . X . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X . O X X O O O O O . . . X . X O .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O . O O X O X X X O X . . . X O . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . O X X O X X . O X X . . X O O O .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X . O O X . O X . . . . . X O . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . . . . . . . . . . . . . . . . . .

Channel: 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 1 0  . . X . X O X . . . . X5X3O2O O O O .
0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0  . O X X O X . . . . . . O4X1. X X O X
0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 1 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 1 1 0 1 1 0 0 0 1 1 1 0 0 1 1 0 0 0  . O O X O O . X . O O O . X O O . . .
0 1 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 1 0  . X . O X X O O O O O . . . X . X O .
0 1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0  . O . O O X O X X X O X . . . X O . .
0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 1 1 1 0  . . O X X O X X . O X X . . X O O O .
0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 1 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 2
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 1 0 1 0 1 0 0 0 0 1 1 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 1 1 0 1 0 0 0 0 0 0 0 1 0 1 1 0 1  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0  . O O O O . . . . . . . . . . X O X .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0  . X . . . . . . . . . . . . . . O X .
0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0  . . X O . O . . X X . . . . . . X . .
0 1 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 1 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 1 1 1 1 0 0 0 1 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 0 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 1 0 1 1 1 0 1 0 0 0 1 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 1 1 0 0 1 1 0 0 1 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 4
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 1  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 5
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0  . O O O O . . . . . . . . . . X O X .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0  . O O X O O . X . O O O . X O O . . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0  . X . O X X O O O O O . . . X . X O .
0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 1 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 6
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 8
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 9
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 10
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 12
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 13
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 14
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 16
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . . . . . . . . . . . . . . . . . . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . . X . X O X . . . . X5X3O2O O O O .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . O X X O X . . . . . . O4X1. X X O X
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . O O O O . . . . . . . . . . X O X .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . X . . . . . . . . . . . . . . O X .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . . X O . O . . X X . . . . . . X . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . X O O . . . . X O . O . X . . . . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . O X . . . . . X O O . O X O . . . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . X . . . . . . . X O O X X O . . . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . . . . X X O . X X O O O O X O . . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . X X X X O O . X O X X X X X . . . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . O O X O O . X . O O O . X O O . . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . O X O O . X . X . . . O O X . . . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . O X X O X . . . . . . . . . . X . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . X . O X X O O O O O . . . X . X O .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . O . O O X O X X X O X . . . X O . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . . O X X O X X . O X X . . X O O O .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . O X . O O X . O X . . . . . X O . .
0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5  . . . . . . . . . . . . . . . . . . .

)%%";
    expect(name,actualNHWC,expected);
    expect(name,actualNCHW,expected);
  }

  {
    const char* name = "NN Inputs V2 Ko and Komi";

    const string sgfStr = "(;FF[4]KM[0.5];B[rj];W[ri];B[si];W[rh];B[sh];W[sg];B[rk];W[sk];B[sl];W[sj];B[eg];W[fg];B[ff];W[gf];B[fh];W[gh];B[gg];W[hg];B[si];W[fg];B[sh];W[sk];B[gg])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

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

    int posLen = 19;
    Hash128 hash = NNInputs::getHashV2(board,hist,nextPla);
    float* row = new float[NNInputs::ROW_SIZE_V2];

    auto run = [&](bool inputsUseNHWC) {
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);

      out << hash << endl;
      int c = 6;
      printNNInputHWAndBoard(out,2,board,hist,posLen,inputsUseNHWC,row,c);
      c = 16;
      printNNInputHWAndBoard(out,2,board,hist,posLen,inputsUseNHWC,row,c);
      return getAndClear(out);
    };

    string actualNHWC = run(true);
    string actualNCHW = run(false);

    delete[] row;
    delete sgf;

    string expected = R"%%(
FE5C02B9D64069A9CCCAD9894A2431C3
Channel: 6
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . X O . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X .2X5O . . . . . . . . . . O
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . X O . . . . . . . . . . O X3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . O X1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1  . . . . . . . . . . . . . . . . . X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . X O4
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 16
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . X O . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . X .2X5O . . . . . . . . . . O
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . X O . . . . . . . . . . O X3
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . O X1
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . X .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . X O4
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . X
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .
0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333 0.0333  . . . . . . . . . . . . . . . . . . .

)%%";
    expect(name,actualNHWC,expected);
    expect(name,actualNCHW,expected);
  }

  {
    const char* name = "NN Inputs V2 7x7";

    const string sgfStr = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Japanese]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

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

    int posLen = 7;
    Hash128 hash = NNInputs::getHashV2(board,hist,nextPla);
    float* row = new float[NNInputs::ROW_SIZE_V2];

    auto run = [&](bool inputsUseNHWC) {
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_V2; c++)
        printNNInputHWAndBoard(out,2,board,hist,posLen,inputsUseNHWC,row,c);
      return getAndClear(out);
    };

    string actualNHWC = run(true);
    string actualNCHW = run(false);

    delete[] row;
    delete sgf;

    string expected = R"%%(

0B50ADBC6ED163777CF0FD0E53647C7D
Channel: 0
1 1 1 1 1 1 1  . . . . . . .
1 1 1 1 1 1 1  . . . . . X .
1 1 1 1 1 1 1  . . . O1. . .
1 1 1 1 1 1 1  . . . X O X4.
1 1 1 1 1 1 1  . . X O5.2O .
1 1 1 1 1 1 1  . X . X O X .
1 1 1 1 1 1 1  . . . . O3. .

Channel: 1
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 1 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 1 0  . . . X O X4.
0 0 1 0 0 0 0  . . X O5.2O .
0 1 0 1 0 1 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 2
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0  . . . O1. . .
0 0 0 0 1 0 0  . . . X O X4.
0 0 0 1 0 1 0  . . X O5.2O .
0 0 0 0 1 0 0  . X . X O X .
0 0 0 0 1 0 0  . . . . O3. .

Channel: 3
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 0 0  . . . X O X4.
0 0 0 1 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 4
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 1 1 0  . . . X O X4.
0 0 0 0 0 1 0  . . X O5.2O .
0 0 0 1 0 1 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 5
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 1 0 0 0 0  . . X O5.2O .
0 0 0 0 1 0 0  . X . X O X .
0 0 0 0 1 0 0  . . . . O3. .

Channel: 6
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 1 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 7
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 1 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 8
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 1 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 9
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 1 0 0  . . . . O3. .

Channel: 10
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 1 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 11
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 12
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 1 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 1 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 13
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 14
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 1 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 15
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 16
0.3 0.3 0.3 0.3 0.3 0.3 0.3  . . . . . . .
0.3 0.3 0.3 0.3 0.3 0.3 0.3  . . . . . X .
0.3 0.3 0.3 0.3 0.3 0.3 0.3  . . . O1. . .
0.3 0.3 0.3 0.3 0.3 0.3 0.3  . . . X O X4.
0.3 0.3 0.3 0.3 0.3 0.3 0.3  . . X O5.2O .
0.3 0.3 0.3 0.3 0.3 0.3 0.3  . X . X O X .
0.3 0.3 0.3 0.3 0.3 0.3 0.3  . . . . O3. .

)%%";
    expect(name,actualNHWC,expected);
    expect(name,actualNCHW,expected);
  }

  {
    const char* name = "NN Inputs V2 7x7 embedded in 9x9";

    const string sgfStr = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Japanese]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

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

    int posLen = 9;
    Hash128 hash = NNInputs::getHashV2(board,hist,nextPla);
    float* row = new float[NNInputs::ROW_SIZE_V2];

    auto run = [&](bool inputsUseNHWC) {
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_V2; c++)
        printNNInputHWAndBoard(out,2,board,hist,posLen,inputsUseNHWC,row,c);
      return getAndClear(out);
    };

    string actualNHWC = run(true);
    string actualNCHW = run(false);

    delete[] row;
    delete sgf;

    string expected = R"%%(

0B50ADBC6ED163777CF0FD0E53647C7D
Channel: 0
1 1 1 1 1 1 1 0 0  . . . . . . .
1 1 1 1 1 1 1 0 0  . . . . . X .
1 1 1 1 1 1 1 0 0  . . . O1. . .
1 1 1 1 1 1 1 0 0  . . . X O X4.
1 1 1 1 1 1 1 0 0  . . X O5.2O .
1 1 1 1 1 1 1 0 0  . X . X O X .
1 1 1 1 1 1 1 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 1
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 1 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 1 0 0 0  . . . X O X4.
0 0 1 0 0 0 0 0 0  . . X O5.2O .
0 1 0 1 0 1 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 2
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0 0 0  . . . O1. . .
0 0 0 0 1 0 0 0 0  . . . X O X4.
0 0 0 1 0 1 0 0 0  . . X O5.2O .
0 0 0 0 1 0 0 0 0  . X . X O X .
0 0 0 0 1 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 3
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 0 0 0 0  . . . X O X4.
0 0 0 1 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 4
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 1 1 0 0 0  . . . X O X4.
0 0 0 0 0 1 0 0 0  . . X O5.2O .
0 0 0 1 0 1 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 5
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 1 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 1 0 0 0 0  . X . X O X .
0 0 0 0 1 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 6
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 1 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 7
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 1 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 8
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 1 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 9
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 1 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 10
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 1 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 11
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 12
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 1 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 1 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 13
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 14
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 1 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 15
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 16
0.3 0.3 0.3 0.3 0.3 0.3 0.3 0 0  . . . . . . .
0.3 0.3 0.3 0.3 0.3 0.3 0.3 0 0  . . . . . X .
0.3 0.3 0.3 0.3 0.3 0.3 0.3 0 0  . . . O1. . .
0.3 0.3 0.3 0.3 0.3 0.3 0.3 0 0  . . . X O X4.
0.3 0.3 0.3 0.3 0.3 0.3 0.3 0 0  . . X O5.2O .
0.3 0.3 0.3 0.3 0.3 0.3 0.3 0 0  . X . X O X .
0.3 0.3 0.3 0.3 0.3 0.3 0.3 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

)%%";
    expect(name,actualNHWC,expected);
    expect(name,actualNCHW,expected);

  }

}


//==================================================================================================================
//==================================================================================================================
//==================================================================================================================
//==================================================================================================================



void Tests::runNNInputsV3Tests() {
  cout << "Running NN inputs V3 tests" << endl;
  ostringstream out;
  out << std::setprecision(5);

  {
    const char* name = "NN Inputs V3 Basic";

    const string sgfStr = "(;FF[4]KM[7.5];B[pd];W[pq];B[dq];W[dd];B[qo];W[pl];B[qq];W[qr];B[pp];W[rq];B[oq];W[qp];B[pr];W[qq];B[oo];W[ro];B[qn];W[do];B[dl];W[gp];B[eo];W[en];B[fo];W[dp];B[eq];W[cq];B[cr];W[br];B[dn];W[bp];B[cn];W[ep];B[fp];W[fq];B[gq];W[fr];B[gr];W[er];B[hp];W[go];B[fn];W[ho];B[ip];W[io];B[jp];W[jo];B[lp];W[kp];B[kq];W[ko];B[lq];W[ir];B[hq];W[jq];B[jr];W[em];B[gm];W[el];B[hl];W[kl];B[ek];W[fk];B[ej];W[fl];B[fj];W[gk];B[ik];W[gj];B[jj];W[dm];B[lk];W[mm];B[nl];W[nm];B[om];W[ol];B[nk];W[ll];B[kk];W[jl];B[im];W[jk];B[ij];W[kj];B[mk];W[ki];B[ih];W[jh];B[ig];W[jg];B[if];W[oi];B[mi];W[mh];B[lh];W[li];B[nh];W[mj];B[ni];W[nj];B[oj];W[lj];B[ok];W[oh];B[ng];W[pj];B[ji];W[kh];B[jf];W[lg];B[cm];W[cl];B[dk];W[bl];B[bk];W[bn];B[ck];W[bm];B[cc];W[cd];B[dc];W[ec];B[eb];W[fb];B[fc];W[ed];B[gb];W[bc];B[cb];W[cg];B[be];W[bd];B[bg];W[bh];B[cf];W[df];B[ch];W[dg];B[bi];W[qd];B[qc];W[rc];B[rd];W[qe];B[re];W[rb];B[pc];W[qb];B[qf];W[ff];B[sc];W[pb];B[bo];W[ob];B[nc];W[nb];B[mb];W[mc];B[lb])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

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

    int posLen = 19;
    double drawEquivalentWinsForWhite = 0.2;
    Hash128 hash = NNInputs::getHashV3(board,hist,nextPla,drawEquivalentWinsForWhite);
    float* rowBin = new float[NNInputs::NUM_FEATURES_BIN_V3 * posLen * posLen];
    float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V3];

    auto run = [&](bool inputsUseNHWC) {
      NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,inputsUseNHWC,rowBin,rowGlobal);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_BIN_V3; c++)
        printNNInputHWAndBoard(out,3,board,hist,posLen,inputsUseNHWC,rowBin,c);
      for(int c = 0; c<NNInputs::NUM_FEATURES_GLOBAL_V3; c++)
        printNNInputGlobal(out,3,rowGlobal,c);
      return getAndClear(out);
    };

    string actualNHWC = run(true);
    string actualNCHW = run(false);

    delete[] rowBin;
    delete[] rowGlobal;
    delete sgf;

    string expected = R"%%(
A5C6AC8DE3EDC8645D2B8AFAF326EC64
Channel: 0
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . . . . . . . . . . . . . . . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . X . X O X . . . . X5X3O2O O O O .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X X O X . . . . . . O4X1. X X O X
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O O O O . . . . . . . . . . X O X .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X . . . . . . . . . . . . . . O X .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . X O . O . . X X . . . . . . X . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X O O . . . . X O . O . X . . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X . . . . . X O O . O X O . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X . . . . . . . X O O X X O . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . . . X X O . X X O O O O X O . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X X X X O O . X O X X X X X . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O O X O O . X . O O O . X O O . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X O O . X . X . . . O O X . . . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X X O X . . . . . . . . . . X . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . X . O X X O O O O O . . . X . X O .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O . O O X O X X X O X . . . X O . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . O X X O X X . O X X . . X O O O .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . O X . O O X . O X . . . . . X O . .
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  . . . . . . . . . . . . . . . . . . .

Channel: 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 1 0  . . X . X O X . . . . X5X3O2O O O O .
0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0  . O X X O X . . . . . . O4X1. X X O X
0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 1 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 1 1 0 1 0 1 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 1 1 0 1 1 0 0 0 1 1 1 0 0 1 1 0 0 0  . O O X O O . X . O O O . X O O . . .
0 1 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 1 0  . X . O X X O O O O O . . . X . X O .
0 1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0  . O . O O X O X X X O X . . . X O . .
0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 1 1 1 0  . . O X X O X X . O X X . . X O O O .
0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 1 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 2
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 1 0 1 0 1 0 0 0 0 1 1 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 1 1 0 1 0 0 0 0 0 0 0 1 0 1 1 0 1  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0  . O O O O . . . . . . . . . . X O X .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0  . X . . . . . . . . . . . . . . O X .
0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0  . . X O . O . . X X . . . . . . X . .
0 1 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 1 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 1 1 1 1 0 0 0 1 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 0 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 1 0 1 1 1 0 1 0 0 0 1 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 1 1 0 0 1 1 0 0 1 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 4
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 0 0 0 1  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 5
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0  . O O O O . . . . . . . . . . X O X .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 1 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0  . O O X O O . X . O O O . X O O . . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0  . X . O X X O O O O O . . . X . X O .
0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 1 0 0 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 6
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 7
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 8
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 9
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 10
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 11
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 12
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 13
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 14
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 15
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 16
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 17
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 18
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 1 1 1 1 1 0  . . X . X O X . . . . X5X3O2O O O O .
0 1 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0  . O X X O X . . . . . . O4X1. X X O X
0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 1 1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 1 0 0 0 0 0 0 0 1 1 1 1 0 1 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 1 1 0 1 1 0 0 0 1 1 1 0 0 1 1 0 0 0  . O O X O O . X . O O O . X O O . . .
0 1 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 1 0 0 1 1 1 1 1 0 0 0 0 0 0 1 0  . X . O X X O O O O O . . . X . X O .
0 1 0 1 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0  . O . O O X O X X X O X . . . X O . .
0 0 1 0 0 1 0 0 0 1 0 0 0 0 0 1 1 1 0  . . O X X O X X . O X X . . X O O O .
0 1 0 0 1 1 0 0 1 0 0 0 0 0 0 0 1 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 19
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 1 0 1 0 1 0 0 0 0 1 1 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 1 1 0 1 0 0 0 0 0 0 0 1 0 1 1 0 1  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0  . O O O O . . . . . . . . . . X O X .
0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0  . X . . . . . . . . . . . . . . O X .
0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0  . . X O . O . . X X . . . . . . X . .
0 1 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 1 0 0 0 0 0 0 0 1 0 0 1 1 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 1 1 0 0 1 1 0 0 0 0 1 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 1 1 1 1 0 0 0 1 0 1 1 1 1 1 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 1 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0  . O X X O X . . . . . . . . . . X . .
0 1 0 0 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 1 0 1 1 1 0 1 0 0 0 1 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 1 1 0 1 1 0 0 1 1 0 0 1 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 1 0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 20
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 21
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X . X O X . . . . X5X3O2O O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . O4X1. X X O X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O O O . . . . . . . . . . X O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . . . . . . . . O X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . X O . O . . X X . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X O O . . . . X O . O . X . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . . . . . X O O . O X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . . . . . . . X O O X X O . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X X O . X X O O O O X O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X X X X O O . X O X X X X X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O O X O O . X . O O O . X O O . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X O O . X . X . . . O O X . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X X O X . . . . . . . . . . X . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . X . O X X O O O O O . . . X . X O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O . O O X O X X X O X . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . O X X O X X . O X X . . X O O O .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . O X . O O X . O X . . . . . X O . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 0: 0
Channel: 1: 0
Channel: 2: 0
Channel: 3: 0
Channel: 4: 0
Channel: 5: 0.5
Channel: 6: 1
Channel: 7: 0.5
Channel: 8: 1
Channel: 9: 0
Channel: 10: 0
Channel: 11: 0
Channel: 12: 0
Channel: 13: 0.5

)%%";
    expect(name,actualNHWC,expected);
    expect(name,actualNCHW,expected);
  }

  {
    const char* name = "NN Inputs V3 Ko";

    const string sgfStr = "(;FF[4]KM[0.5];B[rj];W[ri];B[si];W[rh];B[sh];W[sg];B[rk];W[sk];B[sl];W[sj];B[eg];W[fg];B[ff];W[gf];B[fh];W[gh];B[gg];W[hg];B[si];W[fg];B[sh];W[sk];B[gg])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

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

    int posLen = 19;
    double drawEquivalentWinsForWhite = 0.3;
    Hash128 hash = NNInputs::getHashV3(board,hist,nextPla,drawEquivalentWinsForWhite);
    float* rowBin = new float[NNInputs::NUM_FEATURES_BIN_V3 * posLen * posLen];
    float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V3];

    auto run = [&](bool inputsUseNHWC) {
      NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,inputsUseNHWC,rowBin,rowGlobal);

      out << hash << endl;
      int c = 6;
      printNNInputHWAndBoard(out,3,board,hist,posLen,inputsUseNHWC,rowBin,c);
      for(c = 0; c<NNInputs::NUM_FEATURES_GLOBAL_V3; c++)
        printNNInputGlobal(out,3,rowGlobal,c);
      return getAndClear(out);
    };

    string actualNHWC = run(true);
    string actualNCHW = run(true);

    delete[] rowBin;
    delete[] rowGlobal;
    delete sgf;

    string expected = R"%%(
09B45DF61A5E7554B7716E45B0917FE0
Channel: 6
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . X O . . . . . . . . . . . .
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . X .2X5O . . . . . . . . . . O
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . X O . . . . . . . . . . O X3
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . O X1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1  . . . . . . . . . . . . . . . . . X .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . X O4
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . X
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  . . . . . . . . . . . . . . . . . . .

Channel: 0: 0
Channel: 1: 0
Channel: 2: 0
Channel: 3: 0
Channel: 4: 0
Channel: 5: 0.033333
Channel: 6: 1
Channel: 7: 0.5
Channel: 8: 1
Channel: 9: 0
Channel: 10: 0
Channel: 11: 0
Channel: 12: 0
Channel: 13: -0.5

)%%";
    expect(name,actualNHWC,expected);
    expect(name,actualNCHW,expected);
  }

  {
    const char* name = "NN Inputs V3 7x7";

    const string sgfStr = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Japanese]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

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

    int posLen = 7;
    double drawEquivalentWinsForWhite = 0.5;
    Hash128 hash = NNInputs::getHashV3(board,hist,nextPla,drawEquivalentWinsForWhite);
    float* rowBin = new float[NNInputs::NUM_FEATURES_BIN_V3 * posLen * posLen];
    float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V3];

    auto run = [&](bool inputsUseNHWC) {
      NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,inputsUseNHWC,rowBin,rowGlobal);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_BIN_V3; c++)
        printNNInputHWAndBoard(out,3,board,hist,posLen,inputsUseNHWC,rowBin,c);
      for(int c = 0; c<NNInputs::NUM_FEATURES_GLOBAL_V3; c++)
        printNNInputGlobal(out,3,rowGlobal,c);
      return getAndClear(out);
    };

    string actualNHWC = run(true);
    string actualNCHW = run(false);

    delete[] rowBin;
    delete[] rowGlobal;
    delete sgf;

    string expected = R"%%(

FD2112739B82727E0C1BBB85558C8776
Channel: 0
1 1 1 1 1 1 1  . . . . . . .
1 1 1 1 1 1 1  . . . . . X .
1 1 1 1 1 1 1  . . . O1. . .
1 1 1 1 1 1 1  . . . X O X4.
1 1 1 1 1 1 1  . . X O5.2O .
1 1 1 1 1 1 1  . X . X O X .
1 1 1 1 1 1 1  . . . . O3. .

Channel: 1
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 1 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 1 0  . . . X O X4.
0 0 1 0 0 0 0  . . X O5.2O .
0 1 0 1 0 1 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 2
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0  . . . O1. . .
0 0 0 0 1 0 0  . . . X O X4.
0 0 0 1 0 1 0  . . X O5.2O .
0 0 0 0 1 0 0  . X . X O X .
0 0 0 0 1 0 0  . . . . O3. .

Channel: 3
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 0 0  . . . X O X4.
0 0 0 1 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 4
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 1 1 0  . . . X O X4.
0 0 0 0 0 1 0  . . X O5.2O .
0 0 0 1 0 1 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 5
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 1 0 0 0 0  . . X O5.2O .
0 0 0 0 1 0 0  . X . X O X .
0 0 0 0 1 0 0  . . . . O3. .

Channel: 6
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 1 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 7
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 8
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 9
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 1 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 10
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 1 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 11
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 1 0 0  . . . . O3. .

Channel: 12
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 1 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 13
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 14
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 1 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 1 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 15
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 16
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 1 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 17
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 18
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 1 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 1 0  . . . X O X4.
0 0 1 0 0 0 0  . . X O5.2O .
0 1 0 1 0 1 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 19
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0  . . . O1. . .
0 0 0 0 1 0 0  . . . X O X4.
0 0 0 1 1 1 0  . . X O5.2O .
0 0 0 0 1 0 0  . X . X O X .
0 0 0 0 1 0 0  . . . . O3. .

Channel: 20
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 21
0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0  . . . . O3. .

Channel: 0: 0
Channel: 1: 0
Channel: 2: 0
Channel: 3: 0
Channel: 4: 0
Channel: 5: 0.3
Channel: 6: 1
Channel: 7: 0.5
Channel: 8: 1
Channel: 9: 0
Channel: 10: 0
Channel: 11: 0
Channel: 12: 0
Channel: 13: -0.5

)%%";
    expect(name,actualNHWC,expected);
    expect(name,actualNCHW,expected);
  }

  {
    const char* name = "NN Inputs V3 7x7 embedded in 9x9";

    const string sgfStr = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Japanese]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

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

    int posLen = 9;
    double drawEquivalentWinsForWhite = 0.8;
    Hash128 hash = NNInputs::getHashV3(board,hist,nextPla,drawEquivalentWinsForWhite);
    float* rowBin = new float[NNInputs::NUM_FEATURES_BIN_V3 * posLen * posLen];
    float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V3];

    auto run = [&](bool inputsUseNHWC) {
      NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,inputsUseNHWC,rowBin,rowGlobal);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_BIN_V3; c++)
        printNNInputHWAndBoard(out,3,board,hist,posLen,inputsUseNHWC,rowBin,c);
      for(int c = 0; c<NNInputs::NUM_FEATURES_GLOBAL_V3; c++)
        printNNInputGlobal(out,3,rowGlobal,c);
      return getAndClear(out);
    };

    string actualNHWC = run(true);
    string actualNCHW = run(false);

    delete[] rowBin;
    delete[] rowGlobal;
    delete sgf;

    string expected = R"%%(

FD2112739B82727E0C1BBB85558C8776
Channel: 0
1 1 1 1 1 1 1 0 0  . . . . . . .
1 1 1 1 1 1 1 0 0  . . . . . X .
1 1 1 1 1 1 1 0 0  . . . O1. . .
1 1 1 1 1 1 1 0 0  . . . X O X4.
1 1 1 1 1 1 1 0 0  . . X O5.2O .
1 1 1 1 1 1 1 0 0  . X . X O X .
1 1 1 1 1 1 1 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 1
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 1 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 1 0 0 0  . . . X O X4.
0 0 1 0 0 0 0 0 0  . . X O5.2O .
0 1 0 1 0 1 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 2
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0 0 0  . . . O1. . .
0 0 0 0 1 0 0 0 0  . . . X O X4.
0 0 0 1 0 1 0 0 0  . . X O5.2O .
0 0 0 0 1 0 0 0 0  . X . X O X .
0 0 0 0 1 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 3
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 0 0 0 0  . . . X O X4.
0 0 0 1 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 4
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 1 1 0 0 0  . . . X O X4.
0 0 0 0 0 1 0 0 0  . . X O5.2O .
0 0 0 1 0 1 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 5
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 1 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 1 0 0 0 0  . X . X O X .
0 0 0 0 1 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 6
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 1 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 7
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 8
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 9
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 1 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 10
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 1 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 11
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 1 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 12
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 1 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 13
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 14
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 1 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 1 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 15
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 16
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 1 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 17
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 18
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 1 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 1 0 1 0 0 0  . . . X O X4.
0 0 1 0 0 0 0 0 0  . . X O5.2O .
0 1 0 1 0 1 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 19
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 1 0 0 0 0 0  . . . O1. . .
0 0 0 0 1 0 0 0 0  . . . X O X4.
0 0 0 1 1 1 0 0 0  . . X O5.2O .
0 0 0 0 1 0 0 0 0  . X . X O X .
0 0 0 0 1 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 20
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 21
0 0 0 0 0 0 0 0 0  . . . . . . .
0 0 0 0 0 0 0 0 0  . . . . . X .
0 0 0 0 0 0 0 0 0  . . . O1. . .
0 0 0 0 0 0 0 0 0  . . . X O X4.
0 0 0 0 0 0 0 0 0  . . X O5.2O .
0 0 0 0 0 0 0 0 0  . X . X O X .
0 0 0 0 0 0 0 0 0  . . . . O3. .
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0

Channel: 0: 0
Channel: 1: 0
Channel: 2: 0
Channel: 3: 0
Channel: 4: 0
Channel: 5: 0.3
Channel: 6: 1
Channel: 7: 0.5
Channel: 8: 1
Channel: 9: 0
Channel: 10: 0
Channel: 11: 0
Channel: 12: 0
Channel: 13: -0.5

)%%";
    expect(name,actualNHWC,expected);
    expect(name,actualNCHW,expected);

  }

  {
    const char* name = "NN Inputs V3 Area Komi";

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

    int posLen = 7;
    double drawEquivalentWinsForWhite = 0.3;
    float* rowBin = new float[NNInputs::NUM_FEATURES_BIN_V3 * posLen * posLen];
    float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V3];

    NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,true,rowBin,rowGlobal);
    int c = 18;
    printNNInputHWAndBoard(out,3,board,hist,posLen,true,rowBin,c);
    c = 19;
    printNNInputHWAndBoard(out,3,board,hist,posLen,true,rowBin,c);
    for(c = 0; c<NNInputs::NUM_FEATURES_GLOBAL_V3; c++)
      printNNInputGlobal(out,3,rowGlobal,c);

    nextPla = P_WHITE;
    hist.clear(board,nextPla,initialRules,0);
    NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,true,rowBin,rowGlobal);
    for(c = 0; c<NNInputs::NUM_FEATURES_GLOBAL_V3; c++)
      printNNInputGlobal(out,3,rowGlobal,c);

    nextPla = P_BLACK;
    initialRules.komi = 1;
    hist.clear(board,nextPla,initialRules,0);
    NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,true,rowBin,rowGlobal);
    for(c = 0; c<NNInputs::NUM_FEATURES_GLOBAL_V3; c++)
      printNNInputGlobal(out,3,rowGlobal,c);

    delete[] rowBin;
    delete[] rowGlobal;

    string expected = R"%%(
Channel: 18
0 0 0 0 0 0 0  . X O . O O .
0 0 0 0 0 0 0  X X O . X O X
0 0 0 0 0 0 0  O O O O O O O
1 1 1 0 0 1 1  X X X . . X X
1 1 1 0 0 1 1  . . X O O X .
1 1 1 1 1 1 1  . . X X X X X
1 1 1 1 1 1 1  . . X O . O X

Channel: 19
1 1 1 1 1 1 1  . X O . O O .
1 1 1 1 1 1 1  X X O . X O X
1 1 1 1 1 1 1  O O O O O O O
0 0 0 0 0 0 0  X X X . . X X
0 0 0 1 1 0 0  . . X O O X .
0 0 0 0 0 0 0  . . X X X X X
0 0 0 0 0 0 0  . . X O . O X

Channel: 0: 0
Channel: 1: 0
Channel: 2: 0
Channel: 3: 0
Channel: 4: 0
Channel: 5: -0.12
Channel: 6: 1
Channel: 7: 0.5
Channel: 8: 1
Channel: 9: 0
Channel: 10: 0
Channel: 11: 0
Channel: 12: 0
Channel: 13: -0.2
Channel: 0: 0
Channel: 1: 0
Channel: 2: 0
Channel: 3: 0
Channel: 4: 0
Channel: 5: 0.12
Channel: 6: 1
Channel: 7: 0.5
Channel: 8: 1
Channel: 9: 0
Channel: 10: 0
Channel: 11: 0
Channel: 12: 0
Channel: 13: 0.2
Channel: 0: 0
Channel: 1: 0
Channel: 2: 0
Channel: 3: 0
Channel: 4: 0
Channel: 5: -0.053333
Channel: 6: 1
Channel: 7: 0.5
Channel: 8: 1
Channel: 9: 0
Channel: 10: 0
Channel: 11: 0
Channel: 12: 0
Channel: 13: 0.2

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "NN Inputs V3 Rules";

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

      int posLen = size;
      double drawEquivalentWinsForWhite = 0.47;
      float* rowBin = new float[NNInputs::NUM_FEATURES_BIN_V3 * posLen * posLen];
      float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V3];

      for(int c = 0; c<NNInputs::NUM_FEATURES_GLOBAL_V3; c++) {
        for(int i = 0; i<rules.size(); i++) {
          BoardHistory hist(board,nextPla,rules[i],0);
          NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,true,rowBin,rowGlobal);
          out << rowGlobal[c] << " ";
        }
        out << endl;
      }

      delete[] rowBin;
      delete[] rowGlobal;
    }

    string expected = R"%%(
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
-0.064667 -0.1 -0.13133 -0.16667 -0.198 -0.23333 -0.26467 -0.3 -0.33133 -0.36667 -0.398 -0.43333 -0.064667 -0.1 -0.13133 -0.16667 -0.198 -0.23333 -0.26467 -0.3 -0.33133 -0.36667 -0.398 -0.43333
0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0.5 0.5 0.5 0.5 -0.5 -0.5 -0.5 -0.5 0 0 0 0 0.5 0.5 0.5 0.5 -0.5 -0.5 -0.5 -0.5
0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0.03 -0.5 -0.03 0.5 0.03 -0.5 -0.03 0.5 0.03 -0.5 -0.03 0.5 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
-0.064667 -0.1 -0.13133 -0.16667 -0.198 -0.23333 -0.26467 -0.3 -0.33133 -0.36667 -0.398 -0.43333 -0.064667 -0.1 -0.13133 -0.16667 -0.198 -0.23333 -0.26467 -0.3 -0.33133 -0.36667 -0.398 -0.43333
0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 1 1 1 1
0 0 0 0 0.5 0.5 0.5 0.5 -0.5 -0.5 -0.5 -0.5 0 0 0 0 0.5 0.5 0.5 0.5 -0.5 -0.5 -0.5 -0.5
0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1
0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
-0.03 0.5 0.03 -0.5 -0.03 0.5 0.03 -0.5 -0.03 0.5 0.03 -0.5 0 0 0 0 0 0 0 0 0 0 0 0

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "NN Inputs V3 Ko Prohib and pass hist and whitebonus and encorestart";

    //Immediately enters encore via b0 pass w1 pass. Through w19, sets up various ko shapes. Then starts ko captures. b26 pass b27 pass switches to second encore.
    const string sgfStr = "(;GM[1]FF[4]SZ[6]KM[0.00];B[];W[];B[ab];W[bb];B[ba];W[ca];B[ec];W[ed];B[fd];W[fe];B[fb];W[dc];B[db];W[ae];B[ea];W[bf];B[be];W[ad];B[cf];W[dd];B[af];W[aa];B[];W[fc];B[bd];W[eb];B[];W[];B[ec];W[bf];B[ac];W[eb];B[af];W[eb])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = Rules(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, false, 0.0f);
    sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
    vector<Move>& moves = sgf->moves;

    int posLen = 6;
    double drawEquivalentWinsForWhite = 0.0;
    float* rowBin = new float[NNInputs::NUM_FEATURES_BIN_V3 * posLen * posLen];
    float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V3];

    for(size_t i = 0; i<moves.size(); i++) {
      assert(hist.isLegal(board,moves[i].loc,moves[i].pla));
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
        NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,true,rowBin,rowGlobal);
        out << "Pass Hist Channels: ";
        for(int c = 0; c<5; c++)
          out << rowGlobal[c] << " ";
        out << endl;
        out << "Selfkomi channel times 15: " << rowGlobal[5]*15 << endl;
        out << "EncorePhase channel 10,11: " << rowGlobal[10] << " " << rowGlobal[11] << endl;
        out << "PassWouldEndPhase channel 12: " << rowGlobal[12] << endl;
        printNNInputHWAndBoard(out,3,board,hist,posLen,true,rowBin,7);
        printNNInputHWAndBoard(out,3,board,hist,posLen,true,rowBin,8);
        printNNInputHWAndBoard(out,3,board,hist,posLen,true,rowBin,20);
        printNNInputHWAndBoard(out,3,board,hist,posLen,true,rowBin,21);
      }
    }

    delete[] rowBin;
    delete[] rowGlobal;
    delete sgf;

    string expected = R"%%(
Move 24
010100 001010 111000 001001 001110 010111
Pass Hist Channels: 0 0 1 0 0
Selfkomi channel times 15: -0.5
EncorePhase channel 10,11: 1 0
PassWouldEndPhase channel 12: 0
Channel: 7
0 0 0 0 0 0  O2. O . X .
0 0 0 0 0 0  X O . X . X
0 0 0 0 0 0  . . . O X O4
0 0 0 0 0 0  O X5. O O .
0 0 0 0 0 0  O X . . . O
0 1 0 0 0 0  X1. X . . .

Channel: 8
0 1 0 0 0 0  O2. O . X .
0 0 0 0 0 0  X O . X . X
0 0 0 0 0 0  . . . O X O4
0 0 0 0 0 1  O X5. O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  X1. X . . .

Channel: 20
0 0 0 0 0 0  O2. O . X .
0 0 0 0 0 0  X O . X . X
0 0 0 0 0 0  . . . O X O4
0 0 0 0 0 0  O X5. O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  X1. X . . .

Channel: 21
0 0 0 0 0 0  O2. O . X .
0 0 0 0 0 0  X O . X . X
0 0 0 0 0 0  . . . O X O4
0 0 0 0 0 0  O X5. O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  X1. X . . .

Move 25
010101 001000 111010 001000 001110 010111
Pass Hist Channels: 0 0 0 1 0
Selfkomi channel times 15: 1.5
EncorePhase channel 10,11: 1 0
PassWouldEndPhase channel 12: 0
Channel: 7
0 1 0 0 0 0  O1. O . X .
0 0 0 0 0 0  X O . X O5X
0 0 0 0 1 0  . . . O . O3
0 0 0 0 0 0  O X4. O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  X . X . . .

Channel: 8
0 0 0 0 0 0  O1. O . X .
0 0 0 0 0 0  X O . X O5X
0 0 0 0 0 0  . . . O . O3
0 0 0 0 0 0  O X4. O O .
0 0 0 0 0 0  O X . . . O
0 1 0 0 0 0  X . X . . .

Channel: 20
0 0 0 0 0 0  O1. O . X .
0 0 0 0 0 0  X O . X O5X
0 0 0 0 0 0  . . . O . O3
0 0 0 0 0 0  O X4. O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  X . X . . .

Channel: 21
0 0 0 0 0 0  O1. O . X .
0 0 0 0 0 0  X O . X O5X
0 0 0 0 0 0  . . . O . O3
0 0 0 0 0 0  O X4. O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  X . X . . .

Move 26
010101 001000 111010 001001 001110 010111
Pass Hist Channels: 1 0 0 0 1
Selfkomi channel times 15: -1.5
EncorePhase channel 10,11: 1 0
PassWouldEndPhase channel 12: 1
Channel: 7
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X O4X
0 0 0 0 0 0  . . . O . O2
0 0 0 0 0 0  O X3. O O .
0 0 0 0 0 0  O X . . . O
0 1 0 0 0 0  X . X . . .

Channel: 8
0 1 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X O4X
0 0 0 0 1 0  . . . O . O2
0 0 0 0 0 0  O X3. O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  X . X . . .

Channel: 20
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X O4X
0 0 0 0 0 0  . . . O . O2
0 0 0 0 0 0  O X3. O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  X . X . . .

Channel: 21
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X O4X
0 0 0 0 0 0  . . . O . O2
0 0 0 0 0 0  O X3. O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  X . X . . .

Move 30
010100 001010 011000 001001 001110 000111
Pass Hist Channels: 0 0 0 1 1
Selfkomi channel times 15: -1.5
EncorePhase channel 10,11: 1 1
PassWouldEndPhase channel 12: 0
Channel: 7
0 0 0 0 0 0  O . O . X .
0 0 0 0 1 0  X O . X . X
0 0 0 0 0 0  X5. . O X3O
0 0 0 0 0 0  O X . O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  . O4X . . .

Channel: 8
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X . X
0 0 0 0 0 0  X5. . O X3O
0 0 0 0 0 0  O X . O O .
0 0 0 0 0 0  O X . . . O
1 0 0 0 0 0  . O4X . . .

Channel: 20
1 0 1 0 0 0  O . O . X .
0 1 0 0 1 0  X O . X . X
0 0 0 1 0 1  X5. . O X3O
1 0 0 1 1 0  O X . O O .
1 0 0 0 0 1  O X . . . O
0 0 0 0 0 0  . O4X . . .

Channel: 21
0 0 0 0 1 0  O . O . X .
1 0 0 1 0 1  X O . X . X
0 0 0 0 0 0  X5. . O X3O
0 1 0 0 0 0  O X . O O .
0 1 0 0 0 0  O X . . . O
1 0 1 0 0 0  . O4X . . .

Move 31
010101 001010 011000 001001 001110 100111
Pass Hist Channels: 0 0 0 0 1
Selfkomi channel times 15: 1.5
EncorePhase channel 10,11: 1 1
PassWouldEndPhase channel 12: 0
Channel: 7
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X .5X
0 0 0 0 0 0  X4. . O X2O
0 0 0 0 0 0  O X . O O .
0 0 0 0 0 0  O X . . . O
1 0 0 0 0 0  . O3X . . .

Channel: 8
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X .5X
0 0 0 0 0 0  X4. . O X2O
0 0 0 0 0 0  O X . O O .
0 0 0 0 0 0  O X . . . O
0 0 0 0 0 0  . O3X . . .

Channel: 20
0 0 0 0 1 0  O . O . X .
1 0 0 1 0 1  X O . X .5X
0 0 0 0 0 0  X4. . O X2O
0 1 0 0 0 0  O X . O O .
0 1 0 0 0 0  O X . . . O
1 0 1 0 0 0  . O3X . . .

Channel: 21
1 0 1 0 0 0  O . O . X .
0 1 0 0 1 0  X O . X .5X
0 0 0 1 0 1  X4. . O X2O
1 0 0 1 1 0  O X . O O .
1 0 0 0 0 1  O X . . . O
0 0 0 0 0 0  . O3X . . .

Move 32
010100 001010 011000 101001 101110 000111
Pass Hist Channels: 0 0 0 0 0
Selfkomi channel times 15: -1.5
EncorePhase channel 10,11: 1 1
PassWouldEndPhase channel 12: 0
Channel: 7
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X .4X
0 0 0 0 0 0  X3. . O X1O
0 0 0 0 0 0  . X . O O .
0 0 0 0 0 0  . X . . . O
0 0 0 0 0 0  X5.2X . . .

Channel: 8
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X .4X
0 0 0 0 0 0  X3. . O X1O
0 0 0 0 0 0  . X . O O .
0 0 0 0 0 0  . X . . . O
0 0 0 0 0 0  X5.2X . . .

Channel: 20
1 0 1 0 0 0  O . O . X .
0 1 0 0 1 0  X O . X .4X
0 0 0 1 0 1  X3. . O X1O
1 0 0 1 1 0  . X . O O .
1 0 0 0 0 1  . X . . . O
0 0 0 0 0 0  X5.2X . . .

Channel: 21
0 0 0 0 1 0  O . O . X .
1 0 0 1 0 1  X O . X .4X
0 0 0 0 0 0  X3. . O X1O
0 1 0 0 0 0  . X . O O .
0 1 0 0 0 0  . X . . . O
1 0 1 0 0 0  X5.2X . . .

Move 33
010101 001000 011010 101000 101110 010111
Pass Hist Channels: 0 0 0 0 0
Selfkomi channel times 15: 1.5
EncorePhase channel 10,11: 1 1
PassWouldEndPhase channel 12: 0
Channel: 7
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X O3X
0 0 0 0 1 0  X2. . O . O
0 0 0 0 0 0  . X . O O .
0 0 0 0 0 0  . X . . . O
0 0 0 0 0 0  X4.1X . . .

Channel: 8
0 0 0 0 0 0  O . O . X .
0 0 0 0 0 0  X O . X O3X
0 0 0 0 0 0  X2. . O . O
0 0 0 0 0 0  . X . O O .
0 0 0 0 0 0  . X . . . O
0 0 0 0 0 0  X4.1X . . .

Channel: 20
0 0 0 0 1 0  O . O . X .
1 0 0 1 0 1  X O . X O3X
0 0 0 0 0 0  X2. . O . O
0 1 0 0 0 0  . X . O O .
0 1 0 0 0 0  . X . . . O
1 0 1 0 0 0  X4.1X . . .

Channel: 21
1 0 1 0 0 0  O . O . X .
0 1 0 0 1 0  X O . X O3X
0 0 0 1 0 1  X2. . O . O
1 0 0 1 1 0  . X . O O .
1 0 0 0 0 1  . X . . . O
0 0 0 0 0 0  X4.1X . . .

)%%";
    expect(name,out,expected);
  }

  {
    const char* name = "NN Inputs V3 some other test positions";

    const string sgfStr = "(;FF[4]GM[1]SZ[13]PB[s75411712-d5152283-b8c128]PW[s78621440-d5365731-b8c128]HA[0]KM[7.5]RU[koPOSITIONALscoreAREAsui0]RE[B+11.5];B[ck];W[lb];B[ke];W[ld];B[jd];W[kc];B[jc];W[jb];B[ib];W[kk];B[ki];W[kh];B[ja];W[le];B[ic];W[kf];B[lj];W[li];B[kj];W[lk];B[jk];W[jl];B[ik];W[mj];B[kb];W[jj];B[ji];W[ij];B[ii];W[hj];B[lh];W[mi];B[kg];W[jg];B[jh];W[lg];B[hk];W[hi];B[mh];W[gk];B[mk];W[il];B[jf];W[lf];B[ig];W[cc];B[dc];W[cd];B[ed];W[kd];B[dj];W[el];B[eg];W[de];B[ee];W[ec];B[je];W[db];B[fc];W[eb];B[bj];W[fd];B[gc];W[cl];B[df];W[dd];B[cf];W[dl];B[gh];W[fk];B[la];W[hh];B[hg];W[fi];B[gg];W[mc];B[bk];W[fb];B[gb];W[ei];B[gi];W[fe];B[ef];W[ej];B[gj];W[hl];B[bh];W[mg];B[be];W[bd];B[ad];W[bb];B[ae];W[di];B[me];W[ci];B[bi];W[bl];B[ab];W[ba];B[ac];W[ml];B[ga];W[fa];B[al];W[bc];B[bf];W[mj];B[mi];W[mb];B[ge];W[mk];B[dk];W[md];B[ek];W[fj];B[jb];W[fh];B[ff];W[bm];B[ka];W[ce];B[ak];W[cj];B[ch];W[];B[id];W[fl];B[hc];W[am];B[ik];W[jk];B[ma];W[];B[mm];W[gl];B[aa];W[ca];B[dh];W[fg];B[];W[lm];B[bg];W[];B[hd];W[];B[ag];W[];B[hf];W[];B[gd];W[];B[ih];W[];B[li];W[];B[hb];W[];B[af];W[];B[ia];W[];B[kl];W[];B[])";

    CompactSgf* sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules;
    sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
    vector<Move>& moves = sgf->moves;

    int posLen = 13;
    double drawEquivalentWinsForWhite = 0.0;
    float* rowBin = new float[NNInputs::NUM_FEATURES_BIN_V3 * posLen * posLen];
    float* rowGlobal = new float[NNInputs::NUM_FEATURES_GLOBAL_V3];

    for(size_t i = 0; i<moves.size(); i++) {
      assert(hist.isLegal(board,moves[i].loc,moves[i].pla));
      hist.makeBoardMoveAssumeLegal(board,moves[i].loc,moves[i].pla,NULL);
      nextPla = getOpp(moves[i].pla);

      if(i == 163) {
        out << "Move " << i << endl;
        NNInputs::fillRowV3(board,hist,nextPla,drawEquivalentWinsForWhite,posLen,true,rowBin,rowGlobal);
        printNNInputHWAndBoard(out,3,board,hist,posLen,true,rowBin,18);
        printNNInputHWAndBoard(out,3,board,hist,posLen,true,rowBin,19);
      }
    }

    delete[] rowBin;
    delete[] rowGlobal;
    delete sgf;

    string expected = R"%%(
Move 163
Channel: 18
1 0 0 0 0 0 1 1 1 1 1 1 1  X O O . . O X . X2X X X X
1 0 0 0 0 0 1 1 1 1 1 0 0  X O . O O O X X X X X O O
1 0 0 0 0 1 1 1 1 1 0 0 0  X O O . O X X X X X O . O
1 0 0 0 1 1 1 1 1 1 0 0 0  X O O O X . X X X X O O O
1 1 0 0 1 1 1 1 1 1 1 0 0  X X O O X . X . . X X O X
1 1 1 1 1 1 1 1 1 1 0 0 0  X X X X X X . X . X O O .
1 1 1 1 1 0 1 1 1 1 1 0 0  X X . . X O X X X . X O O
1 1 1 1 0 0 1 0 1 1 1 1 1  . X X X . O X O X X . X X
1 1 0 0 0 0 1 0 1 1 1 1 1  . X O O O O X O X X X X X
1 1 0 1 0 0 1 0 0 0 1 1 0  . X O X O O X O O O X X O
1 1 1 1 1 0 0 0 1 0 0 0 0  X X X X X O O . X O O O O
1 0 0 0 0 0 0 0 0 0 1 0 0  X O O O O O O O O O X4. O
0 0 0 0 0 0 0 0 0 0 0 0 0  O O . . . . . . . . . O .

Channel: 19
0 1 1 1 1 1 0 0 0 0 0 0 0  X O O . . O X . X2X X X X
0 1 1 1 1 1 0 0 0 0 0 1 1  X O . O O O X X X X X O O
0 1 1 1 1 0 0 0 0 0 1 1 1  X O O . O X X X X X O . O
0 1 1 1 0 0 0 0 0 0 1 1 1  X O O O X . X X X X O O O
0 0 1 1 0 0 0 0 0 0 0 1 1  X X O O X . X . . X X O X
0 0 0 0 0 0 0 0 0 0 1 1 1  X X X X X X . X . X O O .
0 0 0 0 0 1 0 0 0 0 0 1 1  X X . . X O X X X . X O O
0 0 0 0 0 1 0 1 0 0 0 0 0  . X X X . O X O X X . X X
0 0 1 1 1 1 0 1 0 0 0 0 0  . X O O O O X O X X X X X
0 0 1 0 1 1 0 1 1 1 0 0 1  . X O X O O X O O O X X O
0 0 0 0 0 1 1 0 0 1 1 1 1  X X X X X O O . X O O O O
0 1 1 1 1 1 1 1 1 1 0 0 1  X O O O O O O O O O X4. O
1 1 0 0 0 0 0 0 0 0 0 1 1  O O . . . . . . . . . O .

)%%";
    expect(name,out,expected);
  }
}
