#include "../tests/tests.h"
#include "../neuralnet/nninputs.h"
#include "../dataio/sgf.h"
using namespace TestCommon;

#include <iomanip>

void Tests::runNNInputsTests() {
  cout << "Running NN inputs tests" << endl;
  ostringstream out;
  out << std::setprecision(3);

  //TODO switch to V3 and add more tests for more various positions, as well as all the new aux features
  auto printNNInputAndBoard = [&out](const Board& board, const BoardHistory& hist, int posLen, bool inputsUseNHWC, float* row, int c) {
    for(int y = 0; y<posLen; y++) {
      for(int x = 0; x<posLen; x++) {
        int pos = NNPos::xyToPos(x,y,posLen);
        if(x > 0)
          out << " ";
        if(inputsUseNHWC)
          out << row[pos * NNInputs::NUM_FEATURES_V2 + c];
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
  };


  {
    const char* name = "NN Inputs Basic";

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

    string actualNHWC;
    {
      bool inputsUseNHWC = true;
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_V2; c++) {
        out << "Channel: " << c << endl;
        printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
        out << endl;
      }
      actualNHWC = out.str();
      out.str("");
      out.clear();
    }

    string actualNCHW;
    {
      bool inputsUseNHWC = false;
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_V2; c++) {
        out << "Channel: " << c << endl;
        printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
        out << endl;
      }
      actualNCHW = out.str();
      out.str("");
      out.clear();
    }

    delete row;
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
    const char* name = "NN Inputs Ko and Komi";

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


    string actualNHWC;
    {
      bool inputsUseNHWC = true;
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);

      out << hash << endl;
      int c = 6;
      out << "Channel: " << c << endl;
      printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
      out << endl;
      c = 16;
      out << "Channel: " << c << endl;
      printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
      out << endl;
      actualNHWC = out.str();
      out.str("");
      out.clear();
    }
    string actualNCHW;
    {
      bool inputsUseNHWC = false;
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);

      out << hash << endl;
      int c = 6;
      out << "Channel: " << c << endl;
      printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
      out << endl;
      c = 16;
      out << "Channel: " << c << endl;
      printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
      out << endl;
      actualNCHW = out.str();
      out.str("");
      out.clear();
    }

    delete row;
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
    const char* name = "NN Inputs 7x7";

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

    string actualNHWC;
    {
      bool inputsUseNHWC = true;
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_V2; c++) {
        out << "Channel: " << c << endl;
        printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
        out << endl;
      }
      actualNHWC = out.str();
      out.str("");
      out.clear();
    }

    string actualNCHW;
    {
      bool inputsUseNHWC = false;
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_V2; c++) {
        out << "Channel: " << c << endl;
        printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
        out << endl;
      }
      actualNCHW = out.str();
      out.str("");
      out.clear();
    }

    delete row;
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
    const char* name = "NN Inputs 7x7 embedded in 9x9";

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

    string actualNHWC;
    {
      bool inputsUseNHWC = true;
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_V2; c++) {
        out << "Channel: " << c << endl;
        printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
        out << endl;
      }
      actualNHWC = out.str();
      out.str("");
      out.clear();
    }

    string actualNCHW;
    {
      bool inputsUseNHWC = false;
      NNInputs::fillRowV2(board,hist,nextPla,posLen,inputsUseNHWC,row);
      out << hash << endl;
      for(int c = 0; c<NNInputs::NUM_FEATURES_V2; c++) {
        out << "Channel: " << c << endl;
        printNNInputAndBoard(board,hist,posLen,inputsUseNHWC,row,c);
        out << endl;
      }
      actualNCHW = out.str();
      out.str("");
      out.clear();
    }

    delete row;
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
