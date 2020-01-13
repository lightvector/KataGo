#include "core/global.h"
#include "core/config_parser.h"
#include "core/timer.h"
#include "dataio/sgf.h"
#include "search/asyncbot.h"
#include "program/setup.h"
#include "main.h"

#include <chrono>

#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

using namespace std;

int MainCmds::benchmark(int argc, const char* const* argv) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  const int64_t defaultMaxVisits = 600;

  string configFile;
  string modelFile;
  string sgfFile;
  int boardSize;
  int64_t maxVisits;
  string desiredThreadsStr;
  int numPositionsPerGame;
  try {
    TCLAP::CmdLine cmd("Benchmark to test speed with different numbers of threads", ' ', Version::getKataGoVersionForHelp(),true);
    TCLAP::ValueArg<string> configFileArg("","config","Config file to use, same as for gtp (see gtp_example.cfg)",true,string(),"FILE");
    TCLAP::ValueArg<string> modelFileArg("","model","Neural net model file to use",true,string(),"FILE");
    TCLAP::ValueArg<string> sgfFileArg("","sgf", "Optional game to sample positions from (default: uses a built-in-set of positions)",false,string(),"FILE");
    TCLAP::ValueArg<int> boardSizeArg("","boardsize", "Size of board to benchmark on (9-19), default 19",false,-1,"SIZE");
    TCLAP::ValueArg<long> visitsArg("v","visits","How many visits to use per search (default " + Global::int64ToString(defaultMaxVisits) + ")",false,(long)defaultMaxVisits,"VISITS");
    TCLAP::ValueArg<string> threadsArg("t","threads","Test using these many threads, comma-separated (default 1,2,4,6,8,12,16)",false,string("1,2,4,6,8,12,16"),"THREADS");
    TCLAP::ValueArg<int> numPositionsPerGameArg("n","numpositions","How many positions to sample from a game (default 10)",false,10,"NUM");
    cmd.add(configFileArg);
    cmd.add(modelFileArg);
    cmd.add(sgfFileArg);
    cmd.add(boardSizeArg);
    cmd.add(visitsArg);
    cmd.add(threadsArg);
    cmd.add(numPositionsPerGameArg);
    cmd.parse(argc,argv);
    configFile = configFileArg.getValue();
    modelFile = modelFileArg.getValue();
    sgfFile = sgfFileArg.getValue();
    boardSize = boardSizeArg.getValue();
    maxVisits = (int64_t)visitsArg.getValue();
    desiredThreadsStr = threadsArg.getValue();
    numPositionsPerGame = numPositionsPerGameArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  if(boardSize != -1 && sgfFile != "")
    throw StringError("Cannot specify both -sgf and -boardsize at the same time");
  if(boardSize != -1 && (boardSize < 9 || boardSize > 19))
    throw StringError("Board size to test: invalid value " + Global::intToString(boardSize));
  if(maxVisits <= 1)
    throw StringError("Number of visits to use: invalid value " + Global::int64ToString(maxVisits));
  if(numPositionsPerGame <= 0 || numPositionsPerGame > 100000)
    throw StringError("Number of positions per game to use: invalid value " + Global::intToString(numPositionsPerGame));

  vector<int> numThreadsToTest;
  int maxNumThreadsInATest = 1;
  {
    vector<string> desiredThreadsPieces = Global::split(desiredThreadsStr,',');
    for(int i = 0; i<desiredThreadsPieces.size(); i++) {
      string s = Global::trim(desiredThreadsPieces[i]);
      if(s == "")
        continue;
      int desiredThreads;
      bool suc = Global::tryStringToInt(s,desiredThreads);
      if(!suc || desiredThreads <= 0 || desiredThreads > 1024)
        throw StringError("Number of threads to use: invalid value: " + s);
      numThreadsToTest.push_back(desiredThreads);
      maxNumThreadsInATest = std::max(maxNumThreadsInATest,desiredThreads);
    }

    if(numThreadsToTest.size() <= 0) {
      throw StringError("Must specify at least one valid value for -threads");
    }
  }

  ConfigParser cfg(configFile);

  CompactSgf* sgf;
  if(sgfFile != "") {
    sgf = CompactSgf::loadFile(sgfFile);
  }
  else {
    string sgfData;
    if(boardSize == -1)
      boardSize = 19;

    if(boardSize == 19) {
      sgfData = "(;FF[4]GM[1]SZ[19]HA[0]KM[7.5];B[dd];W[pp];B[dp];W[pd];B[qq];W[pq];B[qp];W[qo];B[ro];W[rn];B[qn];W[po];B[rm];W[rp];B[sn];W[rq];B[nc];W[oc];B[qr];W[rr];B[nd];W[pf];B[re];W[lc];B[jc];W[le];B[qc];W[qd];B[ob];W[pc];B[pb];W[rc];B[qb];W[rd];B[lb];W[mb];B[ld];W[kb];B[kc];W[la];B[mc];W[qi];B[ke];W[cc];B[dc];W[cd];B[cf];W[ce];B[de];W[bf];B[cb];W[bb];B[be];W[db];B[bc];W[ca];B[bd];W[cb];B[bg];W[df];B[cg];W[fc];B[nr];W[nq];B[pg];W[qg];B[of];W[ef];B[mq];W[cq];B[dq];W[cp];B[co];W[bo];B[bn];W[cn];B[do];W[bm];B[bp];W[an];B[bq];W[cr];B[br];W[kq];B[iq];W[ko];B[kr];W[cj];B[lq];W[bi];B[qj];W[ph];B[og];W[rf];B[gc];W[gd];B[hc];W[nb];B[rb];W[sb];B[lb];W[fe];B[ri];W[rh];B[pi];W[qh];B[af];W[lc];B[dn];W[dm];B[em];W[in];B[lb];W[fn];B[lc];W[en];B[fp];W[fm];B[pj];W[hp];B[hq];W[ij];B[di];W[dj];B[jl];W[il];B[jm];W[im];B[jk];W[jj];B[kj];W[ki];B[lj];W[li];B[mi];W[jh];B[pn];W[or];B[np];W[ie];B[dh];W[ei];B[eh];W[fh];B[kn];W[oo];B[mn];W[oh];B[ni];W[oq];B[fg];W[gh];B[hg];W[hh];B[ig];W[ih];B[fb];W[eb];B[gb];W[mj];B[mg];W[kf];B[lf];W[je];B[kg];W[kd];B[jg];W[ln];B[lm];W[lo];B[nn];W[ra];B[na];W[gp];B[gq];W[fo];B[rj];W[oe];B[me];W[mr];B[lr];W[ns];B[sp];W[on];B[om];W[pm];B[nm];W[ip];B[jp];W[jo];B[kp];W[lp];B[jq];W[ap];B[dr];W[ar];B[cs];W[aq];B[sr];W[qs];B[ab];W[ea];B[ec];W[fd];B[fa];W[ee];B[ke];W[oi];B[oj];W[le];B[ik];W[hk];B[ke];W[ep];B[fq];W[le];B[eo];W[ke];B[sq];W[pr];B[qm];W[ml];B[mk];W[lk];B[nj];W[kk];B[mj];W[km];B[kl];W[qa];B[hd];W[he];B[gf];W[ge];B[bh];W[nf];B[ng];W[ne];B[ci];W[ai];B[mf];W[eg];B[gg];W[dg];B[ch];W[si];B[ah];W[bj];B[sj];W[sh];B[ma];W[sc];B[pa];W[id];B[ic];W[ls];B[ks];W[ms];B[sa];W[ra];B[od];W[pe];B[kh];W[lh];B[ji];W[ii];B[mh];W[ji];B[lg];W[mp];B[no];W[jn];B[km];W[as];B[fi];W[ej])";
    }
    else if(boardSize == 18) {
      sgfData = "(;FF[4]GM[1]SZ[18]HA[0]KM[7.5];B[od];W[do];B[oo];W[dd];B[cp];W[dp];B[co];W[cn];B[bn];W[bm];B[cm];W[dn];B[bl];W[bo];B[am];W[bp];B[cq];W[bq];B[fc];W[pp];B[op];W[po];B[pm];W[pn];B[on];W[qm];B[pq];W[pc];B[qn];W[rn];B[ql];W[qo];B[pl];W[qq];B[pd];W[oc];B[nd];W[mb];B[cf];W[fd];B[gd];W[fe];B[dc];W[cc];B[ec];W[df];B[ed];W[de];B[cb];W[bd];B[bc];W[cd];B[ch];W[bb];B[ba];W[ac];B[lc];W[lb];B[kc];W[mk];B[lm];W[mi];B[lj];W[mj];B[kl];W[mm];B[mn];W[nm];B[pi];W[ln];B[kn];W[lo];B[om];W[ko];B[jo];W[jp];B[io];W[ip];B[ho];W[hp];B[kh];W[oi];B[ph];W[oh];B[go];W[gp];B[fp];W[fq];B[fo];W[lq];B[eq];W[dq];B[mp];W[pg];B[og];W[ng];B[of];W[pj];B[qj];W[qi];B[qh];W[qk];B[ri];W[pk];B[rk];W[ki];B[li];W[lh];B[kj];W[ji];B[jj];W[kg];B[jh];W[ii];B[jg];W[kf];B[gq];W[hq];B[hh];W[mg];B[fr];W[lp];B[eh];W[dm];B[nk];W[oj];B[rl];W[dk];B[hf];W[gg];B[gh];W[fg];B[fh];W[db];B[da];W[ca];B[nq];W[ea];B[rp];W[rq];B[rm];W[qr];B[qn];W[cj];B[bj];W[qm];B[qc];W[qb];B[qn];W[pr];B[oq];W[qm];B[qd];W[gl];B[fm];W[gj];B[el];W[dl];B[ek];W[ej];B[fj];W[ij];B[il];W[bi];B[bk];W[ci];B[cl];W[ih];B[ig];W[fk];B[fi];W[ik];B[jk];W[im];B[jl];W[jn];B[hm];W[km];B[ll];W[ml];B[in];W[qn];B[lf];W[le];B[mf];W[lg];B[kb];W[hg];B[je];W[ie];B[he];W[id];B[if];W[gc];B[hd];W[gb];B[ol];W[jd];B[hc];W[kd];B[hb];W[md];B[mc];W[nb];B[me];W[jf];B[rb];W[oa];B[fb];W[cg];B[bh];W[bg];B[dh];W[ah];B[mo];W[ld];B[nf];W[jb];B[jc];W[ka];B[ic];W[ke];B[fl];W[eb];B[ga];W[hr];B[ia];W[ja];B[qa];W[pb];B[gk];W[kn];B[aj];W[ai];B[nj];W[ni];B[jm];W[ao];B[an];W[nn];B[no];W[lk];B[dr];W[cr];B[en];W[er];B[ib];W[la];B[dr];W[mr];B[jq];W[jr];B[kr];W[kq];B[ir];W[iq];B[mq];W[jr];B[er];W[pa];B[rc];W[ra])";
    }
    else if(boardSize == 17) {
      sgfData = "(;FF[4]GM[1]SZ[17]HA[0]KM[7.5];B[dd];W[nn];B[nd];W[dn];B[oo];W[on];B[no];W[mo];B[mp];W[lp];B[lo];W[mn];B[kp];W[np];B[lq];W[op];B[po];W[pp];B[cn];W[co];B[cm];W[do];B[dm];W[cc];B[cd];W[dc];B[fc];W[ec];B[ed];W[fb];B[bc];W[bb];B[gb];W[fd];B[gc];W[bd];B[pl];W[pm];B[be];W[ac];B[fe];W[nf];B[of];W[og];B[oe];W[ng];B[ld];W[kg];B[ig];W[ke];B[go];W[fn];B[fl];W[fo];B[ji];W[bf];B[cf];W[ae];B[li];W[mj];B[lf];W[ne];B[md];W[od];B[pd];W[oc];B[pc];W[ob];B[pg];W[ph];B[pb];W[lg];B[mf];W[mg];B[kf];W[mb];B[lb];W[kb];B[lc];W[pf];B[pe];W[qg];B[ch];W[la];B[pa];W[jf];B[jg];W[hb];B[jc];W[hc];B[gd];W[ga];B[jb];W[ce];B[hd];W[cg];B[bh];W[ho];B[ip];W[kk];B[mi];W[nj];B[ik];W[hl];B[hk];W[dg];B[dh];W[fg];B[eh];W[ff];B[gp];W[gn];B[hp];W[jo];B[jp];W[bo];B[bn];W[hf];B[if];W[ln];B[io];W[ko];B[hn];W[fp];B[gh];W[ha];B[ib];W[jj];B[ij];W[jm];B[lp];W[il];B[gl];W[jk];B[kh];W[hg];B[hh];W[an];B[am];W[em];B[el];W[ao];B[fq];W[ep];B[ni];W[oi];B[eb];W[fa];B[ic];W[db];B[df];W[de];B[ef];W[eg];B[ee];W[bg];B[nq];W[in];B[pq];W[qo];B[oq];W[pn];B[gq];W[lj];B[ki];W[he];B[ie];W[be];B[ge];W[ah];B[ai];W[ag];B[bj];W[qe];B[qd];W[hm];B[eq];W[dq];B[fh];W[me];B[le];W[qf];B[fm];W[en];B[ia];W[ea];B[nh];W[oh];B[ho];W[qq];B[mq];W[gm];B[qp];W[dl];B[cl];W[qq];B[kj];W[mh];B[qp];W[dk];B[ck];W[qq];B[gk];W[gf];B[qp];W[qb];B[nb];W[qq];B[oj];W[pj];B[qp];W[nc];B[na];W[qq];B[lh];W[gg];B[qp];W[ma];B[oa];W[qq];B[dp];W[cp];B[qp];W[id];B[jd];W[qq];B[cq];W[bq];B[qp];W[bi];B[ci];W[qq];B[qi];W[ok];B[qp];W[bm];B[bl];W[qq];B[qh];W[pg];B[qp];W[iq];B[jq];W[qq];B[pi];W[qj];B[qp];W[qa];B[mc];W[qq];B[da];W[ca];B[qp];W[jh];B[ii];W[qq];B[kn];W[jn];B[qp];W[aj];B[qq];W[bi];B[po];W[ai];B[qn];W[qm])";
    }
    else if(boardSize == 16) {
      sgfData = "(;FF[4]GM[1]SZ[16]HA[0]KM[7.5];B[mm];W[dd];B[md];W[dm];B[cc];W[cd];B[dc];W[ed];B[fb];W[ck];B[nf];W[nn];B[mn];W[nm];B[nl];W[ol];B[ok];W[nk];B[ml];W[oj];B[om];W[pk];B[on];W[no];B[oo];W[fn];B[bc];W[gd];B[ni];W[mk];B[dj];W[cj];B[dh];W[gh];B[ci];W[cg];B[dk];W[bi];B[dl];W[cl];B[em];W[en];B[hi];W[gi];B[hk];W[hj];B[fm];W[gm];B[fk];W[ij];B[dn];W[cm];B[gn];W[go];B[hn];W[ho];B[in];W[io];B[jn];W[jo];B[do];W[eo];B[ep];W[fo];B[cn];W[kn];B[bm];W[jm];B[gj];W[gl];B[bl];W[bk];B[gk];W[al];B[bn];W[mc];B[nc];W[lc];B[nb];W[me];B[nd];W[ld];B[hc];W[ch];B[di];W[fi];B[jj];W[ii];B[jl];W[mo];B[lo];W[ll];B[ln];W[im];B[hl];W[hm];B[kk];W[lk];B[kh];W[fl];B[ig];W[ih];B[jh];W[ie];B[el];W[lg];B[mh];W[jg];B[kg];W[jf];B[kf];W[ke];B[lf];W[le];B[og];W[bd];B[jd];W[je];B[hd];W[li];B[lj];W[mj];B[kj];W[ac];B[ab];W[ad];B[cb];W[he];B[dg];W[jb];B[am];W[mi];B[bj];W[aj];B[ak];W[lm];B[np];W[al];B[cf];W[bf];B[ak];W[ko];B[mp];W[al];B[kl];W[oi];B[ak];W[gb];B[gc];W[al];B[nh];W[km];B[ak];W[ne];B[oe];W[al];B[hg];W[gg];B[ak];W[fc];B[ec];W[fd];B[hb];W[al];B[hh];W[hf];B[ak];W[eb];B[ga];W[al];B[ik];W[if];B[ak];W[ca];B[ea];W[al];B[mb];W[ak];B[lb];W[lh];B[mf];W[id];B[kb];W[ao];B[bo];W[cp];B[dp];W[bp];B[jc];W[ic];B[ib];W[kc];B[ef];W[ja];B[bg];W[bh];B[be];W[af];B[ce];W[ff];B[ee];W[fe];B[fj];W[ei];B[ej];W[oh];B[ph];W[pi];B[pg];W[kd];B[pm];W[mg];B[ng];W[ae];B[pl];W[ok];B[fp];W[eg];B[df];W[eh];B[gp];W[ia];B[ha])";
    }
    else if(boardSize == 15) {
      sgfData = "(;FF[4]GM[1]SZ[15]HA[0]KM[7.5];B[dd];W[ll];B[ld];W[dl];B[dm];W[cm];B[em];W[cl];B[im];W[mc];B[lc];W[md];B[le];W[nf];B[el];W[km];B[dj];W[bj];B[mj];W[mg];B[nl];W[mm];B[kk];W[jl];B[nm];W[il];B[hm];W[hl];B[bi];W[cj];B[ci];W[gm];B[gn];W[gl];B[ei];W[fn];B[cn];W[bn];B[en];W[co];B[fm];W[dc];B[ec];W[cd];B[ed];W[cc];B[in];W[ii];B[cf];W[ce];B[df];W[eb];B[fb];W[da];B[ki];W[ig];B[lg];W[gh];B[gj];W[jj];B[kj];W[fj];B[fk];W[gf];B[gi];W[hd];B[hh];W[ih];B[hi];W[hj];B[hg];W[hf];B[if];W[gk];B[fi];W[lh];B[ie];W[kg];B[lf];W[kh];B[gc];W[nn];B[nh];W[mh];B[ni];W[ml];B[mk];W[om];B[ok];W[on];B[ng];W[og];B[lk];W[li];B[kl];W[mf];B[mn];W[lm];B[no];W[ln];B[mo];W[lo];B[jm];W[oo];B[mo];W[jn];B[jo];W[ko];B[kn];W[id];B[mn];W[jc];B[kb];W[jb];B[mb];W[nb];B[je];W[gd];B[na];W[nc];B[ff];W[jd];B[hc];W[ke];B[kf];W[kd];B[gg];W[he];B[jf];W[fe];B[ja];W[ia];B[la];W[ka];B[dn];W[ef];B[de];W[bl];B[ja];W[ib];B[fg];W[fd];B[fc];W[ka];B[jk];W[ik];B[ja];W[kc];B[hb];W[me];B[ee];W[jg];B[lb];W[ka];B[bo];W[ao];B[ja];W[oa];B[ob];W[ka];B[am];W[dk];B[ja];W[bf];B[ha];W[ka];B[an];W[bo];B[ja];W[ek];B[ej];W[ka];B[aj];W[ak];B[ja];W[ai];B[bg];W[be];B[af];W[ka];B[ji];W[ij];B[ja];W[ae];B[ag];W[ka];B[ea];W[ma];B[bb];W[cb];B[db];W[ca];B[bc];W[eb];B[fa];W[ab];B[ac])";
    }
    else if(boardSize == 14) {
      sgfData = "(;FF[4]GM[1]SZ[14]HA[0]KM[7.5];B[dk];W[kd];B[kk];W[dd];B[dc];W[cc];B[ec];W[cb];B[ic];W[ed];B[fd];W[jc];B[gc];W[ef];B[id];W[le];B[cg];W[bf];B[il];W[cl];B[dl];W[ck];B[cj];W[bj];B[bi];W[ci];B[dj];W[bh];B[bk];W[ai];B[bl];W[cm];B[bm];W[di];B[lg];W[jb];B[lj];W[fe];B[ei];W[eh];B[fi];W[kg];B[kh];W[he];B[ib];W[jg];B[hd];W[jh];B[ja];W[lh];B[kb];W[lc];B[lb];W[mb];B[db];W[hk];B[hi];W[ik];B[jk];W[hl];B[ij];W[jl];B[jm];W[hj];B[ii];W[gi];B[gh];W[gj];B[fh];W[im];B[km];W[em];B[dm];W[fk];B[gm];W[hh];B[fj];W[el];B[gl];W[en];B[gk];W[hm];B[gn];W[cn];B[dn];W[ej];B[bn];W[ek];B[gf];W[jj];B[eg];W[ih];B[dh];W[mi];B[in];W[ji];B[hn];W[ll];B[kl];W[mj];B[mk];W[ml];B[lk];W[nk];B[mm];W[nm];B[lm];W[bg];B[cd];W[ce];B[ge];W[bd];B[ma];W[mc];B[ca];W[ba];B[da];W[mn];B[li];W[mh];B[ki];W[mg];B[bb];W[bc];B[aa];W[je];B[ff];W[ee];B[nl];W[ml];B[if];W[ie];B[hf];W[jf])";
    }
    else if(boardSize == 13) {
      sgfData = "(;FF[4]GM[1]SZ[13]HA[0]KM[7.5];B[dd];W[jj];B[kk];W[jd];B[kj];W[dj];B[jc];W[dc];B[cc];W[ec];B[ed];W[fc];B[fd];W[ic];B[hc];W[kc];B[gc];W[ch];B[jb];W[ib];B[id];W[ie];B[hd];W[kb];B[je];W[ja];B[kd];W[jc];B[jf];W[jk];B[ji];W[ii];B[fk];W[kl];B[ll];W[il];B[km];W[dk];B[hk];W[jg];B[kg];W[jh];B[ki];W[gj];B[hj];W[gk];B[hl];W[kh];B[if];W[lg];B[gi];W[li];B[jl];W[le];B[bg];W[bh];B[cg];W[dg];B[df];W[gb];B[dh];W[di];B[eg];W[cb];B[bb];W[db];B[bj];W[bi];B[aj];W[ai];B[cl];W[bk];B[dl];W[fj];B[fl];W[fi];B[cj];W[ck];B[bl];W[gh];B[hi];W[eh];B[ig];W[al];B[ek];W[ej];B[fg];W[fh];B[ih];W[kf];B[ba];W[gl];B[gm];W[hb];B[ke];W[ld];B[gg];W[ag];B[bf];W[bc];B[cd];W[ca];B[bd];W[lj];B[lk];W[hh];B[hg];W[mk];B[ml];W[mj];B[dg];W[bm];B[cm];W[ci];B[am];W[ac];B[ad];W[bm];B[af];W[ah];B[am];W[em];B[ak];W[al];B[fm];W[bm];B[ak];W[aj];B[am];W[ab];B[aa];W[bm];B[fb];W[fa];B[am];W[el];B[dm];W[bm];B[hm];W[ak];B[am];W[em];B[el];W[bm];B[ea];W[eb];B[am];W[gd])";
    }
    else if(boardSize == 12) {
      sgfData = "(;FF[4]GM[1]SZ[12]HA[0]KM[7.5];B[ii];W[dd];B[cc];W[cd];B[dc];W[ed];B[ec];W[di];B[fd];W[jd];B[cg];W[fe];B[gd];W[bc];B[jf];W[hc];B[hd];W[ic];B[eg];W[be];B[fi];W[gf];B[gb];W[fj];B[gj];W[ej];B[gk];W[fh];B[fg];W[if];B[je];W[gi];B[gg];W[hi];B[hh];W[hj];B[ij];W[hk];B[ik];W[il];B[jl];W[hl];B[hb];W[ie];B[id];W[jg];B[kg];W[kh];B[ig];W[jh];B[hf];W[ih];B[he];W[kf];B[ke];W[lg];B[bb];W[ab];B[ba];W[bg];B[bh];W[cf];B[ag];W[bf];B[ch];W[df];B[eh];W[ei];B[gh];W[jk];B[ck];W[le];B[ld];W[lf];B[kd];W[cj];B[bj];W[dk];B[ci];W[cl];B[ee];W[ef];B[ff];W[de];B[fi];W[kl];B[dj];W[bk];B[cj];W[fk];B[ak];W[bl];B[af];W[ae];B[ah];W[fh];B[ge];W[fi];B[ee];W[cb];B[db];W[fe];B[ac];W[ad];B[ee];W[dl];B[aj];W[fe];B[li];W[ki];B[ee];W[ac];B[al];W[fe];B[ek];W[el];B[ee];W[aa];B[ca];W[fe];B[lh];W[lj];B[ee];W[ea];B[da])";
    }
    else if(boardSize == 11) {
      sgfData = "(;FF[4]GM[1]SZ[11]HA[0]KM[7.5];B[hd];W[dh];B[dc];W[hh];B[cf];W[id];B[ie];W[he];B[hc];W[dd];B[cd];W[ec];B[de];W[ed];B[db];W[ic];B[ib];W[jb];B[hb];W[je];B[if];W[jf];B[ig];W[ge];B[ih];W[fb];B[ja];W[kb];B[jd];W[jc];B[kd];W[ke];B[jg];W[kc];B[eb];W[fc];B[ee];W[fe];B[gg];W[ff];B[fi];W[ej];B[bh];W[hi];B[fg];W[fj];B[ef];W[gd];B[ii];W[cj];B[ci];W[bj];B[gj];W[gi];B[hj];W[di];B[hg];W[ch];B[bg];W[bi];B[eh];W[dg];B[ea];W[gb];B[ai];W[aj];B[ah];W[kg];B[kh];W[ei];B[fh];W[gk];B[hk];W[fk];B[cg];W[eg];B[kf];W[cc];B[bc];W[df];B[ce];W[kg];B[jd];W[kd];B[cb];W[kf];B[hf];W[ga];B[fa];W[ha];B[ia];W[gc];B[gf])";
    }
    else if(boardSize == 10) {
      sgfData = "(;FF[4]GM[1]SZ[10]HA[0]KM[7.5];B[gc];W[gg];B[dh];W[cd];B[dc];W[cg];B[ch];W[cc];B[he];W[hf];B[ie];W[fh];B[bg];W[gd];B[hc];W[ec];B[db];W[dd];B[eb];W[bf];B[cf];W[dg];B[be];W[bh];B[af];W[bi];B[eg];W[df];B[ef];W[de];B[eh];W[ce];B[ci];W[bf];B[gf];W[hh];B[cf];W[bj];B[cj];W[bf];B[fi];W[gi];B[fg];W[gh];B[if];W[ei];B[cf];W[fe];B[ff];W[bf];B[ah];W[ag];B[ii];W[ih];B[bg];W[hd];B[id];W[ag];B[ej];W[fj];B[bg];W[dj];B[cf];W[fc];B[fb];W[bf];B[di];W[ag];B[ej];W[bg];B[ji];W[dj];B[jh];W[ge];B[hg];W[ig])";
    }
    else if(boardSize == 9) {
      sgfData = "(;FF[4]GM[1]SZ[9]HA[0]KM[7.5];B[ef];W[ed];B[ge];W[gc];B[cc];W[cd];B[bd];W[ce];B[be];W[dg];B[cf];W[df];B[de];W[dd];B[ee];W[cg];B[bf];W[cb];B[eg];W[bc];B[bh];W[he];B[hd];W[gf];B[fe];W[hf];B[fc];W[eb];B[gd];W[fh];B[eh];W[hh];B[ac];W[dc];B[fb];W[ab];B[fg];W[gg];B[fi];W[bg];B[dh];W[gh];B[ea];W[da];B[fa];W[ad];B[ch];W[id];B[ic];W[ie];B[gb];W[gi];B[ec];W[hc];B[hb];W[ei];B[db];W[ae];B[ag];W[eb];B[ig];W[db];B[ih];W[ii];B[di];W[ac];B[fi];W[hg];B[ei];W[af];B[ff];W[if];B[fd];W[bb])";
    }
    else {
      ASSERT_UNREACHABLE;
    }

    sgf = CompactSgf::parse(sgfData);
  }

  Logger logger;
  logger.setLogToStdout(true);
  logger.write("Loading model and initializing benchmark...");

  Rules initialRules = Setup::loadSingleRulesExceptForKomi(cfg);
  //Take the komi from the sgf, otherwise ignore the rules in the sgf
  initialRules.komi = sgf->komi;


  //Pick random positions from the SGF file, but deterministically
  vector<Move> moves = sgf->moves;
  string posSeed = "benchmarkPosSeed|";
  for(int i = 0; i<moves.size(); i++) {
    posSeed += Global::intToString((int)moves[i].loc);
    posSeed += "|";
  }
  Rand posRand(posSeed);
  vector<int> possiblePositionIdxs;
  for(int i = 0; i<moves.size(); i++) {
    possiblePositionIdxs.push_back(i);
  }
  for(int i = possiblePositionIdxs.size()-1; i > 1; i--) {
    int r = posRand.nextUInt(i);
    int tmp = possiblePositionIdxs[i];
    possiblePositionIdxs[i] = possiblePositionIdxs[r];
    possiblePositionIdxs[r] = tmp;
  }
  if(possiblePositionIdxs.size() > numPositionsPerGame)
    possiblePositionIdxs.resize(numPositionsPerGame);

  std::sort(possiblePositionIdxs.begin(),possiblePositionIdxs.end());

  SearchParams params = Setup::loadSingleParams(cfg);
  params.maxVisits = maxVisits;
  params.maxPlayouts = maxVisits;
  params.maxTime = 1e20;

  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    int maxConcurrentEvals = maxNumThreadsInATest * 2 + 16; // * 2 + 16 just to give plenty of headroom
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,cfg,logger,seedRand,maxConcurrentEvals,
      sgf->xSize,sgf->ySize
    );
  }
  logger.write("Loaded neural net");

  //Run on a sample position just to get any initialization and logs out of the way
  {
    Board board(sgf->xSize,sgf->ySize);
    BoardHistory hist;
    Player nextPla = P_BLACK;
    SearchParams thisParams = params;
    thisParams.numThreads = 1;
    thisParams.maxVisits = 5;
    thisParams.maxPlayouts = 5;
    thisParams.maxTime = 1e20;
    AsyncBot* bot = new AsyncBot(thisParams, nnEval, &logger, Global::uint64ToString(seedRand.nextUInt64()));
    bot->setPosition(nextPla,board,hist);
    bot->genMoveSynchronous(nextPla,TimeControls());
  }
  cout.flush();
  cerr.flush();
  //Sleep a bit
  std::this_thread::sleep_for(std::chrono::duration<double>(0.2));

  auto testNumThreads = [&](
    int numThreads,
    int64_t& totalVisits,
    double& totalSeconds,
    int64_t& totalPositions,
    int64_t& numNNEvals,
    int64_t& numNNBatches,
    double& avgBatchSize
  ) {
    totalPositions = 0;
    totalVisits = 0;
    totalSeconds = 0.0;

    nnEval->clearCache();
    nnEval->clearStats();

    SearchParams thisParams = params;
    thisParams.numThreads = numThreads;
    AsyncBot* bot = new AsyncBot(thisParams, nnEval, &logger, Global::uint64ToString(seedRand.nextUInt64()));

    Board board;
    Player nextPla;
    BoardHistory hist;
    sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);

    int moveNum = 0;

    cout << endl;
    for(int i = 0; i<possiblePositionIdxs.size(); i++) {
      cout << "\rnumSearchThreads = " << Global::strprintf("%2d",numThreads) << ":"
           << " " << totalPositions << " / " << possiblePositionIdxs.size() << " positions,"
           << " visits/s = " << Global::strprintf("%.2f",totalVisits / totalSeconds)
           << " (" << Global::strprintf("%.1f", totalSeconds) << " secs)"
           << "      "
           << std::flush;
      int nextIdx = possiblePositionIdxs[i];
      while(moveNum < moves.size() && moveNum < nextIdx) {
        bool suc = hist.makeBoardMoveTolerant(board,moves[moveNum].loc,moves[moveNum].pla);
        if(!suc) {
          cerr << endl;
          cerr << board << endl;
          cerr << "SGF Illegal move " << (moveNum+1) << " for " << PlayerIO::colorToChar(moves[moveNum].pla) << ": " << Location::toString(moves[moveNum].loc,board) << endl;
          throw StringError("Illegal move in SGF");
        }
        nextPla = getOpp(moves[moveNum].pla);
        moveNum += 1;
      }

      bot->clearSearch();
      bot->setPosition(nextPla,board,hist);
      nnEval->clearCache();

      ClockTimer timer;
      bot->genMoveSynchronous(nextPla,TimeControls());
      double seconds = timer.getSeconds();

      totalPositions += 1;
      totalSeconds += seconds;
      totalVisits += bot->getSearch()->getRootVisits();
    }

    numNNEvals = nnEval->numRowsProcessed();
    numNNBatches = nnEval->numBatchesProcessed();
    avgBatchSize = nnEval->averageProcessedBatchSize();

    delete bot;
  };

  //From some test matches by lightvector using g104
  const double eloCostPerThread = 8;
  const double eloGainPerDoubling = 250;

  cout << endl;
  cout << "Testing using " << maxVisits << " visits.";
  if(maxVisits == defaultMaxVisits)
    cout << " If you have a good GPU, you might increase this using -visits N to get more accurate results." << endl;
  else
    cout << endl;
  cout << endl;
#ifdef USE_CUDA_BACKEND
  cout << "Your GTP config is currently set to cudaUseFP16 = " << Global::boolToString(nnEval->getUsingFP16())
       << " and cudaUseNHWC = " << Global::boolToString(nnEval->getUsingNHWC()) << endl;
  if(!nnEval->getUsingFP16())
    cout << "If you have a strong GPU capable of FP16 tensor cores (e.g. RTX2080) setting these both to true may give a large performance boost." << endl;
#endif
#ifdef USE_OPENCL_BACKEND
  cout << "You are currently using the OpenCL version of KataGo." << endl;
  //TODO update when we have FP16 opencl
  cout << "If you have a strong GPU capable of FP16 tensor cores (e.g. RTX2080), "
       << "downloading or compiling the Cuda version of KataGo and setting cudaUseFP16=true and cudaUseNHWC=true may give a large performance boost." << endl;
#endif
  cout << endl;
  cout << "Your GTP config is currently set to use numSearchThreads = " << params.numThreads << endl;
  if(numThreadsToTest.size() > 1)
    cout << "Testing different numbers of threads: " << endl;
  vector<double> eloEffects;
  for(int i = 0; i<numThreadsToTest.size(); i++) {
    int numThreads = numThreadsToTest[i];
    int64_t totalVisits;
    double totalSeconds;
    int64_t totalPositions;
    int64_t numNNEvals;
    int64_t numNNBatches;
    double avgBatchSize;
    testNumThreads(numThreads,totalVisits,totalSeconds,totalPositions,numNNEvals,numNNBatches,avgBatchSize);
    eloEffects.push_back(eloGainPerDoubling * log(totalVisits / totalSeconds) / log(2) - eloCostPerThread * numThreads);
    cout << "\rnumSearchThreads = " << Global::strprintf("%2d",numThreads) << ":"
         << " " << totalPositions << " / " << possiblePositionIdxs.size() << " positions,"
         << " visits/s = " << Global::strprintf("%.2f",totalVisits / totalSeconds)
         << " nnEvals/s = " << Global::strprintf("%.2f",numNNEvals / totalSeconds)
         << " nnBatches/s = " << Global::strprintf("%.2f",numNNBatches / totalSeconds)
         << " avgBatchSize = " << Global::strprintf("%.2f",avgBatchSize)
         << " (" << Global::strprintf("%.1f", totalSeconds) << " secs)";
    if(numThreadsToTest.size() > 1) {
      if(i == 0)
        cout << " (EloDiff baseline)";
      else
        cout << " (EloDiff " << Global::strprintf("%+.0f",eloEffects[i] - eloEffects[0]) << ")";
    }
    cout << std::flush;
  }
  cout << endl;

  if(numThreadsToTest.size() > 1) {
    cout << endl;
    cout << "Based on some test data, each thread costs perhaps ~" << eloCostPerThread << " Elo holding visits fixed (by making MCTS worse)." << endl;
    cout << "Based on some test data, each speed doubling gains perhaps ~" << eloGainPerDoubling << " Elo by searching deeper." << endl;
    cout << "So APPROXIMATELY based on this benchmark: " << endl;
    for(int i = 0; i<numThreadsToTest.size(); i++) {
      int numThreads = numThreadsToTest[i];
      double eloEffect = eloEffects[i] - eloEffects[0];
      cout << "numSearchThreads = " << Global::strprintf("%2d",numThreads) << ": ";
      if(i == 0)
        cout << "(baseline)" << endl;
      else
        cout << Global::strprintf("%+5.0f",eloEffect) << " Elo" << endl;
    }
    cout << endl;
    cout << "If you care about performance, you may want to edit numSearchThreads in " << configFile << " based on the above results!" << endl;
    cout << "If interested see also other notes about performance and mem usage in the top of that file." << endl;
  }
  cout << endl;

  delete nnEval;
  NeuralNet::globalCleanup();
  delete sgf;
  ScoreValue::freeTables();

  return 0;
}
