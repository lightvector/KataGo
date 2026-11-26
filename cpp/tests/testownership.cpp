#include "../tests/tests.h"

#include "../neuralnet/nninputs.h"
#include "../search/search.h"
#include "../program/playutils.h"
#include "../program/setup.h"

using namespace std;

void Tests::runOwnershipTests(const string& configFile, const string& modelFile) {
  ConfigParser cfg(configFile);

  const bool logToStdoutDefault = false;
  const bool logToStderrDefault = true;
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);

  Rand seedRand;

  Rules ttRules = Rules::parseRules("tromp-taylor");
  Rules jpRules = Rules::parseRules("japanese");

  int nnXLen = 19;
  int nnYLen = 19;
  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = params.numThreads;
    const bool requireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      nnXLen,nnYLen,Setup::MaxBatchSizeRequest::fromConcurrency(),requireExactNNLen,disableFP16,
      Setup::SETUP_FOR_GTP
    );
  }

  Search* bot = new Search(params, nnEval, &logger, Global::uint64ToString(seedRand.nextUInt64()));

  auto runOnBoard = [&](const Board& board, const Rules& rules) {
    Player nextPla = P_BLACK;
    BoardHistory hist(board,nextPla,rules,0,BoardHistoryModes(false,false));
    int64_t numVisits = 100;
    vector<double> ownership = PlayUtils::computeOwnership(bot,board,hist,nextPla,numVisits);
    cout << "=================================================================================" << endl;
    cout << rules << endl;
    cout << board << endl;
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        int ownershipValue = (int)round(100*ownership[pos]);
        string s;
        if(ownershipValue >= 99)
          s = "    W";
        else if(ownershipValue <= -99)
          s = "    B";
        else
          s = Global::strprintf(" %+4d", ownershipValue);
        cout << s;
      }
      cout << endl;
    }
    cout << endl;
  };

  {
    Board board = Board::parseBoard(17,17,R"%%(
.................
.................
.................
...*....*....*...
.................
.................
.................
.................
...*....*....*...
.................
.................
.................
.................
...*....*....*...
.................
.................
.................
)%%");

    runOnBoard(board,ttRules);
    runOnBoard(board,jpRules);
  }

  {
    Board board = Board::parseBoard(17,17,R"%%(
.............xo.x
...........o.xoo.
...o...o..o.x.xo.
.oxo..o.*.ox.*xo.
.xoo.....oxx..xx.
.........o.......
...o....o.x..x...
.........ox......
...o....*ox..*...
.........ox......
..o.o...ox...x...
....xo..ox.......
...o.o..ox....x..
...*oxxx*x.x.*...
oooooxox.x...xx..
xxxxxooox...oxo..
.oox.o.ox........
)%%");

    runOnBoard(board,ttRules);
    runOnBoard(board,jpRules);
  }

  {
    Board board = Board::parseBoard(17,17,R"%%(
x.o.......oxx.xo.
xoox..x..xoox.x.o
xo.x....x.x.oxxxx
.ox*...x*xoo.oooo
oox..xxoxxxox....
.xx.xo.oooo.ox...
....xo......o....
....xo........o..
...x.o..*....*...
....xo......o....
.....xo.o.....o..
..x..x...........
.....x.oooooo....
xxxx..xx*x..xoooo
ooooxx...xoooxxxx
x.o.ox..x.oxxxx.o
.xo.ox....oxo.xo.
)%%");

    runOnBoard(board,ttRules);
    runOnBoard(board,jpRules);
  }

  {
    Board board = Board::parseBoard(17,17,R"%%(
x.o......xoxx.xo.
xoox..x..xoox.x.o
xo.x....xxxooxxxx
.ox*...x*xoo.oooo
oox..xxoxxxox....
.xx.xoxoooo.ox...
....xoo.....o....
....xo.o......o..
...xxo..*....*...
....xo......o....
.....xo.o.....o..
..x..xxo.........
.....x.oooooo....
xxxx..xxoxo.xoooo
ooooxx..xxoooxxxx
x.o.ox.xx.oxxxx.o
.xo.ox.xoooxo.xo.
)%%");

    runOnBoard(board,ttRules);
    runOnBoard(board,jpRules);
  }

  {
    Board board = Board::parseBoard(17,17,R"%%(
....oxx......xxo.
..oxoox......xoxx
..oo.ox..x.x.xoxx
...*oox.*....xooo
..o.oxxx.....xo..
....oox.xxxxxxox.
ooooox.xooxooo.o.
xoxxxxxo.oo...o..
xx.x..xo*...o*...
..x...xo.....o...
.....xxoooooo.o..
...x.xoo.oxxxo.o.
......xooxx..xoo.
xxxxxxxox.x.xxxoo
xoooooxxx...xoxxo
ooxx.oxox.x....xx
o.xx.ox..........
)%%");

    runOnBoard(board,ttRules);
    runOnBoard(board,jpRules);
  }



  delete bot;
  delete nnEval;
}

void Tests::runNearEndByOwnershipTests() {
  cout << "Running near end by ownership tests" << endl;

  //Checks the largest count of affected locations that is still accepted, for each kind of count.
  auto testBoard = [&](int xSize, int ySize, int nnXLen, int nnYLen, int maxAccepted) {
    Board board(xSize,ySize);
    for(int count = maxAccepted; count <= maxAccepted+1; count++) {
      for(int kind = 0; kind < 3; kind++) {
        //Locations off the board are left unsettled to check that they are ignored.
        vector<double> ownershipIfBlackFirst(nnXLen*nnYLen, 0.0);
        vector<double> ownershipIfWhiteFirst(nnXLen*nnYLen, 0.0);
        for(int y = 0; y<ySize; y++) {
          for(int x = 0; x<xSize; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            bool affected = y * xSize + x < count;
            ownershipIfBlackFirst[pos] = (affected && kind == 0) ? 0.5 : 1.0;
            ownershipIfWhiteFirst[pos] = (affected && kind == 1) ? 0.5 : (affected && kind == 2) ? -1.0 : 1.0;
          }
        }
        bool expected = count <= maxAccepted;
        testAssert(PlayUtils::isGameNearEndByOwnership(board,ownershipIfBlackFirst,ownershipIfWhiteFirst,nnXLen) == expected);
      }
    }
  };
  testBoard(19,19,19,19,42);
  testBoard(9,9,19,19,14);
  testBoard(13,7,19,19,15);
}
