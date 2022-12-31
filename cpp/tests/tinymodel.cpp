#include "../tests/tinymodel.h"

#include <fstream>
#include <sstream>

#include "../core/fileutils.h"
#include "../core/rand.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"

using namespace std;

static void decodeBase64(const string& input, string& output) {
  output = "";

  string valueToCharMap = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int32_t charToValueMap[256];
  for(int i = 0; i<256; i++)
    charToValueMap[i] = -1;
  for(int i = 0; i<valueToCharMap.size(); i++)
    charToValueMap[(int)valueToCharMap[i]] = i;

  int32_t carry = 0;
  int numBitsInCarry = 0;
  for(unsigned char c : input) {
    if(c == '\n' || c == '\r' || c == ' ')
      continue;
    int value = charToValueMap[c];
    if(value < 0)
      break;

    carry = (carry << 6) + value;
    numBitsInCarry += 6;
    if(numBitsInCarry >= 8) {
      int extracted = carry >> (numBitsInCarry-8);
      carry -= (extracted << (numBitsInCarry-8));
      numBitsInCarry -= 8;
      assert(extracted >= 0 && extracted < 256);
      output.push_back((char)extracted);
    }
  }
  if(carry != 0)
    throw StringError("decodeBase64 got leftover bits");
}

static void requireApproxEqual(double x, double expected, double scale, const NNResultBuf& buf, const Board& board) {
  if(!std::isfinite(x) || !std::isfinite(expected) || std::fabs(x-expected) > scale) {
    buf.result->debugPrint(cout,board);
    throw StringError("Tiny neural net test got invalid values - is the GPU working?");
  }
}

NNEvaluator* TinyModelTest::runTinyModelTest(const string& baseDir, Logger& logger, ConfigParser& cfg, bool randFileName) {
  logger.write("Running tiny net to sanity-check that GPU is working");

  string base64Data;
  base64Data += TinyModelTest::tinyModelBase64Part0;
  base64Data += TinyModelTest::tinyModelBase64Part1;
  base64Data += TinyModelTest::tinyModelBase64Part2;
  base64Data += TinyModelTest::tinyModelBase64Part3;
  base64Data += TinyModelTest::tinyModelBase64Part4;
  base64Data += TinyModelTest::tinyModelBase64Part5;
  base64Data += TinyModelTest::tinyModelBase64Part6;
  string binaryData;
  decodeBase64(base64Data, binaryData);

  Rand rand;
  const string tmpModelFile =
    randFileName ?
    (baseDir + "/" + "tmpTinyModel_" + Global::uint64ToHexString(rand.nextUInt64()) + ".bin.gz") :
    (baseDir + "/" + "tmpTinyModel.bin.gz");
  ofstream outModel;
  FileUtils::open(outModel,tmpModelFile.c_str(),ios::binary);
  outModel << binaryData;
  outModel.close();

  const int maxConcurrentEvals = 8;
  const int expectedConcurrentEvals = 1;
  const int maxBatchSize = 8;
  const bool requireExactNNLen = false;
  const bool disableFP16 = false;
  const string expectedSha256 = "";
  NNEvaluator* nnEval = Setup::initializeNNEvaluator(
    "tinyModel",tmpModelFile,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
    NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,maxBatchSize,requireExactNNLen,disableFP16,
    Setup::SETUP_FOR_DISTRIBUTED
  );
  nnEval->setDoRandomize(false);
  nnEval->setDefaultSymmetry(6);

  Board board = Board::parseBoard(19,19,R"%%(
...................
...................
..xx......o.x..oo..
..xo.o.........ox..
................x..
...x..........oox..
...............x...
...................
...................
..x................
...................
...................
...................
.....o.......o.x...
...o.x.............
. xo.........o.x...
..xxooo......ox....
....xx.............
...................
)%%");

  const Player nextPla = P_BLACK;
  const Rules rules = Rules::getTrompTaylorish();
  const BoardHistory hist(board,nextPla,rules,0);

  auto runOneTest = [&]() {
    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    bool includeOwnerMap = true;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache,includeOwnerMap);

    //ostringstream out;
    //buf.result->debugPrint(out,board);
    //cout << out.str() << endl;

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale), buf, board)
    NNOutput& nnOutput = *(buf.result);
    EQ(nnOutput.whiteWinProb, .43298, 0.01);
    EQ(nnOutput.whiteLossProb, .56702, 0.01);
    EQ(nnOutput.whiteNoResultProb, .0000, 0.002);
    EQ(nnOutput.whiteScoreMean, -0.81, 0.2);
    EQ(nnOutput.whiteScoreMeanSq, 130.83, 0.5);
    EQ(nnOutput.whiteLead, -0.61, 0.2);
    EQ(nnOutput.varTimeLeft, 15.79, 0.5);
    EQ(nnOutput.shorttermWinlossError, .22408, 0.01);
    EQ(nnOutput.shorttermScoreError, 2.787, 0.05);

    double expectedPolicy[361] = {
      0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,
      0,    0,    0,    0,    3,   46,    7,    9,   14,    9,   39,   67,   12,   57,   52,    1,    1,   11,    0,
      0,    0,   -1,   -1,   36,   25,   83,  121,   79,   18,   -1,   22,   -1,    7,    3,   -1,   -1,  304,    1,
      0,    0,   -1,   -1,   33,   -1,    7,   95,   42,   37,   83,   78,   22,   33,   19,   -1,   -1,   31,    1,
      0,    0,    1,    3,    7,    3,   16,   22,   11,   20,   33,   84,  108,   14,    5,  147,   -1,    0,    0,
      0,    0,    0,   -1,    5,   23,   12,    8,    4,    5,    6,    8,   27,    4,   -1,   -1,   -1,    0,    0,
      0,    0,    2,    0,    1,    4,    4,    4,    3,    2,    3,    4,   14,   24,  294,   -1,   46,    7,    0,
      0,    0,    6,    4,    1,    2,    2,    2,    2,    2,    2,    3,    7,   45,   36,   57,   80,    4,    0,
      0,    1,    2,    3,    2,    1,    2,    2,    2,    2,    2,    2,    3,    6,    9,   91,   49,    3,    0,
      0,    0,   -1,    1,    3,    2,    2,    2,    2,    2,    2,    2,    2,    3,    4,   33,   45,    2,    0,
      0,    1,    2,    6,    4,    3,    3,    3,    2,    2,    3,    3,    4,    3,    5,   45,   77,    3,    0,
      0,    3,   40,   15,    8,   15,    7,    5,    3,    3,    4,    5,    6,   18,   31,  143,   98,    3,    0,
      0,   13,  152,  120,   30,   65,   21,   14,    4,    4,    5,   10,   13,   14,   22,   57,   46,    2,    0,
      0,   47,  270,   66,  724,   -1, 1055,   21,    6,    7,    6,   19,    2,   -1,   18,   -1,    3,    2,    0,
      0,   19,  490,   -1,   19,   -1,  480,  185,   19,   10,   11,   17,    6,    7,   36,    6,    8,    2,    0,
      0,    0,   -1,   -1,   40,    0,   10,   89,   69,   50,   40,   78,    1,   -1,   12,   -1,    5,    2,    0,
      0,    0,   -1,   -1,   -1,   -1,   -1,   89,  103,   19,   30,   85,    2,   -1,   -1,    1,   23,    1,    0,
      0,    1,    0,    4,   -1,   -1,  465,  195,   13,    1,    2,   17,   11,  288,  435,   33,    4,    0,    0,
      0,    0,    0,    0,    0,    0,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    };
    for(int pos = 0; pos<361; pos++) {
      double prob = nnOutput.policyProbs[pos];
      if(expectedPolicy[pos] >= 0) {
        EQ(prob*10000, expectedPolicy[pos], std::min(40.0, expectedPolicy[pos] * 0.1 + 1.0));
      }
    }

    double expectedOwnership[361] = {
      -6334, -6906, -6958, -5779, -3636,  -722,  1417,  2059,  2397,  2729,  2500,  1762,  1535,  1847,  4035,  5789,  6028,  6529,  6013,
      -6957, -8376, -7921, -7229, -3072,  1525,  1702,  2352,  2644,  3310,  2634,  2001,   712,  1506,  3976,  6828,  7159,  5612,  4395,
      -6957, -8035, -9373, -9111,     0,  1509,  1903,  1429,  2007,  2206,  7261,   373, -4878,   145,  3420,  8908,  8632,  3277,  1820,
      -6695, -7518, -9239, -3754, -1587,  5671,  1390,  1334,   823,  1965,  1752,  1137,    -8,   196,  4567,  8704, -7778,  -946, -1346,
      -6069, -6785, -6344, -3785,   -34,  1656,  1828,   691,   453,   558,   968,   255,   306,  1497,  4482,   437, -8290, -4663, -3208,
      -5524, -5731, -4610, -8527, -1151,   618,    91,   292,     0,   194,   169,   318,  1034,  1736,  8053,  8247, -8541, -5412, -4288,
      -4593, -4550, -3511, -2336, -1105,     0,    13,     0,     0,     0,     0,    35,   398,  1508,   172, -6878, -3003, -4050, -4192,
      -4126, -3721, -2403,  -665,     0,     0,     0,     0,     0,     0,     0,     0,   118,    88,     0,  -174, -1577, -2594, -3181,
      -3847, -3635, -2893, -1044,     0,     0,     0,     0,     0,     0,     0,     0,     0,    55,     0,     0,     0, -1185, -1558,
      -3321, -3414, -8571, -1752,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  -121,  -734,
      -2815, -2060, -1968,  -413,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    30,    45,  -386,
      -2255, -1002,     0,     0,     0,  1102,   308,   182,     0,     0,     0,     0,   197,   507,     0,     0,     0,  -232,  -798,
      -1534,  -125,   253,  1150,  1824,  1406,  1390,   342,     0,     0,   295,   593,  1988,   928,     0, -1780, -1535,  -998, -1624,
      -1942,  -624,  1433,  1827,  2878,  8255,   400,   473,    39,     0,   744,  1431,  2018,  8272,  -975, -8493, -3678, -2663, -2885,
      -3253, -2106,     0,  8275,  2996,  -299,  1219,   351,   219,   200,  1099,  1655,  3988,  4856,     0, -2935, -4009, -3458, -3758,
      -4675, -5018, -8646,  8619,  5631,  3303,  2735,  1526,   114,    23,   963,  1884,  4524,  8485, -1602, -8646, -4367, -3987, -4216,
      -6035, -7075, -9062, -8585,  8344,  8468,  8040,  1607,  1052,   362,  1144,  1837,  4533,  8249, -7754, -4197, -4242, -3966, -4009,
      -6308, -7702, -7512, -6024, -8292, -7443,  1230,  2138,  1192,  1304,  1824,  2393,  3535,  3321,     0, -2770, -3210, -4067, -3169,
      -6356, -6400, -7044, -6989, -6759, -5093, -1934,  1007,  1366,  1482,  2363,  3037,  3605,  1452,  -952, -3103, -3531, -3474, -3516,
    };
    for(int pos = 0; pos<361; pos++) {
      double ownership = nnOutput.whiteOwnerMap[pos];
      EQ(ownership*10000, expectedOwnership[pos], 200.0);
    }
  };

  runOneTest();

  auto runAFewTests = [&]() {
    try {
      for(int i = 0; i<10; i++)
        runOneTest();
    }
    catch(std::exception& e) {
      cout << "Tiny model test exception: " << e.what() << endl;
      throw;
    }
  };

  vector<std::thread> testThreads;
  for(int i = 0; i<4; i++)
    testThreads.push_back(std::thread(runAFewTests));
  for(int i = 0; i<4; i++)
    testThreads[i].join();

  if(!FileUtils::tryRemoveFile(tmpModelFile)) {
    logger.write("Warning: could not delete " + tmpModelFile);
  }
  logger.write("Tiny net sanity check complete");

  return nnEval;
}

