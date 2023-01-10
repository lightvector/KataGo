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

static void requireApproxEqual(double x, double expected, double scale, const NNResultBuf& buf, const Board& board, const char *file, int line) {
  if(!std::isfinite(x) || !std::isfinite(expected) || std::fabs(x-expected) > scale) {
    buf.result->debugPrint(cout,board);
    throw StringError(
      "Tiny neural net test got invalid values - is the GPU working? " +
      string(file) + " line " + Global::intToString(line) + " " + Global::doubleToString(x) + " " + Global::doubleToString(expected)
    );
  }
}

NNEvaluator* TinyModelTest::runTinyModelTest(const string& baseDir, Logger& logger, ConfigParser& cfg, bool randFileName) {
  logger.write("Running tiny net to sanity-check that GPU is working");

  NNEvaluator* nnEvalRet;
  {
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

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale), buf, board, __FILE__, __LINE__)
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

    // We return this one
    nnEvalRet = nnEval;
  }

  {
    string base64Data;
    base64Data += TinyModelTest::tinyMishModelBase64;
    string binaryData;
    decodeBase64(base64Data, binaryData);

    Rand rand;
    const string tmpModelFile =
      randFileName ?
      (baseDir + "/" + "tmpTinyMishModel_" + Global::uint64ToHexString(rand.nextUInt64()) + ".bin.gz") :
      (baseDir + "/" + "tmpTinyMishModel.bin.gz");
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
    nnEval->setDefaultSymmetry(7);

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

      // ostringstream out;
      // buf.result->debugPrint(out,board);
      // cout << out.str() << endl;

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale), buf, board, __FILE__, __LINE__)
      NNOutput& nnOutput = *(buf.result);
      EQ(nnOutput.whiteWinProb, .4792, 0.02);
      EQ(nnOutput.whiteLossProb, .5208, 0.02);
      EQ(nnOutput.whiteNoResultProb, .0000, 0.004);
      EQ(nnOutput.whiteScoreMean, -0.7, 0.4);
      EQ(nnOutput.whiteScoreMeanSq, 212.0, 2.0);
      EQ(nnOutput.whiteLead, -0.8, 0.4);
      EQ(nnOutput.varTimeLeft, 15.2, 1.0);
      EQ(nnOutput.shorttermWinlossError, .531, 0.02);
      EQ(nnOutput.shorttermScoreError, 4.7, 0.4);

      double expectedPolicy[361] = {
        1,    5,    4,    4,    4,    5,    4,    4,    4,    5,    6,    8,    8,    6,    6,    6,    6,    7,    1,
        5,    8,    4,    8,   37,   40,   34,   19,   25,   52,   64,   56,   22,   42,   53,   20,   18,   31,    8,
        4,    4,   -1,   -1,   41,   51,   66,   37,   37,   56,   -1,   54,   -1,   35,   34,   -1,   -1,  467,    9,
        3,    5,   -1,   -1,   48,   -1,   48,   25,   24,   46,   69,   64,   20,   59,   30,   -1,   -1,   60,   14,
        3,   11,   20,   15,   10,   22,   47,   22,   19,   30,   37,   37,   51,   56,   36,  109,   -1,    3,   10,
        4,   12,    8,   -1,   12,   34,   28,   21,   22,   23,   23,   23,   37,   39,   -1,   -1,   -1,    6,    9,
        4,   19,   21,   11,   21,   24,   27,   24,   24,   24,   23,   22,   31,   60,  132,   -1,   66,   37,    6,
        4,   14,   22,   19,   19,   24,   25,   24,   24,   25,   24,   23,   28,   54,   59,   54,   70,   27,    5,
        4,   16,   12,   22,   22,   29,   24,   24,   24,   24,   24,   24,   26,   28,   34,   52,   47,   25,    4,
        5,    8,   -1,   12,   26,   28,   24,   24,   24,   24,   24,   25,   26,   26,   28,   33,   38,   23,    4,
        5,   21,   15,   25,   25,   26,   22,   23,   24,   24,   24,   25,   24,   27,   30,   34,   39,   22,    4,
        5,   27,   47,   30,   33,   36,   33,   25,   24,   24,   23,   22,   32,   41,   46,   42,   34,   21,    4,
        5,   35,   70,   64,   58,   96,   39,   36,   25,   24,   20,   24,   45,   62,   65,   30,   32,   19,    4,
        5,   55,  146,   76,  268,   -1,  236,   29,   24,   23,   18,   19,   28,   -1,   49,   -1,   12,   16,    4,
        5,   38,  230,   -1,   58,   -1,  114,   55,   24,   21,   15,   17,   50,   29,   60,    9,   13,   13,    4,
        6,    7,   -1,   -1,   30,   12,   19,   51,   37,   21,   16,   22,   11,   -1,   31,   -1,   11,   17,    4,
        6,    4,   -1,   -1,   -1,   -1,   -1,   81,   57,   31,   27,   37,   17,   -1,   -1,    5,   34,   23,    4,
        8,   22,    7,   19,   -1,   -1,  185,   76,   31,   19,   18,   24,   45,  202,  141,   41,   24,   17,    6,
        1,    8,    8,   11,    3,    4,    9,    7,    4,    4,    4,    4,    5,    6,    9,    6,    5,    6,    1,
      };
      for(int pos = 0; pos<361; pos++) {
        double prob = nnOutput.policyProbs[pos];
        if(expectedPolicy[pos] >= 0) {
          EQ(prob*10000, expectedPolicy[pos], std::min(40.0, expectedPolicy[pos] * 0.1 + 1.0));
        }
      }

      double expectedOwnership[361] = {
        -504, -2858, -3717, -2882,  -937,  1088,  3018,  3874,  4277,  3999,  3562,  2456,  1886,  2882,  4287,  5781,  5471,  4781,  4283,
        -3032, -4807, -6195, -5256, -2810,  -358,  2463,  4055,  4297,  3932,  3082,  2013,  1174,  2923,  5080,  6432,  5475,  4629,  3018,
        -4336, -6263, -9391, -9039, -4993, -2251,  2516,  3776,  4073,  2849,  7914,  1050, -3592,  2720,  5610,  9267,  8881,  1304,  1563,
        -4130, -6029, -9141,  -310, -2871,  7148,  3767,  5102,  5040,  4394,  3401,  3262,  3541,  4299,  6363,  9317, -3822,   -85,  -221,
        -3416, -5048, -6531, -5878, -1973,   363,  4077,  4886,  4815,  3977,  3584,  3711,  4122,  4476,  5660,  3560, -7556, -3626,  -777,
        -2417, -4751, -5865, -7076, -2788,   570,  2878,  4114,  3925,  3549,  3425,  3803,  4053,  4324,  8686,  8681, -7855, -4249, -2488,
        -1230, -3763, -4971, -4016, -1708,   498,  2054,  3040,  2898,  2808,  3056,  3732,  3649,  3528,  2470, -5399, -4690, -4348, -2824,
        -914, -2844, -4374, -3410, -2242,  -347,  1432,  2272,  2312,  2356,  2602,  3069,  2846,  2178,   811, -1469, -3936, -3339, -1647,
        -916, -2889, -4579, -3175, -2032,  -422,  1517,  2087,  2187,  2206,  2333,  2550,  2359,  1755,   671,  -957, -2914, -1860,   198,
        -938, -2871, -6158, -2905, -1171,   226,  1980,  2294,  2272,  2211,  2312,  2458,  2378,  1900,   966,  -281, -1666,  -354,  1371,
        -333, -1552, -2996,  -837,   735,  1925,  3012,  2880,  2584,  2437,  2655,  3012,  3098,  2623,  1466,    43, -1452,  -161,  1488,
        838,   -53,  -536,  1284,  2474,  3197,  3502,  3225,  2914,  2768,  3267,  3903,  3717,  3029,  1521,  -285, -1843,  -924,   880,
        1886,  1028,   611,  2851,  3923,  3571,  3842,  3373,  3310,  3265,  4110,  4841,  4815,  3524,  1696, -1834, -2963, -2027,   -72,
        1601,  1384,  1444,  4346,  3763,  8537,  3332,  4364,  3845,  3925,  4908,  5752,  5185,  8477,   606, -5406, -4172, -3214, -1019,
        147,  -251,   697,  8523,  6164,  -362,  4770,  4456,  4196,  4454,  5457,  6176,  6067,  5389,  1969, -3425, -4053, -3575, -1425,
        -1755, -3440, -7007,  8807,  6435,  7124,  6598,  5238,  4205,  4571,  5437,  6100,  6842,  9025,  1268, -7004, -4633, -4130, -1337,
        -3217, -5238, -8542, -7821,  8438,  8955,  8881,  4227,  3511,  3715,  4386,  5062,  5904,  9041, -4878, -4002, -4519, -4180,  -998,
        -3118, -4743, -5967, -5505, -7403, -5399,  2896,  3736,  3700,  4071,  4456,  4785,  4893,  2911,   910, -2049, -2965, -2259,   558,
        -1020, -3475, -4110, -3074, -2727,  -202,  3265,  4305,  4249,  4198,  4376,  4625,  4061,  3241,  1078,  -175, -1142,   492,  2393,
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

    //We delete this one
    delete nnEval;
  }

  {
    string base64Data;
    base64Data += TinyModelTest::tinyMishModelBase64;
    string binaryData;
    decodeBase64(base64Data, binaryData);

    Rand rand;
    const string tmpModelFile =
      randFileName ?
      (baseDir + "/" + "tmpTinyMishModel_" + Global::uint64ToHexString(rand.nextUInt64()) + ".bin.gz") :
      (baseDir + "/" + "tmpTinyMishModel.bin.gz");
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
    nnEval->setDefaultSymmetry(1);

    Board board = Board::parseBoard(13,6,R"%%(
.............
....o....xxx.
..xx......ox.
..xo.o.xoo.o.
...xo..oxxxxx
........x....
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

      // ostringstream out;
      // buf.result->debugPrint(out,board);
      // cout << out.str() << endl;

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale), buf, board, __FILE__, __LINE__)
      NNOutput& nnOutput = *(buf.result);
      EQ(nnOutput.whiteWinProb, .3913, 0.02);
      EQ(nnOutput.whiteLossProb, .6087, 0.02);
      EQ(nnOutput.whiteNoResultProb, .0000, 0.004);
      EQ(nnOutput.whiteScoreMean, -0.5, 0.4);
      EQ(nnOutput.whiteScoreMeanSq, 347.9, 4.0);
      EQ(nnOutput.whiteLead, -0.9, 0.4);
      EQ(nnOutput.varTimeLeft, 10.0, 1.0);
      EQ(nnOutput.shorttermWinlossError, .516, 0.02);
      EQ(nnOutput.shorttermScoreError, 3.4, 0.4);

      double expectedPolicy[78] = {
        3,   24,   26,   34,   31,   24,   14,   15,   24,   14,   17,   17,    5,
        21,   45,   16,  158,   -1,  180,  157,  231,  139,   -1,   -1,   -1,   26,
        14,   12,   -1,   -1,  366,  155,  658,  838,  200,  146,   -1,   -1,   68,
        13,   14,   -1,   -1, 2203,   -1,  401,   -1,   -1,   -1,  261,   -1,  188,
        18,   41,   45,   -1,   -1,   74, 2120,   -1,   -1,   -1,   -1,   -1,   -1,
        2,   19,   52,  163,  241,   27,   37,  208,   -1,    7,   15,   23,   10,
      };
      for(int idx = 0; idx<78; idx++) {
        int pos = (idx % 13) + idx / 13 * NNPos::MAX_BOARD_LEN;
        double prob = nnOutput.policyProbs[pos];
        if(expectedPolicy[idx] >= 0) {
          EQ(prob*10000, expectedPolicy[idx], std::min(40.0, expectedPolicy[idx] * 0.1 + 1.0));
        }
      }

      double expectedOwnership[78] = {
        -938, -2902, -2819,    -5,  3151,  5084,  3942,   945, -2246, -3962, -5563, -5590, -3173,
        -3442, -4439, -4766, -3102,  7477,  3195,  2761, -1298, -5488, -8944, -9622, -9650, -6429,
        -4796, -5690, -9138, -9129, -3273,   343,  2319, -1063, -4934, -7612, -5206, -9699, -7295,
        -5084, -6429, -9485, -2736, -2237,  7636,  2328, -5824,  3863,   414, -9128, -6615, -7423,
        -3776, -5245, -6296, -8555,  7304,  1059,  2602,  1281, -9204, -9666, -9811, -9693, -9382,
        -971, -3594, -3639, -1953,  3102,  3018,  1695, -4079, -9005, -6389, -5881, -6173, -3985,
      };
      for(int idx = 0; idx<78; idx++) {
        int pos = (idx % 13) + idx / 13 * NNPos::MAX_BOARD_LEN;
        double ownership = nnOutput.whiteOwnerMap[pos];
        EQ(ownership*10000, expectedOwnership[idx], 200.0);
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

    //We delete this one
    delete nnEval;
  }

  logger.write("Tiny net sanity check complete");

  return nnEvalRet;
}

