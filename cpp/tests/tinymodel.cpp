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

NNEvaluator* TinyModelTest::runTinyModelTest(const string& baseDir, Logger& logger, ConfigParser& cfg, bool randFileName, double errorTolFactor) {
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

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale*errorTolFactor), buf, board, __FILE__, __LINE__)
      NNOutput& nnOutput = *(buf.result);
      EQ(nnOutput.whiteWinProb, .43298, 0.01);
      EQ(nnOutput.whiteLossProb, .56702, 0.01);
      EQ(nnOutput.whiteNoResultProb, .0000, 0.002);
      EQ(nnOutput.whiteScoreMean, -0.81, 0.2);
      EQ(nnOutput.whiteScoreMeanSq, 130.83, 1.0);
      EQ(nnOutput.whiteLead, -0.61, 0.2);
      EQ(nnOutput.varTimeLeft, 15.79, 1.0);
      EQ(nnOutput.shorttermWinlossError, .22408, 0.01);
      EQ(nnOutput.shorttermScoreError, 2.787, 0.2);

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
        0,    1,    2,    6,    4,    3,    3,    3,    2,    2,    3,    3,    4,    3,    5,   45,   78,    3,    0,
        0,    3,   40,   15,    8,   15,    7,    5,    3,    3,    4,    5,    6,   18,   31,  143,   98,    3,    0,
        0,   13,  152,  120,   30,   65,   21,   14,    4,    4,    5,   10,   13,   14,   22,   57,   46,    2,    0,
        0,   47,  270,   66,  723,   -1, 1055,   21,    6,    7,    6,   19,    2,   -1,   18,   -1,    3,    2,    0,
        0,   19,  489,   -1,   19,   -1,  480,  185,   19,   10,   11,   17,    6,    7,   36,    6,    8,    2,    0,
        0,    0,   -1,   -1,   40,    0,   10,   89,   69,   50,   40,   78,    1,   -1,   12,   -1,    5,    2,    0,
        0,    0,   -1,   -1,   -1,   -1,   -1,   89,  103,   19,   30,   85,    2,   -1,   -1,    1,   23,    1,    0,
        0,    1,    0,    4,   -1,   -1,  465,  195,   13,    1,    2,   17,   11,  287,  435,   33,    4,    0,    0,
        0,    0,    0,    0,    0,    0,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
      };
      for(int pos = 0; pos<361; pos++) {
        double prob = nnOutput.policyProbs[pos];
        if(expectedPolicy[pos] >= 0) {
          EQ(prob*10000, expectedPolicy[pos], std::min(60.0, expectedPolicy[pos] * 0.1 + 2.0) + std::min(10.0, expectedPolicy[pos] * 0.1));
        }
      }

      double expectedOwnership[361] = {
        -6334, -6907, -6959, -5781, -3638,  -725,  1417,  2059,  2398,  2731,  2500,  1762,  1534,  1847,  4036,  5790,  6030,  6531,  6013,
        -6958, -8377, -7921, -7230, -3075,  1524,  1701,  2351,  2644,  3310,  2634,  2000,   710,  1504,  3975,  6828,  7159,  5613,  4397,
        -6958, -8036, -9374, -9112,     0,  1509,  1903,  1429,  2007,  2206,  7262,   373, -4882,   144,  3418,  8908,  8632,  3278,  1820,
        -6696, -7519, -9240, -3753, -1589,  5671,  1389,  1334,   823,  1964,  1751,  1137,    -8,   195,  4565,  8705, -7779,  -947, -1347,
        -6070, -6786, -6344, -3786,   -36,  1656,  1828,   691,   453,   558,   968,   255,   306,  1496,  4481,   436, -8291, -4664, -3208,
        -5526, -5732, -4611, -8527, -1152,   618,    91,   292,     0,   194,   169,   317,  1033,  1735,  8053,  8247, -8541, -5412, -4289,
        -4595, -4552, -3513, -2337, -1107,     0,    13,     0,     0,     0,     0,    35,   398,  1508,   172, -6878, -3004, -4052, -4193,
        -4128, -3723, -2405,  -667,     0,     0,     0,     0,     0,     0,     0,     0,   118,    88,     0,  -175, -1578, -2595, -3181,
        -3849, -3636, -2894, -1044,     0,     0,     0,     0,     0,     0,     0,     0,     0,    55,     0,     0,     0, -1186, -1558,
        -3322, -3416, -8571, -1753,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  -123,  -734,
        -2816, -2062, -1969,  -414,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    30,    44,  -386,
        -2256, -1004,     0,     0,     0,  1102,   308,   182,     0,     0,     0,     0,   196,   507,     0,     0,     0,  -233,  -798,
        -1536,  -128,   252,  1149,  1824,  1407,  1389,   342,     0,     0,   295,   593,  1988,   927,     0, -1781, -1536, -1000, -1624,
        -1944,  -628,  1433,  1826,  2878,  8256,   399,   473,    39,     0,   744,  1431,  2020,  8273,  -976, -8493, -3679, -2664, -2885,
        -3255, -2108,     0,  8274,  2996,  -299,  1220,   351,   218,   200,  1100,  1656,  3989,  4856,     0, -2935, -4009, -3459, -3759,
        -4677, -5020, -8647,  8619,  5631,  3303,  2735,  1525,   114,    22,   963,  1884,  4524,  8485, -1602, -8646, -4368, -3988, -4217,
        -6036, -7076, -9063, -8586,  8344,  8467,  8040,  1608,  1053,   362,  1145,  1837,  4533,  8249, -7754, -4197, -4244, -3968, -4010,
        -6308, -7702, -7512, -6025, -8292, -7443,  1230,  2138,  1193,  1305,  1825,  2393,  3537,  3323,     0, -2771, -3212, -4068, -3169,
        -6356, -6400, -7044, -6989, -6758, -5093, -1935,  1007,  1367,  1482,  2366,  3039,  3606,  1452,  -952, -3103, -3532, -3474, -3515,
      };
      for(int pos = 0; pos<361; pos++) {
        double ownership = nnOutput.whiteOwnerMap[pos];
        EQ(ownership*10000, expectedOwnership[pos], 300.0);
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

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale*errorTolFactor), buf, board, __FILE__, __LINE__)
      NNOutput& nnOutput = *(buf.result);
      EQ(nnOutput.whiteWinProb, .4792, 0.02);
      EQ(nnOutput.whiteLossProb, .5208, 0.02);
      EQ(nnOutput.whiteNoResultProb, .0000, 0.004);
      EQ(nnOutput.whiteScoreMean, -0.7, 0.4);
      EQ(nnOutput.whiteScoreMeanSq, 212.0, 2.0);
      EQ(nnOutput.whiteLead, -0.8, 0.4);
      EQ(nnOutput.varTimeLeft, 15.2, 2.0);
      EQ(nnOutput.shorttermWinlossError, .531, 0.02);
      EQ(nnOutput.shorttermScoreError, 4.7, 0.5);

      double expectedPolicy[361] = {
        1,    5,    4,    4,    4,    5,    4,    4,    4,    5,    6,    8,    8,    6,    6,    6,    6,    7,    1,
        5,    8,    4,    9,   37,   40,   34,   19,   26,   51,   64,   56,   23,   43,   53,   20,   18,   31,    8,
        4,    4,   -1,   -1,   42,   51,   66,   37,   36,   56,   -1,   55,   -1,   35,   34,   -1,   -1,  468,    9,
        3,    5,   -1,   -1,   48,   -1,   48,   25,   24,   46,   69,   63,   20,   58,   30,   -1,   -1,   61,   14,
        3,   11,   20,   15,   10,   22,   46,   22,   19,   30,   37,   37,   51,   56,   36,  109,   -1,    3,   10,
        4,   13,    8,   -1,   12,   34,   28,   21,   22,   23,   23,   23,   37,   39,   -1,   -1,   -1,    6,    9,
        4,   19,   21,   11,   21,   24,   27,   24,   24,   24,   23,   22,   31,   60,  132,   -1,   66,   38,    6,
        4,   14,   22,   19,   20,   24,   25,   24,   24,   25,   24,   23,   28,   53,   59,   53,   70,   27,    5,
        4,   16,   12,   22,   23,   29,   24,   24,   24,   24,   24,   24,   26,   28,   34,   52,   48,   26,    4,
        5,    8,   -1,   12,   26,   28,   24,   24,   24,   24,   24,   25,   26,   26,   28,   33,   38,   23,    4,
        5,   21,   15,   25,   26,   26,   22,   23,   24,   24,   24,   24,   24,   27,   30,   34,   39,   22,    4,
        5,   27,   46,   30,   33,   36,   33,   25,   24,   24,   23,   22,   32,   41,   46,   42,   34,   21,    4,
        5,   35,   70,   64,   58,   96,   39,   36,   25,   24,   20,   24,   45,   62,   65,   30,   33,   19,    4,
        5,   55,  146,   75,  268,   -1,  237,   29,   24,   23,   18,   19,   28,   -1,   49,   -1,   11,   17,    4,
        5,   38,  231,   -1,   58,   -1,  113,   55,   24,   21,   15,   17,   50,   29,   60,    9,   14,   13,    4,
        6,    7,   -1,   -1,   30,   12,   19,   51,   37,   21,   16,   22,   11,   -1,   31,   -1,   11,   17,    4,
        7,    5,   -1,   -1,   -1,   -1,   -1,   82,   57,   31,   27,   37,   17,   -1,   -1,    5,   35,   24,    4,
        8,   22,    7,   19,   -1,   -1,  183,   76,   31,   19,   18,   24,   45,  201,  141,   41,   25,   17,    6,
        1,    8,    8,   11,    3,    4,    9,    7,    4,    4,    4,    4,    5,    6,    9,    6,    5,    6,    1,
      };
      for(int pos = 0; pos<361; pos++) {
        double prob = nnOutput.policyProbs[pos];
        if(expectedPolicy[pos] >= 0) {
          EQ(prob*10000, expectedPolicy[pos], std::min(60.0, expectedPolicy[pos] * 0.1 + 2.0) + std::min(10.0, expectedPolicy[pos] * 0.1));
        }
      }

      double expectedOwnership[361] = {
        -501, -2888, -3755, -2901,  -954,  1075,  3006,  3883,  4275,  3994,  3557,  2453,  1874,  2856,  4296,  5780,  5469,  4763,  4266,
        -3045, -4817, -6228, -5265, -2820,  -372,  2451,  4041,  4274,  3928,  3041,  1988,  1159,  2895,  5076,  6414,  5460,  4627,  3010,
        -4368, -6274, -9387, -9028, -5019, -2260,  2485,  3782,  4073,  2843,  7907,   998, -3606,  2684,  5600,  9267,  8878,  1265,  1563,
        -4164, -6033, -9145,  -317, -2883,  7145,  3774,  5101,  5030,  4386,  3377,  3269,  3540,  4305,  6354,  9318, -3791,   -91,  -268,
        -3437, -5049, -6552, -5881, -1992,   339,  4064,  4879,  4813,  3977,  3587,  3695,  4126,  4468,  5648,  3533, -7558, -3662,  -828,
        -2449, -4771, -5868, -7086, -2804,   563,  2897,  4133,  3932,  3546,  3432,  3802,  4060,  4340,  8685,  8675, -7863, -4266, -2502,
        -1251, -3786, -4977, -4039, -1708,   465,  2014,  3025,  2917,  2808,  3059,  3729,  3653,  3510,  2461, -5382, -4710, -4400, -2861,
        -919, -2860, -4408, -3429, -2264,  -379,  1443,  2275,  2334,  2350,  2583,  3076,  2830,  2148,   812, -1496, -3962, -3359, -1644,
        -911, -2901, -4602, -3174, -2043,  -436,  1505,  2081,  2168,  2185,  2316,  2551,  2347,  1735,   682,  -981, -2927, -1897,   211,
        -931, -2855, -6152, -2897, -1174,   223,  1983,  2293,  2263,  2227,  2318,  2456,  2382,  1890,   931,  -281, -1697,  -358,  1361,
        -347, -1600, -3014,  -843,   709,  1920,  3013,  2871,  2562,  2424,  2661,  3009,  3104,  2597,  1443,    33, -1453,  -170,  1469,
        835,   -48,  -522,  1293,  2478,  3184,  3501,  3241,  2929,  2781,  3293,  3903,  3719,  3012,  1511,  -297, -1836,  -927,   894,
        1892,   985,   600,  2859,  3889,  3548,  3836,  3376,  3306,  3281,  4123,  4837,  4818,  3518,  1683, -1837, -2954, -2066,   -81,
        1604,  1374,  1444,  4346,  3741,  8535,  3322,  4367,  3840,  3923,  4922,  5752,  5171,  8478,   586, -5400, -4177, -3232, -1024,
        138,  -265,   675,  8525,  6166,  -375,  4764,  4451,  4187,  4451,  5461,  6174,  6058,  5367,  1933, -3439, -4074, -3601, -1411,
        -1776, -3459, -6994,  8812,  6438,  7140,  6598,  5231,  4207,  4572,  5433,  6102,  6840,  9021,  1238, -6992, -4654, -4166, -1331,
        -3222, -5261, -8531, -7822,  8428,  8958,  8885,  4203,  3505,  3696,  4378,  5053,  5900,  9040, -4853, -4023, -4534, -4206, -1003,
        -3128, -4756, -5981, -5522, -7401, -5402,  2900,  3696,  3684,  4064,  4450,  4777,  4878,  2896,   903, -2068, -2982, -2291,   538,
        -1021, -3489, -4131, -3075, -2723,  -201,  3282,  4290,  4230,  4198,  4363,  4613,  4028,  3213,  1057,  -188, -1137,   457,  2373,
      };
      for(int pos = 0; pos<361; pos++) {
        double ownership = nnOutput.whiteOwnerMap[pos];
        EQ(ownership*10000, expectedOwnership[pos], 400.0);
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

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale*errorTolFactor), buf, board, __FILE__, __LINE__)
      NNOutput& nnOutput = *(buf.result);
      EQ(nnOutput.whiteWinProb, .3913, 0.02);
      EQ(nnOutput.whiteLossProb, .6087, 0.02);
      EQ(nnOutput.whiteNoResultProb, .0000, 0.004);
      EQ(nnOutput.whiteScoreMean, -0.5, 0.4);
      EQ(nnOutput.whiteScoreMeanSq, 347.9, 4.0);
      EQ(nnOutput.whiteLead, -0.9, 0.4);
      EQ(nnOutput.varTimeLeft, 10.0, 2.0);
      EQ(nnOutput.shorttermWinlossError, .516, 0.02);
      EQ(nnOutput.shorttermScoreError, 3.4, 0.5);

      double expectedPolicy[78] = {
        3,   24,   27,   34,   31,   24,   15,   15,   24,   14,   17,   17,    5,
        21,   45,   17,  159,   -1,  180,  160,  231,  141,   -1,   -1,   -1,   26,
        14,   12,   -1,   -1,  369,  156,  655,  841,  201,  148,   -1,   -1,   69,
        13,   15,   -1,   -1, 2189,   -1,  401,   -1,   -1,   -1,  262,   -1,  188,
        18,   41,   45,   -1,   -1,   74, 2108,   -1,   -1,   -1,   -1,   -1,   -1,
        2,   20,   52,  163,  241,   27,   38,  210,   -1,    7,   15,   23,   10,
      };
      for(int idx = 0; idx<78; idx++) {
        int pos = (idx % 13) + idx / 13 * NNPos::MAX_BOARD_LEN;
        double prob = nnOutput.policyProbs[pos];
        if(expectedPolicy[idx] >= 0) {
          EQ(prob*10000, expectedPolicy[idx], std::min(120.0, expectedPolicy[idx] * 0.1 + 2.0) + std::min(10.0, expectedPolicy[idx] * 0.1));
        }
      }

      double expectedOwnership[78] = {
        -921, -2937, -2842,    -9,  3148,  5068,  3930,   923, -2285, -3973, -5551, -5587, -3177,
        -3454, -4464, -4781, -3138,  7471,  3160,  2737, -1289, -5484, -8947, -9619, -9647, -6429,
        -4805, -5707, -9136, -9129, -3252,   334,  2304, -1068, -4957, -7621, -5213, -9697, -7298,
        -5103, -6423, -9482, -2733, -2219,  7638,  2321, -5834,  3880,   416, -9124, -6601, -7420,
        -3797, -5265, -6283, -8551,  7316,  1073,  2613,  1309, -9205, -9663, -9808, -9689, -9381,
        -983, -3626, -3655, -1982,  3086,  3018,  1692, -4079, -9009, -6395, -5868, -6171, -3983,
      };
      for(int idx = 0; idx<78; idx++) {
        int pos = (idx % 13) + idx / 13 * NNPos::MAX_BOARD_LEN;
        double ownership = nnOutput.whiteOwnerMap[pos];
        EQ(ownership*10000, expectedOwnership[idx], 400.0);
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

