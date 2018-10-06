
#ifdef _WIN32
 #define _RAND_IS_WINDOWS
#elif _WIN64
 #define _RAND_IS_WINDOWS
#elif __unix || __APPLE__
  #define _RAND_IS_UNIX
#else
 #error Unknown OS!
#endif

#ifdef _RAND_IS_WINDOWS
  #include <winsock.h>
#endif
#ifdef _RAND_IS_UNIX
  #include <unistd.h>
#endif

#include <ctime>
#include <cstdlib>
#include <sstream>
#include <atomic>
#include "../core/timer.h"
#include "../core/global.h"
#include "../core/hash.h"
#include "../core/sha2.h"
#include "../core/rand.h"
#include "../core/test.h"

XorShift1024Mult::XorShift1024Mult(const uint64_t* init_a)
{
  init(init_a);
}

void XorShift1024Mult::init(const uint64_t* init_a)
{
  a_idx = 0;
  for(int i = 0; i<XORMULT_LEN; i++)
    a[i] = init_a[i];
}

void XorShift1024Mult::test()
{
  const uint64_t init_a[XORMULT_LEN] = {
    15148282349006049087ULL,
    3601266951833665894ULL,
    16929445066801446424ULL,
    13475938501103070154ULL,
    15713138009143754412ULL,
    4148159782736716337ULL,
    16035594834001032141ULL,
    5555591070439871209ULL,
    4101130512537511022ULL,
    12821547636792886909ULL,
    9050874162294428797ULL,
    6187760405891629771ULL,
    10053646276519763308ULL,
    2219782655280501359ULL,
    3719698449347562208ULL,
    5421263376768154227ULL
  };

  XorShift1024Mult xorm(init_a);

  const uint32_t expected[32] = {
    0x749746d1u,
    0x9242ca14u,
    0x98db98a1u,
    0x1348e491u,
    0xde60e668u,
    0x77e37a69u,
    0xeb51a9d3u,
    0xd44b4727u,
    0x341895b0u,
    0xc7b1b3f4u,
    0xe7ef0529u,
    0x8e72ea7eu,
    0x5855da19u,
    0xfffcd2b2u,
    0xa684e430u,
    0xb76a7e0du,
    0x5af3820eu,
    0x320b0699u,
    0xdbb85ee0u,
    0xc1dcd25cu,
    0x4b395e3eu,
    0x4007756fu,
    0x76a0c667u,
    0xaa6041f6u,
    0x756f94bbu,
    0x39527d1bu,
    0x6e1232efu,
    0xb3027668u,
    0x776ea832u,
    0x35a0ed1bu,
    0x1f2f0268u,
    0xadd59669u,
  };

  for(int i = 0; i<32; i++) {
    uint32_t r = xorm.nextUInt();
    if(r != expected[i]) {
      cout << i << endl;
      cout << Global::uint32ToHexString(r) << endl;
      cout << Global::uint32ToHexString(expected[i]) << endl;
      Global::fatalError("XorShift1024Mult generated unexpected values");
    }
  }
}

//-----------------------------------------------------------------------------

PCG32::PCG32(uint64_t state)
{
  init(state);
}

void PCG32::init(uint64_t state)
{
  s = state;
}

void PCG32::test()
{
  PCG32 pcg(123);

  const uint32_t expected[16] = {
    0xb3766cbdu,
    0x65fdd305u,
    0x2a3b9b9cu,
    0x09a2dee9u,
    0x1a86aabcu,
    0x36a98234u,
    0x82e6e2b4u,
    0x10c077e5u,
    0x29755fc7u,
    0xf7fa7b5cu,
    0x1cb7ae7du,
    0xcce0e3d9u,
    0x065ec08bu,
    0x505d1cdbu,
    0x8b778f3cu,
    0xdb72f217u,
  };

  for(int i = 0; i<16; i++) {
    uint32_t r = pcg.nextUInt();
    if(r != expected[i]) {
      cout << i << endl;
      cout << Global::uint32ToHexString(r) << endl;
      cout << Global::uint32ToHexString(expected[i]) << endl;
      Global::fatalError("PCG32 generated unexpected values");
    }
  }
}

//---------------------------------------------------------------------------------

static const uint64_t zeros[XorShift1024Mult::XORMULT_LEN] = {
  0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,0ULL,
};


Rand::Rand()
  :xorm(zeros),pcg32(0ULL) //Dummy values, overridden by init
{
  init();
}

Rand::Rand(const char* seed)
  :xorm(zeros),pcg32(0ULL) //Dummy values, overridden by init
{
  init(seed);
}
Rand::Rand(const string& seed)
  :xorm(zeros),pcg32(0ULL) //Dummy values, overridden by init
{
  init(seed);
}
Rand::Rand(uint64_t seed)
  :xorm(zeros),pcg32(0ULL) //Dummy values, overridden by init
{
  init(seed);
}

Rand::~Rand()
{

}

static atomic<int> inits(0);

void Rand::init()
{
  int x = inits++;
  uint64_t time0 = (uint64_t)time(NULL);
  uint32_t clock0 = (uint32_t)clock();
  int64_t precisionTime = ClockTimer::getPrecisionSystemTime();

  string s =
    Global::intToString(x) +
    Global::uint64ToHexString(time0) +
    Global::uint32ToHexString(clock0) +
    Global::int64ToString(precisionTime);

  //Mix the hostname into the seed so that starting two things on different computers almost certainly
  //pick different seeds.
  //It turns out in this one case that the windows and unix implementations are the same...
#ifdef _RAND_IS_WINDOWS
  {
    s += "|";
    int bufSize = 1024;
    char hostNameBuf[bufSize];
    int result = gethostname(hostNameBuf,bufSize);
    if(result == 0)
      s += string(hostNameBuf);
  }
#endif
#ifdef _RAND_IS_UNIX
  {
    s += "|";
    int bufSize = 1024;
    char hostNameBuf[bufSize];
    int result = gethostname(hostNameBuf,bufSize);
    if(result == 0)
      s += string(hostNameBuf);
  }
#endif

  uint64_t hash[4];
  SHA2::get256(s.c_str(), hash);

  init(Global::uint64ToHexString(hash[0]));
}

void Rand::init(uint64_t seed)
{
  init(Global::uint64ToHexString(seed));
}

void Rand::init(const char* seed)
{
  init(string(seed));
}

void Rand::init(const string& seed)
{
  initSeed = seed;

  string s = seed;
  int x = 0;

  auto getNonzero = [&s,&x]() -> uint64_t {
    uint64_t hash[4];
    do {
      string tmp = s + Global::intToString(x);
      x += 1;
      SHA2::get256(tmp.c_str(), hash);
    } while(hash[0] == 0);
    return hash[0];
  };

  uint64_t init_a[XorShift1024Mult::XORMULT_LEN];
  for(int i = 0; i<XorShift1024Mult::XORMULT_LEN; i++)
    init_a[i] = getNonzero();

  xorm.init(init_a);
  pcg32.init(getNonzero());

  hasGaussian = false;
  storedGaussian = 0.0;
  numCalls = 0;
}

class RandToURNGWrapper {
public:
  typedef size_t result_type;
  Rand* rand;
  RandToURNGWrapper(Rand* r)
    :rand(r)
  {}
  ~RandToURNGWrapper()
  {}

  static size_t min() { return 0; }
  static size_t max() { return (size_t)0xFFFFFFFF; }
  size_t operator()() {
    return (size_t)rand->nextUInt();
  }
};

#include <random>
double Rand::nextGamma(double a) {
  std::gamma_distribution<double> distribution(a,1.0);
  RandToURNGWrapper wrapped(this);
  return distribution(wrapped);
}

static void simpleTest()
{
  Rand rand("abc");

  const uint32_t expected[24] = {
    0x8d25293fu,
    0xe87be2d9u,
    0xc5424597u,
    0xe0608f99u,
    0x55dd51f1u,
    0x8bec9862u,
    0xf2aaa3bcu,
    0x077e767du,
    0xa9e3208eu,
    0x68bf343du,
    0xd8f91fa9u,
    0xe703df24u,
    0xd382c1feu,
    0xbcfcb106u,
    0x10fc902eu,
    0x4e88ff31u,
    0x8f003fd8u,
    0xdebeb950u,
    0x73a832cfu,
    0x8b04a3edu,
    0x5cb5a5c9u,
    0x36fd9d1fu,
    0xa5d6e74fu,
    0x274e20a5u,
  };

  for(int i = 0; i<24; i++) {
    uint32_t r = rand.nextUInt();
    if(r != expected[i]) {
      cout << i << endl;
      cout << Global::uint32ToHexString(r) << endl;
      cout << Global::uint32ToHexString(expected[i]) << endl;
      Global::fatalError("Rand generated unexpected values");
    }
  }

  char hash[65];
  SHA2::get256("The quick brown fox jumps over the lazy dog.", hash);
  if(string(hash) != string("ef537f25c895bfa782526529a9b63d97aa631564d5d789c2b765448c8635fb6c")) {
    cout << hash << endl;
    cout << "SHA2 generated unexpected hash" << endl;
  }
}


void Rand::runTests() {
  cout << "Running rng and hash tests" << endl;
  simpleTest();

  ostringstream out;

  {
    const char* name = "Rand tests";
    Rand rand("abc");

    out << "rand.nextUInt()" << endl;
    for(int i = 0; i<16; i++) out << rand.nextUInt() << endl;

    out << "rand.nextUInt(27)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextUInt(27) << endl;

    out << "rand.nextInt()" << endl;
    for(int i = 0; i<16; i++) out << rand.nextInt() << endl;

    out << "rand.nextInt(-8,8)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextInt(-8,8) << endl;

    out << "rand.nextUInt64()" << endl;
    for(int i = 0; i<16; i++) out << rand.nextUInt64() << endl;

    out << "rand.nextUInt64(0xFFffffFFFFULL)" << endl;
    for(int i = 0; i<16; i++) out << Global::uint64ToHexString(rand.nextUInt64(0xFFffffFFFFULL)) << endl;

    out << "rand.nextDouble()" << endl;
    for(int i = 0; i<16; i++) out << rand.nextDouble() << endl;

    out << "rand.nextDouble(12)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextDouble(12) << endl;

    out << "rand.nextDouble(-100,100)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextDouble(-100,100) << endl;

    out << "rand.nextGaussian()" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGaussian() << endl;

    out << "rand.nextLogistic()" << endl;
    for(int i = 0; i<16; i++) out << rand.nextLogistic() << endl;

    out << "rand.nextGamma(1)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(1) << endl;

    out << "rand.nextGamma(0.1)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(0.1) << endl;

    out << "rand.nextGamma(4)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(4) << endl;

    string expected = R"%%(
rand.nextUInt()
2368022847
3900433113
3309454743
3764424601
1440567793
2347538530
4071269308
125728381
2850234510
1757361213
3640205225
3875790628
3548561918
3170676998
284987438
1317601073
rand.nextUInt(27)
8
14
24
21
24
9
20
5
4
25
19
19
22
21
9
12
rand.nextInt()
2032136641
-1484690267
-1162496884
-740885019
1341706441
277308650
546758271
1569362505
1722893617
-280154620
-1420292415
-570959358
631721904
-1703885216
2121986197
1409159894
rand.nextInt(-8,8)
8
1
6
0
0
-5
-2
8
-8
8
-8
0
4
-8
-6
-7
rand.nextUInt64()
14418078273872605708
15656272520058952416
1815003353704009443
16763916499353172813
12487021701330837924
3078632268415805218
5899695363489918344
2316637589152137835
3936601489392549938
3172688217440043471
8729586732234677986
4188965795452063568
10742528258072691946
11328279065428005175
9328994114736784186
16748335411137583871
rand.nextUInt64(0xFFffffFFFFULL)
000000A3B324F34A
0000004458B3422E
000000E1C37C2662
00000098B80D8319
0000004FC0DC3BCF
000000A13B506E1D
000000C71DAF87D8
00000034A1026F53
000000450CAF1770
000000CE1FD7E402
000000318460E228
0000009FBECC5B03
000000602629752A
0000005C170F3F7E
000000924AE74CD1
0000000DFEDC1D82
rand.nextDouble()
0.080862
0.547609
0.796694
0.577365
0.997252
0.993346
0.997749
0.881844
0.186606
0.930101
0.500096
0.258518
0.987045
0.384286
0.343331
0.501281
rand.nextDouble(12)
8.04997
9.68375
6.79008
5.47923
2.05584
3.74097
9.23618
3.24588
0.506804
6.11743
2.44672
6.41922
3.01077
7.08368
10.1102
3.49022
rand.nextDouble(-100,100)
24.5489
-91.3789
-30.2139
4.20313
-4.4925
4.33685
82.1587
-23.792
26.9889
-1.93572
60.4881
-54.8498
37.7642
77.027
68.3438
-11.1699
rand.nextGaussian()
-1.04102
1.02086
-2.21387
1.18239
0.464025
0.201022
0.924891
-1.77215
0.783248
0.43083
1.22297
-0.779088
0.437384
-0.712373
0.921093
-0.690558
rand.nextLogistic()
-0.949963
-0.309666
1.25673
0.0234923
-4.13501
1.48687
-2.68628
-0.197865
-2.87159
-4.15327
4.69788
0.164818
1.87224
-0.208472
-1.73322
-0.607461
rand.nextGamma(1)
0.274704
0.0744126
0.471163
0.0774803
0.963182
0.165412
0.213661
0.730311
2.24201
0.468865
0.0571607
0.905897
2.60238
0.64582
0.158352
0.685324
rand.nextGamma(0.1)
0.000316875
3.15072e-09
1.08349e-06
0.108733
0.00341049
2.32469e-11
4.49661e-07
0.0029474
0.669966
1.00563e-08
0.64905
7.22175e-13
1.2857e-07
0.0177569
1.85445e-09
3.17753e-07
rand.nextGamma(4)
1.57075
4.32638
1.22375
2.30653
2.28736
5.2053
3.4774
3.89106
3.4363
1.49415
3.48808
6.74786
3.68859
1.31681
2.39111
2.90081
)%%";
    TestCommon::expect(name,out,expected);
    out.str("");
    out.clear();
  }

  {
    const char* name = "Rand moment tests";
    
    int bufLen = 100000;
    double* buf = new double[bufLen];

    auto printMoments = [&out,&buf,bufLen](const string& distrName, double expectedMean, double expectedVariance, double expectedSkew, double expectedExcessKurt) {
      double m1 = 0;
      double m2 = 0;
      double m3 = 0;
      double m4 = 0;
      for(int i = 0; i<bufLen; i++) {
        double x = buf[i];
        m1 += x;
        m2 += x*x;
        m3 += x*x*x;
        m4 += x*x*x*x;
      }
      m1 /= bufLen;
      m2 /= bufLen;
      m3 /= bufLen;
      m4 /= bufLen;

      double mean = m1;
      double variance = m2 - m1*m1;
      double skew = (m3 - 3*m1*variance - m1*m1*m1) / sqrt(variance*variance*variance);
      double excessKurt = (m4 - 4*m3*m1 + 6*m2*m1*m1 - 3*m1*m1*m1*m1)/(variance*variance) - 3;
      out << Global::strprintf(
        "%s sample: Mean %f Variance %f Skew %f ExcessKurt %f",
        distrName.c_str(), mean, variance, skew, excessKurt
      ) << endl;
      out << Global::strprintf(
        "%s expected: Mean %f Variance %f Skew %f ExcessKurt %f",
        distrName.c_str(), expectedMean, expectedVariance, expectedSkew, expectedExcessKurt
      ) << endl;

    };

    Rand rand("test");
    
    for(int i = 0; i<bufLen; i++)
      buf[i] = rand.nextDouble();
    printMoments("Uniform", 0.5, 1.0/12.0, 0, -6.0/5.0);

    for(int i = 0; i<bufLen; i++)
      buf[i] = rand.nextGaussian();
    printMoments("Gaussian", 0, 1.0, 0, 0.0);

    double pi = 3.14159265358979323846264;
    for(int i = 0; i<bufLen; i++)
      buf[i] = rand.nextLogistic();
    printMoments("Logistic", 0, pi*pi/3.0, 0, 1.2);

    for(int i = 0; i<bufLen; i++)
      buf[i] = rand.nextGamma(0.5);
    printMoments("Gamma", 0.5, 0.5, 2.0/sqrt(0.5), 6.0/0.5);

    for(int i = 0; i<bufLen; i++)
      buf[i] = rand.nextGamma(4.0);
    printMoments("Gamma", 4.0, 4.0, 2.0/sqrt(4.0), 6.0/4.0);
    
    delete[] buf;
    string expected = R"%%(
Uniform sample: Mean 0.499714 Variance 0.083704 Skew 0.003446 ExcessKurt -1.202311
Uniform expected: Mean 0.500000 Variance 0.083333 Skew 0.000000 ExcessKurt -1.200000
Gaussian sample: Mean -0.000782 Variance 1.001046 Skew 0.015581 ExcessKurt 0.012558
Gaussian expected: Mean 0.000000 Variance 1.000000 Skew 0.000000 ExcessKurt 0.000000
Logistic sample: Mean 0.003819 Variance 3.283142 Skew 0.003153 ExcessKurt 1.149271
Logistic expected: Mean 0.000000 Variance 3.289868 Skew 0.000000 ExcessKurt 1.200000
Gamma sample: Mean 0.499243 Variance 0.497366 Skew 2.832579 ExcessKurt 12.005515
Gamma expected: Mean 0.500000 Variance 0.500000 Skew 2.828427 ExcessKurt 12.000000
Gamma sample: Mean 3.998701 Variance 3.992548 Skew 0.986482 ExcessKurt 1.410314
Gamma expected: Mean 4.000000 Variance 4.000000 Skew 1.000000 ExcessKurt 1.500000
)%%";

    TestCommon::expect(name,out,expected);
    out.str("");
    out.clear();
  }
  
}

