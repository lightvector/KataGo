#include "../core/rand.h"
#include "../core/os.h"

#ifdef OS_IS_WINDOWS
  #include <winsock.h>
#endif
#ifdef OS_IS_UNIX_OR_APPLE
  #include <unistd.h>
#endif

#include <atomic>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <thread>

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/md5.h"
#include "../core/sha2.h"
#include "../core/test.h"
#include "../core/timer.h"
#include "../core/bsearch.h"

using namespace std;

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
  //Assemble entropy sources

  //Atomic incrementing counter, within this run of this program
  int x = inits++;

  //Various clocks
  uint64_t time0 = (uint64_t)time(NULL);
  uint32_t clock0 = (uint32_t)clock();
  int64_t precisionTime = ClockTimer::getPrecisionSystemTime();

  string s =
    Global::intToString(x) +
    Global::uint64ToHexString(time0) +
    Global::uint32ToHexString(clock0) +
    Global::int64ToString(precisionTime);

  //Mix the hostname and pid into the seed so that starting two things on different computers almost certainly
  //pick different seeds.
#ifdef OS_IS_WINDOWS
  {
    s += "|";
    DWORD processId = GetCurrentProcessId();
    s += Global::int64ToString((int64_t)processId);
    s += "|";
    static const int bufSize = 1024;
    char hostNameBuf[bufSize];
    int result = gethostname(hostNameBuf,bufSize);
    if(result == 0)
      s += string(hostNameBuf);
  }
#endif
#ifdef OS_IS_UNIX_OR_APPLE
  {
    s += "|";
    pid_t processId = getpid();
    s += Global::int64ToString((int64_t)processId);
    s += "|";
    static const int bufSize = 1024;
    char hostNameBuf[bufSize];
    int result = gethostname(hostNameBuf,bufSize);
    if(result == 0)
      s += string(hostNameBuf);
  }
#endif

  //Mix thread id.
  {
    std::hash<std::thread::id> hashOfThread;
    size_t hash = hashOfThread(std::this_thread::get_id());
    s += "|";
    s += Global::uint64ToHexString((uint64_t)hash);
  }

  //Mix address of stack and heap
  {
    int stackVal = 0;
    int* heapVal = new int[1];
    size_t stackAddr = (size_t)(&stackVal);
    size_t heapAddr = (size_t)(heapVal);
    delete[] heapVal;
    s += "|";
    s += Global::uint64ToHexString((uint64_t)stackAddr);
    s += Global::uint64ToHexString((uint64_t)heapAddr);
  }

  //cout << s << endl;

  char hash[65];
  SHA2::get256(s.c_str(), hash);
  assert(hash[64] == '\0');
  string hashed(hash);
  init(hashed);
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

  string s;
  {
    uint32_t hash[4];
    MD5::get(seed.c_str(), seed.size(), hash);
    s += "|";
    s += std::to_string(hash[0]);
    s += "|";
    s += seed;
  }

  int counter = 0;
  int nextHashIdx = 4;
  uint64_t hash[4];
  auto getNonzero = [&s,&counter,&nextHashIdx,&hash]() -> uint64_t {
    uint64_t nextValue;
    do {
      if(nextHashIdx >= 4) {
        string tmp = std::to_string(counter) + s;
        //cout << tmp << endl;
        counter += 37;
        SHA2::get256(tmp.c_str(), hash);
        nextHashIdx = 0;
      }
      nextValue = hash[nextHashIdx];
      nextHashIdx += 1;
    } while(nextValue == 0);
    return nextValue;
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

size_t Rand::nextIndexCumulative(const double* cumRelProbs, size_t n)
{
  assert(n > 0);
  assert(n < 0xFFFFFFFF);
  double_t sum = cumRelProbs[n-1];
  double d = nextDouble(sum);
  size_t r = BSearch::findFirstGt(cumRelProbs,d,0,n);
  if(r == n)
    return n-1;
  return r;
}


//Marsaglia and Tsang's algorithm
double Rand::nextGamma(double a) {
  if(!(a > 0.0))
    throw StringError("Rand::nextGamma: invalid value for a: " + Global::doubleToString(a));

  if(a <= 1.0) {
    double r = nextGamma(a + 1.0);
    double inva = 1.0 / a;
    //Technically in C++, pow(0,0) could be implementation-dependent or result in an error
    //so we explicitly force the desired behavior
    double scale = inva == 0.0 ? 1.0 : pow(nextDouble(), inva);
    return r * scale;
  }

  double d = a - 1.0/3.0;
  double c = (1.0/3.0) / sqrt(d);

  while(true) {
    double x = nextGaussian();
    double vtmp = 1.0 + c * x;
    if(vtmp <= 0.0)
      continue;
    double v = vtmp * vtmp * vtmp;
    double u = nextDouble();
    double xx = x * x;
    if(u < 1.0 - 0.0331 * xx * xx)
      return d * v;
    if(u == 0.0 || log(u) < 0.5 * xx + d * (1.0 - v + log(v)))
      return d * v;
  }

  //Numeric analysis notes:
  // d >= 2/3
  // c:  0 <= 1/sqrt(9d) <= 0.4 ish
  // vtmp > 0  vtmp < some large number since gaussian can't return infinity
  // u [0,1)
  // v > 0  < some large number
  // xx >= 0 < some large number
}

static void simpleTest()
{
  Rand rand("abc");

  const uint32_t expected[24] = {
    0x1C6B83BDu,
    0xFB7677DBu,
    0x698688D5u,
    0xA3CD21C3u,
    0xD0AD5B77u,
    0x8F889E6Eu,
    0x22852278u,
    0xD71A114Du,
    0x295EF301u,
    0xAA0CCA48u,
    0x0B7271BBu,
    0x4FE798FBu,
    0x26B4DD4Bu,
    0x78B77C1Bu,
    0x231C4DFBu,
    0x17FB87C6u,
    0x9CC23870u,
    0x1C2C2CF7u,
    0x62D51240u,
    0xF1D1A7FFu,
    0x44C45C0Au,
    0xF93ACFCEu,
    0x42B1D236u,
    0xC1069B75u,
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

  {
    uint32_t hash[4];
    const string s = "The quick brown fox jumps over the lazy dog.";
    MD5::get(s.c_str(),s.length(),hash);
    testAssert(hash[0] == 0xC209D9E4);
    testAssert(hash[1] == 0x1CFBD090);
    testAssert(hash[2] == 0xADFF68A0);
    testAssert(hash[3] == 0xD0CB22DF);
  }

  char hash[129];
  SHA2::get256("The quick brown fox jumps over the lazy dog.", hash);
  if(string(hash) != string("ef537f25c895bfa782526529a9b63d97aa631564d5d789c2b765448c8635fb6c")) {
    cout << hash << endl;
    cout << "SHA2 generated unexpected hash" << endl;
  }

  ostringstream out;

  string s;
  for(int i = 0; i<10; i++) {
    SHA2::get256(s.c_str(), hash);
    out << s << endl;
    out << hash << endl;
    for(int j = 0; j<37; j++)
      s += (char)('a'+(i % 26));
  }
  s = "";
  for(int i = 0; i<10; i++) {
    SHA2::get512(s.c_str(), hash);
    out << s << endl;
    out << hash << endl;
    for(int j = 0; j<37; j++)
      s += (char)('a'+(i % 26));
  }

  string expectedOutput = R"%%(

e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
21ec055b38ce759cd4d0f477e9bdec2c5b8199945db4439bae334a964df6246c
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
530324237c4062cee93afcc3433135c4a4729e6b234ecd83b08992a6032efafb
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbccccccccccccccccccccccccccccccccccccc
acc0500a6ebb7fc2c7b265db4e22d1ad1d55908dfdc89913520ba9be50a7720d
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccddddddddddddddddddddddddddddddddddddd
280a8797e00868c757e92d9e13e8e51eaffc4673c53a1d1a8e361cdbf010a328
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
1e01ec99dbb0c2f3fd950046138824070024554faeab5bf1e52207d445de223c
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeefffffffffffffffffffffffffffffffffffff
e5d833a44b2d96fc759f6fbed2e7f303bb7da4400fedb8b3faf395a2fde67c10
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeefffffffffffffffffffffffffffffffffffffggggggggggggggggggggggggggggggggggggg
d1add41e943cf0a880d20847366b573c9fa5181c83ac0284066186d1838c76e0
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeefffffffffffffffffffffffffffffffffffffggggggggggggggggggggggggggggggggggggghhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
2bbaf6f3a2feef7c8edd672aad9919ce18db46919ab18844ab07585188de0860
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeefffffffffffffffffffffffffffffffffffffggggggggggggggggggggggggggggggggggggghhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
d1cb4132d32c95e9146d1f439b1a58c8dcc27c0fc15939cfbe2e1bbad8099c0f

cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
ae77859a42c40e3973aa42bc8fbe8713444f65173580507d7c4bcc7c85d7f8c93204f433d506e912504ea37c766af17e649bdf6c8356f6e8e65bf4e9321987cb
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
5e7e8082ac407147815aeb5419db505d456d7d4cdd4f20c62433f4e2bf09a5c8f649f5f032d55e650f7e696408b5aa24226153988dad515eb5338e2c142c7a84
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbccccccccccccccccccccccccccccccccccccc
714169c309360077960236926fb18c77b5dfa407729b6574105dd1fd8806d04e17f9fff91c99235e1d45a307699039a41753f30cadb2759aed84c4e97d14d382
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccddddddddddddddddddddddddddddddddddddd
3f427dadf9a6ca0e194236c243f51f37b0be5811ce17abc43c80fea7dd0fc73a76a26416192b68fb2bf49be8c2f07ce365ea041672293c81cf76b9ed8f106bcb
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
52ac4b2cf1588a5804800cb26b9356824c75044a1ed5c0dbf0a088e8de77557dcec5a36f60691392eb4b7ed54243e90dc6d743143b336ce36bd30ba8c4dd787d
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeefffffffffffffffffffffffffffffffffffff
1c2983945b90a8f520becaa4d7f2ece194116e1e5f2fec15dd2838e97f4df3a7a2133b8a4353339f3952fc9c1c566783e5f8519457d5bbdcc6e5ec168173dba8
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeefffffffffffffffffffffffffffffffffffffggggggggggggggggggggggggggggggggggggg
06d9cf68dca1e30acbf0e4a02896262ef698580da1a531506f791e3189747226be5da53b085aeba40795f20aa2771b1c6ab69cb320e7b6dc3c1adb3bdac475c1
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeefffffffffffffffffffffffffffffffffffffggggggggggggggggggggggggggggggggggggghhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
a9fda6993a622f9e72076492050cd04625dbd85140bfbba6d08a3fb24eb42143dbc7b1c4f8cd61cf6ccd67eb8b9825f448f8c44b312a6c762c7c1ca5eb5f34b7
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbcccccccccccccccccccccccccccccccccccccdddddddddddddddddddddddddddddddddddddeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeefffffffffffffffffffffffffffffffffffffggggggggggggggggggggggggggggggggggggghhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
404c54bfd3552a11352a3c70172d706b159506b4cc0d40126a4291b48e3c1e506ab882d8b6a8380442b5fd7cdca1fc4e9e9ff51379447181fe214a2a07b477ff

)%%";

  TestCommon::expect("hash test",out,expectedOutput);
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

    string expected = R"%%(
rand.nextUInt()
476808125
4218845147
1770424533
2748129731
3501022071
2408095342
579150456
3608809805
694088449
2852964936
192049595
1340578043
649387339
2025290779
589057531
402360262
rand.nextUInt(27)
5
3
15
8
18
15
12
0
10
20
23
18
24
20
9
14
rand.nextInt()
-334527988
756912801
632180158
-1273579939
-1918485732
-1069916434
-2014707267
1657792760
-1002776568
1890146240
1996933002
1450387446
1700927140
129162816
1321628292
1374562123
rand.nextInt(-8,8)
4
8
-1
4
3
2
6
6
-2
5
-8
1
3
5
-3
7
rand.nextUInt64()
10638269307294496533
18411047602699644662
866485615523685959
11468038378131977312
16992521898711914992
8030137482563021028
255386410622334748
17040394105744604861
8702305142317123362
7011499938802287143
14406966689924300975
17729451932997526225
6624400776209450803
14445168003370372628
12783387423358787378
15169533704952901762
rand.nextUInt64(0xFFffffFFFFULL)
0000001F6569C181
00000031F6FA94F6
0000007D3140612A
00000030F0CE30AF
000000DB09241242
0000002894F80A15
0000001F9AF78FCF
000000AA9EEBFBAF
000000791CD0B9DA
0000007B5A365995
000000C65D52AC90
0000000129D8C5CF
0000008A81AD3722
0000004E7F01EA56
0000000FE3DD0088
000000F0656F65E2
rand.nextDouble()
0.773755
0.456202
0.613086
0.96546
0.542037
0.604754
0.505963
0.900103
0.285093
0.516344
0.438879
0.836596
0.230791
0.767963
0.868906
0.690635
rand.nextDouble(12)
7.40591
2.53628
1.4952
8.53554
1.42112
11.7662
9.05463
4.45068
8.83543
3.00543
7.80766
7.32897
7.94891
4.2592
5.89748
4.71898
rand.nextDouble(-100,100)
-72.0807
-31.9268
90.4578
-89.2463
-72.0024
87.1695
-47.2369
51.9239
84.0686
64.4074
-97.5942
-22.6104
21.778
94.8935
82.9729
-72.1948
rand.nextGaussian()
-0.967472
-0.863108
-1.42761
1.43227
0.0918199
-0.418531
1.15332
0.723536
0.483234
-0.662346
0.360351
-0.374887
-1.08184
0.789806
-0.0137953
1.43438
rand.nextLogistic()
3.80282
-0.248743
-0.0823388
-2.60478
-2.13827
-1.35933
-0.696443
-0.469175
1.11787
0.845649
0.690096
0.598526
0.345431
-0.228918
0.501948
-3.36977
)%%";
    TestCommon::expect(name,out,expected);
  }

  {
    const char* name = "nextIndexCumulative tests";
    Rand rand(name);

    double probs[5] = {1.0, 4.0, 2.5, 0.5, 2.0};
    double cumProbs[5];
    for(int i = 0; i<5; i++)
      cumProbs[i] = (i == 0 ? probs[i] : probs[i] + cumProbs[i-1]);

    int frequencies[5] = {0,0,0,0,0};
    for(int i = 0; i<10000; i++) {
      int r = (int)rand.nextIndexCumulative(cumProbs,5);
      testAssert(r >= 0 && r < 5);
      frequencies[r]++;
    }
    for(int i = 0; i<5; i++)
      out << frequencies[i] << endl;

    string expected = R"%%(
1014
3923
2469
525
2069
)%%";
    TestCommon::expect(name,out,expected);
  }

  {
    const char* name = "Gamma tests";
    Rand rand("def");

    double tinySubnormal = 4.9406564584124654e-324;
    double tinyNormal = 2.2250738585072014e-308;
    double maxDouble = 1.7976931348623157e308;
    out << "pow(0.5,1e300) " << pow(0.5, 1.0e300) << endl;
    out << "tinySubnormal " << Global::strprintf("%.10g",tinySubnormal) << endl;
    out << "tinyNormal " << Global::strprintf("%.10g",tinyNormal) << endl;
    out << "maxDouble " << Global::strprintf("%.10g",maxDouble) << endl;
    out << "log(tinySubnormal) " << Global::strprintf("%.10g",log(tinySubnormal)) << endl;
    out << "log(tinyNormal) " << Global::strprintf("%.10g",log(tinyNormal)) << endl;

    out << "rand.nextGamma(tinySubnormal)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(tinySubnormal) << endl;
    out << "rand.nextGamma(tinyNormal)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(tinyNormal) << endl;
    out << "rand.nextGamma(0.001)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(0.001) << endl;
    out << "rand.nextGamma(0.1)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(0.1) << endl;
    out << "rand.nextGamma(1)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(1) << endl;
    out << "rand.nextGamma(1.0000000000001)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(1.0000000000001) << endl;
    out << "rand.nextGamma(4)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(4) << endl;
    out << "rand.nextGamma(100)" << endl;
    for(int i = 0; i<16; i++) out << rand.nextGamma(100) << endl;
    out << "rand.nextGamma(1e308)" << endl;
    for(int i = 0; i<8; i++) out << rand.nextGamma(1e308) << endl;
    out << "rand.nextGamma(maxDouble)" << endl;
    for(int i = 0; i<8; i++) out << rand.nextGamma(maxDouble) << endl;

    string expected = R"%%(
pow(0.5,1e300) 0
tinySubnormal 4.940656458e-324
tinyNormal 2.225073859e-308
maxDouble 1.797693135e+308
log(tinySubnormal) -744.4400719
log(tinyNormal) -708.3964185
rand.nextGamma(tinySubnormal)
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
rand.nextGamma(tinyNormal)
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
rand.nextGamma(0.001)
1.05907e-101
1.79564e-171
0
3.99929e-164
0
0
1.86086e-92
0
0
0
6.21457e-150
5.98916e-180
0
0
0
1.24842e-40
rand.nextGamma(0.1)
4.0043e-05
0.111808
0.015495
2.33564e-16
0.513398
2.35636
0.000129372
0.000854086
0.00742527
0.0990275
6.59447e-08
5.98863e-05
7.90218e-19
4.9846e-11
0.0923923
0.0258861
rand.nextGamma(1)
2.12756
4.84922
0.171508
0.341829
0.207699
1.0533
0.40069
3.39204
3.79859
0.115891
0.277927
0.133629
1.1966
0.185524
0.40222
0.765241
rand.nextGamma(1.0000000000001)
1.71238
1.35466
0.866271
0.263334
0.197441
0.532812
1.97214
1.42988
0.151373
1.52947
0.0350315
1.02099
0.893285
0.840119
1.09308
1.05674
rand.nextGamma(4)
4.73018
2.71548
1.46158
3.56187
7.45509
0.930581
3.03565
4.60005
7.72403
2.25881
4.60951
2.17532
1.8698
6.90699
2.49718
0.647821
rand.nextGamma(100)
108.41
107.61
93.1365
118.167
84.7977
89.0776
112.248
110.362
98.8992
104.624
101.665
87.0437
80.8363
93.7856
100.486
98.0087
rand.nextGamma(1e308)
1e+308
1e+308
1e+308
1e+308
1e+308
1e+308
1e+308
1e+308
rand.nextGamma(maxDouble)
1.79769e+308
1.79769e+308
1.79769e+308
1.79769e+308
1.79769e+308
1.79769e+308
1.79769e+308
1.79769e+308
)%%";
    TestCommon::expect(name,out,expected);

    if(std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::has_infinity) {
      double inf = INFINITY;
      out << "pow(0.5,inf) " << pow(0.5, inf) << endl;
      out << "pow(1.0,inf) " << pow(1.0, inf) << endl;
      out << "log(0) " << Global::doubleToString(log(0.0)) << endl;
      out << "rand.nextGamma(inf)" << endl;
      for(int i = 0; i<8; i++) out << rand.nextGamma(inf) << endl;
      expected = R"%%(
pow(0.5,inf) 0
pow(1.0,inf) 1
log(0) -inf
rand.nextGamma(inf)
inf
inf
inf
inf
inf
inf
inf
inf
)%%";
      TestCommon::expect(name,out,expected);
    }

  }

  {
    const char* name = "Rand moment tests";

    int bufLen = 200000;
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
      buf[i] = rand.nextExponential();
    printMoments("Exponential", 1.0, 1.0, 2.0, 6.0);

    for(int i = 0; i<bufLen; i++)
      buf[i] = rand.nextGamma(0.05);
    printMoments("Gamma(0.05)", 0.05, 0.05, 2.0/sqrt(0.05), 6.0/0.05);

    for(int i = 0; i<bufLen; i++)
      buf[i] = rand.nextGamma(0.5);
    printMoments("Gamma(0,5)", 0.5, 0.5, 2.0/sqrt(0.5), 6.0/0.5);

    for(int i = 0; i<bufLen; i++)
      buf[i] = rand.nextGamma(1.01);
    printMoments("Gamma(1.01)", 1.01, 1.01, 2.0/sqrt(1.01), 6.0/1.01);

    for(int i = 0; i<bufLen; i++)
      buf[i] = rand.nextGamma(4.0);
    printMoments("Gamma(4.0)", 4.0, 4.0, 2.0/sqrt(4.0), 6.0/4.0);

    delete[] buf;
    string expected = R"%%(
Uniform sample: Mean 0.499861 Variance 0.083304 Skew 0.000470 ExcessKurt -1.198536
Uniform expected: Mean 0.500000 Variance 0.083333 Skew 0.000000 ExcessKurt -1.200000
Gaussian sample: Mean -0.002168 Variance 1.009814 Skew -0.006133 ExcessKurt 0.009611
Gaussian expected: Mean 0.000000 Variance 1.000000 Skew 0.000000 ExcessKurt 0.000000
Logistic sample: Mean -0.006190 Variance 3.288051 Skew 0.004409 ExcessKurt 1.217031
Logistic expected: Mean 0.000000 Variance 3.289868 Skew 0.000000 ExcessKurt 1.200000
Exponential sample: Mean 1.004176 Variance 1.015094 Skew 2.038615 ExcessKurt 6.431784
Exponential expected: Mean 1.000000 Variance 1.000000 Skew 2.000000 ExcessKurt 6.000000
Gamma(0.05) sample: Mean 0.049499 Variance 0.049302 Skew 8.716963 ExcessKurt 108.076402
Gamma(0.05) expected: Mean 0.050000 Variance 0.050000 Skew 8.944272 ExcessKurt 120.000000
Gamma(0,5) sample: Mean 0.496322 Variance 0.492312 Skew 2.839173 ExcessKurt 12.203226
Gamma(0,5) expected: Mean 0.500000 Variance 0.500000 Skew 2.828427 ExcessKurt 12.000000
Gamma(1.01) sample: Mean 1.007725 Variance 1.012255 Skew 2.011365 ExcessKurt 6.055355
Gamma(1.01) expected: Mean 1.010000 Variance 1.010000 Skew 1.990074 ExcessKurt 5.940594
Gamma(4.0) sample: Mean 4.002986 Variance 4.015291 Skew 1.007541 ExcessKurt 1.536908
Gamma(4.0) expected: Mean 4.000000 Variance 4.000000 Skew 1.000000 ExcessKurt 1.500000
)%%";

    TestCommon::expect(name,out,expected);
  }

}
