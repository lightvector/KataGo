
#include <ctime>
#include "../core/timer.h"
#include "../core/global.h"
#include "../core/hash.h"
#include "../core/sha2.h"
#include "../core/rand.h"

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


static int inits = 0;
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

void Rand::init()
{
  //Note that inits++ is not threadsafe
  int x = inits++;
  uint64_t time0 = (uint64_t)time(NULL);
  uint32_t clock0 = (uint32_t)clock();
  int64_t precisionTime = ClockTimer::getPrecisionSystemTime();

  string s =
    Global::intToString(x) +
    Global::uint64ToHexString(time0) +
    Global::uint32ToHexString(clock0) +
    Global::int64ToString(precisionTime);

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

void Rand::test()
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
