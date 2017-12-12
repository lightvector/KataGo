#include "core/global.h"
#include "core/rand.h"

int main(int argc, const char* argv[]) {
  XorShift1024Mult::test();
  PCG32::test();
  Rand::test();
  
  return 0;
}
