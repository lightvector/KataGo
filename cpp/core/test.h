#ifndef CORE_TEST_HELPERS_H
#define CORE_TEST_HELPERS_H

#include "../core/global.h"
#include <sstream>

//A version of assert that's always defined, regardless of NDEBUG
#define testAssert(EX) (void)((EX) || (TestCommon::testAssertFailed(#EX, __FILE__, __LINE__),0))

namespace TestCommon {
  inline void testAssertFailed(const char *msg, const char *file, int line) {
    Global::fatalError(string("Failed test assert: ") + string(msg) + "\n" + string("file: ") + string(file) + "\n" + string("line: ") + Global::intToString(line));
  }

  inline void expect(const char* name, ostringstream& out, const string& expected) {
    if(Global::trim(out.str()) != Global::trim(expected)) {
      cout << "Expect test failure!" << endl;
      cout << name << endl;
      cout << "Expected===============================================================" << endl;
      cout << expected << endl;
      cout << "Got====================================================================:" << endl;
      cout << out.str() << endl;
      exit(1);
    }
  }
}

#endif
