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

  inline void expect(const char* name, const string& actual, const string& expected) {
    string a = Global::trim(actual);
    string e = Global::trim(expected);
    vector<string> alines = Global::split(a,'\n');
    vector<string> elines = Global::split(e,'\n');

    bool matches = true;
    int firstLineDiff = 0;
    for(int i = 0; i<std::max(alines.size(),elines.size()); i++) {
      if(i >= alines.size() || i >= elines.size()) {
        firstLineDiff = i;
        matches = false;
        break;
      }
      if(Global::trim(alines[i]) != Global::trim(elines[i])) {
        firstLineDiff = i;
        matches = false;
        break;
      }
    }

    if(!matches) {
      cout << "Expect test failure!" << endl;
      cout << name << endl;
      cout << "Expected===============================================================" << endl;
      cout << expected << endl;
      cout << "Got====================================================================" << endl;
      cout << actual << endl;

      cout << "=======================================================================" << endl;
      cout << "First line different (0-indexed) = " << firstLineDiff << endl;
      string actualLine = firstLineDiff >= alines.size() ? string() : alines[firstLineDiff];
      string expectedLine = firstLineDiff >= elines.size() ? string() : elines[firstLineDiff];
      cout << "Actual  : " << actualLine << endl;
      cout << "Expected: " << expectedLine << endl;
      int j;
      for(j = 0; j<actualLine.size() && j < expectedLine.size(); j++)
        if(actualLine[j] != expectedLine[j])
          break;
      cout << "Char " << j << " differs" << endl;

      exit(1);
    }
  }

  inline void expect(const char* name, ostringstream& actual, const string& expected) {
    expect(name,actual.str(),expected);
    actual.str("");
    actual.clear();
  }

}

#endif
