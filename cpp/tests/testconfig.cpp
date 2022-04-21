#include "../tests/tests.h"

#include "../core/config_parser.h"

using namespace std;
using namespace TestCommon;

void Tests::runConfigTests() {
  ConfigParser parser("data/test.cfg", true);
  cout << parser.getAllKeyVals();
}

int main(int, char *argv[]) {
  Tests::runConfigTests();
  return 0;
}
