
#ifndef CORE_BASE64_H_
#define CORE_BASE64_H_

#include <iostream>

#include "../core/global.h"

namespace Base64
{
  std::string encode(const std::string& s);
  std::string decode(const std::string& s);

  void runTests();
}

#endif  // CORE_BASE64_H_
