#ifndef CORE_MAKEDIR_H_
#define CORE_MAKEDIR_H_

#include "../core/global.h"

namespace MakeDir {
  //Does nothing if already exists
  void make(const std::string& path);
}

#endif  // CORE_MAKEDIR_H_
