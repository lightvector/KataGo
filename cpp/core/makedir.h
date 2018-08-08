#ifndef MAKEDIR_H_
#define MAKEDIR_H_

#include "../core/global.h"

namespace MakeDir {
  //Does nothing if already exists
  void make(const string& path);
}

#endif
