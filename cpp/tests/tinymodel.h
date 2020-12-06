#ifndef TESTS_TINYMODEL_H
#define TESTS_TINYMODEL_H

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"

namespace TinyModelTest {
  extern const char* tinyModelBase64;

  void runTinyModelTest(const std::string& baseDir, Logger& logger, ConfigParser& cfg);
}



#endif
