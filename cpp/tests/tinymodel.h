#ifndef TESTS_TINYMODEL_H
#define TESTS_TINYMODEL_H

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"

class NNEvaluator;

namespace TinyModelTest {
  extern const char* tinyModelBase64Part0;
  extern const char* tinyModelBase64Part1;
  extern const char* tinyModelBase64Part2;
  extern const char* tinyModelBase64Part3;
  extern const char* tinyModelBase64Part4;
  extern const char* tinyModelBase64Part5;
  extern const char* tinyModelBase64Part6;

  NNEvaluator* runTinyModelTest(const std::string& baseDir, Logger& logger, ConfigParser& cfg, bool randFileName);
}



#endif
