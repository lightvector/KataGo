#ifndef LOADMODEL_H
#define LOADMODEL_H

#include "../core/global.h"
#include "../core/logger.h"

namespace LoadModel {

  bool findLatestModel(const string& modelsDir, Logger& logger, string& modelName, string& modelFile, string& modelDir, time_t& modelTime);

}


#endif
