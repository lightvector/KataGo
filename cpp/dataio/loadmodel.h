#ifndef DATAIO_LOADMODEL_H_
#define DATAIO_LOADMODEL_H_

#include "../core/global.h"
#include "../core/logger.h"

namespace LoadModel {

  bool findLatestModel(const std::string& modelsDir, Logger& logger, std::string& modelName, std::string& modelFile, std::string& modelDir, time_t& modelTime);

}


#endif  // DATAIO_LOADMODEL_H_
