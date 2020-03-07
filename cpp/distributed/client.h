#ifndef DISTRIBUTED_CLIENT_H_
#define DISTRIBUTED_CLIENT_H_

#include "../core/logger.h"

namespace Client {

  struct RunParameters {
    std::string runId;
    int dataBoardLen;
    int inputsVersion;
    int maxSearchThreadsAllowed;
  };

  struct Task {
    std::string taskId;
    std::string taskGroup;
    std::string runId;

    std::string modelNameBlack;
    std::string modelNameWhite;

    std::string config;
    bool doWriteTrainingData;
    bool isEvaluationGame;
  };

  RunParameters getRunParameters();
  Task getNextTask(Logger& logger, const std::string& baseDir);
  std::string getModelPath(const std::string& modelName, const std::string& modelDir);
  void downloadModelIfNotPresent(const std::string& modelName, const std::string& modelDir);

  void uploadTrainingData(const Task& task, const std::string& filePath);
  void uploadSGF(const Task& task, const std::string& filePath);
}

#endif //DISTRIBUTED_CLIENT_H_
