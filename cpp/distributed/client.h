#ifdef BUILD_DISTRIBUTED

#ifndef DISTRIBUTED_CLIENT_H_
#define DISTRIBUTED_CLIENT_H_

#include "../core/logger.h"
#include "../distributed/httplib_wrapper.h"
#include "../core/multithread.h"
#include "../dataio/trainingwrite.h"

namespace Client {

  struct RunParameters {
    std::string runName;
    int dataBoardLen;
    int inputsVersion;
    int maxSearchThreadsAllowed;
  };

  struct Task {
    std::string taskId;
    std::string taskGroup;
    std::string runName;

    std::string modelNameBlack;
    std::string modelUrlBlack;
    std::string modelNameWhite;
    std::string modelUrlWhite;

    std::string config;
    bool doWriteTrainingData;
    bool isEvaluationGame;
  };

  class Connection {
  public:
    Connection(const std::string& serverUrl, const std::string& username, const std::string& password, Logger* logger);
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&&) = delete;
    Connection& operator=(Connection&&) = delete;

    RunParameters getRunParameters();
    Task getNextTask(const std::string& baseDir);

    static std::string getModelPath(const std::string& modelName, const std::string& modelDir);
    void downloadModelIfNotPresent(const std::string& modelName, const std::string& modelDir, const std::string& modelUrl);

    void uploadTrainingGameAndData(const Task& task, const FinishedGameData* gameData, const std::string& sgfFilePath, const std::string& npzFilePath);
    void uploadEvaluationGame(const Task& task, const FinishedGameData* gameData, const std::string& sgfFilePath);

  private:
    std::shared_ptr<httplib::Response> get(const std::string& subPath);


    httplib::Client* httpClient;
    httplib::SSLClient* httpsClient;
    bool isSSL;
    std::string baseResourcePath;

    Logger* logger;

    //TODO if httplib is thread-safe, then we can remove this
    std::mutex mutex;
  };

}

#endif //DISTRIBUTED_CLIENT_H_

#endif //BUILD_DISTRIBUTED
