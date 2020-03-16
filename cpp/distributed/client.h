#ifdef BUILD_DISTRIBUTED

#ifndef DISTRIBUTED_CLIENT_H_
#define DISTRIBUTED_CLIENT_H_

#include "../core/logger.h"
#include "../distributed/httplib_wrapper.h"
#include "../core/multithread.h"

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

  class Connection {
  public:
    Connection(const std::string& serverUrl, int serverPort, const std::string& username, const std::string& password, Logger* logger);
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&&) = delete;
    Connection& operator=(Connection&&) = delete;

    RunParameters getRunParameters();
    Task getNextTask(const std::string& baseDir);

    static std::string getModelPath(const std::string& modelName, const std::string& modelDir);
    void downloadModelIfNotPresent(const std::string& modelName, const std::string& modelDir);

    void uploadTrainingGameAndData(const Task& task, const std::string& sgfFilePath, const std::string& npzFilePath);
    void uploadEvaluationGame(const Task& task, const std::string& sgfFilePath);

  private:
    httplib::Client* httpClient;
    httplib::SSLClient* httpsClient;
    Logger* logger;

    //TODO if httplib is thread-safe, then we can remove this
    std::mutex mutex;
  };

}

#endif //DISTRIBUTED_CLIENT_H_

#endif //BUILD_DISTRIBUTED
