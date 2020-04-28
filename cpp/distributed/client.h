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

  struct ModelInfo {
    std::string name;
    std::string url;
    int64_t bytes;
    std::string sha256;
  };

  struct Task {
    std::string taskId;
    std::string taskGroup;
    std::string runName;

    ModelInfo modelBlack;
    ModelInfo modelWhite;

    std::string config;
    bool doWriteTrainingData;
    bool isRatingGame;
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
    Task getNextTask(const std::string& baseDir, bool retryOnFailure);

    static std::string getModelPath(const Client::ModelInfo& modelInfo, const std::string& modelDir);
    void downloadModelIfNotPresent(const Client::ModelInfo& modelInfo, const std::string& modelDir, bool retryOnFailure);

    void uploadTrainingGameAndData(const Task& task, const FinishedGameData* gameData, const std::string& sgfFilePath, const std::string& npzFilePath, bool retryOnFailure);
    void uploadRatingGame(const Task& task, const FinishedGameData* gameData, const std::string& sgfFilePath, bool retryOnFailure);

  private:
    std::shared_ptr<httplib::Response> get(const std::string& subPath);
    std::shared_ptr<httplib::Response> post(const std::string& subPath, const std::string& data, const std::string& dtype);
    std::shared_ptr<httplib::Response> getBigFile(const std::string& fullPath, std::function<bool(const char* data, size_t data_length)> f);
    std::shared_ptr<httplib::Response> postMulti(const std::string& subPath, const httplib::MultipartFormDataItems& data);


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
