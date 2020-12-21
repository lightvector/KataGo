#ifdef BUILD_DISTRIBUTED

#ifndef DISTRIBUTED_CLIENT_H_
#define DISTRIBUTED_CLIENT_H_

#include "../core/logger.h"
#include "../core/rand.h"
#include "../distributed/httplib_wrapper.h"
#include "../core/multithread.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"

namespace Client {

  struct RunParameters {
    std::string runName;
    std::string infoUrl;
    int dataBoardLen;
    int inputsVersion;
    int maxSearchThreadsAllowed;
  };

  struct ModelInfo {
    std::string name;
    std::string infoUrl;
    std::string downloadUrl;
    int64_t bytes;
    std::string sha256;
    bool isRandom;

    void failIfSha256Mismatch(const std::string& modelPath) const;
  };

  struct DownloadState {
    std::condition_variable downloadingInProgressVar;
    bool downloadingInProgress;
    DownloadState();
    ~DownloadState();
  };

  struct Task {
    std::string taskId;
    std::string taskGroup;
    std::string runName;
    std::string runInfoUrl;

    ModelInfo modelBlack;
    ModelInfo modelWhite;

    std::string config;
    std::vector<Sgf::PositionSample> startPoses;
    bool doWriteTrainingData;
    bool isRatingGame;
  };

  class Connection {
  public:
    Connection(
      const std::string& serverUrl,
      const std::string& username,
      const std::string& password,
      const std::string& caCertsFile,
      const std::string& proxyHost,
      int proxyPort,
      Logger* logger
    );
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&&) = delete;
    Connection& operator=(Connection&&) = delete;

    RunParameters getRunParameters();
    //Returns true if a task was obtained. Returns false if no task was obtained, but not due to an error (e.g. shouldStop).
    //Raises an exception upon a repeated error that persists long enough.
    bool getNextTask(
      Task& task,
      const std::string& baseDir,
      bool retryOnFailure,
      bool allowRatingTask,
      int taskRepFactor,
      std::atomic<bool>& shouldStop
    );

    static std::string getModelPath(const Client::ModelInfo& modelInfo, const std::string& modelDir);

    //Returns true if a model was downloaded or download was not necessary.
    //Returns false if a model count not be downloaded, but not due to an error (e.g. shouldStop).
    //Raises an exception upon a repeated error that persists long enough, or if the connection to the
    //server is working but somehow there is a mismatch on file length or hash or other model integrity
    bool downloadModelIfNotPresent(
      const Client::ModelInfo& modelInfo, const std::string& modelDir,
      std::atomic<bool>& shouldStop
    );

    //Query server for newest model and maybe download it, even if it is not being used by tasks yet.
    bool maybeDownloadNewestModel(
      const std::string& modelDir, std::atomic<bool>& shouldStop
    );

    //Returns true if data was uploaded or upload was not needed.
    //Returns false if it was not, but not due to an error (e.g. shouldStop).
    //Raises an exception upon a repeated error that persists long enough.
    bool uploadTrainingGameAndData(
      const Task& task, const FinishedGameData* gameData, const std::string& sgfFilePath, const std::string& npzFilePath, const int64_t numDataRows,
      bool retryOnFailure, std::atomic<bool>& shouldStop
    );
    bool uploadRatingGame(
      const Task& task, const FinishedGameData* gameData, const std::string& sgfFilePath,
      bool retryOnFailure, std::atomic<bool>& shouldStop
    );

  private:
    httplib::Result get(const std::string& subPath);
    httplib::Result post(const std::string& subPath, const std::string& data, const std::string& dtype);
    httplib::Result postMulti(const std::string& subPath, const httplib::MultipartFormDataItems& data);

    std::string getTmpModelPath(const Client::ModelInfo& modelInfo, const std::string& modelDir);
    bool retryLoop(const char* errorLabel, int maxTries, std::atomic<bool>& shouldStop, std::function<void(int&)> f);

    std::unique_ptr<httplib::Client> httpClient;
    std::unique_ptr<httplib::SSLClient> httpsClient;
    bool isSSL;

    std::string serverUrl;
    std::string username;
    std::string password;

    std::string baseResourcePath;
    std::string caCertsFile;
    std::string proxyHost;
    int proxyPort;

    //Fixed string different on every startup but shared across all requests for this run of the client
    std::string clientInstanceId;

    Logger* logger;
    Rand rand;

    std::mutex downloadStateMutex;
    std::map<std::string,std::shared_ptr<DownloadState>> downloadStateByUrl;

    //TODO if httplib is thread-safe, then we can remove this
    std::mutex mutex;

    void recreateClients();

    bool actuallyDownloadModel(
      const Client::ModelInfo& modelInfo, const std::string& modelDir,
      std::atomic<bool>& shouldStop
    );
  };

}

#endif //DISTRIBUTED_CLIENT_H_

#endif //BUILD_DISTRIBUTED
