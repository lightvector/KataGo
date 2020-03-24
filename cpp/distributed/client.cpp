#ifdef BUILD_DISTRIBUTED

#include "../distributed/client.h"

using namespace std;

using Client::Connection;
using Client::Task;
using Client::RunParameters;

Connection::Connection(const string& serverUrl, int serverPort, const string& username, const string& password, Logger* lg)
  :httpClient(NULL),
   httpsClient(NULL), //TODO currently unused
   logger(lg),
   mutex()
{
  (void)serverUrl;
  (void)username;
  (void)password;

  httpClient = new httplib::Client(serverUrl, serverPort);
  httpClient->set_basic_auth(username.c_str(), password.c_str());
  auto res = httpClient->Get("/api/users/");
  logger->write(res->body);
}

Connection::~Connection() {
  delete httpClient;
  delete httpsClient;
}

RunParameters Connection::getRunParameters() {
  std::lock_guard<std::mutex> lock(mutex);

  RunParameters runParams;
  runParams.runName = "testrun";
  runParams.dataBoardLen = 19;
  runParams.inputsVersion = 7;
  runParams.maxSearchThreadsAllowed = 8;
  return runParams;
}


Task Connection::getNextTask(const string& baseDir) {
  std::lock_guard<std::mutex> lock(mutex);

  Task task;
  task.taskId = "test";
  task.taskGroup = "testgroup";
  task.runName = "testrun";
  task.modelNameBlack = "g170-b10c128-s197428736-d67404019";
  task.modelUrlBlack = "TODO";
  task.modelNameWhite = "g170-b10c128-s197428736-d67404019";
  task.modelUrlWhite = "TODO";
  task.doWriteTrainingData = true;
  task.isEvaluationGame = false;

  string config = Global::readFile(baseDir + "/" + "testDistributedConfig.cfg");
  task.config = config;
  return task;
}

//STATIC method
string Connection::getModelPath(const string& modelName, const string& modelDir) {
  return modelDir + "/" + modelName + ".bin.gz";
}

void Connection::downloadModelIfNotPresent(const string& modelName, const string& modelDir, const string& modelUrl) {
  std::lock_guard<std::mutex> lock(mutex);

  (void)modelUrl;

  string path = getModelPath(modelName,modelDir);
  ifstream test(path.c_str());
  if(!test.good()) {
    throw StringError("Currently for testing, " + path + " is expected to be a valid KataGo model file");
  }
}

void Connection::uploadTrainingGameAndData(const Task& task, const string& sgfFilePath, const string& npzFilePath) {
  std::lock_guard<std::mutex> lock(mutex);
  cout << "UPLOAD TRAINING DATA " << task.taskId << " " << task.taskGroup << " " << task.runName << " " << sgfFilePath << " " << npzFilePath << endl;
}

void Connection::uploadEvaluationGame(const Task& task, const string& sgfFilePath) {
  std::lock_guard<std::mutex> lock(mutex);
  cout << "UPLOAD SGF " << task.taskId << " " << task.taskGroup << " " << task.runName << " " << sgfFilePath << endl;
}

#endif //BUILD_DISTRIBUTED
