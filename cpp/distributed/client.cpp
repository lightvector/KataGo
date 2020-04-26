#ifdef BUILD_DISTRIBUTED

#include "../distributed/client.h"
#include "../game/board.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/desc.h"
#include "../external/nlohmann_json/json.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;

using namespace std;
using json = nlohmann::json;

using Client::Connection;
using Client::Task;
using Client::RunParameters;

static void debugPrintResponse(ostream& out, const std::shared_ptr<httplib::Response>& response) {
  out << "---RESPONSE---------------------" << endl;
  if(response == nullptr)
    out << "nullptr" << endl;
  else {
    out << "Status Code: " << response->status << endl;
    for(auto it = response->headers.begin(); it != response->headers.end(); ++it) {
      out << "Header: " << it->first + ": " + it->second << endl;
    }
    out << "Body:" << endl;
    if(response->body.size() <= 1000)
      out << response->body << endl;
    else {
      out << response->body.substr(0,1000) << endl;
      out << "<TRUNCATED due to length>" << endl;
    }
  }
}

static json parseJson(const std::shared_ptr<httplib::Response>& response) {
  if(response == nullptr)
    throw StringError("No response from server");
  if(response->status != 200) {
    ostringstream out;
    debugPrintResponse(out,response);
    throw StringError("Server gave response that was not status code 200 OK\n" + out.str());
  }
  try {
    return json::parse(response->body);
  }
  catch(nlohmann::detail::exception& e) {
    ostringstream out;
    debugPrintResponse(out,response);
    throw StringError("Server gave response with body that did not parse as json\n" + out.str());
  }
}

Connection::Connection(const string& serverUrl, const string& username, const string& password, Logger* lg)
  :httpClient(NULL),
   httpsClient(NULL),
   isSSL(false),
   baseResourcePath(),
   logger(lg),
   mutex()
{
  //Hacky custom URL parsing, probably isn't fully general but should be good enough for now.
  string url = serverUrl;
  int port;
  if(Global::isPrefix(url,"http://")) {
    url = Global::chopPrefix(url,"http://");
    isSSL = false;
    port = 80;
  }
  else if(Global::isPrefix(url,"https://")) {
    url = Global::chopPrefix(url,"https://");
    isSSL = true;
    port = 443;
  }
  else {
    throw StringError("serverUrl must start with 'http://' or 'https://'");
  }
  string hostAndPort = url.find_first_of("/") == string::npos ? url : url.substr(0, url.find_first_of("/"));
  url = Global::chopPrefix(url,hostAndPort);

  string host;
  if(hostAndPort.find_first_of(":") == string::npos) {
    host = hostAndPort;
  }
  else {
    host = hostAndPort.substr(0,hostAndPort.find_first_of(":"));
    bool suc = Global::tryStringToInt(hostAndPort.substr(hostAndPort.find_first_of(":")+1),port);
    if(!suc)
      throw StringError("Could not parse port in serverUrl as int: " + hostAndPort.substr(hostAndPort.find_first_of(":")+1));
    if(port < 0)
      throw StringError("serverUrl port was negative: " + hostAndPort.substr(hostAndPort.find_first_of(":")+1));
  }

  baseResourcePath = url;
  if(Global::isSuffix(baseResourcePath,"/"))
    baseResourcePath = Global::chopSuffix(baseResourcePath,"/");
  if(baseResourcePath.size() <= 0)
    baseResourcePath = "/";

  logger->write("Attempting to connect to server");
  logger->write("isSSL: " + string(isSSL ? "true" : "false"));
  logger->write("host: " + host);
  logger->write("port: " + Global::intToString(port));
  logger->write("baseResourcePath: " + baseResourcePath);

  if(!isSSL) {
    httpClient = new httplib::Client(host, port);
  }
  else {
    httpsClient = new httplib::SSLClient(host, port);
    // httpsClient->set_ca_cert_path("./ca-bundle.crt");
    // httpsClient->enable_server_certificate_verification(true);
  }

  //Do an initial test query to make sure the server's there!
  auto response = get("/");
  if(response == nullptr) {
    throw StringError("Could not connect to server at " + serverUrl + ", invalid host or port or otherwise no response");
  }
  else if(response->status != 200) {
    ostringstream out;
    debugPrintResponse(out,response);
    throw StringError("Server did not give status 200 for initial query, response was:\n" + out.str());
  }

  //Now set up auth as specified for any subsequent queries
  if(!isSSL) {
    httpClient->set_basic_auth(username.c_str(), password.c_str());
  }
  else {
    httpsClient->set_basic_auth(username.c_str(), password.c_str());
  }

}

Connection::~Connection() {
  delete httpClient;
  delete httpsClient;
}

static string concatPaths(const string& baseResourcePath, const string& subPath) {
  string queryPath;
  if(Global::isSuffix(baseResourcePath,"/") && Global::isPrefix(subPath,"/"))
    queryPath = Global::chopSuffix(baseResourcePath,"/") + subPath;
  else if(Global::isSuffix(baseResourcePath,"/") || Global::isPrefix(subPath,"/"))
    queryPath = baseResourcePath + subPath;
  else
    queryPath = baseResourcePath + "/" + subPath;
  return queryPath;
}

std::shared_ptr<httplib::Response> Connection::get(const string& subPath) {
  string queryPath = concatPaths(baseResourcePath,subPath);

  std::lock_guard<std::mutex> lock(mutex);
  if(isSSL) {
    std::shared_ptr<httplib::Response> response = httpsClient->Get(queryPath.c_str());
    if(response == nullptr) {
      auto result = httpsClient->get_openssl_verify_result();
      if(result) {
        string err = X509_verify_cert_error_string(result);
        logger->write("SSL certificate validation error (X509) - is the website secure?: " + err);
      }
    }
    return response;
  }
  else {
    return httpClient->Get(queryPath.c_str());
  }
}

std::shared_ptr<httplib::Response> Connection::getBigFile(const string& fullPath, std::function<bool(const char *data, size_t data_length)> f) {
  std::lock_guard<std::mutex> lock(mutex);
  if(isSSL) {
    std::shared_ptr<httplib::Response> response = httpsClient->Get(fullPath.c_str(),f);
    if(response == nullptr) {
      auto result = httpsClient->get_openssl_verify_result();
      if(result) {
        string err = X509_verify_cert_error_string(result);
        logger->write("SSL certificate validation error (X509) - is the website secure?: " + err);
      }
    }
    return response;
  }
  else {
    return httpClient->Get(fullPath.c_str(),f);
  }
}


std::shared_ptr<httplib::Response> Connection::post(const string& subPath, const string& data, const string& dtype) {
  string queryPath = concatPaths(baseResourcePath,subPath);

  std::lock_guard<std::mutex> lock(mutex);
  if(isSSL) {
    std::shared_ptr<httplib::Response> response = httpsClient->Post(queryPath.c_str(),data.c_str(),dtype.c_str());
    if(response == nullptr) {
      auto result = httpsClient->get_openssl_verify_result();
      if(result) {
        string err = X509_verify_cert_error_string(result);
        logger->write("SSL certificate validation error (X509) - is the website secure?: " + err);
      }
    }
    return response;
  }
  else {
    return httpClient->Post(queryPath.c_str(),data.c_str(),dtype.c_str());
  }
}

static void throwFieldNotFound(const json& response, const char* field) {
  throw StringError(string("Field ") + field + " not found in json response: " + response.dump());
}
static void throwInvalidValue(const json& response, const char* field) {
  throw StringError(string("Field ") + field + " had invalid value in json response: " + response.dump());
}

template <typename T>
static T parse(const json& response, const char* field) {
  if(response.find(field) == response.end())
    throwFieldNotFound(response,field);
  try {
    T x = response[field].get<T>();
    return x;
  }
  catch(nlohmann::detail::exception& e) {
    throwInvalidValue(response,field);
  }
  throw StringError("BUG, should not reach here");
}

template <typename T>
static T parseInteger(const json& response, const char* field, T min, T max) {
  if(response.find(field) == response.end())
    throwFieldNotFound(response,field);
  try {
    if(!response[field].is_number_integer())
      throwInvalidValue(response,field);
    T x = response[field].get<T>();
    if(x < min || x > max)
      throwInvalidValue(response,field);
    return x;
  }
  catch(nlohmann::detail::exception& e) {
    throwInvalidValue(response,field);
  }
  throw StringError("BUG, should not reach here");
}

template <typename T>
static T parseReal(const json& response, const char* field, T min, T max) {
  if(response.find(field) == response.end())
    throwFieldNotFound(response,field);
  try {
    if(!response[field].is_number_float())
      throwInvalidValue(response,field);
    T x = response[field].get<T>();
    if(x < min || x > max || !isfinite(x))
      throwInvalidValue(response,field);
    return x;
  }
  catch(nlohmann::detail::exception& e) {
    throwInvalidValue(response,field);
  }
  throw StringError("BUG, should not reach here");
}

RunParameters Connection::getRunParameters() {
  try {
    json response = parseJson(get("/api/runs/"));

    vector<json> runs;
    try {
      runs = response["results"].get<vector<json>>();
    }
    catch(nlohmann::detail::exception& e) {
      throw StringError(string("Could not parse runs from server response: ") + e.what() + "\nResponse was:\n" + response.dump());
    }
    if(runs.size() <= 0)
      throw StringError("No active runs found from server, response was:\n" + response.dump());

    //TODO do something better here
    //For now just choose the first
    json run = runs[0];

    RunParameters runParams;
    runParams.runName = parse<string>(run,"name");
    runParams.dataBoardLen = parseInteger<int>(run,"data_board_len",3,Board::MAX_LEN);
    runParams.inputsVersion = parseInteger<int>(run,"inputs_version",NNModelVersion::oldestInputsVersionImplemented,NNModelVersion::latestInputsVersionImplemented);
    runParams.maxSearchThreadsAllowed = parseInteger<int>(run,"max_search_threads_allowed",1,16384);
    return runParams;
  }
  catch(const StringError& e) {
    throw StringError(string("Error when requesting initial run parameters from server: ") + e.what());
  }
}

static void retryLoop(const char* errorLabel, bool retryOnFailure, Logger* logger, std::function<void()> f) {
  double failureInterval = 5.0;
  int maxTries = retryOnFailure ? 40 : 1;
  for(int i = 0; i<maxTries; i++) {
    try {
      f();
    }
    catch(const StringError& e) {
      if(i >= maxTries-1)
        throw;
      logger->write(string(errorLabel) + "Error connecting to server, possibly an internet blip, or possibly the server is down or temporarily misconfigured, waiting " + Global::doubleToString(failureInterval) + " seconds and trying again.");
      logger->write(string("Error was:\n") + e.what());
      std::this_thread::sleep_for(std::chrono::duration<double>(failureInterval));
      failureInterval = round(failureInterval * 1.3 + 1.0);
      continue;
    }
    if(i > 0)
      logger->write(string(errorLabel) + "Connection to server is back!");
    break;
  }
}

Task Connection::getNextTask(const string& baseDir, bool retryOnFailure) {
  (void)baseDir;
  Task task;

  auto f = [&]() {
    json response = parseJson(post("/api/tasks/","","text/plain"));
    //TODO
    //lightvector: when I tried this I got back this kind of response
    //{"type":"dynamic","kind":"training","config":"FILL ME","network":{"name":"","model_file":null}}

    string kind = parse<string>(response,"kind");
    if(kind == "training") {
      json networkProperties = parse<json>(response,"network");

      task.taskId = ""; //TODO Server doesn't care? What about avoiding multiply reporting games?
      task.taskGroup = parse<string>(networkProperties,"name");
      task.runName = ""; //TODO have server report run name here
      task.config = parse<string>(response,"config");
      task.modelNameBlack = parse<string>(networkProperties,"name");
      task.modelUrlBlack = parse<string>(networkProperties,"model_file");
      task.modelNameWhite = task.modelNameBlack;
      task.modelUrlWhite = task.modelUrlWhite;
      task.doWriteTrainingData = true;
      task.isEvaluationGame = false;
    }
    else if(kind == "ranking_estimation") {
      //TODO is this right?
      json blackNetworkProperties = parse<json>(response,"black_network");
      json whiteNetworkProperties = parse<json>(response,"white_network");

      task.taskId = ""; //TODO Server doesn't care? What about avoiding multiply reporting games?
      //TODO can we have this be the newest network?
      task.taskGroup = "evaluation_" + parse<string>(blackNetworkProperties,"name");
      task.runName = ""; //TODO have server report run name here
      task.config = parse<string>(response,"config");
      task.modelNameBlack = parse<string>(blackNetworkProperties,"name");
      task.modelUrlBlack = parse<string>(blackNetworkProperties,"model_file");
      task.modelNameWhite = parse<string>(whiteNetworkProperties,"name");
      task.modelUrlWhite = parse<string>(whiteNetworkProperties,"model_file");
      task.doWriteTrainingData = false;
      task.isEvaluationGame = true;
    }
    else {
      throw StringError("kind was neither 'training' or 'ranking_estimation' in json response: " + response.dump());
    }
  };
  retryLoop("getNextTask",retryOnFailure,logger,f);

  //TODO just for reference
  // string config = Global::readFile(baseDir + "/" + "testDistributedConfig.cfg");
  // task.config = config;
  return task;
}

//STATIC method
string Connection::getModelPath(const string& modelName, const string& modelDir) {
  return modelDir + "/" + modelName + ".bin.gz";
}

void Connection::downloadModelIfNotPresent(const string& modelName, const string& modelDir, const string& modelUrl, bool retryOnFailure) {
  string path = getModelPath(modelName,modelDir);
  string tmpPath = path + ".tmp";

  //Model already exists
  if(bfs::exists(bfs::path(path)))
    return;

  auto f = [&]() {
    size_t totalDataSize = 0;
    ofstream out(tmpPath,ios::binary);

    //TODO can we also have the server tell us the total data length expected as well as the sha256 hash of the file
    //so that we can verify download integrity?
    std::shared_ptr<httplib::Response> response = getBigFile(
      modelUrl, [&out,&totalDataSize](const char* data, size_t data_length) {
        out.write(data, data_length);
        totalDataSize += data_length;
        return true;
      }
    );
    out.close();

    if(response == nullptr)
      throw StringError("No response from server");
    if(response->status != 200) {
      ostringstream outs;
      debugPrintResponse(outs,response);
      throw StringError("Server gave response that was not status code 200 OK\n" + outs.str());
    }

    //TODO
    //if(totalDataSize != expectedDataSize) ...

    //Attempt to load the model file to verify integrity
    {
      ModelDesc* descBuf = new ModelDesc();
      ModelDesc::loadFromFileMaybeGZipped(tmpPath,*descBuf);
      delete descBuf;
    }

    //Done! Rename the file into the right place
    std::rename(tmpPath.c_str(),path.c_str());
  };
  retryLoop("downloadModelIfNotPresent",retryOnFailure,logger,f);
}

void Connection::uploadTrainingGameAndData(const Task& task, const FinishedGameData* gameData, const string& sgfFilePath, const string& npzFilePath, bool retryOnFailure) {
  cout << "UPLOAD TRAINING DATA " << task.taskId << " " << task.taskGroup << " " << task.runName << " " << sgfFilePath << " " << npzFilePath << endl;
}

void Connection::uploadEvaluationGame(const Task& task, const FinishedGameData* gameData, const string& sgfFilePath, bool retryOnFailure) {
  cout << "UPLOAD SGF " << task.taskId << " " << task.taskGroup << " " << task.runName << " " << sgfFilePath << endl;
}

#endif //BUILD_DISTRIBUTED
