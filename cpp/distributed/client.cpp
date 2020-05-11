#ifdef BUILD_DISTRIBUTED

#include "../distributed/client.h"

#include "../core/config_parser.h"
#include "../game/board.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/desc.h"
#include "../search/searchparams.h"
#include "../program/playsettings.h"
#include "../program/setup.h"
#include "../dataio/sgf.h"
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

static constexpr int MAX_RUN_NAME_LEN = 32;
static constexpr int MAX_NETWORK_NAME_LEN = 128;
static constexpr int MAX_URL_LEN = 4096;
static constexpr int MAX_TIME_LEN = 128;
static constexpr int MAX_CONFIG_NAME_LEN = 32768;

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
    if(response->body.size() <= 3000)
      out << response->body << endl;
    else {
      out << response->body.substr(0,3000) << endl;
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

//Hacky custom URL parsing, probably isn't fully general but should be good enough for now.
struct Url {
  string originalString;
  bool isSSL;
  string host;
  int port;
  string path;

  static Url parse(const string& s) {
    if(s.size() > MAX_URL_LEN)
      throw StringError("Invalid URL, too long: " + s);
    Url ret;
    ret.originalString = s;

    string url = s;
    if(Global::isPrefix(url,"http://")) {
      url = Global::chopPrefix(url,"http://");
      ret.isSSL = false;
      ret.port = 80;
    }
    else if(Global::isPrefix(url,"https://")) {
      url = Global::chopPrefix(url,"https://");
      ret.isSSL = true;
      ret.port = 443;
    }
    else {
      throw StringError("Url must start with 'http://' or 'https://', got: " + s);
    }

    string hostAndPort = url.find_first_of("/") == string::npos ? url : url.substr(0, url.find_first_of("/"));
    url = Global::chopPrefix(url,hostAndPort);

    string host;
    if(hostAndPort.find_first_of(":") == string::npos) {
      ret.host = hostAndPort;
    }
    else {
      ret.host = hostAndPort.substr(0,hostAndPort.find_first_of(":"));
      bool suc = Global::tryStringToInt(hostAndPort.substr(hostAndPort.find_first_of(":")+1),ret.port);
      if(!suc)
        throw StringError("Could not parse port in url as int: " + hostAndPort.substr(hostAndPort.find_first_of(":")+1));
      if(ret.port < 0)
        throw StringError("Url port was negative: " + hostAndPort.substr(hostAndPort.find_first_of(":")+1));
    }

    if(url.size() <= 0)
      ret.path = "/";
    else
      ret.path = url;

    return ret;
  }
};

static std::shared_ptr<httplib::Response> oneShotDownload(Logger* logger, const Url& url, std::function<bool(const char *data, size_t data_length)> f) {
  if(!url.isSSL) {
    std::unique_ptr<httplib::Client> httpClient = std::unique_ptr<httplib::Client>(new httplib::Client(url.host, url.port));
    return httpClient->Get(url.path.c_str(),f);
  }
  else {
    std::unique_ptr<httplib::SSLClient> httpsClient = std::unique_ptr<httplib::SSLClient>(new httplib::SSLClient(url.host, url.port));
    std::shared_ptr<httplib::Response> response = httpsClient->Get(url.path.c_str(),f);
    if(response == nullptr) {
      auto result = httpsClient->get_openssl_verify_result();
      if(result) {
        string err = X509_verify_cert_error_string(result);
        logger->write("SSL certificate validation error (X509) - is the website secure?: " + err);
      }
    }
    return response;
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
  Url url;
  try {
    url = Url::parse(serverUrl);
  }
  catch(const StringError& e) {
    throw StringError(string("Could not parse serverUrl in config: ") + e.what());
  }

  isSSL = url.isSSL;

  baseResourcePath = url.path;
  if(Global::isSuffix(baseResourcePath,"/"))
    baseResourcePath = Global::chopSuffix(baseResourcePath,"/");
  if(baseResourcePath.size() <= 0)
    baseResourcePath = "/";

  logger->write("Attempting to connect to server");
  logger->write("isSSL: " + string(isSSL ? "true" : "false"));
  logger->write("host: " + url.host);
  logger->write("port: " + Global::intToString(url.port));
  logger->write("baseResourcePath: " + baseResourcePath);

  if(!isSSL) {
    httpClient = new httplib::Client(url.host, url.port);
  }
  else {
    httpsClient = new httplib::SSLClient(url.host, url.port);
    //TODO
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

std::shared_ptr<httplib::Response> Connection::postMulti(const string& subPath, const httplib::MultipartFormDataItems& data) {
  string queryPath = concatPaths(baseResourcePath,subPath);

  std::lock_guard<std::mutex> lock(mutex);
  if(isSSL) {
    std::shared_ptr<httplib::Response> response = httpsClient->Post(queryPath.c_str(),data);
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
    return httpClient->Post(queryPath.c_str(),data);
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

static string parseString(const json& response, const char* field, size_t maxLen) {
  if(response.find(field) == response.end())
    throwFieldNotFound(response,field);
  try {
    string x = response[field].get<string>();
    if(x.size() >= maxLen)
      throw StringError(string("Field ") + " had Invalid response, length too long: " + Global::uint64ToString(x.size()));
    return x;
  }
  catch(nlohmann::detail::exception& e) {
    throwInvalidValue(response,field);
  }
  throw StringError("BUG, should not reach here");
}

static string parseStringOrNull(const json& response, const char* field, size_t maxLen) {
  if(response.find(field) == response.end())
    throwFieldNotFound(response,field);
  try {
    json fieldJson = response[field];
    if(fieldJson.is_null())
      return string();
    string x = fieldJson.get<string>();
    if(x.size() >= maxLen)
      throw StringError(string("Field ") + " had Invalid response, length too long: " + Global::uint64ToString(x.size()));
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
    json run = parseJson(get("/api/runs/current_for_client/"));
    RunParameters runParams;
    runParams.runName = parseString(run,"name",MAX_RUN_NAME_LEN);
    runParams.dataBoardLen = parseInteger<int>(run,"data_board_len",3,Board::MAX_LEN);
    runParams.inputsVersion = parseInteger<int>(run,"inputs_version",NNModelVersion::oldestInputsVersionImplemented,NNModelVersion::latestInputsVersionImplemented);
    runParams.maxSearchThreadsAllowed = parseInteger<int>(run,"max_search_threads_allowed",1,16384);
    return runParams;
  }
  catch(const StringError& e) {
    throw StringError(string("Error when requesting initial run parameters from server: ") + e.what());
  }
}

static bool retryLoop(const char* errorLabel, bool retryOnFailure, Logger* logger, std::atomic<bool>& shouldStop, std::function<void()> f) {
  if(shouldStop.load())
    return false;
  double stopPollFrequency = 2.0;
  double failureInterval = 5.0;
  int maxTries = retryOnFailure ? 40 : 1;
  for(int i = 0; i<maxTries; i++) {
    try {
      f();
    }
    catch(const StringError& e) {
      if(i >= maxTries-1)
        throw;
      logger->write(string(errorLabel) + ": Error connecting to server, possibly an internet blip, or possibly the server is down or temporarily misconfigured, waiting " + Global::doubleToString(failureInterval) + " seconds and trying again.");
      logger->write(string("Error was:\n") + e.what());


      double intervalRemaining = failureInterval;
      while(intervalRemaining > 0.0) {
        double sleepTime = std::min(intervalRemaining, stopPollFrequency);
        if(shouldStop.load())
          return false;
        intervalRemaining -= sleepTime;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
      }
      failureInterval = round(failureInterval * 1.3 + 1.0);
      continue;
    }
    if(i > 0)
      logger->write(string(errorLabel) + "Connection to server is back!");
    break;
  }
  return true;
}

static Client::ModelInfo parseModelInfo(const json& networkProperties) {
  Client::ModelInfo model;
  model.name = parseString(networkProperties,"name",MAX_NETWORK_NAME_LEN);
  model.url = parseStringOrNull(networkProperties,"model_file",MAX_URL_LEN);
  model.bytes = parse<int64_t>(networkProperties,"model_file_bytes");
  model.sha256 = parseString(networkProperties,"model_file_sha256",64);
  model.isRandom = parse<bool>(networkProperties,"is_random");
  return model;
}

bool Connection::getNextTask(Task& task, const string& baseDir, bool retryOnFailure, std::atomic<bool>& shouldStop) {
  (void)baseDir;

  auto f = [&]() {
    json response = parseJson(post("/api/tasks/","","text/plain"));
    string kind = parseString(response,"kind",32);
    if(kind == "selfplay") {
      json networkProperties = parse<json>(response,"network");
      json runProperties = parse<json>(response,"run");

      task.taskId = ""; //Not currently used by server
      task.taskGroup = parseString(networkProperties,"name",MAX_NETWORK_NAME_LEN);
      task.runName = parseString(runProperties,"name",MAX_RUN_NAME_LEN);
      task.config = parseString(response,"config",MAX_CONFIG_NAME_LEN);
      task.modelBlack = parseModelInfo(networkProperties);
      task.modelWhite = task.modelBlack;
      task.doWriteTrainingData = true;
      task.isRatingGame = false;
    }
    else if(kind == "rating") {
      json content = parse<json>(response,"content");
      json blackNetworkProperties = parse<json>(content,"black_network");
      json whiteNetworkProperties = parse<json>(content,"white_network");
      json runProperties = parse<json>(response,"run");

      string blackCreatedAt = parseString(blackNetworkProperties,"created_at",MAX_TIME_LEN);
      string whiteCreatedAt = parseString(whiteNetworkProperties,"created_at",MAX_TIME_LEN);
      //A bit hacky - we rely on the fact that the server reports these in ISO 8601 and therefore
      //lexicographic compare is correct to determine recency
      string mostRecentName;
      if(std::lexicographical_compare(blackCreatedAt.begin(),blackCreatedAt.end(),whiteCreatedAt.begin(),whiteCreatedAt.end()))
        mostRecentName = parseString(whiteNetworkProperties,"name",MAX_NETWORK_NAME_LEN);
      else
        mostRecentName = parseString(blackNetworkProperties,"name",MAX_NETWORK_NAME_LEN);

      task.taskId = ""; //Not currently used by server

      task.taskGroup = "rating_" + mostRecentName;
      task.runName = parseString(runProperties,"name",MAX_RUN_NAME_LEN);
      task.config = parseString(response,"config",MAX_CONFIG_NAME_LEN);
      task.modelBlack = parseModelInfo(blackNetworkProperties);
      task.modelWhite = parseModelInfo(whiteNetworkProperties);
      task.doWriteTrainingData = false;
      task.isRatingGame = true;
    }
    else {
      throw StringError("kind was neither 'selfplay' or 'rating' in json response: " + response.dump());
    }

    //Go ahead and try to parse most of the normal fields out of the task config, so as to catch errors early
    try {
      istringstream taskCfgIn(task.config);
      ConfigParser taskCfg(taskCfgIn);
      SearchParams baseParams = Setup::loadSingleParams(taskCfg);
      PlaySettings playSettings = PlaySettings::loadForSelfplay(taskCfg);
      (void)baseParams;
      (void)playSettings;
    }
    catch(StringError& e) {
      throw StringError(string("Error parsing task config from server: ") + e.what() + "\nConfig was:\n" + task.config);
    }
  };
  return retryLoop("getNextTask",retryOnFailure,logger,shouldStop,f);
}

//STATIC method
string Connection::getModelPath(const Client::ModelInfo& modelInfo, const string& modelDir) {
  if(modelInfo.isRandom)
    return "/dev/null";
  return modelDir + "/" + modelInfo.name + ".bin.gz";
}

bool Connection::downloadModelIfNotPresent(
  const Client::ModelInfo& modelInfo, const string& modelDir,
  bool retryOnFailure, std::atomic<bool>& shouldStop
) {
  if(modelInfo.isRandom)
    return true;

  string path = getModelPath(modelInfo,modelDir);
  string tmpPath = path + ".tmp";

  //Model already exists
  if(bfs::exists(bfs::path(path)))
    return true;

  Url url;
  try {
    url = Url::parse(modelInfo.url);
  }
  catch(const StringError& e) {
    throw StringError(string("Could not parse URL to download model: ") + e.what());
  }

  auto f = [&]() {
    size_t totalDataSize = 0;
    ofstream out(tmpPath,ios::binary);

    std::shared_ptr<httplib::Response> response = oneShotDownload(
      logger, url, [&out,&totalDataSize,&shouldStop](const char* data, size_t data_length) {
        out.write(data, data_length);
        totalDataSize += data_length;
        return !shouldStop.load();
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

    if(totalDataSize != modelInfo.bytes)
      throw StringError(
        "Model file was incompletely downloaded, only got " + Global::int64ToString(totalDataSize) +
        " bytes out of " + Global::int64ToString(modelInfo.bytes)
      );

    //TODO maybe also verify sha256 matches the one gotten from the server

    //Attempt to load the model file to verify integrity
    {
      ModelDesc* descBuf = new ModelDesc();
      ModelDesc::loadFromFileMaybeGZipped(tmpPath,*descBuf);
      delete descBuf;
    }

    //Done! Rename the file into the right place
    std::rename(tmpPath.c_str(),path.c_str());
  };
  return retryLoop("downloadModelIfNotPresent",retryOnFailure,logger,shouldStop,f);
}

bool Connection::uploadTrainingGameAndData(
  const Task& task, const FinishedGameData* gameData, const string& sgfFilePath, const string& npzFilePath,
  bool retryOnFailure, std::atomic<bool>& shouldStop
) {
  ifstream sgfIn(sgfFilePath);
  if(!sgfIn.good())
    throw IOError(string("Error: sgf file was deleted or wasn't written out for upload?") + sgfFilePath);
  string sgfContents((istreambuf_iterator<char>(sgfIn)), istreambuf_iterator<char>());
  sgfIn.close();

  ifstream npzIn(npzFilePath,ios::in|ios::binary);
  if(!npzIn.good())
    throw IOError(string("Error: npz file was deleted or wasn't written out for upload?") + npzFilePath);
  string npzContents((istreambuf_iterator<char>(npzIn)), istreambuf_iterator<char>());
  npzIn.close();

  auto f = [&]() {
    int boardSizeX = gameData->startBoard.x_size;
    int boardSizeY = gameData->startBoard.y_size;
    int handicap = (gameData->numExtraBlack > 0 ? (gameData->numExtraBlack + 1) : 0);
    double komi = gameData->startHist.rules.komi;
    string rules = gameData->startHist.rules.toJsonStringNoKomi();
    json extraMetadata;
    extraMetadata["playout_doubling_advantage"] = gameData->playoutDoublingAdvantage;
    extraMetadata["playout_doubling_advantage_pla"] = PlayerIO::playerToString(gameData->playoutDoublingAdvantagePla);
    extraMetadata["draw_equivalent_wins_for_white"] = gameData->drawEquivalentWinsForWhite;
    static_assert(FinishedGameData::NUM_MODES == 6,"");
    extraMetadata["mode"] = (
      gameData->mode == FinishedGameData::MODE_NORMAL ? "normal" :
      gameData->mode == FinishedGameData::MODE_CLEANUP_TRAINING ? "cleanup_training" :
      gameData->mode == FinishedGameData::MODE_FORK ? "fork" :
      gameData->mode == FinishedGameData::MODE_HANDICAP ? "handicap" :
      gameData->mode == FinishedGameData::MODE_SGFPOS ? "sgfpos" :
      gameData->mode == FinishedGameData::MODE_HINTPOS ? "hintpos" :
      "unknown"
    );
    string winner = gameData->endHist.winner == P_WHITE ? "W" : gameData->endHist.winner == P_BLACK ? "B" : gameData->endHist.isNoResult ? "-" : "0";
    double score = gameData->endHist.finalWhiteMinusBlackScore;
    string hasResigned = gameData->endHist.isResignation ? "true" : "false";
    string gameUid;
    {
      ostringstream o;
      o << gameData->gameHash;
      gameUid = o.str();
    }
    string run = task.runName;
    string whiteNetwork = task.modelWhite.name;
    string blackNetwork = task.modelBlack.name;

    httplib::MultipartFormDataItems items = {
      { "board_size_x", Global::intToString(boardSizeX), "", "" },
      { "board_size_y", Global::intToString(boardSizeY), "", "" },
      { "handicap", Global::intToString(handicap), "", "" },
      { "komi", Global::doubleToString(komi), "", "" },
      { "rules", rules, "", "" },
      { "extra_metadata", extraMetadata.dump(), "", "" },
      { "winner", winner, "", "" },
      { "score", Global::doubleToString(score), "", "" },
      { "resigned", hasResigned, "", "" },
      { "kg_game_uid", gameUid, "", "" },
      { "run", run, "", ""},
      { "white_network", whiteNetwork, "", ""},
      { "black_network", blackNetwork, "", ""},
      { "sgf_file", sgfContents, "", "text/plain" },
      { "unpacked_file", npzContents, "", "application/octet-stream" },
    };

    std::shared_ptr<httplib::Response> response = postMulti("/games/training/",items);

    if(response == nullptr)
      throw StringError("No response from server");
    if(response->status == 409) {
      logger->write("Server returned 409, data is uploaded already or has a key conflict, so skipping, response was: " + response->body);
    }
    if(response->status != 200 && response->status != 201 && response->status != 202) {
      ostringstream outs;
      debugPrintResponse(outs,response);
      throw StringError("Server gave response that was not status code 200 OK or 201 Created or 202 Accepted\n" + outs.str());
    }
  };
  return retryLoop("uploadTrainingGameAndData",retryOnFailure,logger,shouldStop,f);
}

bool Connection::uploadRatingGame(
  const Task& task, const FinishedGameData* gameData, const string& sgfFilePath,
  bool retryOnFailure, std::atomic<bool>& shouldStop
) {
  ifstream sgfIn(sgfFilePath);
  if(!sgfIn.good())
    throw IOError(string("Error: sgf file was deleted or wasn't written out for upload?") + sgfFilePath);
  string sgfContents((istreambuf_iterator<char>(sgfIn)), istreambuf_iterator<char>());
  sgfIn.close();

  auto f = [&]() {
    int boardSizeX = gameData->startBoard.x_size;
    int boardSizeY = gameData->startBoard.y_size;
    int handicap = (gameData->numExtraBlack > 0 ? (gameData->numExtraBlack + 1) : 0);
    double komi = gameData->startHist.rules.komi;
    string rules = gameData->startHist.rules.toJsonStringNoKomi();
    json extraMetadata = json({});
    string winner = gameData->endHist.winner == P_WHITE ? "W" : gameData->endHist.winner == P_BLACK ? "B" : gameData->endHist.isNoResult ? "-" : "0";
    double score = gameData->endHist.finalWhiteMinusBlackScore;
    string hasResigned = gameData->endHist.isResignation ? "true" : "false";
    string gameUid;
    {
      ostringstream o;
      o << gameData->gameHash;
      gameUid = o.str();
    }
    string run = task.runName;
    string whiteNetwork = task.modelWhite.name;
    string blackNetwork = task.modelBlack.name;

    httplib::MultipartFormDataItems items = {
      { "board_size_x", Global::intToString(boardSizeX), "", "" },
      { "board_size_y", Global::intToString(boardSizeY), "", "" },
      { "handicap", Global::intToString(handicap), "", "" },
      { "komi", Global::doubleToString(komi), "", "" },
      { "rules", rules, "", "" },
      { "extra_metadata", extraMetadata.dump(), "", "" },
      { "winner", winner, "", "" },
      { "score", Global::doubleToString(score), "", "" },
      { "resigned", hasResigned, "", "" },
      { "kg_game_uid", gameUid, "", "" },
      { "run", run, "", ""},
      { "white_network", whiteNetwork, "", ""},
      { "black_network", blackNetwork, "", ""},
      { "sgf_file", sgfContents, "", "text/plain" },
    };

    std::shared_ptr<httplib::Response> response = postMulti("/games/ranking_estimation/",items);

    if(response == nullptr)
      throw StringError("No response from server");
    if(response->status == 409) {
      logger->write("Server returned 409, data is uploaded already or has a key conflict, so skipping, response was: " + response->body);
    }
    if(response->status != 200 && response->status != 201 && response->status != 202) {
      ostringstream outs;
      debugPrintResponse(outs,response);
      throw StringError("Server gave response that was not status code 200 OK or 201 Created or 202 Accepted\n" + outs.str());
    }
  };
  return retryLoop("uploadRatingGame",retryOnFailure,logger,shouldStop,f);
}

#endif //BUILD_DISTRIBUTED
