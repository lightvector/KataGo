#include <ctime>
#include "../dataio/loadmodel.h"

#include "../external/filesystem-1.3.6/include/ghc/filesystem.hpp"

using namespace std;

template <typename TP>
std::time_t to_time_t(TP tp)
{
  using namespace std::chrono;
  auto sctp = time_point_cast<system_clock::duration>(
    tp - TP::clock::now() + system_clock::now()
  );
  return system_clock::to_time_t(sctp);
}

bool LoadModel::findLatestModel(const string& modelsDir, Logger& logger, string& modelName, string& modelFile, string& modelDir, time_t& modelTime) {
  namespace gfs = ghc::filesystem;

  bool hasLatestTime = false;
  std::time_t latestTime = 0;
  gfs::path latestPath;
  for(gfs::directory_iterator iter(modelsDir); iter != gfs::directory_iterator(); ++iter) {
    gfs::path dirPath = iter->path();
    if(!gfs::is_directory(dirPath))
      continue;

    time_t thisTime = to_time_t(gfs::last_write_time(dirPath));
    if(!hasLatestTime || thisTime > latestTime) {
      hasLatestTime = true;
      latestTime = thisTime;
      latestPath = dirPath;
    }
  }

  modelName = "random";
  modelFile = "/dev/null";
  modelDir = "/dev/null";
  if(hasLatestTime) {
    modelName = latestPath.filename().string();
    modelDir = modelsDir + "/" + modelName;
    modelFile = modelsDir + "/" + modelName + "/model.bin.gz";
    if(!gfs::exists(gfs::path(modelFile))) {
      modelFile = modelsDir + "/" + modelName + "/model.txt.gz";
      if(!gfs::exists(gfs::path(modelFile))) {
        modelFile = modelsDir + "/" + modelName + "/model.bin";
        if(!gfs::exists(gfs::path(modelFile))) {
          modelFile = modelsDir + "/" + modelName + "/model.txt";
          if(!gfs::exists(gfs::path(modelFile))) {
            logger.write("Warning: Skipping model " + modelName + " due to not finding model.{bin,txt} or model.{bin,txt}.gz");
            return false;
          }
        }
      }
    }
  }
  modelTime = latestTime;

  return true;
}

void LoadModel::setLastModifiedTimeToNow(const string& filePath, Logger& logger) {
  namespace gfs = ghc::filesystem;
  gfs::path path(filePath);
  try {
    gfs::last_write_time(path, gfs::file_time_type::clock::now());
  }
  catch(gfs::filesystem_error& e) {
    logger.write("Warning: could not set last modified time for " + filePath + ": " + e.what());
  }
}

void LoadModel::deleteModelsOlderThan(const string& modelsDir, Logger& logger, const time_t& time) {
  namespace gfs = ghc::filesystem;
  vector<gfs::path> pathsToRemove;
  for(gfs::directory_iterator iter(modelsDir); iter != gfs::directory_iterator(); ++iter) {
    gfs::path filePath = iter->path();
    if(gfs::is_directory(filePath))
      continue;
    string filePathStr = filePath.string();
    if(Global::isSuffix(filePathStr,".bin.gz") ||
       Global::isSuffix(filePathStr,".txt.gz") ||
       Global::isSuffix(filePathStr,".bin") ||
       Global::isSuffix(filePathStr,".txt")) {
      time_t thisTime = to_time_t(gfs::last_write_time(filePath));
      if(thisTime < time) {
        pathsToRemove.push_back(filePath);
      }
    }
  }

  for(size_t i = 0; i<pathsToRemove.size(); i++) {
    logger.write("Deleting old unused model file: " + pathsToRemove[i].string());
    try {
      gfs::remove(pathsToRemove[i]);
    }
    catch(gfs::filesystem_error& e) {
      logger.write("Warning: could not delete " + pathsToRemove[i].string() + ": " + e.what());
    }
  }

}
