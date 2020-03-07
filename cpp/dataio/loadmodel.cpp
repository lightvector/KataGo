#include "../dataio/loadmodel.h"

#include <boost/filesystem.hpp>

using namespace std;

bool LoadModel::findLatestModel(const string& modelsDir, Logger& logger, string& modelName, string& modelFile, string& modelDir, time_t& modelTime) {
  namespace bfs = boost::filesystem;

  bool hasLatestTime = false;
  std::time_t latestTime = 0;
  bfs::path latestPath;
  for(bfs::directory_iterator iter(modelsDir); iter != bfs::directory_iterator(); ++iter) {
    bfs::path dirPath = iter->path();
    if(!bfs::is_directory(dirPath))
      continue;

    time_t thisTime = bfs::last_write_time(dirPath);
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
    if(!bfs::exists(bfs::path(modelFile))) {
      modelFile = modelsDir + "/" + modelName + "/model.txt.gz";
      if(!bfs::exists(bfs::path(modelFile))) {
        modelFile = modelsDir + "/" + modelName + "/model.bin";
        if(!bfs::exists(bfs::path(modelFile))) {
          modelFile = modelsDir + "/" + modelName + "/model.txt";
          if(!bfs::exists(bfs::path(modelFile))) {
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
  namespace bfs = boost::filesystem;
  bfs::path path(filePath);
  try {
    bfs::last_write_time(path,std::time(NULL));
  }
  catch(bfs::filesystem_error& e) {
    logger.write("Warning: could not set last modified time for " + filePath + ": " + e.what());
  }
}

void LoadModel::deleteModelsOlderThan(const string& modelsDir, Logger& logger, const time_t& time) {
  namespace bfs = boost::filesystem;
  vector<bfs::path> pathsToRemove;
  for(bfs::directory_iterator iter(modelsDir); iter != bfs::directory_iterator(); ++iter) {
    bfs::path filePath = iter->path();
    if(bfs::is_directory(filePath))
      continue;

    time_t thisTime = bfs::last_write_time(filePath);
    if(thisTime < time) {
      pathsToRemove.push_back(filePath);
    }
  }

  for(size_t i = 0; i<pathsToRemove.size(); i++) {
    logger.write("Deleting old unused model file: " + pathsToRemove[i].string());
    try {
      bfs::remove(pathsToRemove[i]);
    }
    catch(bfs::filesystem_error& e) {
      logger.write("Warning: could not delete " + pathsToRemove[i].string() + ": " + e.what());
    }
  }

}
