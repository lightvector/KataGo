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
