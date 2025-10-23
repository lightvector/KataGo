#include <ctime>
#include "../dataio/loadmodel.h"

#include <ghc/filesystem.hpp>

//------------------------
#include "../core/using.h"
//------------------------

template <typename TP>
std::time_t to_time_t(TP tp)
{
  using namespace std::chrono;
  auto sctp = time_point_cast<system_clock::duration>(
    tp - TP::clock::now() + system_clock::now()
  );
  return system_clock::to_time_t(sctp);
}

static const vector<string> ACCEPTABLE_MODEL_SUFFIXES {
  ".bin.gz",
  "model.txt.gz",
  "model.txt"
};
static const vector<string> GENERIC_MODEL_NAMES {
  "model.bin.gz",
  "model.bin",
  "model.txt.gz",
  "model.txt"
  "Model.bin.gz",
  "Model.bin",
  "Model.txt.gz",
  "Model.txt"
  "MODEL.bin.gz",
  "MODEL.bin",
  "MODEL.txt.gz",
  "MODEL.txt"
  "model.ckpt",
  "Model.ckpt"
  "MODEL.ckpt",
  "model.checkpoint",
  "Model.checkpoint"
  "MODEL.checkpoint",
  "model",
  "Model"
  "MODEL",
};

static bool endsWithAnySuffix(const string& path, const vector<string>& suffixes) {
  for(const string& suffix: suffixes) {
    if(Global::isSuffix(path,suffix))
      return true;
  }
  return false;
}

bool LoadModel::findLatestModel(const string& modelsDir, Logger& logger, string& modelName, string& modelFile, string& modelDir, time_t& modelTime) {
  namespace gfs = ghc::filesystem;
  (void)logger;

  bool hasLatestTime = false;
  gfs::file_time_type latestTime;
  gfs::path latestPath;
  for(const auto& dirEntry: gfs::recursive_directory_iterator(gfs::u8path(modelsDir))) {
    gfs::path filePath = dirEntry.path();
    if(gfs::is_regular_file(filePath) && endsWithAnySuffix(filePath.filename().u8string(), ACCEPTABLE_MODEL_SUFFIXES)) {
      gfs::file_time_type thisTime = gfs::last_write_time(filePath);
      if(!hasLatestTime || thisTime > latestTime) {
        hasLatestTime = true;
        latestTime = thisTime;
        latestPath = filePath;
      }
    }
  }

  modelName = "random";
  modelFile = "/dev/null";
  modelDir = "/dev/null";
  modelTime = (std::time_t)(0);
  if(hasLatestTime) {
    modelFile = latestPath.u8string();
    modelDir = latestPath.parent_path().u8string();
    if(contains(GENERIC_MODEL_NAMES, latestPath.filename().u8string())) {
      modelName = latestPath.parent_path().filename().u8string();
    }
    else {
      modelName = latestPath.filename().u8string();
    }
    modelTime = to_time_t(latestTime);
  }

  return true;
}

void LoadModel::setLastModifiedTimeToNow(const string& filePath, Logger& logger) {
  namespace gfs = ghc::filesystem;
  gfs::path path(gfs::u8path(filePath));
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
  for(gfs::directory_iterator iter(gfs::u8path(modelsDir)); iter != gfs::directory_iterator(); ++iter) {
    gfs::path filePath = iter->path();
    if(gfs::is_directory(filePath))
      continue;
    string filePathStr = filePath.u8string();
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
    logger.write("Deleting old unused model file: " + pathsToRemove[i].u8string());
    try {
      gfs::remove(pathsToRemove[i]);
    }
    catch(gfs::filesystem_error& e) {
      logger.write("Warning: could not delete " + pathsToRemove[i].u8string() + ": " + e.what());
    }
  }

}
