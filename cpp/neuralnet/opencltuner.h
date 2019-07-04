#ifndef NEURALNET_OPENCL_TUNER_H_
#define NEURALNET_OPENCL_TUNER_H_

#include "../core/global.h"
#include "../core/logger.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/openclincludes.h"

struct OpenCLTuneParams {
  struct XGemmDirectParams {
    int WGD = 8;
    int MDIMCD = 8;
    int NDIMCD = 8;
    int MDIMAD = 8;
    int NDIMBD = 8;
    int KWID = 1;
    int VWMD = 1;
    int VWND = 1;
    int PADA = 1;
    int PADB = 1;
    std::string desc() const;
    std::string compileOptions() const;
    void fillFromDesc(const std::string& fileName, const std::string& desc);
    bool isValid() const;
  };
  XGemmDirectParams xGemmDirect = XGemmDirectParams();

  struct Conv3x3Params {
    //Winograd input and output tile sizes
    int INTILE_XSIZE = 4;
    int INTILE_YSIZE = 4;
    int OUTTILE_XSIZE = 2;
    int OUTTILE_YSIZE = 2;
    std::string desc() const;
    std::string compileOptions() const;
    void fillFromDesc(const std::string& fileName, const std::string& desc);
    bool isValid() const;
  };
  Conv3x3Params conv3x3 = Conv3x3Params();

  bool operator==(const OpenCLTuneParams& other) const;
  bool isValid() const;

  static void save(const std::string& filename, const OpenCLTuneParams& config);
  static OpenCLTuneParams load(const std::string& filename);
};

namespace OpenCLTuner {
  constexpr int DEFAULT_X_SIZE = NNPos::MAX_BOARD_LEN;
  constexpr int DEFAULT_Y_SIZE = NNPos::MAX_BOARD_LEN;
  constexpr int DEFAULT_BATCH_SIZE = 2;

  void tune(
    const OpenCLTuneParams& initialConfig,
    int gpuIdx,
    Logger* logger,
    int batchSize,
    int nnXLen,
    int nnYLen,
    const ModelDesc* model,
    bool full,
    std::ostream& out,
    std::function<void(const OpenCLTuneParams&)> handleBestSoFar
  );

  std::string defaultDirectory(bool makeDir);
  std::string defaultFileName(int gpuIdx, int nnXLen, int nnYLen, const ModelDesc* model);

  OpenCLTuneParams loadOrAutoTune(
    std::string openCLTunerFile,
    int gpuIdx,
    Logger* logger,
    int nnXLen,
    int nnYLen,
    const ModelDesc* model,
    bool full
  );

}


#endif //NEURALNET_OPENCL_TUNER_H_
