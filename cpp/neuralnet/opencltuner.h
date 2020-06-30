#ifndef NEURALNET_OPENCL_TUNER_H_
#define NEURALNET_OPENCL_TUNER_H_

#include "../core/global.h"
#include "../core/logger.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/openclincludes.h"
#include "../neuralnet/openclhelpers.h"

struct OpenCLTuneParams {
  struct XGemmDirectParams {
    int WGD = 8;
    int MDIMCD = 1;
    int NDIMCD = 1;
    int MDIMAD = 1;
    int NDIMBD = 1;
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

  struct XGemmParams {
    int MWG = 8;
    int NWG = 8;
    int KWG = 8;
    int MDIMC = 1;
    int NDIMC = 1;
    int MDIMA = 1;
    int NDIMB = 1;
    int KWI = 1;
    int VWM = 1;
    int VWN = 1;
    int STRM = 0;
    int STRN = 0;
    int SA = 0;
    int SB = 0;

    std::string desc() const;
    std::string compileOptions() const;
    void fillFromDesc(const std::string& fileName, const std::string& desc);
    bool isValid() const;
    bool isSimple() const;
  };
  XGemmParams xGemm = XGemmParams();

  struct Conv3x3Params {
    //Winograd input and output tile sizes
    int INTILE_XSIZE = 4;
    int INTILE_YSIZE = 4;
    int OUTTILE_XSIZE = 2;
    int OUTTILE_YSIZE = 2;

    int transLocalSize0 = 1;
    int transLocalSize1 = 1;

    int untransLocalSize0 = 1;
    int untransLocalSize1 = 1;
    int untransLocalSize2 = 1;

    std::string desc() const;
    std::string transDesc() const;
    std::string untransDesc() const;
    std::string compileOptions() const;
    void fillFromDesc(const std::string& fileName, const std::string& desc);
    bool isValid() const;
  };
  Conv3x3Params conv3x3 = Conv3x3Params();

  struct Conv5x5Params {
    //Winograd input and output tile sizes
    int INTILE_XSIZE = 6;
    int INTILE_YSIZE = 6;
    int OUTTILE_XSIZE = 2;
    int OUTTILE_YSIZE = 2;

    int transLocalSize0 = 1;
    int transLocalSize1 = 1;

    int untransLocalSize0 = 1;
    int untransLocalSize1 = 1;
    int untransLocalSize2 = 1;

    std::string desc() const;
    std::string transDesc() const;
    std::string untransDesc() const;
    std::string compileOptions() const;
    void fillFromDesc(const std::string& fileName, const std::string& desc);
    bool isValid() const;
  };
  Conv5x5Params conv5x5 = Conv5x5Params();

  struct GPoolParams {
    int XYSTRIDE = 1;
    int CHANNELSTRIDE = 1;
    int BATCHSTRIDE = 1;

    std::string desc() const;
    std::string compileOptions() const;
    void fillFromDesc(const std::string& fileName, const std::string& desc);
    bool isValid() const;
  };
  GPoolParams gPool = GPoolParams();

  bool operator==(const OpenCLTuneParams& other) const;
  bool isValid() const;

  static void save(const std::string& filename, const OpenCLTuneParams& config);
  static OpenCLTuneParams load(const std::string& filename);
};

namespace OpenCLTuner {
  constexpr int DEFAULT_X_SIZE = NNPos::MAX_BOARD_LEN;
  constexpr int DEFAULT_Y_SIZE = NNPos::MAX_BOARD_LEN;
  constexpr int DEFAULT_BATCH_SIZE = 2;
  constexpr int DEFAULT_WINOGRAD_3X3_TILE_SIZE = 4;

  void tune(
    const OpenCLTuneParams& initialConfig,
    DevicesContext& devicesContext,
    int gpuIdx,
    int batchSize,
    int nnXLen,
    int nnYLen,
    const ModelDesc* model,
    bool full,
    int winograd3x3TileSize,
    std::ostream& out,
    std::function<void(const OpenCLTuneParams&)> handleBestSoFar
  );

  std::string defaultDirectory(bool makeDir, const std::string& homeDataDirOverride);
  std::string defaultFileName(const std::string& gpuName, int nnXLen, int nnYLen, const ModelDesc* model);

  OpenCLTuneParams loadOrAutoTune(
    std::string openCLTunerFile,
    const std::string& homeDataDirOverride,
    const std::string& gpuName,
    int gpuIdxForTuning,
    Logger* logger,
    bool openCLReTunePerBoardSize,
    int nnXLen,
    int nnYLen,
    const ModelDesc* model,
    bool full
  );

}


#endif //NEURALNET_OPENCL_TUNER_H_
