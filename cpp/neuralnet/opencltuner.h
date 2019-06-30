#ifndef NEURALNET_OPENCL_TUNER_H_
#define NEURALNET_OPENCL_TUNER_H_

#include "../core/global.h"
#include "../core/logger.h"
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
  };
  XGemmDirectParams xGemmDirect = XGemmDirectParams();
};

namespace OpenCLTuner {
  OpenCLTuneParams tune(int gpuIdx, Logger* logger);

}


#endif //NEURALNET_OPENCL_TUNER_H_
