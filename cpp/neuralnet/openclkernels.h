#ifndef NEURALNET_OPENCL_KERNELS_H
#define NEURALNET_OPENCL_KERNELS_H

#include "../core/global.h"

namespace OpenCLKernels {
  extern std::string conv2dNCHW;
  extern std::string winogradConvNCHW;
  extern std::string scaleBiasMaskNCHW;
  extern std::string scaleBiasMaskReluNCHW;
  extern std::string addPointWise;
  extern std::string matMul;
  extern std::string matMulTransBatched;
  extern std::string sumChannelsNCHW;
  extern std::string gPoolChannelsNCHW;
  extern std::string valueHeadPoolChannelsNCHW;
  extern std::string addChannelBiasesNCHW;
  extern std::string addCBiasesNC;
  extern std::string addCBiasesNCRelu;
  extern std::string transposeNCHW;
  extern std::string mirror;
  extern std::string extractChannel0NCHW;

  extern std::string xgemmDirect;
  extern std::string xgemm;

}





#endif //NEURALNET_OPENCL_KERNELS_H
