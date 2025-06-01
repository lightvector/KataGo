#ifndef NEURALNET_OPENCL_KERNELS_H
#define NEURALNET_OPENCL_KERNELS_H

#include "../core/global.h"

namespace OpenCLKernels {
  extern std::string fp16StorageDefine;
  extern std::string fp16ComputeDefine;
  extern std::string actIdenDefine;
  extern std::string actReluDefine;
  extern std::string actMishDefine;
  extern std::string actMishScale8Define;

  extern std::string common;
  extern std::string conv2dNCHW;
  extern std::string winogradTransformNCHW;
  extern std::string winogradBNActTransformNCHW;
  extern std::string winogradUntransformNCHW;
  extern std::string scaleBiasMaskActNCHW;
  extern std::string addPointWise;
  extern std::string matMul;
  extern std::string matMulTransBatched;
  extern std::string sumChannelsNCHW;
  extern std::string gPoolChannelsNCHWMask;
  extern std::string valueHeadPoolChannelsNCHW;
  extern std::string addChannelBiasesNCHW;
  extern std::string addCBiasesNCAct;
  extern std::string extractChannel0NCHW;

  extern std::string xgemmDirect;
  extern std::string xgemm;
  extern std::string hgemmWmma;
  extern std::string hgemmWmmaNCHW;
}





#endif //NEURALNET_OPENCL_KERNELS_H
