#ifndef NEURALNET_OPENCL_KERNELS_H
#define NEURALNET_OPENCL_KERNELS_H

#include "../core/global.h"

namespace OpenCLKernels {
  extern std::string conv2dNCHW;
  extern std::string scaleBiasMaskNCHW;
  extern std::string scaleBiasMaskReluNCHW;
  extern std::string addPointWise;


}





#endif //NEURALNET_OPENCL_KERNELS_H
