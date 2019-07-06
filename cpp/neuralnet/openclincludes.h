#ifndef NEURALNET_OPENCLINCLUDES_H
#define NEURALNET_OPENCLINCLUDES_H

//Ensures a consistent opencl version everywhere we include opencl
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#endif //NEURALNET_OPENCLINCLUDES_H
