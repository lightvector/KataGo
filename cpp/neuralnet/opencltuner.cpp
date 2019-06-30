#include "../neuralnet/opencltuner.h"

OpenCLTuneParams OpenCLTuner::tune(int gpuIdx, Logger* logger) {
  OpenCLTuneParams params;

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

  return params;
}
