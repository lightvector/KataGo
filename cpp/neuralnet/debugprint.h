#ifndef NEURALNET_DEBUGPRINT_H_
#define NEURALNET_DEBUGPRINT_H_

#include <string>

// Shared debug printing utilities for neural net backends.
// All functions operate on host-side float data. Each backend is responsible
// for copying device data to host before calling these.
//
// Summary functions print one-line stats (min/max/mean/rms) over mask-valid
// positions, plus the first few raw values. These are ordering-invariant so
// NCHW vs NHWC backends produce identical stats for the same logical tensor.
//
// Verbose functions print every element. They are gated by
// DEBUG_INTERMEDIATE_VALUES_VERBOSE - summary is the default.

namespace DebugPrint {

  // Summary stats over a flat buffer.
  // mask is optional [totalSize] - only elements where mask[i] != 0 are counted.
  // numToPrint: how many of the first raw values to show.
  void printSummary(const std::string& name, const float* data, int totalSize,
                    const float* mask = nullptr, int numToPrint = 8);

  // 3D spatial summary: tensor with dims [dim0, dim1, dim2].
  // dimOrder: label string like "NCS" or "NSC" for the print header (describes dim0,dim1,dim2).
  // nSize, spatialSize: logical batch and spatial sizes for mask indexing.
  //   mask shape [nSize, spatialSize], nullptr = all valid.
  // Stats are computed over all elements at mask-valid spatial positions.
  void print3DSummary(const std::string& name, const float* data,
                      int dim0, int dim1, int dim2, const std::string& dimOrder,
                      int nSize, int spatialSize, const float* mask = nullptr,
                      int numToPrint = 8);

  // 2D summary (non-spatial, e.g. pooled values): [nSize, cSize].
  void print2DSummary(const std::string& name, const float* data,
                      int nSize, int cSize, int numToPrint = 8);

  // Verbose: dump all elements of a flat buffer.
  void printVerbose(const std::string& name, const float* data, int totalSize);

  // Verbose 3D: dump with dim labels.
  void print3DVerbose(const std::string& name, const float* data,
                      int dim0, int dim1, int dim2, const std::string& dimOrder);

  // Verbose 2D: dump [nSize, cSize].
  void print2DVerbose(const std::string& name, const float* data,
                      int nSize, int cSize);
}

#endif  // NEURALNET_DEBUGPRINT_H_
