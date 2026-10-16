#include "../neuralnet/debugprint.h"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

using namespace std;

// Helper: compute summary stats over elements where mask is valid.
// If mask is null, all elements are valid.
static void computeStats(
  const float* data, int totalSize, const float* mask,
  int& validCount, float& minVal, float& maxVal, double& sum, double& sumSq
) {
  validCount = 0;
  minVal = 1e30f;
  maxVal = -1e30f;
  sum = 0.0;
  sumSq = 0.0;

  for(int i = 0; i < totalSize; i++) {
    if(mask != nullptr) {
      if(mask[i] == 0.0f)
        continue;
    }
    float v = data[i];
    validCount++;
    minVal = std::min(minVal, v);
    maxVal = std::max(maxVal, v);
    sum += (double)v;
    sumSq += (double)v * (double)v;
  }
  if(validCount == 0) {
    minVal = 0.0f;
    maxVal = 0.0f;
  }
}

static void printStatsLine(
  const string& name, const string& shapeStr,
  int validCount, float minVal, float maxVal, double sum, double sumSq,
  const float* data, int totalSize, int numToPrint
) {
  double mean = (validCount > 0) ? sum / validCount : 0.0;
  double rms = (validCount > 0) ? sqrt(sumSq / validCount) : 0.0;

  fprintf(stderr, "DEBUG %s %s valid=%d min=%.6g max=%.6g mean=%.6g rms=%.6g",
    name.c_str(), shapeStr.c_str(),
    validCount, minVal, maxVal, mean, rms);

  int n = std::min(numToPrint, totalSize);
  if(n > 0) {
    fprintf(stderr, " first%d=", n);
    for(int i = 0; i < n; i++)
      fprintf(stderr, " %.6g", data[i]);
  }
  fprintf(stderr, "\n");
}

void DebugPrint::printSummary(
  const string& name, const float* data, int totalSize,
  const float* mask, int numToPrint
) {
  int validCount;
  float minVal, maxVal;
  double sum, sumSq;
  computeStats(data, totalSize, mask, validCount, minVal, maxVal, sum, sumSq);

  char shapeStr[64];
  snprintf(shapeStr, sizeof(shapeStr), "[%d]", totalSize);
  printStatsLine(name, shapeStr, validCount, minVal, maxVal, sum, sumSq, data, totalSize, numToPrint);
}

void DebugPrint::print3DSummary(
  const string& name, const float* data,
  int dim0, int dim1, int dim2, const string& dimOrder,
  int nSize, int spatialSize, const float* mask,
  int numToPrint
) {
  int totalSize = dim0 * dim1 * dim2;
  char shapeStr[128];
  snprintf(shapeStr, sizeof(shapeStr), "[%s %dx%dx%d]", dimOrder.c_str(), dim0, dim1, dim2);

  // Build a flat mask array [totalSize] from the spatial mask [nSize, spatialSize].
  // We need to figure out which elements correspond to valid spatial positions.
  // Since we don't reorder, we just expand the mask to cover all channels at each spatial position.
  int validCount;
  float minVal, maxVal;
  double sum, sumSq;

  if(mask == nullptr) {
    computeStats(data, totalSize, nullptr, validCount, minVal, maxVal, sum, sumSq);
  }
  else {
    // Expand mask to flat array covering all elements
    // For any dim ordering, an element at flat index i corresponds to spatial position:
    //   For NCS (NCHW-like): n = i/(dim1*dim2), s = i%dim2, and mask[n*spatialSize + s]
    //   For NSC (NHWC-like): n = i/(dim1*dim2), s = (i/dim2)%dim1, and mask[n*spatialSize + s]
    // But we need to know the ordering to do this correctly.
    // Since the whole point is ordering-invariant stats, let's build a flat mask.
    vector<float> flatMask(totalSize);
    int cSize = totalSize / (nSize * spatialSize);

    if(dimOrder.size() >= 3 && dimOrder[1] == 'S') {
      // NSC ordering: dim0=N, dim1=S, dim2=C
      for(int n = 0; n < nSize; n++)
        for(int s = 0; s < spatialSize; s++) {
          float m = mask[n * spatialSize + s];
          for(int c = 0; c < cSize; c++)
            flatMask[(n * spatialSize + s) * cSize + c] = m;
        }
    }
    else if(dimOrder.size() >= 3 && dimOrder[1] == 'C') {
      // NCS ordering: dim0=N, dim1=C, dim2=S
      for(int n = 0; n < nSize; n++)
        for(int c = 0; c < cSize; c++)
          for(int s = 0; s < spatialSize; s++)
            flatMask[(n * cSize + c) * spatialSize + s] = mask[n * spatialSize + s];
    }
    else {
      // Unknown ordering, treat all as valid
      for(int i = 0; i < totalSize; i++)
        flatMask[i] = 1.0f;
    }
    computeStats(data, totalSize, flatMask.data(), validCount, minVal, maxVal, sum, sumSq);
  }

  printStatsLine(name, shapeStr, validCount, minVal, maxVal, sum, sumSq, data, totalSize, numToPrint);
}

void DebugPrint::print2DSummary(
  const string& name, const float* data,
  int nSize, int cSize, int numToPrint
) {
  int totalSize = nSize * cSize;
  char shapeStr[64];
  snprintf(shapeStr, sizeof(shapeStr), "[NC %dx%d]", nSize, cSize);

  int validCount;
  float minVal, maxVal;
  double sum, sumSq;
  computeStats(data, totalSize, nullptr, validCount, minVal, maxVal, sum, sumSq);
  printStatsLine(name, shapeStr, validCount, minVal, maxVal, sum, sumSq, data, totalSize, numToPrint);
}

void DebugPrint::printVerbose(const string& name, const float* data, int totalSize) {
  cerr << "=========================================================" << endl;
  cerr << name << " [" << totalSize << "]" << endl;
  cerr << setprecision(8);
  for(int i = 0; i < totalSize; i++) {
    cerr << data[i] << " ";
    if((i + 1) % 16 == 0) cerr << endl;
  }
  if(totalSize % 16 != 0) cerr << endl;
  cerr << "=========================================================" << endl;
}

void DebugPrint::print3DVerbose(
  const string& name, const float* data,
  int dim0, int dim1, int dim2, const string& dimOrder
) {
  cerr << "=========================================================" << endl;
  cerr << name << " [" << dimOrder << " " << dim0 << "x" << dim1 << "x" << dim2 << "]" << endl;
  cerr << setprecision(8);
  int i = 0;
  for(int d0 = 0; d0 < dim0; d0++) {
    cerr << "-(" << dimOrder[0] << "=" << d0 << ")--------------------" << endl;
    for(int d1 = 0; d1 < dim1; d1++) {
      for(int d2 = 0; d2 < dim2; d2++) {
        cerr << data[i++] << " ";
      }
      cerr << endl;
    }
    cerr << endl;
  }
  cerr << "=========================================================" << endl;
}

void DebugPrint::print2DVerbose(const string& name, const float* data, int nSize, int cSize) {
  cerr << "=========================================================" << endl;
  cerr << name << " [NC " << nSize << "x" << cSize << "]" << endl;
  cerr << setprecision(8);
  int i = 0;
  for(int n = 0; n < nSize; n++) {
    cerr << "-(n=" << n << ")--------------------" << endl;
    for(int c = 0; c < cSize; c++)
      cerr << data[i++] << " ";
    cerr << endl;
  }
  cerr << endl;
  cerr << "=========================================================" << endl;
}
