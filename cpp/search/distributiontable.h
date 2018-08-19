#ifndef DISTRIBUTIONTABLE_H
#define DISTRIBUTIONTABLE_H

#include "../core/global.h"

struct DistributionTable {
  double* pdfTable;
  double* cdfTable;
  int size;
  double minZ;
  double maxZ;

  DistributionTable(function<double(double)> pdf, function<double(double)> cdf, double minZ, double maxZ, int size);
  ~DistributionTable();

  DistributionTable(const DistributionTable& other) = delete;
  DistributionTable& operator=(const DistributionTable& other) = delete;
  DistributionTable(DistributionTable&& other) = delete;
  DistributionTable& operator=(DistributionTable&& other) = delete;

  inline double getPdf(double z) const {
    double d = (size-1) * (z-minZ) / (maxZ-minZ);
    if(d <= 0)
      return 0.0;
    int idx = (int)d;
    if(idx >= size-1)
      return 0.0;
    double lambda = d - idx;
    double y0 = pdfTable[idx];
    double y1 = pdfTable[idx+1];
    return y0 + lambda * (y1 - y0);
  };

  inline double getCdf(double z) const {
    double d = (size-1) * (z-minZ) / (maxZ-minZ);
    if(d <= 0)
      return 0.0;
    int idx = (int)d;
    if(idx >= size-1)
      return 1.0;
    double lambda = d - idx;
    double y0 = cdfTable[idx];
    double y1 = cdfTable[idx+1];
    return y0 + lambda * (y1 - y0);
  };
};


#endif
