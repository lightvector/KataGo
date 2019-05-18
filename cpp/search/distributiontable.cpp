#include "../search/distributiontable.h"

using namespace std;

DistributionTable::DistributionTable(function<double(double)> pdf, function<double(double)> cdf, double minz, double maxz, int sz) {
  size = sz;
  minZ = minz;
  maxZ = maxz;
  pdfTable = new double[size];
  cdfTable = new double[size];

  for(int i = 0; i<size; i++) {
    if(i == 0) {
      pdfTable[i] = 0.0;
      cdfTable[i] = 0.0;
    }
    else if(i == size-1) {
      pdfTable[i] = 0.0;
      cdfTable[i] = 1.0;
    }
    else {
      double z = minZ + i * (maxZ-minZ) / (double)(size-1);
      pdfTable[i] = pdf(z);
      cdfTable[i] = cdf(z);
    }
  }
}

DistributionTable::~DistributionTable() {
  delete[] pdfTable;
  delete[] cdfTable;
}
