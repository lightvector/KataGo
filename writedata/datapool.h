#ifndef DATAPOOL_H
#define DATAPOOL_H

#include <functional>
#include "core/global.h"
#include "core/rand.h"

class DataPool {
  size_t rowWidth;
  size_t numRowsAdded;

  float* trainPool;
  size_t trainPoolSize;
  size_t trainPoolCapacity;

  float* testPool;
  size_t testPoolSize;
  size_t testPoolCapacity;

  bool finished;

  std::function<void(const float*)> writeTrainRow;

public:
  DataPool(int rowWidth, int trainPoolMaxCapacity, int testPoolMaxCapacity, std::function<void(const float*)> writeTrainRow);
  ~DataPool();

  float* addNewRow(Rand& rand);
  void finishAndWriteTrainPool(Rand& rand);
  void writeTestPool(std::function<void(const float*)> writeTestRow, Rand& rand);

private:
  float* addTrainRowHelper(Rand& rand);

};


#endif
