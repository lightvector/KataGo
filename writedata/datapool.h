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

  std::function<void(const float*,size_t)> writeTrainRow;
  float* writeBuf;
  size_t writeBufSize;
  size_t writeBufCapacity;

public:
  DataPool(size_t rowWidth, size_t trainPoolMaxCapacity, size_t testPoolMaxCapacity, size_t writeBufCapacity, std::function<void(const float*,size_t)> writeTrainRow);
  ~DataPool();

  float* addNewRow(Rand& rand);
  void finishAndWriteTrainPool(Rand& rand);
  void writeTestPool(std::function<void(const float*,size_t)> writeTestRow, Rand& rand);

private:
  float* addTrainRowHelper(Rand& rand);
  void flushWriteBuf(std::function<void(const float*,size_t)> write);
  void accumWriteBuf(const float* row, std::function<void(const float*,size_t)> write);

};


#endif
