#ifndef DATAPOOL_H
#define DATAPOOL_H

#include <functional>
#include "../core/global.h"
#include "../core/rand.h"

class DataPool {
  size_t rowWidth;
  size_t numRowsAdded;

  float* pool;
  size_t poolSize;
  size_t poolCapacity;

  bool finished;

  std::function<void(const float*,size_t)> writeRow;
  float* writeBuf;
  size_t writeBufSize;
  size_t writeBufCapacity;

public:
  DataPool(size_t rowWidth, size_t poolMaxCapacity, size_t writeBufCapacity, std::function<void(const float*,size_t)> writeRow);
  ~DataPool();

  //No copy assignment or constructor
  DataPool(const DataPool&) = delete;
  DataPool& operator=(const DataPool&) = delete;

  float* addNewRow(Rand& rand);
  void finishAndWritePool(Rand& rand);

private:
  float* addRowHelper(Rand& rand);
  void flushWriteBuf(std::function<void(const float*,size_t)> write);
  void accumWriteBuf(const float* row, std::function<void(const float*,size_t)> write);

};


#endif
