#include <functional>
#include <cstring>
#include "../core/global.h"
#include "../core/rand.h"
#include "../dataio/datapool.h"

DataPool::DataPool(size_t rowWidth, size_t poolCapacity, size_t writeBufCapacity, std::function<void(const float*,size_t)> writeRow)
  :rowWidth(rowWidth),
   numRowsAdded(0),
   poolSize(0),
   poolCapacity(poolCapacity),
   finished(false),
   writeRow(writeRow),
   writeBufSize(0),
   writeBufCapacity(writeBufCapacity)
{
  assert(sizeof(size_t) == 8);
  pool = new float[rowWidth * poolCapacity];
  writeBuf = new float[rowWidth * writeBufCapacity];

  //Zero out everything to start
  std::memset(pool,0,sizeof(float)*rowWidth*poolCapacity);
}

DataPool::~DataPool() {
  delete[] pool;
  delete[] writeBuf;
}

void DataPool::flushWriteBuf(std::function<void(const float*,size_t)> write) {
  write(writeBuf,writeBufSize);
  writeBufSize = 0;
}

void DataPool::accumWriteBuf(const float* row, std::function<void(const float*,size_t)> write) {
  std::memcpy(&(writeBuf[rowWidth*writeBufSize]),row,sizeof(float)*rowWidth);
  writeBufSize++;
  if(writeBufSize >= writeBufCapacity)
    flushWriteBuf(write);
}

//Helper, add a new row
//Does NOT increment numRowsAdded.
//Does NOT zero out the new row
float* DataPool::addRowHelper(Rand& rand) {
  //If pool is not full, then simply add it to the next spot.
  if(poolSize < poolCapacity) {
    float* row = &(pool[rowWidth*poolSize]);
    poolSize++;
    return row;
  }
  //Otherwise, randomly evict a row.
  size_t evictIdx = (size_t)rand.nextUInt64(poolCapacity);
  float* row = &(pool[rowWidth*evictIdx]);
  accumWriteBuf(row,writeRow);
  return row;
}


float* DataPool::addNewRow(Rand& rand) {
  assert(!finished);
  numRowsAdded++;

  float* trainRow = addRowHelper(rand);
  std::memset(trainRow,0,sizeof(float)*rowWidth);
  return trainRow;
}

static void fillRandomPermutation(size_t* arr, size_t len, Rand& rand) {
  for(size_t i = 0; i<len; i++)
    arr[i] = i;
  for(size_t i = 1; i<len; i++) {
    size_t r = (size_t)rand.nextUInt64(i+1);
    size_t tmp = arr[r];
    arr[r] = arr[i];
    arr[i] = tmp;
  }
}

void DataPool::finishAndWritePool(Rand& rand) {
  assert(!finished);
  finished = true;
  //Pick indices to write in a random order
  size_t* indices = new size_t[poolSize];
  fillRandomPermutation(indices,poolSize,rand);
  for(size_t i = 0; i<poolSize; i++) {
    size_t r = indices[i];
    float* row = &(pool[rowWidth*r]);
    accumWriteBuf(row,writeRow);
  }
  flushWriteBuf(writeRow);

  delete[] indices;
}


