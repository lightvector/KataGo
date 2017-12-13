#include <functional>
#include <cstring>
#include "core/global.h"
#include "core/rand.h"
#include "datapool.h"

DataPool::DataPool(int rowWidth, int trainPoolCapacity, int testPoolCapacity, int writeBufCapacity, std::function<void(const float*,size_t)> writeTrainRow)
  :rowWidth(rowWidth),
   numRowsAdded(0),
   trainPoolSize(0),
   trainPoolCapacity(trainPoolCapacity),
   testPoolSize(0),
   testPoolCapacity(testPoolCapacity),
   finished(false),
   writeTrainRow(writeTrainRow),
   writeBufSize(0),
   writeBufCapacity(writeBufCapacity)
{
  assert(sizeof(size_t) == 8);
  trainPool = new float[rowWidth * trainPoolCapacity];
  testPool = new float[rowWidth * testPoolCapacity];
  writeBuf = new float[rowWidth * writeBufCapacity];

  //Zero out everything to start
  std::memset(trainPool,0,sizeof(float)*rowWidth*trainPoolCapacity);
  std::memset(testPool,0,sizeof(float)*rowWidth*testPoolCapacity);
}

DataPool::~DataPool() {
  delete[] trainPool;
  delete[] testPool;
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

//Helper, add a new row and definitely put it in the training set, don't check the test set.
//Does NOT increment numRowsAdded.
//Does NOT zero out the new row
float* DataPool::addTrainRowHelper(Rand& rand) {
  //If training is not full, then simply add it to the next spot.
  if(trainPoolSize < trainPoolCapacity) {
    float* row = &(trainPool[rowWidth*trainPoolSize]);
    trainPoolSize++;
    return row;
  }
  //Otherwise, randomly evict a row.
  size_t trainIdx = (size_t)rand.nextUInt64(trainPoolCapacity);
  assert(trainIdx >= 0);
  float* row = &(trainPool[rowWidth*trainIdx]);
  accumWriteBuf(row,writeTrainRow);
  return row;
}


float* DataPool::addNewRow(Rand& rand) {
  assert(!finished);
  numRowsAdded++;

  //Reservoir sample to figure out whether to add it to train or test.
  //If test is not full, definitely add it to test.
  if(testPoolSize < testPoolCapacity) {
    float* ret = &(testPool[rowWidth*testPoolSize]);
    testPoolSize++;
    assert(numRowsAdded == testPoolSize);
    return ret;
  }

  //Otherwise, see if we should evict a spot to add a new test row.
  size_t testIdx = (size_t)rand.nextUInt64(numRowsAdded);
  assert(testIdx >= 0);
  if(testIdx < testPoolCapacity) {
    //Evict!
    float* testRow = &(testPool[rowWidth*testIdx]);
    //Grab a new training row, possibly evicting a training row to make space
    float* trainRow = addTrainRowHelper(rand);
    std::memcpy(trainRow,testRow,sizeof(float)*rowWidth);

    //Zero out and return the new test row.
    std::memset(testRow,0,sizeof(float)*rowWidth);
    return testRow;
  }

  //Not adding a new test row, so just add a new training row
  float* trainRow = addTrainRowHelper(rand);
  std::memset(trainRow,0,sizeof(float)*rowWidth);
  return trainRow;
}

static void fillRandomPermutation(size_t* arr, size_t len, Rand& rand) {
  for(size_t i = 0; i<len; i++)
    arr[i] = i;
  for(size_t i = 1; i<len; i++) {
    size_t r = (size_t)rand.nextUInt64(i+1);
    assert(r >= 0);
    size_t tmp = arr[r];
    arr[r] = arr[i];
    arr[i] = tmp;
  }
}

void DataPool::finishAndWriteTrainPool(Rand& rand) {
  assert(!finished);
  finished = true;
  //Pick indices to write in a random order
  size_t* indices = new size_t[trainPoolCapacity];
  fillRandomPermutation(indices,trainPoolCapacity,rand);
  for(size_t i = 0; i<trainPoolCapacity; i++) {
    size_t r = indices[i];
    float* row = &(trainPool[rowWidth*r]);
    accumWriteBuf(row,writeTrainRow);
  }
  flushWriteBuf(writeTrainRow);

  delete[] indices;
}

void DataPool::writeTestPool(std::function<void(const float*,size_t)> writeTestRow, Rand& rand) {
  assert(finished);
  assert(writeBufSize == 0);
  //Pick indices to write in a random order
  size_t* indices = new size_t[testPoolCapacity];
  fillRandomPermutation(indices,testPoolCapacity,rand);
  for(size_t i = 0; i<testPoolCapacity; i++) {
    size_t r = indices[i];
    float* row = &(testPool[rowWidth*r]);
    accumWriteBuf(row,writeTestRow);
  }
  flushWriteBuf(writeTestRow);
  delete[] indices;
}

