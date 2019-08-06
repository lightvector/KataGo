#ifndef DATAIO_NUMPYWRITE_H_
#define DATAIO_NUMPYWRITE_H_

#include "../core/global.h"

/*
  Usage: Users should create this with the appropriate shape, then write into "data".
  Then, call prepareHeaderWithNumRows providing the actual batch size written, and
  dataIncludingHeader will contain exactly the necessary bytes representing the numpy array.
  Supported template types:
  float, double, bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
*/
template <typename T>
struct NumpyBuffer {
  T* dataIncludingHeader;
  T* data;
  int64_t headerLen;
  int64_t dataLen;
  std::vector<int64_t> shape;
  std::string dtype;
  int shapeStartByte;

  //Numpy specifies that this is must be a multiple of 64
  static const int TOTAL_HEADER_BYTES = 256;

  NumpyBuffer(const std::vector<int64_t>& shp);
  ~NumpyBuffer();

  NumpyBuffer(const NumpyBuffer&) = delete;
  NumpyBuffer& operator=(const NumpyBuffer&) = delete;

  int64_t getActualDataLen(int64_t numWriteableRows);

  //Writes the header of the buffer and returns the total size of the writeable portion of
  //the buffer, in bytes.
  //Writes the header and computes the size treating the writeable length of the leading dimension
  //of the shape to be just numRows rather than the specified size at creation time.
  //This is so that users can preallocate one buffer at the start and still write it
  //if there were not as many rows as expected ("partial batch").
  uint64_t prepareHeaderWithNumRows(int64_t numWriteableRows);

private:
  NumpyBuffer(const std::vector<int64_t>& shp, const char* dt);
};

//Simple class for writing zip-compressed data.
//No current support for reading it.
class ZipFile {
 public:
  ZipFile(const std::string& fileName);
  ~ZipFile();

  ZipFile(const ZipFile&) = delete;
  ZipFile& operator=(const ZipFile&) = delete;

  void writeBuffer(const char* nameWithinZip, void* data, uint64_t numBytes);
  void close();

  private:
  std::string fileName;
  void* file;
};

#endif  // DATAIO_NUMPYWRITE_H_
