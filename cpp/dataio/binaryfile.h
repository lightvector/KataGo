#ifndef BINARYFILE_H
#define BINARYFILE_H

#include <fstream>
#include "../core/global.h"

/*
  Write raw binary data to disk.

  dataFile format:
  Consists of concatenated chunks of raw binary data
  Each chunk is an (optionally compressed via blosc) c-layout matrix of shape [nrows][ncols] where each entry has sizeOfElt bytes.
  (for integer elements, endianness is whatever it was on the writing machine)

  metaFile format:
  Consists of ascii lines of "{nrows} {ncols} {sizeofElt} {compressionLevel} {starting byte} {compressed byte len}" indicating the shape of each
  chunk and where it's located in the dataFile.

  NOTE: Currently there is a limitation in blosc that prevents chunks from being larger than approximately 2^31 pre-compressed.
*/

class BinaryFile {
 public:
  /* Global initialization - Call this prior to any other functions here */
  static void init();
  /* Global cleanup - Call at the end of the program to clean up */
  static void finish();

  BinaryFile(const string& dataFile, const string& metaFile, int compressionLevel);
  ~BinaryFile();

  BinaryFile(const BinaryFile&) = delete;
  BinaryFile& operator=(const BinaryFile&) = delete;

  /* Write one chunk of binary data compressed using blosc compression. */
  void write(const void* buf, size_t nrows, size_t ncols, size_t sizeOfElt);

 private:
  size_t updateBufferSize(size_t nrows, size_t ncols, size_t sizeOfElt);

  int compressionLevel;
  size_t compressedIndex;
  char* outBuf;
  size_t outBufLen;
  fstream* dataStream;
  ofstream* metaStream;
};

//Auto-chunking version of BinaryFile
class BinaryFileAutoChunking {

 public:
  BinaryFileAutoChunking(const string& dataFile, const string& metaFile, int compressionLevel, size_t nrows, size_t ncols, size_t sizeOfElt);
  ~BinaryFileAutoChunking();

  //No copy assignment or constructor
  BinaryFileAutoChunking(const BinaryFileAutoChunking&) = delete;
  BinaryFileAutoChunking& operator=(const BinaryFileAutoChunking&) = delete;

  void writeRow(const void* row, size_t ncols, size_t sizeOfElt);

 private:
  size_t nrows;
  size_t ncols;
  size_t sizeOfElt;
  size_t rowLen;
  char* buf;
  size_t curNumRows;
  BinaryFile* binaryFile;
};

#endif
