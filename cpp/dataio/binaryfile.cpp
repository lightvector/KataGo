#include "../dataio/binaryfile.h"

#include <blosc.h>
#include <cstring>

static bool initialized = false;

void BinaryFile::init() {
  blosc_init();
  int result = blosc_set_compressor("zstd");
  assert(result >= 0);
  initialized = true;
}
void BinaryFile::finish() {
  blosc_destroy();
}

BinaryFile::BinaryFile(const string& dataFile, const string& metaFile, int compressionLevel)
  :compressionLevel(compressionLevel)
{
  dataStream = new std::fstream(dataFile.c_str(), std::ios::out | std::ios::binary);
  metaStream = new ofstream(metaFile);

  outBuf = NULL;
  outBufLen = 0;

  compressedIndex = 0;
}

BinaryFile::~BinaryFile() {
  dataStream->close();
  metaStream->close();

  delete dataStream;
  delete metaStream;
  delete[] outBuf;
}

/* Obtain a size of buffer guaranteed not to fail */
size_t BinaryFile::updateBufferSize(size_t nrows, size_t ncols, size_t sizeOfElt) {
  size_t size = nrows * ncols * sizeOfElt + BLOSC_MAX_OVERHEAD;
  if(outBuf == NULL || size > outBufLen) {
    if(outBuf != NULL)
      delete[] outBuf;
    outBuf = new char[size];
    outBufLen = size;
  }
}

void BinaryFile::write(const void* buf, size_t nrows, size_t ncols, size_t sizeOfElt) {
  size_t size = nrows * ncols * sizeOfElt;
  if(size <= 0)
    return;

  size_t compressedSize;
  if(compressionLevel <= 0) {
    dataStream->write((char*)buf, size);
    compressedSize = size;
  }
  else {
    assert(size < (size_t)(1 << 31) - BLOSC_MAX_OVERHEAD);
    updateBufferSize(nrows,ncols,sizeOfElt);

    int doshuffle = BLOSC_BITSHUFFLE;
    int typesize = (int)sizeOfElt;
    int result = blosc_compress(compressionLevel,doshuffle,typesize,size,buf,outBuf,outBufLen);
    assert(result > 0);

    compressedSize = (size_t)result;
    dataStream->write((char*)outBuf, compressedSize);
  }

  dataStream->flush();
  (*metaStream) << nrows << " " << ncols << " " << sizeOfElt << " " << compressionLevel << " " << compressedIndex << " " << compressedSize << "\n";
  metaStream->flush();
  compressedIndex += compressedSize;
}


BinaryFileAutoChunking::BinaryFileAutoChunking(const string& dataFile, const string& metaFile, int compressionLevel, size_t nrows, size_t ncols, size_t sizeOfElt)
  :nrows(nrows),ncols(ncols),sizeOfElt(sizeOfElt)
{
  rowLen = ncols * sizeOfElt;
  buf = new char[nrows * ncols * sizeOfElt];
  curNumRows = 0;
  binaryFile = new BinaryFile(dataFile,metaFile,compressionLevel);
}

BinaryFileAutoChunking::~BinaryFileAutoChunking() {
  if(curNumRows > 0) {
    binaryFile->write(buf,curNumRows,ncols,sizeOfElt);
    curNumRows = 0;
  }

  delete[] buf;
  delete binaryFile;
}

void BinaryFileAutoChunking::writeRow(const void* row, size_t nc, size_t soe) {
  assert(curNumRows >= 0 && curNumRows < nrows);
  assert(ncols = nc);
  assert(sizeOfElt == soe);

  std::memcpy(buf + (curNumRows * rowLen), row, rowLen);

  curNumRows++;
  if(curNumRows >= nrows) {
    binaryFile->write(buf,nrows,ncols,sizeOfElt);
    curNumRows = 0;
  }
}


