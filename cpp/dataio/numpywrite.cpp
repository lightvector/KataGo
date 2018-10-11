
#include "../dataio/numpywrite.h"
#include <cstring>
#include <zip.h>

#if !defined(BYTE_ORDER) || (BYTE_ORDER != LITTLE_ENDIAN && BYTE_ORDER != BIG_ENDIAN)
#error Define BYTE_ORDER to be equal to either LITTLE_ENDIAN or BIG_ENDIAN
#endif

template <>
NumpyBuffer<uint8_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"|u1")
{}
template <>
NumpyBuffer<int8_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"|i1")
{}
template <>
NumpyBuffer<bool>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"|b1")
{}

#if BYTE_ORDER == LITTLE_ENDIAN
template <>
NumpyBuffer<float>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<f4")
{}
template <>
NumpyBuffer<double>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<f8")
{}
template <>
NumpyBuffer<uint16_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<u2")
{}
template <>
NumpyBuffer<int16_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<i2")
{}
template <>
NumpyBuffer<uint32_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<u4")
{}
template <>
NumpyBuffer<int32_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<i4")
{}
template <>
NumpyBuffer<uint64_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<u8")
{}
template <>
NumpyBuffer<int64_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,"<i8")
{}
#else
template <>
NumpyBuffer<float>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">f4")
{}
template <>
NumpyBuffer<double>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">f8")
{}
template <>
NumpyBuffer<uint16_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">u2")
{}
template <>
NumpyBuffer<int16_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">i2")
{}
template <>
NumpyBuffer<uint32_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">u4")
{}
template <>
NumpyBuffer<int32_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">i4")
{}
template <>
NumpyBuffer<uint64_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">u8")
{}
template <>
NumpyBuffer<int64_t>::NumpyBuffer(const vector<int64_t>& shp)
  : NumpyBuffer(shp,">i8")
{}
#endif

template <typename T>
NumpyBuffer<T>::NumpyBuffer(const vector<int64_t>& shp, const char* dt)
  : shape(shp),dtype(dt)
{
  dataLen = 1;
  assert(shape.size() > 0);
  for(size_t i = 0; i<shape.size(); i++) {
    assert(shape[i] >= 0);
    if((uint64_t)dataLen * (uint64_t)shape[i] < (uint64_t)dataLen)
      throw StringError("NumpyBuffer shape overflows");
    dataLen *= shape[i];
  }

  //Leave 256 bytes at the start for the header
  int sizeOfT = sizeof(T);
  assert(sizeOfT > 0 && sizeOfT <= TOTAL_HEADER_BYTES);

  headerLen = TOTAL_HEADER_BYTES / sizeOfT;
  assert(headerLen * sizeOfT == TOTAL_HEADER_BYTES);

  dataIncludingHeader = new T[headerLen+dataLen];
  data = dataIncludingHeader + headerLen;

  //Go ahead and write all the magic bytes and such
  char* s = (char*)dataIncludingHeader;
  s[0] = 0x93;
  s[1] = 'N';
  s[2] = 'U';
  s[3] = 'M';
  s[4] = 'P';
  s[5] = 'Y';
  s[6] = 0x1;
  s[7] = 0x0;
  //Remaining bytes in header = 246 past these two bytes.
  s[8] = (char)((TOTAL_HEADER_BYTES - 10) & 0xFF);
  s[9] = (char)((TOTAL_HEADER_BYTES - 10) >> 8);

  string dictStrFirstHalf = Global::strprintf(
    "{'descr':'%s','fortran_order':False,'shape':(",
    dt
  );

  if(dictStrFirstHalf.size() > TOTAL_HEADER_BYTES - 40)
    throw StringError("Numpy header dict is too long, datatype string is too long: " + string(dt));
  strcpy(s+10, dictStrFirstHalf.c_str());

  //Record where we should start writing the shape and finish off the dict
  shapeStartByte = dictStrFirstHalf.size() + 10;
}

template <typename T>
NumpyBuffer<T>::~NumpyBuffer() {
  delete[] dataIncludingHeader;
}

//Writes the header of the buffer and returns the total size of the writeable portion of
//the buffer, in bytes.
//Writes the header and computes the size treating the writeable length of the leading dimension
//of the shape to be just numRows rather than the specified size at creation time.
//This is so that users can preallocate one buffer at the start and still write it
//if there were not as many rows as expected ("partial batch").
template <typename T>
uint64_t NumpyBuffer<T>::prepareHeaderWithNumRows(int64_t numWriteableRows) {
  //Continue writing the shape
  int idx = shapeStartByte;
  char* s = (char*)dataIncludingHeader;

  //Write each number
  int64_t actualDataLen = 1;
  for(size_t i = 0; i<shape.size(); i++) {
    if(i > 0) {
      s[idx] = ',';
      idx += 1;
      if(idx >= TOTAL_HEADER_BYTES)
        throw StringError("Numpy header is too long, datatype and shape are too long");
    }

    int64_t x = (i == 0) ? numWriteableRows : shape[i];
    actualDataLen *= x;

    int numDigits = 0;
    char digitsRev[32];
    if(x == 0) {
      digitsRev[0] = '0';
      numDigits = 1;
    }
    else {
      while(x > 0) {
        digitsRev[numDigits++] = '0' + (x % 10);
        x /= 10;
      }
    }

    for(int j = numDigits-1; j >= 0; j--) {
      s[idx] = digitsRev[j];
      idx += 1;
      if(idx >= TOTAL_HEADER_BYTES)
        throw StringError("Numpy header is too long, datatype and shape are too long");
    }
  }
  //Finish
  s[idx] = ')'; //close tuple for shape
  idx += 1;
  if(idx >= TOTAL_HEADER_BYTES)
    throw StringError("Numpy header is too long, datatype and shape are too long");
  s[idx] = '}'; //close dict literal
  idx += 1;
  if(idx >= TOTAL_HEADER_BYTES)
    throw StringError("Numpy header is too long, datatype and shape are too long");

  //Pad with spaces
  while(idx < TOTAL_HEADER_BYTES-1) {
    s[idx] = ' ';
    idx += 1;
  }
  s[idx] = '\n'; //newline, as specified by numpy
  idx += 1;

  return (uint64_t)(TOTAL_HEADER_BYTES + actualDataLen * sizeof(T));
}

template struct NumpyBuffer<float>;
template struct NumpyBuffer<double>;
template struct NumpyBuffer<bool>;
template struct NumpyBuffer<uint8_t>;
template struct NumpyBuffer<uint16_t>;
template struct NumpyBuffer<uint32_t>;
template struct NumpyBuffer<uint64_t>;
template struct NumpyBuffer<int8_t>;
template struct NumpyBuffer<int16_t>;
template struct NumpyBuffer<int32_t>;
template struct NumpyBuffer<int64_t>;

struct ZipError {
  zip_error_t value;
  ZipError() { zip_error_init(&value); }
  ~ZipError() { zip_error_fini(&value); }
  ZipError(const ZipError&) = delete;
  ZipError& operator=(const ZipError&) = delete;
};

ZipFile::ZipFile(const string& fName)
  :fileName(fName),file(NULL)
{
  ZipError zipError;
  zip_source_t* zipFileSource = zip_source_file_create(fileName.c_str(),0,-1,&(zipError.value));
  if(zipFileSource == NULL)
    throw StringError("Could not open zip file " + fileName + " due to error " + zip_error_strerror(&(zipError.value)));
  zip_t* fileHandle = zip_open_from_source(zipFileSource, ZIP_CREATE | ZIP_TRUNCATE, &(zipError.value));
  file = fileHandle;
  if(file == NULL) {
    zip_source_free(zipFileSource);
    throw StringError("Could not open zip file " + fileName + " due to error " + zip_error_strerror(&(zipError.value)));
  }
}

ZipFile::~ZipFile() {
  if(file != NULL)
    zip_discard((zip_t*)file);
}

void ZipFile::writeBuffer(const char* nameWithinZip, void* data, uint64_t numBytes) {
  ZipError zipError;
  zip_source_t* dataSource = zip_source_buffer((zip_t*)file,data,numBytes,0);
  if(dataSource == NULL)
    throw StringError(
      "Could not initialize zip write data buffer for " + string(nameWithinZip) +
      " within " + fileName + " due to error " + zip_error_strerror(&(zipError.value))
    );

  zip_int64_t idx = zip_file_add((zip_t*)file, nameWithinZip, dataSource, ZIP_FL_OVERWRITE);
  if(idx < 0) {
    zip_source_free(dataSource);
    throw StringError(
      "Could not write to " + string(nameWithinZip) +
      " within zip file " + fileName + " due to error " + zip_strerror((zip_t*)file)
    );
  }
}

void ZipFile::close() {
  int result = zip_close((zip_t*)file);
  if(result < 0)
    throw StringError("Could not close zip file " + fileName + " due to error " + zip_strerror((zip_t*)file));
  else
    file = NULL;
}


// void test() {
//   string fileName = "abc.npz";

//   NumpyBuffer<float> np({4,3,4});
//   for(int i = 0; i<2*3*4; i++)
//     np.data[i] = 0.1*i;

//   uint64_t npBytes = np.prepareHeaderWithNumRows(2);

//   ZipFile zipFile(fileName);
//   zipFile.writeBuffer("nptest",np.dataIncludingHeader,npBytes);
//   zipFile.close();
// }
