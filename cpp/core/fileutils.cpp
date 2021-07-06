#include "../core/fileutils.h"

#include <fstream>
#include <zlib.h>

#include "../core/global.h"
#include "../core/sha2.h"
#include <ghc/filesystem.hpp>
namespace gfs = ghc::filesystem;
using namespace std;

void FileUtils::loadFileIntoString(const string& filename, const string& expectedSha256, string& str) {
  #ifdef _WIN32
    std::wstring wfilename = gfs::detail::fromUtf8<std::wstring>(filename);
    gfs::ifstream in(wfilename, ios::in | ios::binary | ios::ate);
  #else
    ifstream in(filename.c_str(), ios::in | ios::binary | ios::ate);
  #endif
  if(!in.good())
    throw StringError("Could not open file - does not exist or invalid permissions?");

  ifstream::pos_type fileSize = in.tellg();
  if(fileSize < 0)
    throw StringError("tellg failed to determine size");

  in.seekg(0, ios::beg);
  str.resize(fileSize);
  in.read(&str[0], fileSize);
  in.close();

  if(expectedSha256 != "") {
    char hashResultBuf[65];
    SHA2::get256((const uint8_t*)str.data(), str.size(), hashResultBuf);
    string hashResult(hashResultBuf);
    bool matching = Global::toLower(expectedSha256) == Global::toLower(hashResult);
    if(!matching)
      throw StringError("File " + filename + " sha256 was " + hashResult + " which does not match the expected sha256 " + expectedSha256);
  }
}

void FileUtils::uncompressAndLoadFileIntoString(const string& filename, const string& expectedSha256, string& uncompressed) {
  std::unique_ptr<string> compressed = std::make_unique<string>();
  loadFileIntoString(filename,expectedSha256,*compressed);

  static constexpr size_t CHUNK_SIZE = 262144;

  int zret;
  z_stream zs;
  zs.zalloc = Z_NULL;
  zs.zfree = Z_NULL;
  zs.opaque = Z_NULL;
  zs.avail_in = 0;
  zs.next_in = Z_NULL;
  int windowBits = 15 + 32; //Add 32 according to zlib docs to enable gzip decoding
  zret = inflateInit2(&zs,windowBits);
  if(zret != Z_OK) {
    (void)inflateEnd(&zs);
    throw StringError("Error while ungzipping file. Invalid file? File: " + filename);
  }

  //TODO zs.avail_in is 32 bit, may fail with files larger than 4GB.
  zs.avail_in = compressed->size();
  zs.next_in = (Bytef*)(&(*compressed)[0]);
  while(true) {
    size_t uncompressedSoFar = uncompressed.size();
    uncompressed.resize(uncompressedSoFar + CHUNK_SIZE);
    zs.next_out = (Bytef*)(&uncompressed[uncompressedSoFar]);
    zs.avail_out = CHUNK_SIZE;
    zret = inflate(&zs,Z_FINISH);
    assert(zret != Z_STREAM_ERROR);
    switch(zret) {
    case Z_NEED_DICT:
      (void)inflateEnd(&zs);
      throw StringError("Error while ungzipping file, Z_NEED_DICT. Invalid file? File: " + filename);
    case Z_DATA_ERROR:
      (void)inflateEnd(&zs);
      throw StringError("Error while ungzipping file, Z_DATA_ERROR. Invalid file? File: " + filename);
    case Z_MEM_ERROR:
      (void)inflateEnd(&zs);
      throw StringError("Error while ungzipping file, Z_MEM_ERROR. Invalid file? File: " + filename);
    default:
      break;
    }
    //Output buffer space remaining?
    if(zs.avail_out != 0) {
      assert(zs.avail_out > 0);
      //It must be the case that we're done
      if(zret == Z_STREAM_END)
        break;
      //Otherwise, we're in trouble
      (void)inflateEnd(&zs);
      throw StringError("Error while ungzipping file, reached unexpected end of input. File: " + filename);
    }
  }
  //Prune string down to just what we need
  uncompressed.resize(uncompressed.size()-zs.avail_out);
  //Clean up
  (void)inflateEnd(&zs);
}
