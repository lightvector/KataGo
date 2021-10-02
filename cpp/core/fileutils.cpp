#include "../core/fileutils.h"

#include <fstream>
#include <iomanip>
#include <zlib.h>
#include <ghc/filesystem.hpp>

#include "../core/global.h"
#include "../core/sha2.h"

namespace gfs = ghc::filesystem;

//------------------------
#include "../core/using.h"
//------------------------

bool FileUtils::exists(const string& path) {
  try {
    gfs::path gfsPath(gfs::u8path(path));
    return gfs::exists(gfsPath);
  }
  catch(const gfs::filesystem_error&) {
    return false;
  }
}

bool FileUtils::tryOpen(ifstream& in, const char* filename, std::ios_base::openmode mode) {
  in.open(gfs::u8path(filename), mode);
  return in.good();
}
bool FileUtils::tryOpen(ofstream& out, const char* filename, std::ios_base::openmode mode) {
  out.open(gfs::u8path(filename), mode);
  return out.good();
}
bool FileUtils::tryOpen(ifstream& in, const string& filename, std::ios_base::openmode mode) {
  return tryOpen(in, filename.c_str(), mode);
}
bool FileUtils::tryOpen(ofstream& out, const string& filename, std::ios_base::openmode mode) {
  return tryOpen(out, filename.c_str(), mode);
}
void FileUtils::open(ifstream& in, const char* filename, std::ios_base::openmode mode) {
  in.open(gfs::u8path(filename), mode);
  if(!in.good())
    throw IOError("Could not open file " + string(filename) + " - does not exist or invalid permissions?");
}
void FileUtils::open(ofstream& out, const char* filename, std::ios_base::openmode mode) {
  out.open(gfs::u8path(filename), mode);
  if(!out.good())
    throw IOError("Could not write to file " + string(filename) + " - invalid path or permissions?");
}
void FileUtils::open(ifstream& in, const string& filename, std::ios_base::openmode mode) {
  open(in, filename.c_str(), mode);
}
void FileUtils::open(ofstream& out, const string& filename, std::ios_base::openmode mode) {
  open(out, filename.c_str(), mode);
}

void FileUtils::loadFileIntoString(const string& filename, const string& expectedSha256, string& str) {
  ifstream in;
  open(in, filename, std::ios::in | std::ios::binary | std::ios::ate);
  
  ifstream::pos_type fileSize = in.tellg();
  if(fileSize < 0)
    throw StringError("tellg failed to determine size");

  in.seekg(0, std::ios::beg);
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


//TODO someday there's a bit of duplication of funtionality here versus above, if at some point we care to clean it up.
string FileUtils::readFile(const char* filename)
{
  ifstream ifs;
  open(ifs,filename);
  string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return str;
}

string FileUtils::readFile(const string& filename)
{
  return readFile(filename.c_str());
}

string FileUtils::readFileBinary(const char* filename)
{
  ifstream ifs;
  open(ifs,filename,std::ios::binary);
  string str((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return str;
}

string FileUtils::readFileBinary(const string& filename)
{
  return readFileBinary(filename.c_str());
}

//Read file into separate lines, using the specified delimiter character(s).
//The delimiter characters are NOT included.
vector<string> FileUtils::readFileLines(const char* filename, char delimiter)
{
  ifstream ifs;
  open(ifs,filename);

  vector<string> vec;
  string line;
  while(getline(ifs,line,delimiter))
    vec.push_back(line);
  return vec;
}

vector<string> FileUtils::readFileLines(const string& filename, char delimiter)
{
  return readFileLines(filename.c_str(), delimiter);
}

void FileUtils::collectFiles(const string& dirname, std::function<bool(const string&)> fileFilter, vector<string>& collected)
{
  namespace gfs = ghc::filesystem;
  try {
    for(const gfs::directory_entry& entry: gfs::recursive_directory_iterator(dirname)) {
      if(!gfs::is_directory(entry.status())) {
        const gfs::path& path = entry.path();
        string fileName = path.filename().string();
        if(fileFilter(fileName)) {
          collected.push_back(path.string());
        }
      }
    }
  }
  catch(const gfs::filesystem_error& e) {
    cerr << "Error recursively collecting files: " << e.what() << endl;
    throw StringError(string("Error recursively collecting files: ") + e.what());
  }
}
