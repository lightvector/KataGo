#include "onnxprotoreader.h"
#include "modelversion.h"
#include "nneval.h"
#include "../core/sha2.h"
using namespace std;
// Minimal protobuf parser helper

ProtoReader::ProtoReader(const uint8_t* p, size_t len) : ptr(p), end(p + len) {}

bool ProtoReader::hasBytes() const { return ptr < end; }

uint32_t ProtoReader::readVarint() {
  uint32_t result = 0;
  int shift = 0;
  while(ptr < end) {
    uint8_t byte = *ptr++;
    result |= (uint32_t)(byte & 0x7F) << shift;
    if(!(byte & 0x80))
      return result;
    shift += 7;
    if(shift >= 32)
      break;  // Overflow protection
  }
  return result;
}

// Returns true if tag read, false if EOF
bool ProtoReader::readTag(uint32_t& fieldNum, uint32_t& wireType) {
  if(ptr >= end)
    return false;
  uint32_t tag = readVarint();
  fieldNum = tag >> 3;
  wireType = tag & 7;
  return true;
}

void ProtoReader::skipField(uint32_t wireType) {
  if(wireType == 0) {  // Varint
    readVarint();
  } else if(wireType == 1) {  // 64-bit
    if(ptr + 8 <= end)
      ptr += 8;
  } else if(wireType == 2) {  // Length delimited
    uint32_t len = readVarint();
    if(ptr + len <= end)
      ptr += len;
    else
      ptr = end;
  } else if(wireType == 5) {  // 32-bit
    if(ptr + 4 <= end)
      ptr += 4;
  }
  // Groups (3,4) deprecated/not supported here
}

string ProtoReader::readString() {
  uint32_t len = readVarint();
  if(ptr + len > end)
    return "";
  string s((const char*)ptr, len);
  ptr += len;
  return s;
}

ONNXModelHeader::ONNXModelHeader() {
  clear();
}

void ONNXModelHeader::clear() {
  isOnnx = false;
  allmetadata.clear();
  modelVersion = 0;
  modelName = "";
  num_spatial_inputs = 0;
  num_global_inputs = 0;
  has_mask = false;
  pos_len_x = 0;
  pos_len_y = 0;
  model_config = "";
  model_config_sha256 = "";
}
void ONNXModelHeader::load(const std::string& onnxFile) {
  assert(Global::isSuffix(onnxFile, ".onnx"));
  clear();
  isOnnx = true;
  // Read entire file into memory
  ifstream in(onnxFile, ios::binary | ios::ate);
  if(!in)
    throw StringError("Could not open ONNX file: " + onnxFile);
  size_t fileSize = in.tellg();
  in.seekg(0, ios::beg);

  vector<uint8_t> buffer(fileSize);
  if(!in.read((char*)buffer.data(), fileSize))
    throw StringError("Failed to read ONNX file: " + onnxFile);

  ProtoReader reader(buffer.data(), fileSize);
  // std::map<string, string> metadata;
  allmetadata.clear();

  uint32_t fieldNum, wireType;
  while(reader.readTag(fieldNum, wireType)) {
    if(fieldNum == 14 && wireType == 2) {  // metadata_props (repeated)
      // Read nested message length
      uint32_t msgLen = reader.readVarint();
      const uint8_t* msgEnd = reader.ptr + msgLen;
      if(msgEnd > reader.end)
        break;

      // Parse StringStringEntryProto
      ProtoReader entryReader(reader.ptr, msgLen);
      reader.ptr += msgLen;  // Advance main reader

      string key, value;
      uint32_t eField, eWire;
      while(entryReader.readTag(eField, eWire)) {
        if(eField == 1 && eWire == 2)
          key = entryReader.readString();
        else if(eField == 2 && eWire == 2)
          value = entryReader.readString();
        else
          entryReader.skipField(eWire);
      }
      if(!key.empty())
        allmetadata[key] = value;
    } else {
      reader.skipField(wireType);
    }
  }
  if(!allmetadata.count("modelVersion"))
    throw StringError("ONNX model requires a modelVersion metadata field");
  else if(!Global::tryStringToInt(allmetadata["modelVersion"], modelVersion))
    throw StringError(
      "ONNX model requires a valid modelVersion metadata field, but got: " + allmetadata["modelVersion"]);

  if(!allmetadata.count("name"))
    throw StringError("ONNX model requires a name metadata field");
  modelName = allmetadata["name"];

  if(!allmetadata.count("num_spatial_inputs"))
    throw StringError("ONNX model requires a num_spatial_inputs metadata field");
  else if(!Global::tryStringToInt(allmetadata["num_spatial_inputs"], num_spatial_inputs))
    throw StringError(
      "ONNX model requires a valid num_spatial_inputs metadata field, but got: " + allmetadata["num_spatial_inputs"]);
  if(num_spatial_inputs != NNModelVersion::getNumSpatialFeatures(modelVersion))
    throw StringError("ONNX model requires num_spatial_inputs metadata field to match modelVersion");

  if(!allmetadata.count("num_global_inputs"))
    throw StringError("ONNX model requires a num_global_inputs metadata field");
  else if(!Global::tryStringToInt(allmetadata["num_global_inputs"], num_global_inputs))
    throw StringError(
      "ONNX model requires a valid num_global_inputs metadata field, but got: " + allmetadata["num_global_inputs"]);
  if(num_global_inputs != NNModelVersion::getNumGlobalFeatures(modelVersion))
    throw StringError("ONNX model requires num_global_inputs metadata field to match modelVersion");

  if(!allmetadata.count("has_mask"))
    throw StringError("ONNX model requires a has_mask metadata field");
  else if(!Global::tryStringToBool(allmetadata["has_mask"], has_mask))
    throw StringError("ONNX model requires a valid has_mask metadata field, but got: " + allmetadata["has_mask"]);
  if(!allmetadata.count("model_config") || allmetadata["model_config"].empty())
    throw StringError("ONNX model requires a model_config metadata field");

  if(!allmetadata.count("pos_len_x"))
    throw StringError("ONNX model requires a pos_len_x metadata field");
  else if(!Global::tryStringToInt(allmetadata["pos_len_x"], pos_len_x))
    throw StringError("ONNX model requires a valid pos_len_x metadata field, but got: " + allmetadata["pos_len_x"]);
  if(!allmetadata.count("pos_len_y"))
    throw StringError("ONNX model requires a pos_len_y metadata field");
  else if(!Global::tryStringToInt(allmetadata["pos_len_y"], pos_len_y))
    throw StringError("ONNX model requires a valid pos_len_y metadata field, but got: " + allmetadata["pos_len_y"]);

  if(!allmetadata.count("model_config") || allmetadata["model_config"].empty())
    throw StringError("ONNX model requires a model_config metadata field");
  model_config = allmetadata["model_config"];

  {
    char hashResultBuf[65];
    SHA2::get256((const uint8_t*)model_config.data(), model_config.size(), hashResultBuf);
    string hashResult(hashResultBuf);
    model_config_sha256 = hashResult;
  }
}






void ONNXModelHeader::maybeChangeNNLen(NNEvaluator& nneval) const {
  if(!isOnnx)
    return; // not onnx, do nothing

  if(!has_mask) {
    if(nneval.nnXLen != pos_len_x || nneval.nnYLen != pos_len_y || !nneval.requireExactNNLen)
      throw StringError(
        "ONNX model requires pos_len_x and pos_len_y metadata fields to match nnXLen and nnYLen if has_mask is false");
  } else {
    nneval.requireExactNNLen = false;
    if(nneval.nnXLen > pos_len_x || nneval.nnYLen > pos_len_y)
      throw StringError(
        "ONNX model requires pos_len_x and pos_len_y metadata fields to be at least as large as nnXLen and nnYLen if "
        "has_mask is true");
  }
  nneval.nnXLen = pos_len_x;
  nneval.nnYLen = pos_len_y;
  nneval.policySize = NNPos::getPolicySize(nneval.nnXLen, nneval.nnYLen);
}

