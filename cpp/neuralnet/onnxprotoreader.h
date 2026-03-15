#ifndef ONNX_PROTO_READER_H
#define ONNX_PROTO_READER_H

#include <string>
#include <map>
#include "../core/global.h"
#include "../core/logger.h"

// Minimal protobuf parser helper
struct ProtoReader {
  const uint8_t* ptr;
  const uint8_t* end;

  ProtoReader(const uint8_t* p, size_t len) ;

  bool hasBytes() const ;

  uint32_t readVarint();

  // Returns true if tag read, false if EOF
  bool readTag(uint32_t& fieldNum, uint32_t& wireType) ;

  void skipField(uint32_t wireType) ;

  std::string readString() ;
};
class ModelDesc;
class NNEvaluator;
//static void loadModelDescFromONNX(const std::string& onnxFile, ModelDesc& desc) ;

struct ONNXModelHeader {
    bool isOnnx;//True if the model is in ONNX format, false if in .bin.gz format and all fields are default values
    std::map<std::string, std::string> allmetadata;
    int modelVersion;
    std::string modelName;
    int num_spatial_inputs;
    int num_global_inputs;
    bool has_mask;
    int pos_len_x;
    int pos_len_y;
    std::string model_config;
    std::string model_config_sha256;
    ONNXModelHeader();
    void clear();
    void load(const std::string& onnxFile);
    void maybeChangeNNLen(NNEvaluator& nneval) const;
};

#endif  // ONNX_PROTO_READER_H_