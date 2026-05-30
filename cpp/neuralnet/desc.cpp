#include "../neuralnet/desc.h"

#include <cmath>
#include <fstream>
#include <zlib.h>

#include "../core/global.h"
#include "../core/fileutils.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/sgfmetadata.h"
#include "../neuralnet/nninterface.h"

#include "../core/test.h"

using namespace std;

#if !defined(BYTE_ORDER) || (BYTE_ORDER != LITTLE_ENDIAN && BYTE_ORDER != BIG_ENDIAN)
#error Define BYTE_ORDER to be equal to either LITTLE_ENDIAN or BIG_ENDIAN
#endif

static void checkWeightFinite(float f, const string& name) {
  if(!isfinite(f))
    throw StringError(name + ": Nan or infinite neural net weight or parameter");
}
#define CHECKFINITE(x, name) \
  { checkWeightFinite((x), name); }

//For some strange reason, this function is noticeably faster than
//float x; in >> x;
static float readFloatFast(istream& in, string& tmp) {
  in >> tmp;
  char* endPtr;
  const char* cstr = tmp.c_str();
  float x = strtof(cstr,&endPtr);
  if(endPtr == cstr)
    in.setstate(ios_base::failbit);
  return x;
}

static void readFloats(istream& in, size_t numFloats, bool binaryFloats, const string& name, vector<float>& buf) {
  buf.resize(numFloats);
  if(!binaryFloats) {
    string tmp;
    for(size_t i = 0; i<numFloats; i++) {
      float x = readFloatFast(in,tmp);
      CHECKFINITE(x,name);
      buf[i] = x;
    }
    if(in.fail())
      throw StringError(name + ": could not read float weights. Invalid model - perhaps you are trying to load a .bin.gz model as a .txt.gz model?");
  }
  else {
    //KataGo hacky model format - "@BIN@" followed by the expected number of 32 bit floats, in little-endian binary
    testAssert(sizeof(float) == 4);
    {
      string s;
      int numCharsBeforeAt = 0;
      while((char)in.get() != '@') {
        numCharsBeforeAt++;
        //Something is wrong, there should not be this much whitespace
        if(numCharsBeforeAt > 100 || in.fail()) {
          throw StringError(name + ": could not read float weights. Invalid model - perhaps you are trying to load a .txt.gz model as a .bin.gz model?");
        }
      }
      s += (char)in.get();
      s += (char)in.get();
      s += (char)in.get();
      s += (char)in.get();
      if(s != "BIN@")
        throw StringError(name + ": did not find expected header for binary float block");
    }
    float* data = buf.data();
    char* bytes = (char*)data;
    in.read(bytes, numFloats*sizeof(float));

    if(in.fail())
      throw StringError(name + ": did not find the expected number of floats in binary float block");

#if BYTE_ORDER == BIG_ENDIAN
    for(size_t i = 0; i<numFloats; i++) {
      //Reverse byte order for big endian
      std::swap(bytes[i*4 + 0], bytes[i*4 + 3]);
      std::swap(bytes[i*4 + 1], bytes[i*4 + 2]);
    }
#endif
    for(size_t i = 0; i<numFloats; i++) {
      CHECKFINITE(buf[i],name);
    }
  }
}

//-----------------------------------------------------------------------------

static void parseResidualBlockStack(
  std::istream& in,
  int modelVersion,
  bool binaryFloats,
  const std::string& name,
  int numBlocks,
  int trunkNumChannels,
  std::vector<std::pair<int, unique_ptr_void>>& blocks
);


//-----------------------------------------------------------------------------

ConvLayerDesc::ConvLayerDesc()
  : convYSize(0), convXSize(0), inChannels(0), outChannels(0), dilationY(1), dilationX(1) {}

ConvLayerDesc::ConvLayerDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> convYSize;
  in >> convXSize;
  in >> inChannels;
  in >> outChannels;
  in >> dilationY;
  in >> dilationX;

  if(in.fail())
    throw StringError(name + ": convlayer failed to parse sizes and channels and dilations");

  if(convXSize <= 0 || convYSize <= 0)
    throw StringError(name + ": convolution filter sizes must be positive");
  if(inChannels <= 0 || outChannels <= 0)
    throw StringError(name + ": number of in and out channels must be positive");
  if(dilationX <= 0 || dilationY <= 0)
    throw StringError(name + ": dilation factors must be positive");
  if(convXSize % 2 != 1 || convYSize % 2 != 1)
    throw StringError(name + ": convolution filter sizes must be odd, found even sizes");

  // Model file order is y,x,ic,oc
  // Cuda's order is oc,ic,y,x
  int numWeights = convYSize * convXSize * inChannels * outChannels;
  weights.resize(numWeights);
  int ocStride = convYSize * convXSize * inChannels;
  int icStride = convYSize * convXSize;
  int yStride = convXSize;
  int xStride = 1;

  vector<float> floats;
  readFloats(in, (size_t)convYSize * convXSize * inChannels * outChannels, binaryFloats, name, floats);
  size_t idx = 0;
  for(int y = 0; y < convYSize; y++) {
    for(int x = 0; x < convXSize; x++) {
      for(int ic = 0; ic < inChannels; ic++) {
        for(int oc = 0; oc < outChannels; oc++) {
          float w = floats[idx++];
          weights[oc * ocStride + ic * icStride + y * yStride + x * xStride] = w;
        }
      }
    }
  }
  if(in.fail())
    throw StringError(name + ": convlayer failed to expected number of float weights");
}

ConvLayerDesc::ConvLayerDesc(ConvLayerDesc&& other) {
  *this = std::move(other);
}

ConvLayerDesc& ConvLayerDesc::operator=(ConvLayerDesc&& other) {
  name = std::move(other.name);
  convYSize = other.convYSize;
  convXSize = other.convXSize;
  inChannels = other.inChannels;
  outChannels = other.outChannels;
  dilationY = other.dilationY;
  dilationX = other.dilationX;
  weights = std::move(other.weights);
  return *this;
}

double ConvLayerDesc::getSpatialConvDepth() const {
  // 1x1 = 0
  // 3x3 = 1
  // 5x5 = 2
  // ...
  return (convYSize + convXSize - 2) / 4.0;
}

int64_t ConvLayerDesc::getNumParameters() const {
  return (int64_t)weights.size();
}

void ConvLayerDesc::scaleOutputChannels(const std::vector<float>& scaling) {
  testAssert(weights.size() == convYSize * convXSize * inChannels * outChannels);
  testAssert(scaling.size() == outChannels);
  size_t idx = 0;
  for(int oc = 0; oc < outChannels; oc++) {
    for(int ic = 0; ic < inChannels; ic++) {
      for(int y = 0; y < convYSize; y++) {
        for(int x = 0; x < convXSize; x++) {
          weights[idx++] *= scaling[oc];
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------

BatchNormLayerDesc::BatchNormLayerDesc() : numChannels(0), epsilon(0.001f), hasScale(false), hasBias(false) {}

BatchNormLayerDesc::BatchNormLayerDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> numChannels;
  in >> epsilon;
  in >> hasScale;
  in >> hasBias;

  if(in.fail())
    throw StringError(name + ": bnlayer failed to parse num channels and epsilon and hasScale and hasBias");

  if(numChannels < 1)
    throw StringError(name + ": numChannels (" + Global::intToString(numChannels) + ") < 1");
  if(epsilon <= 0)
    throw StringError(name + ": epsilon (" + Global::floatToString(epsilon) + ") <= 0");

  vector<float> floats;
  readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
  mean = floats;
  readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
  variance = floats;

  if(hasScale) {
    readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
    scale = floats;
  }
  else {
    scale.resize(numChannels);
    for(int c = 0; c < numChannels; c++)
      scale[c] = 1.0;
  }

  if(hasBias) {
    readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
    bias = floats;
  }
  else {
    bias.resize(numChannels);
    for(int c = 0; c < numChannels; c++)
      bias[c] = 0.0;
  }

  if(in.fail())
    throw StringError(
      name + ": bnlayer failed to parse expected number of batch norm mean, variance, bias, scale values");

  computeMerged();
}

BatchNormLayerDesc::BatchNormLayerDesc(BatchNormLayerDesc&& other) {
  *this = std::move(other);
}

BatchNormLayerDesc& BatchNormLayerDesc::operator=(BatchNormLayerDesc&& other) {
  name = std::move(other.name);
  numChannels = other.numChannels;
  epsilon = other.epsilon;
  hasScale = other.hasScale;
  hasBias = other.hasBias;
  mean = std::move(other.mean);
  variance = std::move(other.variance);
  scale = std::move(other.scale);
  bias = std::move(other.bias);
  mergedScale = std::move(other.mergedScale);
  mergedBias = std::move(other.mergedBias);
  return *this;
}


int64_t BatchNormLayerDesc::getNumParameters() const {
  // Count the learnable scale and bias (gamma/beta); mean and variance are running
  // statistics rather than learned parameters.
  return (int64_t)(hasScale ? numChannels : 0) + (int64_t)(hasBias ? numChannels : 0);
}

void BatchNormLayerDesc::computeMerged() {
  mergedScale.resize(numChannels);
  mergedBias.resize(numChannels);
  for(int c = 0; c < numChannels; c++) {
    mergedScale[c] = scale[c] / sqrt(variance[c] + epsilon);
    mergedBias[c] = bias[c] - mergedScale[c] * mean[c];
  }
}

void BatchNormLayerDesc::scaleInputChannels(const std::vector<float>& scaling) {
  testAssert(mergedScale.size() == numChannels);
  testAssert(scaling.size() == numChannels);
  epsilon = (float)(1e-20);
  for(int c = 0; c < numChannels; c++) {
    mergedScale[c] *= scaling[c];

    mean[c] = 0.0f;
    variance[c] = 1.0f - epsilon;
    scale[c] = mergedScale[c];
    bias[c] = mergedBias[c];
  }
}

// Purpose of this is to grab any factors that downscale the values, and instead of applying
// them at this layer, return those values so they can be folded in to a different layer.
void BatchNormLayerDesc::extractChannelFactorsAbsLtOne(std::vector<float>& channelFactors) {
  epsilon = (float)(1e-20);

  channelFactors = std::vector<float>(numChannels);
  for(int i = 0; i < numChannels; i++) {
    if(abs(mergedScale[i]) < 1.0f) {
      channelFactors[i] = mergedScale[i];
      mergedScale[i] = 1.0f;
    }
    else {
      channelFactors[i] = 1.0f;
    }

    mean[i] = 0.0f;
    variance[i] = 1.0f - epsilon;
    scale[i] = mergedScale[i];
    bias[i] = mergedBias[i];
  }
}
void BatchNormLayerDesc::extractChannelFactorsAbsLtOneWithInverses(std::vector<float>& channelFactors, std::vector<float>& invChannelFactors) {
  epsilon = (float)(1e-20);

  channelFactors = std::vector<float>(numChannels);
  invChannelFactors = std::vector<float>(numChannels);
  for(int i = 0; i < numChannels; i++) {
    // Much more gentle, since if we want the inverse channels, then there might be places that still upscale.
    if(abs(mergedScale[i]) < 1.0f / 2.0f) {
      channelFactors[i] = 1.0f / 2.0f;
      invChannelFactors[i] = 2.0f;
      mergedScale[i] *= 2.0f;
    }
    else if(abs(mergedScale[i]) < 1.0f) {
      channelFactors[i] = mergedScale[i];
      invChannelFactors[i] = 1.0f / mergedScale[i];
      mergedScale[i] = 1.0f;
    }
    else {
      channelFactors[i] = 1.0f;
      invChannelFactors[i] = 1.0f;
    }

    mean[i] = 0.0f;
    variance[i] = 1.0f - epsilon;
    scale[i] = mergedScale[i];
    bias[i] = mergedBias[i];
  }
}

void BatchNormLayerDesc::applyScale8ToReduceActivations() {
  epsilon = (float)(1e-20);

  for(int c = 0; c < numChannels; c++) {
    mergedBias[c] *= 0.125f;

    mean[c] = 0.0f;
    variance[c] = 1.0f - epsilon;
    scale[c] = mergedScale[c];
    bias[c] = mergedBias[c];
  }
}


//-----------------------------------------------------------------------------

ActivationLayerDesc::ActivationLayerDesc() : name(), activation(ACTIVATION_RELU) {}

ActivationLayerDesc::ActivationLayerDesc(istream& in, int modelVersion) {
  in >> name;
  if(modelVersion >= 11) {
    string kind;
    in >> kind;
    if(kind == "ACTIVATION_IDENTITY")
      activation = ACTIVATION_IDENTITY;
    else if(kind == "ACTIVATION_RELU")
      activation = ACTIVATION_RELU;
    else if(kind == "ACTIVATION_MISH")
      activation = ACTIVATION_MISH;
    else if(kind == "ACTIVATION_SILU")
      activation = ACTIVATION_SILU;
    else
      throw StringError(
        name + ": unknown activation " + kind
      );
  }
  else {
    activation = ACTIVATION_RELU;
  }
}

ActivationLayerDesc::ActivationLayerDesc(ActivationLayerDesc&& other) {
  *this = std::move(other);
}

ActivationLayerDesc& ActivationLayerDesc::operator=(ActivationLayerDesc&& other) {
  name = std::move(other.name);
  activation = other.activation;
  return *this;
}

// Scale 8 means that all activations in the relevant part of the net are divided by 8
// and we adapt all functions so that the computations we do are equivalent in this space.
// In particular, since we are transforming any input activation x => x/8, if the output
// of a layer was f(x) before, it now needs to be f(x)/8.
// Therefore, we need an activation function g s.t. g(x/8) => f(x)/8, or equivalently
// g(x) = f(8x)/8
void ActivationLayerDesc::applyScale8ToReduceActivations() {
  if(activation == ACTIVATION_IDENTITY) {
    // pass. If f(x) = x, then g(x) = f(8x)/8 = 8x/8 = x = f(x)
  }
  else if(activation == ACTIVATION_RELU) {
    // pass. If f(x) = max(x,0), then g(x) = f(8x)/8 = max(8x,0)/8 = max(x,0) = f(x)
  }
  else if(activation == ACTIVATION_MISH) {
    // If f(x) = x * tanh(softplus(x)) then g(x) = f(8x)/8 = 8x * tanh(softplus(8x)) / 8 = x * tanh(softplus(8x))
    // So we need a new activation function "mish_scale8" which we define as mish_scale8(x) = x * tanh(softplus(8x))
    activation = ACTIVATION_MISH_SCALE8;
  }
  else if(activation == ACTIVATION_SILU) {
    // Not implemented right now, but if we wanted to, it would be:
    // If f(x) = x / (1+exp(-x)) then g(x) = f(8x)/8 = 8x / (1+exp(-8x)) / 8 = x / (1+exp(-8x))
    // So we need a new activation function "silu_scale8" which we define as silu_scale8(x) = x / (1+exp(-8x))
    throw StringError("applyScale8ToReduceActivations not supported for ACTIVATION_SILU");
  }
  else if(activation == ACTIVATION_MISH_SCALE8) {
    throw StringError("Cannot applyScale8ToReduceActivations twice, already applied");
  }
  else {
    ASSERT_UNREACHABLE;
  }
}

//-----------------------------------------------------------------------------

MatMulLayerDesc::MatMulLayerDesc() : name(), inChannels(0), outChannels(0), weights() {}

MatMulLayerDesc::MatMulLayerDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> inChannels;
  in >> outChannels;

  if(in.fail())
    throw StringError(name + ": matmullayer failed to parse num channels");
  if(inChannels <= 0 || outChannels <= 0)
    throw StringError(name + ": number of in and out channels must be positive");

  // Model file order is ic,oc
  // Cublas order used is also ic,oc since we transpose
  int numWeights = inChannels * outChannels;
  weights.resize(numWeights);
  int icStride = outChannels;
  int ocStride = 1;

  vector<float> floats;
  readFloats(in, (size_t)inChannels * outChannels, binaryFloats, name, floats);
  size_t idx = 0;
  for(int ic = 0; ic < inChannels; ic++) {
    for(int oc = 0; oc < outChannels; oc++) {
      float w = floats[idx++];
      weights[oc * ocStride + ic * icStride] = w;
    }
  }
  if(in.fail())
    throw StringError(name + ": matmullayer failed to parse expected number of matmul weights");
}

MatMulLayerDesc::MatMulLayerDesc(MatMulLayerDesc&& other) {
  *this = std::move(other);
}

MatMulLayerDesc& MatMulLayerDesc::operator=(MatMulLayerDesc&& other) {
  name = std::move(other.name);
  inChannels = other.inChannels;
  outChannels = other.outChannels;
  weights = std::move(other.weights);
  return *this;
}


int64_t MatMulLayerDesc::getNumParameters() const {
  return (int64_t)weights.size();
}

void MatMulLayerDesc::scaleOutputChannels(const std::vector<float>& scaling) {
  testAssert(weights.size() == inChannels * outChannels);
  testAssert(scaling.size() == outChannels);
  size_t idx = 0;
  for(int ic = 0; ic < inChannels; ic++) {
    for(int oc = 0; oc < outChannels; oc++) {
      weights[idx++] *= scaling[oc];
    }
  }
}


//-----------------------------------------------------------------------------

MatBiasLayerDesc::MatBiasLayerDesc() : name(), numChannels(0), weights() {}

MatBiasLayerDesc::MatBiasLayerDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> numChannels;

  if(in.fail())
    throw StringError(name + ": matbiaslayer failed to parse num channels");
  if(numChannels <= 0)
    throw StringError(name + ": number of channels must be positive");

  weights.resize(numChannels);

  vector<float> floats;
  readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
  weights = floats;

  if(in.fail())
    throw StringError(name + ": matbiaslayer failed to parse expected number of matbias weights");
}

MatBiasLayerDesc::MatBiasLayerDesc(MatBiasLayerDesc&& other) {
  *this = std::move(other);
}

MatBiasLayerDesc& MatBiasLayerDesc::operator=(MatBiasLayerDesc&& other) {
  name = std::move(other.name);
  numChannels = other.numChannels;
  weights = std::move(other.weights);
  return *this;
}

int64_t MatBiasLayerDesc::getNumParameters() const {
  return (int64_t)weights.size();
}

void MatBiasLayerDesc::applyScale8ToReduceActivations() {
  for(int c = 0; c < numChannels; c++) {
    weights[c] *= 0.125f;
  }
}

//-----------------------------------------------------------------------------

ResidualBlockDesc::ResidualBlockDesc() {}

ResidualBlockDesc::ResidualBlockDesc(istream& in, int modelVersion, bool binaryFloats) {
  in >> name;
  if(in.fail())
    throw StringError(name + ": res block failed to parse name");

  preBN = BatchNormLayerDesc(in,binaryFloats);
  preActivation = ActivationLayerDesc(in,modelVersion);
  regularConv = ConvLayerDesc(in,binaryFloats);
  midBN = BatchNormLayerDesc(in,binaryFloats);
  midActivation = ActivationLayerDesc(in,modelVersion);
  finalConv = ConvLayerDesc(in,binaryFloats);

  if(preBN.numChannels != regularConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": preBN.numChannels (%d) != regularConv.inChannels (%d)", preBN.numChannels, regularConv.inChannels));
  if(midBN.numChannels != regularConv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": midBN.numChannels (%d) != regularConv.outChannels (%d)", midBN.numChannels, regularConv.outChannels));
  if(midBN.numChannels != finalConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": midBN.numChannels (%d) != finalConv.inChannels (%d)", midBN.numChannels, finalConv.inChannels));

  if(in.fail())
    throw StringError(name + ": res block parse failure (istream fail() return true)");
}

ResidualBlockDesc::ResidualBlockDesc(ResidualBlockDesc&& other) {
  *this = std::move(other);
}

ResidualBlockDesc& ResidualBlockDesc::operator=(ResidualBlockDesc&& other) {
  name = std::move(other.name);
  preBN = std::move(other.preBN);
  preActivation = std::move(other.preActivation);
  regularConv = std::move(other.regularConv);
  midBN = std::move(other.midBN);
  midActivation = std::move(other.midActivation);
  finalConv = std::move(other.finalConv);
  return *this;
}

void ResidualBlockDesc::iterConvLayers(const std::function<void(const ConvLayerDesc& desc)>& f) const {
  f(regularConv);
  f(finalConv);
}

double ResidualBlockDesc::getSpatialConvDepth() const {
  return regularConv.getSpatialConvDepth() + finalConv.getSpatialConvDepth();
}

int64_t ResidualBlockDesc::getNumParameters() const {
  return
    preBN.getNumParameters() +
    regularConv.getNumParameters() +
    midBN.getNumParameters() +
    finalConv.getNumParameters();
}

void ResidualBlockDesc::transformToReduceActivations() {
  // Merge in any multiplications by values less than 1.0 to happen earlier.
  std::vector<float> channelFactors;
  midBN.extractChannelFactorsAbsLtOne(channelFactors);
  regularConv.scaleOutputChannels(channelFactors);
}

void ResidualBlockDesc::applyScale8ToReduceActivations() {
  preBN.applyScale8ToReduceActivations();
  preActivation.applyScale8ToReduceActivations();
  midBN.applyScale8ToReduceActivations();
  midActivation.applyScale8ToReduceActivations();
}

//-----------------------------------------------------------------------------

GlobalPoolingResidualBlockDesc::GlobalPoolingResidualBlockDesc() {}

GlobalPoolingResidualBlockDesc::GlobalPoolingResidualBlockDesc(istream& in, int vrsn, bool binaryFloats) {
  in >> name;
  if(in.fail())
    throw StringError(name + ": gpool res block failed to parse name");
  modelVersion = vrsn;
  preBN = BatchNormLayerDesc(in,binaryFloats);
  preActivation = ActivationLayerDesc(in,modelVersion);
  regularConv = ConvLayerDesc(in,binaryFloats);
  gpoolConv = ConvLayerDesc(in,binaryFloats);
  gpoolBN = BatchNormLayerDesc(in,binaryFloats);
  gpoolActivation = ActivationLayerDesc(in,modelVersion);
  gpoolToBiasMul = MatMulLayerDesc(in,binaryFloats);
  midBN = BatchNormLayerDesc(in,binaryFloats);
  midActivation = ActivationLayerDesc(in,modelVersion);
  finalConv = ConvLayerDesc(in,binaryFloats);

  if(preBN.numChannels != regularConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": preBN.numChannels (%d) != regularConv.inChannels (%d)", preBN.numChannels, regularConv.inChannels));
  if(preBN.numChannels != gpoolConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": preBN.numChannels (%d) != gpoolConv.inChannels (%d)", preBN.numChannels, gpoolConv.inChannels));
  if(gpoolBN.numChannels != gpoolConv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": gpoolBN.numChannels (%d) != gpoolConv.outChannels (%d)", gpoolBN.numChannels, gpoolConv.outChannels));
  if(gpoolBN.numChannels * 3 != gpoolToBiasMul.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": gpoolBN.numChannels * 3 (%d) != gpoolToBiasMul.inChannels (%d)",
               gpoolBN.numChannels * 3,
               gpoolToBiasMul.inChannels));
  if(midBN.numChannels != regularConv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": midBN.numChannels (%d) != regularConv.outChannels (%d)", midBN.numChannels, regularConv.outChannels));
  if(midBN.numChannels != gpoolToBiasMul.outChannels)
    throw StringError(
      name +
      Global::strprintf(
        ": midBN.numChannels (%d) != gpoolToBiasMul.outChannels (%d)", midBN.numChannels, gpoolToBiasMul.outChannels));
  if(midBN.numChannels != finalConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": midBN.numChannels (%d) != finalConv.inChannels (%d)", midBN.numChannels, finalConv.inChannels));

  if(in.fail())
    throw StringError(name + ": gpool res block parse failure (istream fail() return true)");
}

GlobalPoolingResidualBlockDesc::GlobalPoolingResidualBlockDesc(GlobalPoolingResidualBlockDesc&& other) {
  *this = std::move(other);
}

GlobalPoolingResidualBlockDesc& GlobalPoolingResidualBlockDesc::operator=(GlobalPoolingResidualBlockDesc&& other) {
  name = std::move(other.name);
  preBN = std::move(other.preBN);
  preActivation = std::move(other.preActivation);
  regularConv = std::move(other.regularConv);
  gpoolConv = std::move(other.gpoolConv);
  gpoolBN = std::move(other.gpoolBN);
  gpoolActivation = std::move(other.gpoolActivation);
  gpoolToBiasMul = std::move(other.gpoolToBiasMul);
  midBN = std::move(other.midBN);
  midActivation = std::move(other.midActivation);
  finalConv = std::move(other.finalConv);
  return *this;
}

void GlobalPoolingResidualBlockDesc::iterConvLayers(const std::function<void(const ConvLayerDesc& desc)>& f) const {
  f(regularConv);
  f(gpoolConv);
  f(finalConv);
}

double GlobalPoolingResidualBlockDesc::getSpatialConvDepth() const {
  return regularConv.getSpatialConvDepth() + finalConv.getSpatialConvDepth();
}

int64_t GlobalPoolingResidualBlockDesc::getNumParameters() const {
  return
    preBN.getNumParameters() +
    regularConv.getNumParameters() +
    gpoolConv.getNumParameters() +
    gpoolBN.getNumParameters() +
    gpoolToBiasMul.getNumParameters() +
    midBN.getNumParameters() +
    finalConv.getNumParameters();
}


void GlobalPoolingResidualBlockDesc::transformToReduceActivations() {
  // Merge in any multiplications by values less than 1.0 to happen earlier.
  {
    std::vector<float> channelFactors;
    midBN.extractChannelFactorsAbsLtOne(channelFactors);
    regularConv.scaleOutputChannels(channelFactors);
    gpoolToBiasMul.scaleOutputChannels(channelFactors);
  }
  {
    std::vector<float> channelFactors;
    gpoolBN.extractChannelFactorsAbsLtOne(channelFactors);
    gpoolConv.scaleOutputChannels(channelFactors);
  }
}

void GlobalPoolingResidualBlockDesc::applyScale8ToReduceActivations() {
  preBN.applyScale8ToReduceActivations();
  preActivation.applyScale8ToReduceActivations();
  gpoolBN.applyScale8ToReduceActivations();
  gpoolActivation.applyScale8ToReduceActivations();
  midBN.applyScale8ToReduceActivations();
  midActivation.applyScale8ToReduceActivations();
}

//-----------------------------------------------------------------------------

NestedBottleneckResidualBlockDesc::NestedBottleneckResidualBlockDesc() {}

NestedBottleneckResidualBlockDesc::NestedBottleneckResidualBlockDesc(istream& in, int modelVersion, bool binaryFloats) {
  in >> name;
  if(in.fail())
    throw StringError(name + ": res block failed to parse name");
  in >> numBlocks;
  if(in.fail())
    throw StringError(name + ": nested bottleneck res block failed to parse num blocks");
  if(numBlocks < 1)
    throw StringError(name + ": nested bottleneck res block num blocks must be positive");

  preBN = BatchNormLayerDesc(in,binaryFloats);
  preActivation = ActivationLayerDesc(in,modelVersion);
  preConv = ConvLayerDesc(in,binaryFloats);

  parseResidualBlockStack(in, modelVersion, binaryFloats, name, numBlocks, preConv.outChannels, blocks);

  postBN = BatchNormLayerDesc(in,binaryFloats);
  postActivation = ActivationLayerDesc(in,modelVersion);
  postConv = ConvLayerDesc(in,binaryFloats);

  if(preBN.numChannels != preConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": preBN.numChannels (%d) != preConv.inChannels (%d)", preBN.numChannels, preConv.inChannels));
  if(postBN.numChannels != preConv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": postBN.numChannels (%d) != preConv.outChannels (%d)", postBN.numChannels, preConv.outChannels));
  if(postBN.numChannels != postConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": postBN.numChannels (%d) != postConv.inChannels (%d)", postBN.numChannels, postConv.inChannels));

  if(in.fail())
    throw StringError(name + ": nested res block parse failure (istream fail() return true)");
}

NestedBottleneckResidualBlockDesc::NestedBottleneckResidualBlockDesc(NestedBottleneckResidualBlockDesc&& other) {
  *this = std::move(other);
}

NestedBottleneckResidualBlockDesc& NestedBottleneckResidualBlockDesc::operator=(NestedBottleneckResidualBlockDesc&& other) {
  name = std::move(other.name);
  numBlocks = other.numBlocks;
  preBN = std::move(other.preBN);
  preActivation = std::move(other.preActivation);
  preConv = std::move(other.preConv);
  blocks = std::move(other.blocks);
  postBN = std::move(other.postBN);
  postActivation = std::move(other.postActivation);
  postConv = std::move(other.postConv);
  return *this;
}

void NestedBottleneckResidualBlockDesc::iterConvLayers(const std::function<void(const ConvLayerDesc& desc)>& f) const {
  f(preConv);
  for(int i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc* desc = (const ResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc* desc = (const GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* desc = (const NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      // No conv layers in transformer attention blocks
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // No conv layers in transformer FFN blocks
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  f(postConv);
}

double NestedBottleneckResidualBlockDesc::getSpatialConvDepth() const {
  double depth = 0;
  depth += preConv.getSpatialConvDepth();

  for(int i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc* desc = (const ResidualBlockDesc*)blocks[i].second.get();
      depth += desc->getSpatialConvDepth();
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc* desc = (const GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      depth += desc->getSpatialConvDepth();
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* desc = (const NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      depth += desc->getSpatialConvDepth();
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      // Transformer blocks don't technically contribute spatial conv depth but in practice
      // we count it as 2 for things that want to get a crude idea of model size.
      depth += 2;
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // Transformer blocks don't technically contribute spatial conv depth but in practice
      // we count it as 2 for things that want to get a crude idea of model size.
      depth += 2;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  depth += postConv.getSpatialConvDepth();
  return depth;
}

int64_t NestedBottleneckResidualBlockDesc::getNumParameters() const {
  int64_t numParameters = 0;
  numParameters += preBN.getNumParameters();
  numParameters += preConv.getNumParameters();

  for(int i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc* desc = (const ResidualBlockDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc* desc = (const GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* desc = (const NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      const TransformerAttentionDesc* desc = (const TransformerAttentionDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      const TransformerFFNDesc* desc = (const TransformerFFNDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  numParameters += postBN.getNumParameters();
  numParameters += postConv.getNumParameters();
  return numParameters;
}

static bool blocksContainTransformer(const std::vector<std::pair<int, unique_ptr_void>>& blocks) {
  for(size_t i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
       blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND)
      return true;
  }
  return false;
}

void NestedBottleneckResidualBlockDesc::transformToReduceActivations() {
  // Extract per-channel scale factors from postBN and push into adjacent layers.
  // Skip the per-channel scaling if any immediate child is a transformer block
  // (RMSNorm is not invariant to per-channel scaling). But still recurse into
  // child blocks for their own internal optimizations.
  bool hasTransformerChild = blocksContainTransformer(blocks);

  if(!hasTransformerChild) {
    std::vector<float> channelFactors;
    std::vector<float> invChannelFactors;
    postBN.extractChannelFactorsAbsLtOneWithInverses(channelFactors,invChannelFactors);
    preConv.scaleOutputChannels(channelFactors);

    for(size_t i = 0; i < blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second.get();
        desc->preBN.scaleInputChannels(invChannelFactors);
        desc->finalConv.scaleOutputChannels(channelFactors);
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
        desc->preBN.scaleInputChannels(invChannelFactors);
        desc->finalConv.scaleOutputChannels(channelFactors);
      }
      else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
        NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
        desc->preBN.scaleInputChannels(invChannelFactors);
        desc->postConv.scaleOutputChannels(channelFactors);
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
  }

  // Recurse into child blocks for their own internal optimizations
  for(size_t i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second.get();
      desc->transformToReduceActivations();
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      desc->transformToReduceActivations();
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      desc->transformToReduceActivations();
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
            blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // No internal optimization needed for transformer blocks
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}

void NestedBottleneckResidualBlockDesc::applyScale8ToReduceActivations() {
  preBN.applyScale8ToReduceActivations();
  preActivation.applyScale8ToReduceActivations();

  for(size_t i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second.get();
      desc->applyScale8ToReduceActivations();
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      desc->applyScale8ToReduceActivations();
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      desc->applyScale8ToReduceActivations();
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
            blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // Should not be reached - ModelDesc::applyScale8 guards against models with transformers
      throw StringError("applyScale8ToReduceActivations called on block stack containing transformer blocks");
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  postBN.applyScale8ToReduceActivations();
  postActivation.applyScale8ToReduceActivations();
}

//-----------------------------------------------------------------------------

RMSNormLayerDesc::RMSNormLayerDesc() : numChannels(0), epsilon(0), spatial(false), cgroupSize(0) {}

RMSNormLayerDesc::RMSNormLayerDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> numChannels;
  in >> epsilon;
  int spatialInt;
  in >> spatialInt;
  spatial = (spatialInt != 0);
  in >> cgroupSize;

  if(in.fail())
    throw StringError(name + ": rmsnorm layer failed to parse parameters");
  if(epsilon <= 0 || epsilon > 1.0f)
    throw StringError(name + ": rmsnorm epsilon (" + Global::doubleToString(epsilon) + ") is not positive or is too large");
  if(numChannels < 1)
    throw StringError(name + ": rmsnorm numChannels (" + Global::intToString(numChannels) + ") < 1");
  if(cgroupSize != 0)
    throw StringError(name + ": rmsnorm cgroupSize (" + Global::intToString(cgroupSize) + ") != 0, grouped spatial RMSNorm is not supported");

  vector<float> floats;
  readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
  gamma = floats;
  readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
  beta = floats;

  if(in.fail())
    throw StringError(name + ": rmsnorm layer failed to parse gamma/beta weights");
}

RMSNormLayerDesc::RMSNormLayerDesc(RMSNormLayerDesc&& other) {
  *this = std::move(other);
}

RMSNormLayerDesc& RMSNormLayerDesc::operator=(RMSNormLayerDesc&& other) {
  name = std::move(other.name);
  numChannels = other.numChannels;
  epsilon = other.epsilon;
  spatial = other.spatial;
  cgroupSize = other.cgroupSize;
  gamma = std::move(other.gamma);
  beta = std::move(other.beta);
  return *this;
}

int64_t RMSNormLayerDesc::getNumParameters() const {
  return (int64_t)gamma.size() + (int64_t)beta.size();
}

//-----------------------------------------------------------------------------

TransformerRMSNormDesc::TransformerRMSNormDesc() : numChannels(0), epsilon(0) {}

TransformerRMSNormDesc::TransformerRMSNormDesc(istream& in, bool binaryFloats) {
  in >> name;
  in >> numChannels;
  in >> epsilon;

  if(in.fail())
    throw StringError(name + ": transformer rmsnorm failed to parse parameters");
  if(numChannels < 1)
    throw StringError(name + ": transformer rmsnorm numChannels (" + Global::intToString(numChannels) + ") < 1");
  if(epsilon <= 0 || epsilon > 1.0f)
    throw StringError(name + ": transformer rmsnorm epsilon (" + Global::doubleToString(epsilon) + ") is not positive or is too large");

  vector<float> floats;
  readFloats(in, (size_t)numChannels, binaryFloats, name, floats);
  weight = floats;

  if(in.fail())
    throw StringError(name + ": transformer rmsnorm failed to parse weights");
}

TransformerRMSNormDesc::TransformerRMSNormDesc(TransformerRMSNormDesc&& other) {
  *this = std::move(other);
}

TransformerRMSNormDesc& TransformerRMSNormDesc::operator=(TransformerRMSNormDesc&& other) {
  name = std::move(other.name);
  numChannels = other.numChannels;
  epsilon = other.epsilon;
  weight = std::move(other.weight);
  return *this;
}

int64_t TransformerRMSNormDesc::getNumParameters() const {
  return (int64_t)weight.size();
}

//-----------------------------------------------------------------------------

TransformerAttentionDesc::TransformerAttentionDesc()
  : numHeads(0), numKVHeads(0), qHeadDim(0), vHeadDim(0),
    useRope(false), learnableRope(false),
    ropeNumKVHeads(0), ropeNumPairs(0), ropeTheta(0.0f)
{}

TransformerAttentionDesc::TransformerAttentionDesc(istream& in, bool binaryFloats) {
  in >> name;
  if(in.fail())
    throw StringError(name + ": transformer attention block failed to parse name");

  in >> numHeads;
  in >> numKVHeads;
  in >> qHeadDim;
  in >> vHeadDim;
  int useRopeInt, learnableRopeInt;
  in >> useRopeInt;
  in >> learnableRopeInt;
  useRope = (useRopeInt != 0);
  learnableRope = (learnableRopeInt != 0);

  if(in.fail())
    throw StringError(name + ": transformer attention block failed to parse header");
  if(numHeads < 1 || numKVHeads < 1)
    throw StringError(name + ": transformer attention numHeads and numKVHeads must be positive");
  if(numHeads % numKVHeads != 0)
    throw StringError(name + ": numHeads must be divisible by numKVHeads");
  if(qHeadDim < 1 || vHeadDim < 1)
    throw StringError(name + ": head dims must be positive");
  // RoPE rotates interleaved channel pairs (2p, 2p+1), so qHeadDim must be even when rope is used.
  // All backends assume this (qHeadDim/2 pairs); guard here so an odd qHeadDim fails loudly at load
  // rather than silently dropping the last channel.
  if(useRope && qHeadDim % 2 != 0)
    throw StringError(name + Global::strprintf(": qHeadDim (%d) must be even when RoPE is used", qHeadDim));

  preLN = TransformerRMSNormDesc(in, binaryFloats);
  qProj = MatMulLayerDesc(in, binaryFloats);
  kProj = MatMulLayerDesc(in, binaryFloats);
  vProj = MatMulLayerDesc(in, binaryFloats);
  outProj = MatMulLayerDesc(in, binaryFloats);

  if(qProj.outChannels != numHeads * qHeadDim)
    throw StringError(name + Global::strprintf(": qProj.outChannels (%d) != numHeads*qHeadDim (%d)", qProj.outChannels, numHeads * qHeadDim));
  if(kProj.outChannels != numKVHeads * qHeadDim)
    throw StringError(name + Global::strprintf(": kProj.outChannels (%d) != numKVHeads*qHeadDim (%d)", kProj.outChannels, numKVHeads * qHeadDim));
  if(vProj.outChannels != numKVHeads * vHeadDim)
    throw StringError(name + Global::strprintf(": vProj.outChannels (%d) != numKVHeads*vHeadDim (%d)", vProj.outChannels, numKVHeads * vHeadDim));
  if(outProj.inChannels != numHeads * vHeadDim)
    throw StringError(name + Global::strprintf(": outProj.inChannels (%d) != numHeads*vHeadDim (%d)", outProj.inChannels, numHeads * vHeadDim));

  ropeNumKVHeads = 0;
  ropeNumPairs = 0;
  ropeTheta = 0.0f;

  if(useRope) {
    if(learnableRope) {
      string ropeFreqsName;
      in >> ropeFreqsName;
      in >> ropeNumKVHeads;
      in >> ropeNumPairs;
      int ropeDim2;
      in >> ropeDim2;
      if(in.fail())
        throw StringError(name + ": failed to parse learnable rope freq header");
      if(ropeNumKVHeads != numKVHeads)
        throw StringError(name + Global::strprintf(": ropeNumKVHeads (%d) != numKVHeads (%d)", ropeNumKVHeads, numKVHeads));
      if(ropeNumPairs != qHeadDim / 2)
        throw StringError(name + Global::strprintf(": ropeNumPairs (%d) != qHeadDim/2 (%d)", ropeNumPairs, qHeadDim / 2));
      if(ropeDim2 != 2)
        throw StringError(name + ": rope freq dim2 must be 2");

      vector<float> floats;
      readFloats(in, (size_t)ropeNumKVHeads * ropeNumPairs * 2, binaryFloats, name, floats);
      ropeFreqs = floats;
    }
    else {
      string ropeThetaName;
      in >> ropeThetaName;
      in >> ropeTheta;
      if(in.fail())
        throw StringError(name + ": failed to parse rope theta");
      if(ropeTheta <= 0.0f)
        throw StringError(name + ": rope theta must be positive");
    }
  }

  if(in.fail())
    throw StringError(name + ": transformer attention block parse failure (istream fail() return true)");
}

TransformerAttentionDesc::TransformerAttentionDesc(TransformerAttentionDesc&& other) {
  *this = std::move(other);
}

TransformerAttentionDesc& TransformerAttentionDesc::operator=(TransformerAttentionDesc&& other) {
  name = std::move(other.name);
  numHeads = other.numHeads;
  numKVHeads = other.numKVHeads;
  qHeadDim = other.qHeadDim;
  vHeadDim = other.vHeadDim;
  useRope = other.useRope;
  learnableRope = other.learnableRope;
  preLN = std::move(other.preLN);
  qProj = std::move(other.qProj);
  kProj = std::move(other.kProj);
  vProj = std::move(other.vProj);
  outProj = std::move(other.outProj);
  ropeNumKVHeads = other.ropeNumKVHeads;
  ropeNumPairs = other.ropeNumPairs;
  ropeFreqs = std::move(other.ropeFreqs);
  ropeTheta = other.ropeTheta;
  return *this;
}

int64_t TransformerAttentionDesc::getNumParameters() const {
  return
    preLN.getNumParameters() +
    qProj.getNumParameters() +
    kProj.getNumParameters() +
    vProj.getNumParameters() +
    outProj.getNumParameters() +
    (int64_t)ropeFreqs.size();  // learnable RoPE frequencies, empty for fixed/no RoPE
}

void TransformerAttentionDesc::computeRopeCosSin(int nnXLen, int nnYLen, int paddedNNXYLen, std::vector<float>& cosTable, std::vector<float>& sinTable) const {
  if(!useRope)
    throw StringError("TransformerAttentionDesc::computeRopeCosSin called but useRope is false");

  int numPairs = qHeadDim / 2;

  if(learnableRope) {
    // Precompute from learnable frequencies: ropeFreqs is (numKVHeads, numPairs, 2) flattened
    // For each KV head, pair, and position, angle = x*freq_x + y*freq_y
    testAssert(ropeNumKVHeads == numKVHeads);
    testAssert(ropeNumPairs == numPairs);
    testAssert(ropeFreqs.size() == (size_t)(numKVHeads * numPairs * 2));

    cosTable.resize(numKVHeads * numPairs * paddedNNXYLen, 0.0f);
    sinTable.resize(numKVHeads * numPairs * paddedNNXYLen, 0.0f);

    for(int h = 0; h < numKVHeads; h++) {
      for(int p = 0; p < numPairs; p++) {
        float freqX = ropeFreqs[(h * numPairs + p) * 2 + 0];
        float freqY = ropeFreqs[(h * numPairs + p) * 2 + 1];
        for(int y = 0; y < nnYLen; y++) {
          for(int x = 0; x < nnXLen; x++) {
            int xy = y * nnXLen + x;
            float angle = (float)x * freqX + (float)y * freqY;
            int idx = (h * numPairs + p) * paddedNNXYLen + xy;
            cosTable[idx] = cosf(angle);
            sinTable[idx] = sinf(angle);
          }
        }
      }
    }
  }
  else {
    // Fixed RoPE from theta.
    // Python: freqs = 1/(theta^(arange(0, dimHalf, 2) / dimHalf)) where dimHalf = headDim/2
    // emb = cat([y*freqs, x*freqs]).repeat_interleave(2)
    // First numPairsPerDim pairs are height, next numPairsPerDim are width.
    int numPairsPerDim = numPairs / 2;
    int dimHalf = qHeadDim / 2;

    cosTable.resize(numPairs * paddedNNXYLen, 0.0f);
    sinTable.resize(numPairs * paddedNNXYLen, 0.0f);

    for(int p = 0; p < numPairs; p++) {
      for(int y = 0; y < nnYLen; y++) {
        for(int x = 0; x < nnXLen; x++) {
          int xy = y * nnXLen + x;
          float angle;
          if(p < numPairsPerDim) {
            float freq = 1.0f / powf(ropeTheta, (float)(2 * p) / (float)dimHalf);
            angle = (float)y * freq;
          } else {
            int pAdj = p - numPairsPerDim;
            float freq = 1.0f / powf(ropeTheta, (float)(2 * pAdj) / (float)dimHalf);
            angle = (float)x * freq;
          }
          int idx = p * paddedNNXYLen + xy;
          cosTable[idx] = cosf(angle);
          sinTable[idx] = sinf(angle);
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------

TransformerFFNDesc::TransformerFFNDesc()
  : numChannels(0), ffnChannels(0), useSwiGLU(false)
{}

TransformerFFNDesc::TransformerFFNDesc(istream& in, bool binaryFloats) {
  in >> name;
  if(in.fail())
    throw StringError(name + ": transformer ffn block failed to parse name");

  in >> numChannels;
  in >> ffnChannels;
  int useSwiGLUInt;
  in >> useSwiGLUInt;
  useSwiGLU = (useSwiGLUInt != 0);

  if(in.fail())
    throw StringError(name + ": transformer ffn block failed to parse header");
  if(numChannels < 1 || ffnChannels < 1)
    throw StringError(name + ": transformer ffn channels must be positive");

  preLN = TransformerRMSNormDesc(in, binaryFloats);
  linear1 = MatMulLayerDesc(in, binaryFloats);
  if(useSwiGLU) {
    linearGate = MatMulLayerDesc(in, binaryFloats);
  }
  linear2 = MatMulLayerDesc(in, binaryFloats);

  if(linear1.inChannels != numChannels)
    throw StringError(name + Global::strprintf(": linear1.inChannels (%d) != numChannels (%d)", linear1.inChannels, numChannels));
  if(linear1.outChannels != ffnChannels)
    throw StringError(name + Global::strprintf(": linear1.outChannels (%d) != ffnChannels (%d)", linear1.outChannels, ffnChannels));
  if(useSwiGLU && linearGate.inChannels != numChannels)
    throw StringError(name + Global::strprintf(": linearGate.inChannels (%d) != numChannels (%d)", linearGate.inChannels, numChannels));
  if(useSwiGLU && linearGate.outChannels != ffnChannels)
    throw StringError(name + Global::strprintf(": linearGate.outChannels (%d) != ffnChannels (%d)", linearGate.outChannels, ffnChannels));
  if(linear2.inChannels != ffnChannels)
    throw StringError(name + Global::strprintf(": linear2.inChannels (%d) != ffnChannels (%d)", linear2.inChannels, ffnChannels));
  if(linear2.outChannels != numChannels)
    throw StringError(name + Global::strprintf(": linear2.outChannels (%d) != numChannels (%d)", linear2.outChannels, numChannels));

  if(in.fail())
    throw StringError(name + ": transformer ffn block parse failure (istream fail() return true)");
}

TransformerFFNDesc::TransformerFFNDesc(TransformerFFNDesc&& other) {
  *this = std::move(other);
}

TransformerFFNDesc& TransformerFFNDesc::operator=(TransformerFFNDesc&& other) {
  name = std::move(other.name);
  numChannels = other.numChannels;
  ffnChannels = other.ffnChannels;
  useSwiGLU = other.useSwiGLU;
  preLN = std::move(other.preLN);
  linear1 = std::move(other.linear1);
  linearGate = std::move(other.linearGate);
  linear2 = std::move(other.linear2);
  return *this;
}

int64_t TransformerFFNDesc::getNumParameters() const {
  return
    preLN.getNumParameters() +
    linear1.getNumParameters() +
    linearGate.getNumParameters() +  // empty when not using SwiGLU
    linear2.getNumParameters();
}

//-----------------------------------------------------------------------------

static void parseResidualBlockStack(
  std::istream& in,
  int modelVersion,
  bool binaryFloats,
  const std::string& name,
  int numBlocks,
  int trunkNumChannels,
  std::vector<std::pair<int, unique_ptr_void>>& blocks
) {
  string kind;
  for(int i = 0; i < numBlocks; i++) {
    in >> kind;
    if(in.fail())
      throw StringError(name + ": failed to parse block kind");
    if(kind == "ordinary_block") {
      unique_ptr_void descPtr = make_unique_void(new ResidualBlockDesc(in,modelVersion,binaryFloats));
      ResidualBlockDesc& desc = *((ResidualBlockDesc*)descPtr.get());

      if(desc.preBN.numChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s preBN.numChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.preBN.numChannels,
                   trunkNumChannels));
      if(desc.finalConv.outChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s finalConv.outChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.finalConv.outChannels,
                   trunkNumChannels));

      blocks.emplace_back(ORDINARY_BLOCK_KIND, std::move(descPtr));
    }
    else if(kind == "gpool_block") {
      unique_ptr_void descPtr = make_unique_void(new GlobalPoolingResidualBlockDesc(in, modelVersion, binaryFloats));
      GlobalPoolingResidualBlockDesc& desc = *((GlobalPoolingResidualBlockDesc*)descPtr.get());

      if(desc.preBN.numChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s preBN.numChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.preBN.numChannels,
                   trunkNumChannels));
      if(desc.finalConv.outChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s finalConv.outChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.finalConv.outChannels,
                   trunkNumChannels));

      blocks.emplace_back(GLOBAL_POOLING_BLOCK_KIND, std::move(descPtr));
    }
    else if(kind == "nested_bottleneck_block") {
      unique_ptr_void descPtr = make_unique_void(new NestedBottleneckResidualBlockDesc(in,modelVersion,binaryFloats));
      NestedBottleneckResidualBlockDesc& desc = *((NestedBottleneckResidualBlockDesc*)descPtr.get());

      if(desc.preBN.numChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s preBN.numChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.preBN.numChannels,
                   trunkNumChannels));
      if(desc.postConv.outChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s postConv.outChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.postConv.outChannels,
                   trunkNumChannels));

      blocks.emplace_back(NESTED_BOTTLENECK_BLOCK_KIND, std::move(descPtr));
    }
    else if(kind == "transformer_attention_block") {
      unique_ptr_void descPtr = make_unique_void(new TransformerAttentionDesc(in, binaryFloats));
      TransformerAttentionDesc& desc = *((TransformerAttentionDesc*)descPtr.get());

      if(desc.qProj.inChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s qProj.inChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.qProj.inChannels,
                   trunkNumChannels));
      if(desc.outProj.outChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s outProj.outChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.outProj.outChannels,
                   trunkNumChannels));

      blocks.emplace_back(TRANSFORMER_ATTENTION_BLOCK_KIND, std::move(descPtr));
    }
    else if(kind == "transformer_ffn_block") {
      unique_ptr_void descPtr = make_unique_void(new TransformerFFNDesc(in, binaryFloats));
      TransformerFFNDesc& desc = *((TransformerFFNDesc*)descPtr.get());

      if(desc.numChannels != trunkNumChannels)
        throw StringError(
          name + Global::strprintf(
                   ": %s numChannels (%d) != trunkNumChannels (%d)",
                   desc.name.c_str(),
                   desc.numChannels,
                   trunkNumChannels));

      blocks.emplace_back(TRANSFORMER_FFN_BLOCK_KIND, std::move(descPtr));
    }
    else
      throw StringError(name + ": found unknown block kind: " + kind);

    if(in.fail())
      throw StringError(name + ": trunk istream fail after parsing block");
  }
}

//-----------------------------------------------------------------------------

SGFMetadataEncoderDesc::SGFMetadataEncoderDesc()
  : metaEncoderVersion(0),
    numInputMetaChannels(0)
{}

SGFMetadataEncoderDesc::SGFMetadataEncoderDesc(istream& in, int modelVersion, int metaEncVersion, bool binaryFloats) {
  in >> name;

  if(in.fail())
    throw StringError(name + ": sgf metadata encoder failed to parse name");

  metaEncoderVersion = metaEncVersion;
  in >> numInputMetaChannels;

  if(in.fail())
    throw StringError(name + ": sgf metadata encoder failed to parse num input channels");
  int expectedNumInputMetaChannels = NNModelVersion::getNumInputMetaChannels(metaEncoderVersion);
  if(numInputMetaChannels != expectedNumInputMetaChannels)
    throw StringError(
      name + Global::strprintf(": number of in channels (%d) did not match expected (%d)", numInputMetaChannels, expectedNumInputMetaChannels)
    );

  mul1 = MatMulLayerDesc(in,binaryFloats);
  bias1 = MatBiasLayerDesc(in,binaryFloats);
  act1 = ActivationLayerDesc(in,modelVersion);
  mul2 = MatMulLayerDesc(in,binaryFloats);
  bias2 = MatBiasLayerDesc(in,binaryFloats);
  act2 = ActivationLayerDesc(in,modelVersion);
  mul3 = MatMulLayerDesc(in,binaryFloats);

  if(in.fail())
    throw StringError(name + ": sgf metadata encoder istream fail after parsing layers");

  if(mul1.outChannels != bias1.numChannels)
    throw StringError(
      name +
      Global::strprintf(": mul1.outChannels (%d) != bias1.numChannels (%d)", mul1.outChannels, bias1.numChannels));

  if(mul2.inChannels != mul1.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": mul2.inChannels (%d) != mul1.outChannels (%d)", mul2.inChannels, mul1.outChannels));

  if(mul2.outChannels != bias2.numChannels)
    throw StringError(
      name +
      Global::strprintf(": mul2.outChannels (%d) != bias2.numChannels (%d)", mul2.outChannels, bias2.numChannels));
  if(mul2.outChannels != mul3.inChannels)
    throw StringError(
      name +
      Global::strprintf(": mul2.outChannels (%d) != mul3.inChannels (%d)", mul2.outChannels, mul3.inChannels));
}

SGFMetadataEncoderDesc::~SGFMetadataEncoderDesc() {}

SGFMetadataEncoderDesc::SGFMetadataEncoderDesc(SGFMetadataEncoderDesc&& other) {
  *this = std::move(other);
}

SGFMetadataEncoderDesc& SGFMetadataEncoderDesc::operator=(SGFMetadataEncoderDesc&& other) {
  name = std::move(other.name);
  metaEncoderVersion = other.metaEncoderVersion;
  numInputMetaChannels = other.numInputMetaChannels;
  mul1 = std::move(other.mul1);
  bias1 = std::move(other.bias1);
  act1 = std::move(other.act1);
  mul2 = std::move(other.mul2);
  bias2 = std::move(other.bias2);
  act2 = std::move(other.act2);
  mul3 = std::move(other.mul3);
  return *this;
}

int64_t SGFMetadataEncoderDesc::getNumParameters() const {
  return
    mul1.getNumParameters() +
    bias1.getNumParameters() +
    mul2.getNumParameters() +
    bias2.getNumParameters() +
    mul3.getNumParameters();
}

//-----------------------------------------------------------------------------

TrunkDesc::TrunkDesc()
  : modelVersion(-1),
    numBlocks(0),
    trunkNumChannels(0),
    midNumChannels(0),
    regularNumChannels(0),
    gpoolNumChannels(0),
    metaEncoderVersion(0),
    trunkNormKind(TRUNK_NORM_KIND_STANDARD)
{}

TrunkDesc::TrunkDesc(istream& in, int vrsn, bool binaryFloats, int metaEncVersion) {
  in >> name;
  modelVersion = vrsn;
  in >> numBlocks;
  in >> trunkNumChannels;
  in >> midNumChannels;
  in >> regularNumChannels;
  int dilatedNumChannels; //unused
  in >> dilatedNumChannels;
  in >> gpoolNumChannels;

  metaEncoderVersion = metaEncVersion;

  trunkNormKind = TRUNK_NORM_KIND_STANDARD;
  if(modelVersion >= 15) {
    int unused = 0;
    in >> trunkNormKind;
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported trunk option B: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported trunk option C: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported trunk option D: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported trunk option E: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported trunk option F: " + Global::intToString(unused));
    if(in.fail())
      throw StringError(name + ": trunk failed to parse trunk norm kind / unused params");
    if(trunkNormKind != TRUNK_NORM_KIND_STANDARD && trunkNormKind != TRUNK_NORM_KIND_RMSNORM)
      throw StringError(name + ": unknown or unsupported trunk norm kind: " + Global::intToString(trunkNormKind));
  }

  if(in.fail())
    throw StringError(name + ": trunk failed to parse num blocks or various channel parameters");
  if(numBlocks < 1)
    throw StringError(name + ": trunk num blocks must be positive");
  if(
    trunkNumChannels <= 0 || midNumChannels <= 0 || regularNumChannels <= 0 ||
    gpoolNumChannels <= 0)
    throw StringError(name + ": all numbers of channels must be positive");

  initialConv = ConvLayerDesc(in,binaryFloats);
  if(initialConv.outChannels != trunkNumChannels)
    throw StringError(
      name + Global::strprintf(
               ": %s initialConv.outChannels (%d) != trunkNumChannels (%d)",
               initialConv.name.c_str(),
               initialConv.outChannels,
               trunkNumChannels));

  initialMatMul = MatMulLayerDesc(in,binaryFloats);
  if(initialMatMul.outChannels != trunkNumChannels)
    throw StringError(
      name + Global::strprintf(
               ": %s initialMatMul.outChannels (%d) != trunkNumChannels (%d)",
               initialMatMul.name.c_str(),
               initialMatMul.outChannels,
               trunkNumChannels));

  if(metaEncoderVersion > 0) {
    sgfMetadataEncoder = SGFMetadataEncoderDesc(in,modelVersion,metaEncoderVersion,binaryFloats);
    int numInputMetaChannels = NNModelVersion::getNumInputMetaChannels(metaEncoderVersion);
    if(numInputMetaChannels != sgfMetadataEncoder.mul1.inChannels)
      throw StringError(
        name + Global::strprintf(
               ": %s sgfMetadataEncoder.mul1.inChannels (%d) != numInputMetaChannels (%d)",
               sgfMetadataEncoder.name.c_str(),
               sgfMetadataEncoder.mul1.inChannels,
               numInputMetaChannels));
    if(sgfMetadataEncoder.mul3.outChannels != trunkNumChannels)
      throw StringError(
        name + Global::strprintf(
               ": %s sgfMetadataEncoder.mul3.outChannels (%d) != trunkNumChannels (%d)",
               sgfMetadataEncoder.name.c_str(),
               sgfMetadataEncoder.mul3.outChannels,
               trunkNumChannels));
  }

  parseResidualBlockStack(in, modelVersion, binaryFloats, name, numBlocks, trunkNumChannels, blocks);

  if(trunkNormKind == TRUNK_NORM_KIND_STANDARD) {
    trunkTipBN = BatchNormLayerDesc(in,binaryFloats);
    if(trunkTipBN.numChannels != trunkNumChannels)
      throw StringError(
        name + Global::strprintf(
                 ": trunkTipBN.numChannels (%d) != trunkNumChannels (%d)", trunkTipBN.numChannels, trunkNumChannels));
  }
  else {
    trunkTipRMSNorm = RMSNormLayerDesc(in,binaryFloats);
    if(trunkTipRMSNorm.numChannels != trunkNumChannels)
      throw StringError(
        name + Global::strprintf(
                 ": trunkTipRMSNorm.numChannels (%d) != trunkNumChannels (%d)", trunkTipRMSNorm.numChannels, trunkNumChannels));
  }
  trunkTipActivation = ActivationLayerDesc(in,modelVersion);

  if(in.fail())
    throw StringError(name + ": trunk istream fail after parsing tip");
}

TrunkDesc::~TrunkDesc() {
}

TrunkDesc::TrunkDesc(TrunkDesc&& other) {
  name = std::move(other.name);
  modelVersion = other.modelVersion;
  numBlocks = other.numBlocks;
  trunkNumChannels = other.trunkNumChannels;
  midNumChannels = other.midNumChannels;
  regularNumChannels = other.regularNumChannels;
  gpoolNumChannels = other.gpoolNumChannels;
  metaEncoderVersion = other.metaEncoderVersion;
  trunkNormKind = other.trunkNormKind;
  initialConv = std::move(other.initialConv);
  initialMatMul = std::move(other.initialMatMul);
  sgfMetadataEncoder = std::move(other.sgfMetadataEncoder);
  blocks = std::move(other.blocks);
  trunkTipBN = std::move(other.trunkTipBN);
  trunkTipRMSNorm = std::move(other.trunkTipRMSNorm);
  trunkTipActivation = std::move(other.trunkTipActivation);
}

TrunkDesc& TrunkDesc::operator=(TrunkDesc&& other) {
  name = std::move(other.name);
  modelVersion = other.modelVersion;
  numBlocks = other.numBlocks;
  trunkNumChannels = other.trunkNumChannels;
  midNumChannels = other.midNumChannels;
  regularNumChannels = other.regularNumChannels;
  gpoolNumChannels = other.gpoolNumChannels;
  metaEncoderVersion = other.metaEncoderVersion;
  trunkNormKind = other.trunkNormKind;
  initialConv = std::move(other.initialConv);
  initialMatMul = std::move(other.initialMatMul);
  sgfMetadataEncoder = std::move(other.sgfMetadataEncoder);
  blocks = std::move(other.blocks);
  trunkTipBN = std::move(other.trunkTipBN);
  trunkTipRMSNorm = std::move(other.trunkTipRMSNorm);
  trunkTipActivation = std::move(other.trunkTipActivation);
  return *this;
}

void TrunkDesc::iterConvLayers(const std::function<void(const ConvLayerDesc& desc)>& f) const {
  f(initialConv);
  for(int i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      desc->iterConvLayers(f);
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      // No conv layers in transformer attention blocks
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // No conv layers in transformer FFN blocks
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}

double TrunkDesc::getSpatialConvDepth() const {
  double depth = 0;
  depth += initialConv.getSpatialConvDepth();

  for(int i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc* desc = (const ResidualBlockDesc*)blocks[i].second.get();
      depth += desc->getSpatialConvDepth();
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc* desc = (const GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      depth += desc->getSpatialConvDepth();
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* desc = (const NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      depth += desc->getSpatialConvDepth();
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      // Transformer blocks don't technically contribute spatial conv depth but in practice
      // we count it as 2 for things that want to get a crude idea of model size.
      depth += 2;
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // Transformer blocks don't technically contribute spatial conv depth but in practice
      // we count it as 2 for things that want to get a crude idea of model size.
      depth += 2;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  return depth;
}

int64_t TrunkDesc::getNumParameters() const {
  int64_t numParameters = 0;
  numParameters += initialConv.getNumParameters();
  numParameters += initialMatMul.getNumParameters();
  if(metaEncoderVersion > 0)
    numParameters += sgfMetadataEncoder.getNumParameters();

  for(int i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc* desc = (const ResidualBlockDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc* desc = (const GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* desc = (const NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND) {
      const TransformerAttentionDesc* desc = (const TransformerAttentionDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else if(blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      const TransformerFFNDesc* desc = (const TransformerFFNDesc*)blocks[i].second.get();
      numParameters += desc->getNumParameters();
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  // Whichever trunk tip norm is unused has empty parameter vectors, so summing both is safe.
  numParameters += trunkTipBN.getNumParameters();
  numParameters += trunkTipRMSNorm.getNumParameters();
  return numParameters;
}

void TrunkDesc::transformToReduceActivations() {
  // Top-level per-channel scaling: extract factors from trunk tip BN and push into
  // adjacent layers. This requires a standard BatchNorm trunk tip and no immediate-child
  // transformer blocks (RMSNorm is not invariant to per-channel scaling).
  bool canDoTopLevelScaling = (trunkNormKind == TRUNK_NORM_KIND_STANDARD) && !blocksContainTransformer(blocks);

  if(canDoTopLevelScaling) {
    std::vector<float> channelFactors;
    std::vector<float> invChannelFactors;
    trunkTipBN.extractChannelFactorsAbsLtOneWithInverses(channelFactors,invChannelFactors);

    initialConv.scaleOutputChannels(channelFactors);
    initialMatMul.scaleOutputChannels(channelFactors);
    if(metaEncoderVersion > 0) {
      sgfMetadataEncoder.mul3.scaleOutputChannels(channelFactors);
    }

    for(size_t i = 0; i < blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second.get();
        desc->preBN.scaleInputChannels(invChannelFactors);
        desc->finalConv.scaleOutputChannels(channelFactors);
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
        desc->preBN.scaleInputChannels(invChannelFactors);
        desc->finalConv.scaleOutputChannels(channelFactors);
      }
      else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
        NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
        desc->preBN.scaleInputChannels(invChannelFactors);
        desc->postConv.scaleOutputChannels(channelFactors);
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
  }

  // Always recurse into child blocks for their own internal optimizations
  for(size_t i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second.get();
      desc->transformToReduceActivations();
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      desc->transformToReduceActivations();
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      desc->transformToReduceActivations();
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
            blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // No internal optimization needed for transformer blocks
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}

void TrunkDesc::applyScale8ToReduceActivations() {
  std::vector<float> channelFactors(trunkNumChannels);
  for(int i = 0; i<trunkNumChannels; i++)
    channelFactors[i] = 0.125f;

  initialConv.scaleOutputChannels(channelFactors);
  initialMatMul.scaleOutputChannels(channelFactors);
  if(metaEncoderVersion > 0) {
    sgfMetadataEncoder.mul3.scaleOutputChannels(channelFactors);
  }

  trunkTipBN.applyScale8ToReduceActivations();
  trunkTipActivation.applyScale8ToReduceActivations();

  for(int i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second.get();
      desc->applyScale8ToReduceActivations();
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second.get();
      desc->applyScale8ToReduceActivations();
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      desc->applyScale8ToReduceActivations();
    }
    else if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
            blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND) {
      // Should not be reached - ModelDesc::applyScale8 guards against models with transformers
      throw StringError("applyScale8ToReduceActivations called on trunk containing transformer blocks");
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}


//-----------------------------------------------------------------------------

PolicyHeadDesc::PolicyHeadDesc() : modelVersion(-1) {}

PolicyHeadDesc::PolicyHeadDesc(istream& in, int vrsn, bool binaryFloats) {
  in >> name;
  modelVersion = vrsn;

  if(in.fail())
    throw StringError(name + ": policy head failed to parse name");

  if(modelVersion >= 17) {
    in >> policyOutChannels;
    if(in.fail())
      throw StringError(name + ": policy head failed to parse policyOutChannels");
    // version 17 supports q value predictions based on whether channels are 2 (optimistic policy only) or 4 (optimistic policy + q winloss and score)
    if(policyOutChannels != 2 && policyOutChannels != 4)
      throw StringError(name + ": policy head got invalid policyOutChannels " + Global::intToString(policyOutChannels));
  }
  else if(modelVersion == 16)
    policyOutChannels = 4; // added q value predictions
  else if(modelVersion >= 12)
    policyOutChannels = 2;
  else
    policyOutChannels = 1;

  if(modelVersion >= 17) {
    int unused = 0;
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported policy option A: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported policy option B: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported policy option C: " + Global::intToString(unused));
    if(in.fail())
      throw StringError(name + ": model failed to parse unused params");
  }

  p1Conv = ConvLayerDesc(in,binaryFloats);
  g1Conv = ConvLayerDesc(in,binaryFloats);
  g1BN = BatchNormLayerDesc(in,binaryFloats);
  g1Activation = ActivationLayerDesc(in,modelVersion);
  gpoolToBiasMul = MatMulLayerDesc(in,binaryFloats);
  p1BN = BatchNormLayerDesc(in,binaryFloats);
  p1Activation = ActivationLayerDesc(in,modelVersion);
  p2Conv = ConvLayerDesc(in,binaryFloats);
  gpoolToPassMul = MatMulLayerDesc(in,binaryFloats);
  if(modelVersion >= 15) {
    gpoolToPassBias = MatBiasLayerDesc(in,binaryFloats);
    passActivation = ActivationLayerDesc(in,modelVersion);
    gpoolToPassMul2 = MatMulLayerDesc(in,binaryFloats);
  }
  else {
    gpoolToPassBias = MatBiasLayerDesc();
    passActivation = ActivationLayerDesc();
    gpoolToPassMul2 = MatMulLayerDesc();
  }

  if(in.fail())
    throw StringError(name + ": policy head istream fail after parsing layers");

  if(p1Conv.outChannels != p1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(": p1Conv.outChannels (%d) != p1BN.numChannels (%d)", p1Conv.outChannels, p1BN.numChannels));
  if(g1Conv.outChannels != g1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(": g1Conv.outChannels (%d) != g1BN.numChannels (%d)", g1Conv.outChannels, g1BN.numChannels));
  if(gpoolToBiasMul.inChannels != g1BN.numChannels * 3)
    throw StringError(
      name + Global::strprintf(
               ": gpoolToBiasMul.inChannels (%d) != g1BN.numChannels*3 (%d)",
               gpoolToBiasMul.inChannels,
               g1BN.numChannels * 3));
  if(gpoolToBiasMul.outChannels != p1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(
        ": gpoolToBiasMul.outChannels (%d) != p1BN.numChannels (%d)", gpoolToBiasMul.outChannels, p1BN.numChannels));
  if(p2Conv.inChannels != p1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(": p2Conv.inChannels (%d) != p1BN.numChannels (%d)", p2Conv.inChannels, p1BN.numChannels));
  if(gpoolToPassMul.inChannels != g1BN.numChannels * 3)
    throw StringError(
      name + Global::strprintf(
               ": gpoolToPassMul.inChannels (%d) != g1BN.numChannels*3 (%d)",
               gpoolToPassMul.inChannels,
               g1BN.numChannels * 3));
  if(modelVersion >= 15) {
    if(p2Conv.outChannels != policyOutChannels)
      throw StringError(name + Global::strprintf(": p2Conv.outChannels (%d) != %d", p2Conv.outChannels, policyOutChannels));
    if(gpoolToPassMul.outChannels != gpoolToPassBias.numChannels)
      throw StringError(name + Global::strprintf(": gpoolToPassMul.outChannels (%d) != gpoolToPassBias.numChannels (%d)", gpoolToPassMul.outChannels, gpoolToPassBias.numChannels));
    if(gpoolToPassMul.outChannels != gpoolToPassMul2.inChannels)
      throw StringError(name + Global::strprintf(": gpoolToPassMul.outChannels (%d) != gpoolToPassMul2.inChannels (%d)", gpoolToPassMul.outChannels, gpoolToPassMul2.inChannels));
    if(gpoolToPassMul.outChannels != p1Conv.outChannels)
      throw StringError(name + Global::strprintf(": gpoolToPassMul.outChannels (%d) != p1Conv.outChannels (%d)", gpoolToPassMul.outChannels, p1Conv.outChannels));
    if(gpoolToPassMul2.outChannels != policyOutChannels)
      throw StringError(name + Global::strprintf(": gpoolToPassMul2.outChannels (%d) != %d", gpoolToPassMul2.outChannels, policyOutChannels));
  }
  else {
    if(p2Conv.outChannels != policyOutChannels)
      throw StringError(name + Global::strprintf(": p2Conv.outChannels (%d) != %d", p2Conv.outChannels, policyOutChannels));
    if(gpoolToPassMul.outChannels != policyOutChannels)
      throw StringError(name + Global::strprintf(": gpoolToPassMul.outChannels (%d) != %d", gpoolToPassMul.outChannels, policyOutChannels));
  }
}

PolicyHeadDesc::~PolicyHeadDesc() {}

PolicyHeadDesc::PolicyHeadDesc(PolicyHeadDesc&& other) {
  *this = std::move(other);
}

PolicyHeadDesc& PolicyHeadDesc::operator=(PolicyHeadDesc&& other) {
  name = std::move(other.name);
  modelVersion = other.modelVersion;
  policyOutChannels = other.policyOutChannels;
  p1Conv = std::move(other.p1Conv);
  g1Conv = std::move(other.g1Conv);
  g1BN = std::move(other.g1BN);
  g1Activation = std::move(other.g1Activation);
  gpoolToBiasMul = std::move(other.gpoolToBiasMul);
  p1BN = std::move(other.p1BN);
  p1Activation = std::move(other.p1Activation);
  p2Conv = std::move(other.p2Conv);
  gpoolToPassMul = std::move(other.gpoolToPassMul);
  gpoolToPassBias = std::move(other.gpoolToPassBias);
  passActivation = std::move(other.passActivation);
  gpoolToPassMul2 = std::move(other.gpoolToPassMul2);
  return *this;
}

void PolicyHeadDesc::iterConvLayers(const std::function<void(const ConvLayerDesc& desc)>& f) const {
  f(p1Conv);
  f(g1Conv);
  f(p2Conv);
}

int64_t PolicyHeadDesc::getNumParameters() const {
  return
    p1Conv.getNumParameters() +
    g1Conv.getNumParameters() +
    g1BN.getNumParameters() +
    gpoolToBiasMul.getNumParameters() +
    p1BN.getNumParameters() +
    p2Conv.getNumParameters() +
    gpoolToPassMul.getNumParameters() +
    gpoolToPassBias.getNumParameters() +
    gpoolToPassMul2.getNumParameters();  // empty for older model versions
}


void PolicyHeadDesc::transformToReduceActivations() {
  // Merge in any multiplications by values less than 1.0 to happen earlier.
  {
    std::vector<float> channelFactors;
    p1BN.extractChannelFactorsAbsLtOne(channelFactors);
    p1Conv.scaleOutputChannels(channelFactors);
    gpoolToBiasMul.scaleOutputChannels(channelFactors);
  }
  {
    std::vector<float> channelFactors;
    g1BN.extractChannelFactorsAbsLtOne(channelFactors);
    g1Conv.scaleOutputChannels(channelFactors);
  }
}

void PolicyHeadDesc::applyScale8ToReduceActivations() {
  g1BN.applyScale8ToReduceActivations();
  g1Activation.applyScale8ToReduceActivations();
  p1BN.applyScale8ToReduceActivations();
  p1Activation.applyScale8ToReduceActivations();
  gpoolToPassBias.applyScale8ToReduceActivations();
  passActivation.applyScale8ToReduceActivations();
}

//-----------------------------------------------------------------------------

ValueHeadDesc::ValueHeadDesc() : modelVersion(-1) {}

ValueHeadDesc::ValueHeadDesc(istream& in, int vrsn, bool binaryFloats) {
  in >> name;
  modelVersion = vrsn;

  if(in.fail())
    throw StringError(name + ": value head failed to parse name");

  if(modelVersion >= 17) {
    int unused = 0;
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported value option A: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported value option B: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported value option C: " + Global::intToString(unused));
    if(in.fail())
      throw StringError(name + ": model failed to parse unused params");
  }

  v1Conv = ConvLayerDesc(in,binaryFloats);
  v1BN = BatchNormLayerDesc(in,binaryFloats);
  v1Activation = ActivationLayerDesc(in,modelVersion);
  v2Mul = MatMulLayerDesc(in,binaryFloats);
  v2Bias = MatBiasLayerDesc(in,binaryFloats);
  v2Activation = ActivationLayerDesc(in,modelVersion);
  v3Mul = MatMulLayerDesc(in,binaryFloats);
  v3Bias = MatBiasLayerDesc(in,binaryFloats);

  sv3Mul = MatMulLayerDesc(in,binaryFloats);
  sv3Bias = MatBiasLayerDesc(in,binaryFloats);
  vOwnershipConv = ConvLayerDesc(in,binaryFloats);

  if(in.fail())
    throw StringError(name + ": value head istream fail after parsing layers");

  if(v1Conv.outChannels != v1BN.numChannels)
    throw StringError(
      name +
      Global::strprintf(": v1Conv.outChannels (%d) != v1BN.numChannels (%d)", v1Conv.outChannels, v1BN.numChannels));

  if(v2Mul.inChannels != v1BN.numChannels * 3)
    throw StringError(
      name + Global::strprintf(
               ": v2Mul.inChannels (%d) != v1BN.numChannels*3 (%d)", v2Mul.inChannels, v1BN.numChannels * 3));

  if(v2Mul.outChannels != v2Bias.numChannels)
    throw StringError(
      name +
      Global::strprintf(": v2Mul.outChannels (%d) != v2Bias.numChannels (%d)", v2Mul.outChannels, v2Bias.numChannels));
  if(v2Mul.outChannels != v3Mul.inChannels)
    throw StringError(
      name +
      Global::strprintf(": v2Mul.outChannels (%d) != v3Mul.inChannels (%d)", v2Mul.outChannels, v3Mul.inChannels));
  if(v3Mul.outChannels != 3)
    throw StringError(name + Global::strprintf(": v3Mul.outChannels (%d) != 3", v3Mul.outChannels));
  if(v3Bias.numChannels != 3)
    throw StringError(name + Global::strprintf(": v3Bias.numChannels (%d) != 3", v3Bias.numChannels));

  if(sv3Mul.inChannels != v2Mul.outChannels)
    throw StringError(
      name +
      Global::strprintf(": sv3Mul.inChannels (%d) != v2Mul.outChannels (%d)", sv3Mul.inChannels, v2Mul.outChannels));

  if(modelVersion >= 9) {
    if(sv3Mul.outChannels != 6)
      throw StringError(name + Global::strprintf(": sv3Mul.outChannels (%d) != 6", sv3Mul.outChannels));
    if(sv3Bias.numChannels != 6)
      throw StringError(name + Global::strprintf(": sv3Bias.numChannels (%d) != 6", sv3Bias.numChannels));
  }
  else if(modelVersion >= 8) {
    if(sv3Mul.outChannels != 4)
      throw StringError(name + Global::strprintf(": sv3Mul.outChannels (%d) != 4", sv3Mul.outChannels));
    if(sv3Bias.numChannels != 4)
      throw StringError(name + Global::strprintf(": sv3Bias.numChannels (%d) != 4", sv3Bias.numChannels));
  }
  else if(modelVersion >= 4) {
    if(sv3Mul.outChannels != 2)
      throw StringError(name + Global::strprintf(": sv3Mul.outChannels (%d) != 2", sv3Mul.outChannels));
    if(sv3Bias.numChannels != 2)
      throw StringError(name + Global::strprintf(": sv3Bias.numChannels (%d) != 2", sv3Bias.numChannels));
  }
  else {
    if(sv3Mul.outChannels != 1)
      throw StringError(name + Global::strprintf(": sv3Mul.outChannels (%d) != 1", sv3Mul.outChannels));
    if(sv3Bias.numChannels != 1)
      throw StringError(name + Global::strprintf(": sv3Bias.numChannels (%d) != 1", sv3Bias.numChannels));
  }

  if(vOwnershipConv.inChannels != v1Conv.outChannels)
    throw StringError(
      name + Global::strprintf(
               ": vOwnershipConv.outChannels (%d) != v1Conv.outChannels (%d)",
               vOwnershipConv.inChannels,
               v1Conv.outChannels));
  if(vOwnershipConv.outChannels != 1)
    throw StringError(name + Global::strprintf(": vOwnershipConv.outChannels (%d) != 1", vOwnershipConv.outChannels));
}

ValueHeadDesc::~ValueHeadDesc() {}

ValueHeadDesc::ValueHeadDesc(ValueHeadDesc&& other) {
  *this = std::move(other);
}

ValueHeadDesc& ValueHeadDesc::operator=(ValueHeadDesc&& other) {
  name = std::move(other.name);
  modelVersion = other.modelVersion;
  v1Conv = std::move(other.v1Conv);
  v1BN = std::move(other.v1BN);
  v1Activation = std::move(other.v1Activation);
  v2Mul = std::move(other.v2Mul);
  v2Bias = std::move(other.v2Bias);
  v2Activation = std::move(other.v2Activation);
  v3Mul = std::move(other.v3Mul);
  v3Bias = std::move(other.v3Bias);
  sv3Mul = std::move(other.sv3Mul);
  sv3Bias = std::move(other.sv3Bias);
  vOwnershipConv = std::move(other.vOwnershipConv);
  return *this;
}

void ValueHeadDesc::iterConvLayers(const std::function<void(const ConvLayerDesc& desc)>& f) const {
  f(v1Conv);
  f(vOwnershipConv);
}

int64_t ValueHeadDesc::getNumParameters() const {
  return
    v1Conv.getNumParameters() +
    v1BN.getNumParameters() +
    v2Mul.getNumParameters() +
    v2Bias.getNumParameters() +
    v3Mul.getNumParameters() +
    v3Bias.getNumParameters() +
    sv3Mul.getNumParameters() +
    sv3Bias.getNumParameters() +
    vOwnershipConv.getNumParameters();
}

void ValueHeadDesc::transformToReduceActivations() {
  // Merge in any multiplications by values less than 1.0 to happen earlier.
  std::vector<float> channelFactors;
  v1BN.extractChannelFactorsAbsLtOne(channelFactors);
  v1Conv.scaleOutputChannels(channelFactors);
}


void ValueHeadDesc::applyScale8ToReduceActivations() {
  v1BN.applyScale8ToReduceActivations();
  v1Activation.applyScale8ToReduceActivations();
  v2Bias.applyScale8ToReduceActivations();
  v2Activation.applyScale8ToReduceActivations();
  v3Bias.applyScale8ToReduceActivations();
  sv3Bias.applyScale8ToReduceActivations();
}


//-----------------------------------------------------------------------------

ModelPostProcessParams::ModelPostProcessParams()
  : tdScoreMultiplier(20.0),
    scoreMeanMultiplier(20.0),
    scoreStdevMultiplier(20.0),
    leadMultiplier(20.0),
    varianceTimeMultiplier(40.0),
    shorttermValueErrorMultiplier(0.25),
    shorttermScoreErrorMultiplier(30.0),
    outputScaleMultiplier(1.0f)
{}
ModelPostProcessParams::~ModelPostProcessParams()
{}

//-----------------------------------------------------------------------------

ModelDesc::ModelDesc()
  : modelVersion(-1),
    numInputChannels(0),
    numInputGlobalChannels(0),
    numInputMetaChannels(0),
    numPolicyChannels(0),
    numValueChannels(0),
    numScoreValueChannels(0),
    numOwnershipChannels(0),
    metaEncoderVersion(0),
    postProcessParams()
{}

ModelDesc::ModelDesc(istream& in, const string& sha256_, bool binaryFloats) {
  in >> name;
  sha256 = sha256_;
  in >> modelVersion;
  if(in.fail())
    throw StringError("Model failed to parse name or version. Is this a valid model file? You probably specified the wrong file.");

  // The model name is embedded into on-disk cache filenames (e.g. the TensorRT plan cache), so keep
  // it short and restricted to filesystem-safe characters: at most 96 chars of [A-Za-z0-9_-].
  if(name.size() > 96)
    throw StringError("Model name is too long (" + Global::intToString((int)name.size()) + " chars, max 96): " + name);
  for(char c : name) {
    bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    if(!ok)
      throw StringError("Model name must contain only alphanumeric characters, underscores, and hyphens: " + name);
  }

  if(modelVersion < 0)
    throw StringError("This neural net has an invalid version, you probably specified the wrong file. Supposed model version: " + Global::intToString(modelVersion));
  if(modelVersion < 3)
    throw StringError("This neural net is from an extremely old version of KataGo and is no longer supported by the engine. Model version: " + Global::intToString(modelVersion));
  if(modelVersion > NNModelVersion::latestModelVersionImplemented)
    throw StringError("This neural net requires a newer KataGo version. Obtain a newer KataGo at https://github.com/lightvector/KataGo. Model version: " + Global::intToString(modelVersion));

  in >> numInputChannels;
  if(in.fail())
    throw StringError(name + ": model failed to parse numInputChannels");
  if(numInputChannels <= 0)
    throw StringError(name + ": model numInputChannels must be positive");

  in >> numInputGlobalChannels;
  if(in.fail())
    throw StringError(name + ": model failed to parse numInputGlobalChannels");
  if(numInputGlobalChannels <= 0)
    throw StringError(name + ": model numInputGlobalChannels must be positive");

  if(modelVersion >= 13) {
    in >> postProcessParams.tdScoreMultiplier;
    if(in.fail())
      throw StringError(name + ": model failed to parse tdScoreMultiplier");
    if(postProcessParams.tdScoreMultiplier <= 0)
      throw StringError(name + ": model tdScoreMultiplier must be positive");
    in >> postProcessParams.scoreMeanMultiplier;
    if(in.fail())
      throw StringError(name + ": model failed to parse scoreMeanMultiplier");
    if(postProcessParams.scoreMeanMultiplier <= 0)
      throw StringError(name + ": model scoreMeanMultiplier must be positive");
    in >> postProcessParams.scoreStdevMultiplier;
    if(in.fail())
      throw StringError(name + ": model failed to parse scoreStdevMultiplier");
    if(postProcessParams.scoreStdevMultiplier <= 0)
      throw StringError(name + ": model scoreStdevMultiplier must be positive");
    in >> postProcessParams.leadMultiplier;
    if(in.fail())
      throw StringError(name + ": model failed to parse leadMultiplier");
    if(postProcessParams.leadMultiplier <= 0)
      throw StringError(name + ": model leadMultiplier must be positive");
    in >> postProcessParams.varianceTimeMultiplier;
    if(in.fail())
      throw StringError(name + ": model failed to parse varianceTimeMultiplier");
    if(postProcessParams.varianceTimeMultiplier <= 0)
      throw StringError(name + ": model varianceTimeMultiplier must be positive");
    in >> postProcessParams.shorttermValueErrorMultiplier;
    if(in.fail())
      throw StringError(name + ": model failed to parse shorttermValueErrorMultiplier");
    if(postProcessParams.shorttermValueErrorMultiplier <= 0)
      throw StringError(name + ": model shorttermValueErrorMultiplier must be positive");
    in >> postProcessParams.shorttermScoreErrorMultiplier;
    if(in.fail())
      throw StringError(name + ": model failed to parse shorttermScoreErrorMultiplier");
    if(postProcessParams.shorttermScoreErrorMultiplier <= 0)
      throw StringError(name + ": model shorttermScoreErrorMultiplier must be positive");
  }
  else {
    postProcessParams = ModelPostProcessParams();
  }

  if(modelVersion >= 15) {
    in >> metaEncoderVersion;
    if(in.fail())
      throw StringError(name + ": model failed to parse metaEncoderVersion");
    if(metaEncoderVersion < 0)
      throw StringError(name + ": model metaEncoderVersion unexpected value: " + Global::intToString(metaEncoderVersion));
    if(metaEncoderVersion > 1)
      throw StringError(
        name + ": model metaEncoderVersion not implemented, you may need a newer KataGo version, value was: " +
        Global::intToString(metaEncoderVersion)
      );
    numInputMetaChannels = NNModelVersion::getNumInputMetaChannels(metaEncoderVersion);
    if(metaEncoderVersion > 0 && numInputMetaChannels != SGFMetadata::METADATA_INPUT_NUM_CHANNELS) {
      throw StringError(
        name + Global::strprintf(
          ": numInputMetaChannels (%d) != METADATA_INPUT_NUM_CHANNELS (%d)",
          numInputMetaChannels,
          SGFMetadata::METADATA_INPUT_NUM_CHANNELS));
    }

    int unused = 0;
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported model option B: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported model option C: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported model option D: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported model option E: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported model option F: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported model option G: " + Global::intToString(unused));
    in >> unused;
    if(unused != 0) throw StringError(name + ": unknown/unsupported model option H: " + Global::intToString(unused));
    if(in.fail())
      throw StringError(name + ": model failed to parse unused params");
  }
  else {
    metaEncoderVersion = 0;
    numInputMetaChannels = 0;
  }

  trunk = TrunkDesc(in, modelVersion, binaryFloats, metaEncoderVersion);
  policyHead = PolicyHeadDesc(in, modelVersion, binaryFloats);
  valueHead = ValueHeadDesc(in, modelVersion, binaryFloats);

  numPolicyChannels = policyHead.policyOutChannels;
  numValueChannels = valueHead.v3Mul.outChannels;
  numScoreValueChannels = valueHead.sv3Mul.outChannels;
  numOwnershipChannels = valueHead.vOwnershipConv.outChannels;

  if(in.fail())
    throw StringError(name + ": model desc istream fail after parsing model");

  if(numInputChannels != trunk.initialConv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": numInputChannels (%d) != trunk.initialConv.inChannels (%d)",
               numInputChannels,
               trunk.initialConv.inChannels));
  if(numInputGlobalChannels != trunk.initialMatMul.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": numInputChannels (%d) != trunk.initialMatMul.inChannels (%d)",
               numInputGlobalChannels,
               trunk.initialMatMul.inChannels));

  if(trunk.trunkNumChannels != policyHead.p1Conv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": trunk.trunkNumChannels (%d) != policyHead.p1Conv.inChannels (%d)",
               trunk.trunkNumChannels,
               policyHead.p1Conv.inChannels));
  if(trunk.trunkNumChannels != policyHead.g1Conv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": trunk.trunkNumChannels (%d) != policyHead.g1Conv.inChannels (%d)",
               trunk.trunkNumChannels,
               policyHead.g1Conv.inChannels));
  if(trunk.trunkNumChannels != valueHead.v1Conv.inChannels)
    throw StringError(
      name + Global::strprintf(
               ": trunk.trunkNumChannels (%d) != valueHead.v1Conv.inChannels (%d)",
               trunk.trunkNumChannels,
               valueHead.v1Conv.inChannels));
}

ModelDesc::~ModelDesc() {}

ModelDesc::ModelDesc(ModelDesc&& other) {
  *this = std::move(other);
}

ModelDesc& ModelDesc::operator=(ModelDesc&& other) {
  name = std::move(other.name);
  sha256 = std::move(other.sha256);
  modelVersion = other.modelVersion;
  numInputChannels = other.numInputChannels;
  numInputGlobalChannels = other.numInputGlobalChannels;
  numInputMetaChannels = other.numInputMetaChannels;
  numPolicyChannels = other.numPolicyChannels;
  numValueChannels = other.numValueChannels;
  numScoreValueChannels = other.numScoreValueChannels;
  numOwnershipChannels = other.numOwnershipChannels;
  metaEncoderVersion = other.metaEncoderVersion;
  postProcessParams = other.postProcessParams;
  trunk = std::move(other.trunk);
  policyHead = std::move(other.policyHead);
  valueHead = std::move(other.valueHead);
  return *this;
}

void ModelDesc::iterConvLayers(const std::function<void(const ConvLayerDesc& desc)>& f) const {
  trunk.iterConvLayers(f);
  policyHead.iterConvLayers(f);
  valueHead.iterConvLayers(f);
}

int ModelDesc::maxConvChannels(int convXSize, int convYSize) const {
  int c = 0;
  auto f = [&c,convXSize,convYSize](const ConvLayerDesc& desc) noexcept {
    if(desc.convXSize == convXSize && desc.convYSize == convYSize) {
      if(desc.inChannels > c)
        c = desc.inChannels;
      if(desc.outChannels > c)
        c = desc.outChannels;
    }
  };
  iterConvLayers(f);
  return c;
}

double ModelDesc::getTrunkSpatialConvDepth() const {
  return trunk.getSpatialConvDepth();
}

int64_t ModelDesc::getNumParameters() const {
  return trunk.getNumParameters() + policyHead.getNumParameters() + valueHead.getNumParameters();
}

string ModelDesc::getShortInfoString() const {
  bool isTransformer = hasAnyTransformerBlocks();
  bool isNbt = false;
  for(size_t i = 0; i < trunk.blocks.size(); i++) {
    if(trunk.blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      isNbt = true;
      break;
    }
  }
  string kind;
  if(isNbt)
    kind = isTransformer ? "nbt transformer" : "nbt convnet";
  else
    kind = isTransformer ? "transformer" : "convnet";
  return kind + ", " + Global::int64ToString(getNumParameters()) + " params";
}

void ModelDesc::transformToReduceActivations() {
  trunk.transformToReduceActivations();
  policyHead.transformToReduceActivations();
  valueHead.transformToReduceActivations();
}

static bool blocksContainTransformerRecursive(const std::vector<std::pair<int, unique_ptr_void>>& blocks) {
  for(size_t i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == TRANSFORMER_ATTENTION_BLOCK_KIND ||
       blocks[i].first == TRANSFORMER_FFN_BLOCK_KIND)
      return true;
    if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* desc = (NestedBottleneckResidualBlockDesc*)blocks[i].second.get();
      if(blocksContainTransformerRecursive(desc->blocks))
        return true;
    }
  }
  return false;
}

bool NestedBottleneckResidualBlockDesc::hasAnyTransformerBlocks() const {
  return blocksContainTransformerRecursive(blocks);
}
bool TrunkDesc::hasAnyTransformerBlocks() const {
  return blocksContainTransformerRecursive(blocks);
}
bool ModelDesc::hasAnyTransformerBlocks() const {
  return trunk.hasAnyTransformerBlocks();
}

void ModelDesc::applyScale8ToReduceActivations() {
  // Scale8 scales the entire net's activations by 1/8 and compensates with MISH_SCALE8.
  // This is unsafe when:
  // - Non-standard trunk norm (RMSNorm is scale-invariant, so it would undo the 1/8 scaling)
  // - SiLU activation anywhere (no SiLU_SCALE8 variant exists)
  // - Any transformer blocks (their internal RMSNorm would undo the scaling)
  if(trunk.trunkNormKind != TRUNK_NORM_KIND_STANDARD)
    return;
  if(trunk.trunkTipActivation.activation == ACTIVATION_SILU)
    return;
  if(blocksContainTransformerRecursive(trunk.blocks))
    return;

  trunk.applyScale8ToReduceActivations();
  policyHead.applyScale8ToReduceActivations();
  valueHead.applyScale8ToReduceActivations();

  postProcessParams.outputScaleMultiplier *= 8.0f;
}

static void releaseVec(std::vector<float>& v) { std::vector<float>().swap(v); }

static void releaseConv(ConvLayerDesc& c) { releaseVec(c.weights); }

static void releaseBN(BatchNormLayerDesc& b) {
  releaseVec(b.mean); releaseVec(b.variance); releaseVec(b.scale);
  releaseVec(b.bias); releaseVec(b.mergedScale); releaseVec(b.mergedBias);
}

static void releaseMatMul(MatMulLayerDesc& m) { releaseVec(m.weights); }
static void releaseMatBias(MatBiasLayerDesc& m) { releaseVec(m.weights); }

static void releaseResidual(ResidualBlockDesc& b) {
  releaseBN(b.preBN); releaseConv(b.regularConv);
  releaseBN(b.midBN); releaseConv(b.finalConv);
}

static void releaseGPool(GlobalPoolingResidualBlockDesc& b) {
  releaseBN(b.preBN); releaseConv(b.regularConv); releaseConv(b.gpoolConv);
  releaseBN(b.gpoolBN); releaseMatMul(b.gpoolToBiasMul);
  releaseBN(b.midBN); releaseConv(b.finalConv);
}

static void releaseBlocks(std::vector<std::pair<int, unique_ptr_void>>& blocks);

static void releaseNested(NestedBottleneckResidualBlockDesc& b) {
  releaseBN(b.preBN); releaseConv(b.preConv);
  releaseBlocks(b.blocks);
  releaseBN(b.postBN); releaseConv(b.postConv);
}

static void releaseBlocks(std::vector<std::pair<int, unique_ptr_void>>& blocks) {
  for(size_t i = 0; i < blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND)
      releaseResidual(*(ResidualBlockDesc*)blocks[i].second.get());
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND)
      releaseGPool(*(GlobalPoolingResidualBlockDesc*)blocks[i].second.get());
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND)
      releaseNested(*(NestedBottleneckResidualBlockDesc*)blocks[i].second.get());
    else
      ASSERT_UNREACHABLE;
  }
}

static void releaseSGFEncoder(SGFMetadataEncoderDesc& e) {
  releaseMatMul(e.mul1); releaseMatBias(e.bias1);
  releaseMatMul(e.mul2); releaseMatBias(e.bias2);
  releaseMatMul(e.mul3);
}

void ModelDesc::releaseWeights() {
  releaseConv(trunk.initialConv);
  releaseMatMul(trunk.initialMatMul);
  if(trunk.metaEncoderVersion > 0)
    releaseSGFEncoder(trunk.sgfMetadataEncoder);
  releaseBlocks(trunk.blocks);
  releaseBN(trunk.trunkTipBN);
  releaseConv(policyHead.p1Conv); releaseConv(policyHead.g1Conv);
  releaseBN(policyHead.g1BN); releaseMatMul(policyHead.gpoolToBiasMul);
  releaseBN(policyHead.p1BN); releaseConv(policyHead.p2Conv);
  releaseMatMul(policyHead.gpoolToPassMul); releaseMatBias(policyHead.gpoolToPassBias);
  releaseMatMul(policyHead.gpoolToPassMul2);
  releaseConv(valueHead.v1Conv); releaseBN(valueHead.v1BN);
  releaseMatMul(valueHead.v2Mul); releaseMatBias(valueHead.v2Bias);
  releaseMatMul(valueHead.v3Mul); releaseMatBias(valueHead.v3Bias);
  releaseMatMul(valueHead.sv3Mul); releaseMatBias(valueHead.sv3Bias);
  releaseConv(valueHead.vOwnershipConv);
}

struct NonCopyingStreamBuf : public std::streambuf
{
  NonCopyingStreamBuf(string& str) {
    char* s = &str[0];
    size_t n = str.size();
    setg(s, s, s + n);
  }
};

void ModelDesc::loadFromFileMaybeGZipped(const string& fileName, ModelDesc& descBuf, const string& expectedSha256) {
  try {
    string lower = Global::toLower(fileName);
    //Read model file with no compression if it's directly named .txt or .bin
    if(Global::isSuffix(lower,".txt")) {
      bool binaryFloats = false;
      string uncompressed;
      string sha256Buf;
      FileUtils::loadFileIntoString(fileName,expectedSha256,uncompressed,&sha256Buf);
      NonCopyingStreamBuf uncompressedStreamBuf(uncompressed);
      std::istream uncompressedIn(&uncompressedStreamBuf);
      descBuf = ModelDesc(uncompressedIn,sha256Buf,binaryFloats);
    }
    else if(Global::isSuffix(lower,".bin")) {
      bool binaryFloats = true;
      string uncompressed;
      string sha256Buf;
      FileUtils::loadFileIntoString(fileName,expectedSha256,uncompressed,&sha256Buf);
      NonCopyingStreamBuf uncompressedStreamBuf(uncompressed);
      std::istream uncompressedIn(&uncompressedStreamBuf);
      descBuf = ModelDesc(uncompressedIn,sha256Buf,binaryFloats);
    }
    else if(Global::isSuffix(lower,".txt.gz") || Global::isSuffix(lower,".bin.gz") || Global::isSuffix(lower,".gz")) {
      string uncompressed;
      string sha256Buf;
      FileUtils::uncompressAndLoadFileIntoString(fileName,expectedSha256,uncompressed,&sha256Buf);

      bool binaryFloats = !Global::isSuffix(lower,".txt.gz");
      try {
        //Now, initialize an istream to read from the string
        NonCopyingStreamBuf uncompressedStreamBuf(uncompressed);
        std::istream uncompressedIn(&uncompressedStreamBuf);
        //And read in the model desc
        descBuf = ModelDesc(uncompressedIn,sha256Buf,binaryFloats);
      }
      catch(const StringError& e) {
        //On failure, try again to read as a .txt.gz file if the extension was ambiguous
        bool tryAgain = binaryFloats && !Global::isSuffix(lower,".bin.gz");
        if(!tryAgain)
          throw;
        else {
          binaryFloats = false;
          try {
            NonCopyingStreamBuf uncompressedStreamBuf(uncompressed);
            std::istream uncompressedIn(&uncompressedStreamBuf);
            descBuf = ModelDesc(uncompressedIn,sha256Buf,binaryFloats);
          }
          catch(const StringError& e2) {
            throw StringError(string("Could neither parse .gz model as .txt.gz model nor as .bin.gz model, errors were:\n") + e2.what() + "\n" + e.what());
          }
        }
      }
    }
    else {
      throw StringError("Model file should end with .txt, .bin, .txt.gz, .bin.gz, or possibly just .gz. (If it doesn't have one of these extensions already, it's probably the wrong file, renaming will probably NOT help).");
    }

    descBuf.transformToReduceActivations();
  }
  catch(const StringError& e) {
    throw StringError("Error loading or parsing model file " + fileName + ": " + e.what());
  }
}


Rules ModelDesc::getSupportedRules(const Rules& desiredRules, bool& supported) const {
  Rules rules = desiredRules;
  supported = true;
  if(modelVersion <= 6) {
    if(rules.koRule == Rules::KO_SIMPLE || rules.koRule == Rules::KO_SPIGHT) {
      rules.koRule = Rules::KO_SITUATIONAL;
      supported = false;
    }
    if(rules.scoringRule == Rules::SCORING_TERRITORY) {
      rules.scoringRule = Rules::SCORING_AREA;
      supported = false;
    }
    if(rules.taxRule != Rules::TAX_NONE) {
      rules.taxRule = Rules::TAX_NONE;
      supported = false;
    }
    if(rules.hasButton) {
      rules.hasButton = false;
      supported = false;
    }
  }
  else {
    if(rules.koRule == Rules::KO_SPIGHT) {
      rules.koRule = Rules::KO_SITUATIONAL;
      supported = false;
    }
    if(rules.hasButton && rules.scoringRule != Rules::SCORING_AREA) {
      rules.hasButton = false;
      supported = false;
    }
  }

  return rules;
}
