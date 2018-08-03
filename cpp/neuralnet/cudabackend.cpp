
#ifdef USE_CUDA_BACKEND

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <fstream>

#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"

void NeuralNet::globalInitialize(
  const string& tensorflowGpuVisibleDeviceList,
  double tensorflowPerProcessGpuMemoryFraction
) {
  (void)tensorflowGpuVisibleDeviceList;
  (void)tensorflowPerProcessGpuMemoryFraction;
  //Empty for cudnn backend
}

void NeuralNet::globalCleanup() {
  cudaDeviceReset();
}

static void checkWeightFinite(float f, const string& name) {
  if(!isfinite(f))
    throw StringError(name + ": Nan or infinite neural net weight or parameter");
}
#define CHECKFINITE(x,name) { checkWeightFinite((x),name); }


struct CudaHandles {
  cublasHandle_t cublas;
  cudnnHandle_t cudnn;

  CudaHandles() {
    CUBLAS_ERR("CudaHandles",cublasCreate(&cublas));
    CUDNN_ERR("CudaHandles",cudnnCreate(&cudnn));
  }

  ~CudaHandles() {
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
  }

  CudaHandles(const CudaHandles&) = delete;
  CudaHandles& operator=(const CudaHandles&) = delete;
};



//---------------------------------------------------------------------------------

struct ConvLayerDesc {
  string name;
  int convYSize;
  int convXSize;
  int inChannels;
  int outChannels;
  int dilationY;
  int dilationX;
  vector<float> weights;

  ConvLayerDesc() {}

  ConvLayerDesc(istream& in) {
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

    //Model file order is y,x,ic,oc
    //Cuda's order is oc,ic,y,x
    int numWeights = convYSize * convXSize * inChannels * outChannels;
    weights.resize(numWeights);
    int ocStride = convYSize * convXSize * inChannels;
    int icStride = convYSize * convXSize;
    int yStride = convXSize;
    int xStride = 1;

    for(int y = 0; y < convYSize; y++) {
      for(int x = 0; x < convXSize; x++) {
        for(int ic = 0; ic < inChannels; ic++) {
          for(int oc = 0; oc < outChannels; oc++) {
            float w;
            in >> w;
            CHECKFINITE(w,name);
            weights[oc * ocStride + ic * icStride + y * yStride + x * xStride] = w;
          }
        }
      }
    }
    if(in.fail())
      throw StringError(name + ": convlayer failed to expected number of float weights");
  }

  ConvLayerDesc(const ConvLayerDesc&) = delete;
  ConvLayerDesc& operator=(const ConvLayerDesc&) = delete;

  ConvLayerDesc(ConvLayerDesc&& other) {
    *this = std::move(other);
  }

  ConvLayerDesc& operator=(ConvLayerDesc&& other) {
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

};

struct ConvLayer {
  string name;
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
  cudnnConvolutionFwdAlgo_t* convolutionAlgorithms; //array of one for each batch size
  float* filterBuf;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(
    CudaHandles* cudaHandles,
    const ConvLayerDesc* desc,
    int maxBatchSize,
    const cudnnTensorDescriptor_t* inputDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* outputDescriptors //array of one for each batch size
  ) {
    name = desc->name;
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int inChannels = desc->inChannels;
    int outChannels = desc->outChannels;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    CUDNN_ERR(name.c_str(),cudnnCreateFilterDescriptor(&filterDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetFilter4dDescriptor(
      filterDescriptor,
      CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW,
      outChannels,
      inChannels,
      convYSize,
      convXSize
    ));

    int yStride = 1;
    int xStride = 1;

    CUDNN_ERR(name.c_str(),cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetConvolution2dDescriptor(
      convolutionDescriptor,
      paddingY,
      paddingX,
      yStride,
      xStride,
      dilationY,
      dilationX,
      CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT
    ));

    convolutionAlgorithms = new cudnnConvolutionFwdAlgo_t[maxBatchSize];
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {

      const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];
      const cudnnTensorDescriptor_t& outputDescriptor = outputDescriptors[batchSize-1];

      size_t bytesMemoryLimit = 0;
      CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm(
        cudaHandles->cudnn,
        inputDescriptor,
        filterDescriptor,
        convolutionDescriptor,
        outputDescriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        bytesMemoryLimit,
        &(convolutionAlgorithms[batchSize-1])
      ));
    }

    assert(desc->weights.size() == convYSize * convXSize * inChannels * outChannels);
    size_t filterBytes = sizeof(float) * convYSize * convXSize * inChannels * outChannels;

    CUDA_ERR(name.c_str(),cudaMalloc(&filterBuf, filterBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(filterBuf, desc->weights.data(), filterBytes, cudaMemcpyHostToDevice));

  }

  ~ConvLayer() {
    cudaFree(filterBuf);
    cudnnDestroyFilterDescriptor(filterDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    delete[] convolutionAlgorithms;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& inputDescriptor,
    const cudnnTensorDescriptor_t& outputDescriptor,
    int batchSize
  ) const {
    size_t workspaceBytes = 0;
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptor,
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptor,
      convolutionAlgorithms[batchSize-1],
      &workspaceBytes
    ));

    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& inputDescriptor,
    const cudnnTensorDescriptor_t& outputDescriptor,
    int batchSize,
    float* inputBuf,
    float* outputBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptor,
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      convolutionAlgorithms[batchSize-1],
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptor,
      outputBuf
    ));
  }

};


//---------------------------------------------------------------------------------

struct BNLayerDesc {
  string name;
  int numChannels;
  float epsilon;
  bool hasScale;
  bool hasBias;
  vector<float> mean;
  vector<float> variance;
  vector<float> scale;
  vector<float> bias;

  BNLayerDesc() {}

  BNLayerDesc(istream& in) {
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

    float w;
    mean.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      in >> w;
      CHECKFINITE(w,name);
      mean[c] = w;
    }
    variance.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      in >> w;
      CHECKFINITE(w,name);
      variance[c] = w;
    }
    scale.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      if(hasScale) in >> w; else w = 1.0;
      CHECKFINITE(w,name);
      scale[c] = w;
    }
    bias.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      if(hasBias) in >> w; else w = 1.0;
      CHECKFINITE(w,name);
      bias[c] = w;
    }

    if(in.fail())
      throw StringError(name + ": bnlayer failed to parse expected number of batch norm mean, variance, bias, scale values");
  }

  BNLayerDesc(const BNLayerDesc&) = delete;
  BNLayerDesc& operator=(const BNLayerDesc&) = delete;

  BNLayerDesc(BNLayerDesc&& other) {
    *this = std::move(other);
  }

  BNLayerDesc& operator=(BNLayerDesc&& other) {
    name = std::move(other.name);
    numChannels = other.numChannels;
    epsilon = other.epsilon;
    hasScale = other.hasScale;
    hasBias = other.hasBias;
    mean = std::move(other.mean);
    variance = std::move(other.variance);
    scale = std::move(other.scale);
    bias = std::move(other.bias);
    return *this;
  }

};

struct BNLayer {
  string name;
  int numChannels;
  float epsilon;
  cudnnTensorDescriptor_t bufDescriptor;
  float* meanBuf;
  float* varianceBuf;
  float* scaleBuf;
  float* biasBuf;

  BNLayer() = delete;
  BNLayer(const BNLayer&) = delete;
  BNLayer& operator=(const BNLayer&) = delete;

  BNLayer(
    CudaHandles* cudaHandles,
    const BNLayerDesc* desc
  ) {
    (void)cudaHandles;

    name = desc->name;
    numChannels = desc->numChannels;
    epsilon = desc->epsilon;

    CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&bufDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
      bufDescriptor,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      1,
      numChannels,
      1,
      1
    ));

    size_t bufBytes = sizeof(float) * numChannels;

    assert(desc->mean.size() == numChannels);
    CUDA_ERR(name.c_str(),cudaMalloc(&meanBuf, bufBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(meanBuf, desc->mean.data(), bufBytes, cudaMemcpyHostToDevice));

    assert(desc->variance.size() == numChannels);
    CUDA_ERR(name.c_str(),cudaMalloc(&varianceBuf, bufBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(varianceBuf, desc->variance.data(), bufBytes, cudaMemcpyHostToDevice));

    assert(desc->scale.size() == numChannels);
    CUDA_ERR(name.c_str(),cudaMalloc(&scaleBuf, bufBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(scaleBuf, desc->scale.data(), bufBytes, cudaMemcpyHostToDevice));

    assert(desc->bias.size() == numChannels);
    CUDA_ERR(name.c_str(),cudaMalloc(&biasBuf, bufBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(biasBuf, desc->bias.data(), bufBytes, cudaMemcpyHostToDevice));
  }

  ~BNLayer() {
    cudaFree(meanBuf);
    cudaFree(varianceBuf);
    cudaFree(scaleBuf);
    cudaFree(biasBuf);
    cudnnDestroyTensorDescriptor(bufDescriptor);
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& inputDescriptor,
    const cudnnTensorDescriptor_t& outputDescriptor,
    float* inputBuf,
    float* outputBuf
  ) const {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_ERR(name.c_str(),cudnnBatchNormalizationForwardInference(
      cudaHandles->cudnn,
      CUDNN_BATCHNORM_SPATIAL,
      &alpha,
      &beta,
      inputDescriptor,
      inputBuf,
      outputDescriptor,
      outputBuf,
      bufDescriptor,
      scaleBuf,
      biasBuf,
      meanBuf,
      varianceBuf,
      epsilon
    ));
  }

};


//---------------------------------------------------------------------------------

struct ActivationLayerDesc {
  string name;

  ActivationLayerDesc() {}

  ActivationLayerDesc(istream& in) {
    in >> name;
  }

  ActivationLayerDesc(const ActivationLayerDesc&) = delete;
  ActivationLayerDesc& operator=(const ActivationLayerDesc&) = delete;

  ActivationLayerDesc(ActivationLayerDesc&& other) {
    *this = std::move(other);
  }

  ActivationLayerDesc& operator=(ActivationLayerDesc&& other) {
    name = std::move(other.name);
    return *this;
  }

};

struct ActivationLayer {
  string name;
  cudnnActivationDescriptor_t activationDescriptor;

  ActivationLayer() = delete;
  ActivationLayer(const ActivationLayer&) = delete;
  ActivationLayer& operator=(const ActivationLayer&) = delete;

  ActivationLayer(
    CudaHandles* cudaHandles,
    const ActivationLayerDesc* desc
  ) {
    (void)cudaHandles;
    name = desc->name;

    CUDNN_ERR(name.c_str(),cudnnCreateActivationDescriptor(&activationDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetActivationDescriptor(
      activationDescriptor,
      CUDNN_ACTIVATION_RELU,
      CUDNN_PROPAGATE_NAN,
      0.0
    ));
  }

  ~ActivationLayer() {
    cudnnDestroyActivationDescriptor(activationDescriptor);
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& inputDescriptor,
    const cudnnTensorDescriptor_t& outputDescriptor,
    float* inputBuf,
    float* outputBuf
  ) const {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUDNN_ERR(name.c_str(),cudnnActivationForward(
      cudaHandles->cudnn,
      activationDescriptor,
      &alpha,
      inputDescriptor,
      inputBuf,
      &beta,
      outputDescriptor,
      outputBuf
    ));
  }

};


//---------------------------------------------------------------------------------

struct MatMulLayerDesc {
  string name;
  int inChannels;
  int outChannels;
  vector<float> weights;

  MatMulLayerDesc() {}

  MatMulLayerDesc(istream& in) {
    in >> name;
    in >> inChannels;
    in >> outChannels;

    if(in.fail())
      throw StringError(name + ": matmullayer failed to parse num channels");
    if(inChannels <= 0 || outChannels <= 0)
      throw StringError(name + ": number of in and out channels must be positive");

    //Model file order is ic,oc
    //Cublas order used is also ic,oc since we transpose
    int numWeights = inChannels * outChannels;
    weights.resize(numWeights);
    int icStride = outChannels;
    int ocStride = 1;

    for(int ic = 0; ic < inChannels; ic++) {
      for(int oc = 0; oc < outChannels; oc++) {
        float w;
        in >> w;
        CHECKFINITE(w,name);
        weights[oc * ocStride + ic * icStride] = w;
      }
    }
    if(in.fail())
      throw StringError(name + ": matmullayer failed to parse expected number of matmul weights");
  }

  MatMulLayerDesc(const MatMulLayerDesc&) = delete;
  MatMulLayerDesc& operator=(const MatMulLayerDesc&) = delete;

  MatMulLayerDesc(MatMulLayerDesc&& other) {
    *this = std::move(other);
  }

  MatMulLayerDesc& operator=(MatMulLayerDesc&& other) {
    name = std::move(other.name);
    inChannels = other.inChannels;
    outChannels = other.outChannels;
    weights = std::move(other.weights);
    return *this;
  }

};

struct MatMulLayer {
  string name;
  int inChannels;
  int outChannels;
  float* matBuf;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(
    CudaHandles* cudaHandles,
    const MatMulLayerDesc* desc
  ) {
    (void)cudaHandles;
    name = desc->name;
    inChannels = desc->inChannels;
    outChannels = desc->outChannels;

    assert(desc->weights.size() == inChannels * outChannels);
    size_t matBytes = sizeof(float) * inChannels * outChannels;

    CUDA_ERR(name.c_str(),cudaMalloc(&matBuf, matBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(matBuf, desc->weights.data(), matBytes, cudaMemcpyHostToDevice));
  }

  ~MatMulLayer() {
    cudaFree(matBuf);
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles
  ) const {
    (void)cudaHandles;
    size_t workspaceBytes = 0;
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    float* inputBuf,
    float* outputBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) const {
    (void)workspaceBuf;
    (void)workspaceBytes;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_ERR(name.c_str(),cublasSgemm(
      cudaHandles->cublas,
      CUBLAS_OP_T,
      CUBLAS_OP_T,
      batchSize,outChannels,inChannels,
      &alpha,
      inputBuf,inChannels,
      matBuf,outChannels,
      &beta,
      outputBuf,batchSize
    ));
  }

};

//---------------------------------------------------------------------------------

struct MatBiasLayerDesc {
  string name;
  int numChannels;
  vector<float> weights;

  MatBiasLayerDesc() {}

  MatBiasLayerDesc(istream& in) {
    in >> name;
    in >> numChannels;

    if(in.fail())
      throw StringError(name + ": matbiaslayer failed to parse num channels");
    if(numChannels <= 0)
      throw StringError(name + ": number of channels must be positive");

    weights.resize(numChannels);

    for(int c = 0; c < numChannels; c++) {
      float w;
      in >> w;
      CHECKFINITE(w,name);
      weights[c] = w;
    }
    if(in.fail())
      throw StringError(name + ": matbiaslayer failed to parse expected number of matbias weights");
  }

  MatBiasLayerDesc(const MatBiasLayerDesc&) = delete;
  MatBiasLayerDesc& operator=(const MatBiasLayerDesc&) = delete;

  MatBiasLayerDesc(MatBiasLayerDesc&& other) {
    *this = std::move(other);
  }

  MatBiasLayerDesc& operator=(MatBiasLayerDesc&& other) {
    name = std::move(other.name);
    numChannels = other.numChannels;
    weights = std::move(other.weights);
    return *this;
  }
};

struct MatBiasLayer {
  string name;
  int numChannels;
  float* biasBuf;
  float* maxBatchSizeOnesBuf;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(
    CudaHandles* cudaHandles,
    const MatBiasLayerDesc* desc,
    int maxBatchSize
  ) {
    (void)cudaHandles;
    name = desc->name;
    numChannels = desc->numChannels;

    assert(desc->weights.size() == numChannels);
    size_t biasBytes = sizeof(float) * numChannels;

    size_t maxBatchSizeOnesBytes = sizeof(float) * maxBatchSize;
    float maxBatchSizeOnesArr[maxBatchSize];
    for(int i = 0; i<maxBatchSize; i++)
      maxBatchSizeOnesArr[i] = 1.0f;

    CUDA_ERR(name.c_str(),cudaMalloc(&biasBuf, biasBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(biasBuf, desc->weights.data(), biasBytes, cudaMemcpyHostToDevice));
    CUDA_ERR(name.c_str(),cudaMalloc(&maxBatchSizeOnesBuf, maxBatchSizeOnesBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(maxBatchSizeOnesBuf, maxBatchSizeOnesArr, maxBatchSizeOnesBytes, cudaMemcpyHostToDevice));
  }

  ~MatBiasLayer() {
    cudaFree(biasBuf);
    cudaFree(maxBatchSizeOnesBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    float* matBuf
  ) const {
    const float alpha = 1.0f;
    CUBLAS_ERR(name.c_str(),cublasSger(
      cudaHandles->cublas,
      numChannels,
      batchSize,
      &alpha,
      biasBuf,
      1,
      maxBatchSizeOnesBuf,
      1,
      matBuf,
      numChannels
    ));
  }

};



//---------------------------------------------------------------------------------

struct ResidualBlockDesc {
  string name;
  BNLayerDesc preBN;
  ActivationLayerDesc preActivation;
  ConvLayerDesc regularConv;
  BNLayerDesc midBN;
  ActivationLayerDesc midActivation;
  ConvLayerDesc finalConv;

  ResidualBlockDesc() {}

  ResidualBlockDesc(istream& in) {
    in >> name;
    if(in.fail())
      throw StringError(name + ": res block failed to parse name");

    preBN = BNLayerDesc(in);
    preActivation = ActivationLayerDesc(in);
    regularConv = ConvLayerDesc(in);
    midBN = BNLayerDesc(in);
    midActivation = ActivationLayerDesc(in);
    finalConv = ConvLayerDesc(in);

    if(preBN.numChannels != regularConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": preBN.numChannels (%d) != regularConv.inChannels (%d)", preBN.numChannels, regularConv.inChannels
      ));
    if(midBN.numChannels != regularConv.outChannels)
      throw StringError(name+Global::strprintf(
        ": midBN.numChannels (%d) != regularConv.outChannels (%d)", midBN.numChannels, regularConv.outChannels
      ));
    if(midBN.numChannels != finalConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": midBN.numChannels (%d) != finalConv.inChannels (%d)", midBN.numChannels, finalConv.inChannels
      ));

    if(in.fail())
      throw StringError(name + ": res block parse failure (istream fail() return true)");
  }

  ResidualBlockDesc(const ResidualBlockDesc&) = delete;
  ResidualBlockDesc& operator=(const ResidualBlockDesc&) = delete;

  ResidualBlockDesc(ResidualBlockDesc&& other) {
    *this = std::move(other);
  }

  ResidualBlockDesc& operator=(ResidualBlockDesc&& other) {
    name = std::move(other.name);
    preBN = std::move(other.preBN);
    preActivation = std::move(other.preActivation);
    regularConv = std::move(other.regularConv);
    midBN = std::move(other.midBN);
    midActivation = std::move(other.midActivation);
    finalConv = std::move(other.finalConv);
    return *this;
  }

};

struct ResidualBlock {
  string name;
  BNLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  BNLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  int xSize;
  int ySize;
  int regularChannels;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(
    CudaHandles* cudaHandles,
    const ResidualBlockDesc* desc,
    int maxBatchSize,
    int xS,
    int yS,
    const cudnnTensorDescriptor_t* trunkDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* midInDescriptors //array of one for each batch size
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN),
     preActivation(cudaHandles,&desc->preActivation),
     regularConv(cudaHandles,&desc->regularConv,maxBatchSize,trunkDescriptors,midInDescriptors),
     midBN(cudaHandles,&desc->midBN),
     midActivation(cudaHandles,&desc->midActivation),
     finalConv(cudaHandles,&desc->finalConv,maxBatchSize,midInDescriptors,trunkDescriptors),
     xSize(xS),
     ySize(yS),
     regularChannels(desc->regularConv.outChannels)
  {
  }

  ~ResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    const cudnnTensorDescriptor_t& midInDescriptor,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,trunkDescriptor,midInDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = finalConv.requiredWorkspaceBytes(cudaHandles,midInDescriptor,trunkDescriptor,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    const cudnnTensorDescriptor_t& midInDescriptor,
    int batchSize,
    int trunkBufSize,
    float* trunkInBuf,
    float* trunkOutBuf,
    float* midInBuf,
    float* midScratchBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) const {
    preBN.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkInBuf,trunkOutBuf);
    preActivation.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkOutBuf,trunkOutBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,midInDescriptor,batchSize,trunkOutBuf,midInBuf,workspaceBuf,workspaceBytes);
    midBN.apply(cudaHandles,midInDescriptor,midInDescriptor,midInBuf,midScratchBuf);
    midActivation.apply(cudaHandles,midInDescriptor,midInDescriptor,midScratchBuf,midScratchBuf);
    finalConv.apply(cudaHandles,midInDescriptor,trunkDescriptor,batchSize,midScratchBuf,trunkOutBuf,workspaceBuf,workspaceBytes);

    const float alpha = 1.0f;
    CUBLAS_ERR(name.c_str(),cublasSaxpy(cudaHandles->cublas,trunkBufSize,&alpha,trunkInBuf,1,trunkOutBuf,1));
  }

};


//-----------------------------------------------------------------------------

struct DilatedResidualBlockDesc {
  string name;
  BNLayerDesc preBN;
  ActivationLayerDesc preActivation;
  ConvLayerDesc regularConv;
  ConvLayerDesc dilatedConv;
  BNLayerDesc midBN;
  ActivationLayerDesc midActivation;
  ConvLayerDesc finalConv;

  DilatedResidualBlockDesc() {}

  DilatedResidualBlockDesc(istream& in) {
    in >> name;
    if(in.fail())
      throw StringError(name + ": dilated res block failed to parse name");

    preBN = BNLayerDesc(in);
    preActivation = ActivationLayerDesc(in);
    regularConv = ConvLayerDesc(in);
    dilatedConv = ConvLayerDesc(in);
    midBN = BNLayerDesc(in);
    midActivation = ActivationLayerDesc(in);
    finalConv = ConvLayerDesc(in);

    if(preBN.numChannels != regularConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": preBN.numChannels (%d) != regularConv.inChannels (%d)", preBN.numChannels, regularConv.inChannels
      ));
    if(preBN.numChannels != dilatedConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": preBN.numChannels (%d) != dilatedConv.inChannels (%d)", preBN.numChannels, dilatedConv.inChannels
      ));
    if(midBN.numChannels != regularConv.outChannels + dilatedConv.outChannels)
      throw StringError(name+Global::strprintf(
        ": midBN.numChannels (%d) != regularConv.outChannels (%d) + dilatedConv.outChannels (%d)", midBN.numChannels, regularConv.outChannels, dilatedConv.outChannels
      ));
    if(midBN.numChannels != finalConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": midBN.numChannels (%d) != finalConv.inChannels (%d)", midBN.numChannels, finalConv.inChannels
      ));

    if(in.fail())
      throw StringError(name + ": dilated res block parse failure (istream fail() return true)");
  }

  DilatedResidualBlockDesc(const DilatedResidualBlockDesc&) = delete;
  DilatedResidualBlockDesc& operator=(const DilatedResidualBlockDesc&) = delete;

  DilatedResidualBlockDesc(DilatedResidualBlockDesc&& other) {
    *this = std::move(other);
  }

  DilatedResidualBlockDesc& operator=(DilatedResidualBlockDesc&& other) {
    name = std::move(other.name);
    preBN = std::move(other.preBN);
    preActivation = std::move(other.preActivation);
    regularConv = std::move(other.regularConv);
    dilatedConv = std::move(other.dilatedConv);
    midBN = std::move(other.midBN);
    midActivation = std::move(other.midActivation);
    finalConv = std::move(other.finalConv);
    return *this;
  }

};

struct DilatedResidualBlock {
  string name;
  BNLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  ConvLayer dilatedConv;
  BNLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  int xSize;
  int ySize;
  int regularChannels;
  int dilatedChannels;

  DilatedResidualBlock() = delete;
  DilatedResidualBlock(const DilatedResidualBlock&) = delete;
  DilatedResidualBlock& operator=(const DilatedResidualBlock&) = delete;

  DilatedResidualBlock(
    CudaHandles* cudaHandles,
    const DilatedResidualBlockDesc* desc,
    int maxBatchSize,
    int xS,
    int yS,
    const cudnnTensorDescriptor_t* trunkDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* regularOutDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* dilatedOutDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* midInDescriptors //array of one for each batch size
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN),
     preActivation(cudaHandles,&desc->preActivation),
     regularConv(cudaHandles,&desc->regularConv,maxBatchSize,trunkDescriptors,regularOutDescriptors),
     dilatedConv(cudaHandles,&desc->dilatedConv,maxBatchSize,trunkDescriptors,dilatedOutDescriptors),
     midBN(cudaHandles,&desc->midBN),
     midActivation(cudaHandles,&desc->midActivation),
     finalConv(cudaHandles,&desc->finalConv,maxBatchSize,midInDescriptors,trunkDescriptors),
     xSize(xS),
     ySize(yS),
     regularChannels(desc->regularConv.outChannels),
     dilatedChannels(desc->dilatedConv.outChannels)
  {
  }

  ~DilatedResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    const cudnnTensorDescriptor_t& regularOutDescriptor,
    const cudnnTensorDescriptor_t& dilatedOutDescriptor,
    const cudnnTensorDescriptor_t& midInDescriptor,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,trunkDescriptor,regularOutDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = dilatedConv.requiredWorkspaceBytes(cudaHandles,trunkDescriptor,dilatedOutDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = finalConv.requiredWorkspaceBytes(cudaHandles,midInDescriptor,trunkDescriptor,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    const cudnnTensorDescriptor_t& regularOutDescriptor,
    const cudnnTensorDescriptor_t& dilatedOutDescriptor,
    const cudnnTensorDescriptor_t& midInDescriptor,
    int batchSize,
    int trunkBufSize,
    float* trunkInBuf,
    float* trunkOutBuf,
    float* regularOutBuf,
    float* dilatedOutBuf,
    float* midInBuf,
    float* midScratchBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) const {
    preBN.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkInBuf,trunkOutBuf);
    preActivation.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkOutBuf,trunkOutBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,regularOutDescriptor,batchSize,trunkOutBuf,regularOutBuf,workspaceBuf,workspaceBytes);
    dilatedConv.apply(cudaHandles,trunkDescriptor,dilatedOutDescriptor,batchSize,trunkOutBuf,dilatedOutBuf,workspaceBuf,workspaceBytes);
    customCudaChannelConcat(
      regularOutBuf,dilatedOutBuf,midInBuf,
      xSize*ySize*regularChannels,
      xSize*ySize*dilatedChannels,
      batchSize
    );
    midBN.apply(cudaHandles,midInDescriptor,midInDescriptor,midInBuf,midScratchBuf);
    midActivation.apply(cudaHandles,midInDescriptor,midInDescriptor,midScratchBuf,midScratchBuf);
    finalConv.apply(cudaHandles,midInDescriptor,trunkDescriptor,batchSize,midScratchBuf,trunkOutBuf,workspaceBuf,workspaceBytes);

    const float alpha = 1.0f;
    CUBLAS_ERR(name.c_str(),cublasSaxpy(cudaHandles->cublas,trunkBufSize,&alpha,trunkInBuf,1,trunkOutBuf,1));
  }

};



//----------------------------------------------------------------------------

struct GlobalPoolingResidualBlockDesc {
  string name;
  BNLayerDesc preBN;
  ActivationLayerDesc preActivation;
  ConvLayerDesc regularConv;
  ConvLayerDesc gpoolConv;
  BNLayerDesc gpoolBN;
  ActivationLayerDesc gpoolActivation;
  MatMulLayerDesc gpoolToBiasMul;
  BNLayerDesc midBN;
  ActivationLayerDesc midActivation;
  ConvLayerDesc finalConv;

  GlobalPoolingResidualBlockDesc() {}

  GlobalPoolingResidualBlockDesc(istream& in) {
    in >> name;
    if(in.fail())
      throw StringError(name + ": gpool res block failed to parse name");

    preBN = BNLayerDesc(in);
    preActivation = ActivationLayerDesc(in);
    regularConv = ConvLayerDesc(in);
    gpoolConv = ConvLayerDesc(in);
    gpoolBN = BNLayerDesc(in);
    gpoolActivation = ActivationLayerDesc(in);
    gpoolToBiasMul = MatMulLayerDesc(in);
    midBN = BNLayerDesc(in);
    midActivation = ActivationLayerDesc(in);
    finalConv = ConvLayerDesc(in);

    if(preBN.numChannels != regularConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": preBN.numChannels (%d) != regularConv.inChannels (%d)", preBN.numChannels, regularConv.inChannels
      ));
    if(preBN.numChannels != gpoolConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": preBN.numChannels (%d) != gpoolConv.inChannels (%d)", preBN.numChannels, gpoolConv.inChannels
      ));
    if(gpoolBN.numChannels != gpoolConv.outChannels)
      throw StringError(name+Global::strprintf(
        ": gpoolBN.numChannels (%d) != gpoolConv.outChannels (%d)", gpoolBN.numChannels, gpoolConv.outChannels
      ));
    if(gpoolBN.numChannels * 2 != gpoolToBiasMul.inChannels)
      throw StringError(name+Global::strprintf(
        ": gpoolBN.numChannels * 2 (%d) != gpoolToBiasMul.inChannels (%d)", gpoolBN.numChannels * 2, gpoolToBiasMul.inChannels
      ));
    if(midBN.numChannels != regularConv.outChannels)
      throw StringError(name+Global::strprintf(
        ": midBN.numChannels (%d) != regularConv.outChannels (%d)", midBN.numChannels, regularConv.outChannels
      ));
    if(midBN.numChannels != gpoolToBiasMul.outChannels)
      throw StringError(name+Global::strprintf(
        ": midBN.numChannels (%d) != gpoolToBiasMul.outChannels (%d)", midBN.numChannels, gpoolToBiasMul.outChannels
      ));
    if(midBN.numChannels != finalConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": midBN.numChannels (%d) != finalConv.inChannels (%d)", midBN.numChannels, finalConv.inChannels
      ));

    if(in.fail())
      throw StringError(name + ": gpool res block parse failure (istream fail() return true)");

  }


  GlobalPoolingResidualBlockDesc(const GlobalPoolingResidualBlockDesc&) = delete;
  GlobalPoolingResidualBlockDesc& operator=(const GlobalPoolingResidualBlockDesc&) = delete;

  GlobalPoolingResidualBlockDesc(GlobalPoolingResidualBlockDesc&& other) {
    *this = std::move(other);
  }

  GlobalPoolingResidualBlockDesc& operator=(GlobalPoolingResidualBlockDesc&& other) {
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

};

struct GlobalPoolingResidualBlock {
  string name;
  BNLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  ConvLayer gpoolConv;
  BNLayer gpoolBN;
  ActivationLayer gpoolActivation;
  MatMulLayer gpoolToBiasMul;
  BNLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  int xSize;
  int ySize;
  int regularChannels;
  int gpoolChannels;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(
    CudaHandles* cudaHandles,
    const GlobalPoolingResidualBlockDesc* desc,
    int maxBatchSize,
    int xS,
    int yS,
    const cudnnTensorDescriptor_t* trunkDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* regularOutDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* gpoolOutDescriptors //array of one for each batch size
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN),
     preActivation(cudaHandles,&desc->preActivation),
     regularConv(cudaHandles,&desc->regularConv,maxBatchSize,trunkDescriptors,regularOutDescriptors),
     gpoolConv(cudaHandles,&desc->gpoolConv,maxBatchSize,trunkDescriptors,gpoolOutDescriptors),
     gpoolBN(cudaHandles,&desc->gpoolBN),
     gpoolActivation(cudaHandles,&desc->gpoolActivation),
     gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul),
     midBN(cudaHandles,&desc->midBN),
     midActivation(cudaHandles,&desc->midActivation),
     finalConv(cudaHandles,&desc->finalConv,maxBatchSize,regularOutDescriptors,trunkDescriptors),
     xSize(xS),
     ySize(yS),
     regularChannels(desc->regularConv.outChannels),
     gpoolChannels(desc->gpoolConv.outChannels)
  {
  }

  ~GlobalPoolingResidualBlock() {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    const cudnnTensorDescriptor_t& regularOutDescriptor,
    const cudnnTensorDescriptor_t& gpoolOutDescriptor,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,trunkDescriptor,regularOutDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolConv.requiredWorkspaceBytes(cudaHandles,trunkDescriptor,gpoolOutDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = finalConv.requiredWorkspaceBytes(cudaHandles,regularOutDescriptor,trunkDescriptor,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    const cudnnTensorDescriptor_t& regularOutDescriptor,
    const cudnnTensorDescriptor_t& gpoolOutDescriptor,
    const cudnnTensorDescriptor_t& gpoolBiasDescriptor,
    int batchSize,
    int trunkBufSize,
    float* trunkInBuf,
    float* trunkOutBuf,
    float* regularOutBuf,
    float* gpoolOutBuf,
    float* gpoolOutBuf2,
    float* gpoolMeanBuf,
    float* gpoolMaxBuf,
    float* gpoolConcatBuf,
    float* gpoolBiasBuf,
    float* regularScratchBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) const {
    preBN.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkInBuf,trunkOutBuf);
    preActivation.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkOutBuf,trunkOutBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,regularOutDescriptor,batchSize,trunkOutBuf,regularOutBuf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(cudaHandles,trunkDescriptor,gpoolOutDescriptor,batchSize,trunkOutBuf,gpoolOutBuf,workspaceBuf,workspaceBytes);
    gpoolBN.apply(cudaHandles,gpoolOutDescriptor,gpoolOutDescriptor,gpoolOutBuf,gpoolOutBuf2);
    gpoolActivation.apply(cudaHandles,gpoolOutDescriptor,gpoolOutDescriptor,gpoolOutBuf2,gpoolOutBuf2);

    customCudaPoolRowsSum(gpoolOutBuf2,gpoolMeanBuf,batchSize*gpoolChannels,xSize*ySize);
    customCudaPoolRowsMax(gpoolOutBuf2,gpoolMaxBuf,batchSize*gpoolChannels,xSize*ySize);
    const float meanScale = 1.0f / (xSize*ySize);
    CUBLAS_ERR(name.c_str(),cublasSscal(cudaHandles->cublas, batchSize*gpoolChannels, &meanScale, gpoolMeanBuf, 1));
    customCudaChannelConcat(
      gpoolMeanBuf,gpoolMaxBuf,gpoolConcatBuf,
      gpoolChannels,
      gpoolChannels,
      batchSize
    );
    gpoolToBiasMul.apply(cudaHandles,batchSize,gpoolConcatBuf,gpoolBiasBuf,workspaceBuf,workspaceBytes);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnAddTensor(cudaHandles->cudnn,&alpha,gpoolBiasDescriptor,gpoolBiasBuf,&beta,regularOutDescriptor,regularOutBuf);

    midBN.apply(cudaHandles,regularOutDescriptor,regularOutDescriptor,regularOutBuf,regularScratchBuf);
    midActivation.apply(cudaHandles,regularOutDescriptor,regularOutDescriptor,regularScratchBuf,regularScratchBuf);
    finalConv.apply(cudaHandles,regularOutDescriptor,trunkDescriptor,batchSize,regularScratchBuf,trunkOutBuf,workspaceBuf,workspaceBytes);

    CUBLAS_ERR(name.c_str(),cublasSaxpy(cudaHandles->cublas,trunkBufSize,&alpha,trunkInBuf,1,trunkOutBuf,1));
  }

};

//------------------------------------------------------------------------------

static const int ORDINARY_BLOCK_KIND = 0;
static const int DILATED_BLOCK_KIND = 1;
static const int GLOBAL_POOLING_BLOCK_KIND = 2;

struct TrunkDesc {
  string name;
  int numBlocks;
  int trunkNumChannels;
  int midNumChannels;     //Currently every plain residual block must have the same number of mid conv channels
  int regularNumChannels; //Currently every dilated or gpool residual block must have the same number of regular conv channels
  int dilatedNumChannels; //Currently every dilated residual block must have the same number of dilated conv channels
  int gpoolNumChannels;   //Currently every gpooling residual block must have the same number of gpooling conv channels
  ConvLayerDesc initialConv;
  vector<pair<int,void*>> blocks;
  BNLayerDesc trunkTipBN;
  ActivationLayerDesc trunkTipActivation;

  TrunkDesc() {}

  TrunkDesc(istream& in) {
    in >> name;
    in >> numBlocks;
    in >> trunkNumChannels;
    in >> midNumChannels;
    in >> regularNumChannels;
    in >> dilatedNumChannels;
    in >> gpoolNumChannels;

    if(in.fail())
      throw StringError(name + ": trunk failed to parse num blocks or various channel parameters");
    if(numBlocks < 1)
      throw StringError(name + ": trunk num blocks must be positive");
    if(trunkNumChannels <= 0 || midNumChannels <= 0 || regularNumChannels <= 0 || dilatedNumChannels <= 0 || gpoolNumChannels <= 0)
      throw StringError(name + ": all numbers of channels must be positive");
    if(midNumChannels != regularNumChannels + dilatedNumChannels)
      throw StringError(name + ": midNumChannels != regularNumChannels + dilatedNumChannels");

    initialConv = ConvLayerDesc(in);

    if(initialConv.outChannels != trunkNumChannels)
      throw StringError(name+Global::strprintf(
        ": %s initialConv.outChannels (%d) != trunkNumChannels (%d)", initialConv.name.c_str(), initialConv.outChannels, trunkNumChannels
        ));

    string kind;
    for(int i = 0; i<numBlocks; i++) {
      in >> kind;
      if(in.fail())
        throw StringError(name + ": failed to parse block kind");
      if(kind == "ordinary_block") {
        ResidualBlockDesc* desc = new ResidualBlockDesc(in);

        if(desc->preBN.numChannels != trunkNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s preBN.numChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->preBN.numChannels, trunkNumChannels
          ));
        if(desc->regularConv.outChannels != midNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s regularConv.outChannels (%d) != regularNumChannels+dilatedNumChannels (%d)",
            desc->name.c_str(), desc->regularConv.outChannels, regularNumChannels+dilatedNumChannels
          ));
        if(desc->regularConv.outChannels != midNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s regularConv.outChannels (%d) != midNumChannels (%d)", desc->name.c_str(), desc->regularConv.outChannels, midNumChannels
          ));
        if(desc->finalConv.outChannels != trunkNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s finalConv.outChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->finalConv.outChannels, trunkNumChannels
          ));

        blocks.push_back(make_pair(ORDINARY_BLOCK_KIND,(void*)desc));
      }
      else if(kind == "dilated_block") {
        DilatedResidualBlockDesc* desc = new DilatedResidualBlockDesc(in);

        if(desc->preBN.numChannels != trunkNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s preBN.numChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->preBN.numChannels, trunkNumChannels
          ));
        if(desc->regularConv.outChannels != regularNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s regularConv.outChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->regularConv.outChannels, regularNumChannels
          ));
        if(desc->dilatedConv.outChannels != dilatedNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s dilatedConv.outChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->dilatedConv.outChannels, dilatedNumChannels
          ));
        if(desc->finalConv.outChannels != trunkNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s finalConv.outChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->finalConv.outChannels, trunkNumChannels
          ));

        blocks.push_back(make_pair(DILATED_BLOCK_KIND,(void*)desc));
      }
      else if(kind == "gpool_block") {
        GlobalPoolingResidualBlockDesc* desc = new GlobalPoolingResidualBlockDesc(in);

        if(desc->preBN.numChannels != trunkNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s preBN.numChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->preBN.numChannels, trunkNumChannels
          ));
        if(desc->regularConv.outChannels != regularNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s regularConv.outChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->regularConv.outChannels, regularNumChannels
          ));
        if(desc->gpoolConv.outChannels != gpoolNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s gpoolConv.outChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->gpoolConv.outChannels, gpoolNumChannels
          ));
        if(desc->finalConv.outChannels != trunkNumChannels)
          throw StringError(name+Global::strprintf(
            ": %s finalConv.outChannels (%d) != trunkNumChannels (%d)", desc->name.c_str(), desc->finalConv.outChannels, trunkNumChannels
          ));

        blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND,(void*)desc));
      }
      else
        throw StringError(name + ": found unknown block kind: " + kind);

      if(in.fail())
        throw StringError(name + ": trunk istream fail after parsing block");
    }

    trunkTipBN = BNLayerDesc(in);
    trunkTipActivation = ActivationLayerDesc(in);

    if(trunkTipBN.numChannels != trunkNumChannels)
      throw StringError(name+Global::strprintf(
        ": trunkTipBN.numChannels (%d) != trunkNumChannels (%d)", trunkTipBN.numChannels, trunkNumChannels
      ));

    if(in.fail())
      throw StringError(name + ": trunk istream fail after parsing tip");
  }

  ~TrunkDesc() {
    for(int i = 0; i<blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlockDesc* desc = (ResidualBlockDesc*)blocks[i].second;
        delete desc;
      }
      else if(blocks[i].first == DILATED_BLOCK_KIND) {
        DilatedResidualBlockDesc* desc = (DilatedResidualBlockDesc*)blocks[i].second;
        delete desc;
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second;
        delete desc;
      }
    }
  }

  TrunkDesc(const TrunkDesc&) = delete;
  TrunkDesc& operator=(const TrunkDesc&) = delete;

  TrunkDesc(TrunkDesc&& other) {
    name = std::move(other.name);
    numBlocks = other.numBlocks;
    trunkNumChannels = other.trunkNumChannels;
    midNumChannels = other.midNumChannels;
    regularNumChannels = other.regularNumChannels;
    dilatedNumChannels = other.dilatedNumChannels;
    gpoolNumChannels = other.gpoolNumChannels;
    initialConv = std::move(other.initialConv);
    blocks = std::move(other.blocks);
    trunkTipBN = std::move(other.trunkTipBN);
    trunkTipActivation = std::move(other.trunkTipActivation);
  }

  TrunkDesc& operator=(TrunkDesc&& other) {
    name = std::move(other.name);
    numBlocks = other.numBlocks;
    trunkNumChannels = other.trunkNumChannels;
    midNumChannels = other.midNumChannels;
    regularNumChannels = other.regularNumChannels;
    dilatedNumChannels = other.dilatedNumChannels;
    gpoolNumChannels = other.gpoolNumChannels;
    initialConv = std::move(other.initialConv);
    blocks = std::move(other.blocks);
    trunkTipBN = std::move(other.trunkTipBN);
    trunkTipActivation = std::move(other.trunkTipActivation);
    return *this;
  }

};


struct Trunk {
  string name;
  int numBlocks;
  int trunkNumChannels;
  int midNumChannels;
  int regularNumChannels;
  int dilatedNumChannels;
  int gpoolNumChannels;

  int maxBatchSize;
  int xSize;
  int ySize;

  cudnnTensorDescriptor_t* trunkDescriptors;
  cudnnTensorDescriptor_t* regularOutDescriptors;
  cudnnTensorDescriptor_t* gpoolOutDescriptors;
  cudnnTensorDescriptor_t* gpoolBiasDescriptors;
  cudnnTensorDescriptor_t* dilatedOutDescriptors;
  cudnnTensorDescriptor_t* midInDescriptors;

  ConvLayer* initialConv;
  vector<pair<int,void*>> blocks;
  BNLayer* trunkTipBN;
  ActivationLayer* trunkTipActivation;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    CudaHandles* cudaHandles,
    const TrunkDesc* desc,
    int maxBatchSz,
    int xS,
    int yS,
    const cudnnTensorDescriptor_t* inputDescriptors
  ) {
    name = desc->name;
    numBlocks = desc->numBlocks;
    trunkNumChannels = desc->trunkNumChannels;
    midNumChannels = desc->midNumChannels;
    regularNumChannels = desc->regularNumChannels;
    dilatedNumChannels = desc->dilatedNumChannels;
    gpoolNumChannels = desc->gpoolNumChannels;

    maxBatchSize = maxBatchSz;
    xSize = xS;
    ySize = yS;

    trunkDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    regularOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    gpoolOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    gpoolBiasDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    dilatedOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    midInDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& trunkDescriptor = trunkDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& regularOutDescriptor = regularOutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& gpoolOutDescriptor = gpoolOutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& gpoolBiasDescriptor = gpoolBiasDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& dilatedOutDescriptor = dilatedOutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& midInDescriptor = midInDescriptors[batchSize-1];

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&trunkDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        trunkDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        trunkNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&regularOutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        regularOutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        regularNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&dilatedOutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        dilatedOutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        dilatedNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&gpoolOutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        gpoolOutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        gpoolNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&gpoolBiasDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        gpoolBiasDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        regularNumChannels,
        1,
        1
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&midInDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        midInDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        regularNumChannels+dilatedNumChannels,
        ySize,
        xSize
      ));
    }

    initialConv = new ConvLayer(cudaHandles,&desc->initialConv,maxBatchSize,inputDescriptors,trunkDescriptors);
    trunkTipBN = new BNLayer(cudaHandles,&desc->trunkTipBN);
    trunkTipActivation = new ActivationLayer(cudaHandles,&desc->trunkTipActivation);

    assert(desc->blocks.size() == numBlocks);
    for(int i = 0; i<numBlocks; i++) {
      if(desc->blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)desc->blocks[i].second;
        ResidualBlock* block = new ResidualBlock(
          cudaHandles,
          blockDesc,
          maxBatchSize,
          xSize,
          ySize,
          trunkDescriptors,
          midInDescriptors
        );
        blocks.push_back(make_pair(ORDINARY_BLOCK_KIND,(void*)block));
      }
      else if(desc->blocks[i].first == DILATED_BLOCK_KIND) {
        DilatedResidualBlockDesc* blockDesc = (DilatedResidualBlockDesc*)desc->blocks[i].second;
        DilatedResidualBlock* block = new DilatedResidualBlock(
          cudaHandles,
          blockDesc,
          maxBatchSize,
          xSize,
          ySize,
          trunkDescriptors,
          regularOutDescriptors,
          dilatedOutDescriptors,
          midInDescriptors
        );
        blocks.push_back(make_pair(DILATED_BLOCK_KIND,(void*)block));
      }
      else if(desc->blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)desc->blocks[i].second;
        GlobalPoolingResidualBlock* block = new GlobalPoolingResidualBlock(
          cudaHandles,
          blockDesc,
          maxBatchSize,
          xSize,
          ySize,
          trunkDescriptors,
          regularOutDescriptors,
          gpoolOutDescriptors
        );
        blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND,(void*)block));
      }
      else {
        assert(false);
      }
    }
  }

  ~Trunk()
  {
    for(int i = 0; i<blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlock* block = (ResidualBlock*)blocks[i].second;
        delete block;
      }
      else if(blocks[i].first == DILATED_BLOCK_KIND) {
        DilatedResidualBlock* block = (DilatedResidualBlock*)blocks[i].second;
        delete block;
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second;
        delete block;
      }
    }

    delete initialConv;
    delete trunkTipBN;
    delete trunkTipActivation;

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnDestroyTensorDescriptor(trunkDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(regularOutDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(dilatedOutDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(gpoolOutDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(gpoolBiasDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(midInDescriptors[batchSize-1]);
    }

    delete[] trunkDescriptors;
    delete[] regularOutDescriptors;
    delete[] dilatedOutDescriptors;
    delete[] gpoolOutDescriptors;
    delete[] gpoolBiasDescriptors;
    delete[] midInDescriptors;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& inputDescriptor,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    const cudnnTensorDescriptor_t& trunkDescriptor = trunkDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& regularOutDescriptor = regularOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& gpoolOutDescriptor = gpoolOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& dilatedOutDescriptor = dilatedOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& midInDescriptor = midInDescriptors[batchSize-1];

    b = initialConv->requiredWorkspaceBytes(cudaHandles,inputDescriptor,trunkDescriptor,batchSize);
    bytes = std::max(bytes,b);

    for(int i = 0; i<blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlock* block = (ResidualBlock*)blocks[i].second;
        b = block->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,midInDescriptor,batchSize);
        bytes = std::max(bytes,b);
      }
      else if(blocks[i].first == DILATED_BLOCK_KIND) {
        DilatedResidualBlock* block = (DilatedResidualBlock*)blocks[i].second;
        b = block->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,regularOutDescriptor,dilatedOutDescriptor,midInDescriptor,batchSize);
        bytes = std::max(bytes,b);
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second;
        b = block->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,regularOutDescriptor,gpoolOutDescriptor,batchSize);
        bytes = std::max(bytes,b);
      }
      else {
        assert(false);
      }
    }
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& inputDescriptor,
    int batchSize,
    float* inputBuf,
    float* trunkScratchBuf,
    float* trunkOutBuf,
    float* regularOutBuf,
    float* dilatedOutBuf,
    float* midInBuf,
    float* midScratchBuf,
    float* gpoolOutBuf,
    float* gpoolOutBuf2,
    float* gpoolMeanBuf,
    float* gpoolMaxBuf,
    float* gpoolConcatBuf,
    float* gpoolBiasBuf,
    float* regularScratchBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) const {
    float* currentTrunkBuf = trunkScratchBuf;
    float* nextTrunkBuf = trunkOutBuf;

    const cudnnTensorDescriptor_t& trunkDescriptor = trunkDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& regularOutDescriptor = regularOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& gpoolOutDescriptor = gpoolOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& gpoolBiasDescriptor = gpoolBiasDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& dilatedOutDescriptor = dilatedOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& midInDescriptor = midInDescriptors[batchSize-1];

    int trunkBufSize = batchSize * trunkNumChannels * xSize * ySize;

    initialConv->apply(cudaHandles,inputDescriptor,trunkDescriptor,batchSize,inputBuf,currentTrunkBuf,workspaceBuf,workspaceBytes);

    for(int i = 0; i<blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlock* block = (ResidualBlock*)blocks[i].second;
        block->apply(
          cudaHandles,
          trunkDescriptor,
          midInDescriptor,
          batchSize,
          trunkBufSize,
          currentTrunkBuf,
          nextTrunkBuf,
          midInBuf,
          midScratchBuf,
          workspaceBuf,
          workspaceBytes
        );
        std::swap(currentTrunkBuf,nextTrunkBuf);
      }
      else if(blocks[i].first == DILATED_BLOCK_KIND) {
        DilatedResidualBlock* block = (DilatedResidualBlock*)blocks[i].second;
        block->apply(
          cudaHandles,
          trunkDescriptor,
          regularOutDescriptor,
          dilatedOutDescriptor,
          midInDescriptor,
          batchSize,
          trunkBufSize,
          currentTrunkBuf,
          nextTrunkBuf,
          regularOutBuf,
          dilatedOutBuf,
          midInBuf,
          midScratchBuf,
          workspaceBuf,
          workspaceBytes
        );
        std::swap(currentTrunkBuf,nextTrunkBuf);
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second;
        block->apply(
          cudaHandles,
          trunkDescriptor,
          regularOutDescriptor,
          gpoolOutDescriptor,
          gpoolBiasDescriptor,
          batchSize,
          trunkBufSize,
          currentTrunkBuf,
          nextTrunkBuf,
          regularOutBuf,
          gpoolOutBuf,
          gpoolOutBuf2,
          gpoolMeanBuf,
          gpoolMaxBuf,
          gpoolConcatBuf,
          gpoolBiasBuf,
          regularScratchBuf,
          workspaceBuf,
          workspaceBytes
        );
        std::swap(currentTrunkBuf,nextTrunkBuf);
      }
      else {
        assert(false);
      }
    }

    trunkTipBN->apply(cudaHandles,trunkDescriptor,trunkDescriptor,currentTrunkBuf,nextTrunkBuf);
    trunkTipActivation->apply(cudaHandles,trunkDescriptor,trunkDescriptor,currentTrunkBuf,trunkOutBuf);
  }

};


//------------------------------------------------------------------------------

struct PolicyHeadDesc {
  string name;
  ConvLayerDesc p1Conv;
  ConvLayerDesc g1Conv;
  BNLayerDesc g1BN;
  ActivationLayerDesc g1Activation;
  MatMulLayerDesc gpoolToBiasMul;
  BNLayerDesc p1BN;
  ActivationLayerDesc p1Activation;
  ConvLayerDesc p2Conv;
  MatMulLayerDesc gpoolToPassMul;

  PolicyHeadDesc() {}

  PolicyHeadDesc(istream& in) {
    in >> name;

    if(in.fail())
      throw StringError(name + ": policy head failed to parse name");

    p1Conv = ConvLayerDesc(in);
    g1Conv = ConvLayerDesc(in);
    g1BN = BNLayerDesc(in);
    g1Activation = ActivationLayerDesc(in);
    gpoolToBiasMul = MatMulLayerDesc(in);
    p1BN = BNLayerDesc(in);
    p1Activation = ActivationLayerDesc(in);
    p2Conv = ConvLayerDesc(in);
    gpoolToPassMul = MatMulLayerDesc(in);

    if(in.fail())
      throw StringError(name + ": policy head istream fail after parsing layers");

    if(p1Conv.outChannels != p1BN.numChannels)
      throw StringError(name+Global::strprintf(
        ": p1Conv.outChannels (%d) != p1BN.numChannels (%d)", p1Conv.outChannels, p1BN.numChannels
      ));
    if(g1Conv.outChannels != g1BN.numChannels)
      throw StringError(name+Global::strprintf(
        ": g1Conv.outChannels (%d) != g1BN.numChannels (%d)", g1Conv.outChannels, g1BN.numChannels
      ));
    if(gpoolToBiasMul.inChannels != g1BN.numChannels*2)
      throw StringError(name+Global::strprintf(
        ": gpoolToBiasMul.inChannels (%d) != g1BN.numChannels*2 (%d)", gpoolToBiasMul.inChannels, g1BN.numChannels*2
      ));
    if(gpoolToBiasMul.outChannels != p1BN.numChannels)
      throw StringError(name+Global::strprintf(
        ": gpoolToBiasMul.outChannels (%d) != p1BN.numChannels (%d)", gpoolToBiasMul.outChannels, p1BN.numChannels
      ));
    if(p2Conv.inChannels != p1BN.numChannels*2)
      throw StringError(name+Global::strprintf(
        ": p2Conv.inChannels (%d) != p1BN.numChannels*2 (%d)", p2Conv.inChannels, p1BN.numChannels*2
      ));
    if(p2Conv.outChannels != 1)
      throw StringError(name+Global::strprintf(
        ": p2Conv.outChannels (%d) != 1", p2Conv.outChannels
      ));
    if(gpoolToPassMul.inChannels != g1BN.numChannels*2)
      throw StringError(name+Global::strprintf(
        ": gpoolToPassMul.inChannels (%d) != g1BN.numChannels*2 (%d)", gpoolToPassMul.inChannels, g1BN.numChannels*2
      ));
    if(gpoolToPassMul.outChannels != 1)
      throw StringError(name+Global::strprintf(
        ": gpoolToPassMul.outChannels (%d) != 1", gpoolToPassMul.outChannels
      ));
  }

  ~PolicyHeadDesc() {
  }

  PolicyHeadDesc(const PolicyHeadDesc&) = delete;
  PolicyHeadDesc& operator=(const PolicyHeadDesc&) = delete;

  PolicyHeadDesc(PolicyHeadDesc&& other) {
    *this = std::move(other);
  }

  PolicyHeadDesc& operator=(PolicyHeadDesc&& other) {
    name = std::move(other.name);
    p1Conv = std::move(other.p1Conv);
    g1Conv = std::move(other.g1Conv);
    g1BN = std::move(other.g1BN);
    g1Activation = std::move(other.g1Activation);
    gpoolToBiasMul = std::move(other.gpoolToBiasMul);
    p1BN = std::move(other.p1BN);
    p1Activation = std::move(other.p1Activation);
    p2Conv = std::move(other.p2Conv);
    gpoolToPassMul = std::move(other.gpoolToPassMul);
    return *this;
  }

};

struct PolicyHead {
  string name;
  int maxBatchSize;
  int xSize;
  int ySize;
  int p1Channels;
  int g1Channels;
  int p2Channels;

  cudnnTensorDescriptor_t* p1OutDescriptors;
  cudnnTensorDescriptor_t* g1OutDescriptors;
  cudnnTensorDescriptor_t* g1BiasDescriptors;
  cudnnTensorDescriptor_t* p2InDescriptors;
  cudnnTensorDescriptor_t* p2OutDescriptors;

  ConvLayer* p1Conv;
  ConvLayer* g1Conv;
  BNLayer* g1BN;
  ActivationLayer* g1Activation;
  MatMulLayer* gpoolToBiasMul;
  BNLayer* p1BN;
  ActivationLayer* p1Activation;
  ConvLayer* p2Conv;
  MatMulLayer* gpoolToPassMul;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    CudaHandles* cudaHandles,
    const PolicyHeadDesc* desc,
    int maxBatchSz,
    int xS,
    int yS,
    const cudnnTensorDescriptor_t* trunkDescriptors
  ) {
    name = desc->name;
    maxBatchSize = maxBatchSz;
    xSize = xS;
    ySize = yS;
    p1Channels = desc->p1Conv.outChannels;
    g1Channels = desc->g1Conv.outChannels;
    p2Channels = desc->p2Conv.outChannels;

    p1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    g1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    g1BiasDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    p2InDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    p2OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& p1OutDescriptor = p1OutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& g1OutDescriptor = g1OutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& g1BiasDescriptor = g1BiasDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& p2InDescriptor = p2InDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& p2OutDescriptor = p2OutDescriptors[batchSize-1];

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&p1OutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        p1OutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->p1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&g1OutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        g1OutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->g1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&g1BiasDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        g1BiasDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->p1Conv.outChannels,
        1,
        1
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&p2InDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        p2InDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->p2Conv.inChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&p2OutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        p2OutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->p2Conv.outChannels,
        ySize,
        xSize
      ));
    }

    p1Conv = new ConvLayer(cudaHandles,&desc->p1Conv,maxBatchSize,trunkDescriptors,p1OutDescriptors);
    g1Conv = new ConvLayer(cudaHandles,&desc->g1Conv,maxBatchSize,trunkDescriptors,g1OutDescriptors);
    g1BN = new BNLayer(cudaHandles,&desc->g1BN);
    g1Activation = new ActivationLayer(cudaHandles,&desc->g1Activation);
    gpoolToBiasMul = new MatMulLayer(cudaHandles,&desc->gpoolToBiasMul);
    p1BN = new BNLayer(cudaHandles,&desc->p1BN);
    p1Activation = new ActivationLayer(cudaHandles,&desc->p1Activation);
    p2Conv = new ConvLayer(cudaHandles,&desc->p2Conv,maxBatchSize,p2InDescriptors,p2OutDescriptors);
    gpoolToPassMul = new MatMulLayer(cudaHandles,&desc->gpoolToPassMul);
  }

  ~PolicyHead()
  {
    delete p1Conv;
    delete g1Conv;
    delete g1BN;
    delete g1Activation;
    delete gpoolToBiasMul;
    delete p1BN;
    delete p1Activation;
    delete p2Conv;
    delete gpoolToPassMul;

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnDestroyTensorDescriptor(p1OutDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(g1OutDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(g1BiasDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(p2InDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(p2OutDescriptors[batchSize-1]);
    }

    delete[] p1OutDescriptors;
    delete[] g1OutDescriptors;
    delete[] g1BiasDescriptors;
    delete[] p2InDescriptors;
    delete[] p2OutDescriptors;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    const cudnnTensorDescriptor_t& p1OutDescriptor = p1OutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& g1OutDescriptor = g1OutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& p2InDescriptor = p2InDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& p2OutDescriptor = p2OutDescriptors[batchSize-1];

    b = p1Conv->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,p1OutDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = g1Conv->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,g1OutDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = p2Conv->requiredWorkspaceBytes(cudaHandles,p2InDescriptor,p2OutDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    int batchSize,
    float* trunkOutBuf,
    float* p1OutBuf,
    float* p1OutBuf2,
    float* g1OutBuf,
    float* g1OutBuf2,
    float* g1MeanBuf,
    float* g1MaxBuf,
    float* g1ConcatBuf,
    float* g1BiasBuf,
    float* p2InBuf,
    float* p2OutBuf,
    float* g1PassBuf,
    float* policyBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const cudnnTensorDescriptor_t& p1OutDescriptor = p1OutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& g1OutDescriptor = g1OutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& g1BiasDescriptor = g1BiasDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& p2InDescriptor = p2InDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& p2OutDescriptor = p2OutDescriptors[batchSize-1];

    p1Conv->apply(cudaHandles,trunkDescriptor,p1OutDescriptor,batchSize,trunkOutBuf,p1OutBuf,workspaceBuf,workspaceBytes);
    g1Conv->apply(cudaHandles,trunkDescriptor,g1OutDescriptor,batchSize,trunkOutBuf,g1OutBuf,workspaceBuf,workspaceBytes);
    g1BN->apply(cudaHandles,g1OutDescriptor,g1OutDescriptor,g1OutBuf,g1OutBuf2);
    g1Activation->apply(cudaHandles,g1OutDescriptor,g1OutDescriptor,g1OutBuf2,g1OutBuf2);

    customCudaPoolRowsSum(g1OutBuf2,g1MeanBuf,batchSize*g1Channels,xSize*ySize);
    customCudaPoolRowsMax(g1OutBuf2,g1MaxBuf,batchSize*g1Channels,xSize*ySize);
    const float meanScale = 1.0f / (xSize*ySize);
    CUBLAS_ERR(name.c_str(),cublasSscal(cudaHandles->cublas, batchSize*g1Channels, &meanScale, g1MeanBuf, 1));
    customCudaChannelConcat(
      g1MeanBuf,g1MaxBuf,g1ConcatBuf,
      g1Channels,
      g1Channels,
      batchSize
    );
    gpoolToBiasMul->apply(cudaHandles,batchSize,g1ConcatBuf,g1BiasBuf,workspaceBuf,workspaceBytes);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudnnAddTensor(cudaHandles->cudnn,&alpha,g1BiasDescriptor,g1BiasBuf,&beta,p1OutDescriptor,p1OutBuf);

    p1BN->apply(cudaHandles,p1OutDescriptor,p1OutDescriptor,p1OutBuf,p1OutBuf2);

    //Negate and concat for crelu
    CUDA_ERR(name.c_str(),cudaMemcpy(p1OutBuf,p1OutBuf2,batchSize*p1Channels*xSize*ySize*sizeof(float),cudaMemcpyDeviceToDevice));
    const float invScale = -1.0f;
    CUBLAS_ERR(name.c_str(),cublasSscal(cudaHandles->cublas, batchSize*p1Channels*xSize*ySize, &invScale, p1OutBuf, 1));
    customCudaChannelConcat(
      p1OutBuf2,p1OutBuf,p2InBuf,
      p1Channels*xSize*ySize,
      p1Channels*xSize*ySize,
      batchSize
    );

    p1Activation->apply(cudaHandles,p2InDescriptor,p2InDescriptor,p2InBuf,p2InBuf);
    p2Conv->apply(cudaHandles,p2InDescriptor,p2OutDescriptor,batchSize,p2InBuf,p2OutBuf,workspaceBuf,workspaceBytes);

    gpoolToPassMul->apply(cudaHandles,batchSize,g1ConcatBuf,g1PassBuf,workspaceBuf,workspaceBytes);

    customCudaChannelConcat(
      p2OutBuf,g1PassBuf,policyBuf,
      xSize*ySize,
      1,
      batchSize
    );

  }

};



//------------------------------------------------------------------------------


struct ValueHeadDesc {
  string name;
  ConvLayerDesc v1Conv;
  BNLayerDesc v1BN;
  ActivationLayerDesc v1Activation;
  MatMulLayerDesc v2Mul;
  MatBiasLayerDesc v2Bias;
  ActivationLayerDesc v2Activation;
  MatMulLayerDesc v3Mul;
  MatBiasLayerDesc v3Bias;

  ValueHeadDesc() {}

  ValueHeadDesc(istream& in) {
    in >> name;

    if(in.fail())
      throw StringError(name + ": value head failed to parse name");

    v1Conv = ConvLayerDesc(in);
    v1BN = BNLayerDesc(in);
    v1Activation = ActivationLayerDesc(in);
    v2Mul = MatMulLayerDesc(in);
    v2Bias = MatBiasLayerDesc(in);
    v2Activation = ActivationLayerDesc(in);
    v3Mul = MatMulLayerDesc(in);
    v3Bias = MatBiasLayerDesc(in);

    if(in.fail())
      throw StringError(name + ": value head istream fail after parsing layers");

    if(v1Conv.outChannels != v1BN.numChannels)
      throw StringError(name+Global::strprintf(
        ": v1Conv.outChannels (%d) != v1BN.numChannels (%d)", v1Conv.outChannels, v1BN.numChannels
      ));
    if(v2Mul.inChannels != v1BN.numChannels)
      throw StringError(name+Global::strprintf(
        ": v2Mul.inChannels (%d) != v1BN.numChannels (%d)", v2Mul.inChannels, v1BN.numChannels
      ));
    if(v2Mul.outChannels != v2Bias.numChannels)
      throw StringError(name+Global::strprintf(
        ": v2Mul.outChannels (%d) != v2Bias.numChannels (%d)", v2Mul.outChannels, v2Bias.numChannels
      ));
    if(v2Mul.outChannels*2 != v3Mul.inChannels)
      throw StringError(name+Global::strprintf(
        ": v2Mul.outChannels*2 (%d) != v3Mul.inChannels (%d)", v2Mul.outChannels*2, v3Mul.inChannels
      ));
    if(v3Mul.outChannels != 1)
      throw StringError(name+Global::strprintf(
        ": v3Mul.outChannels (%d) != 1", v3Mul.outChannels
      ));
    if(v3Bias.numChannels != 1)
      throw StringError(name+Global::strprintf(
        ": v3Bias.numChannels (%d) != 1", v3Bias.numChannels
      ));
  }

  ~ValueHeadDesc() {
  }

  ValueHeadDesc(const ValueHeadDesc&) = delete;
  ValueHeadDesc& operator=(const ValueHeadDesc&) = delete;

  ValueHeadDesc(ValueHeadDesc&& other) {
    *this = std::move(other);
  }

  ValueHeadDesc& operator=(ValueHeadDesc&& other) {
    name = std::move(other.name);
    v1Conv = std::move(other.v1Conv);
    v1BN = std::move(other.v1BN);
    v1Activation = std::move(other.v1Activation);
    v2Mul = std::move(other.v2Mul);
    v2Bias = std::move(other.v2Bias);
    v2Activation = std::move(other.v2Activation);
    v3Mul = std::move(other.v3Mul);
    v3Bias = std::move(other.v3Bias);
    return *this;
  }

};



struct ValueHead {
  string name;
  int maxBatchSize;
  int xSize;
  int ySize;
  int v1Channels;
  int v2Channels;
  int valueChannels;

  cudnnTensorDescriptor_t* v1OutDescriptors;
  cudnnTensorDescriptor_t* v3InDescriptors;

  ConvLayer* v1Conv;
  BNLayer* v1BN;
  ActivationLayer* v1Activation;
  MatMulLayer* v2Mul;
  MatBiasLayer* v2Bias;
  ActivationLayer* v2Activation;
  MatMulLayer* v3Mul;
  MatBiasLayer* v3Bias;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    CudaHandles* cudaHandles,
    const ValueHeadDesc* desc,
    int maxBatchSz,
    int xS,
    int yS,
    const cudnnTensorDescriptor_t* trunkDescriptors
  ) {
    name = desc->name;
    maxBatchSize = maxBatchSz;
    xSize = xS;
    ySize = yS;
    v1Channels = desc->v1Conv.outChannels;
    v2Channels = desc->v2Mul.outChannels;
    valueChannels = desc->v3Mul.outChannels;

    v1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    v3InDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& v1OutDescriptor = v1OutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& v3InDescriptor = v3InDescriptors[batchSize-1];

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&v1OutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        v1OutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->v1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&v3InDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        v3InDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->v2Mul.outChannels*2,
        1,
        1
      ));
    }

    v1Conv = new ConvLayer(cudaHandles,&desc->v1Conv,maxBatchSize,trunkDescriptors,v1OutDescriptors);
    v1BN = new BNLayer(cudaHandles,&desc->v1BN);
    v1Activation = new ActivationLayer(cudaHandles,&desc->v1Activation);
    v2Mul = new MatMulLayer(cudaHandles,&desc->v2Mul);
    v2Bias = new MatBiasLayer(cudaHandles,&desc->v2Bias,maxBatchSize);
    v2Activation = new ActivationLayer(cudaHandles,&desc->v2Activation);
    v3Mul = new MatMulLayer(cudaHandles,&desc->v3Mul);
    v3Bias = new MatBiasLayer(cudaHandles,&desc->v3Bias,maxBatchSize);
  }

  ~ValueHead()
  {
    delete v1Conv;
    delete v1BN;
    delete v1Activation;
    delete v2Mul;
    delete v2Bias;
    delete v2Activation;
    delete v3Mul;
    delete v3Bias;

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnDestroyTensorDescriptor(v1OutDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(v3InDescriptors[batchSize-1]);
    }

    delete[] v1OutDescriptors;
    delete[] v3InDescriptors;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    const cudnnTensorDescriptor_t& v1OutDescriptor = v1OutDescriptors[batchSize-1];

    b = v1Conv->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,v1OutDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = v2Mul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = v3Mul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    return bytes;
  }


  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    int batchSize,
    float* trunkOutBuf,
    float* v1OutBuf,
    float* v1OutBuf2,
    float* v1MeanBuf,
    float* v2OutBuf,
    float* v2OutBuf2,
    float* v3InBuf,
    float* valueBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const cudnnTensorDescriptor_t& v1OutDescriptor = v1OutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& v3InDescriptor = v3InDescriptors[batchSize-1];

    v1Conv->apply(cudaHandles,trunkDescriptor,v1OutDescriptor,batchSize,trunkOutBuf,v1OutBuf,workspaceBuf,workspaceBytes);
    v1BN->apply(cudaHandles,v1OutDescriptor,v1OutDescriptor,v1OutBuf,v1OutBuf2);
    v1Activation->apply(cudaHandles,v1OutDescriptor,v1OutDescriptor,v1OutBuf2,v1OutBuf2);

    customCudaPoolRowsSum(v1OutBuf2,v1MeanBuf,batchSize*v1Channels,xSize*ySize);
    const float meanScale = 1.0f / (xSize*ySize);
    CUBLAS_ERR(name.c_str(),cublasSscal(cudaHandles->cublas, batchSize*v1Channels, &meanScale, v1MeanBuf, 1));

    v2Mul->apply(cudaHandles,batchSize,v1MeanBuf,v2OutBuf,workspaceBuf,workspaceBytes);
    v2Bias->apply(cudaHandles,batchSize,v2OutBuf);

    //Negate and concat for crelu
    CUDA_ERR(name.c_str(),cudaMemcpy(v2OutBuf2,v2OutBuf,batchSize*v2Channels*sizeof(float),cudaMemcpyDeviceToDevice));
    const float invScale = -1.0f;
    CUBLAS_ERR(name.c_str(),cublasSscal(cudaHandles->cublas, batchSize*v2Channels, &invScale, v2OutBuf2, 1));
    customCudaChannelConcat(
      v2OutBuf,v2OutBuf2,v3InBuf,
      v2Channels,
      v2Channels,
      batchSize
    );

    v2Activation->apply(cudaHandles,v3InDescriptor,v3InDescriptor,v3InBuf,v3InBuf);

    v3Mul->apply(cudaHandles,batchSize,v3InBuf,valueBuf,workspaceBuf,workspaceBytes);
    v3Bias->apply(cudaHandles,batchSize,valueBuf);
  }

};


//------------------------------------------------------------------------------

struct ModelDesc {
  string name;
  int xSize;
  int ySize;
  int numInputChannels;

  TrunkDesc trunk;
  PolicyHeadDesc policyHead;
  ValueHeadDesc valueHead;

  ModelDesc() {}

  ModelDesc(istream& in) {
    in >> name;
    in >> xSize;
    in >> ySize;
    in >> numInputChannels;

    if(in.fail())
      throw StringError(name + ": model failed to parse name or xSize or ySize");
    if(xSize <= 0 || ySize <= 0)
      throw StringError(name + ": model xSize and ySize must be positive");
    if(numInputChannels <= 0)
      throw StringError(name + ": model numInputChannels must be positive");

    trunk = TrunkDesc(in);
    policyHead = PolicyHeadDesc(in);
    valueHead = ValueHeadDesc(in);

    if(in.fail())
      throw StringError(name + ": model desc istream fail after parsing model");

    if(numInputChannels != trunk.initialConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": numInputChannels (%d) != trunk.initialConv.inChannels (%d)", numInputChannels, trunk.initialConv.inChannels
      ));
    if(trunk.trunkNumChannels != policyHead.p1Conv.inChannels)
      throw StringError(name+Global::strprintf(
        ": trunk.trunkNumChannels (%d) != policyHead.p1Conv.inChannels (%d)", trunk.trunkNumChannels, policyHead.p1Conv.inChannels
      ));
    if(trunk.trunkNumChannels != policyHead.g1Conv.inChannels)
      throw StringError(name+Global::strprintf(
        ": trunk.trunkNumChannels (%d) != policyHead.g1Conv.inChannels (%d)", trunk.trunkNumChannels, policyHead.g1Conv.inChannels
      ));
    if(trunk.trunkNumChannels != valueHead.v1Conv.inChannels)
      throw StringError(name+Global::strprintf(
        ": trunk.trunkNumChannels (%d) != valueHead.v1Conv.inChannels (%d)", trunk.trunkNumChannels, valueHead.v1Conv.inChannels
      ));
  }

  ~ModelDesc() {
  }

  ModelDesc(const ModelDesc&) = delete;
  ModelDesc& operator=(const ModelDesc&) = delete;

  ModelDesc(ModelDesc&& other) {
    *this = std::move(other);
  }
  ModelDesc& operator=(ModelDesc&& other) {
    name = std::move(other.name);
    xSize = other.xSize;
    ySize = other.ySize;
    numInputChannels = other.numInputChannels;
    trunk = std::move(other.trunk);
    policyHead = std::move(other.policyHead);
    valueHead = std::move(other.valueHead);
    return *this;
  }
};


struct Model {
  string name;
  int maxBatchSize;
  int xSize;
  int ySize;
  int numInputChannels;

  cudnnTensorDescriptor_t* inputDescriptors;

  Trunk* trunk;
  PolicyHead* policyHead;
  ValueHead* valueHead;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    CudaHandles* cudaHandles,
    const ModelDesc* desc,
    int maxBatchSz
  ) {
    name = desc->name;
    maxBatchSize = maxBatchSz;
    xSize = desc->xSize;
    ySize = desc->ySize;
    numInputChannels = desc->numInputChannels;

    if(xSize != NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("Currently neural net xSize (%d) must be NNPos::MAX_BOARD_LEN (%d)",
        xSize, NNPos::MAX_BOARD_LEN
      ));
    if(ySize != NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("Currently neural net ySize (%d) must be NNPos::MAX_BOARD_LEN (%d)",
        ySize, NNPos::MAX_BOARD_LEN
      ));
    if(numInputChannels != NNInputs::NUM_FEATURES_V1)
      throw StringError(Global::strprintf("Currently neural net numInputChannels (%d) must be NNPos::NUM_FEATURES_V1 (%d)",
        numInputChannels, NNInputs::NUM_FEATURES_V1
      ));

    inputDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&inputDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        inputDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        numInputChannels,
        ySize,
        xSize
      ));
    }

    trunk = new Trunk(cudaHandles,&desc->trunk,maxBatchSize,xSize,ySize,inputDescriptors);
    policyHead = new PolicyHead(cudaHandles,&desc->policyHead,maxBatchSize,xSize,ySize,trunk->trunkDescriptors);
    valueHead = new ValueHead(cudaHandles,&desc->valueHead,maxBatchSize,xSize,ySize,trunk->trunkDescriptors);
  }

  ~Model()
  {
    delete valueHead;
    delete policyHead;
    delete trunk;

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnDestroyTensorDescriptor(inputDescriptors[batchSize-1]);
    }

    delete[] inputDescriptors;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& trunkDescriptor = trunk->trunkDescriptors[batchSize-1];

    b = trunk->requiredWorkspaceBytes(cudaHandles,inputDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = policyHead->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = valueHead->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,batchSize);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    float* inputBuf,
    float* trunkScratchBuf,
    float* trunkOutBuf,
    float* regularOutBuf,
    float* dilatedOutBuf,
    float* midInBuf,
    float* midScratchBuf,
    float* gpoolOutBuf,
    float* gpoolOutBuf2,
    float* gpoolMeanBuf,
    float* gpoolMaxBuf,
    float* gpoolConcatBuf,
    float* gpoolBiasBuf,
    float* regularScratchBuf,

    float* p1OutBuf,
    float* p1OutBuf2,
    float* g1OutBuf,
    float* g1OutBuf2,
    float* g1MeanBuf,
    float* g1MaxBuf,
    float* g1ConcatBuf,
    float* g1BiasBuf,
    float* p2InBuf,
    float* p2OutBuf,
    float* g1PassBuf,
    float* policyBuf,

    float* v1OutBuf,
    float* v1OutBuf2,
    float* v1MeanBuf,
    float* v2OutBuf,
    float* v2OutBuf2,
    float* v3InBuf,
    float* valueBuf,

    float* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& trunkDescriptor = trunk->trunkDescriptors[batchSize-1];

    trunk->apply(
      cudaHandles,
      inputDescriptor,
      batchSize,
      inputBuf,
      trunkScratchBuf,
      trunkOutBuf,
      regularOutBuf,
      dilatedOutBuf,
      midInBuf,
      midScratchBuf,
      gpoolOutBuf,
      gpoolOutBuf2,
      gpoolMeanBuf,
      gpoolMaxBuf,
      gpoolConcatBuf,
      gpoolBiasBuf,
      regularScratchBuf,
      workspaceBuf,
      workspaceBytes
    );
    policyHead->apply(
      cudaHandles,
      trunkDescriptor,
      batchSize,
      trunkOutBuf,
      p1OutBuf,
      p1OutBuf2,
      g1OutBuf,
      g1OutBuf2,
      g1MeanBuf,
      g1MaxBuf,
      g1ConcatBuf,
      g1BiasBuf,
      p2InBuf,
      p2OutBuf,
      g1PassBuf,
      policyBuf,
      workspaceBuf,
      workspaceBytes
    );
    valueHead->apply(
      cudaHandles,
      trunkDescriptor,
      batchSize,
      trunkOutBuf,
      v1OutBuf,
      v1OutBuf2,
      v1MeanBuf,
      v2OutBuf,
      v2OutBuf2,
      v3InBuf,
      valueBuf,
      workspaceBuf,
      workspaceBytes
    );
  }

};


//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(istream& in) {
    modelDesc = std::move(ModelDesc(in));
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  (void)modelFileIdx;

  ifstream in(file);
  LoadedModel* loadedModel = new LoadedModel(in);
  in.close();
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}


//------------------------------------------------------------------------------

struct Buffers {
  //All of these are device pointers

  float* inputBuf;
  size_t inputBufBytes;

  float* trunkScratchBuf;
  float* trunkOutBuf;
  float* regularOutBuf;
  float* dilatedOutBuf;
  float* midInBuf;
  float* midScratchBuf;
  float* gpoolOutBuf;
  float* gpoolOutBuf2;
  float* gpoolMeanBuf;
  float* gpoolMaxBuf;
  float* gpoolConcatBuf;
  float* gpoolBiasBuf;
  float* regularScratchBuf;

  float* p1OutBuf;
  float* p1OutBuf2;
  float* g1OutBuf;
  float* g1OutBuf2;
  float* g1MeanBuf;
  float* g1MaxBuf;
  float* g1ConcatBuf;
  float* g1BiasBuf;
  float* p2InBuf;
  float* p2OutBuf;
  float* g1PassBuf;
  float* policyBuf;
  size_t policyBufBytes;

  float* v1OutBuf;
  float* v1OutBuf2;
  float* v1MeanBuf;
  float* v2OutBuf;
  float* v2OutBuf2;
  float* v3InBuf;
  float* valueBuf;
  size_t valueBufBytes;

  float* workspaceBuf;
  size_t workspaceBytes;

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(CudaHandles* cudaHandles, const Model& m) {
    size_t batchXYFloat = m.maxBatchSize * m.xSize * m.ySize * sizeof(float);
    size_t batchFloat = m.maxBatchSize * sizeof(float);

    inputBufBytes = m.numInputChannels * batchXYFloat;
    CUDA_ERR("Buffers",cudaMalloc(&inputBuf, inputBufBytes));

    CUDA_ERR("Buffers",cudaMalloc(&trunkScratchBuf, m.trunk->trunkNumChannels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&trunkOutBuf, m.trunk->trunkNumChannels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&regularOutBuf, m.trunk->regularNumChannels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&dilatedOutBuf, m.trunk->dilatedNumChannels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&midInBuf, (m.trunk->regularNumChannels + m.trunk->dilatedNumChannels) * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&midScratchBuf, (m.trunk->regularNumChannels + m.trunk->dilatedNumChannels) * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolOutBuf, m.trunk->gpoolNumChannels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolOutBuf2, m.trunk->gpoolNumChannels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolMeanBuf, m.trunk->gpoolNumChannels * batchFloat));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolMaxBuf, m.trunk->gpoolNumChannels * batchFloat));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolConcatBuf, m.trunk->gpoolNumChannels * batchFloat * 2));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolBiasBuf, m.trunk->regularNumChannels * batchFloat));
    CUDA_ERR("Buffers",cudaMalloc(&regularScratchBuf, m.trunk->regularNumChannels * batchXYFloat));

    CUDA_ERR("Buffers",cudaMalloc(&p1OutBuf, m.policyHead->p1Channels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&p1OutBuf2, m.policyHead->p1Channels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&g1OutBuf, m.policyHead->g1Channels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&g1OutBuf2, m.policyHead->g1Channels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&g1MeanBuf, m.policyHead->g1Channels * batchFloat));
    CUDA_ERR("Buffers",cudaMalloc(&g1MaxBuf, m.policyHead->g1Channels * batchFloat));
    CUDA_ERR("Buffers",cudaMalloc(&g1ConcatBuf, m.policyHead->g1Channels * batchFloat * 2));
    CUDA_ERR("Buffers",cudaMalloc(&g1BiasBuf, m.policyHead->p1Channels * batchFloat));
    CUDA_ERR("Buffers",cudaMalloc(&p2InBuf, m.policyHead->p1Channels * 2 * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&p2OutBuf, m.policyHead->p2Channels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&g1PassBuf, m.policyHead->p2Channels * batchFloat));

    policyBufBytes = m.policyHead->p2Channels * (batchXYFloat + batchFloat);
    CUDA_ERR("Buffers",cudaMalloc(&policyBuf, policyBufBytes));
    assert(m.policyHead->p2Channels == 1);

    CUDA_ERR("Buffers",cudaMalloc(&v1OutBuf, m.valueHead->v1Channels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&v1OutBuf2, m.valueHead->v1Channels * batchXYFloat));
    CUDA_ERR("Buffers",cudaMalloc(&v1MeanBuf, m.valueHead->v1Channels * batchFloat));
    CUDA_ERR("Buffers",cudaMalloc(&v2OutBuf, m.valueHead->v2Channels * batchFloat));
    CUDA_ERR("Buffers",cudaMalloc(&v2OutBuf2, m.valueHead->v2Channels * batchFloat));
    CUDA_ERR("Buffers",cudaMalloc(&v3InBuf, m.valueHead->v2Channels * 2 * batchFloat));

    valueBufBytes = m.valueHead->valueChannels * batchFloat;
    CUDA_ERR("Buffers",cudaMalloc(&valueBuf, valueBufBytes));

    //In theory the requiredWorkspaceBytes calls could give us values non-monotone in batch size
    //such as if the convolution algorithm changes between batch size 1 and larger.
    //So we call it for all the batch sizes.
    size_t bytes = 0;
    size_t b;
    for(int batchSize = 1; batchSize <= m.maxBatchSize; batchSize++) {
      b = m.requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }

    CUDA_ERR("Buffers",cudaMalloc(&workspaceBuf, bytes));
    workspaceBytes = bytes;
  }

  ~Buffers() {
    cudaFree(inputBuf);
    cudaFree(trunkScratchBuf);
    cudaFree(trunkOutBuf);
    cudaFree(regularOutBuf);
    cudaFree(dilatedOutBuf);
    cudaFree(midInBuf);
    cudaFree(midScratchBuf);
    cudaFree(gpoolOutBuf);
    cudaFree(gpoolOutBuf2);
    cudaFree(gpoolMeanBuf);
    cudaFree(gpoolMaxBuf);
    cudaFree(gpoolConcatBuf);
    cudaFree(gpoolBiasBuf);
    cudaFree(regularScratchBuf);

    cudaFree(p1OutBuf);
    cudaFree(p1OutBuf2);
    cudaFree(g1OutBuf);
    cudaFree(g1OutBuf2);
    cudaFree(g1MeanBuf);
    cudaFree(g1MaxBuf);
    cudaFree(g1ConcatBuf);
    cudaFree(g1BiasBuf);
    cudaFree(p2InBuf);
    cudaFree(p2OutBuf);
    cudaFree(g1PassBuf);
    cudaFree(policyBuf);

    cudaFree(v1OutBuf);
    cudaFree(v1OutBuf2);
    cudaFree(v1MeanBuf);
    cudaFree(v2OutBuf);
    cudaFree(v2OutBuf2);
    cudaFree(v3InBuf);
    cudaFree(valueBuf);

    cudaFree(workspaceBuf);
  }

};



//------------------------------------------------------------------------------

struct LocalGpuHandle {
  CudaHandles* cudaHandles;
  Model* model;
  Buffers* buffers;

  LocalGpuHandle(const LoadedModel* loadedModel, int maxBatchSize) {
    cudaHandles = new CudaHandles();
    model = new Model(cudaHandles,&(loadedModel->modelDesc),maxBatchSize);
    buffers = new Buffers(cudaHandles,*model);
  }
  ~LocalGpuHandle() {
    delete buffers;
    delete model;
    delete cudaHandles;
  }

  LocalGpuHandle() = delete;
  LocalGpuHandle(const LocalGpuHandle&) = delete;
  LocalGpuHandle& operator=(const LocalGpuHandle&) = delete;
};

LocalGpuHandle* NeuralNet::createLocalGpuHandle(const LoadedModel* loadedModel, int maxBatchSize, int cudaDeviceIdxForThisThread) {
  CUDA_ERR("createLocalGpuHandle",cudaSetDevice(cudaDeviceIdxForThisThread));

  LocalGpuHandle* gpuHandle = new LocalGpuHandle(loadedModel,maxBatchSize);
  return gpuHandle;
}

void NeuralNet::freeLocalGpuHandle(LocalGpuHandle* gpuHandle) {
  delete gpuHandle;
}

//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleBatchItemBytes;
  size_t singlePolicyResultBytes;
  size_t singleValueResultBytes;

  size_t userInputBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;

  float* userInputBuffer; //Host pointer
  bool* symmetriesBuffer; //Host pointer

  float* policyResults;
  float* valueResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleBatchItemBytes = m.numInputChannels * m.xSize * m.ySize;
    singlePolicyResultBytes = (1 + m.xSize * m.ySize) * sizeof(float);
    singleValueResultBytes = sizeof(float);

    userInputBufferBytes = m.numInputChannels * maxBatchSize * m.xSize * m.ySize * sizeof(float);
    policyResultBufferBytes = maxBatchSize * (1 + m.xSize * m.ySize) * sizeof(float);
    valueResultBufferBytes = maxBatchSize * sizeof(float);

    userInputBuffer = new float[m.numInputChannels * maxBatchSize * m.xSize * m.ySize];
    symmetriesBuffer = new bool[NNInputs::NUM_SYMMETRY_BOOLS];

    policyResults = new float[maxBatchSize * (1 + m.xSize * m.ySize)];
    valueResults = new float[maxBatchSize];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] symmetriesBuffer;
    delete[] policyResults;
    delete[] valueResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize) {
  return new InputBuffers(loadedModel,maxBatchSize);
}
void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  delete buffers;
}

float* NeuralNet::getRowInplace(InputBuffers* buffers, int rowIdx) {
  assert(rowIdx < buffers->maxBatchSize);
  return buffers->userInputBuffer + (buffers->singleBatchItemBytes * rowIdx);
}

//TODO use symmetries!!
bool* NeuralNet::getSymmetriesInplace(InputBuffers* buffers) {
  return buffers->symmetriesBuffer;
}


//---------------------------------------------------------------------------------------


void NeuralNet::getOutput(LocalGpuHandle* gpuHandle, InputBuffers* inputBuffers, int numFilledRows, vector<NNOutput*>& outputs) {
  assert(numFilledRows <= inputBuffers->maxBatchSize);
  assert(numFilledRows > 0);

  assert(inputBuffers->userInputBufferBytes == gpuHandle->buffers->inputBufBytes);
  assert(inputBuffers->policyResultBufferBytes == gpuHandle->buffers->policyBufBytes);
  assert(inputBuffers->valueResultBufferBytes == gpuHandle->buffers->valueBufBytes);
  assert(inputBuffers->singlePolicyResultBytes == NNPos::NN_POLICY_SIZE * sizeof(float));

  int batchSize = numFilledRows;

  Buffers* buffers = gpuHandle->buffers;
  CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleBatchItemBytes*batchSize, cudaMemcpyHostToDevice));

  gpuHandle->model->apply(
    gpuHandle->cudaHandles,
    batchSize,

    buffers->inputBuf,
    buffers->trunkScratchBuf,
    buffers->trunkOutBuf,
    buffers->regularOutBuf,
    buffers->dilatedOutBuf,
    buffers->midInBuf,
    buffers->midScratchBuf,
    buffers->gpoolOutBuf,
    buffers->gpoolOutBuf2,
    buffers->gpoolMeanBuf,
    buffers->gpoolMaxBuf,
    buffers->gpoolConcatBuf,
    buffers->gpoolBiasBuf,
    buffers->regularScratchBuf,

    buffers->p1OutBuf,
    buffers->p1OutBuf2,
    buffers->g1OutBuf,
    buffers->g1OutBuf2,
    buffers->g1MeanBuf,
    buffers->g1MaxBuf,
    buffers->g1ConcatBuf,
    buffers->g1BiasBuf,
    buffers->p2InBuf,
    buffers->p2OutBuf,
    buffers->g1PassBuf,
    buffers->policyBuf,

    buffers->v1OutBuf,
    buffers->v1OutBuf2,
    buffers->v1MeanBuf,
    buffers->v2OutBuf,
    buffers->v2OutBuf2,
    buffers->v3InBuf,
    buffers->valueBuf,

    buffers->workspaceBuf,
    buffers->workspaceBytes
  );

  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes*batchSize, cudaMemcpyDeviceToHost));

  outputs.clear();

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = new NNOutput();
    float* policyProbs = output->policyProbs;

    //These are not actually correct, the client does the postprocessing to turn them into
    //probabilities and white value
    //Also we don't fill in the nnHash here either
    std::copy(
      inputBuffers->policyResults + row * NNPos::NN_POLICY_SIZE,
      inputBuffers->policyResults + (row+1) * NNPos::NN_POLICY_SIZE,
      policyProbs
    );
    output->whiteValue = inputBuffers->valueResults[row];
    outputs.push_back(output);
  }

}





#endif
