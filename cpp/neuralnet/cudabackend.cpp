
//TODO
#define USE_CUDA_BACKEND
#ifdef USE_CUDA_BACKEND

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

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
    CUBLAS_ERR(cublasCreate(&cublas));
    CUDNN_ERR(cudnnCreate(&cudnn));
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
};

struct ConvLayer {
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
    cudnnTensorDescriptor_t* inputDescriptors, //array of one for each batch size
    cudnnTensorDescriptor_t* outputDescriptors //array of one for each batch size
  ) {
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int inChannels = desc->inChannels;
    int outChannels = desc->outChannels;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = convXSize / 2 + dilationX;
    int paddingY = convYSize / 2 + dilationY;

    assert(convXSize % 2 == 0);
    assert(convYSize % 2 == 0);

    CUDNN_ERR(cudnnCreateFilterDescriptor(&filterDescriptor));
    CUDNN_ERR(cudnnSetFilter4dDescriptor(
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

    CUDNN_ERR(cudnnCreateConvolutionDescriptor(&convolutionDescriptor));
    CUDNN_ERR(cudnnSetConvolution2dDescriptor(
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
      size_t bytesMemoryLimit = 0;
      CUDNN_ERR(cudnnGetConvolutionForwardAlgorithm(
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

    CUDA_ERR(cudaMalloc(&filterBuf, filterBytes));
    CUDA_ERR(cudaMemcpy(filterBuf, desc->weights.data(), filterBytes, cudaMemcpyHostToDevice));

  }

  ~ConvLayer() {
    cudaFree(filterBuf);
    cudnnDestroyFilterDescriptor(filterDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    delete[] convolutionAlgorithms;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor
  ) {
    size_t workspaceBytes = 0;
    CUDNN_ERR(cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptor,
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptor,
      convolutionAlgorithm,
      &workspaceBytes
    ));

    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor,
    float* inputBuf,
    float* outputBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) {
    const float alpha = 1;
    const float beta = 0;
    CUDNN_ERR(cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptor,
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      convolutionAlgorithm,
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
};

struct BNLayer {
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
    const BNLayerDesc* desc,
  ) {
    (void)cudaHandles;

    numChannels = desc->numChannels;
    epsilon = desc->epsilon;

    CUDNN_ERR(cudnnCreateTensorDescriptor(&bufDescriptor));
    CUDNN_ERR(cudnnSetTensor4dDescriptor(
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
    CUDA_ERR(cudaMalloc(&meanBuf, bufBytes));
    CUDA_ERR(cudaMemcpy(meanBuf, desc->mean.data(), bufBytes, cudaMemcpyHostToDevice));

    assert(desc->variance.size() == numChannels);
    CUDA_ERR(cudaMalloc(&varianceBuf, bufBytes));
    CUDA_ERR(cudaMemcpy(varianceBuf, desc->variance.data(), bufBytes, cudaMemcpyHostToDevice));

    assert(desc->scale.size() == numChannels);
    CUDA_ERR(cudaMalloc(&scaleBuf, bufBytes));
    CUDA_ERR(cudaMemcpy(scaleBuf, desc->scale.data(), bufBytes, cudaMemcpyHostToDevice));

    assert(desc->bias.size() == numChannels);
    CUDA_ERR(cudaMalloc(&biasBuf, bufBytes));
    CUDA_ERR(cudaMemcpy(biasBuf, desc->bias.data(), bufBytes, cudaMemcpyHostToDevice));
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
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor,
    float* inputBuf,
    float* outputBuf
  ) {
    const float alpha = 1;
    const float beta = 0;
    CUDNN_ERR(cudnnBatchNormalizationForwardInference(
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

  ActivationLayerDesc() {}

  ActivationLayerDesc(istream& in) {
    (void)in;
  }
};

struct ActivationLayer {
  cudnnActivationDescriptor_t activationDescriptor;

  ActivationLayer() = delete;
  ActivationLayer(const ActivationLayer&) = delete;
  ActivationLayer& operator=(const ActivationLayer&) = delete;

  ActivationLayer(
    CudaHandles* cudaHandles,
    const ActivationLayerDesc* desc,
  ) {
    (void)cudaHandles;
    (void)desc;

    CUDNN_ERR(cudnnCreateActivationDescriptor(&activationDescriptor));
    CUDNN_ERR(cudnnSetActivationDescriptor(
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
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor,
    float* inputBuf,
    float* outputBuf
  ) {
    const float alpha = 1;
    const float beta = 0;

    CUDNN_ERR(cudnnActivationForward(
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
};

struct MatMulLayer {
  int inChannels;
  int outChannels;
  float* matBuf;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(
    CudaHandles* cudaHandles,
    const MatMulLayerDesc* desc,
  ) {
    (void)cudaHandles;
    inChannels = desc->inChannels;
    outChannels = desc->outChannels;

    assert(desc->weights.size() == inChannels * outChannels);
    size_t matBytes = sizeof(float) * inChannels * outChannels;

    CUDA_ERR(cudaMalloc(&matBuf, matBytes));
    CUDA_ERR(cudaMemcpy(matBuf, desc->weights.data(), matBytes, cudaMemcpyHostToDevice));
  }

  ~MatMulLayer() {
    cudaFree(matBuf);
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles
  ) {
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
  ) {
    (void)workspaceBuf;
    (void)workspaceBytes;

    const float alpha = 1;
    const float beta = 0;
    CUBLAS_ERR(cublasSgemm(
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
};

struct MatBiasLayer {
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
    numChannels = desc->numChannels;

    assert(desc->weights.size() == numChannels);
    size_t biasBytes = sizeof(float) * numChannels;

    size_t maxBatchSizeOnesBytes = sizeof(float) * maxBatchSize;
    float maxBatchSizeOnesArr[maxBatchSize];
    for(int i = 0; i<maxBatchSize; i++)
      maxBatchSizeOnesArr[i] = 1.0f;

    CUDA_ERR(cudaMalloc(&biasBuf, biasBytes));
    CUDA_ERR(cudaMemcpy(biasBuf, desc->weights.data(), biasBytes, cudaMemcpyHostToDevice));
    CUDA_ERR(cudaMalloc(&maxBatchSizeOnesBuf, maxBatchSizeOnesBytes));
    CUDA_ERR(cudaMemcpy(maxBatchSizeOnesBuf, maxBatchSizeOnesArr, maxBatchSizeOnesBytes, cudaMemcpyHostToDevice));
  }

  ~MatBiasLayer() {
    cudaFree(biasBuf);
    cudaFree(maxBatchSizeOnesBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    float* matBuf
  ) {
    const float alpha = 1;
    CUBLAS_ERR(cublasSger(
      cudaHandles->cublas,
      numChannels,
      batchSize,
      &alpha,
      biasBuf,
      1,
      maxBatchSizeOnesBuf,
      1
      matBuf,
      numChannels
    ));
  }

};


//---------------------------------------------------------------------------------

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
};

struct DilatedResidualBlock {
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
    cudnnTensorDescriptor_t* trunkDescriptors, //array of one for each batch size
    cudnnTensorDescriptor_t* regularOutDescriptors, //array of one for each batch size
    cudnnTensorDescriptor_t* dilatedOutDescriptors, //array of one for each batch size
    cudnnTensorDescriptor_t* midInDescriptors //array of one for each batch size
  ): preBN(cudaHandles,&desc->preBN),
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
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& dilatedOutDescriptor,
    cudnnTensorDescriptor_t& midInDescriptor
  ) {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,trunkDescriptor,regularOutDescriptor);
    bytes = std::max(bytes,b);
    b = dilatedConv.requiredWorkspaceBytes(cudaHandles,trunkDescriptor,dilatedOutDescriptor);
    bytes = std::max(bytes,b);
    b = finalConv.requiredWorkspaceBytes(cudaHandles,midInDescriptor,trunkDescriptor);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& dilatedOutDescriptor,
    cudnnTensorDescriptor_t& midInDescriptor,
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
  ) {
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
    CUBLAS_ERR(cublasSaxpy(cudaHandles->cublas,trunkBufSize,&alpha,trunkInBuf,1,trunkOutBuf,1));
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
};

struct GlobalPoolingResidualBlock {
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
    cudnnTensorDescriptor_t* trunkDescriptors, //array of one for each batch size
    cudnnTensorDescriptor_t* regularOutDescriptors, //array of one for each batch size
    cudnnTensorDescriptor_t* gpoolOutDescriptors //array of one for each batch size
  ): preBN(cudaHandles,&desc->preBN),
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
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& gpoolOutDescriptor
  ) {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,trunkDescriptor,regularOutDescriptor);
    bytes = std::max(bytes,b);
    b = gpoolConv.requiredWorkspaceBytes(cudaHandles,trunkDescriptor,gpoolOutDescriptor);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = finalConv.requiredWorkspaceBytes(cudaHandles,regularOutDescriptor,trunkDescriptor);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& gpoolOutDescriptor,
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
  ) {
    preBN.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkInBuf,trunkOutBuf);
    preActivation.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkOutBuf,trunkOutBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,regularOutDescriptor,batchSize,trunkOutBuf,regularOutBuf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(cudaHandles,trunkDescriptor,gpoolOutDescriptor,batchSize,trunkOutBuf,gpoolOutBuf,workspaceBuf,workspaceBytes);
    gpoolBN.apply(cudaHandles,gpoolOutDescriptor,gpoolOutBuf,gpoolOutBuf2);
    gpoolActivation.apply(cudaHandles,gpoolOutDescriptor,gpoolOutBuf2,gpoolOutBuf2);

    customCudaPoolRowsSum(gpoolOutBuf2,gpoolMeanBuf,batchSize*gpoolChannels,xSize*ySize);
    customCudaPoolRowsMax(gpoolOutBuf2,gpoolMaxBuf,batchSize*gpoolChannels,xSize*ySize);
    float meanScale = 1.0f / (xSize*ySize);
    CUBLAS_ERR(cublasSscal(cudaHandles->cublas, batchSize*gpoolChannels, &meanScale, gpoolMeanBuf, 1));
    customCudaChannelConcat(
      gpoolMeanBuf,gpoolMaxBuf,gpoolConcatBuf,
      xSize*ySize*gpoolChannels,
      xSize*ySize*gpoolChannels,
      batchSize
    );
    gpoolToBiasMul.apply(cudaHandles,batchSize,gpoolConcatBuf,gpoolBiasBuf,workspaceBuf,workspaceBytes);

    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnAddTensor(cudaHandles->cudnn,&alpha,gpoolBiasBuf,&beta,regularOutDescriptor,regularOutBuf);

    midBN.apply(cudaHandles,regularOutDescriptor,regularOutDescriptor,regularOutBuf,regularScratchBuf);
    midActivation.apply(cudaHandles,regularOutDescriptor,regularOutDescriptor,regularScratchBuf,regularScratchBuf);
    finalConv.apply(cudaHandles,regularOutDescriptor,trunkDescriptor,batchSize,regularScratchBuf,trunkOutBuf,workspaceBuf,workspaceBytes);

    const float alpha = 1.0f;
    CUBLAS_ERR(cublasSaxpy(cudaHandles->cublas,trunkBufSize,&alpha,trunkInBuf,1,trunkOutBuf,1));
  }

};

//------------------------------------------------------------------------------

static const int DILATED_BLOCK_KIND = 0;
static const int GLOBAL_POOLING_BLOCK_KIND = 1;

struct TrunkDesc {
  string name;
  int numBlocks;
  int trunkNumChannels;
  int regularNumChannels; //Currently every residual block must have the same number of regular conv channels
  int dilatedNumChannels; //Currently every dilated residual block must have the same number of dilated conv channels
  int gpoolNumChannels;   //Currently every gpooling residual block must have the same number of gpooling conv channels
  ConvLayerDesc initialConv;
  vector<pair<int,void*>> blocks;
  BNLayer trunkTipBN;
  ActivationLayer trunkTipActivation;

  TrunkDesc() {}

  TrunkDesc(istream& in) {
    in >> name;
    in >> numBlocks;
    in >> trunkNumChannels;
    in >> regularNumChannels;
    in >> dilatedNumChannels;
    in >> gpoolNumChannels;

    if(in.fail())
      throw StringError(name + ": trunk failed to parse num blocks or various channel parameters");
    if(numBlocks < 1)
      throw StringError(name + ": trunk num blocks must be positive");
    if(trunkNumChannels <= 0 || regularNumChannels <= 0 || dilatedNumChannels <= 0 || gpoolNumChannels <= 0)
      throw StringError(name + ": all numbers of channels must be positive");

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
      if(kind == "dilated_block") {
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
        throw StringError("name" + ": found unknown block kind: " + kind);

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
      if(blocks[i].first == DILATED_BLOCK_KIND) {
        DilatedResidualBlockDesc* desc = (DilatedResidualBlockDesc*)blocks[i].second;
        delete desc;
      }
      else if(blocks[i].second == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)blocks[i].second;
        delete desc;
      }
    }
  }
};


struct Trunk {
  int numBlocks;
  int trunkNumChannels;
  int regularNumChannels;
  int dilatedNumChannels;
  int gpoolNumChannels;

  int maxBatchSize;
  int xSize;
  int ySize;

  cudnnTensorDescriptor_t* trunkDescriptors;
  cudnnTensorDescriptor_t* regularOutDescriptors;
  cudnnTensorDescriptor_t* gpoolOutDescriptors;
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
    cudnnTensorDescriptor_t* inputDescriptors
  ) {
    numBlocks = desc->numBlocks;
    trunkNumChannels = desc->trunkNumChannels;
    regularNumChannels = desc->regularNumChannels;
    dilatedNumChannels = desc->dilatedNumChannels;
    gpoolNumChannels = desc->gpoolNumChannels;

    maxBatchSize = maxBatchSz;
    xSize = xS;
    ySize = yS;

    trunkDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    regularOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    gpoolOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    dilatedOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    midInDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; i++) {
      cudnnTensorDescriptor_t& trunkDescriptor = trunkDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& regularOutDescriptor = regularOutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& gpoolOutDescriptor = gpoolOutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& dilatedOutDescriptor = dilatedOutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& midInDescriptor = midInDescriptors[batchSize-1];

      CUDNN_ERR(cudnnCreateTensorDescriptor(&trunkDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
        trunkDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        trunkNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(cudnnCreateTensorDescriptor(&regularOutDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
        regularOutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        regularNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(cudnnCreateTensorDescriptor(&dilatedOutDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
        dilatedOutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        dilatedNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(cudnnCreateTensorDescriptor(&gpoolOutDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
        gpoolOutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        gpoolNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(cudnnCreateTensorDescriptor(&midInDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
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
      if(desc->blocks[i].first == DILATED_BLOCK_KIND) {
        DilatedResidualBlockDesc* desc = (DilatedResidualBlockDesc*)desc->blocks[i].second;
        DilatedResidualBlock* block = new DilatedResidualBlock(
          cudaHandles,
          desc,
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
        GlobalPoolingResidualBlockDesc* desc = (GlobalPoolingResidualBlockDesc*)desc->blocks[i].second;
        GlobalPoolingResidualBlock* block = new GlobalPoolingResidualBlock(
          cudaHandles,
          desc,
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
      if(blocks[i].first == DILATED_BLOCK_KIND) {
        DilatedResidualBlock* block = (DilatedResidualBlock*)blocks[i].second;
        delete block;
      }
      else if(blocks[i].second == GLOBAL_POOLING_BLOCK_KIND) {
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
      cudnnDestroyTensorDescriptor(midInDescriptors[batchSize-1]);
    }

    delete[] trunkDescriptors;
    delete[] regularOutDescriptors;
    delete[] dilatedOutDescriptors;
    delete[] gpoolOutDescriptors;
    delete[] midInDescriptors;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    cudnnTensorDescriptor_t& inputDescriptor,
    int batchSize
  ) {
    size_t bytes = 0;
    size_t b;

    cudnnTensorDescriptor_t& trunkDescriptor = trunkDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& regularOutDescriptor = regularOutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& gpoolOutDescriptor = gpoolOutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& dilatedOutDescriptor = dilatedOutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& midInDescriptor = midInDescriptors[batchSize-1];

    b = initialConv->requiredWorkspaceBytes(cudaHandles,inputDescriptor,trunkDescriptor);
    bytes = std::max(bytes,b);

    for(int i = 0; i<blocks.size(); i++) {
      if(blocks[i].first == DILATED_BLOCK_KIND) {
        DilatedResidualBlock* block = (DilatedResidualBlock*)blocks[i].second;
        b = block->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,regularOutDescriptor,dilatedOutDescriptor,midInDescriptor);
        bytes = std::max(bytes,b);
      }
      else if(blocks[i].second == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second;
        b = block->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,regularOutDescriptor,gpoolOutDescriptor);
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
    cudnnTensorDescriptor_t& inputDescriptor,
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
  ) {
    float* currentTrunkBuf = trunkScratchBuf;
    float* nextTrunkBuf = trunkOutBuf;

    cudnnTensorDescriptor_t& trunkDescriptor = trunkDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& regularOutDescriptor = regularOutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& gpoolOutDescriptor = gpoolOutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& dilatedOutDescriptor = dilatedOutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& midInDescriptor = midInDescriptors[batchSize-1];

    int trunkBufSize = batchSize * trunkNumChannels * xSize * ySize;

    initialConv->apply(cudaHandles,inputDescriptor,trunkDescriptor,inputBuf,currentTrunkBuf);

    for(int i = 0; i<blocks.size(); i++) {
      if(blocks[i].first == DILATED_BLOCK_KIND) {
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
      else if(blocks[i].second == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second;
        block->apply(
          cudaHandles,
          trunkDescriptor,
          regularOutDescriptor,
          gpoolOutDescriptor,
          batchSize,
          trunkBufSize,
          currentTrunkBuf,
          nextTrunkBuf,
          regularOutBuf,
          gpoolOutBuf,
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
    if(p2Conv.inChannels != p1BN.numChannels)
      throw StringError(name+Global::strprintf(
        ": p2Conv.inChannels (%d) != p1BN.numChannels (%d)", p2Conv.inChannels, p1BN.numChannels
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
};

struct PolicyHead() {
  int maxBatchSize;
  int xSize;
  int ySize;
  int p1Channels;
  int g1Channels;

  cudnnTensorDescriptor_t* p1OutDescriptors;
  cudnnTensorDescriptor_t* g1OutDescriptors;
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
    cudnnTensorDescriptor_t* trunkDescriptors
  ) {
    maxBatchSize = maxBatchSz;
    xSize = xS;
    ySize = yS;
    p1Channels = desc->p1Conv.outChannels;
    g1Channels = desc->g1Conv.outChannels;

    p1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    g1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    p2OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; i++) {
      cudnnTensorDescriptor_t& p1OutDescriptor = p1OutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& g1OutDescriptor = g1OutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& p2OutDescriptor = p2OutDescriptors[batchSize-1];

      CUDNN_ERR(cudnnCreateTensorDescriptor(&p1OutDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
        p1OutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->p1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(cudnnCreateTensorDescriptor(&g1OutDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
        g1OutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->g1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(cudnnCreateTensorDescriptor(&p2OutDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
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
    p1BN = new BNLayer(cudaHandles,&desc->p1BN,p1OutDescriptor,p1OutDescriptor);
    p1Activation = new ActivationLayer(cudaHandles,&desc->p1Activation);
    p2Conv = new ConvLayer(cudaHandles,&desc->p2Conv,maxBatchSize,p1OutDescriptors,p2OutDescriptors);
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
      cudnnDestroyTensorDescriptor(p2OutDescriptors[batchSize-1]);
    }

    delete[] p1OutDescriptors;
    delete[] g1OutDescriptors;
    delete[] p2OutDescriptors;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    cudnnTensorDescriptor_t& trunkDescriptor,
    int batchSize
  ) {
    size_t bytes = 0;
    size_t b;

    cudnnTensorDescriptor_t& p1OutDescriptor = p1OutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& g1OutDescriptor = g1OutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& p2OutDescriptor = p2OutDescriptors[batchSize-1];

    b = p1Conv->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,p1OutDescriptor);
    bytes = std::max(bytes,b);
    b = g1Conv->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,g1OutDescriptor);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = p2Conv->requiredWorkspaceBytes(cudaHandles,p1OutDescriptor,p2OutDescriptor);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    cudnnTensorDescriptor_t& trunkDescriptor,
    float* trunkOutBuf,
    float* p1OutBuf,
    float* p1OutBuf2,
    float* g1OutBuf,
    float* g1OutBuf2,
    float* g1MeanBuf,
    float* g1MaxBuf,
    float* g1ConcatBuf,
    float* g1BiasBuf,
    float* p2OutBuf,
    float* g1PassBuf,
    float* policyBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) {
    cudnnTensorDescriptor_t& p1OutDescriptor = p1OutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& g1OutDescriptor = g1OutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& p2OutDescriptor = p2OutDescriptors[batchSize-1];

    p1Conv->apply(cudaHandles,trunkDescriptor,p1OutDescriptor,batchSize,trunkOutBuf,p1OutBuf);
    g1Conv->apply(cudaHandles,trunkDescriptor,g1OutDescriptor,batchSize,trunkOutBuf,g1OutBuf);
    g1BN->apply(cudaHandles,g1OutDescriptor,g1OutDescriptor,g1OutBuf,g1OutBuf2);
    g1Activation->apply(cudaHandles,g1OutDescriptor,g1OutDescriptor,g1OutBuf2,g1OutBuf2);

    customCudaPoolRowsSum(g1OutBuf2,g1MeanBuf,batchSize*g1Channels,xSize*ySize);
    customCudaPoolRowsMax(g1OutBuf2,g1MaxBuf,batchSize*g1Channels,xSize*ySize);
    float meanScale = 1.0f / (xSize*ySize);
    CUBLAS_ERR(cublasSscal(cudaHandles->cublas, batchSize*g1Channels, &meanScale, g1MeanBuf, 1));
    customCudaChannelConcat(
      g1MeanBuf,g1MaxBuf,g1ConcatBuf,
      xSize*ySize*g1Channels,
      xSize*ySize*g1Channels,
      batchSize
    );
    gpoolToBiasMul.apply(cudaHandles,batchSize,g1ConcatBuf,g1BiasBuf,workspaceBuf,workspaceBytes);

    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnAddTensor(cudaHandles->cudnn,&alpha,g1BiasBuf,&beta,p1OutDescriptor,p1OutBuf);

    p1BN->apply(cudaHandles,p1OutDescriptor,p1OutDescriptor,p1OutBuf,p1OutBuf2);
    //TODO CRELU
    p1Activation->apply(cudaHandles,p1OutDescriptor,p1OutDescriptor,p1OutBuf2,p1OutBuf2);

    p2Conv->apply(cudaHandles,p1OutDescriptor,p2OutDescriptor,batchSize,p1OutBuf2,p2OutBuf);

    gpoolToPassMul.apply(cudaHandles,batchSize,g1ConcatBuf,g1PassBuf,workspaceBuf,workspaceBytes);

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
    v3Activation = ActivationLayerDesc(in);

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
    if(v2Mul.outChannels != v3Mul.inChannels)
      throw StringError(name+Global::strprintf(
        ": v2Mul.outChannels (%d) != v3Mul.inChannels (%d)", v2Mul.outChannels, v3Mul.inChannels
      ));
    if(v3Mul.outChannels != 1)
      throw StringError(name+Global::strprintf(
        ": v3Mul.outChannels (%d) != 1", v3Mul.outChannels
      ));
    if(v3Bias.outChannels != 1)
      throw StringError(name+Global::strprintf(
        ": v3Bias.outChannels (%d) != 1", v3Bias.outChannels
      ));
  }

  ~ValueHeadDesc() {
  }
};



struct ValueHead() {
  int maxBatchSize;
  int xSize;
  int ySize;
  int v1Channels;

  cudnnTensorDescriptor_t* v1OutDescriptors;
  cudnnTensorDescriptor_t* v2OutDescriptors;

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
    cudnnTensorDescriptor_t& trunkDescriptor
  ) {
    maxBatchSize = maxBatchSz;
    xSize = xS;
    ySize = yS;
    v1Channels = desc->v1Conv.outChannels;

    v1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    v2OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; i++) {
      cudnnTensorDescriptor_t& v1OutDescriptor = v1OutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& v2OutDescriptor = v2OutDescriptors[batchSize-1];

      CUDNN_ERR(cudnnCreateTensorDescriptor(&v1OutDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
        v1OutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->v1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(cudnnCreateTensorDescriptor(&v2OutDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
        v2OutDescriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->v2Mul.outChannels,
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
      cudnnDestroyTensorDescriptor(v2OutDescriptors[batchSize-1]);
    }

    delete[] v1OutDescriptors;
    delete[] v2OutDescriptors;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    cudnnTensorDescriptor_t& trunkDescriptor,
    int batchSize
  ) {
    size_t bytes = 0;
    size_t b;

    cudnnTensorDescriptor_t& v1OutDescriptor = v1OutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& v2OutDescriptor = v2OutDescriptors[batchSize-1];

    b = v1Conv->requiredWorkspaceBytes(cudaHandles,trunkDescriptor,v1OutDescriptor);
    bytes = std::max(bytes,b);
    b = v2Mul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = v3Mul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    cudnnTensorDescriptor_t& trunkDescriptor,
    int batchSize,
    float* trunkOutBuf,
    float* v1OutBuf,
    float* v1OutBuf2,
    float* v1MeanBuf,
    float* v2Buf,
    float* valueBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) {
    cudnnTensorDescriptor_t& v1OutDescriptor = v1OutDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& v2OutDescriptor = v2OutDescriptors[batchSize-1];

    v1Conv->apply(cudaHandles,trunkDescriptor,v1OutDescriptor,batchSize,trunkOutBuf,v1OutBuf);
    v1BN->apply(cudaHandles,v1OutDescriptor,v1OutDescriptor,v1OutBuf,v1OutBuf2);
    v1Activation->apply(cudaHandles,v1OutDescriptor,v1OutDescriptor,v1OutBuf2,v1OutBuf2);

    customCudaPoolRowsSum(v1OutBuf2,v1MeanBuf,batchSize*v1Channels,xSize*ySize);
    float meanScale = 1.0f / (xSize*ySize);
    CUBLAS_ERR(cublasSscal(cudaHandles->cublas, batchSize*v1Channels, &meanScale, v1MeanBuf, 1));

    v2Mul.apply(cudaHandles,batchSize,v1MeanBuf,v2Buf,workspaceBuf,workspaceBytes);
    v2Bias.apply(cudaHandles,batchSize,v2Buf);
    v2Activation->apply(cudaHandles,v2OutDescriptor,v2OutDescriptor,v2Buf,v2Buf);
    //TODO CRELU

    v3Mul.apply(cudaHandles,batchSize,v2MeanBuf,valueBuf,workspaceBuf,workspaceBytes);
    v3Bias.apply(cudaHandles,batchSize,valueBuf);
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

    if(numInputChannels != trunk->initialConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": numInputChannels (%d) != trunk->initialConv.inChannels (%d)", numInputChannels, trunk->initialConv.inChannels
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
};


struct Model {
  int maxBatchSize;
  int xSize;
  int ySize;
  int numInputChannels;

  cudnnTensorDescriptor_t* inputDescriptors;

  Trunk* trunk;
  PolicyHead* policyHead;
  ValueHead* valueHead;

  Model(
    CudaHandles* cudaHandles,
    const ModelDesc* desc,
    int maxBatchSz,
  ) {
    maxBatchSize = maxBatchSz;
    xSize = desc->xSize;
    ySize = desc->ySize;
    numInputChannels = desc->numInputChannels;

    if(xSize != NNPos::MAX_BOARD_LEN)
      throw StringError("Currently neural net xSize must be NNPos::MAX_BOARD_LEN (" + Global::intToString(NNPos::MAX_BOARD_LEN) + ")");
    if(ySize != NNPos::MAX_BOARD_LEN)
      throw StringError("Currently neural net ySize must be NNPos::MAX_BOARD_LEN (" + Global::intToString(NNPos::MAX_BOARD_LEN) + ")");
    if(numInputChannels != NNInputs::NUM_FEATURES_V1)
      throw StringError("Currently neural net numInputChannels must be NNInputs::NUM_FEATURES_V1 (" + Global::intToString(NNInputs::NUM_FEATURES_V1) + ")");

    inputDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];

      CUDNN_ERR(cudnnCreateTensorDescriptor(&inputDescriptor));
      CUDNN_ERR(cudnnSetTensor4dDescriptor(
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
  ) {
    size_t bytes = 0;
    size_t b;

    cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& trunkDescriptor = trunk->trunkDescriptors[batchSize-1];

    b = trunk->requiredWorkspaceBytes(cudaHandles,inputDescriptor);
    bytes = std::max(bytes,b);
    b = policyHead->requiredWorkspaceBytes(cudaHandles,trunkDescriptor);
    bytes = std::max(bytes,b);
    b = valueHead->requiredWorkspaceBytes(cudaHandles,trunkDescriptor);
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
    float* p2OutBuf,
    float* g1PassBuf,
    float* policyBuf,

    float* v1OutBuf,
    float* v1OutBuf2,
    float* v1MeanBuf,
    float* v2Buf,
    float* valueBuf,

    float* workspaceBuf,
    size_t workspaceBytes
  ) {
    cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& trunkDescriptor = trunk->trunkDescriptors[batchSize-1];

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
      p2OutBuf,
      g1PassBuf,
      policyBuf,
      workspaceBuf,
      workspaceBytes
    );
    valueHead->apply(
      cudaHandles,
      trunkDescriptors,
      batchSize,
      trunkOutBuf,
      v1OutBuf,
      v1OutBuf2,
      v1MeanBuf,
      v2Buf,
      valueBuf,
    );
  }

};


//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
}

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  LoadedModel* loadedModel = new LoadedModel();

  ifstream in(file);
  loadedModel->modelDesc = ModelDesc(in);
  in.close();
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}


//------------------------------------------------------------------------------

struct LocalGpuHandle {
  CudaHandles* cudaHandles;
  Model* model;

  LocalGpuHandle(LoadedModel* loadedModel, int maxBatchSize) {
    cudaHandles = new CudaHandles();
    model = new Model(cudaHandles,&(loadedModel->modelDesc),maxBatchSize);
  }
  ~LocalGpuHandle() {
    delete model;
    delete cudaHandles;
  }

  LocalGpuHandle() = delete;
  LocalGpuHandle(const LocalGpuHandle&) = delete;
  LocalGpuHandle& operator=(const LocalGpuHandle&) = delete;
}

LocalGpuHandle* NeuralNet::createLocalGpuHandle(LoadedModel* loadedModel, int maxBatchSize, int cudaDeviceIdxForThisThread) {
  CUDA_ERR(cudaSetDevice(cudaDeviceIdxForThisThread));

  LocalGpuHandle* gpuHandle = new LocalGpuHandle(loadedModel,maxBatchSize);
  return gpuHandle;
}

void NeuralNet::freeLocalGpuHandle(LocalGpuHandle* gpuHandle) {
  delete gpuHandle;
}

//------------------------------------------------------------------------------

struct InputBuffers {

  float* inputBuf;
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
  float* p2OutBuf;
  float* g1PassBuf;
  float* policyBuf;

  float* v1OutBuf;
  float* v1OutBuf2;
  float* v1MeanBuf;
  float* v2Buf;
  float* valueBuf;

  float* workspaceBuf;
  size_t workspaceBytes

};




#endif
