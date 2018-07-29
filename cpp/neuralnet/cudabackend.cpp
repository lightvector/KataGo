
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


struct LocalGpuHandle {
  cublasHandle_t cublas;
  cudnnHandle_t cudnn;
};

LocalGpuHandle* NeuralNet::createLocalGpuHandle(int cudaDeviceIdxForThisThread) {
  CUDA_ERR(cudaSetDevice(cudaDeviceIdxForThisThread));

  LocalGpuHandle* gpuHandle = new LocalGpuHandle();
  CUBLAS_ERR(cublasCreate(&(gpuHandle->cublas)));
  CUDNN_ERR(cudnnCreate(&(gpuHandle->cudnn)));
  return gpuHandle;
}

void NeuralNet::freeLocalGpuHandle(LocalGpuHandle* gpuHandle) {
  cublasDestroy(gpuHandle->cublas);
  cudnnDestroy(gpuHandle->cudnn);
  delete gpuHandle;
}


//---------------------------------------------------------------------------------

struct ConvLayerDesc {
  int convYSize;
  int convXSize;
  int inChannels;
  int outChannels;
  int dilationY;
  int dilationX;
  vector<float> weights;

  ConvLayerDesc() {}

  ConvLayerDesc(istream& in) {
    in >> convYSize;
    in >> convXSize;
    in >> inChannels;
    in >> outChannels;
    in >> dilationY;
    in >> dilationX;

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
            weights[oc * ocStride + ic * icStride + y * yStride + x * xStride] = w;
          }
        }
      }
    }
  }
};

struct ConvLayer {
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
  cudnnConvolutionFwdAlgo_t convolutionAlgorithm;
  float* filterBuf;

  ConvLayer(
    LocalGpuHandle* gpuHandle,
    const ConvLayerDesc* desc,
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor
  ) {
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    int inChannels = desc->inChannels;
    int outChannels = desc->outChannels;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = convXSize / 2 + dilationX;
    int paddingY = convYSize / 2 + dilationY;

    if(convXSize % 2 != 1 || convYSize % 2 != 1)
      throw new StringError("Convolution filter sizes must be odd, found even sizes");

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

    size_t bytesMemoryLimit = 0;
    CUDNN_ERR(cudnnGetConvolutionForwardAlgorithm(
      gpuHandle->cudnn,
      inputDescriptor,
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptor,
      CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      bytesMemoryLimit,
      &convolutionAlgorithm
    ));

    assert(desc->weights.size() == convYSize * convXSize * inChannels * outChannels);
    size_t filterBytes = sizeof(float) * convYSize * convXSize * inChannels * outChannels;

    CUDA_ERR(cudaMalloc(&filterBuf, filterBytes));
    CUDA_ERR(cudaMemcpy(filterBuf, desc->weights.data(), filterBytes, cudaMemcpyHostToDevice));

  }

  ~ConvLayer() {
    cudaFree(filterBuf);
    cudnnDestroyFilterDescriptor(filterDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
  }

  size_t requiredWorkspaceBytes(
    LocalGpuHandle* gpuHandle,
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor
  ) {
    size_t workspaceBytes = 0;
    CUDNN_ERR(cudnnGetConvolutionForwardWorkspaceSize(
      gpuHandle->cudnn,
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
    LocalGpuHandle* gpuHandle,
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
      gpuHandle->cudnn,
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
  int numChannels;
  float epsilon;
  vector<float> mean;
  vector<float> variance;
  vector<float> bias;
  vector<float> scale;

  BNLayerDesc() {}

  BNLayerDesc(istream& in) {
    in >> numChannels;
    in >> epsilon;

    float w;
    mean.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      in >> w;
      mean[c] = w;
    }
    variance.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      in >> w;
      variance[c] = w;
    }
    bias.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      in >> w;
      bias[c] = w;
    }
    scale.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      w = 1.0; //No batch norms scaling right now
      scale[c] = w;
    }
  }
};

struct BNLayer {
  int numChannels;
  float epsilon;
  cudnnTensorDescriptor_t bufDescriptor;
  float* meanBuf;
  float* varianceBuf;
  float* biasBuf;
  float* scaleBuf;

  BNLayer(
    LocalGpuHandle* gpuHandle,
    const BNLayerDesc* desc,
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor
  ) {
    (void)gpuHandle;
    (void)inputDescriptor;
    (void)outputDescriptor;

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

    assert(desc->bias.size() == numChannels);
    CUDA_ERR(cudaMalloc(&biasBuf, bufBytes));
    CUDA_ERR(cudaMemcpy(biasBuf, desc->bias.data(), bufBytes, cudaMemcpyHostToDevice));

    assert(desc->scale.size() == numChannels);
    CUDA_ERR(cudaMalloc(&scaleBuf, bufBytes));
    CUDA_ERR(cudaMemcpy(scaleBuf, desc->scale.data(), bufBytes, cudaMemcpyHostToDevice));
  }

  ~BNLayer() {
    cudaFree(meanBuf);
    cudaFree(varianceBuf);
    cudaFree(biasBuf);
    cudaFree(scaleBuf);
    cudnnDestroyTensorDescriptor(bufDescriptor);
  }

  void apply(
    LocalGpuHandle* gpuHandle,
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor,
    float* inputBuf,
    float* outputBuf
  ) {
    const float alpha = 1;
    const float beta = 0;
    CUDNN_ERR(cudnnBatchNormalizationForwardInference(
      gpuHandle->cudnn,
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

  ActivationLayer(
    LocalGpuHandle* gpuHandle,
    const ActivationLayerDesc* desc,
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor
  ) {
    (void)gpuHandle;
    (void)desc;
    (void)inputDescriptor;
    (void)outputDescriptor;

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
    LocalGpuHandle* gpuHandle,
    cudnnTensorDescriptor_t& inputDescriptor,
    cudnnTensorDescriptor_t& outputDescriptor,
    float* inputBuf,
    float* outputBuf
  ) {
    const float alpha = 1;
    const float beta = 0;

    CUDNN_ERR(cudnnActivationForward(
      gpuHandle->cudnn,
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
  int inChannels;
  int outChannels;
  vector<float> weights;

  MatMulLayerDesc() {}

  MatMulLayerDesc(istream& in) {
    in >> inChannels;
    in >> outChannels;

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
        weights[oc * ocStride + ic * icStride] = w;
      }
    }
  }
};

struct MatMulLayer {
  int batchSize;
  int inChannels;
  int outChannels;
  float* matBuf;

  MatMulLayer(
    LocalGpuHandle* gpuHandle,
    const MatMulLayerDesc* desc,
    int batchSz
  ) {
    (void)gpuHandle;
    batchSize = batchSz;
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
    LocalGpuHandle* gpuHandle
  ) {
    (void)gpuHandle;
    size_t workspaceBytes = 0;
    return workspaceBytes;
  }

  void apply(
    LocalGpuHandle* gpuHandle,
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
      gpuHandle->cublas,
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

struct DilatedResidualBlockDesc {
  BNLayerDesc preBN;
  ActivationLayerDesc preActivation;
  ConvLayerDesc regularConv;
  ConvLayerDesc dilatedConv;
  BNLayerDesc midBN;
  ActivationLayerDesc midActivation;
  ConvLayerDesc finalConv;

  //TODO we should have lots of cross-layer size consistency asserts

  DilatedResidualBlockDesc(istream& in) {
    preBN = BNLayerDesc(in);
    preActivation = ActivationLayerDesc(in);
    regularConv = ConvLayerDesc(in);
    dilatedConv = ConvLayerDesc(in);
    midBN = BNLayerDesc(in);
    midActivation = ActivationLayerDesc(in);
    finalConv = ConvLayerDesc(in);
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

  int batchSize;
  int xSize;
  int ySize;
  int regularChannels;
  int dilatedChannels;

  DilatedResidualBlock(
    LocalGpuHandle* gpuHandle,
    const DilatedResidualBlockDesc* desc,
    int batchSz,
    int xS,
    int yS,
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& dilatedOutDescriptor,
    cudnnTensorDescriptor_t& midInDescriptor
  ): preBN(gpuHandle,&desc->preBN,trunkDescriptor,trunkDescriptor),
     preActivation(gpuHandle,&desc->preActivation,trunkDescriptor,trunkDescriptor),
     regularConv(gpuHandle,&desc->regularConv,trunkDescriptor,regularOutDescriptor),
     dilatedConv(gpuHandle,&desc->dilatedConv,trunkDescriptor,dilatedOutDescriptor),
     midBN(gpuHandle,&desc->midBN,midInDescriptor,midInDescriptor),
     midActivation(gpuHandle,&desc->midActivation,midInDescriptor,midInDescriptor),
     finalConv(gpuHandle,&desc->finalConv,midInDescriptor,trunkDescriptor),
     batchSize(batchSz),
     xSize(xS),
     ySize(yS),
     regularChannels(desc->regularConv.outChannels),
     dilatedChannels(desc->dilatedConv.outChannels)
  {
  }

  ~DilatedResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    LocalGpuHandle* gpuHandle,
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& dilatedOutDescriptor,
    cudnnTensorDescriptor_t& midInDescriptor
  ) {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(gpuHandle,trunkDescriptor,regularOutDescriptor);
    bytes = std::max(bytes,b);
    b = dilatedConv.requiredWorkspaceBytes(gpuHandle,trunkDescriptor,dilatedOutDescriptor);
    bytes = std::max(bytes,b);
    b = finalConv.requiredWorkspaceBytes(gpuHandle,midInDescriptor,trunkDescriptor);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    LocalGpuHandle* gpuHandle,
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& dilatedOutDescriptor,
    cudnnTensorDescriptor_t& midInDescriptor,
    float* trunkInBuf,
    float* trunkScratchBuf,
    float* trunkOutBuf,
    float* regularOutBuf,
    float* dilatedOutBuf,
    float* midInBuf,
    float* midScratchBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) {
    preBN.apply(gpuHandle,trunkDescriptor,trunkDescriptor,trunkInBuf,trunkScratchBuf);
    preActivation.apply(gpuHandle,trunkDescriptor,trunkDescriptor,trunkScratchBuf,trunkOutBuf);
    regularConv.apply(gpuHandle,trunkDescriptor,regularOutDescriptor,trunkOutBuf,regularOutBuf,workspaceBuf,workspaceBytes);
    dilatedConv.apply(gpuHandle,trunkDescriptor,dilatedOutDescriptor,trunkOutBuf,dilatedOutBuf,workspaceBuf,workspaceBytes);
    customCudaChannelConcat(
      regularOutBuf,dilatedOutBuf,midInBuf,
      xSize*ySize*regularChannels,
      xSize*ySize*dilatedChannels,
      batchSize
    );
    midBN.apply(gpuHandle,midInDescriptor,midInDescriptor,midInBuf,midScratchBuf);
    midActivation.apply(gpuHandle,midInDescriptor,midInDescriptor,midScratchBuf,midScratchBuf);
    finalConv.apply(gpuHandle,midInDescriptor,trunkDescriptor,midScratchBuf,trunkOutBuf,workspaceBuf,workspaceBytes);
  }

};

struct GlobalPoolingResidualBlockDesc {
  BNLayerDesc preBN;
  ActivationLayerDesc preActivation;
  ConvLayerDesc regularConv;
  ConvLayerDesc gpoolConv;
  BNLayerDesc gpoolBN;
  ActivationLayerDesc gpoolActivation;
  MatMulLayerDesc gpoolToMidMul;
  BNLayerDesc midBN;
  ActivationLayerDesc midActivation;
  ConvLayerDesc finalConv;

  GlobalPoolingResidualBlockDesc(istream& in) {
    preBN = BNLayerDesc(in);
    preActivation = ActivationLayerDesc(in);
    regularConv = ConvLayerDesc(in);
    gpoolConv = ConvLayerDesc(in);
    gpoolBN = BNLayerDesc(in);
    gpoolActivation = ActivationLayerDesc(in);
    gpoolToMidMul = MatMulLayerDesc(in);
    midBN = BNLayerDesc(in);
    midActivation = ActivationLayerDesc(in);
    finalConv = ConvLayerDesc(in);
  }
};

struct GlobalPoolingResidualBlock {
  BNLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  ConvLayer gpoolConv;
  BNLayer gpoolBN;
  ActivationLayer gpoolActivation;
  MatMulLayer gpoolToMidMul;
  BNLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  int batchSize;
  int xSize;
  int ySize;
  int regularChannels;
  int gpoolChannels;

  GlobalPoolingResidualBlock(
    LocalGpuHandle* gpuHandle,
    const GlobalPoolingResidualBlockDesc* desc,
    int batchSz,
    int xS,
    int yS,
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& gpoolOutDescriptor
  ): preBN(gpuHandle,&desc->preBN,trunkDescriptor,trunkDescriptor),
     preActivation(gpuHandle,&desc->preActivation,trunkDescriptor,trunkDescriptor),
     regularConv(gpuHandle,&desc->regularConv,trunkDescriptor,regularOutDescriptor),
     gpoolConv(gpuHandle,&desc->gpoolConv,trunkDescriptor,gpoolOutDescriptor),
     gpoolBN(gpuHandle,&desc->gpoolBN,gpoolOutDescriptor,gpoolOutDescriptor),
     gpoolActivation(gpuHandle,&desc->gpoolActivation,gpoolOutDescriptor,gpoolOutDescriptor),
     gpoolToMidMul(gpuHandle,&desc->gpoolToMidMul,batchSz),
     midBN(gpuHandle,&desc->midBN,regularOutDescriptor,regularOutDescriptor),
     midActivation(gpuHandle,&desc->midActivation,regularOutDescriptor,regularOutDescriptor),
     finalConv(gpuHandle,&desc->finalConv,regularOutDescriptor,trunkDescriptor),
     batchSize(batchSz),
     xSize(xS),
     ySize(yS),
     regularChannels(desc->regularConv.outChannels),
     gpoolChannels(desc->gpoolConv.outChannels)
  {
  }

  ~GlobalPoolingResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    LocalGpuHandle* gpuHandle,
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& gpoolOutDescriptor
  ) {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(gpuHandle,trunkDescriptor,regularOutDescriptor);
    bytes = std::max(bytes,b);
    b = gpoolConv.requiredWorkspaceBytes(gpuHandle,trunkDescriptor,gpoolOutDescriptor);
    bytes = std::max(bytes,b);
    b = gpoolToMidMul.requiredWorkspaceBytes(gpuHandle);
    bytes = std::max(bytes,b);
    b = finalConv.requiredWorkspaceBytes(gpuHandle,regularOutDescriptor,trunkDescriptor);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    LocalGpuHandle* gpuHandle,
    cudnnTensorDescriptor_t& trunkDescriptor,
    cudnnTensorDescriptor_t& regularOutDescriptor,
    cudnnTensorDescriptor_t& gpoolOutDescriptor,
    float* trunkInBuf,
    float* trunkScratchBuf,
    float* trunkOutBuf,
    float* regularOutBuf,
    float* gpoolOutBuf,
    float* gpoolMeanBuf,
    float* gpoolMaxBuf,
    float* gpoolConcatBuf,
    float* gpoolMultipliedBuf,
    float* regularScratchBuf,
    float* workspaceBuf,
    size_t workspaceBytes
  ) {
    preBN.apply(gpuHandle,trunkDescriptor,trunkDescriptor,trunkInBuf,trunkScratchBuf);
    preActivation.apply(gpuHandle,trunkDescriptor,trunkDescriptor,trunkScratchBuf,trunkOutBuf);
    regularConv.apply(gpuHandle,trunkDescriptor,regularOutDescriptor,trunkOutBuf,regularOutBuf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(gpuHandle,trunkDescriptor,gpoolOutDescriptor,trunkOutBuf,gpoolOutBuf,workspaceBuf,workspaceBytes);

    customCudaPoolRowsSum(gpoolOutBuf,gpoolMeanBuf,batchSize*gpoolChannels,xSize*ySize);
    customCudaPoolRowsMax(gpoolOutBuf,gpoolMaxBuf,batchSize*gpoolChannels,xSize*ySize);
    float meanScale = 1.0f / xSize*ySize;
    CUBLAS_ERR(cublasSscal(gpuHandle->cublas, batchSize*gpoolChannels*xSize*ySize, &meanScale, gpoolMeanBuf, 1));
    customCudaChannelConcat(
      gpoolMeanBuf,gpoolMaxBuf,gpoolConcatBuf,
      xSize*ySize*gpoolChannels,
      xSize*ySize*gpoolChannels,
      batchSize
    );
    gpoolToMidMul.apply(gpuHandle,gpoolConcatBuf,gpoolMultipliedBuf,workspaceBuf,workspaceBytes);
    //TODO need to broadcast out again
    //add gpoolMultipliedBuf as per-channel biases to regularOutBuf
    midBN.apply(gpuHandle,regularOutDescriptor,regularOutDescriptor,regularOutBuf,regularScratchBuf);
    midActivation.apply(gpuHandle,regularOutDescriptor,regularOutDescriptor,regularScratchBuf,regularScratchBuf);
    finalConv.apply(gpuHandle,regularOutDescriptor,trunkDescriptor,regularScratchBuf,trunkOutBuf,workspaceBuf,workspaceBytes);
  }

};

static const int DILATED_BLOCK_KIND = 0;
static const int GLOBAL_POOLING_BLOCK_KIND = 1;


    // status = cudnnCreateTensorDescriptor(&inputDescriptor);
    // checkCudnnStatus(status,"cudnnCreateTensorDescriptor");
    // status = cudnnSetTensor4dDescriptor(
    //   inputDescriptor,
    //   (isFirstConvolution ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
    //   CUDNN_DATA_FLOAT,
    //   batchSize,
    //   inChannels,
    //   ySize,
    //   xSize
    // );
    // checkCudnnStatus(status,"cudnnSetTensor4dDescriptor");

    // status = cudnnCreateTensorDescriptor(&outputDescriptor);
    // checkCudnnStatus(status,"cudnnCreateTensorDescriptor");
    // status = cudnnSetTensor4dDescriptor(
    //   outputDescriptor,
    //   CUDNN_TENSOR_NCHW,
    //   CUDNN_DATA_FLOAT,
    //   batchSize,
    //   outChannels,
    //   ySize,
    //   xSize
    // );
    // checkCudnnStatus(status,"cudnnSetTensor4dDescriptor");

// struct LoadedModel {
//   int numResidualBlocks;
//   ConvLayerDesc firstConv;
//   pair<int,void*> blockDescs;

//   cudnnTensorDescriptor_t inputDescriptor;
//   cudnnTensorDescriptor_t trunkDescriptor;

//   cudnnTensorDescriptor_t trunkDescriptor;

// };
// struct InputBuffers {

// };

#endif
