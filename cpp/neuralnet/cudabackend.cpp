#ifdef USE_CUDA_BACKEND
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudaincludes.h"

#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/cudautils.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/desc.h"

#include "../core/simpleallocator.h"

#include "../external/half-2.1.0/include/half.hpp"

//------------------------
#include "../core/using.h"
//------------------------

using half_t = half_float::half;

//Define this to print out some of the intermediate values of the neural net
//#define DEBUG_INTERMEDIATE_VALUES

void NeuralNet::globalInitialize() {
  //Empty for cudnn backend
}

void NeuralNet::globalCleanup() {
  cudaDeviceReset();
}

struct CudaHandles {
  cublasHandle_t cublas;
  cudnnHandle_t cudnn;
  int majorComputeCapability;
  int minorComputeCapability;

  CudaHandles(int major, int minor) {
    CUBLAS_ERR("CudaHandles",cublasCreate(&cublas));
    CUDNN_ERR("CudaHandles",cudnnCreate(&cudnn));

    majorComputeCapability = major;
    minorComputeCapability = minor;
  }

  ~CudaHandles() {
    cublasDestroy(cublas);
    cudnnDestroy(cudnn);
  }

  static CudaHandles* cudaHandlesTesting() {
    const int gpuIdxForThisThread = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,gpuIdxForThisThread);
    return new CudaHandles(prop.major, prop.minor);
  }

  CudaHandles(const CudaHandles&) = delete;
  CudaHandles& operator=(const CudaHandles&) = delete;
};

//---------------------------------------------------------------------------------

template<typename T>
struct ByBatchSize {
  int maxBatchSize;
  T* data;
  cudnnStatus_t (*destroyFunc)(T);

  ByBatchSize()
    : maxBatchSize(0), data(nullptr), destroyFunc(nullptr)
  {}

  ByBatchSize(
    int maxBatchSize_
  ) : maxBatchSize(maxBatchSize_), data(nullptr), destroyFunc(nullptr) {
    data = new T[maxBatchSize];
  }

  ByBatchSize(const ByBatchSize&) = delete;
  ByBatchSize& operator=(const ByBatchSize&) = delete;

  ~ByBatchSize() {
    if(destroyFunc != nullptr && data != nullptr) {
      for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
        (*destroyFunc)(data[batchSize-1]);
      }
    }
    if(data != nullptr) {
      delete[] data;
      data = nullptr;
    }
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

template<typename T>
struct ByBatchSizeView {
  int maxBatchSize;
  T* data;

  ByBatchSizeView()
    : maxBatchSize(0), data(nullptr)
  {}

  ByBatchSizeView(const ByBatchSize<T>& toView)
    : maxBatchSize(toView.maxBatchSize), data(toView.data)
  {}
  ByBatchSizeView& operator=(const ByBatchSize<T>& toView) {
    maxBatchSize = toView.maxBatchSize;
    data = toView.data;
  }

  ~ByBatchSizeView() {
  }
  T& operator[](int batchSize) {
    return data[batchSize-1];
  }
  const T& operator[](int batchSize) const {
    return data[batchSize-1];
  }
};

//---------------------------------------------------------------------------------


//channels, useFP16, useNHWC
typedef std::tuple<int, bool, bool> CudnnTensorDesc4DKey;

struct CudnnManager {
  string name;
  int maxBatchSize;
  int xSize;
  int ySize;
  std::map<CudnnTensorDesc4DKey, ByBatchSize<cudnnTensorDescriptor_t>*> tensorDesc4DByBatchSizeByKey;

  CudnnManager(string name_, int maxBatchSize_, int xSize_, int ySize_)
    :name(name_),
     maxBatchSize(maxBatchSize_),
     xSize(xSize_),
     ySize(ySize_),
     tensorDesc4DByBatchSizeByKey()
  {
  }

  ~CudnnManager() {
    for(auto& iter: tensorDesc4DByBatchSizeByKey) {
      delete iter.second;
    }
  }

  ByBatchSizeView<cudnnTensorDescriptor_t> getTensorDesc4DByBatchSize(
    int channels, bool useFP16, bool useNHWC
  ) {
    auto iter = tensorDesc4DByBatchSizeByKey.find({channels, useFP16, useNHWC});
    if(iter != tensorDesc4DByBatchSizeByKey.end()) {
      return ByBatchSizeView<cudnnTensorDescriptor_t>(*(iter->second));
    }
    ByBatchSize<cudnnTensorDescriptor_t>* descs = new ByBatchSize<cudnnTensorDescriptor_t>(maxBatchSize);
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& desc = (*descs)[batchSize];
      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&desc));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
                  desc,
                  (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
                  (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
                  batchSize,
                  channels,
                  ySize,
                  xSize
                ));
    }
    descs->destroyFunc = cudnnDestroyTensorDescriptor;
    tensorDesc4DByBatchSizeByKey[{channels, useFP16, useNHWC}] = descs;
    return ByBatchSizeView<cudnnTensorDescriptor_t>(*descs);
  }
};

//---------------------------------------------------------------------------------

struct ScratchBuffers {

  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  SimpleAllocator<void*>* allocator;

  // Not scratch, but convenient to have here
  void* zeroBuf;
  void* oneBuf;

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(int maxBatchSize, int xSize, int ySize, bool useFP16)
    : batchXYFloatBytes((size_t)maxBatchSize * xSize * ySize * sizeof(float)),
      batchFloatBytes((size_t)maxBatchSize * sizeof(float)),
      batchXYBytes((size_t)maxBatchSize * xSize * ySize * (useFP16 ? sizeof(half) : sizeof(float))),
      batchBytes((size_t)maxBatchSize * (useFP16 ? sizeof(half) : sizeof(float)))
  {
    std::function<void*(size_t)> allocateFunc = [](size_t size) {
      void* buf;
      CUDA_ERR("ScratchBuffers",cudaMalloc(&buf, size));
      return buf;
    };
    std::function<void(void*)> releaseFunc = [](void* buf) {
      cudaFree(buf);
    };

    allocator = new SimpleAllocator<void*>(allocateFunc, releaseFunc);

    CudaUtils::hostMallocZeroOneBufs(zeroBuf, oneBuf, useFP16);
  }
  ~ScratchBuffers() {
    delete allocator;
    free(zeroBuf);
    free(oneBuf);
  }

  size_t getBufSizeXY(int channels) const {
    return channels * batchXYBytes;
  }
  size_t getBufSizeXYFloat(int channels) const {
    return channels * batchXYFloatBytes;
  }
  size_t getBufSizeFloat(int channels) const {
    return channels * batchFloatBytes;
  }
  size_t getBufSize(int channels) const {
    return channels * batchBytes;
  }

};


//---------------------------------------------------------------------------------

struct ConvLayer {
  string name;
  int inChannels;
  int outChannels;
  ByBatchSizeView<cudnnTensorDescriptor_t> inputDescriptors;
  ByBatchSizeView<cudnnTensorDescriptor_t> outputDescriptors;
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
#if CUDNN_MAJOR >= 8
  ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>* convolutionAlgorithms; //array of one for each batch size
#else
  ByBatchSize<cudnnConvolutionFwdAlgo_t>* convolutionAlgorithms; //array of one for each batch size
#endif
  void* filterBuf;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWC
  ) : ConvLayer(cudaHandles, manager, desc, useFP16, useNHWC, useNHWC)
  {}

  ConvLayer(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ConvLayerDesc* desc,
    bool useFP16,
    bool useNHWCIn,
    bool useNHWCOut
  ) {
    name = desc->name;
    int convYSize = desc->convYSize;
    int convXSize = desc->convXSize;
    inChannels = desc->inChannels;
    outChannels = desc->outChannels;
    int dilationY = desc->dilationY;
    int dilationX = desc->dilationX;
    int paddingX = (convXSize / 2) * dilationX;
    int paddingY = (convYSize / 2) * dilationY;

    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    inputDescriptors = manager->getTensorDesc4DByBatchSize(inChannels,useFP16,useNHWCIn);
    outputDescriptors = manager->getTensorDesc4DByBatchSize(outChannels,useFP16,useNHWCOut);
    int maxBatchSize = manager->maxBatchSize;

    bool filterNHWC = useNHWCOut && dilationY == 1 && dilationX == 1;

    CUDNN_ERR(name.c_str(),cudnnCreateFilterDescriptor(&filterDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetFilter4dDescriptor(
      filterDescriptor,
      (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
      (filterNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
      outChannels,
      inChannels,
      convYSize,
      convXSize
    ));

    int yStride = 1;
    int xStride = 1;

    //NVIDIA compute capability 7 is when we first hit Volta architecture, with tensor cores
    //See https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
    bool tensorCoresSupported = cudaHandles->majorComputeCapability >= 7;

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
      (useFP16 && !tensorCoresSupported) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT
    ));
    if(useFP16 && tensorCoresSupported)
      CUDNN_ERR(name.c_str(),cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH));

#if CUDNN_MAJOR >= 8
    convolutionAlgorithms = new ByBatchSize<cudnnConvolutionFwdAlgoPerf_t>(maxBatchSize);
#else
    convolutionAlgorithms = new ByBatchSize<cudnnConvolutionFwdAlgo_t>(maxBatchSize);
#endif

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      if(useFP16 && dilationX <= 1 && dilationY <= 1) {
#if CUDNN_MAJOR >= 8
        (*convolutionAlgorithms)[batchSize].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#else
        (*convolutionAlgorithms)[batchSize] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
#endif
      }
      else {
        const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize];
        const cudnnTensorDescriptor_t& outputDescriptor = outputDescriptors[batchSize];

#if CUDNN_MAJOR >= 8
        int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
        int returnedAlgoCount = -1;
        cudnnConvolutionFwdAlgoPerf_t results[2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm_v7(
          cudaHandles->cudnn,
          inputDescriptor,
          filterDescriptor,
          convolutionDescriptor,
          outputDescriptor,
          requestedAlgoCount,
          &returnedAlgoCount,
          results
        ));
        if(returnedAlgoCount <= 0)
          throw StringError("cudnnGetConvolutionForwardAlgorithm_v7 returned no algorithms?");
        (*convolutionAlgorithms)[batchSize] = results[0];
#else
        size_t bytesMemoryLimit = 0;
        CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardAlgorithm(
           cudaHandles->cudnn,
           inputDescriptor,
           filterDescriptor,
           convolutionDescriptor,
           outputDescriptor,
           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
           bytesMemoryLimit,
           &((*convolutionAlgorithms)[batchSize])
         ));
#endif
      }
    }

    assert(desc->weights.size() == convYSize * convXSize * inChannels * outChannels);

    if(filterNHWC) {
      vector<float> weightsTransposed(desc->weights.size());
      for(int y = 0; y < convYSize; y++) {
        for(int x = 0; x < convXSize; x++) {
          for(int ic = 0; ic < inChannels; ic++) {
            for(int oc = 0; oc < outChannels; oc++) {
              weightsTransposed[((oc*convYSize + y)*convXSize + x)*inChannels + ic] =
                desc->weights[((oc*inChannels + ic)*convYSize + y)*convXSize + x];
            }
          }
        }
      }
      CudaUtils::mallocAndCopyToDevice(name,weightsTransposed,filterBuf,useFP16);
      cudaDeviceSynchronize();
    }
    else
      CudaUtils::mallocAndCopyToDevice(name,desc->weights,filterBuf,useFP16);
  }

  ~ConvLayer() {
    cudaFree(filterBuf);
    cudnnDestroyFilterDescriptor(filterDescriptor);
    cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    delete convolutionAlgorithms;
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t workspaceBytes = 0;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize].algo,
      &workspaceBytes
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnGetConvolutionForwardWorkspaceSize(
      cudaHandles->cudnn,
      inputDescriptors[batchSize],
      filterDescriptor,
      convolutionDescriptor,
      outputDescriptors[batchSize],
      (*convolutionAlgorithms)[batchSize],
      &workspaceBytes
    ));
#endif
    return workspaceBytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
#if CUDNN_MAJOR >= 8
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize].algo,
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#else
    CUDNN_ERR(name.c_str(),cudnnConvolutionForward(
      cudaHandles->cudnn,
      &alpha,
      inputDescriptors[batchSize],
      inputBuf,
      filterDescriptor,
      filterBuf,
      convolutionDescriptor,
      (*convolutionAlgorithms)[batchSize],
      workspaceBuf,
      workspaceBytes,
      &beta,
      outputDescriptors[batchSize],
      outputBuf
    ));
#endif
  }

};


//---------------------------------------------------------------------------------

struct BatchNormLayer {
  string name;
  int numChannels;
  float epsilon;
  int activation;
  int xSize;
  int ySize;

  void* meanBuf;
  void* varianceBuf;
  void* scaleBuf;
  void* biasBuf;

  void* mergedScaleBuf;
  void* mergedBiasBuf;

  bool usingFP16;
  bool usingNHWC;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(
    CudaHandles* cudaHandles,
    const BatchNormLayerDesc* desc,
    const ActivationLayerDesc* actDesc,
    int xS,
    int yS,
    bool useFP16,
    bool useNHWC
  ) {
    (void)cudaHandles;

    name = desc->name;
    numChannels = desc->numChannels;
    epsilon = desc->epsilon;
    activation = actDesc->activation;
    xSize = xS;
    ySize = yS;
    usingFP16 = useFP16;
    usingNHWC = useNHWC;

    assert(desc->mean.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->mean,meanBuf,useFP16);

    assert(desc->variance.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->variance,varianceBuf,useFP16);

    assert(desc->scale.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->scale,scaleBuf,useFP16);

    assert(desc->bias.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->bias,biasBuf,useFP16);

    vector<float> mergedScale(numChannels);
    vector<float> mergedBias(numChannels);
    for(int i = 0; i<numChannels; i++) {
      mergedScale[i] = desc->scale[i] / sqrt(desc->variance[i] + epsilon);
      mergedBias[i] = desc->bias[i] - mergedScale[i] * desc->mean[i];
    }
    CudaUtils::mallocAndCopyToDevice(name,mergedScale,mergedScaleBuf,useFP16);
    CudaUtils::mallocAndCopyToDevice(name,mergedBias,mergedBiasBuf,useFP16);
  }
  ~BatchNormLayer() {
    cudaFree(meanBuf);
    cudaFree(varianceBuf);
    cudaFree(scaleBuf);
    cudaFree(biasBuf);
    cudaFree(mergedScaleBuf);
    cudaFree(mergedBiasBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* inputBuf,
    const void* maskBuf, //ok to be null
    void* outputBuf
  ) const {
    (void)cudaHandles;
    bool applyRelu = activation == ACTIVATION_RELU;
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf,(const float*)mergedBiasBuf,
                                      (const float*)maskBuf,
                                      batchSize,numChannels,xSize*ySize,applyRelu);
      else
        customCudaApplyCScaleBiasNHWC((const float*)inputBuf,(float*)outputBuf,(const float*)mergedScaleBuf,(const float*)mergedBiasBuf,
                                      (const float*)maskBuf,
                                      batchSize,xSize*ySize,numChannels,applyRelu);
    }
    else {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,numChannels,xSize*ySize,applyRelu);
      else
        customCudaApplyCScaleBiasNHWC((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,
                                      (const half*)maskBuf,
                                      batchSize,xSize*ySize,numChannels,applyRelu);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

  }

};


//---------------------------------------------------------------------------------

struct MatMulLayer {
  string name;
  int inChannels;
  int outChannels;
  void* matBuf;
  bool usingFP16;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(
    CudaHandles* cudaHandles,
    const MatMulLayerDesc* desc,
    bool useFP16
  ) {
    (void)cudaHandles;
    name = desc->name;
    inChannels = desc->inChannels;
    outChannels = desc->outChannels;
    usingFP16 = useFP16;

    assert(desc->weights.size() == inChannels * outChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->weights,matBuf,useFP16);
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
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    (void)workspaceBuf;
    (void)workspaceBytes;

    if(!usingFP16) {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      CUBLAS_ERR(name.c_str(),cublasSgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        &alpha,
        (const float*)matBuf,outChannels,
        (const float*)inputBuf,inChannels,
        &beta,
        (float*)outputBuf,outChannels
      ));
    }
    else {
      const half* alpha = (const half*)scratch->oneBuf;
      const half* beta = (const half*)scratch->zeroBuf;
      CUBLAS_ERR(name.c_str(),cublasHgemm(
        cudaHandles->cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        outChannels,
        batchSize,
        inChannels,
        alpha,
        (const half*)matBuf,outChannels,
        (const half*)inputBuf,inChannels,
        beta,
        (half*)outputBuf,outChannels
      ));
    }

  }

};

//---------------------------------------------------------------------------------

struct MatBiasLayer {
  string name;
  int numChannels;
  void* biasBuf;
  bool usingFP16;
  int activation;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(
    CudaHandles* cudaHandles,
    const MatBiasLayerDesc* desc,
    bool useFP16,
    int activation_
  ) {
    (void)cudaHandles;
    name = desc->name;
    numChannels = desc->numChannels;
    usingFP16 = useFP16;
    activation = activation_;

    assert(desc->weights.size() == numChannels);
    CudaUtils::mallocAndCopyToDevice(name,desc->weights,biasBuf,useFP16);
  }

  ~MatBiasLayer() {
    cudaFree(biasBuf);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    void* matBuf
  ) const {
    (void)cudaHandles;
    if(!usingFP16) {
      customCudaAddCBiasInplaceNC((float*)matBuf,(const float*)biasBuf,batchSize,numChannels,activation);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaAddCBiasInplaceNC((half*)matBuf,(const half*)biasBuf,batchSize,numChannels,activation);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
  }

};

//---------------------------------------------------------------------------------

struct NormActConv {
  BatchNormLayer norm;
  ConvLayer conv;

  int inChannels;
  int outChannels;

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;

  NormActConv(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const BatchNormLayerDesc* normDesc,
    const ActivationLayerDesc* actDesc,
    const ConvLayerDesc* convDesc,
    int xS,
    int yS,
    bool useFP16,
    bool useNHWC
  ): norm(cudaHandles,normDesc,actDesc,xS,yS,useFP16,useNHWC),
     conv(cudaHandles,manager,convDesc,useFP16,useNHWC)
  {
    inChannels = norm.numChannels;
    assert(norm.numChannels = conv.inChannels);
    outChannels = conv.outChannels;
  }

  ~NormActConv()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = conv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    bool accumulate,
    void* inBuf,
    void* inScratchBuf,
    void* outBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    norm.apply(cudaHandles,batchSize,inBuf,maskBuf,inScratchBuf);
    conv.apply(cudaHandles,batchSize,accumulate,inScratchBuf,outBuf,workspaceBuf,workspaceBytes);
  }

};


//---------------------------------------------------------------------------------

struct ResidualBlock {
  string name;
  NormActConv normActConv1;
  NormActConv normActConv2;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ResidualBlockDesc* desc,
    int xS,
    int yS,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->regularConv,xS,yS,useFP16,useNHWC),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,xS,yS,useFP16,useNHWC)
  {
  }

  ~ResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> midIn(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    normActConv1.apply(cudaHandles,batchSize,false,trunkBuf,trunkScratchBuf,midIn.buf,maskBuf,workspaceBuf,workspaceBytes);
    normActConv2.apply(cudaHandles,batchSize,true,midIn.buf,midScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};


//----------------------------------------------------------------------------


struct GlobalPoolingResidualBlock {
  string name;
  BatchNormLayer preBN;
  ConvLayer regularConv;
  ConvLayer gpoolConv;
  BatchNormLayer gpoolBN;
  MatMulLayer gpoolToBiasMul;
  NormActConv normActConv2;

  int xSize;
  int ySize;
  int regularChannels;
  int gpoolChannels;
  bool usingFP16;
  bool usingNHWC;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const GlobalPoolingResidualBlockDesc* desc,
    int xS,
    int yS,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN,&desc->preActivation,xS,yS,useFP16,useNHWC),
     regularConv(cudaHandles,manager,&desc->regularConv,useFP16,useNHWC),
     gpoolConv(cudaHandles,manager,&desc->gpoolConv,useFP16,useNHWC),
     gpoolBN(cudaHandles,&desc->gpoolBN,&desc->gpoolActivation,xS,yS,useFP16,useNHWC),
     gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,useFP16),
     normActConv2(cudaHandles,manager,&desc->midBN,&desc->midActivation,&desc->finalConv,xS,yS,useFP16,useNHWC),
     xSize(xS),
     ySize(yS),
     regularChannels(desc->regularConv.outChannels),
     gpoolChannels(desc->gpoolConv.outChannels),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
  {
  }

  ~GlobalPoolingResidualBlock() {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = regularConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolConv.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul.requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*gpoolChannels*xSize*ySize;
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> regularOut(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> regularScratch(scratch->allocator, scratch->getBufSizeXY(regularChannels));
    SizedBuf<void*> gpoolOut(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolOut2(scratch->allocator, scratch->getBufSizeXY(gpoolChannels));
    SizedBuf<void*> gpoolConcat(scratch->allocator, scratch->getBufSize(gpoolChannels*3));
    SizedBuf<void*> gpoolBias(scratch->allocator, scratch->getBufSize(regularChannels));

    preBN.apply(cudaHandles,batchSize,trunkBuf,maskBuf,trunkScratchBuf);
    regularConv.apply(cudaHandles,batchSize,false,trunkScratchBuf,regularOut.buf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(cudaHandles,batchSize,false,trunkScratchBuf,gpoolOut.buf,workspaceBuf,workspaceBytes);
    gpoolBN.apply(cudaHandles,batchSize,gpoolOut.buf,maskBuf,gpoolOut2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,gpoolChannels,xSize*ySize,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const float*)gpoolOut2.buf,(float*)gpoolConcat.buf,batchSize,xSize*ySize,gpoolChannels,maskSumBuf);
    }
    else {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,gpoolChannels,xSize*ySize,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const half*)gpoolOut2.buf,(half*)gpoolConcat.buf,batchSize,xSize*ySize,gpoolChannels,maskSumBuf);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    gpoolToBiasMul.apply(cudaHandles,scratch,batchSize,gpoolConcat.buf,gpoolBias.buf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,regularChannels,xSize*ySize);
      else
        customCudaAddNCBiasInplaceNHWC((float*)regularOut.buf,(const float*)gpoolBias.buf,batchSize,xSize*ySize,regularChannels);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,regularChannels,xSize*ySize);
      else
        customCudaAddNCBiasInplaceNHWC((half*)regularOut.buf,(const half*)gpoolBias.buf,batchSize,xSize*ySize,regularChannels);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    normActConv2.apply(cudaHandles,batchSize,true,regularOut.buf,regularScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

struct BlockStack {
  int numBlocks;
  vector<pair<int,unique_ptr_void>> blocks;

  BlockStack() = delete;
  BlockStack(const BlockStack&) = delete;
  BlockStack& operator=(const BlockStack&) = delete;

  BlockStack(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    int nBlocks,
    const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
    int xS,
    int yS,
    bool useFP16,
    bool useNHWC
  );
  ~BlockStack();

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const;

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;

};

//------------------------------------------------------------------------------

struct NestedBottleneckResidualBlock {
  string name;
  NormActConv normActConv1;
  BlockStack blocks;
  NormActConv normActConv2;

  NestedBottleneckResidualBlock() = delete;
  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlock&) = delete;
  NestedBottleneckResidualBlock& operator=(const NestedBottleneckResidualBlock&) = delete;

  NestedBottleneckResidualBlock(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const NestedBottleneckResidualBlockDesc* desc,
    int xS,
    int yS,
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     normActConv1(cudaHandles,manager,&desc->preBN,&desc->preActivation,&desc->preConv,xS,yS,useFP16,useNHWC),
     blocks(cudaHandles,manager,desc->numBlocks,desc->blocks,xS,yS,useFP16,useNHWC),
     normActConv2(cudaHandles,manager,&desc->postBN,&desc->postActivation,&desc->postConv,xS,yS,useFP16,useNHWC)
  {
  }

  ~NestedBottleneckResidualBlock()
  {}

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;
    b = normActConv1.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = normActConv2.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> mid(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    SizedBuf<void*> midScratch(scratch->allocator, scratch->getBufSizeXY(normActConv1.outChannels));
    assert(normActConv1.outChannels == normActConv2.inChannels);
    normActConv1.apply(cudaHandles,batchSize,false,trunkBuf,trunkScratchBuf,mid.buf,maskBuf,workspaceBuf,workspaceBytes);
    blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      mid.buf,
      midScratch.buf,
      workspaceBuf,
      workspaceBytes
    );
    normActConv2.apply(cudaHandles,batchSize,true,mid.buf,midScratch.buf,trunkBuf,maskBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

BlockStack::BlockStack(
  CudaHandles* cudaHandles,
  CudnnManager* manager,
  int nBlocks,
  const std::vector<std::pair<int, unique_ptr_void>>& descBlocks,
  int xS,
  int yS,
  bool useFP16,
  bool useNHWC
) {
  int xSize = xS;
  int ySize = yS;
  numBlocks = nBlocks;
  assert(numBlocks == descBlocks.size());
  for(int i = 0; i<numBlocks; i++) {
    if(descBlocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new ResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          xSize,
          ySize,
          useFP16,
          useNHWC
        )
      );
      blocks.push_back(make_pair(ORDINARY_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new GlobalPoolingResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          xSize,
          ySize,
          useFP16,
          useNHWC
        )
      );
      blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND,std::move(blockPtr)));
    }
    else if(descBlocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlockDesc* blockDesc = (NestedBottleneckResidualBlockDesc*)descBlocks[i].second.get();
      unique_ptr_void blockPtr = make_unique_void(
        new NestedBottleneckResidualBlock(
          cudaHandles,
          manager,
          blockDesc,
          xSize,
          ySize,
          useFP16,
          useNHWC
        )
      );
      blocks.push_back(make_pair(NESTED_BOTTLENECK_BLOCK_KIND,std::move(blockPtr)));
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}
BlockStack::~BlockStack() {
}

size_t BlockStack::requiredWorkspaceBytes(
  CudaHandles* cudaHandles,
  int batchSize
) const {
  size_t bytes = 0;
  size_t b;

  for(int i = 0; i<blocks.size(); i++) {
    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      b = block->requiredWorkspaceBytes(cudaHandles,batchSize);
      bytes = std::max(bytes,b);
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
  return bytes;
}

void BlockStack::apply(
  CudaHandles* cudaHandles,
  ScratchBuffers* scratch,
  int batchSize,
  void* maskBuf,
  float* maskSumBuf,
  void* trunkBuf,
  void* trunkScratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {

  for(int i = 0; i<blocks.size(); i++) {
#ifdef DEBUG_INTERMEDIATE_VALUES
    debugPrint4D(string("Blockstack before block " + Global::intToString(i)), trunkBuf, batchSize, trunkNumChannels, xSize, ySize, usingNHWC, usingFP16);
#endif

    if(blocks[i].first == ORDINARY_BLOCK_KIND) {
      ResidualBlock* block = (ResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
      GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
      NestedBottleneckResidualBlock* block = (NestedBottleneckResidualBlock*)blocks[i].second.get();
      block->apply(
        cudaHandles,
        scratch,
        batchSize,
        trunkBuf,
        trunkScratchBuf,
        maskBuf,
        maskSumBuf,
        workspaceBuf,
        workspaceBytes
      );
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }
}

//----------------------------------------------------------------------------


struct Trunk {
  string name;
  int version;
  int numBlocks;
  int trunkNumChannels;

  int xSize;
  int ySize;
  bool usingFP16;
  bool usingNHWC;

  std::unique_ptr<ConvLayer> initialConv;
  std::unique_ptr<MatMulLayer> initialMatMul;
  BlockStack blocks;
  std::unique_ptr<BatchNormLayer> trunkTipBN;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const TrunkDesc* desc,
    int xS,
    int yS,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) : blocks(cudaHandles,manager,desc->numBlocks,desc->blocks,xS,yS,useFP16,useNHWC)
  {
    name = desc->name;
    version = desc->version;
    numBlocks = desc->numBlocks;
    trunkNumChannels = desc->trunkNumChannels;
    int midNumChannels = desc->midNumChannels;
    int regularNumChannels = desc->regularNumChannels;
    int gpoolNumChannels = desc->gpoolNumChannels;

    xSize = xS;
    ySize = yS;
    usingFP16 = useFP16;
    usingNHWC = useNHWC;

    int maxBatchSize = manager->maxBatchSize;
    CudaUtils::checkBufferSize(maxBatchSize,xSize,ySize,trunkNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,xSize,ySize,midNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,xSize,ySize,regularNumChannels);
    CudaUtils::checkBufferSize(maxBatchSize,xSize,ySize,gpoolNumChannels);

    initialConv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->initialConv,useFP16,inputsUseNHWC,useNHWC);
    initialMatMul = std::make_unique<MatMulLayer>(cudaHandles,&desc->initialMatMul,useFP16);

    trunkTipBN = std::make_unique<BatchNormLayer>(cudaHandles,&desc->trunkTipBN,&desc->trunkTipActivation,xSize,ySize,useFP16,useNHWC);
    assert(desc->blocks.size() == numBlocks);
  }

  ~Trunk()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = initialConv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    b = initialMatMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);

    b = blocks.requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* inputBuf,
    void* inputGlobalBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> trunkScratch(scratch->allocator, scratch->getBufSizeXY(trunkNumChannels));

    //Feed the conv into trunkScratch.buf, not trunkBuf
    initialConv->apply(cudaHandles,batchSize,false,inputBuf,trunkScratch.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    debugPrint4D(string("Initial bin features"), inputBuf, batchSize, initialConv->inChannels, xSize, ySize, usingNHWC, usingFP16);
    debugPrint4D(string("After initial conv"), trunkScratch.buf, batchSize, trunkNumChannels, xSize, ySize, usingNHWC, usingFP16);
    #endif

    //Feed the matmul into trunkBuf
    initialMatMul->apply(cudaHandles,scratch,batchSize,inputGlobalBuf,trunkBuf,workspaceBuf,workspaceBytes);
    //Then accumulate it into trunkScratch.buf, broadcasting during the process
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,trunkNumChannels,xSize*ySize);
      else
        customCudaAddNCBiasInplaceNHWC((float*)trunkScratch.buf,(const float*)trunkBuf,batchSize,xSize*ySize,trunkNumChannels);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,trunkNumChannels,xSize*ySize);
      else
        customCudaAddNCBiasInplaceNHWC((half*)trunkScratch.buf,(const half*)trunkBuf,batchSize,xSize*ySize,trunkNumChannels);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    //Flip trunkBuf and trunkScratch.buf so that the result gets accumulated in trunkScratch.buf
    blocks.apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkScratch.buf,
      trunkBuf,
      workspaceBuf,
      workspaceBytes
    );

    //And now with the final BN port it from trunkScratch.buf to trunkBuf.
    trunkTipBN->apply(cudaHandles,batchSize,trunkScratch.buf,maskBuf,trunkBuf);
    #ifdef DEBUG_INTERMEDIATE_VALUES
    debugPrint4D(string("Trunk tip"), trunkBuf, batchSize, trunkNumChannels, xSize, ySize, usingNHWC, usingFP16);
    #endif
  }

};

//------------------------------------------------------------------------------

static void fillMaskFloatBufAndMaskSumBuf(void* maskBuf, float*& maskFloatBuf, float*& maskSumBuf, bool usingFP16, int batchSize, int xSize, int ySize) {
  if(!usingFP16) {
    maskFloatBuf = (float*)maskBuf;
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,xSize*ySize,1.0);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
  else {
    customCudaCopyFromHalf((const half*)maskBuf,maskFloatBuf,batchSize*xSize*ySize);
    CUDA_ERR("copyMaskFromHalf",cudaPeekAtLastError());
    customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,xSize*ySize,1.0);
    CUDA_ERR("sumMask",cudaPeekAtLastError());
  }
}


//------------------------------------------------------------------------------

struct PolicyHead {
  string name;
  int version;
  int xSize;
  int ySize;
  int p1Channels;
  int g1Channels;
  int p2Channels;
  bool usingFP16;
  bool usingNHWC;

  std::unique_ptr<ConvLayer> p1Conv;
  std::unique_ptr<ConvLayer> g1Conv;
  std::unique_ptr<BatchNormLayer> g1BN;
  std::unique_ptr<MatMulLayer> gpoolToBiasMul;
  std::unique_ptr<BatchNormLayer> p1BN;
  std::unique_ptr<ConvLayer> p2Conv;
  std::unique_ptr<MatMulLayer> gpoolToPassMul;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const PolicyHeadDesc* desc,
    int xS,
    int yS,
    bool useFP16,
    bool useNHWC
  ) {
    name = desc->name;
    version = desc->version;
    xSize = xS;
    ySize = yS;
    p1Channels = desc->p1Conv.outChannels;
    g1Channels = desc->g1Conv.outChannels;
    p2Channels = desc->p2Conv.outChannels;
    usingFP16 = useFP16;
    usingNHWC = useNHWC;

    p1Conv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->p1Conv,useFP16,useNHWC);
    g1Conv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->g1Conv,useFP16,useNHWC);
    g1BN = std::make_unique<BatchNormLayer>(cudaHandles,&desc->g1BN,&desc->g1Activation,xSize,ySize,useFP16,useNHWC);
    gpoolToBiasMul = std::make_unique<MatMulLayer>(cudaHandles,&desc->gpoolToBiasMul,false);
    p1BN = std::make_unique<BatchNormLayer>(cudaHandles,&desc->p1BN,&desc->p1Activation,xSize,ySize,false,useNHWC);
    p2Conv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->p2Conv,false,useNHWC);
    gpoolToPassMul = std::make_unique<MatMulLayer>(cudaHandles,&desc->gpoolToPassMul,false);
  }

  ~PolicyHead()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = p1Conv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = g1Conv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToBiasMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = p2Conv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = gpoolToPassMul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*g1Channels*xSize*ySize;
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskFloatBuf,
    float* maskSumBuf,
    void* trunkBuf,
    float* policyBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    SizedBuf<void*> p1Out(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> p1Out2(scratch->allocator, scratch->getBufSizeXYFloat(p1Channels)); //Need to hold floats, not just halfs
    SizedBuf<void*> g1Out(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Out2(scratch->allocator, scratch->getBufSizeXY(g1Channels));
    SizedBuf<void*> g1Concat(scratch->allocator, scratch->getBufSizeFloat(g1Channels*3));
    SizedBuf<void*> g1Bias(scratch->allocator, scratch->getBufSizeFloat(p1Channels));
    SizedBuf<void*> p2Out(scratch->allocator, scratch->getBufSizeXYFloat(p2Channels));
    SizedBuf<void*> g1Pass(scratch->allocator, scratch->getBufSizeFloat(p2Channels));

    p1Conv->apply(cudaHandles,batchSize,false,trunkBuf,p1Out.buf,workspaceBuf,workspaceBytes);
    g1Conv->apply(cudaHandles,batchSize,false,trunkBuf,g1Out.buf,workspaceBuf,workspaceBytes);
    g1BN->apply(cudaHandles,batchSize,g1Out.buf,maskBuf,g1Out2.buf);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,g1Channels,xSize*ySize,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const float*)g1Out2.buf,(float*)g1Concat.buf,batchSize,xSize*ySize,g1Channels,maskSumBuf);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaCopyFromHalf((const half*)g1Out2.buf,(float*)workspaceBuf,batchSize*g1Channels*xSize*ySize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(!usingNHWC)
        customCudaPoolRowsGPoolNCHW((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,g1Channels,xSize*ySize,maskSumBuf);
      else
        customCudaPoolRowsGPoolNHWC((const float*)workspaceBuf,(float*)g1Concat.buf,batchSize,xSize*ySize,g1Channels,maskSumBuf);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    gpoolToBiasMul->apply(cudaHandles,scratch,batchSize,g1Concat.buf,g1Bias.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    debugPrint4D(string("p1 pre-gpool-sum"), p1Out.buf, batchSize, p1Channels, xSize, ySize, usingNHWC, usingFP16);
    debugPrint4D(string("g1 pre-gpool"), g1Out.buf, batchSize, g1Channels, xSize, ySize, usingNHWC, usingFP16);
    debugPrint2D(string("g1 pooled"), g1Concat.buf, batchSize, g1Channels*3, usingFP16);
    debugPrint2D(string("g1 biases"), g1Bias.buf, batchSize, p1Channels, usingFP16);
    #endif

    float* p1OutBufA;
    float* p1OutBufB;
    if(!usingFP16) {
      p1OutBufA = (float*)p1Out.buf;
      p1OutBufB = (float*)p1Out2.buf;
    }
    else {
      customCudaCopyFromHalf((const half*)p1Out.buf,(float*)p1Out2.buf,batchSize*p1Channels*xSize*ySize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      p1OutBufA = (float*)p1Out2.buf;
      p1OutBufB = (float*)p1Out.buf;
    }

    if(!usingNHWC)
      customCudaAddNCBiasInplaceNCHW(p1OutBufA,(float*)g1Bias.buf,batchSize,p1Channels,xSize*ySize);
    else
      customCudaAddNCBiasInplaceNHWC(p1OutBufA,(float*)g1Bias.buf,batchSize,xSize*ySize,p1Channels);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    p1BN->apply(cudaHandles,batchSize,p1OutBufA,maskFloatBuf,p1OutBufB);
    p2Conv->apply(cudaHandles,batchSize,false,p1OutBufB,(float*)p2Out.buf,workspaceBuf,workspaceBytes);

    gpoolToPassMul->apply(cudaHandles,scratch,batchSize,g1Concat.buf,g1Pass.buf,workspaceBuf,workspaceBytes);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    debugPrint4D(string("p1 after-gpool-sum"), p1Out.buf, batchSize, p1Channels, xSize, ySize, usingNHWC, usingFP16);
    debugPrint4D(string("p2"), p2Out.buf, batchSize, p2Channels, xSize, ySize, usingNHWC, usingFP16);
    debugPrint2D(string("p2pass"), g1Pass.buf, batchSize, 1, usingFP16);
    #endif

    customCudaChannelConcat(
      (float*)p2Out.buf,(float*)g1Pass.buf,policyBuf,
      xSize*ySize,
      1,
      batchSize
    );
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

  }

};

//------------------------------------------------------------------------------

struct ValueHead {
  string name;
  int version;
  int xSize;
  int ySize;
  int v1Channels;
  int v2Channels;
  int valueChannels;
  int scoreValueChannels;
  int ownershipChannels;
  bool usingFP16;
  bool usingNHWC;

  std::unique_ptr<ConvLayer> v1Conv;
  std::unique_ptr<BatchNormLayer> v1BN;
  std::unique_ptr<MatMulLayer> v2Mul;
  std::unique_ptr<MatBiasLayer> v2Bias;
  std::unique_ptr<MatMulLayer> v3Mul;
  std::unique_ptr<MatBiasLayer> v3Bias;
  std::unique_ptr<MatMulLayer> sv3Mul;
  std::unique_ptr<MatBiasLayer> sv3Bias;
  std::unique_ptr<ConvLayer> vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    CudaHandles* cudaHandles,
    CudnnManager* manager,
    const ValueHeadDesc* desc,
    int xS,
    int yS,
    bool useFP16,
    bool useNHWC
  ) {
    name = desc->name;
    version = desc->version;
    xSize = xS;
    ySize = yS;
    v1Channels = desc->v1Conv.outChannels;
    v2Channels = desc->v2Mul.outChannels;
    valueChannels = desc->v3Mul.outChannels;
    scoreValueChannels = desc->sv3Mul.outChannels;
    ownershipChannels = desc->vOwnershipConv.outChannels;
    usingFP16 = useFP16;
    usingNHWC = useNHWC;

    v1Conv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->v1Conv,useFP16,useNHWC);
    v1BN = std::make_unique<BatchNormLayer>(cudaHandles,&desc->v1BN,&desc->v1Activation,xSize,ySize,useFP16,useNHWC);
    v2Mul = std::make_unique<MatMulLayer>(cudaHandles,&desc->v2Mul,false);
    v2Bias = std::make_unique<MatBiasLayer>(cudaHandles,&desc->v2Bias,false,desc->v2Activation.activation);
    v3Mul = std::make_unique<MatMulLayer>(cudaHandles,&desc->v3Mul,false);
    v3Bias = std::make_unique<MatBiasLayer>(cudaHandles,&desc->v3Bias,false,ACTIVATION_IDENTITY);
    sv3Mul = std::make_unique<MatMulLayer>(cudaHandles,&desc->sv3Mul,false);
    sv3Bias = std::make_unique<MatBiasLayer>(cudaHandles,&desc->sv3Bias,false,ACTIVATION_IDENTITY);
    vOwnershipConv = std::make_unique<ConvLayer>(cudaHandles,manager,&desc->vOwnershipConv,useFP16,useNHWC);
  }

  ~ValueHead()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = v1Conv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = v2Mul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = v3Mul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*v1Channels*xSize*ySize;
    bytes = std::max(bytes,b);

    b = sv3Mul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = vOwnershipConv->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*ownershipChannels*xSize*ySize;
    bytes = std::max(bytes,b);

    return bytes;
  }


  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> v1Out(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Out2(scratch->allocator, scratch->getBufSizeXY(v1Channels));
    SizedBuf<void*> v1Mean(scratch->allocator, scratch->getBufSizeFloat(v1Channels*3));
    SizedBuf<void*> v2Out(scratch->allocator, scratch->getBufSizeFloat(v2Channels));
    SizedBuf<void*> ownershipScratch(scratch->allocator, scratch->getBufSizeXYFloat(ownershipChannels));

    v1Conv->apply(cudaHandles,batchSize,false,trunkBuf,v1Out.buf,workspaceBuf,workspaceBytes);
    v1BN->apply(cudaHandles,batchSize,v1Out.buf,maskBuf,v1Out2.buf);

    void* bufToBePooled = v1Out2.buf;
    if(usingFP16) {
      customCudaCopyFromHalf((const half*)v1Out2.buf,(float*)workspaceBuf,batchSize*v1Channels*xSize*ySize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      bufToBePooled = workspaceBuf;
    }

    if(!usingNHWC)
      customCudaValueHeadPoolNCHW((float*)bufToBePooled,(float*)v1Mean.buf,batchSize,v1Channels,xSize*ySize,maskSumBuf);
    else
      customCudaValueHeadPoolNHWC((const float*)bufToBePooled,(float*)v1Mean.buf,batchSize,xSize*ySize,v1Channels,maskSumBuf);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    v2Mul->apply(cudaHandles,scratch,batchSize,v1Mean.buf,v2Out.buf,workspaceBuf,workspaceBytes);
    v2Bias->apply(cudaHandles,batchSize,v2Out.buf);
    v3Mul->apply(cudaHandles,scratch,batchSize,v2Out.buf,valueBuf,workspaceBuf,workspaceBytes);
    v3Bias->apply(cudaHandles,batchSize,valueBuf);

    sv3Mul->apply(cudaHandles,scratch,batchSize,v2Out.buf,scoreValueBuf,workspaceBuf,workspaceBytes);
    sv3Bias->apply(cudaHandles,batchSize,scoreValueBuf);

    #ifdef DEBUG_INTERMEDIATE_VALUES
    debugPrint4D(string("v1"), v1Out.buf, batchSize, v1Channels, xSize, ySize, usingNHWC, usingFP16);
    debugPrint2D(string("v1 pooled"), v1Mean.buf, batchSize, v1Channels, usingFP16);
    debugPrint2D(string("v2"), v2Out.buf, batchSize, v1Channels, usingFP16);
    #endif

    if(!usingFP16) {
      vOwnershipConv->apply(cudaHandles,batchSize,false,v1Out2.buf,ownershipBuf,workspaceBuf,workspaceBytes);
    }
    else {
      vOwnershipConv->apply(cudaHandles,batchSize,false,v1Out2.buf,ownershipScratch.buf,workspaceBuf,workspaceBytes);
      customCudaCopyFromHalf((const half*)ownershipScratch.buf,(float*)ownershipBuf,batchSize*ownershipChannels*xSize*ySize);
      CUDA_ERR("vOwnership copy",cudaPeekAtLastError());
    }

  }

};

//------------------------------------------------------------------------------

struct Model {
  string name;
  int version;
  int maxBatchSize;
  int xSize;
  int ySize;
  int numInputChannels;
  int numInputGlobalChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;
  bool usingFP16;
  bool inputsUsingNHWC;

  std::unique_ptr<Trunk> trunk;
  std::unique_ptr<PolicyHead> policyHead;
  std::unique_ptr<ValueHead> valueHead;
  std::unique_ptr<CudnnManager> manager;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(
    CudaHandles* cudaHandles,
    const ModelDesc* desc,
    int maxBatchSz,
    int nnXLen,
    int nnYLen,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) {
    name = desc->name;
    version = desc->version;
    maxBatchSize = maxBatchSz;

    xSize = nnXLen;
    ySize = nnYLen;
    if(nnXLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnXLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnXLen, NNPos::MAX_BOARD_LEN
      ));
    if(nnYLen > NNPos::MAX_BOARD_LEN)
      throw StringError(Global::strprintf("nnYLen (%d) is greater than NNPos::MAX_BOARD_LEN (%d)",
        nnYLen, NNPos::MAX_BOARD_LEN
      ));

    numInputChannels = desc->numInputChannels;
    numInputGlobalChannels = desc->numInputGlobalChannels;
    numValueChannels = desc->numValueChannels;
    numScoreValueChannels = desc->numScoreValueChannels;
    numOwnershipChannels = desc->numOwnershipChannels;
    usingFP16 = useFP16;
    inputsUsingNHWC = inputsUseNHWC;

    int numFeatures = NNModelVersion::getNumSpatialFeatures(version);
    if(numInputChannels != numFeatures)
      throw StringError(Global::strprintf("Neural net numInputChannels (%d) was not the expected number based on version (%d)",
        numInputChannels, numFeatures
      ));
    int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
    if(numInputGlobalChannels != numGlobalFeatures)
      throw StringError(Global::strprintf("Neural net numInputGlobalChannels (%d) was not the expected number based on version (%d)",
        numInputGlobalChannels, numGlobalFeatures
      ));

    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputGlobalChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numScoreValueChannels);
    CudaUtils::checkBufferSize(maxBatchSize,nnXLen,nnYLen,numOwnershipChannels);

    manager = std::make_unique<CudnnManager>(name, maxBatchSize, xSize, ySize);
    trunk = std::make_unique<Trunk>(cudaHandles,manager.get(),&desc->trunk,xSize,ySize,inputsUseNHWC,useFP16,useNHWC);
    policyHead = std::make_unique<PolicyHead>(cudaHandles,manager.get(),&desc->policyHead,xSize,ySize,useFP16,useNHWC);
    valueHead = std::make_unique<ValueHead>(cudaHandles,manager.get(),&desc->valueHead,xSize,ySize,useFP16,useNHWC);
  }

  ~Model()
  {
  }

  size_t requiredWorkspaceBytes(
    CudaHandles* cudaHandles,
    int batchSize
  ) const {
    size_t bytes = 0;
    size_t b;

    b = trunk->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = policyHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);
    b = valueHead->requiredWorkspaceBytes(cudaHandles,batchSize);
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    ScratchBuffers* scratch,
    int batchSize,
    bool requireExactNNLen,

    void* inputBuf,
    void* inputGlobalBuf,

    float* policyBuf,

    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,

    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    SizedBuf<void*> mask(scratch->allocator, scratch->getBufSizeXY(1));
    SizedBuf<void*> maskFloat(scratch->allocator, scratch->getBufSizeXYFloat(1));
    SizedBuf<void*> maskSum(scratch->allocator, scratch->getBufSizeFloat(1));

    void* maskBuf = mask.buf;
    float* maskFloatBuf = (float*)maskFloat.buf;
    float* maskSumBuf = (float*)maskSum.buf;

    if(!usingFP16) {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const float*)inputBuf, (float*)maskBuf, batchSize, xSize*ySize, numInputChannels);
      else
        customCudaChannel0ExtractNCHW((const float*)inputBuf, (float*)maskBuf, batchSize, numInputChannels, xSize*ySize);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }
    else {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const half*)inputBuf, (half*)maskBuf, batchSize, xSize*ySize, numInputChannels);
      else
        customCudaChannel0ExtractNCHW((const half*)inputBuf, (half*)maskBuf, batchSize, numInputChannels, xSize*ySize);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
    }

    fillMaskFloatBufAndMaskSumBuf(maskBuf,maskFloatBuf,maskSumBuf,usingFP16,batchSize,xSize,ySize);

    //Don't do any masking if we know the board is exactly the desired size
    if(requireExactNNLen) {
      //Set to NULL to signal downstream that this buf doesn't need to be used
      maskBuf = NULL;
      maskFloatBuf = NULL;
      //The global pooling structures need this no matter what, for normalizing based on this and its sqrt.
      //maskSumBuf = NULL;
    }

    SizedBuf<void*> trunkBuf(scratch->allocator, scratch->getBufSizeXY(trunk->trunkNumChannels));

    trunk->apply(
      cudaHandles,
      scratch,
      batchSize,
      inputBuf,
      inputGlobalBuf,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      workspaceBuf,
      workspaceBytes
    );
    policyHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskFloatBuf,
      maskSumBuf,
      trunkBuf.buf,
      policyBuf,
      workspaceBuf,
      workspaceBytes
    );
    valueHead->apply(
      cudaHandles,
      scratch,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkBuf.buf,
      valueBuf,
      scoreValueBuf,
      ownershipBuf,
      workspaceBuf,
      workspaceBytes
    );
  }

};


//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc,expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file,expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

string NeuralNet::getModelName(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.name;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

Rules NeuralNet::getSupportedRules(const LoadedModel* loadedModel, const Rules& desiredRules, bool& supported) {
  return loadedModel->modelDesc.getSupportedRules(desiredRules, supported);
}

//------------------------------------------------------------------------------

struct Buffers {
  //All of these are device pointers

  float* inputBufFloat;
  void* inputBuf;
  float* inputGlobalBufFloat;
  void* inputGlobalBuf;
  size_t inputBufBytesFloat;
  size_t inputBufBytes;
  size_t inputGlobalBufBytesFloat;
  size_t inputGlobalBufBytes;

  float* policyBuf;
  size_t policyBufBytes;

  float* valueBuf;
  size_t valueBufBytes;
  float* scoreValueBuf;
  size_t scoreValueBufBytes;
  void* ownershipBuf;
  size_t ownershipBufBytes;

  void* workspaceBuf;
  size_t workspaceBytes;

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(CudaHandles* cudaHandles, const Model& m, bool useFP16) {
    size_t batchXYFloatBytes = (size_t)m.maxBatchSize * m.xSize * m.ySize * sizeof(float);
    size_t batchFloatBytes = (size_t)m.maxBatchSize * sizeof(float);

    size_t batchXYBytes = (size_t)m.maxBatchSize * m.xSize * m.ySize * (useFP16 ? sizeof(half) : sizeof(float));
    size_t batchBytes = (size_t)m.maxBatchSize * (useFP16 ? sizeof(half) : sizeof(float));

    inputBufBytesFloat = m.numInputChannels * batchXYFloatBytes;
    inputBufBytes = m.numInputChannels * batchXYBytes;
    inputGlobalBufBytesFloat = m.numInputGlobalChannels * batchFloatBytes;
    inputGlobalBufBytes = m.numInputGlobalChannels * batchBytes;

    CUDA_ERR("Buffers",cudaMalloc(&inputBufFloat, inputBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputBuf, inputBufBytes));
    CUDA_ERR("Buffers",cudaMalloc(&inputGlobalBufFloat, inputGlobalBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputGlobalBuf, inputGlobalBufBytes));

    policyBufBytes = m.policyHead->p2Channels * (batchXYFloatBytes + batchFloatBytes);
    CUDA_ERR("Buffers",cudaMalloc(&policyBuf, policyBufBytes));
    assert(m.policyHead->p2Channels == 1);

    valueBufBytes = m.valueHead->valueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&valueBuf, valueBufBytes));

    scoreValueBufBytes = m.valueHead->scoreValueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&scoreValueBuf, scoreValueBufBytes));

    //This buf is used for both an intermdiate fp16 result in fp16 mode, and ALSO the final fp32 output, so always must be fp32-sized
    ownershipBufBytes = m.valueHead->ownershipChannels * batchXYFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&ownershipBuf, ownershipBufBytes));

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
    cudaFree(inputBufFloat);
    cudaFree(inputBuf);
    cudaFree(inputGlobalBufFloat);
    cudaFree(inputGlobalBuf);

    cudaFree(policyBuf);

    cudaFree(valueBuf);
    cudaFree(scoreValueBuf);
    cudaFree(ownershipBuf);

    cudaFree(workspaceBuf);
  }

};

//------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  enabled_t useNHWCMode;
};

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  context->useFP16Mode = useFP16Mode;
  context->useNHWCMode = useNHWCMode;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

struct ComputeHandle {
  std::unique_ptr<CudaHandles> cudaHandles;
  std::unique_ptr<Model> model;
  std::unique_ptr<Buffers> buffers;
  std::unique_ptr<ScratchBuffers> scratch;
  bool usingFP16;
  int nnXLen;
  int nnYLen;
  bool requireExactNNLen;
  bool inputsUseNHWC;
  int policySize;

  ComputeHandle(
    const LoadedModel* loadedModel,
    int majorComputeCapability,
    int minorComputeCapability,
    int maxBatchSize,
    int xLen,
    int yLen,
    bool rExactNNLen,
    bool inputsNHWC,
    bool useFP16,
    bool useNHWC
  ) {
    cudaHandles = std::make_unique<CudaHandles>(majorComputeCapability,minorComputeCapability);
    model = std::make_unique<Model>(
      cudaHandles.get(), &(loadedModel->modelDesc), maxBatchSize,
      xLen, yLen, inputsNHWC, useFP16, useNHWC
    );
    buffers = std::make_unique<Buffers>(cudaHandles.get(), *model, useFP16);
    scratch = std::make_unique<ScratchBuffers>(maxBatchSize, xLen, yLen, useFP16);
    usingFP16 = useFP16;
    nnXLen = xLen;
    nnYLen = yLen;
    requireExactNNLen = rExactNNLen;
    inputsUseNHWC = inputsNHWC;
    policySize = NNPos::getPolicySize(nnXLen, nnYLen);

    //Synchronize after creating buffers and copying all the weights, just in case
    CUDA_ERR("ComputeHandle", cudaDeviceSynchronize());
  }
  ~ComputeHandle() {
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  //Use whatever CUDA believes GPU 0 to be.
  if(gpuIdxForThisThread == -1)
    gpuIdxForThisThread = 0;

  CUDA_ERR("createComputeHandle",cudaSetDevice(gpuIdxForThisThread));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,gpuIdxForThisThread);

  bool useFP16 = false;
  bool useNHWC = false;
  //Old GPUs - use FP32 and explicitly fail if FP16 enabled
  if(prop.major < 5 || (prop.major == 5 && prop.minor < 3)) {
    if(context->useFP16Mode == enabled_t::True)
      throw StringError("Cuda device versions below 5.3 do not support useFP16=true");
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //In theory these GPUs support FP16, so allow if the user wants.
  else if(prop.major < 6) {
    if(context->useFP16Mode == enabled_t::True)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //On Pascal architecture, default to using FP16 operations
  //Actually, just use FP32 - there's a risk that on certain cards this might just be a lot worse.
  //A user manually fine-tuning for performance can just enable it themselves if they know how.
  else if(prop.major < 7) {
    if(context->useFP16Mode == enabled_t::True)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True)
      useNHWC = true;
  }
  //On Volta and higher, use FP16 and NHWC together because we have tensor cores.
  else {
    if(context->useFP16Mode == enabled_t::True || context->useFP16Mode == enabled_t::Auto)
      useFP16 = true;
    if(context->useNHWCMode == enabled_t::True || (context->useNHWCMode == enabled_t::Auto && useFP16))
      useNHWC = true;
  }
  int nnXLen = context->nnXLen;
  int nnYLen = context->nnYLen;

  if(logger != NULL) {
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Found GPU " + string(prop.name)
      + " memory " + Global::uint64ToString(prop.totalGlobalMem)
      + " compute capability major " + Global::intToString(prop.major)
      + " minor " + Global::intToString(prop.minor)
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.version) +
      " useFP16 = " + Global::boolToString(useFP16) +
      " useNHWC = " + Global::boolToString(useNHWC)
    );
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name
    );
  }

  ComputeHandle* gpuHandle = new ComputeHandle(
    loadedModel,prop.major,prop.minor,maxBatchSize,nnXLen,nnYLen,requireExactNNLen,inputsUseNHWC,useFP16,useNHWC
  );
  return gpuHandle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

//------------------------------------------------------------------------------

void NeuralNet::printDevices() {
  int numDevices = 0;
  cudaGetDeviceCount(&numDevices);
  for(int i = 0; i<numDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout << "Found CUDA device " << i << ": " << prop.name << endl;
  }
}


//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t userInputBufferBytes;
  size_t userInputGlobalBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  float* userInputBuffer; //Host pointer
  float* userInputGlobalBuffer; //Host pointer

  float* policyResults; //Host pointer
  float* valueResults; //Host pointer
  float* scoreValueResults; //Host pointer
  float* ownershipResults; //Host pointer

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = nnXLen;
    int ySize = nnYLen;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * xSize * ySize;
    singleInputBytes = (size_t)m.numInputChannels * xSize * ySize * sizeof(float);
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputGlobalBytes = (size_t)m.numInputGlobalChannels * sizeof(float);
    singlePolicyResultElts = (size_t)(1 + xSize * ySize);
    singlePolicyResultBytes = (size_t)(1 + xSize * ySize) * sizeof(float);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleValueResultBytes = (size_t)m.numValueChannels * sizeof(float);
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleScoreValueResultBytes = (size_t)m.numScoreValueChannels * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * xSize * ySize;
    singleOwnershipResultBytes = (size_t)m.numOwnershipChannels * xSize * ySize * sizeof(float);

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    userInputBufferBytes = (size_t)m.numInputChannels * maxBatchSize * xSize * ySize * sizeof(float);
    userInputGlobalBufferBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    policyResultBufferBytes = (size_t)maxBatchSize * (1 + xSize * ySize) * sizeof(float);
    valueResultBufferBytes = (size_t)maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = (size_t)maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = (size_t)maxBatchSize * xSize * ySize * m.numOwnershipChannels * sizeof(float);

    userInputBuffer = new float[(size_t)m.numInputChannels * maxBatchSize * xSize * ySize];
    userInputGlobalBuffer = new float[(size_t)m.numInputGlobalChannels * maxBatchSize];

    policyResults = new float[(size_t)maxBatchSize * (1 + xSize * ySize)];
    valueResults = new float[(size_t)maxBatchSize * m.numValueChannels];

    scoreValueResults = new float[(size_t)maxBatchSize * m.numScoreValueChannels];
    ownershipResults = new float[(size_t)maxBatchSize * xSize * ySize * m.numOwnershipChannels];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;

};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel,maxBatchSize,nnXLen,nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//---------------------------------------------------------------------------------------


void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int version = gpuHandle->model->version;

  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
  assert(numSpatialFeatures == gpuHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  for(int nIdx = 0; nIdx<batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobal;
    const float* rowSpatial = inputBufs[nIdx]->rowSpatial;
    std::copy(rowGlobal,rowGlobal+numGlobalFeatures,rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);
  }

  Buffers* buffers = gpuHandle->buffers.get();
  ScratchBuffers* scratch = gpuHandle->scratch.get();

  if(!gpuHandle->usingFP16) {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singlePolicyResultElts == gpuHandle->policySize);
    assert(inputBuffers->singlePolicyResultBytes == gpuHandle->policySize * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice));
    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice));
  }
  else {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytesFloat);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytesFloat);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes*2);
    assert(inputBuffers->userInputGlobalBufferBytes == buffers->inputGlobalBufBytes*2);
    assert(inputBuffers->singleInputBytes == inputBuffers->singleInputElts*4);
    assert(inputBuffers->singleInputGlobalBytes == inputBuffers->singleInputGlobalElts*4);
    assert(inputBuffers->singlePolicyResultElts == gpuHandle->policySize);
    assert(inputBuffers->singlePolicyResultBytes == gpuHandle->policySize * sizeof(float));
    assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
    assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
    assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
    assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice));
    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice));

    customCudaCopyToHalf((const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,inputBuffers->singleInputElts*batchSize);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    customCudaCopyToHalf((const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,inputBuffers->singleInputGlobalElts*batchSize);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
  }

  gpuHandle->model->apply(
    gpuHandle->cudaHandles.get(),
    scratch,
    batchSize,
    gpuHandle->requireExactNNLen,

    buffers->inputBuf,
    buffers->inputGlobalBuf,

    buffers->policyBuf,

    buffers->valueBuf,
    buffers->scoreValueBuf,
    buffers->ownershipBuf,

    buffers->workspaceBuf,
    buffers->workspaceBytes
  );

  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes*batchSize, cudaMemcpyDeviceToHost));
  CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes*batchSize, cudaMemcpyDeviceToHost));

  assert(outputs.size() == batchSize);

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    const float* policySrcBuf = inputBuffers->policyResults + row * gpuHandle->policySize;
    float* policyProbs = output->policyProbs;

    //These are not actually correct, the client does the postprocessing to turn them into
    //policy probabilities and white game outcome probabilities
    //Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    policyProbs[gpuHandle->policySize-1] = policySrcBuf[gpuHandle->policySize-1];

    int numValueChannels = gpuHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * nnXLen * nnYLen;
      assert(gpuHandle->model->numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(version >= 9) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    }
    else if(version >= 8) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(version >= 4) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(version >= 3) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else {
      ASSERT_UNREACHABLE;
    }
  }

}

//TESTING ----------------------------------------------------------------------------------


bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  int xSize = nnXLen;
  int ySize = nnYLen;

  size_t numInputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->inChannels;
  size_t numOutputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateConv: unexpected input buffer size");

  void* deviceInput;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,xSize,ySize);
  ConvLayer* convLayer = new ConvLayer(cudaHandles,manager,desc,useFP16,useNHWC);

  size_t workspaceBytes =
    convLayer->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));


  bool accumulate = false;
  convLayer->apply(
    cudaHandles,
    desiredBatchSize,
    accumulate,
    deviceInput,
    deviceOutput,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  cudaFree(deviceWorkspace);

  delete convLayer;
  delete manager;
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  int xSize = nnXLen;
  int ySize = nnYLen;

  size_t numInputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * xSize * ySize;
  size_t numOutputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->numChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateBatchNorm: unexpected mask buffer size");

  ActivationLayerDesc actDesc;
  actDesc.activation = ACTIVATION_IDENTITY;

  void* deviceInput;
  void* deviceMask;
  void* deviceOutput;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  BatchNormLayer* batchNormLayer = new BatchNormLayer(cudaHandles,desc,&actDesc,xSize,ySize,useFP16,useNHWC);

  batchNormLayer->apply(
    cudaHandles,
    desiredBatchSize,
    deviceInput,
    deviceMask,
    deviceOutput
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  delete batchNormLayer;

  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  int xSize = nnXLen;
  int ySize = nnYLen;

  size_t numInputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * xSize * ySize;
  size_t numOutputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->finalConv.outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, xSize, ySize, useFP16);

  void* deviceInput;
  void* deviceMask;
  void* deviceScratch;
  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,xSize,ySize);
  ResidualBlock* residualBlock = new ResidualBlock(cudaHandles,manager,desc,xSize,ySize,useFP16,useNHWC);

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(cudaHandles,desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;
  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int desiredBatchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  cudaDeviceSynchronize();
  CudaHandles* cudaHandles = CudaHandles::cudaHandlesTesting();

  int xSize = nnXLen;
  int ySize = nnYLen;

  size_t numInputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)desiredBatchSize * xSize * ySize;
  size_t numMaskSumFloats = (size_t)desiredBatchSize;
  size_t numOutputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->finalConv.outChannels;

  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected mask buffer size");

  ScratchBuffers* scratch = new ScratchBuffers(desiredBatchSize, xSize, ySize, useFP16);

  void* deviceInput;
  void* deviceMask;
  float* deviceMaskFloatOrig;
  float* deviceMaskFloat;
  float* deviceMaskSum;
  void* deviceScratch;

  CudaUtils::mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  CudaUtils::mallocAndCopyToDevice("deviceMask", maskBuffer.data(), numMaskFloats, deviceMask, useFP16);
  CUDA_ERR("deviceMaskFloat",cudaMalloc(&deviceMaskFloat, numMaskFloats * sizeof(float)));
  CUDA_ERR("deviceMaskSum",cudaMalloc(&deviceMaskSum, numMaskSumFloats * sizeof(float)));
  deviceMaskFloatOrig = deviceMaskFloat;
  CudaUtils::mallocOnDevice("deviceScratch", numInputFloats, deviceScratch, useFP16);

  fillMaskFloatBufAndMaskSumBuf(deviceMask, deviceMaskFloat, deviceMaskSum, useFP16, desiredBatchSize, xSize, ySize);

  int maxBatchSize = desiredBatchSize;

  CudnnManager* manager = new CudnnManager("manager",maxBatchSize,xSize,ySize);
  GlobalPoolingResidualBlock* residualBlock = new GlobalPoolingResidualBlock(
    cudaHandles,manager,desc,xSize,ySize,useFP16,useNHWC
  );

  size_t workspaceBytes =
    residualBlock->requiredWorkspaceBytes(
      cudaHandles,desiredBatchSize
    );

  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));

  residualBlock->apply(
    cudaHandles,
    scratch,
    desiredBatchSize,
    deviceInput,
    deviceScratch,
    deviceMask,
    deviceMaskSum,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  CudaUtils::expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceInput, useFP16);

  cudaFree(deviceWorkspace);

  delete residualBlock;
  delete manager;

  cudaFree(deviceInput);
  cudaFree(deviceMask);
  cudaFree(deviceMaskFloatOrig);
  cudaFree(deviceMaskSum);
  cudaFree(deviceScratch);
  delete scratch;
  delete cudaHandles;

  return true;
}


#endif  // USE_CUDA_BACKEND
