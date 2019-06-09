#ifdef USE_CUDA_BACKEND
#include "../neuralnet/cudaerrorcheck.h"
#include "../neuralnet/cudaincludes.h"

#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/desc.h"

using namespace std;

void NeuralNet::globalInitialize() {
  //Empty for cudnn backend
}

void NeuralNet::globalCleanup() {
  cudaDeviceReset();
}

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

static void mallocOnDevice(const string& name, int numWeights, void*& deviceBuf, bool useFP16) {
  if(useFP16) {
    size_t halfBytes = numWeights * sizeof(half);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, halfBytes));
  }
  else {
    size_t floatBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, floatBytes));
  }
}

static void mallocAndCopyToDevice(const string& name, const vector<float>& weights, void*& deviceBuf, bool useFP16) {
  size_t numWeights = weights.size();
  if(useFP16) {
    size_t halfBytes = numWeights * sizeof(half);
    size_t floatBytes = numWeights * sizeof(float);
    float* buf;
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, halfBytes));
    CUDA_ERR(name.c_str(),cudaMalloc(&buf, floatBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(buf, weights.data(), floatBytes, cudaMemcpyHostToDevice));
    customCudaCopyToHalf(buf,(half*)deviceBuf,numWeights);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    CUDA_ERR(name.c_str(),cudaDeviceSynchronize());
    cudaFree(buf);
  }
  else {
    size_t floatBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, floatBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(deviceBuf, weights.data(), floatBytes, cudaMemcpyHostToDevice));
  }
}

static void mallocAndCopyToDevice(const string& name, const float* weights, int numWeights, void*& deviceBuf, bool useFP16) {
  if(useFP16) {
    size_t halfBytes = numWeights * sizeof(half);
    size_t floatBytes = numWeights * sizeof(float);
    float* buf;
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, halfBytes));
    CUDA_ERR(name.c_str(),cudaMalloc(&buf, floatBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(buf, weights, floatBytes, cudaMemcpyHostToDevice));
    customCudaCopyToHalf(buf,(half*)deviceBuf,numWeights);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    CUDA_ERR(name.c_str(),cudaDeviceSynchronize());
    cudaFree(buf);
  }
  else {
    size_t floatBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, floatBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(deviceBuf, weights, floatBytes, cudaMemcpyHostToDevice));
  }
}

//Only use in testing, allocates an intermediate buffer in the case of FP16 which will be very slow.
static void expensiveCopyFromDevice(const string& name, float* weights, int numWeights, void* deviceBuf, bool useFP16) {
  if(useFP16) {
    size_t floatBytes = numWeights * sizeof(float);
    float* buf;
    CUDA_ERR(name.c_str(),cudaMalloc(&buf, floatBytes));
    customCudaCopyFromHalf((const half*)deviceBuf,(float*)buf,numWeights);
    CUDA_ERR(name.c_str(),cudaMemcpy(weights, buf, floatBytes, cudaMemcpyDeviceToHost));
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    CUDA_ERR(name.c_str(),cudaDeviceSynchronize());
    cudaFree(buf);
  }
  else {
    size_t floatBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMemcpy(weights, deviceBuf, floatBytes, cudaMemcpyDeviceToHost));
  }
}


static void checkBufferSize(int batchSize, int xSize, int ySize, int channels) {
  if((int64_t)batchSize * xSize * ySize * channels >= (int64_t)1 << 31)
    throw StringError("Batch size too large, resulting GPU buffers might exceed 2^31 entries which is not currently supported");
}

//---------------------------------------------------------------------------------

struct ConvLayer {
  string name;
  cudnnFilterDescriptor_t filterDescriptor;
  cudnnConvolutionDescriptor_t convolutionDescriptor;
  cudnnConvolutionFwdAlgo_t* convolutionAlgorithms; //array of one for each batch size
  void* filterBuf;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(
    CudaHandles* cudaHandles,
    const ConvLayerDesc* desc,
    int maxBatchSize,
    const cudnnTensorDescriptor_t* inputDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* outputDescriptors, //array of one for each batch size
    bool useFP16,
    bool useNHWC
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

    bool filterNHWC = useNHWC && dilationY == 1 && dilationX == 1;

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
    if(useFP16)
      CUDNN_ERR(name.c_str(),cudnnSetConvolutionMathType(convolutionDescriptor, CUDNN_TENSOR_OP_MATH));

    convolutionAlgorithms = new cudnnConvolutionFwdAlgo_t[maxBatchSize];
    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      if(useFP16 && dilationX <= 1 && dilationY <= 1) {
        convolutionAlgorithms[batchSize-1] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      }
      else {
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
      mallocAndCopyToDevice(name,weightsTransposed,filterBuf,useFP16);
      cudaDeviceSynchronize();
    }
    else
      mallocAndCopyToDevice(name,desc->weights,filterBuf,useFP16);
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
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
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

struct BatchNormLayer {
  string name;
  int numChannels;
  float epsilon;
  int xSize;
  int ySize;

  cudnnTensorDescriptor_t bufDescriptor;
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
    int xS,
    int yS,
    bool useFP16,
    bool useNHWC
  ) {
    (void)cudaHandles;

    name = desc->name;
    numChannels = desc->numChannels;
    epsilon = desc->epsilon;
    xSize = xS;
    ySize = yS;
    usingFP16 = useFP16;
    usingNHWC = useNHWC;

    CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&bufDescriptor));
    CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
      bufDescriptor,
      CUDNN_TENSOR_NCHW, //Always NCHW since otherwise cudnn is sad
      (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
      1,
      numChannels,
      1,
      1
    ));

    assert(desc->mean.size() == numChannels);
    mallocAndCopyToDevice(name,desc->mean,meanBuf,useFP16);

    assert(desc->variance.size() == numChannels);
    mallocAndCopyToDevice(name,desc->variance,varianceBuf,useFP16);

    assert(desc->scale.size() == numChannels);
    mallocAndCopyToDevice(name,desc->scale,scaleBuf,useFP16);

    assert(desc->bias.size() == numChannels);
    mallocAndCopyToDevice(name,desc->bias,biasBuf,useFP16);

    vector<float> mergedScale(numChannels);
    vector<float> mergedBias(numChannels);
    for(int i = 0; i<numChannels; i++) {
      mergedScale[i] = desc->scale[i] / sqrt(desc->variance[i] + epsilon);
      mergedBias[i] = desc->bias[i] - mergedScale[i] * desc->mean[i];
    }
    mallocAndCopyToDevice(name,mergedScale,mergedScaleBuf,useFP16);
    mallocAndCopyToDevice(name,mergedBias,mergedBiasBuf,useFP16);
  }
  ~BatchNormLayer() {
    cudaFree(meanBuf);
    cudaFree(varianceBuf);
    cudaFree(scaleBuf);
    cudaFree(biasBuf);
    cudaFree(mergedScaleBuf);
    cudaFree(mergedBiasBuf);
    cudnnDestroyTensorDescriptor(bufDescriptor);
  }

  void apply(
    CudaHandles* cudaHandles,
    int batchSize,
    bool applyRelu,
    void* inputBuf,
    const void* maskBuf, //ok to be null
    void* outputBuf
  ) const {
    (void)cudaHandles;
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
    void* inputBuf,
    void* outputBuf
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
    mallocAndCopyToDevice(name,desc->weights,matBuf,useFP16);
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
    void* inputBuf,
    void* outputBuf,
    const void* zeroBuf,
    const void* oneBuf,
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
      const half* alpha = (const half*)oneBuf;
      const half* beta = (const half*)zeroBuf;
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

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(
    CudaHandles* cudaHandles,
    const MatBiasLayerDesc* desc,
    bool useFP16
  ) {
    (void)cudaHandles;
    name = desc->name;
    numChannels = desc->numChannels;
    usingFP16 = useFP16;

    assert(desc->weights.size() == numChannels);
    mallocAndCopyToDevice(name,desc->weights,biasBuf,useFP16);
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
      customCudaAddCBiasInplaceNC((float*)matBuf,(const float*)biasBuf,batchSize,numChannels);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaAddCBiasInplaceNC((half*)matBuf,(const half*)biasBuf,batchSize,numChannels);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
  }

};

//---------------------------------------------------------------------------------

struct ResidualBlock {
  string name;
  BatchNormLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  BatchNormLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  int xSize;
  int ySize;
  int regularChannels;
  bool usingFP16;

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
    const cudnnTensorDescriptor_t* midInDescriptors, //array of one for each batch size
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN,xS,yS,useFP16,useNHWC),
     preActivation(cudaHandles,&desc->preActivation),
     regularConv(cudaHandles,&desc->regularConv,maxBatchSize,trunkDescriptors,midInDescriptors,useFP16,useNHWC),
     midBN(cudaHandles,&desc->midBN,xS,yS,useFP16,useNHWC),
     midActivation(cudaHandles,&desc->midActivation),
     finalConv(cudaHandles,&desc->finalConv,maxBatchSize,midInDescriptors,trunkDescriptors,useFP16,useNHWC),
     xSize(xS),
     ySize(yS),
     regularChannels(desc->regularConv.outChannels),
     usingFP16(useFP16)
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
    void* trunkBuf,
    void* trunkScratchBuf,
    void* midInBuf,
    void* midScratchBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    bool applyBNRelu = true;
    preBN.apply(cudaHandles,batchSize,applyBNRelu,trunkBuf,maskBuf,trunkScratchBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,midInDescriptor,batchSize,false,trunkScratchBuf,midInBuf,workspaceBuf,workspaceBytes);
    midBN.apply(cudaHandles,batchSize,applyBNRelu,midInBuf,maskBuf,midScratchBuf);
    finalConv.apply(cudaHandles,midInDescriptor,trunkDescriptor,batchSize,true,midScratchBuf,trunkBuf,workspaceBuf,workspaceBytes);
  }

};


//-----------------------------------------------------------------------------

struct DilatedResidualBlock {
  string name;
  BatchNormLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  ConvLayer dilatedConv;
  BatchNormLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

  int xSize;
  int ySize;
  int regularChannels;
  int dilatedChannels;
  bool usingFP16;
  bool usingNHWC;

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
    const cudnnTensorDescriptor_t* midInDescriptors, //array of one for each batch size
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN,xS,yS,useFP16,useNHWC),
     preActivation(cudaHandles,&desc->preActivation),
     regularConv(cudaHandles,&desc->regularConv,maxBatchSize,trunkDescriptors,regularOutDescriptors,useFP16,useNHWC),
     dilatedConv(cudaHandles,&desc->dilatedConv,maxBatchSize,trunkDescriptors,dilatedOutDescriptors,useFP16,useNHWC),
     midBN(cudaHandles,&desc->midBN,xS,yS,useFP16,useNHWC),
     midActivation(cudaHandles,&desc->midActivation),
     finalConv(cudaHandles,&desc->finalConv,maxBatchSize,midInDescriptors,trunkDescriptors,useFP16,useNHWC),
     xSize(xS),
     ySize(yS),
     regularChannels(desc->regularConv.outChannels),
     dilatedChannels(desc->dilatedConv.outChannels),
     usingFP16(useFP16),
     usingNHWC(useNHWC)
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
    void* trunkBuf,
    void* trunkScratchBuf,
    void* regularOutBuf,
    void* dilatedOutBuf,
    void* midInBuf,
    void* midScratchBuf,
    void* maskBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    bool applyBNRelu = true;
    preBN.apply(cudaHandles,batchSize,applyBNRelu,trunkBuf,maskBuf,trunkScratchBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,regularOutDescriptor,batchSize,false,trunkScratchBuf,regularOutBuf,workspaceBuf,workspaceBytes);
    dilatedConv.apply(cudaHandles,trunkDescriptor,dilatedOutDescriptor,batchSize,false,trunkScratchBuf,dilatedOutBuf,workspaceBuf,workspaceBytes);
    if(!usingFP16) {
      if(!usingNHWC)
        customCudaChannelConcat(
          (const float*)regularOutBuf,(const float*)dilatedOutBuf,(float*)midInBuf,
          xSize*ySize*regularChannels,
          xSize*ySize*dilatedChannels,
          batchSize
        );
      else
        customCudaChannelConcat(
          (const float*)regularOutBuf,(const float*)dilatedOutBuf,(float*)midInBuf,
          regularChannels,
          dilatedChannels,
          batchSize*xSize*ySize
        );
    }
    else {
      if(!usingNHWC)
        customCudaChannelConcat(
          (const half*)regularOutBuf,(const half*)dilatedOutBuf,(half*)midInBuf,
          xSize*ySize*regularChannels,
          xSize*ySize*dilatedChannels,
          batchSize
        );
      else
        customCudaChannelConcat(
          (const half*)regularOutBuf,(const half*)dilatedOutBuf,(half*)midInBuf,
          regularChannels,
          dilatedChannels,
          batchSize*xSize*ySize
        );
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    midBN.apply(cudaHandles,batchSize,applyBNRelu,midInBuf,maskBuf,midScratchBuf);
    finalConv.apply(cudaHandles,midInDescriptor,trunkDescriptor,batchSize,true,midScratchBuf,trunkBuf,workspaceBuf,workspaceBytes);
  }

};



//----------------------------------------------------------------------------


struct GlobalPoolingResidualBlock {
  string name;
  BatchNormLayer preBN;
  ActivationLayer preActivation;
  ConvLayer regularConv;
  ConvLayer gpoolConv;
  BatchNormLayer gpoolBN;
  ActivationLayer gpoolActivation;
  MatMulLayer gpoolToBiasMul;
  BatchNormLayer midBN;
  ActivationLayer midActivation;
  ConvLayer finalConv;

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
    const GlobalPoolingResidualBlockDesc* desc,
    int maxBatchSize,
    int xS,
    int yS,
    const cudnnTensorDescriptor_t* trunkDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* regularOutDescriptors, //array of one for each batch size
    const cudnnTensorDescriptor_t* gpoolOutDescriptors, //array of one for each batch size
    bool useFP16,
    bool useNHWC
  ): name(desc->name),
     preBN(cudaHandles,&desc->preBN,xS,yS,useFP16,useNHWC),
     preActivation(cudaHandles,&desc->preActivation),
     regularConv(cudaHandles,&desc->regularConv,maxBatchSize,trunkDescriptors,regularOutDescriptors,useFP16,useNHWC),
     gpoolConv(cudaHandles,&desc->gpoolConv,maxBatchSize,trunkDescriptors,gpoolOutDescriptors,useFP16,useNHWC),
     gpoolBN(cudaHandles,&desc->gpoolBN,xS,yS,useFP16,useNHWC),
     gpoolActivation(cudaHandles,&desc->gpoolActivation),
     gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,useFP16),
     midBN(cudaHandles,&desc->midBN,xS,yS,useFP16,useNHWC),
     midActivation(cudaHandles,&desc->midActivation),
     finalConv(cudaHandles,&desc->finalConv,maxBatchSize,regularOutDescriptors,trunkDescriptors,useFP16,useNHWC),
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
    b = sizeof(float)*batchSize*gpoolChannels*xSize*ySize;
    bytes = std::max(bytes,b);
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    const cudnnTensorDescriptor_t& regularOutDescriptor,
    const cudnnTensorDescriptor_t& gpoolOutDescriptor,
    int batchSize,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* regularOutBuf,
    void* gpoolOutBuf,
    void* gpoolOutBuf2,
    void* gpoolConcatBuf,
    void* gpoolBiasBuf,
    void* regularScratchBuf,
    void* maskBuf,
    float* maskSumBuf,
    const void* zeroBuf,
    const void* oneBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    bool applyBNRelu = true;
    preBN.apply(cudaHandles,batchSize,applyBNRelu,trunkBuf,maskBuf,trunkScratchBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,regularOutDescriptor,batchSize,false,trunkScratchBuf,regularOutBuf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(cudaHandles,trunkDescriptor,gpoolOutDescriptor,batchSize,false,trunkScratchBuf,gpoolOutBuf,workspaceBuf,workspaceBytes);
    gpoolBN.apply(cudaHandles,batchSize,applyBNRelu,gpoolOutBuf,maskBuf,gpoolOutBuf2);

    if(!usingFP16) {
      const float meanScale = 1.0f / (xSize*ySize);
      if(!usingNHWC) {
        if(maskSumBuf != NULL)
          customCudaPoolRowsGPoolNCHW((const float*)gpoolOutBuf2,(float*)gpoolConcatBuf,batchSize,gpoolChannels,xSize*ySize,maskSumBuf);
        else
          customCudaPoolRowsSumAndMaxPositiveNCHW((const float*)gpoolOutBuf2,(float*)gpoolConcatBuf,batchSize,gpoolChannels,xSize*ySize,meanScale);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      else {
        if(maskSumBuf != NULL)
          customCudaPoolRowsGPoolNHWC((const float*)gpoolOutBuf2,(float*)gpoolConcatBuf,batchSize,xSize*ySize,gpoolChannels,maskSumBuf);
        else
          customCudaPoolRowsSumAndMaxPositiveNHWC((const float*)gpoolOutBuf2,(float*)gpoolConcatBuf,batchSize,xSize*ySize,gpoolChannels,meanScale);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
    }
    else {
      const float meanScale = 1.0f / (xSize*ySize);
      if(!usingNHWC) {
        if(maskSumBuf != NULL)
          customCudaPoolRowsGPoolNCHW((const half*)gpoolOutBuf2,(half*)gpoolConcatBuf,batchSize,gpoolChannels,xSize*ySize,maskSumBuf);
        else
          customCudaPoolRowsSumAndMaxPositiveNCHW((const half*)gpoolOutBuf2,(half*)gpoolConcatBuf,batchSize,gpoolChannels,xSize*ySize,meanScale);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      else {
        if(maskSumBuf != NULL)
          customCudaPoolRowsGPoolNHWC((const half*)gpoolOutBuf2,(half*)gpoolConcatBuf,batchSize,xSize*ySize,gpoolChannels,maskSumBuf);
        else
          customCudaPoolRowsSumAndMaxPositiveNHWC((const half*)gpoolOutBuf2,(half*)gpoolConcatBuf,batchSize,xSize*ySize,gpoolChannels,meanScale);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
    }

    gpoolToBiasMul.apply(cudaHandles,batchSize,gpoolConcatBuf,gpoolBiasBuf,zeroBuf,oneBuf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((float*)regularOutBuf,(const float*)gpoolBiasBuf,batchSize,regularChannels,xSize*ySize);
      else
        customCudaAddNCBiasInplaceNHWC((float*)regularOutBuf,(const float*)gpoolBiasBuf,batchSize,xSize*ySize,regularChannels);
    }
    else {
      if(!usingNHWC)
        customCudaAddNCBiasInplaceNCHW((half*)regularOutBuf,(const half*)gpoolBiasBuf,batchSize,regularChannels,xSize*ySize);
      else
        customCudaAddNCBiasInplaceNHWC((half*)regularOutBuf,(const half*)gpoolBiasBuf,batchSize,xSize*ySize,regularChannels);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    midBN.apply(cudaHandles,batchSize,applyBNRelu,regularOutBuf,maskBuf,regularScratchBuf);
    finalConv.apply(cudaHandles,regularOutDescriptor,trunkDescriptor,batchSize,true,regularScratchBuf,trunkBuf,workspaceBuf,workspaceBytes);
  }

};

//------------------------------------------------------------------------------

struct Trunk {
  string name;
  int version;
  int numBlocks;
  int trunkNumChannels;
  int midNumChannels;
  int regularNumChannels;
  int dilatedNumChannels;
  int gpoolNumChannels;

  int maxBatchSize;
  int xSize;
  int ySize;
  bool usingFP16;
  bool usingNHWC;

  cudnnTensorDescriptor_t* trunkDescriptors;
  cudnnTensorDescriptor_t* regularOutDescriptors;
  cudnnTensorDescriptor_t* gpoolOutDescriptors;
  cudnnTensorDescriptor_t* dilatedOutDescriptors;
  cudnnTensorDescriptor_t* midInDescriptors;

  ConvLayer* initialConv;
  MatMulLayer* initialMatMul;
  vector<pair<int,void*>> blocks;
  BatchNormLayer* trunkTipBN;
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
    const cudnnTensorDescriptor_t* inputDescriptors,
    bool useFP16,
    bool useNHWC
  ) {
    name = desc->name;
    version = desc->version;
    numBlocks = desc->numBlocks;
    trunkNumChannels = desc->trunkNumChannels;
    midNumChannels = desc->midNumChannels;
    regularNumChannels = desc->regularNumChannels;
    dilatedNumChannels = desc->dilatedNumChannels;
    gpoolNumChannels = desc->gpoolNumChannels;

    maxBatchSize = maxBatchSz;
    xSize = xS;
    ySize = yS;
    usingFP16 = useFP16;
    usingNHWC = useNHWC;

    checkBufferSize(maxBatchSize,xSize,ySize,trunkNumChannels);
    checkBufferSize(maxBatchSize,xSize,ySize,midNumChannels);
    checkBufferSize(maxBatchSize,xSize,ySize,regularNumChannels);
    checkBufferSize(maxBatchSize,xSize,ySize,dilatedNumChannels);
    checkBufferSize(maxBatchSize,xSize,ySize,gpoolNumChannels);

    trunkDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    regularOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    gpoolOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    dilatedOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    midInDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& trunkDescriptor = trunkDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& regularOutDescriptor = regularOutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& gpoolOutDescriptor = gpoolOutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& dilatedOutDescriptor = dilatedOutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& midInDescriptor = midInDescriptors[batchSize-1];

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&trunkDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        trunkDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        trunkNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&regularOutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        regularOutDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        regularNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&dilatedOutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        dilatedOutDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        dilatedNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&gpoolOutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        gpoolOutDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        gpoolNumChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&midInDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        midInDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        regularNumChannels+dilatedNumChannels,
        ySize,
        xSize
      ));
    }

    initialConv = new ConvLayer(cudaHandles,&desc->initialConv,maxBatchSize,inputDescriptors,trunkDescriptors,useFP16,useNHWC);
    initialMatMul = new MatMulLayer(cudaHandles,&desc->initialMatMul,useFP16);

    trunkTipBN = new BatchNormLayer(cudaHandles,&desc->trunkTipBN,xSize,ySize,useFP16,useNHWC);
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
          midInDescriptors,
          useFP16,
          useNHWC
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
          midInDescriptors,
          useFP16,
          useNHWC
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
          gpoolOutDescriptors,
          useFP16,
          useNHWC
        );
        blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND,(void*)block));
      }
      else {
        ASSERT_UNREACHABLE;
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
    delete initialMatMul;
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

    if(initialMatMul != NULL) {
      b = initialMatMul->requiredWorkspaceBytes(cudaHandles);
      bytes = std::max(bytes,b);
    }

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
        ASSERT_UNREACHABLE;
      }
    }
    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& inputDescriptor,
    int batchSize,
    void* inputBuf,
    void* inputGlobalBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* regularOutBuf,
    void* dilatedOutBuf,
    void* midInBuf,
    void* midScratchBuf,
    void* gpoolOutBuf,
    void* gpoolOutBuf2,
    void* gpoolConcatBuf,
    void* gpoolBiasBuf,
    void* regularScratchBuf,
    const void* zeroBuf,
    const void* oneBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {

    const cudnnTensorDescriptor_t& trunkDescriptor = trunkDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& regularOutDescriptor = regularOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& gpoolOutDescriptor = gpoolOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& dilatedOutDescriptor = dilatedOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& midInDescriptor = midInDescriptors[batchSize-1];

    //int trunkBufSize = batchSize * trunkNumChannels * xSize * ySize;

    //Feed the conv into trunkScratchBuf, not trunkBuf
    initialConv->apply(cudaHandles,inputDescriptor,trunkDescriptor,batchSize,false,inputBuf,trunkScratchBuf,workspaceBuf,workspaceBytes);

    if(initialMatMul != NULL) {
      //Feed the matmul into trunkBuf
      initialMatMul->apply(cudaHandles,batchSize,inputGlobalBuf,trunkBuf,zeroBuf,oneBuf,workspaceBuf,workspaceBytes);
      //Then accumulate it into trunkScratchBuf, broadcasting during the process
      if(!usingFP16) {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((float*)trunkScratchBuf,(const float*)trunkBuf,batchSize,trunkNumChannels,xSize*ySize);
        else
          customCudaAddNCBiasInplaceNHWC((float*)trunkScratchBuf,(const float*)trunkBuf,batchSize,xSize*ySize,trunkNumChannels);
      }
      else {
        if(!usingNHWC)
          customCudaAddNCBiasInplaceNCHW((half*)trunkScratchBuf,(const half*)trunkBuf,batchSize,trunkNumChannels,xSize*ySize);
        else
          customCudaAddNCBiasInplaceNHWC((half*)trunkScratchBuf,(const half*)trunkBuf,batchSize,xSize*ySize,trunkNumChannels);
      }
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    for(int i = 0; i<blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlock* block = (ResidualBlock*)blocks[i].second;
        block->apply(
          cudaHandles,
          trunkDescriptor,
          midInDescriptor,
          batchSize,
          trunkScratchBuf, //Flip trunkBuf and trunkScratchBuf so that the result gets accumulated in trunkScratchBuf
          trunkBuf,
          midInBuf,
          midScratchBuf,
          maskBuf,
          workspaceBuf,
          workspaceBytes
        );
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
          trunkScratchBuf, //Flip trunkBuf and trunkScratchBuf so that the result gets accumulated in trunkScratchBuf
          trunkBuf,
          regularOutBuf,
          dilatedOutBuf,
          midInBuf,
          midScratchBuf,
          maskBuf,
          workspaceBuf,
          workspaceBytes
        );
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = (GlobalPoolingResidualBlock*)blocks[i].second;
        block->apply(
          cudaHandles,
          trunkDescriptor,
          regularOutDescriptor,
          gpoolOutDescriptor,
          batchSize,
          trunkScratchBuf, //Flip trunkBuf and trunkScratchBuf so that the result gets accumulated in trunkScratchBuf
          trunkBuf,
          regularOutBuf,
          gpoolOutBuf,
          gpoolOutBuf2,
          gpoolConcatBuf,
          gpoolBiasBuf,
          regularScratchBuf,
          maskBuf,
          maskSumBuf,
          zeroBuf,
          oneBuf,
          workspaceBuf,
          workspaceBytes
        );
      }
      else {
        ASSERT_UNREACHABLE;
      }

    }

    //And now with the final BN port it from trunkScratchBuf to trunkBuf.
    bool applyBNRelu = true;
    trunkTipBN->apply(cudaHandles,batchSize,applyBNRelu,trunkScratchBuf,maskBuf,trunkBuf);
  }

};

//------------------------------------------------------------------------------

template <typename T>
static void applySymmetriesNCHW(
  const bool* symmetriesBuffer, bool inverse, int batchSize, int cSize, int xSize, int ySize,
  T* inputBuf, T* inputScratchBuf
) {
  if(!symmetriesBuffer[0] && !symmetriesBuffer[1] && !symmetriesBuffer[2])
    return;

  if(inverse) {
    if(symmetriesBuffer[2] && xSize == ySize)
      customCudaNCHWTranspose(inputBuf,inputScratchBuf,xSize,ySize,batchSize*cSize);
    else
      cudaMemcpyAsync(inputScratchBuf,inputBuf,sizeof(T)*batchSize*cSize*ySize*xSize,cudaMemcpyDeviceToDevice);
    CUDA_ERR("applySymmetriesNCHW",cudaPeekAtLastError());

    customCudaMirrorNCHW(inputScratchBuf, inputBuf, batchSize, cSize, ySize, xSize, symmetriesBuffer[0], symmetriesBuffer[1]);
    CUDA_ERR("applySymmetriesNCHW",cudaPeekAtLastError());
  }
  else {
    customCudaMirrorNCHW(inputBuf, inputScratchBuf, batchSize, cSize, ySize, xSize, symmetriesBuffer[0], symmetriesBuffer[1]);
    CUDA_ERR("applySymmetriesNCHW",cudaPeekAtLastError());
    if(symmetriesBuffer[2] && xSize == ySize)
      customCudaNCHWTranspose(inputScratchBuf,inputBuf,xSize,ySize,batchSize*cSize);
    else
      cudaMemcpyAsync(inputBuf,inputScratchBuf,sizeof(T)*batchSize*cSize*ySize*xSize,cudaMemcpyDeviceToDevice);
    CUDA_ERR("applySymmetriesNCHW",cudaPeekAtLastError());
  }
}

template <typename T>
static void applySymmetriesNHWC(
  const bool* symmetriesBuffer, bool inverse, int batchSize, int cSize, int xSize, int ySize,
  T* inputBuf, T* inputScratchBuf
) {
  if(!symmetriesBuffer[0] && !symmetriesBuffer[1] && !symmetriesBuffer[2])
    return;

  if(inverse) {
    if(symmetriesBuffer[2] && xSize == ySize)
      customCudaNHWCTranspose(inputBuf,inputScratchBuf,xSize,ySize,cSize,batchSize);
    else
      cudaMemcpyAsync(inputScratchBuf,inputBuf,sizeof(T)*batchSize*cSize*ySize*xSize,cudaMemcpyDeviceToDevice);
    CUDA_ERR("applySymmetriesNHWC",cudaPeekAtLastError());

    customCudaMirrorNHWC(inputScratchBuf, inputBuf, batchSize, ySize, xSize, cSize, symmetriesBuffer[0], symmetriesBuffer[1]);
    CUDA_ERR("applySymmetriesNHWC",cudaPeekAtLastError());
  }
  else {
    customCudaMirrorNHWC(inputBuf, inputScratchBuf, batchSize, ySize, xSize, cSize, symmetriesBuffer[0], symmetriesBuffer[1]);
    CUDA_ERR("applySymmetriesNHWC",cudaPeekAtLastError());
    if(symmetriesBuffer[2] && xSize == ySize)
      customCudaNHWCTranspose(inputScratchBuf,inputBuf,xSize,ySize,cSize,batchSize);
    else
      cudaMemcpyAsync(inputBuf,inputScratchBuf,sizeof(T)*batchSize*cSize*ySize*xSize,cudaMemcpyDeviceToDevice);
    CUDA_ERR("applySymmetriesNHWC",cudaPeekAtLastError());
  }
}


//------------------------------------------------------------------------------

struct PolicyHead {
  string name;
  int version;
  int maxBatchSize;
  int xSize;
  int ySize;
  int p1Channels;
  int g1Channels;
  int p2Channels;
  bool usingFP16;
  bool usingNHWC;

  cudnnTensorDescriptor_t* p1OutDescriptors;
  cudnnTensorDescriptor_t* g1OutDescriptors;
  cudnnTensorDescriptor_t* p2InDescriptors;
  cudnnTensorDescriptor_t* p2OutDescriptors;

  ConvLayer* p1Conv;
  ConvLayer* g1Conv;
  BatchNormLayer* g1BN;
  ActivationLayer* g1Activation;
  MatMulLayer* gpoolToBiasMul;
  BatchNormLayer* p1BN;
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
    const cudnnTensorDescriptor_t* trunkDescriptors,
    bool useFP16,
    bool useNHWC
  ) {
    name = desc->name;
    version = desc->version;
    maxBatchSize = maxBatchSz;
    xSize = xS;
    ySize = yS;
    p1Channels = desc->p1Conv.outChannels;
    g1Channels = desc->g1Conv.outChannels;
    p2Channels = desc->p2Conv.outChannels;
    usingFP16 = useFP16;
    usingNHWC = useNHWC;

    p1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    g1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    p2InDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    p2OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& p1OutDescriptor = p1OutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& g1OutDescriptor = g1OutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& p2InDescriptor = p2InDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& p2OutDescriptor = p2OutDescriptors[batchSize-1];

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&p1OutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        p1OutDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        desc->p1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&g1OutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        g1OutDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        desc->g1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&p2InDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        p2InDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->p1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&p2OutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        p2OutDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->p2Conv.outChannels,
        ySize,
        xSize
      ));

    }

    p1Conv = new ConvLayer(cudaHandles,&desc->p1Conv,maxBatchSize,trunkDescriptors,p1OutDescriptors,useFP16,useNHWC);
    g1Conv = new ConvLayer(cudaHandles,&desc->g1Conv,maxBatchSize,trunkDescriptors,g1OutDescriptors,useFP16,useNHWC);
    g1BN = new BatchNormLayer(cudaHandles,&desc->g1BN,xSize,ySize,useFP16,useNHWC);
    g1Activation = new ActivationLayer(cudaHandles,&desc->g1Activation);
    gpoolToBiasMul = new MatMulLayer(cudaHandles,&desc->gpoolToBiasMul,false);
    p1BN = new BatchNormLayer(cudaHandles,&desc->p1BN,xSize,ySize,false,useNHWC);
    p1Activation = new ActivationLayer(cudaHandles,&desc->p1Activation);
    p2Conv = new ConvLayer(cudaHandles,&desc->p2Conv,maxBatchSize,p2InDescriptors,p2OutDescriptors,false,useNHWC);
    gpoolToPassMul = new MatMulLayer(cudaHandles,&desc->gpoolToPassMul,false);
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
      cudnnDestroyTensorDescriptor(p2InDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(p2OutDescriptors[batchSize-1]);
    }

    delete[] p1OutDescriptors;
    delete[] g1OutDescriptors;
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
    b = sizeof(float)*batchSize*g1Channels*xSize*ySize;
    bytes = std::max(bytes,b);

    return bytes;
  }

  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    const bool* symmetriesBuffer,
    int batchSize,
    void* maskBuf,
    float* maskFloatBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* p1OutBuf,
    void* p1OutBuf2,
    void* g1OutBuf,
    void* g1OutBuf2,
    float* g1ConcatBuf,
    float* g1BiasBuf,
    float* p2OutBuf,
    float* g1PassBuf,
    float* policyBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const cudnnTensorDescriptor_t& p1OutDescriptor = p1OutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& g1OutDescriptor = g1OutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& p2InDescriptor = p2InDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& p2OutDescriptor = p2OutDescriptors[batchSize-1];

    bool applyBNRelu = true;

    p1Conv->apply(cudaHandles,trunkDescriptor,p1OutDescriptor,batchSize,false,trunkBuf,p1OutBuf,workspaceBuf,workspaceBytes);
    g1Conv->apply(cudaHandles,trunkDescriptor,g1OutDescriptor,batchSize,false,trunkBuf,g1OutBuf,workspaceBuf,workspaceBytes);
    g1BN->apply(cudaHandles,batchSize,applyBNRelu,g1OutBuf,maskBuf,g1OutBuf2);

    const float meanScale = 1.0f / (xSize*ySize);
    if(!usingFP16) {
      if(!usingNHWC) {
        if(maskSumBuf != NULL)
          customCudaPoolRowsGPoolNCHW((const float*)g1OutBuf2,g1ConcatBuf,batchSize,g1Channels,xSize*ySize,maskSumBuf);
        else
          customCudaPoolRowsSumAndMaxPositiveNCHW((const float*)g1OutBuf2,g1ConcatBuf,batchSize,g1Channels,xSize*ySize,meanScale);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      else {
        if(maskSumBuf != NULL)
          customCudaPoolRowsGPoolNHWC((const float*)g1OutBuf2,g1ConcatBuf,batchSize,xSize*ySize,g1Channels,maskSumBuf);
        else
          customCudaPoolRowsSumAndMaxPositiveNHWC((const float*)g1OutBuf2,g1ConcatBuf,batchSize,xSize*ySize,g1Channels,meanScale);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
    }
    else {
      customCudaCopyFromHalf((const half*)g1OutBuf2,(float*)workspaceBuf,batchSize*g1Channels*xSize*ySize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(!usingNHWC) {
        if(maskSumBuf != NULL)
          customCudaPoolRowsGPoolNCHW((const float*)workspaceBuf,g1ConcatBuf,batchSize,g1Channels,xSize*ySize,maskSumBuf);
        else
          customCudaPoolRowsSumAndMaxPositiveNCHW((const float*)workspaceBuf,g1ConcatBuf,batchSize,g1Channels,xSize*ySize,meanScale);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      else {
        if(maskSumBuf != NULL)
          customCudaPoolRowsGPoolNHWC((const float*)workspaceBuf,g1ConcatBuf,batchSize,xSize*ySize,g1Channels,maskSumBuf);
        else
          customCudaPoolRowsSumAndMaxPositiveNHWC((const float*)workspaceBuf,g1ConcatBuf,batchSize,xSize*ySize,g1Channels,meanScale);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
    }

    float zero = 0.0f;
    float one = 1.0f;
    gpoolToBiasMul->apply(cudaHandles,batchSize,g1ConcatBuf,g1BiasBuf,&zero,&one,workspaceBuf,workspaceBytes);

    float* p1OutBufA;
    float* p1OutBufB;
    if(!usingFP16) {
      p1OutBufA = (float*)p1OutBuf;
      p1OutBufB = (float*)p1OutBuf2;
    }
    else {
      customCudaCopyFromHalf((const half*)p1OutBuf,(float*)p1OutBuf2,batchSize*p1Channels*xSize*ySize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      p1OutBufA = (float*)p1OutBuf2;
      p1OutBufB = (float*)p1OutBuf;
    }

    if(!usingNHWC)
      customCudaAddNCBiasInplaceNCHW(p1OutBufA,g1BiasBuf,batchSize,p1Channels,xSize*ySize);
    else
      customCudaAddNCBiasInplaceNHWC(p1OutBufA,g1BiasBuf,batchSize,xSize*ySize,p1Channels);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

    p1BN->apply(cudaHandles,batchSize,true,p1OutBufA,maskFloatBuf,p1OutBufB);
    p2Conv->apply(cudaHandles,p2InDescriptor,p2OutDescriptor,batchSize,false,p1OutBufB,p2OutBuf,workspaceBuf,workspaceBytes);

    bool inverse = true;
    if(!usingNHWC)
      applySymmetriesNCHW<float>(symmetriesBuffer, inverse, batchSize, p2Channels, xSize, ySize, p2OutBuf, policyBuf);
    else
      applySymmetriesNHWC<float>(symmetriesBuffer, inverse, batchSize, p2Channels, xSize, ySize, p2OutBuf, policyBuf);

    gpoolToPassMul->apply(cudaHandles,batchSize,g1ConcatBuf,g1PassBuf,&zero,&one,workspaceBuf,workspaceBytes);

    customCudaChannelConcat(
      p2OutBuf,g1PassBuf,policyBuf,
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
  int maxBatchSize;
  int xSize;
  int ySize;
  int v1Channels;
  int v2Channels;
  int valueChannels;
  int scoreValueChannels;
  int ownershipChannels;
  bool usingFP16;
  bool usingNHWC;

  cudnnTensorDescriptor_t* v1OutDescriptors;
  cudnnTensorDescriptor_t* v3InDescriptors;
  cudnnTensorDescriptor_t* vOwnershipOutDescriptors;

  ConvLayer* v1Conv;
  BatchNormLayer* v1BN;
  ActivationLayer* v1Activation;
  MatMulLayer* v2Mul;
  MatBiasLayer* v2Bias;
  ActivationLayer* v2Activation;
  MatMulLayer* v3Mul;
  MatBiasLayer* v3Bias;
  MatMulLayer* sv3Mul;
  MatBiasLayer* sv3Bias;
  ConvLayer* vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(
    CudaHandles* cudaHandles,
    const ValueHeadDesc* desc,
    int maxBatchSz,
    int xS,
    int yS,
    const cudnnTensorDescriptor_t* trunkDescriptors,
    bool useFP16,
    bool useNHWC
  ) {
    name = desc->name;
    version = desc->version;
    maxBatchSize = maxBatchSz;
    xSize = xS;
    ySize = yS;
    v1Channels = desc->v1Conv.outChannels;
    v2Channels = desc->v2Mul.outChannels;
    valueChannels = desc->v3Mul.outChannels;
    scoreValueChannels = desc->sv3Mul.outChannels;
    ownershipChannels = desc->vOwnershipConv.outChannels;
    usingFP16 = useFP16;
    usingNHWC = useNHWC;

    v1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    v3InDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    vOwnershipOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& v1OutDescriptor = v1OutDescriptors[batchSize-1];
      cudnnTensorDescriptor_t& v3InDescriptor = v3InDescriptors[batchSize-1];

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&v1OutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        v1OutDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        desc->v1Conv.outChannels,
        ySize,
        xSize
      ));

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&v3InDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        v3InDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        CUDNN_DATA_FLOAT,
        batchSize,
        desc->v2Mul.outChannels,
        1,
        1
      ));

      cudnnTensorDescriptor_t& vOwnershipOutDescriptor = vOwnershipOutDescriptors[batchSize-1];

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&vOwnershipOutDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        vOwnershipOutDescriptor,
        (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        desc->vOwnershipConv.outChannels,
        ySize,
        xSize
      ));
    }

    v1Conv = new ConvLayer(cudaHandles,&desc->v1Conv,maxBatchSize,trunkDescriptors,v1OutDescriptors,useFP16,useNHWC);
    v1BN = new BatchNormLayer(cudaHandles,&desc->v1BN,xSize,ySize,useFP16,useNHWC);
    v1Activation = new ActivationLayer(cudaHandles,&desc->v1Activation);
    v2Mul = new MatMulLayer(cudaHandles,&desc->v2Mul,false);
    v2Bias = new MatBiasLayer(cudaHandles,&desc->v2Bias,false);
    v2Activation = new ActivationLayer(cudaHandles,&desc->v2Activation);
    v3Mul = new MatMulLayer(cudaHandles,&desc->v3Mul,false);
    v3Bias = new MatBiasLayer(cudaHandles,&desc->v3Bias,false);
    sv3Mul = new MatMulLayer(cudaHandles,&desc->sv3Mul,false);
    sv3Bias = new MatBiasLayer(cudaHandles,&desc->sv3Bias,false);
    vOwnershipConv = new ConvLayer(cudaHandles,&desc->vOwnershipConv,maxBatchSize,v1OutDescriptors,vOwnershipOutDescriptors,useFP16,useNHWC);
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
    delete sv3Mul;
    delete sv3Bias;
    delete vOwnershipConv;

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnDestroyTensorDescriptor(v1OutDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(v3InDescriptors[batchSize-1]);
      cudnnDestroyTensorDescriptor(vOwnershipOutDescriptors[batchSize-1]);
    }

    delete[] v1OutDescriptors;
    delete[] v3InDescriptors;
    delete[] vOwnershipOutDescriptors;
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
    b = sizeof(float)*batchSize*v1Channels*xSize*ySize;
    bytes = std::max(bytes,b);

    const cudnnTensorDescriptor_t& vOwnershipOutDescriptor = vOwnershipOutDescriptors[batchSize-1];

    b = sv3Mul->requiredWorkspaceBytes(cudaHandles);
    bytes = std::max(bytes,b);
    b = vOwnershipConv->requiredWorkspaceBytes(cudaHandles,v1OutDescriptor,vOwnershipOutDescriptor,batchSize);
    bytes = std::max(bytes,b);
    b = sizeof(float)*batchSize*ownershipChannels*xSize*ySize;
    bytes = std::max(bytes,b);

    return bytes;
  }


  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    const bool* symmetriesBuffer,
    int batchSize,
    void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* v1OutBuf,
    void* v1OutBuf2,
    float* v1MeanBuf,
    float* v2OutBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* ownershipScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const cudnnTensorDescriptor_t& v1OutDescriptor = v1OutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& v3InDescriptor = v3InDescriptors[batchSize-1];

    bool applyBNRelu = true;

    v1Conv->apply(cudaHandles,trunkDescriptor,v1OutDescriptor,batchSize,false,trunkBuf,v1OutBuf,workspaceBuf,workspaceBytes);
    v1BN->apply(cudaHandles,batchSize,applyBNRelu,v1OutBuf,maskBuf,v1OutBuf2);

    const float meanScale = 1.0f / (xSize*ySize);

    void* bufToBePooled = v1OutBuf2;
    if(usingFP16) {
      customCudaCopyFromHalf((const half*)v1OutBuf2,(float*)workspaceBuf,batchSize*v1Channels*xSize*ySize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      bufToBePooled = workspaceBuf;
    }

    if(!usingNHWC) {
      if(maskSumBuf != NULL)
        customCudaValueHeadPoolNCHW((float*)bufToBePooled,v1MeanBuf,batchSize,v1Channels,xSize*ySize,maskSumBuf);
      else
        customCudaPoolRowsSumNCHW((float*)bufToBePooled,v1MeanBuf,batchSize,v1Channels,xSize*ySize,meanScale);
    }
    else {
      if(maskSumBuf != NULL)
        customCudaValueHeadPoolNHWC((const float*)bufToBePooled,v1MeanBuf,batchSize,xSize*ySize,v1Channels,maskSumBuf);
      else
        customCudaPoolRowsSumNHWC((const float*)bufToBePooled,v1MeanBuf,batchSize,xSize*ySize,v1Channels,meanScale);
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());


    float zero = 0.0f;
    float one = 1.0f;
    v2Mul->apply(cudaHandles,batchSize,v1MeanBuf,v2OutBuf,&zero,&one,workspaceBuf,workspaceBytes);
    v2Bias->apply(cudaHandles,batchSize,v2OutBuf);
    v2Activation->apply(cudaHandles,v3InDescriptor,v3InDescriptor,v2OutBuf,v2OutBuf);
    v3Mul->apply(cudaHandles,batchSize,v2OutBuf,valueBuf,&zero,&one,workspaceBuf,workspaceBytes);
    v3Bias->apply(cudaHandles,batchSize,valueBuf);

    sv3Mul->apply(cudaHandles,batchSize,v2OutBuf,scoreValueBuf,&zero,&one,workspaceBuf,workspaceBytes);
    sv3Bias->apply(cudaHandles,batchSize,scoreValueBuf);

    const cudnnTensorDescriptor_t& vOwnershipOutDescriptor = vOwnershipOutDescriptors[batchSize-1];

    bool inverse = true;
    if(!usingFP16) {
      vOwnershipConv->apply(cudaHandles,v1OutDescriptor,vOwnershipOutDescriptor,batchSize,false,v1OutBuf2,ownershipBuf,workspaceBuf,workspaceBytes);
      if(!usingNHWC)
        applySymmetriesNCHW<float>(symmetriesBuffer, inverse, batchSize, ownershipChannels, xSize, ySize, (float*)ownershipBuf, (float*)workspaceBuf);
      else
        applySymmetriesNHWC<float>(symmetriesBuffer, inverse, batchSize, ownershipChannels, xSize, ySize, (float*)ownershipBuf, (float*)workspaceBuf);
    }
    else {
      vOwnershipConv->apply(cudaHandles,v1OutDescriptor,vOwnershipOutDescriptor,batchSize,false,v1OutBuf2,ownershipScratchBuf,workspaceBuf,workspaceBytes);
      if(!usingNHWC)
        applySymmetriesNCHW<half>(symmetriesBuffer, inverse, batchSize, ownershipChannels, xSize, ySize, (half*)ownershipScratchBuf, (half*)workspaceBuf);
      else
        applySymmetriesNHWC<half>(symmetriesBuffer, inverse, batchSize, ownershipChannels, xSize, ySize, (half*)ownershipScratchBuf, (half*)workspaceBuf);

      customCudaCopyFromHalf((const half*)ownershipScratchBuf,(float*)ownershipBuf,batchSize*ownershipChannels*xSize*ySize);
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

    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numInputGlobalChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numValueChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numScoreValueChannels);
    checkBufferSize(maxBatchSize,nnXLen,nnYLen,numOwnershipChannels);

    inputDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

    for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
      cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];

      CUDNN_ERR(name.c_str(),cudnnCreateTensorDescriptor(&inputDescriptor));
      CUDNN_ERR(name.c_str(),cudnnSetTensor4dDescriptor(
        inputDescriptor,
        inputsUseNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
        batchSize,
        numInputChannels,
        ySize,
        xSize
      ));
    }

    trunk = new Trunk(cudaHandles,&desc->trunk,maxBatchSize,xSize,ySize,inputDescriptors,useFP16,useNHWC);
    policyHead = new PolicyHead(cudaHandles,&desc->policyHead,maxBatchSize,xSize,ySize,trunk->trunkDescriptors,useFP16,useNHWC);
    valueHead = new ValueHead(cudaHandles,&desc->valueHead,maxBatchSize,xSize,ySize,trunk->trunkDescriptors,useFP16,useNHWC);
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
    bool requireExactNNLen,
    bool* symmetriesBuffer,

    void* inputBuf,
    void* inputScratchBuf,
    void* inputGlobalBuf,
    void* maskBuf,
    float* maskFloatBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* regularOutBuf,
    void* dilatedOutBuf,
    void* midInBuf,
    void* midScratchBuf,
    void* gpoolOutBuf,
    void* gpoolOutBuf2,
    void* gpoolConcatBuf,
    void* gpoolBiasBuf,
    void* regularScratchBuf,

    void* p1OutBuf,
    void* p1OutBuf2,
    void* g1OutBuf,
    void* g1OutBuf2,
    float* g1ConcatBuf,
    float* g1BiasBuf,
    float* p2OutBuf,
    float* g1PassBuf,
    float* policyBuf,

    void* v1OutBuf,
    void* v1OutBuf2,
    float* v1MeanBuf,
    float* v2OutBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* ownershipScratchBuf,

    const void* zeroBuf,
    const void* oneBuf,

    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& trunkDescriptor = trunk->trunkDescriptors[batchSize-1];

    if(!usingFP16) {
      bool inverse = false;
      if(inputsUsingNHWC)
        applySymmetriesNHWC<float>(symmetriesBuffer, inverse, batchSize, numInputChannels, xSize, ySize, (float*)inputBuf, (float*)inputScratchBuf);
      else
        applySymmetriesNCHW<float>(symmetriesBuffer, inverse, batchSize, numInputChannels, xSize, ySize, (float*)inputBuf, (float*)inputScratchBuf);
    }
    else {
      bool inverse = false;
      if(inputsUsingNHWC)
        applySymmetriesNHWC<half>(symmetriesBuffer, inverse, batchSize, numInputChannels, xSize, ySize, (half*)inputBuf, (half*)inputScratchBuf);
      else
        applySymmetriesNCHW<half>(symmetriesBuffer, inverse, batchSize, numInputChannels, xSize, ySize, (half*)inputBuf, (half*)inputScratchBuf);
    }

    if(!usingFP16) {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const float*)inputBuf, (float*)maskBuf, batchSize, xSize*ySize, numInputChannels);
      else
        customCudaChannel0ExtractNCHW((const float*)inputBuf, (float*)maskBuf, batchSize, numInputChannels, xSize*ySize);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
      maskFloatBuf = (float*)maskBuf;
      customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,xSize*ySize,1.0);
      CUDA_ERR("sumMask",cudaPeekAtLastError());
    }
    else {
      if(inputsUsingNHWC)
        customCudaChannel0ExtractNHWC((const half*)inputBuf, (half*)maskBuf, batchSize, xSize*ySize, numInputChannels);
      else
        customCudaChannel0ExtractNCHW((const half*)inputBuf, (half*)maskBuf, batchSize, numInputChannels, xSize*ySize);
      CUDA_ERR("modelExtractMask",cudaPeekAtLastError());
      customCudaCopyFromHalf((const half*)maskBuf,maskFloatBuf,batchSize*xSize*ySize);
      CUDA_ERR("copyMaskFromHalf",cudaPeekAtLastError());
      customCudaPoolRowsSumNCHW((const float*)maskFloatBuf,maskSumBuf,batchSize,1,xSize*ySize,1.0);
      CUDA_ERR("sumMask",cudaPeekAtLastError());
    }

    //Don't do any masking if we know the board is exactly the desired size
    if(requireExactNNLen) {
      //Set to NULL to signal downstream that this buf doesn't need to be used
      maskBuf = NULL;
      maskFloatBuf = NULL;
      //The global pooling structures need this no matter what, for normalizing based on this and its sqrt.
      //maskSumBuf = NULL;
    }

    trunk->apply(
      cudaHandles,
      inputDescriptor,
      batchSize,
      inputBuf,
      inputGlobalBuf,
      maskBuf,
      maskSumBuf,
      trunkBuf,
      trunkScratchBuf,
      regularOutBuf,
      dilatedOutBuf,
      midInBuf,
      midScratchBuf,
      gpoolOutBuf,
      gpoolOutBuf2,
      gpoolConcatBuf,
      gpoolBiasBuf,
      regularScratchBuf,
      zeroBuf,
      oneBuf,
      workspaceBuf,
      workspaceBytes
    );
    policyHead->apply(
      cudaHandles,
      trunkDescriptor,
      symmetriesBuffer,
      batchSize,
      maskBuf,
      maskFloatBuf,
      maskSumBuf,
      trunkBuf,
      p1OutBuf,
      p1OutBuf2,
      g1OutBuf,
      g1OutBuf2,
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
      trunkDescriptor,
      symmetriesBuffer,
      batchSize,
      maskBuf,
      maskSumBuf,
      trunkBuf,
      v1OutBuf,
      v1OutBuf2,
      v1MeanBuf,
      v2OutBuf,
      valueBuf,
      scoreValueBuf,
      ownershipBuf,
      ownershipScratchBuf,
      workspaceBuf,
      workspaceBytes
    );
  }

};


//------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  (void)modelFileIdx;
  LoadedModel* loadedModel = new LoadedModel(file);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

int NeuralNet::getModelVersion(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc.version;
}

//------------------------------------------------------------------------------

struct Buffers {
  //All of these are device pointers

  float* inputBufFloat;
  void* inputBuf;
  void* inputScratchBuf;
  float* inputGlobalBufFloat;
  void* inputGlobalBuf;
  size_t inputBufBytesFloat;
  size_t inputBufBytes;
  size_t inputGlobalBufBytesFloat;
  size_t inputGlobalBufBytes;

  void* maskBuf;
  float* maskFloatBuf;
  float* maskSumBuf;

  void* trunkBuf;
  void* trunkScratchBuf;
  void* regularOutBuf;
  void* dilatedOutBuf;
  void* midInBuf;
  void* midScratchBuf;
  void* gpoolOutBuf;
  void* gpoolOutBuf2;
  void* gpoolConcatBuf;
  void* gpoolBiasBuf;
  void* regularScratchBuf;

  void* p1OutBuf;
  void* p1OutBuf2;
  void* g1OutBuf;
  void* g1OutBuf2;
  float* g1ConcatBuf;
  float* g1BiasBuf;
  float* p2OutBuf;
  float* g1PassBuf;
  float* policyBuf;
  size_t policyBufBytes;

  void* v1OutBuf;
  void* v1OutBuf2;
  float* v1MeanBuf;
  float* v2OutBuf;
  float* valueBuf;
  size_t valueBufBytes;
  float* scoreValueBuf;
  size_t scoreValueBufBytes;
  void* ownershipBuf;
  void* ownershipScratchBuf;
  size_t ownershipBufBytes;

  void* zeroBuf;
  void* oneBuf;

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
    CUDA_ERR("Buffers",cudaMalloc(&inputScratchBuf, inputBufBytes));
    CUDA_ERR("Buffers",cudaMalloc(&inputGlobalBufFloat, inputGlobalBufBytesFloat));
    CUDA_ERR("Buffers",cudaMalloc(&inputGlobalBuf, inputGlobalBufBytes));

    CUDA_ERR("Buffers",cudaMalloc(&maskBuf, batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&maskFloatBuf, batchXYFloatBytes));
    CUDA_ERR("Buffers",cudaMalloc(&maskSumBuf, batchFloatBytes));

    CUDA_ERR("Buffers",cudaMalloc(&trunkBuf, m.trunk->trunkNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&trunkScratchBuf, m.trunk->trunkNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&regularOutBuf, m.trunk->regularNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&dilatedOutBuf, m.trunk->dilatedNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&midInBuf, (m.trunk->regularNumChannels + m.trunk->dilatedNumChannels) * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&midScratchBuf, (m.trunk->regularNumChannels + m.trunk->dilatedNumChannels) * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolOutBuf, m.trunk->gpoolNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolOutBuf2, m.trunk->gpoolNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolConcatBuf, m.trunk->gpoolNumChannels * batchBytes * 3));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolBiasBuf, m.trunk->regularNumChannels * batchBytes));
    CUDA_ERR("Buffers",cudaMalloc(&regularScratchBuf, m.trunk->regularNumChannels * batchXYBytes));

    CUDA_ERR("Buffers",cudaMalloc(&p1OutBuf, m.policyHead->p1Channels * batchXYFloatBytes)); //need to hold floats in addition to halfs
    CUDA_ERR("Buffers",cudaMalloc(&p1OutBuf2, m.policyHead->p1Channels * batchXYFloatBytes)); //need to hold floats in addition to halfs
    CUDA_ERR("Buffers",cudaMalloc(&g1OutBuf, m.policyHead->g1Channels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&g1OutBuf2, m.policyHead->g1Channels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&g1ConcatBuf, m.policyHead->g1Channels * batchFloatBytes * 3));
    CUDA_ERR("Buffers",cudaMalloc(&g1BiasBuf, m.policyHead->p1Channels * batchFloatBytes));
    CUDA_ERR("Buffers",cudaMalloc(&p2OutBuf, m.policyHead->p2Channels * batchXYFloatBytes));
    CUDA_ERR("Buffers",cudaMalloc(&g1PassBuf, m.policyHead->p2Channels * batchFloatBytes));

    policyBufBytes = m.policyHead->p2Channels * (batchXYFloatBytes + batchFloatBytes);
    CUDA_ERR("Buffers",cudaMalloc(&policyBuf, policyBufBytes));
    assert(m.policyHead->p2Channels == 1);

    CUDA_ERR("Buffers",cudaMalloc(&v1OutBuf, m.valueHead->v1Channels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&v1OutBuf2, m.valueHead->v1Channels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&v1MeanBuf, m.valueHead->v1Channels * 3 * batchFloatBytes));
    CUDA_ERR("Buffers",cudaMalloc(&v2OutBuf, m.valueHead->v2Channels * batchFloatBytes));

    valueBufBytes = m.valueHead->valueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&valueBuf, valueBufBytes));

    scoreValueBufBytes = m.valueHead->scoreValueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&scoreValueBuf, scoreValueBufBytes));

    //This buf is used for both an intermdiate fp16 result in fp16 mode, and ALSO the final fp32 output, so always must be fp32-sized
    ownershipBufBytes = m.valueHead->ownershipChannels * batchXYFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&ownershipBuf, ownershipBufBytes));
    CUDA_ERR("Buffers",cudaMalloc(&ownershipScratchBuf, ownershipBufBytes));

    if(!useFP16) {
      zeroBuf = malloc(sizeof(float));
      oneBuf = malloc(sizeof(float));
      *((float*)zeroBuf) = 0.0f;
      *((float*)oneBuf) = 1.0f;
    }
    else {
      //Convert to FP16 on the device, then copy back so we have it in host memory
      float zero = 0.0f;
      float one = 1.0f;
      void* zeroTmp;
      void* oneTmp;
      mallocAndCopyToDevice("Buffers",&zero,1,zeroTmp,useFP16);
      mallocAndCopyToDevice("Buffers",&one,1,oneTmp,useFP16);
      zeroBuf = malloc(sizeof(half));
      oneBuf = malloc(sizeof(half));
      CUDA_ERR("Buffers",cudaMemcpy(zeroBuf,zeroTmp,sizeof(half),cudaMemcpyDeviceToHost));
      CUDA_ERR("Buffers",cudaMemcpy(oneBuf,oneTmp,sizeof(half),cudaMemcpyDeviceToHost));
      cudaFree(zeroTmp);
      cudaFree(oneTmp);
    }

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
    cudaFree(inputScratchBuf);
    cudaFree(inputGlobalBufFloat);
    cudaFree(inputGlobalBuf);

    cudaFree(maskBuf);
    cudaFree(maskFloatBuf);
    cudaFree(maskSumBuf);

    cudaFree(trunkBuf);
    cudaFree(trunkScratchBuf);
    cudaFree(regularOutBuf);
    cudaFree(dilatedOutBuf);
    cudaFree(midInBuf);
    cudaFree(midScratchBuf);
    cudaFree(gpoolOutBuf);
    cudaFree(gpoolOutBuf2);
    cudaFree(gpoolConcatBuf);
    cudaFree(gpoolBiasBuf);
    cudaFree(regularScratchBuf);

    cudaFree(p1OutBuf);
    cudaFree(p1OutBuf2);
    cudaFree(g1OutBuf);
    cudaFree(g1OutBuf2);
    cudaFree(g1ConcatBuf);
    cudaFree(g1BiasBuf);
    cudaFree(p2OutBuf);
    cudaFree(g1PassBuf);
    cudaFree(policyBuf);

    cudaFree(v1OutBuf);
    cudaFree(v1OutBuf2);
    cudaFree(v1MeanBuf);
    cudaFree(v2OutBuf);
    cudaFree(valueBuf);
    if(scoreValueBuf != NULL)
      cudaFree(scoreValueBuf);
    if(ownershipBuf != NULL)
      cudaFree(ownershipBuf);
    if(ownershipScratchBuf != NULL)
      cudaFree(ownershipScratchBuf);

    free(zeroBuf);
    free(oneBuf);

    cudaFree(workspaceBuf);
  }

};



//------------------------------------------------------------------------------

struct ComputeHandle {
  CudaHandles* cudaHandles;
  Model* model;
  Buffers* buffers;
  bool usingFP16;
  int nnXLen;
  int nnYLen;
  bool requireExactNNLen;
  int policySize;

  ComputeHandle(
    const LoadedModel* loadedModel,
    int maxBatchSize,
    int xLen,
    int yLen,
    bool rExactNNLen,
    bool inputsUseNHWC,
    bool useFP16,
    bool useNHWC
  ) {
    cudaHandles = new CudaHandles();
    model = new Model(
      cudaHandles, &(loadedModel->modelDesc), maxBatchSize,
      xLen, yLen, inputsUseNHWC, useFP16, useNHWC
    );
    buffers = new Buffers(cudaHandles, *model, useFP16);
    usingFP16 = useFP16;
    nnXLen = xLen;
    nnYLen = yLen;
    requireExactNNLen = rExactNNLen;
    policySize = NNPos::getPolicySize(nnXLen, nnYLen);

    //Synchronize after creating buffers and copying all the weights, just in case
    CUDA_ERR("ComputeHandle", cudaDeviceSynchronize());
  }
  ~ComputeHandle() {
    delete buffers;
    delete model;
    delete cudaHandles;
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

ComputeHandle* NeuralNet::createComputeHandle(
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  int nnXLen,
  int nnYLen,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int cudaDeviceIdxForThisThread,
  bool cudaUseFP16,
  bool cudaUseNHWC
) {
  CUDA_ERR("createComputeHandle",cudaSetDevice(cudaDeviceIdxForThisThread));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,cudaDeviceIdxForThisThread);
  if(logger != NULL) {
    logger->write(
      "Cuda backend: Found GPU " + string(prop.name)
      + " memory " + Global::uint64ToString(prop.totalGlobalMem)
      + " compute capability major " + Global::intToString(prop.major)
      + " minor " + Global::intToString(prop.minor)
    );
    logger->write("Cuda backend: Model version " + Global::intToString(loadedModel->modelDesc.version));
  }
  if(cudaUseFP16 && (prop.major < 5 || (prop.major == 5 && prop.minor < 3)))
    throw StringError("Cuda device versions below 5.3 do not support cudaUseFP16=true");

  ComputeHandle* gpuHandle = new ComputeHandle(loadedModel,maxBatchSize,nnXLen,nnYLen,requireExactNNLen,inputsUseNHWC,cudaUseFP16,cudaUseNHWC);
  return gpuHandle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
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
  bool* symmetriesBuffer; //Host pointer

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
    symmetriesBuffer = new bool[NNInputs::NUM_SYMMETRY_BOOLS];

    policyResults = new float[(size_t)maxBatchSize * (1 + xSize * ySize)];
    valueResults = new float[(size_t)maxBatchSize * m.numValueChannels];

    scoreValueResults = new float[(size_t)maxBatchSize * m.numScoreValueChannels];
    ownershipResults = new float[(size_t)maxBatchSize * xSize * ySize * m.numOwnershipChannels];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    delete[] symmetriesBuffer;
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

float* NeuralNet::getBatchEltSpatialInplace(InputBuffers* inputBuffers, int nIdx) {
  assert(nIdx < inputBuffers->maxBatchSize);
  return inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
}

float* NeuralNet::getBatchEltGlobalInplace(InputBuffers* inputBuffers, int nIdx) {
  assert(nIdx < inputBuffers->maxBatchSize);
  return inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);
}

int NeuralNet::getBatchEltSpatialLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputElts;
}
int NeuralNet::getBatchEltGlobalLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputGlobalElts;
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* inputBuffers) {
  return inputBuffers->symmetriesBuffer;
}


//---------------------------------------------------------------------------------------


void NeuralNet::getOutput(ComputeHandle* gpuHandle, InputBuffers* inputBuffers, int numBatchEltsFilled, vector<NNOutput*>& outputs) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  int batchSize = numBatchEltsFilled;
  int nnXLen = gpuHandle->nnXLen;
  int nnYLen = gpuHandle->nnYLen;
  int version = gpuHandle->model->version;
  Buffers* buffers = gpuHandle->buffers;

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
    gpuHandle->cudaHandles,
    batchSize,
    gpuHandle->requireExactNNLen,
    inputBuffers->symmetriesBuffer,

    buffers->inputBuf,
    buffers->inputScratchBuf,
    buffers->inputGlobalBuf,

    buffers->maskBuf,
    buffers->maskFloatBuf,
    buffers->maskSumBuf,

    buffers->trunkBuf,
    buffers->trunkScratchBuf,
    buffers->regularOutBuf,
    buffers->dilatedOutBuf,
    buffers->midInBuf,
    buffers->midScratchBuf,
    buffers->gpoolOutBuf,
    buffers->gpoolOutBuf2,
    buffers->gpoolConcatBuf,
    buffers->gpoolBiasBuf,
    buffers->regularScratchBuf,

    buffers->p1OutBuf,
    buffers->p1OutBuf2,
    buffers->g1OutBuf,
    buffers->g1OutBuf2,
    buffers->g1ConcatBuf,
    buffers->g1BiasBuf,
    buffers->p2OutBuf,
    buffers->g1PassBuf,
    buffers->policyBuf,

    buffers->v1OutBuf,
    buffers->v1OutBuf2,
    buffers->v1MeanBuf,
    buffers->v2OutBuf,
    buffers->valueBuf,
    buffers->scoreValueBuf,
    buffers->ownershipBuf,
    buffers->ownershipScratchBuf,

    buffers->zeroBuf,
    buffers->oneBuf,

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

    float* policyProbs = output->policyProbs;

    //These are not actually correct, the client does the postprocessing to turn them into
    //policy probabilities and white game outcome probabilities
    //Also we don't fill in the nnHash here either
    std::copy(
      inputBuffers->policyResults + row * gpuHandle->policySize,
      inputBuffers->policyResults + (row+1) * gpuHandle->policySize,
      policyProbs
    );

    int numValueChannels = gpuHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      assert(gpuHandle->model->numOwnershipChannels == 1);
      std::copy(
        inputBuffers->ownershipResults + row * nnXLen * nnYLen,
        inputBuffers->ownershipResults + (row+1) * nnXLen * nnYLen,
        output->whiteOwnerMap
      );
    }

    if(version >= 4) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
    }
    else if(version >= 3) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
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
  CudaHandles* cudaHandles = new CudaHandles();

  int xSize = nnXLen;
  int ySize = nnYLen;

  size_t numInputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->inChannels;
  size_t numOutputFloats = (size_t)desiredBatchSize * xSize * ySize * desc->outChannels;
  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateConv: unexpected input buffer size");

  void* deviceInput;
  void* deviceOutput;
  mallocAndCopyToDevice("deviceInput", inputBuffer.data(), numInputFloats, deviceInput, useFP16);
  mallocOnDevice("deviceOutput", numOutputFloats, deviceOutput, useFP16);

  int maxBatchSize = desiredBatchSize;
  cudnnTensorDescriptor_t* inputDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
  cudnnTensorDescriptor_t* outputDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

  for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
    cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];
    cudnnTensorDescriptor_t& outputDescriptor = outputDescriptors[batchSize-1];

    CUDNN_ERR("inputDescriptor",cudnnCreateTensorDescriptor(&inputDescriptor));
    CUDNN_ERR("inputDescriptor",cudnnSetTensor4dDescriptor(
      inputDescriptor,
      (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
      (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
      batchSize,
      desc->inChannels,
      ySize,
      xSize
    ));
    CUDNN_ERR("outputDescriptor",cudnnCreateTensorDescriptor(&outputDescriptor));
    CUDNN_ERR("outputDescriptor",cudnnSetTensor4dDescriptor(
      outputDescriptor,
      (useNHWC ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW),
      (useFP16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT),
      batchSize,
      desc->outChannels,
      ySize,
      xSize
    ));
  }

  ConvLayer* convLayer = new ConvLayer(cudaHandles,desc,maxBatchSize,inputDescriptors,outputDescriptors,useFP16,useNHWC);

  size_t workspaceBytes =
    convLayer->requiredWorkspaceBytes(cudaHandles,inputDescriptors[desiredBatchSize-1],outputDescriptors[desiredBatchSize-1],desiredBatchSize);
  void* deviceWorkspace;
  CUDA_ERR("deviceWorkspace",cudaMalloc(&deviceWorkspace, workspaceBytes));


  bool accumulate = false;
  convLayer->apply(
    cudaHandles,
    inputDescriptors[desiredBatchSize-1],
    outputDescriptors[desiredBatchSize-1],
    desiredBatchSize,
    accumulate,
    deviceInput,
    deviceOutput,
    deviceWorkspace,
    workspaceBytes
  );

  outputBuffer.resize(numOutputFloats);
  expensiveCopyFromDevice("copyResultsToHost", outputBuffer.data(), numOutputFloats, deviceOutput, useFP16);

  cudaFree(deviceWorkspace);

  delete convLayer;

  for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
    cudnnDestroyTensorDescriptor(inputDescriptors[batchSize-1]);
    cudnnDestroyTensorDescriptor(outputDescriptors[batchSize-1]);
  }
  delete[] inputDescriptors;
  delete[] outputDescriptors;
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  delete cudaHandles;

  return true;
}


#endif  // USE_CUDA_BACKEND
