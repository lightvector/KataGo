#ifdef USE_CUDA_BACKEND
#include "../neuralnet/cudaerrorcheck.h"

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>

#include <fstream>
#include <zstr/src/zstr.hpp>

#include "../neuralnet/cudahelpers.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"

using namespace std;

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

static void mallocAndCopyToDevice(const string& name, float* weights, int numWeights, void*& deviceBuf, bool useFP16) {
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

  ConvLayerDesc()
    :convYSize(0),convXSize(0),inChannels(0),outChannels(0),dilationY(1),dilationX(1)
  {}

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

    // cudaDeviceSynchronize();
    // cout << name << endl;
    // float* inputsTmp = new float[224*19*19];
    // CUDA_ERR("DEBUG",cudaMemcpy(inputsTmp, inputBuf, sizeof(float)*224*19*19/5, cudaMemcpyDeviceToHost));
    // cout << name << " inputs ";
    // for(int c = 0; c<2; c++)
    //   for(int y = 0; y<2; y++)
    //     for(int x = 0; x<2; x++)
    //       cout << inputsTmp[y*19*224+x*224+c] << " ";
    // cout << endl;
    // float* filterTmp = new float[3*3*3];
    // CUDA_ERR("DEBUG",cudaMemcpy(filterTmp, filterBuf, sizeof(float)*3*3*3, cudaMemcpyDeviceToHost));
    // cout << name << " filter ";
    // for(int i = 0; i<18; i++)
    //   cout << filterTmp[i] << " ";
    // cout << endl;
    // float* outputsTmp = new float[224*19*19];
    // CUDA_ERR("DEBUG",cudaMemcpy(outputsTmp, outputBuf, sizeof(float)*224*19*19/5, cudaMemcpyDeviceToHost));
    // cout << name << " outputs ";
    // for(int c = 0; c<2; c++)
    //   for(int y = 0; y<2; y++)
    //     for(int x = 0; x<2; x++)
    //       cout << outputsTmp[y*19*224+x*224+c] << " ";
    // cout << endl;

    // cudaDeviceSynchronize();
    // cout << name << endl;
    // float* inputsTmp = new float[224*19*19];
    // CUDA_ERR("DEBUG",cudaMemcpy(inputsTmp, inputBuf, sizeof(float)*224*19*19/5, cudaMemcpyDeviceToHost));
    // cout << name << " inputs ";
    // for(int c = 0; c<2; c++)
    //   for(int y = 0; y<2; y++)
    //     for(int x = 0; x<2; x++)
    //       cout << inputsTmp[c*19*19+y*19+x] << " ";
    // cout << endl;
    // float* filterTmp = new float[3*3*3];
    // CUDA_ERR("DEBUG",cudaMemcpy(filterTmp, filterBuf, sizeof(float)*3*3*3, cudaMemcpyDeviceToHost));
    // cout << name << " filter ";
    // for(int i = 0; i<18; i++)
    //   cout << filterTmp[i] << " ";
    // cout << endl;
    // float* outputsTmp = new float[224*19*19];
    // CUDA_ERR("DEBUG",cudaMemcpy(outputsTmp, outputBuf, sizeof(float)*224*19*19/5, cudaMemcpyDeviceToHost));
    // cout << name << " outputs ";
    // for(int c = 0; c<2; c++)
    //   for(int y = 0; y<2; y++)
    //     for(int x = 0; x<2; x++)
    //       cout << outputsTmp[c*19*19+y*19+x] << " ";
    // cout << endl;

    // cudaDeviceSynchronize();
    // cout << name << endl;
    // float tmp[12];
    // customCudaCopyFromHalf((const half*)inputBuf,(float*)workspaceBuf,12);
    // CUDA_ERR("DEBUG",cudaMemcpy(tmp, workspaceBuf, sizeof(float)*12, cudaMemcpyDeviceToHost));
    // for(int i = 0; i<12; i++)
    //   cout << tmp[i] << " ";
    // cout << endl;

    // customCudaCopyFromHalf((const half*)filterBuf,(float*)workspaceBuf,12);
    // CUDA_ERR("DEBUG",cudaMemcpy(tmp, workspaceBuf, sizeof(float)*12, cudaMemcpyDeviceToHost));
    // for(int i = 0; i<12; i++)
    //   cout << tmp[i] << " ";
    // cout << endl;

    // customCudaCopyFromHalf((const half*)outputBuf,(float*)workspaceBuf,12);
    // CUDA_ERR("DEBUG",cudaMemcpy(tmp, workspaceBuf, sizeof(float)*12, cudaMemcpyDeviceToHost));
    // for(int i = 0; i<12; i++)
    //   cout << tmp[i] << " ";
    // cout << endl;

    // if(name == "conv1") {
    //   float* outputsTmp = new float[224*19*19];
    //   CUDA_ERR("DEBUG",cudaMemcpy(outputsTmp, outputBuf, sizeof(float)*224*19*19, cudaMemcpyDeviceToHost));
    //   for(int ic = 0; ic < 224; ic++) {
    //     for(int dy = 0; dy <= 1; dy++) {
    //       for(int dx = 0; dx <= 1; dx++) {
    //         float x = outputsTmp[19*19*ic + dy*19 + dx];
    //         cout << "TEST " << ic << " " << dy << " " << dx << " " << x << endl;
    //       }
    //     }
    //   }
    // }

    // if(name == "conv1") {
    //   float* outputsTmp = new float[224*19*19];
    //   customCudaCopyFromHalf((const half*)outputBuf,(float*)workspaceBuf,224*19*19);
    //   CUDA_ERR("DEBUG",cudaMemcpy(outputsTmp, workspaceBuf, sizeof(float)*224*19*19, cudaMemcpyDeviceToHost));
    //   for(int ic = 0; ic < 224; ic++) {
    //     for(int dy = 0; dy <= 1; dy++) {
    //       for(int dx = 0; dx <= 1; dx++) {
    //         float x = outputsTmp[19*19*ic + dy*19 + dx];
    //         cout << "TEST " << ic << " " << dy << " " << dx << " " << x << endl;
    //       }
    //     }
    //   }
    // }

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

  BNLayerDesc()
    :numChannels(0),epsilon(0.001),hasScale(false),hasBias(false)
  {}

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

  BNLayer() = delete;
  BNLayer(const BNLayer&) = delete;
  BNLayer& operator=(const BNLayer&) = delete;

  BNLayer(
    CudaHandles* cudaHandles,
    const BNLayerDesc* desc,
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
  ~BNLayer() {
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

struct MatMulLayerDesc {
  string name;
  int inChannels;
  int outChannels;
  vector<float> weights;

  MatMulLayerDesc()
    :inChannels(0),outChannels(0)
  {}

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

struct MatBiasLayerDesc {
  string name;
  int numChannels;
  vector<float> weights;

  MatBiasLayerDesc()
    :numChannels(0)
  {}

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

struct GlobalPoolingResidualBlockDesc {
  string name;
  int version;
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

  GlobalPoolingResidualBlockDesc(istream& in, int vrsn) {
    in >> name;
    if(in.fail())
      throw StringError(name + ": gpool res block failed to parse name");
    version = vrsn;
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
    if(version >= 3) {
      if(gpoolBN.numChannels * 3 != gpoolToBiasMul.inChannels)
        throw StringError(name+Global::strprintf(
          ": gpoolBN.numChannels * 3 (%d) != gpoolToBiasMul.inChannels (%d)", gpoolBN.numChannels * 3, gpoolToBiasMul.inChannels
        ));
    }
    else {
      if(gpoolBN.numChannels * 2 != gpoolToBiasMul.inChannels)
        throw StringError(name+Global::strprintf(
          ": gpoolBN.numChannels * 2 (%d) != gpoolToBiasMul.inChannels (%d)", gpoolBN.numChannels * 2, gpoolToBiasMul.inChannels
        ));
    }
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

static const int ORDINARY_BLOCK_KIND = 0;
static const int DILATED_BLOCK_KIND = 1;
static const int GLOBAL_POOLING_BLOCK_KIND = 2;

struct TrunkDesc {
  string name;
  int version;
  int numBlocks;
  int trunkNumChannels;
  int midNumChannels;     //Currently every plain residual block must have the same number of mid conv channels
  int regularNumChannels; //Currently every dilated or gpool residual block must have the same number of regular conv channels
  int dilatedNumChannels; //Currently every dilated residual block must have the same number of dilated conv channels
  int gpoolNumChannels;   //Currently every gpooling residual block must have the same number of gpooling conv channels
  ConvLayerDesc initialConv;
  MatMulLayerDesc initialMatMul;
  vector<pair<int,void*>> blocks;
  BNLayerDesc trunkTipBN;
  ActivationLayerDesc trunkTipActivation;

  TrunkDesc()
    :version(-1),numBlocks(0),trunkNumChannels(0),midNumChannels(0),regularNumChannels(0),dilatedNumChannels(0),gpoolNumChannels(0)
  {}

  TrunkDesc(istream& in, int vrsn) {
    in >> name;
    version = vrsn;
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

    if(version >= 3) {
      initialMatMul = MatMulLayerDesc(in);
      if(initialMatMul.outChannels != trunkNumChannels)
        throw StringError(name+Global::strprintf(
          ": %s initialMatMul.outChannels (%d) != trunkNumChannels (%d)", initialMatMul.name.c_str(), initialMatMul.outChannels, trunkNumChannels
          ));
    }

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
        GlobalPoolingResidualBlockDesc* desc = new GlobalPoolingResidualBlockDesc(in,version);

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
    version = other.version;
    numBlocks = other.numBlocks;
    trunkNumChannels = other.trunkNumChannels;
    midNumChannels = other.midNumChannels;
    regularNumChannels = other.regularNumChannels;
    dilatedNumChannels = other.dilatedNumChannels;
    gpoolNumChannels = other.gpoolNumChannels;
    initialConv = std::move(other.initialConv);
    initialMatMul = std::move(other.initialMatMul);
    blocks = std::move(other.blocks);
    trunkTipBN = std::move(other.trunkTipBN);
    trunkTipActivation = std::move(other.trunkTipActivation);
  }

  TrunkDesc& operator=(TrunkDesc&& other) {
    name = std::move(other.name);
    version = other.version;
    numBlocks = other.numBlocks;
    trunkNumChannels = other.trunkNumChannels;
    midNumChannels = other.midNumChannels;
    regularNumChannels = other.regularNumChannels;
    dilatedNumChannels = other.dilatedNumChannels;
    gpoolNumChannels = other.gpoolNumChannels;
    initialConv = std::move(other.initialConv);
    initialMatMul = std::move(other.initialMatMul);
    blocks = std::move(other.blocks);
    trunkTipBN = std::move(other.trunkTipBN);
    trunkTipActivation = std::move(other.trunkTipActivation);
    return *this;
  }

};


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
    initialMatMul = NULL;
    if(version >= 3)
      initialMatMul = new MatMulLayer(cudaHandles,&desc->initialMatMul,useFP16);

    trunkTipBN = new BNLayer(cudaHandles,&desc->trunkTipBN,xSize,ySize,useFP16,useNHWC);
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

struct PolicyHeadDesc {
  string name;
  int version;
  ConvLayerDesc p1Conv;
  ConvLayerDesc g1Conv;
  BNLayerDesc g1BN;
  ActivationLayerDesc g1Activation;
  MatMulLayerDesc gpoolToBiasMul;
  BNLayerDesc p1BN;
  ActivationLayerDesc p1Activation;
  ConvLayerDesc p2Conv;
  MatMulLayerDesc gpoolToPassMul;

  PolicyHeadDesc()
    :version(-1)
  {}

  PolicyHeadDesc(istream& in, int vrsn) {
    in >> name;
    version = vrsn;

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
    if(version >= 3) {
      if(gpoolToBiasMul.inChannels != g1BN.numChannels*3)
        throw StringError(name+Global::strprintf(
          ": gpoolToBiasMul.inChannels (%d) != g1BN.numChannels*3 (%d)", gpoolToBiasMul.inChannels, g1BN.numChannels*3
        ));
    }
    else {
      if(gpoolToBiasMul.inChannels != g1BN.numChannels*2)
        throw StringError(name+Global::strprintf(
          ": gpoolToBiasMul.inChannels (%d) != g1BN.numChannels*2 (%d)", gpoolToBiasMul.inChannels, g1BN.numChannels*2
        ));
    }
    if(gpoolToBiasMul.outChannels != p1BN.numChannels)
      throw StringError(name+Global::strprintf(
        ": gpoolToBiasMul.outChannels (%d) != p1BN.numChannels (%d)", gpoolToBiasMul.outChannels, p1BN.numChannels
      ));
    if(version >= 1) {
      if(p2Conv.inChannels != p1BN.numChannels)
        throw StringError(name+Global::strprintf(
          ": p2Conv.inChannels (%d) != p1BN.numChannels (%d)", p2Conv.inChannels, p1BN.numChannels
        ));
    }
    else {
      if(p2Conv.inChannels != p1BN.numChannels*2)
        throw StringError(name+Global::strprintf(
          ": p2Conv.inChannels (%d) != p1BN.numChannels*2 (%d)", p2Conv.inChannels, p1BN.numChannels*2
        ));
    }
    if(p2Conv.outChannels != 1)
      throw StringError(name+Global::strprintf(
        ": p2Conv.outChannels (%d) != 1", p2Conv.outChannels
      ));
    if(version >= 3) {
      if(gpoolToPassMul.inChannels != g1BN.numChannels*3)
        throw StringError(name+Global::strprintf(
          ": gpoolToPassMul.inChannels (%d) != g1BN.numChannels*3 (%d)", gpoolToPassMul.inChannels, g1BN.numChannels*3
        ));
    }
    else {
      if(gpoolToPassMul.inChannels != g1BN.numChannels*2)
        throw StringError(name+Global::strprintf(
          ": gpoolToPassMul.inChannels (%d) != g1BN.numChannels*2 (%d)", gpoolToPassMul.inChannels, g1BN.numChannels*2
        ));
    }
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
    version = other.version;
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
    g1BN = new BNLayer(cudaHandles,&desc->g1BN,xSize,ySize,useFP16,useNHWC);
    g1Activation = new ActivationLayer(cudaHandles,&desc->g1Activation);
    gpoolToBiasMul = new MatMulLayer(cudaHandles,&desc->gpoolToBiasMul,false);
    p1BN = new BNLayer(cudaHandles,&desc->p1BN,xSize,ySize,false,useNHWC);
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


struct ValueHeadDesc {
  string name;
  int version;
  ConvLayerDesc v1Conv;
  BNLayerDesc v1BN;
  ActivationLayerDesc v1Activation;
  MatMulLayerDesc v2Mul;
  MatBiasLayerDesc v2Bias;
  ActivationLayerDesc v2Activation;
  MatMulLayerDesc v3Mul;
  MatBiasLayerDesc v3Bias;
  MatMulLayerDesc sv3Mul;
  MatBiasLayerDesc sv3Bias;
  ConvLayerDesc vOwnershipConv;

  ValueHeadDesc()
    :version(-1)
  {}

  ValueHeadDesc(istream& in, int vrsn) {
    in >> name;
    version = vrsn;

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

    if(version >= 3) {
      sv3Mul = MatMulLayerDesc(in);
      sv3Bias = MatBiasLayerDesc(in);
      vOwnershipConv = ConvLayerDesc(in);
    }

    if(in.fail())
      throw StringError(name + ": value head istream fail after parsing layers");

    if(v1Conv.outChannels != v1BN.numChannels)
      throw StringError(name+Global::strprintf(
        ": v1Conv.outChannels (%d) != v1BN.numChannels (%d)", v1Conv.outChannels, v1BN.numChannels
      ));

    if(version >= 3) {
      if(v2Mul.inChannels != v1BN.numChannels*3)
        throw StringError(name+Global::strprintf(
          ": v2Mul.inChannels (%d) != v1BN.numChannels*3 (%d)", v2Mul.inChannels, v1BN.numChannels*3
        ));
    }
    else {
      if(v2Mul.inChannels != v1BN.numChannels)
        throw StringError(name+Global::strprintf(
          ": v2Mul.inChannels (%d) != v1BN.numChannels (%d)", v2Mul.inChannels, v1BN.numChannels
        ));
    }

    if(v2Mul.outChannels != v2Bias.numChannels)
      throw StringError(name+Global::strprintf(
        ": v2Mul.outChannels (%d) != v2Bias.numChannels (%d)", v2Mul.outChannels, v2Bias.numChannels
      ));
    if(version >= 1) {
      if(v2Mul.outChannels != v3Mul.inChannels)
        throw StringError(name+Global::strprintf(
          ": v2Mul.outChannels (%d) != v3Mul.inChannels (%d)", v2Mul.outChannels, v3Mul.inChannels
        ));
    }
    else {
      if(v2Mul.outChannels*2 != v3Mul.inChannels)
        throw StringError(name+Global::strprintf(
          ": v2Mul.outChannels*2 (%d) != v3Mul.inChannels (%d)", v2Mul.outChannels*2, v3Mul.inChannels
        ));
    }
    if(version >= 3) {
      if(v3Mul.outChannels != 3)
        throw StringError(name+Global::strprintf(
          ": v3Mul.outChannels (%d) != 3", v3Mul.outChannels
        ));
      if(v3Bias.numChannels != 3)
        throw StringError(name+Global::strprintf(
          ": v3Bias.numChannels (%d) != 3", v3Bias.numChannels
        ));
    }
    else {
      if(v3Mul.outChannels != 1)
        throw StringError(name+Global::strprintf(
          ": v3Mul.outChannels (%d) != 1", v3Mul.outChannels
        ));
      if(v3Bias.numChannels != 1)
        throw StringError(name+Global::strprintf(
          ": v3Bias.numChannels (%d) != 1", v3Bias.numChannels
        ));
    }

    if(version >= 3) {
      if(sv3Mul.inChannels != v2Mul.outChannels)
        throw StringError(name+Global::strprintf(
          ": sv3Mul.inChannels (%d) != v2Mul.outChannels (%d)", sv3Mul.inChannels, v2Mul.outChannels
        ));

      if(version >= 4) {
        if(sv3Mul.outChannels != 2)
          throw StringError(name+Global::strprintf(
            ": sv3Mul.outChannels (%d) != 2", sv3Mul.outChannels
          ));
        if(sv3Bias.numChannels != 2)
          throw StringError(name+Global::strprintf(
            ": sv3Bias.numChannels (%d) != 2", sv3Bias.numChannels
          ));
      }
      else {
        if(sv3Mul.outChannels != 1)
          throw StringError(name+Global::strprintf(
            ": sv3Mul.outChannels (%d) != 1", sv3Mul.outChannels
          ));
        if(sv3Bias.numChannels != 1)
          throw StringError(name+Global::strprintf(
            ": sv3Bias.numChannels (%d) != 1", sv3Bias.numChannels
          ));
      }

      if(vOwnershipConv.inChannels != v1Conv.outChannels)
        throw StringError(name+Global::strprintf(
          ": vOwnershipConv.outChannels (%d) != v1Conv.outChannels (%d)", vOwnershipConv.inChannels, v1Conv.outChannels
        ));
      if(vOwnershipConv.outChannels != 1)
        throw StringError(name+Global::strprintf(
          ": vOwnershipConv.outChannels (%d) != 1", vOwnershipConv.outChannels
        ));
    }

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
    version = other.version;
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

};



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
  BNLayer* v1BN;
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
    if(version >= 3)
      vOwnershipOutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    else
      vOwnershipOutDescriptors = NULL;

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
        (version >= 1 ? desc->v2Mul.outChannels : desc->v2Mul.outChannels*2),
        1,
        1
      ));

      if(version >= 3) {
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
    }

    v1Conv = new ConvLayer(cudaHandles,&desc->v1Conv,maxBatchSize,trunkDescriptors,v1OutDescriptors,useFP16,useNHWC);
    v1BN = new BNLayer(cudaHandles,&desc->v1BN,xSize,ySize,useFP16,useNHWC);
    v1Activation = new ActivationLayer(cudaHandles,&desc->v1Activation);
    v2Mul = new MatMulLayer(cudaHandles,&desc->v2Mul,false);
    v2Bias = new MatBiasLayer(cudaHandles,&desc->v2Bias,false);
    v2Activation = new ActivationLayer(cudaHandles,&desc->v2Activation);
    v3Mul = new MatMulLayer(cudaHandles,&desc->v3Mul,false);
    v3Bias = new MatBiasLayer(cudaHandles,&desc->v3Bias,false);
    if(version >= 3) {
      sv3Mul = new MatMulLayer(cudaHandles,&desc->sv3Mul,false);
      sv3Bias = new MatBiasLayer(cudaHandles,&desc->sv3Bias,false);
      vOwnershipConv = new ConvLayer(cudaHandles,&desc->vOwnershipConv,maxBatchSize,v1OutDescriptors,vOwnershipOutDescriptors,useFP16,useNHWC);
    }
    else {
      sv3Mul = NULL;
      sv3Bias = NULL;
      vOwnershipConv = NULL;
    }
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
      if(version >= 3)
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


    if(version >= 3) {
      const cudnnTensorDescriptor_t& vOwnershipOutDescriptor = vOwnershipOutDescriptors[batchSize-1];

      b = sv3Mul->requiredWorkspaceBytes(cudaHandles);
      bytes = std::max(bytes,b);
      b = vOwnershipConv->requiredWorkspaceBytes(cudaHandles,v1OutDescriptor,vOwnershipOutDescriptor,batchSize);
      bytes = std::max(bytes,b);
      b = sizeof(float)*batchSize*ownershipChannels*xSize*ySize;
      bytes = std::max(bytes,b);
    }

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

    if(version >= 3) {
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

  }

};


//------------------------------------------------------------------------------

struct ModelDesc {
  string name;
  int version;
  int xSizePreV3;
  int ySizePreV3;
  int numInputChannels;
  int numInputGlobalChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  TrunkDesc trunk;
  PolicyHeadDesc policyHead;
  ValueHeadDesc valueHead;

  ModelDesc()
    :version(-1),xSizePreV3(0),ySizePreV3(0),numInputChannels(0),numInputGlobalChannels(0),numValueChannels(0),numScoreValueChannels(0),numOwnershipChannels(0)
  {}

  ModelDesc(istream& in) {
    in >> name;
    in >> version;
    if(in.fail())
      throw StringError(name + ": model failed to parse name or version");

    if(version < 0 || version > NNModelVersion::latestModelVersionImplemented)
      throw StringError(name + ": model found unsupported version " + Global::intToString(version));
    if(version < 1)
      throw StringError("Version 0 neural nets no longer supported in cuda backend");

    if(version >= 3) {
      xSizePreV3 = 0; //Unused, V3 uses nnXLen instead
      ySizePreV3 = 0; //Unused, V3 uses nnYLen instead
    }
    else {
      in >> xSizePreV3;
      in >> ySizePreV3;
      if(in.fail())
        throw StringError(name + ": model failed to parse xSize or ySize");
      if(xSizePreV3 <= 0 || ySizePreV3 <= 0)
        throw StringError(name + ": model xSize and ySize must be positive");
    }

    in >> numInputChannels;
    if(in.fail())
      throw StringError(name + ": model failed to parse numInputChannels");
    if(numInputChannels <= 0)
      throw StringError(name + ": model numInputChannels must be positive");

    if(version >= 3) {
      in >> numInputGlobalChannels;
      if(in.fail())
        throw StringError(name + ": model failed to parse numInputGlobalChannels");
      if(numInputGlobalChannels <= 0)
        throw StringError(name + ": model numInputGlobalChannels must be positive");
    }
    else
      numInputGlobalChannels = 0;

    trunk = TrunkDesc(in,version);
    policyHead = PolicyHeadDesc(in,version);
    valueHead = ValueHeadDesc(in,version);

    numValueChannels = valueHead.v3Mul.outChannels;
    numScoreValueChannels = valueHead.sv3Mul.outChannels;
    numOwnershipChannels = valueHead.vOwnershipConv.outChannels;

    if(in.fail())
      throw StringError(name + ": model desc istream fail after parsing model");

    if(numInputChannels != trunk.initialConv.inChannels)
      throw StringError(name+Global::strprintf(
        ": numInputChannels (%d) != trunk.initialConv.inChannels (%d)", numInputChannels, trunk.initialConv.inChannels
      ));
    if(version >= 3) {
      if(numInputGlobalChannels != trunk.initialMatMul.inChannels)
        throw StringError(name+Global::strprintf(
          ": numInputChannels (%d) != trunk.initialMatMul.inChannels (%d)", numInputGlobalChannels, trunk.initialMatMul.inChannels
        ));
    }

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
    version = other.version;
    xSizePreV3 = other.xSizePreV3;
    ySizePreV3 = other.ySizePreV3;
    numInputChannels = other.numInputChannels;
    numInputGlobalChannels = other.numInputGlobalChannels;
    numValueChannels = other.numValueChannels;
    numScoreValueChannels = other.numScoreValueChannels;
    numOwnershipChannels = other.numOwnershipChannels;
    trunk = std::move(other.trunk);
    policyHead = std::move(other.policyHead);
    valueHead = std::move(other.valueHead);
    return *this;
  }
};


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

    if(version >= 3) {
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
    }
    else {
      xSize = desc->xSizePreV3;
      ySize = desc->ySizePreV3;

      if(xSize != NNPos::MAX_BOARD_LEN)
        throw StringError(Global::strprintf("For V2 models and lower xSize (%d) must be NNPos::MAX_BOARD_LEN (%d)",
          xSize, NNPos::MAX_BOARD_LEN
        ));
      if(ySize != NNPos::MAX_BOARD_LEN)
        throw StringError(Global::strprintf("For V2 models and lower ySize (%d) must be NNPos::MAX_BOARD_LEN (%d)",
          ySize, NNPos::MAX_BOARD_LEN
        ));
      if(nnXLen != xSize)
        throw StringError(Global::strprintf("For V2 models and lower nnXLen (%d) must match xSize (%d)",
          nnXLen, xSize
        ));
      if(nnYLen != ySize)
        throw StringError(Global::strprintf("For V2 models and lower nnYLen (%d) must match ySize (%d)",
          nnYLen, ySize
        ));
    }

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

    if(version >= 3) {
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
    }
    //Older versions need to set this to NULL, in particular various parts of the code use maskSumBuf being non-null
    //as an indicator to perform V3 operations.
    else {
      maskBuf = NULL;
      maskFloatBuf = NULL;
      maskSumBuf = NULL;
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

  LoadedModel(istream& in) {
    modelDesc = std::move(ModelDesc(in));
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, int modelFileIdx) {
  (void)modelFileIdx;

  try {
    //zstr has a bad property of simply aborting if the file doesn't exist
    //So we try to catch this common error by explicitly testing first if the file exists by trying to open it normally
    //to turn it into a regular C++ exception.
    {
      ifstream testIn(file);
      if(!testIn.good())
        throw StringError("File does not exist or could not be opened");
    }
    zstr::ifstream in(file);
    LoadedModel* loadedModel = new LoadedModel(in);
    return loadedModel;
  }
  catch(const StringError& e) {
    throw StringError("Error parsing model file " + file + ": " + e.what());
  }
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
    size_t batchXYFloatBytes = m.maxBatchSize * m.xSize * m.ySize * sizeof(float);
    size_t batchFloatBytes = m.maxBatchSize * sizeof(float);

    size_t batchXYBytes = m.maxBatchSize * m.xSize * m.ySize * (useFP16 ? sizeof(half) : sizeof(float));
    size_t batchBytes = m.maxBatchSize * (useFP16 ? sizeof(half) : sizeof(float));

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
    if(m.version >= 3)
    {CUDA_ERR("Buffers",cudaMalloc(&v1MeanBuf, m.valueHead->v1Channels * 3 * batchFloatBytes));}
    else
    {CUDA_ERR("Buffers",cudaMalloc(&v1MeanBuf, m.valueHead->v1Channels * batchFloatBytes));}
    CUDA_ERR("Buffers",cudaMalloc(&v2OutBuf, m.valueHead->v2Channels * batchFloatBytes));

    valueBufBytes = m.valueHead->valueChannels * batchFloatBytes;
    CUDA_ERR("Buffers",cudaMalloc(&valueBuf, valueBufBytes));

    if(m.version >= 3) {
      scoreValueBufBytes = m.valueHead->scoreValueChannels * batchFloatBytes;
      CUDA_ERR("Buffers",cudaMalloc(&scoreValueBuf, scoreValueBufBytes));

      //This buf is used for both an intermdiate fp16 result in fp16 mode, and ALSO the final fp32 output, so always must be fp32-sized
      ownershipBufBytes = m.valueHead->ownershipChannels * batchXYFloatBytes;
      CUDA_ERR("Buffers",cudaMalloc(&ownershipBuf, ownershipBufBytes));
      CUDA_ERR("Buffers",cudaMalloc(&ownershipScratchBuf, ownershipBufBytes));
    }
    else {
      scoreValueBuf = NULL;
      ownershipBuf = NULL;
      ownershipScratchBuf = NULL;
    }

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

struct LocalGpuHandle {
  CudaHandles* cudaHandles;
  Model* model;
  Buffers* buffers;
  bool usingFP16;
  int nnXLen;
  int nnYLen;
  bool requireExactNNLen;
  int policySize;

  LocalGpuHandle(const LoadedModel* loadedModel, int maxBatchSize, int xLen, int yLen, bool rExactNNLen, bool inputsUseNHWC, bool useFP16, bool useNHWC) {
    cudaHandles = new CudaHandles();
    model = new Model(cudaHandles,&(loadedModel->modelDesc),maxBatchSize,xLen,yLen,inputsUseNHWC,useFP16,useNHWC);
    buffers = new Buffers(cudaHandles,*model,useFP16);
    usingFP16 = useFP16;
    nnXLen = xLen;
    nnYLen = yLen;
    requireExactNNLen = rExactNNLen;
    policySize = NNPos::getPolicySize(nnXLen,nnYLen);

    //Synchronize after creating all the buffers and copying all the weights, just in case
    CUDA_ERR("LocalGpuHandle",cudaDeviceSynchronize());
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

LocalGpuHandle* NeuralNet::createLocalGpuHandle(
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
  CUDA_ERR("createLocalGpuHandle",cudaSetDevice(cudaDeviceIdxForThisThread));

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

  LocalGpuHandle* gpuHandle = new LocalGpuHandle(loadedModel,maxBatchSize,nnXLen,nnYLen,requireExactNNLen,inputsUseNHWC,cudaUseFP16,cudaUseNHWC);
  return gpuHandle;
}

void NeuralNet::freeLocalGpuHandle(LocalGpuHandle* gpuHandle) {
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

    int xSize = m.version >= 3 ? nnXLen : m.xSizePreV3;
    int ySize = m.version >= 3 ? nnYLen : m.ySizePreV3;

    maxBatchSize = maxBatchSz;
    singleInputElts = m.numInputChannels * xSize * ySize;
    singleInputBytes = m.numInputChannels * xSize * ySize * sizeof(float);
    singleInputGlobalElts = m.numInputGlobalChannels;
    singleInputGlobalBytes = m.numInputGlobalChannels * sizeof(float);
    singlePolicyResultElts = (1 + xSize * ySize);
    singlePolicyResultBytes = (1 + xSize * ySize) * sizeof(float);
    singleValueResultElts = m.numValueChannels;
    singleValueResultBytes = m.numValueChannels * sizeof(float);
    singleScoreValueResultElts = m.numScoreValueChannels;
    singleScoreValueResultBytes = m.numScoreValueChannels * sizeof(float);
    singleOwnershipResultElts = m.numOwnershipChannels * xSize * ySize;
    singleOwnershipResultBytes = m.numOwnershipChannels * xSize * ySize * sizeof(float);

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);
    if(m.version < 3)
      assert(NNModelVersion::getRowSize(m.version) == singleInputElts);

    userInputBufferBytes = m.numInputChannels * maxBatchSize * xSize * ySize * sizeof(float);
    userInputGlobalBufferBytes = m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    policyResultBufferBytes = maxBatchSize * (1 + xSize * ySize) * sizeof(float);
    valueResultBufferBytes = maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = maxBatchSize * xSize * ySize * m.numOwnershipChannels * sizeof(float);

    userInputBuffer = new float[m.numInputChannels * maxBatchSize * xSize * ySize];
    userInputGlobalBuffer = new float[m.numInputGlobalChannels * maxBatchSize];
    symmetriesBuffer = new bool[NNInputs::NUM_SYMMETRY_BOOLS];

    policyResults = new float[maxBatchSize * (1 + xSize * ySize)];
    valueResults = new float[maxBatchSize * m.numValueChannels];

    if(m.version >= 3) {
      scoreValueResults = new float[maxBatchSize * m.numScoreValueChannels];
      ownershipResults = new float[maxBatchSize * xSize * ySize * m.numOwnershipChannels];
    }
    else {
      scoreValueResults = NULL;
      ownershipResults = NULL;
    }
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

float* NeuralNet::getRowInplace(InputBuffers* inputBuffers, int rowIdx) {
  assert(rowIdx < inputBuffers->maxBatchSize);
  return inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * rowIdx);
}

float* NeuralNet::getRowGlobalInplace(InputBuffers* inputBuffers, int rowIdx) {
  assert(rowIdx < inputBuffers->maxBatchSize);
  return inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * rowIdx);
}

int NeuralNet::getRowLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputElts;
}
int NeuralNet::getRowGlobalLen(const InputBuffers* inputBuffers) {
  return inputBuffers->singleInputGlobalElts;
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* inputBuffers) {
  return inputBuffers->symmetriesBuffer;
}


//---------------------------------------------------------------------------------------


void NeuralNet::getOutput(LocalGpuHandle* gpuHandle, InputBuffers* inputBuffers, int numFilledRows, vector<NNOutput*>& outputs) {
  assert(numFilledRows <= inputBuffers->maxBatchSize);
  assert(numFilledRows > 0);
  int batchSize = numFilledRows;
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
    if(version >= 3) {
      assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
      assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
      assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
      assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));
    }

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice));
    if(version >= 3)
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
    if(version >= 3) {
      assert(inputBuffers->scoreValueResultBufferBytes == buffers->scoreValueBufBytes);
      assert(inputBuffers->ownershipResultBufferBytes == buffers->ownershipBufBytes);
      assert(inputBuffers->singleOwnershipResultElts == nnXLen*nnYLen);
      assert(inputBuffers->singleOwnershipResultBytes == nnXLen*nnYLen * sizeof(float));
    }

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBufFloat, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes*batchSize, cudaMemcpyHostToDevice));
    if(version >= 3)
      CUDA_ERR("getOutput",cudaMemcpy(buffers->inputGlobalBufFloat, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes*batchSize, cudaMemcpyHostToDevice));

    customCudaCopyToHalf((const float*)buffers->inputBufFloat,(half*)buffers->inputBuf,inputBuffers->singleInputElts*batchSize);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
    if(version >= 3) {
      customCudaCopyToHalf((const float*)buffers->inputGlobalBufFloat,(half*)buffers->inputGlobalBuf,inputBuffers->singleInputGlobalElts*batchSize);
      CUDA_ERR("getOutput",cudaPeekAtLastError());
    }
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
  if(version >= 3) {
    CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes*batchSize, cudaMemcpyDeviceToHost));
    CUDA_ERR("getOutput",cudaMemcpy(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes*batchSize, cudaMemcpyDeviceToHost));
  }

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

    if(version >= 4) {
      int numValueChannels = gpuHandle->model->numValueChannels;
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numValueChannels == 3);
      assert(numScoreValueChannels == 2);
      output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
      output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
      output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];

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

    }
    else if(version >= 3) {
      int numValueChannels = gpuHandle->model->numValueChannels;
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numValueChannels == 3);
      assert(numScoreValueChannels == 1);
      output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
      output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
      output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;

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

    }
    else {
      output->whiteWinProb = inputBuffers->valueResults[row];
      output->whiteLossProb = 0.0;
      output->whiteNoResultProb = 0.0;
      output->whiteScoreMean = 0.0;
      output->whiteScoreMeanSq = 0.0;

      //Older versions don't have an ownership map, so zero fill
      if(output->whiteOwnerMap != NULL)
        std::fill(output->whiteOwnerMap, output->whiteOwnerMap + nnXLen * nnYLen, 0.0f);

    }
  }

}

#endif  // USE_CUDA_BACKEND
