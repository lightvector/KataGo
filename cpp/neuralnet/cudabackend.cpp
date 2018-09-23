
#ifdef USE_CUDA_BACKEND

#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_fp16.h>

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

static void mallocAndCopyToDevice(const string& name, const vector<float>& weights, void*& deviceBuf, bool useFP16) {
  size_t numWeights = weights.size();
  if(useFP16) {
    size_t halfBytes = numWeights * sizeof(half);
    size_t singleBytes = numWeights * sizeof(float);
    float* buf;
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, halfBytes));
    CUDA_ERR(name.c_str(),cudaMalloc(&buf, singleBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(buf, weights.data(), singleBytes, cudaMemcpyHostToDevice));
    customCudaCopyToHalf(buf,(half*)deviceBuf,numWeights);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    CUDA_ERR(name.c_str(),cudaDeviceSynchronize());
    cudaFree(buf);
  }
  else {
    size_t singleBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, singleBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(deviceBuf, weights.data(), singleBytes, cudaMemcpyHostToDevice));
  }
}

static void mallocAndCopyToDevice(const string& name, float* weights, int numWeights, void*& deviceBuf, bool useFP16) {
  if(useFP16) {
    size_t halfBytes = numWeights * sizeof(half);
    size_t singleBytes = numWeights * sizeof(float);
    float* buf;
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, halfBytes));
    CUDA_ERR(name.c_str(),cudaMalloc(&buf, singleBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(buf, weights, singleBytes, cudaMemcpyHostToDevice));
    customCudaCopyToHalf(buf,(half*)deviceBuf,numWeights);
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    CUDA_ERR(name.c_str(),cudaDeviceSynchronize());
    cudaFree(buf);
  }
  else {
    size_t singleBytes = numWeights * sizeof(float);
    CUDA_ERR(name.c_str(),cudaMalloc(&deviceBuf, singleBytes));
    CUDA_ERR(name.c_str(),cudaMemcpy(deviceBuf, weights, singleBytes, cudaMemcpyHostToDevice));
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
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
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
    const cudnnTensorDescriptor_t& inputDescriptor,
    const cudnnTensorDescriptor_t& outputDescriptor,
    int batchSize,
    void* inputBuf,
    void* outputBuf
  ) const {
    if(!usingFP16) {
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
    else {
      if(!usingNHWC)
        customCudaApplyCScaleBiasNCHW((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,batchSize,numChannels,xSize*ySize);
      else
        customCudaApplyCScaleBiasNHWC((const half*)inputBuf,(half*)outputBuf,(const half*)mergedScaleBuf,(const half*)mergedBiasBuf,batchSize,xSize*ySize,numChannels);
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
    int trunkBufSize,
    void* trunkInBuf,
    void* trunkOutBuf,
    void* midInBuf,
    void* midScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    preBN.apply(cudaHandles,trunkDescriptor,trunkDescriptor,batchSize,trunkInBuf,trunkOutBuf);
    preActivation.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkOutBuf,trunkOutBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,midInDescriptor,batchSize,trunkOutBuf,midInBuf,workspaceBuf,workspaceBytes);
    midBN.apply(cudaHandles,midInDescriptor,midInDescriptor,batchSize,midInBuf,midScratchBuf);
    midActivation.apply(cudaHandles,midInDescriptor,midInDescriptor,midScratchBuf,midScratchBuf);
    finalConv.apply(cudaHandles,midInDescriptor,trunkDescriptor,batchSize,midScratchBuf,trunkOutBuf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      const float alpha = 1.0f;
      CUBLAS_ERR(name.c_str(),cublasSaxpy(cudaHandles->cublas,trunkBufSize,&alpha,(const float*)trunkInBuf,1,(float*)trunkOutBuf,1));
    }
    else {
      customCudaAddTensorInplace((half*)trunkOutBuf,(const half*)trunkInBuf,trunkBufSize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
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
    int trunkBufSize,
    void* trunkInBuf,
    void* trunkOutBuf,
    void* regularOutBuf,
    void* dilatedOutBuf,
    void* midInBuf,
    void* midScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    preBN.apply(cudaHandles,trunkDescriptor,trunkDescriptor,batchSize,trunkInBuf,trunkOutBuf);
    preActivation.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkOutBuf,trunkOutBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,regularOutDescriptor,batchSize,trunkOutBuf,regularOutBuf,workspaceBuf,workspaceBytes);
    dilatedConv.apply(cudaHandles,trunkDescriptor,dilatedOutDescriptor,batchSize,trunkOutBuf,dilatedOutBuf,workspaceBuf,workspaceBytes);
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
    midBN.apply(cudaHandles,midInDescriptor,midInDescriptor,batchSize,midInBuf,midScratchBuf);
    midActivation.apply(cudaHandles,midInDescriptor,midInDescriptor,midScratchBuf,midScratchBuf);
    finalConv.apply(cudaHandles,midInDescriptor,trunkDescriptor,batchSize,midScratchBuf,trunkOutBuf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      const float alpha = 1.0f;
      CUBLAS_ERR(name.c_str(),cublasSaxpy(cudaHandles->cublas,trunkBufSize,&alpha,(const float*)trunkInBuf,1,(float*)trunkOutBuf,1));
    }
    else {
      customCudaAddTensorInplace((half*)trunkOutBuf,(const half*)trunkInBuf,trunkBufSize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
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
    int trunkBufSize,
    void* trunkInBuf,
    void* trunkOutBuf,
    void* regularOutBuf,
    void* gpoolOutBuf,
    void* gpoolOutBuf2,
    float* gpoolMeanBufSingle,
    float* gpoolMaxBufSingle,
    void* gpoolMeanBuf,
    void* gpoolMaxBuf,
    void* gpoolConcatBuf,
    void* gpoolBiasBuf,
    void* regularScratchBuf,
    const void* zeroBuf,
    const void* oneBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    preBN.apply(cudaHandles,trunkDescriptor,trunkDescriptor,batchSize,trunkInBuf,trunkOutBuf);
    preActivation.apply(cudaHandles,trunkDescriptor,trunkDescriptor,trunkOutBuf,trunkOutBuf);
    regularConv.apply(cudaHandles,trunkDescriptor,regularOutDescriptor,batchSize,trunkOutBuf,regularOutBuf,workspaceBuf,workspaceBytes);
    gpoolConv.apply(cudaHandles,trunkDescriptor,gpoolOutDescriptor,batchSize,trunkOutBuf,gpoolOutBuf,workspaceBuf,workspaceBytes);
    gpoolBN.apply(cudaHandles,gpoolOutDescriptor,gpoolOutDescriptor,batchSize,gpoolOutBuf,gpoolOutBuf2);
    gpoolActivation.apply(cudaHandles,gpoolOutDescriptor,gpoolOutDescriptor,gpoolOutBuf2,gpoolOutBuf2);

    if(!usingFP16) {
      if(!usingNHWC) {
        customCudaPoolRowsSumNCHW((float*)gpoolOutBuf2,(float*)gpoolMeanBuf,batchSize*gpoolChannels,xSize*ySize);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        customCudaPoolRowsMaxNCHW((float*)gpoolOutBuf2,(float*)gpoolMaxBuf,batchSize*gpoolChannels,xSize*ySize);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      else {
        customCudaPoolRowsSumNHWC((const float*)gpoolOutBuf2,(float*)gpoolMeanBuf,batchSize,xSize*ySize,gpoolChannels);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        customCudaPoolRowsMaxNHWC((const float*)gpoolOutBuf2,(float*)gpoolMaxBuf,batchSize,xSize*ySize,gpoolChannels);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }

      const float meanScale = 1.0f / (xSize*ySize);
      CUBLAS_ERR(name.c_str(),cublasSscal(cudaHandles->cublas, batchSize*gpoolChannels, &meanScale, (float*)gpoolMeanBuf, 1));
    }
    else {
      customCudaCopyFromHalf((const half*)gpoolOutBuf2,(float*)workspaceBuf,batchSize*gpoolChannels*xSize*ySize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(!usingNHWC) {
        customCudaPoolRowsSumNCHW((float*)workspaceBuf,gpoolMeanBufSingle,batchSize*gpoolChannels,xSize*ySize);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        customCudaPoolRowsMaxNCHW((float*)workspaceBuf,gpoolMaxBufSingle,batchSize*gpoolChannels,xSize*ySize);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      else {
        customCudaPoolRowsSumNHWC((const float*)workspaceBuf,gpoolMeanBufSingle,batchSize,xSize*ySize,gpoolChannels);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        customCudaPoolRowsMaxNHWC((const float*)workspaceBuf,gpoolMaxBufSingle,batchSize,xSize*ySize,gpoolChannels);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      const float meanScale = 1.0f / (xSize*ySize);
      CUBLAS_ERR(name.c_str(),cublasSscal(cudaHandles->cublas, batchSize*gpoolChannels, &meanScale, gpoolMeanBufSingle, 1));
      customCudaCopyToHalf((const float*)gpoolMeanBufSingle,(half*)gpoolMeanBuf,batchSize*gpoolChannels);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      customCudaCopyToHalf((const float*)gpoolMaxBufSingle,(half*)gpoolMaxBuf,batchSize*gpoolChannels);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }

    // float* maxTmp = new float[5];
    // CUDA_ERR("DEBUG",cudaMemcpy(maxTmp, gpoolMaxBuf, sizeof(float)*5, cudaMemcpyDeviceToHost));
    // cout << name << " MAX ";
    // for(int i = 0; i<5; i++)
    //   cout << maxTmp[i] << " ";
    // cout << endl;
    // float* meanTmp = new float[5];
    // CUDA_ERR("DEBUG",cudaMemcpy(meanTmp, gpoolMeanBuf, sizeof(float)*5, cudaMemcpyDeviceToHost));
    // cout << name << " MEAN ";
    // for(int i = 0; i<5; i++)
    //   cout << meanTmp[i] << " ";
    // cout << endl;


    if(!usingFP16) {
      customCudaChannelConcat(
        (const float*)gpoolMeanBuf,(const float*)gpoolMaxBuf,(float*)gpoolConcatBuf,
        gpoolChannels,
        gpoolChannels,
        batchSize
      );
    }
    else {
      customCudaChannelConcat(
        (const half*)gpoolMeanBuf,(const half*)gpoolMaxBuf,(half*)gpoolConcatBuf,
        gpoolChannels,
        gpoolChannels,
        batchSize
      );
    }
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

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

    midBN.apply(cudaHandles,regularOutDescriptor,regularOutDescriptor,batchSize,regularOutBuf,regularScratchBuf);
    midActivation.apply(cudaHandles,regularOutDescriptor,regularOutDescriptor,regularScratchBuf,regularScratchBuf);
    finalConv.apply(cudaHandles,regularOutDescriptor,trunkDescriptor,batchSize,regularScratchBuf,trunkOutBuf,workspaceBuf,workspaceBytes);

    if(!usingFP16) {
      const float alpha = 1.0f;
      CUBLAS_ERR(name.c_str(),cublasSaxpy(cudaHandles->cublas,trunkBufSize,&alpha,(const float*)trunkInBuf,1,(float*)trunkOutBuf,1));
    }
    else {
      customCudaAddTensorInplace((half*)trunkOutBuf,(const half*)trunkInBuf,trunkBufSize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
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
    const cudnnTensorDescriptor_t* inputDescriptors,
    bool useFP16,
    bool useNHWC
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
    void* inputBuf,
    void* trunkScratchBuf,
    void* trunkOutBuf,
    void* regularOutBuf,
    void* dilatedOutBuf,
    void* midInBuf,
    void* midScratchBuf,
    void* gpoolOutBuf,
    void* gpoolOutBuf2,
    float* gpoolMeanBufSingle,
    float* gpoolMaxBufSingle,
    void* gpoolMeanBuf,
    void* gpoolMaxBuf,
    void* gpoolConcatBuf,
    void* gpoolBiasBuf,
    void* regularScratchBuf,
    const void* zeroBuf,
    const void* oneBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    void* currentTrunkBuf = trunkScratchBuf;
    void* nextTrunkBuf = trunkOutBuf;

    const cudnnTensorDescriptor_t& trunkDescriptor = trunkDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& regularOutDescriptor = regularOutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& gpoolOutDescriptor = gpoolOutDescriptors[batchSize-1];
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
          batchSize,
          trunkBufSize,
          currentTrunkBuf,
          nextTrunkBuf,
          regularOutBuf,
          gpoolOutBuf,
          gpoolOutBuf2,
          gpoolMeanBufSingle,
          gpoolMaxBufSingle,
          gpoolMeanBuf,
          gpoolMaxBuf,
          gpoolConcatBuf,
          gpoolBiasBuf,
          regularScratchBuf,
          zeroBuf,
          oneBuf,
          workspaceBuf,
          workspaceBytes
        );
        std::swap(currentTrunkBuf,nextTrunkBuf);
      }
      else {
        assert(false);
      }

    }

    trunkTipBN->apply(cudaHandles,trunkDescriptor,trunkDescriptor,batchSize,currentTrunkBuf,nextTrunkBuf);
    trunkTipActivation->apply(cudaHandles,trunkDescriptor,trunkDescriptor,nextTrunkBuf,trunkOutBuf);
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
    if(symmetriesBuffer[2])
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
    if(symmetriesBuffer[2])
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
    if(symmetriesBuffer[2])
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
    if(symmetriesBuffer[2])
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

  PolicyHeadDesc() {}

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
    if(gpoolToBiasMul.inChannels != g1BN.numChannels*2)
      throw StringError(name+Global::strprintf(
        ": gpoolToBiasMul.inChannels (%d) != g1BN.numChannels*2 (%d)", gpoolToBiasMul.inChannels, g1BN.numChannels*2
      ));
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
    void* trunkOutBuf,
    void* p1OutBuf,
    void* p1OutBuf2,
    void* g1OutBuf,
    void* g1OutBuf2,
    float* g1MeanBuf,
    float* g1MaxBuf,
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

    p1Conv->apply(cudaHandles,trunkDescriptor,p1OutDescriptor,batchSize,trunkOutBuf,p1OutBuf,workspaceBuf,workspaceBytes);
    g1Conv->apply(cudaHandles,trunkDescriptor,g1OutDescriptor,batchSize,trunkOutBuf,g1OutBuf,workspaceBuf,workspaceBytes);
    g1BN->apply(cudaHandles,g1OutDescriptor,g1OutDescriptor,batchSize,g1OutBuf,g1OutBuf2);
    g1Activation->apply(cudaHandles,g1OutDescriptor,g1OutDescriptor,g1OutBuf2,g1OutBuf2);

    if(!usingFP16) {
      if(!usingNHWC) {
        customCudaPoolRowsSumNCHW((float*)g1OutBuf2,g1MeanBuf,batchSize*g1Channels,xSize*ySize);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        customCudaPoolRowsMaxNCHW((float*)g1OutBuf2,g1MaxBuf,batchSize*g1Channels,xSize*ySize);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      else {
        customCudaPoolRowsSumNHWC((const float*)g1OutBuf2,g1MeanBuf,batchSize,xSize*ySize,g1Channels);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        customCudaPoolRowsMaxNHWC((const float*)g1OutBuf2,g1MaxBuf,batchSize,xSize*ySize,g1Channels);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
    }
    else {
      customCudaCopyFromHalf((const half*)g1OutBuf2,(float*)workspaceBuf,batchSize*g1Channels*xSize*ySize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(!usingNHWC) {
        customCudaPoolRowsSumNCHW((float*)workspaceBuf,g1MeanBuf,batchSize*g1Channels,xSize*ySize);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        customCudaPoolRowsMaxNCHW((float*)workspaceBuf,g1MaxBuf,batchSize*g1Channels,xSize*ySize);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
      else {
        customCudaPoolRowsSumNHWC((const float*)workspaceBuf,g1MeanBuf,batchSize,xSize*ySize,g1Channels);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
        customCudaPoolRowsMaxNHWC((const float*)workspaceBuf,g1MaxBuf,batchSize,xSize*ySize,g1Channels);
        CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      }
    }
    const float meanScale = 1.0f / (xSize*ySize);
    CUBLAS_ERR(name.c_str(),cublasSscal(cudaHandles->cublas, batchSize*g1Channels, &meanScale, g1MeanBuf, 1));

    customCudaChannelConcat(
      g1MeanBuf,g1MaxBuf,g1ConcatBuf,
      g1Channels,
      g1Channels,
      batchSize
    );
    CUDA_ERR(name.c_str(),cudaPeekAtLastError());

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

    p1BN->apply(cudaHandles,p2InDescriptor,p2InDescriptor,batchSize,p1OutBufA,p1OutBufB);

    if(version >= 1) {
      p1Activation->apply(cudaHandles,p2InDescriptor,p2InDescriptor,p1OutBufB,p1OutBufB);
      p2Conv->apply(cudaHandles,p2InDescriptor,p2OutDescriptor,batchSize,p1OutBufB,p2OutBuf,workspaceBuf,workspaceBytes);
    }
    else {
      throw StringError("Version 0 neural nets no longer supported in cudnn");
    }

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

  ValueHeadDesc() {}

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
    version = other.version;
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
  int version;
  int maxBatchSize;
  int xSize;
  int ySize;
  int v1Channels;
  int v2Channels;
  int valueChannels;
  bool usingFP16;
  bool usingNHWC;

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
    usingFP16 = useFP16;
    usingNHWC = useNHWC;

    v1OutDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];
    v3InDescriptors = new cudnnTensorDescriptor_t[maxBatchSize];

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
    }

    v1Conv = new ConvLayer(cudaHandles,&desc->v1Conv,maxBatchSize,trunkDescriptors,v1OutDescriptors,useFP16,useNHWC);
    v1BN = new BNLayer(cudaHandles,&desc->v1BN,xSize,ySize,useFP16,useNHWC);
    v1Activation = new ActivationLayer(cudaHandles,&desc->v1Activation);
    v2Mul = new MatMulLayer(cudaHandles,&desc->v2Mul,false);
    v2Bias = new MatBiasLayer(cudaHandles,&desc->v2Bias,false);
    v2Activation = new ActivationLayer(cudaHandles,&desc->v2Activation);
    v3Mul = new MatMulLayer(cudaHandles,&desc->v3Mul,false);
    v3Bias = new MatBiasLayer(cudaHandles,&desc->v3Bias,false);
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
    b = sizeof(float)*batchSize*v1Channels*xSize*ySize;
    bytes = std::max(bytes,b);

    return bytes;
  }


  void apply(
    CudaHandles* cudaHandles,
    const cudnnTensorDescriptor_t& trunkDescriptor,
    int batchSize,
    void* trunkOutBuf,
    void* v1OutBuf,
    void* v1OutBuf2,
    float* v1MeanBuf,
    float* v2OutBuf,
    float* valueBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const cudnnTensorDescriptor_t& v1OutDescriptor = v1OutDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& v3InDescriptor = v3InDescriptors[batchSize-1];

    v1Conv->apply(cudaHandles,trunkDescriptor,v1OutDescriptor,batchSize,trunkOutBuf,v1OutBuf,workspaceBuf,workspaceBytes);
    v1BN->apply(cudaHandles,v1OutDescriptor,v1OutDescriptor,batchSize,v1OutBuf,v1OutBuf2);
    v1Activation->apply(cudaHandles,v1OutDescriptor,v1OutDescriptor,v1OutBuf2,v1OutBuf2);

    if(!usingFP16) {
      if(!usingNHWC)
        customCudaPoolRowsSumNCHW((float*)v1OutBuf2,v1MeanBuf,batchSize*v1Channels,xSize*ySize);
      else
        customCudaPoolRowsSumNHWC((const float*)v1OutBuf2,v1MeanBuf,batchSize,xSize*ySize,v1Channels);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    else {
      customCudaCopyFromHalf((const half*)v1OutBuf2,(float*)workspaceBuf,batchSize*v1Channels*xSize*ySize);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
      if(!usingNHWC)
        customCudaPoolRowsSumNCHW((float*)workspaceBuf,v1MeanBuf,batchSize*v1Channels,xSize*ySize);
      else
        customCudaPoolRowsSumNHWC((const float*)workspaceBuf,v1MeanBuf,batchSize,xSize*ySize,v1Channels);
      CUDA_ERR(name.c_str(),cudaPeekAtLastError());
    }
    const float meanScale = 1.0f / (xSize*ySize);
    CUBLAS_ERR(name.c_str(),cublasSscal(cudaHandles->cublas, batchSize*v1Channels, &meanScale, v1MeanBuf, 1));

    float zero = 0.0f;
    float one = 1.0f;
    v2Mul->apply(cudaHandles,batchSize,v1MeanBuf,v2OutBuf,&zero,&one,workspaceBuf,workspaceBytes);
    v2Bias->apply(cudaHandles,batchSize,v2OutBuf);

    if(version >= 1) {
      v2Activation->apply(cudaHandles,v3InDescriptor,v3InDescriptor,v2OutBuf,v2OutBuf);
      v3Mul->apply(cudaHandles,batchSize,v2OutBuf,valueBuf,&zero,&one,workspaceBuf,workspaceBytes);
    }
    else {
      throw StringError("Version 0 neural nets no longer supported in cudnn");
    }
    v3Bias->apply(cudaHandles,batchSize,valueBuf);
  }

};


//------------------------------------------------------------------------------

struct ModelDesc {
  string name;
  int version;
  int xSize;
  int ySize;
  int numInputChannels;

  TrunkDesc trunk;
  PolicyHeadDesc policyHead;
  ValueHeadDesc valueHead;

  ModelDesc() {}

  ModelDesc(istream& in) {
    in >> name;
    in >> version;
    in >> xSize;
    in >> ySize;
    in >> numInputChannels;

    if(in.fail())
      throw StringError(name + ": model failed to parse name or xSize or ySize");
    if(xSize <= 0 || ySize <= 0)
      throw StringError(name + ": model xSize and ySize must be positive");
    if(numInputChannels <= 0)
      throw StringError(name + ": model numInputChannels must be positive");

    if(version < 0 || version > 1)
      throw StringError(name + ": model found unsupported version " + Global::intToString(version));

    trunk = TrunkDesc(in);
    policyHead = PolicyHeadDesc(in,version);
    valueHead = ValueHeadDesc(in,version);

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
    version = other.version;
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
  bool usingFP16;

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
    bool useFP16,
    bool useNHWC
  ) {
    name = desc->name;
    maxBatchSize = maxBatchSz;
    xSize = desc->xSize;
    ySize = desc->ySize;
    numInputChannels = desc->numInputChannels;
    usingFP16 = useFP16;

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
        CUDNN_TENSOR_NHWC, //Always NHWC since this is the tensor that we receive initial user input from
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
    bool* symmetriesBuffer,

    void* inputBuf,
    void* inputScratchBuf,
    void* trunkScratchBuf,
    void* trunkOutBuf,
    void* regularOutBuf,
    void* dilatedOutBuf,
    void* midInBuf,
    void* midScratchBuf,
    void* gpoolOutBuf,
    void* gpoolOutBuf2,
    float* gpoolMeanBufSingle,
    float* gpoolMaxBufSingle,
    void* gpoolMeanBuf,
    void* gpoolMaxBuf,
    void* gpoolConcatBuf,
    void* gpoolBiasBuf,
    void* regularScratchBuf,

    void* p1OutBuf,
    void* p1OutBuf2,
    void* g1OutBuf,
    void* g1OutBuf2,
    float* g1MeanBuf,
    float* g1MaxBuf,
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

    const void* zeroBuf,
    const void* oneBuf,

    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    const cudnnTensorDescriptor_t& inputDescriptor = inputDescriptors[batchSize-1];
    const cudnnTensorDescriptor_t& trunkDescriptor = trunk->trunkDescriptors[batchSize-1];

    if(!usingFP16) {
      bool inverse = false;
      applySymmetriesNHWC<float>(symmetriesBuffer, inverse, batchSize, numInputChannels, xSize, ySize, (float*)inputBuf, (float*)inputScratchBuf);
    }
    else {
      bool inverse = false;
      applySymmetriesNHWC<half>(symmetriesBuffer, inverse, batchSize, numInputChannels, xSize, ySize, (half*)inputBuf, (half*)inputScratchBuf);
    }

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
      gpoolMeanBufSingle,
      gpoolMaxBufSingle,
      gpoolMeanBuf,
      gpoolMaxBuf,
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
      trunkDescriptor,
      batchSize,
      trunkOutBuf,
      v1OutBuf,
      v1OutBuf2,
      v1MeanBuf,
      v2OutBuf,
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
  try {
    LoadedModel* loadedModel = new LoadedModel(in);
    in.close();
    return loadedModel;
  }
  catch(const StringError& e) {
    throw StringError("Error parsing model file " + file + ": " + e.what());
  }
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}


//------------------------------------------------------------------------------

struct Buffers {
  //All of these are device pointers

  float* inputBufSingle;
  void* inputBuf;
  void* inputScratchBuf;
  size_t inputBufBytesSingle;
  size_t inputBufBytes;

  void* trunkScratchBuf;
  void* trunkOutBuf;
  void* regularOutBuf;
  void* dilatedOutBuf;
  void* midInBuf;
  void* midScratchBuf;
  void* gpoolOutBuf;
  void* gpoolOutBuf2;
  float* gpoolMeanBufSingle;
  float* gpoolMaxBufSingle;
  void* gpoolMeanBuf;
  void* gpoolMaxBuf;
  void* gpoolConcatBuf;
  void* gpoolBiasBuf;
  void* regularScratchBuf;

  void* p1OutBuf;
  void* p1OutBuf2;
  void* g1OutBuf;
  void* g1OutBuf2;
  float* g1MeanBuf;
  float* g1MaxBuf;
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

  void* zeroBuf;
  void* oneBuf;

  void* workspaceBuf;
  size_t workspaceBytes;

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;

  Buffers(CudaHandles* cudaHandles, const Model& m, bool useFP16) {
    size_t batchXYSingleBytes = m.maxBatchSize * m.xSize * m.ySize * sizeof(float);
    size_t batchSingleBytes = m.maxBatchSize * sizeof(float);

    size_t batchXYBytes = m.maxBatchSize * m.xSize * m.ySize * (useFP16 ? sizeof(half) : sizeof(float));
    size_t batchBytes = m.maxBatchSize * (useFP16 ? sizeof(half) : sizeof(float));

    inputBufBytesSingle = m.numInputChannels * batchXYSingleBytes;
    inputBufBytes = m.numInputChannels * batchXYBytes;
    CUDA_ERR("Buffers",cudaMalloc(&inputBufSingle, inputBufBytesSingle));
    CUDA_ERR("Buffers",cudaMalloc(&inputBuf, inputBufBytes));
    CUDA_ERR("Buffers",cudaMalloc(&inputScratchBuf, inputBufBytes));

    CUDA_ERR("Buffers",cudaMalloc(&trunkScratchBuf, m.trunk->trunkNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&trunkOutBuf, m.trunk->trunkNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&regularOutBuf, m.trunk->regularNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&dilatedOutBuf, m.trunk->dilatedNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&midInBuf, (m.trunk->regularNumChannels + m.trunk->dilatedNumChannels) * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&midScratchBuf, (m.trunk->regularNumChannels + m.trunk->dilatedNumChannels) * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolOutBuf, m.trunk->gpoolNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolOutBuf2, m.trunk->gpoolNumChannels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolMeanBufSingle, m.trunk->gpoolNumChannels * batchSingleBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolMaxBufSingle, m.trunk->gpoolNumChannels * batchSingleBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolMeanBuf, m.trunk->gpoolNumChannels * batchBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolMaxBuf, m.trunk->gpoolNumChannels * batchBytes));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolConcatBuf, m.trunk->gpoolNumChannels * batchBytes * 2));
    CUDA_ERR("Buffers",cudaMalloc(&gpoolBiasBuf, m.trunk->regularNumChannels * batchBytes));
    CUDA_ERR("Buffers",cudaMalloc(&regularScratchBuf, m.trunk->regularNumChannels * batchXYBytes));

    CUDA_ERR("Buffers",cudaMalloc(&p1OutBuf, m.policyHead->p1Channels * batchXYSingleBytes)); //need to hold floats in addition to halfs
    CUDA_ERR("Buffers",cudaMalloc(&p1OutBuf2, m.policyHead->p1Channels * batchXYSingleBytes)); //need to hold floats in addition to halfs
    CUDA_ERR("Buffers",cudaMalloc(&g1OutBuf, m.policyHead->g1Channels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&g1OutBuf2, m.policyHead->g1Channels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&g1MeanBuf, m.policyHead->g1Channels * batchSingleBytes));
    CUDA_ERR("Buffers",cudaMalloc(&g1MaxBuf, m.policyHead->g1Channels * batchSingleBytes));
    CUDA_ERR("Buffers",cudaMalloc(&g1ConcatBuf, m.policyHead->g1Channels * batchSingleBytes * 2));
    CUDA_ERR("Buffers",cudaMalloc(&g1BiasBuf, m.policyHead->p1Channels * batchSingleBytes));
    CUDA_ERR("Buffers",cudaMalloc(&p2OutBuf, m.policyHead->p2Channels * batchXYSingleBytes));
    CUDA_ERR("Buffers",cudaMalloc(&g1PassBuf, m.policyHead->p2Channels * batchSingleBytes));

    policyBufBytes = m.policyHead->p2Channels * (batchXYSingleBytes + batchSingleBytes);
    CUDA_ERR("Buffers",cudaMalloc(&policyBuf, policyBufBytes));
    assert(m.policyHead->p2Channels == 1);

    CUDA_ERR("Buffers",cudaMalloc(&v1OutBuf, m.valueHead->v1Channels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&v1OutBuf2, m.valueHead->v1Channels * batchXYBytes));
    CUDA_ERR("Buffers",cudaMalloc(&v1MeanBuf, m.valueHead->v1Channels * batchSingleBytes));
    CUDA_ERR("Buffers",cudaMalloc(&v2OutBuf, m.valueHead->v2Channels * batchSingleBytes));

    valueBufBytes = m.valueHead->valueChannels * batchSingleBytes;
    CUDA_ERR("Buffers",cudaMalloc(&valueBuf, valueBufBytes));

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
    cudaFree(inputBufSingle);
    cudaFree(inputBuf);
    cudaFree(inputScratchBuf);
    cudaFree(trunkScratchBuf);
    cudaFree(trunkOutBuf);
    cudaFree(regularOutBuf);
    cudaFree(dilatedOutBuf);
    cudaFree(midInBuf);
    cudaFree(midScratchBuf);
    cudaFree(gpoolOutBuf);
    cudaFree(gpoolOutBuf2);
    cudaFree(gpoolMeanBufSingle);
    cudaFree(gpoolMaxBufSingle);
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
    cudaFree(p2OutBuf);
    cudaFree(g1PassBuf);
    cudaFree(policyBuf);

    cudaFree(v1OutBuf);
    cudaFree(v1OutBuf2);
    cudaFree(v1MeanBuf);
    cudaFree(v2OutBuf);
    cudaFree(valueBuf);

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

  LocalGpuHandle(const LoadedModel* loadedModel, int maxBatchSize, bool useFP16, bool useNHWC) {
    cudaHandles = new CudaHandles();
    model = new Model(cudaHandles,&(loadedModel->modelDesc),maxBatchSize,useFP16,useNHWC);
    buffers = new Buffers(cudaHandles,*model,useFP16);
    usingFP16 = useFP16;

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
  }
  if(cudaUseFP16 && (prop.major < 5 || (prop.major == 5 && prop.minor < 3)))
    throw new StringError("Cuda device versions below 5.3 do not support cudaUseFP16=true");

  LocalGpuHandle* gpuHandle = new LocalGpuHandle(loadedModel,maxBatchSize,cudaUseFP16,cudaUseNHWC);
  return gpuHandle;
}

void NeuralNet::freeLocalGpuHandle(LocalGpuHandle* gpuHandle) {
  delete gpuHandle;
}

//------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleBatchItemElts;
  size_t singleBatchItemBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
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
    singleBatchItemElts = m.numInputChannels * m.xSize * m.ySize;
    singleBatchItemBytes = m.numInputChannels * m.xSize * m.ySize * sizeof(float);
    singlePolicyResultElts = (1 + m.xSize * m.ySize);
    singlePolicyResultBytes = (1 + m.xSize * m.ySize) * sizeof(float);
    singleValueResultElts = 1;
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
  return buffers->userInputBuffer + (buffers->singleBatchItemElts * rowIdx);
}

bool* NeuralNet::getSymmetriesInplace(InputBuffers* buffers) {
  return buffers->symmetriesBuffer;
}


//---------------------------------------------------------------------------------------


void NeuralNet::getOutput(LocalGpuHandle* gpuHandle, InputBuffers* inputBuffers, int numFilledRows, vector<NNOutput*>& outputs) {
  assert(numFilledRows <= inputBuffers->maxBatchSize);
  assert(numFilledRows > 0);
  int batchSize = numFilledRows;
  Buffers* buffers = gpuHandle->buffers;

  if(!gpuHandle->usingFP16) {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->singleBatchItemBytes == inputBuffers->singleBatchItemElts*4);
    assert(inputBuffers->singlePolicyResultElts == NNPos::NN_POLICY_SIZE);
    assert(inputBuffers->singlePolicyResultBytes == NNPos::NN_POLICY_SIZE * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleBatchItemBytes*batchSize, cudaMemcpyHostToDevice));
  }
  else {
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytesSingle);
    assert(inputBuffers->policyResultBufferBytes == buffers->policyBufBytes);
    assert(inputBuffers->valueResultBufferBytes == buffers->valueBufBytes);
    assert(inputBuffers->userInputBufferBytes == buffers->inputBufBytes*2);
    assert(inputBuffers->singleBatchItemBytes == inputBuffers->singleBatchItemElts*4);
    assert(inputBuffers->singlePolicyResultElts == NNPos::NN_POLICY_SIZE);
    assert(inputBuffers->singlePolicyResultBytes == NNPos::NN_POLICY_SIZE * sizeof(float));

    CUDA_ERR("getOutput",cudaMemcpy(buffers->inputBufSingle, inputBuffers->userInputBuffer, inputBuffers->singleBatchItemBytes*batchSize, cudaMemcpyHostToDevice));
    customCudaCopyToHalf((const float*)buffers->inputBufSingle,(half*)buffers->inputBuf,inputBuffers->singleBatchItemElts*batchSize);
    CUDA_ERR("getOutput",cudaPeekAtLastError());
  }

  gpuHandle->model->apply(
    gpuHandle->cudaHandles,
    batchSize,
    inputBuffers->symmetriesBuffer,

    buffers->inputBuf,
    buffers->inputScratchBuf,
    buffers->trunkScratchBuf,
    buffers->trunkOutBuf,
    buffers->regularOutBuf,
    buffers->dilatedOutBuf,
    buffers->midInBuf,
    buffers->midScratchBuf,
    buffers->gpoolOutBuf,
    buffers->gpoolOutBuf2,
    buffers->gpoolMeanBufSingle,
    buffers->gpoolMaxBufSingle,
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
    buffers->p2OutBuf,
    buffers->g1PassBuf,
    buffers->policyBuf,

    buffers->v1OutBuf,
    buffers->v1OutBuf2,
    buffers->v1MeanBuf,
    buffers->v2OutBuf,
    buffers->valueBuf,

    buffers->zeroBuf,
    buffers->oneBuf,

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
