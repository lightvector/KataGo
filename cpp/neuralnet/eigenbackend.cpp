#ifdef USE_EIGEN_BACKEND

/** Eigen3 backend.
 *
 * Only supports float32 computation with NHWC memory layout (at runtime and as input).
 */

//TODO someday - not sure how to make thread pool work with TensorMap. It works with Tensor, but TensorMap doesn't seem to have a device(...) method.
//#define EIGEN_USE_THREADS

#include "../neuralnet/nninterface.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <zstr/src/zstr.hpp>

#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"

using namespace std;
using Eigen::Tensor;
using Eigen::TensorMap;

//Eigen doesn't seem to have a way to make a const tensor map out of a const float* ??
//So we have to cast away qualifiers to build it.
#pragma GCC diagnostic ignored "-Wcast-qual"

// Eigen tensors are stored in column-major order, so an NHWC memory layout is given by Tensor<4>(C,W,H,N).

#define SCALAR float
#define TENSOR2 Tensor<SCALAR, 2>
#define TENSOR3 Tensor<SCALAR, 3>
#define TENSOR4 Tensor<SCALAR, 4>
#define TENSORMAP2 TensorMap<Tensor<SCALAR, 2>>
#define TENSORMAP3 TensorMap<Tensor<SCALAR, 3>>
#define TENSORMAP4 TensorMap<Tensor<SCALAR, 4>>

#define CONSTTENSOR2 const Tensor<SCALAR, 2>
#define CONSTTENSOR3 const Tensor<SCALAR, 3>
#define CONSTTENSOR4 const Tensor<SCALAR, 4>
#define CONSTTENSORMAP2 const TensorMap<Tensor<SCALAR, 2>>
#define CONSTTENSORMAP3 const TensorMap<Tensor<SCALAR, 3>>
#define CONSTTENSORMAP4 const TensorMap<Tensor<SCALAR, 4>>


// Debugging -----------------------------------------------------------------------------------------------------------
// #define DEBUG true

template <typename T>
void printTensorShape(const string& name, const T* t) {
  auto d = t->dimensions();
  cout << name << " rank=" << d.size() << " - (";
  for (int i = 0; i < d.size(); i++) {
    cout << d[i] << ",";
  }
  cout << ")" << endl;
}

#if DEBUG
#define DSHAPE(n, x) printTensorShape(n,x)
#define DTENSOR(n, x) cout << n << *x << endl
#else
#define DSHAPE(n, x)
#define DTENSOR(n, x)
#endif

// LoadedModel / ModelDesc ---------------------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName) {
    ModelDesc::loadFromFileMaybeGZipped(fileName,modelDesc);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file) {
  LoadedModel* loadedModel = new LoadedModel(file);
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


// Helpers --------------------------------------------------------------------------------------------------------------

static void computeMaskSum(CONSTTENSORMAP3* mask, float* maskSum) {
  for (int n = 0; n < mask->dimension(2); n++) {
    float s = 0.f;
    for (int h = 0; h < mask->dimension(1); h++) {
      for (int w = 0; w < mask->dimension(0); w++) {
        s += (*mask)(w, h, n);
      }
    }
    maskSum[n] = s;
  }
}

// in NxHxWxC, bias NxC
static void addNCBiasInplace(TENSORMAP4* in, CONSTTENSORMAP2* bias) {
  assert(in->dimension(0) == bias->dimension(0) && in->dimension(3) == bias->dimension(1));
  for (int n = 0; n < in->dimension(3); n++) {
    for (int h = 0; h < in->dimension(2); h++) {
      for (int w = 0; w < in->dimension(1); w++) {
        for (int c = 0; c < in->dimension(0); c++) {
          (*in)(c,w,h,n) += (*bias)(c,n);
        }
      }
    }
  }
}

static void poolRowsGPool(CONSTTENSORMAP4* in, TENSORMAP2* out, const float* maskSum) {
  for (int n = 0; n < in->dimension(3); n++) {
    for (int c = 0; c < in->dimension(0); c++) {
      float s = 0.f;
      float m = 0.f;
      for (int h = 0; h < in->dimension(2); h++) {
        for (int w = 0; w < in->dimension(1); w++) {
          float x = (*in)(c, w, h, n);
          s += x;
          m = max(m, x);
        }
      }
      float div = maskSum[n];
      float sqrtdiv = sqrt(div);
      float mean = s / div;
      (*out)(c, n) = mean;
      (*out)(c + in->dimension(0), n) = mean * (sqrtdiv - 14.f) * 0.1f;
      (*out)(c + 2*in->dimension(0), n) = m;
    }
  }
}

static void poolRowsValueHead(CONSTTENSORMAP4* in, TENSORMAP2* out, const float* maskSum) {
  for (int n = 0; n < in->dimension(3); n++) {
    for (int c = 0; c < in->dimension(0); c++) {
      float s = 0.f;
      for (int h = 0; h < in->dimension(2); h++) {
        for (int w = 0; w < in->dimension(1); w++) {
          float x = (*in)(c, w, h, n);
          s += x;
        }
      }
      float div = maskSum[n];
      float sqrtdiv = sqrt(div);
      float mean = s / div;
      (*out)(c, n) = mean;
      (*out)(c + in->dimension(0), n) = mean * (sqrtdiv - 14.f) * 0.1f;
      (*out)(c + 2*in->dimension(0), n) = mean * ((sqrtdiv - 14.0f) * (sqrtdiv - 14.0f) * 0.01f - 0.1f);
    }
  }
}

static size_t roundUpToMultiple(size_t size, size_t ofThis) {
  return (size + ofThis - 1) / ofThis * ofThis;
}

// --------------------------------------------------------------------------------------------------------------

struct ComputeHandleInternal {
  //static constexpr int numEigenThreads = 2;
  //Eigen::ThreadPool threadPool;
  //Eigen::ThreadPoolDevice device;

  ComputeHandleInternal()
  {
  }
};

// Layers --------------------------------------------------------------------------------------------------------------

// Convolution layer with zero-padding.
struct ConvLayer {
  string name;

  TENSOR2 imagePatchKernel;
  TENSOR3 winogradKernel;
  int inChannels;
  int outChannels;

  int convYSize;
  int convXSize;
  int nnXLen;
  int nnYLen;

  int imagePatchSize;

  int numTilesX;
  int numTilesY;
  int inTileXYSize;
  int outTileXYSize;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(const ConvLayerDesc& desc, int nnX, int nnY) {
    name = desc.name;
    convYSize = desc.convYSize;
    convXSize = desc.convXSize;
    inChannels = desc.inChannels;
    outChannels = desc.outChannels;
    //Currently eigen impl doesn't support dilated convs
    int dilationY = desc.dilationY;
    int dilationX = desc.dilationX;

    if(dilationX != 1 || dilationY != 1)
      throw StringError("Eigen backend: Encountered convolution dilation factors other than 1, not supported");

    assert(convXSize % 2 == 1);
    assert(convYSize % 2 == 1);

    nnXLen = nnX;
    nnYLen = nnY;

    if((convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)) {
      imagePatchSize = 0; //not used in this branch

      const int inTileXSize = 6;
      const int inTileYSize = 6;
      const int outTileXSize = convXSize == 5 ? 2 : 4;
      const int outTileYSize = convYSize == 5 ? 2 : 4;

      numTilesX = (nnXLen + outTileXSize - 1) / outTileXSize;
      numTilesY = (nnYLen + outTileYSize - 1) / outTileYSize;
      inTileXYSize = inTileXSize * inTileYSize;
      outTileXYSize = outTileXSize * outTileYSize;

      static constexpr int maxTileXSize = 6;
      static constexpr int maxTileYSize = 6;

      //INTILE_YSIZE, INTILE_XSIZE, ic, oc
      vector<float> transWeights(inTileXYSize * inChannels * outChannels);
      auto transform3x3_6 = [](float& a0, float& a1, float& a2, float& a3, float& a4, float& a5) {
        float z0 = a0; float z1 = a1; float z2 = a2;
        a0 = 0.25f * z0;
        a1 = (float)( (1.0 / 6.0) * (-z0 - z1 - z2) );
        a2 = (float)( (1.0 / 6.0) * (-z0 + z1 - z2) );
        a3 = (float)( (1.0 / 24.0) * (z0 + 2.0*z1 + 4.0*z2) );
        a4 = (float)( (1.0 / 24.0) * (z0 - 2.0*z1 + 4.0*z2) );
        a5 = 1.0f * z2;
      };
      auto transform5x5_6 = [](float& a0, float& a1, float& a2, float& a3, float& a4, float& a5) {
        float z0 = a0; float z1 = a1; float z2 = a2; float z3 = a3; float z4 = a4;
        a0 = 0.25f * z0;
        a1 = (float)( (1.0 / 6.0) * (-z0 - z1 - z2 - z3 - z4) );
        a2 = (float)( (1.0 / 6.0) * (-z0 + z1 - z2 + z3 - z4) );
        a3 = (float)( (1.0 / 24.0) * (z0 + 2.0*z1 + 4.0*z2 + 8.0*z3 + 16.0*z4) );
        a4 = (float)( (1.0 / 24.0) * (z0 - 2.0*z1 + 4.0*z2 - 8.0*z3 + 16.0*z4) );
        a5 = 1.0f * z4;
      };

      for(int oc = 0; oc < outChannels; oc++) {
        for(int ic = 0; ic < inChannels; ic++) {
          float tmp[maxTileYSize][maxTileXSize];
          for(int subY = 0; subY < convYSize; subY++) {
            for(int subX = 0; subX < convXSize; subX++) {
              if(oc < outChannels && ic < inChannels)
                tmp[subY][subX] = desc.weights[((oc * inChannels + ic) * convYSize + subY) * convXSize + subX];
              else
                tmp[subY][subX] = 0.0f;
            }
          }

          if(convXSize == 3) {
            for(int subY = 0; subY < convYSize; subY++)
              transform3x3_6(tmp[subY][0], tmp[subY][1], tmp[subY][2], tmp[subY][3], tmp[subY][4], tmp[subY][5]);
          }
          else if(convXSize == 5) {
            for(int subY = 0; subY < convYSize; subY++)
              transform5x5_6(tmp[subY][0], tmp[subY][1], tmp[subY][2], tmp[subY][3], tmp[subY][4], tmp[subY][5]);
          }

          if(convYSize == 3) {
            for(int subX = 0; subX < inTileXSize; subX++)
              transform3x3_6(tmp[0][subX], tmp[1][subX], tmp[2][subX], tmp[3][subX], tmp[4][subX], tmp[5][subX]);
          }
          else if(convYSize == 5) {
            for(int subX = 0; subX < inTileXSize; subX++)
              transform5x5_6(tmp[0][subX], tmp[1][subX], tmp[2][subX], tmp[3][subX], tmp[4][subX], tmp[5][subX]);
          }

          for(int subY = 0; subY < inTileYSize; subY++) {
            for(int subX = 0; subX < inTileXSize; subX++) {
              transWeights[((subY*inTileXSize + subX)*inChannels + ic)*outChannels + oc] = tmp[subY][subX];
            }
          }
        }
      }

      winogradKernel = TensorMap<const Tensor<const SCALAR, 3>>(
        transWeights.data(), outChannels, inChannels, inTileXSize * inTileYSize);
    }

    else {
      numTilesX = 0; //not used in this branch
      numTilesY = 0; //not used in this branch
      inTileXYSize = 0; //not used in this branch
      outTileXYSize = 0; //not used in this branch

      TENSOR4 kernel = TensorMap<const Tensor<const SCALAR, 4>>(desc.weights.data(), convXSize, convYSize, inChannels, outChannels);
      imagePatchSize = convXSize * convYSize * inChannels;
      Eigen::array<Eigen::Index, 4> dimensionPermutatation = {3, 2, 0, 1};
      Eigen::array<Eigen::Index, 2> newShape = {outChannels, imagePatchSize};
      imagePatchKernel = kernel.shuffle(dimensionPermutatation).reshape(newShape);
    }
  }

  size_t requiredConvWorkspaceElts(size_t maxBatchSize) const {
    if((convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)) {
      constexpr int inTileXSize = 6;
      constexpr int inTileYSize = 6;
      size_t totalChannelsRounded = roundUpToMultiple(inChannels,32) + roundUpToMultiple(outChannels,32);
      size_t sizeForTransforms = totalChannelsRounded * maxBatchSize * numTilesY * numTilesX * inTileXSize * inTileYSize;
      size_t sizeForTileBufs = 2 * inTileXSize * inTileYSize * roundUpToMultiple(std::max(inChannels,outChannels),32);
      return sizeForTransforms + sizeForTileBufs;
    }
    return 0;
  }

  void apply(ComputeHandleInternal* handle, CONSTTENSORMAP4* input, TENSORMAP4* output, float* convWorkspace, bool accumulate) const {
    (void)handle;
    assert(output->dimension(0) == outChannels);
    assert(input->dimension(0) == inChannels);
    assert(input->dimension(1) == nnXLen);
    assert(input->dimension(2) == nnYLen);
    const int batchSize = input->dimension(3);
    const int xSize = nnXLen;
    const int ySize = nnYLen;

    if((convXSize == 3 && convYSize == 3) || (convXSize == 5 && convYSize == 5)) {
      constexpr int inTileXSize = 6;
      constexpr int inTileYSize = 6;
      const int inTileXOffset = convXSize == 5 ? -2 : -1;
      const int inTileYOffset = convYSize == 5 ? -2 : -1;
      const int outTileXSize = convXSize == 5 ? 2 : 4;
      const int outTileYSize = convYSize == 5 ? 2 : 4;

      float* tile = convWorkspace;
      float* tile2 = tile + inTileXSize * inTileYSize * roundUpToMultiple(std::max(inChannels,outChannels),32);
      float* convWorkspaceIn = tile2 + inTileXSize * inTileYSize * roundUpToMultiple(std::max(inChannels,outChannels),32);
      float* convWorkspaceOut = convWorkspaceIn + roundUpToMultiple(inChannels,32) * batchSize * numTilesY * numTilesX * inTileXSize * inTileYSize;
      TENSORMAP3 transformedInput(convWorkspaceIn, inChannels, batchSize * numTilesY * numTilesX, inTileXSize * inTileYSize);
      TENSORMAP3 transformedOutput(convWorkspaceOut, outChannels, batchSize * numTilesY * numTilesX, inTileXSize * inTileYSize);
      for(int n = 0; n < batchSize; n++) {
        for(int yTile = 0; yTile < numTilesY; yTile++) {
          for(int xTile = 0; xTile < numTilesX; xTile++) {
            for(int dy = 0; dy < inTileYSize; dy++) {
              for(int dx = 0; dx < inTileXSize; dx++) {
                int x = xTile*outTileXSize+dx+inTileXOffset;
                int y = yTile*outTileYSize+dy+inTileYOffset;
                int subTileIdx = dy * inTileXSize + dx;
                if(x < 0 || y < 0 || x >= xSize || y >= ySize) {
                  std::fill(tile + subTileIdx * inChannels, tile + (subTileIdx+1) * inChannels, 0.0f);
                }
                else {
                  for(int ic = 0; ic < inChannels; ic++) {
                    float z = (*input)(ic,x,y,n);
                    tile[subTileIdx * inChannels + ic] = z;
                  }
                }
              }
            }

            for(int subY = 0; subY < inTileYSize; subY++) {
              float* __restrict t0 = &tile[(subY*inTileXSize+0)*inChannels];
              float* __restrict t1 = &tile[(subY*inTileXSize+1)*inChannels];
              float* __restrict t2 = &tile[(subY*inTileXSize+2)*inChannels];
              float* __restrict t3 = &tile[(subY*inTileXSize+3)*inChannels];
              float* __restrict t4 = &tile[(subY*inTileXSize+4)*inChannels];
              float* __restrict t5 = &tile[(subY*inTileXSize+5)*inChannels];
              for(int ic = 0; ic < inChannels; ic++) {
                float z0 = t0[ic];
                float z1 = t1[ic];
                float z2 = t2[ic];
                float z3 = t3[ic];
                float z4 = t4[ic];
                float z5 = t5[ic];
                t0[ic] = 4.0f*z0 - 5.0f*z2 + z4;
                t1[ic] = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
                t2[ic] =   4.0f*z1 - 4.0f*z2 - z3 + z4;
                t3[ic] = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
                t4[ic] =   2.0f*z1 - z2 - 2.0f*z3 + z4;
                t5[ic] = 4.0f*z1 - 5.0f*z3 + z5;
              }
            }
            for(int subX = 0; subX < inTileXSize; subX++) {
              float* __restrict t0 = &tile[(0*inTileXSize+subX)*inChannels];
              float* __restrict t1 = &tile[(1*inTileXSize+subX)*inChannels];
              float* __restrict t2 = &tile[(2*inTileXSize+subX)*inChannels];
              float* __restrict t3 = &tile[(3*inTileXSize+subX)*inChannels];
              float* __restrict t4 = &tile[(4*inTileXSize+subX)*inChannels];
              float* __restrict t5 = &tile[(5*inTileXSize+subX)*inChannels];
              for(int ic = 0; ic < inChannels; ic++) {
                float z0 = t0[ic];
                float z1 = t1[ic];
                float z2 = t2[ic];
                float z3 = t3[ic];
                float z4 = t4[ic];
                float z5 = t5[ic];
                t0[ic] = 4.0f*z0 - 5.0f*z2 + z4;
                t1[ic] = - 4.0f*z1 - 4.0f*z2 + z3 + z4;
                t2[ic] =   4.0f*z1 - 4.0f*z2 - z3 + z4;
                t3[ic] = - 2.0f*z1 - z2 + 2.0f*z3 + z4;
                t4[ic] =   2.0f*z1 - z2 - 2.0f*z3 + z4;
                t5[ic] = 4.0f*z1 - 5.0f*z3 + z5;
              }
            }
            int batchTileXTileY = n * numTilesY * numTilesX + yTile * numTilesX + xTile;
            for(int dy = 0; dy < inTileYSize; dy++) {
              for(int dx = 0; dx < inTileXSize; dx++) {
                for(int ic = 0; ic < inChannels; ic++) {
                  int subTileIdx = dy * inTileXSize + dx;
                  transformedInput(ic, batchTileXTileY, subTileIdx) = tile[subTileIdx*inChannels+ic];
                }
              }
            }
          }
        }
      }

      //TODO someday: Does eigen have a fast batched matrix multiply?
      //Here we just manually iterate over the 36 matrices that need to get multiplied.
      //Also, if eigen were to support *interleaved* matrices (viewing it as a matrix whose element is
      //a vector of length 36 instead of a float), that might allow for improved transform/untransform implementations.
      for(int dy = 0; dy < inTileYSize; dy++) {
        for(int dx = 0; dx < inTileXSize; dx++) {
          int subTileIdx = dy * inTileXSize + dx;
          auto transformedInputMap = Eigen::Map<Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>(
            (float*)transformedInput.data() + subTileIdx * batchSize * numTilesY * numTilesX * inChannels,
            inChannels,
            batchSize * numTilesY * numTilesX
          );
          auto winogradKernelMap = Eigen::Map<Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>(
            (float*)winogradKernel.data() + subTileIdx * outChannels * inChannels,
            outChannels,
            inChannels
          );
          auto transformedOutputMap = Eigen::Map<Eigen::Matrix<SCALAR,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>>(
            (float*)transformedOutput.data() + subTileIdx * batchSize * numTilesY * numTilesX * outChannels,
            outChannels,
            batchSize * numTilesY * numTilesX
          );
          transformedOutputMap = winogradKernelMap * transformedInputMap;
        }
      }

      for(int n = 0; n < batchSize; n++) {
        for(int yTile = 0; yTile < numTilesY; yTile++) {
          for(int xTile = 0; xTile < numTilesX; xTile++) {
            int batchTileXTileY = n * numTilesY * numTilesX + yTile * numTilesX + xTile;
            for(int dy = 0; dy < inTileYSize; dy++) {
              for(int dx = 0; dx < inTileXSize; dx++) {
                int subTileIdx = dy * inTileXSize + dx;
                for(int oc = 0; oc < outChannels; oc++) {
                  tile[subTileIdx*outChannels+oc] = transformedOutput(oc, batchTileXTileY, subTileIdx);
                }
              }
            }

            if(convXSize == 5 && convYSize == 5) {
              for(int subY = 0; subY < inTileYSize; subY++) {
                float* __restrict t0 = &tile[(subY*inTileXSize+0)*outChannels];
                float* __restrict t1 = &tile[(subY*inTileXSize+1)*outChannels];
                float* __restrict t2 = &tile[(subY*inTileXSize+2)*outChannels];
                float* __restrict t3 = &tile[(subY*inTileXSize+3)*outChannels];
                float* __restrict t4 = &tile[(subY*inTileXSize+4)*outChannels];
                float* __restrict t5 = &tile[(subY*inTileXSize+5)*outChannels];
                for(int oc = 0; oc < outChannels; oc++) {
                  float z0 = t0[oc];
                  float z1 = t1[oc];
                  float z2 = t2[oc];
                  float z3 = t3[oc];
                  float z4 = t4[oc];
                  float z5 = t5[oc];
                  t0[oc] = z0 + z1 + z2 + z3 + z4;
                  t1[oc] = (z1-z2) + 2.0f*(z3-z4) + z5;
                }
              }
              for(int subX = 0; subX < outTileXSize; subX++) {
                float* __restrict t0 = &tile[(0*inTileXSize+subX)*outChannels];
                float* __restrict t1 = &tile[(1*inTileXSize+subX)*outChannels];
                float* __restrict t2 = &tile[(2*inTileXSize+subX)*outChannels];
                float* __restrict t3 = &tile[(3*inTileXSize+subX)*outChannels];
                float* __restrict t4 = &tile[(4*inTileXSize+subX)*outChannels];
                float* __restrict t5 = &tile[(5*inTileXSize+subX)*outChannels];
                for(int oc = 0; oc < outChannels; oc++) {
                  float z0 = t0[oc];
                  float z1 = t1[oc];
                  float z2 = t2[oc];
                  float z3 = t3[oc];
                  float z4 = t4[oc];
                  float z5 = t5[oc];
                  t0[oc] = z0 + z1 + z2 + z3 + z4;
                  t1[oc] = (z1-z2) + 2.0f*(z3-z4) + z5;
                }
              }
            }
            else {
              for(int subY = 0; subY < inTileYSize; subY++) {
                float* __restrict t0 = &tile[(subY*inTileXSize+0)*outChannels];
                float* __restrict t1 = &tile[(subY*inTileXSize+1)*outChannels];
                float* __restrict t2 = &tile[(subY*inTileXSize+2)*outChannels];
                float* __restrict t3 = &tile[(subY*inTileXSize+3)*outChannels];
                float* __restrict t4 = &tile[(subY*inTileXSize+4)*outChannels];
                float* __restrict t5 = &tile[(subY*inTileXSize+5)*outChannels];
                for(int oc = 0; oc < outChannels; oc++) {
                  float z0 = t0[oc];
                  float z1 = t1[oc];
                  float z2 = t2[oc];
                  float z3 = t3[oc];
                  float z4 = t4[oc];
                  float z5 = t5[oc];
                  t0[oc] = z0 + z1 + z2 + z3 + z4;
                  t1[oc] = (z1-z2) + 2.0f*(z3-z4);
                  t2[oc] = (z1+z2) + 4.0f*(z3+z4);
                  t3[oc] = (z1-z2) + 8.0f*(z3-z4) + z5;
                }
              }
              for(int subX = 0; subX < outTileXSize; subX++) {
                float* __restrict t0 = &tile[(0*inTileXSize+subX)*outChannels];
                float* __restrict t1 = &tile[(1*inTileXSize+subX)*outChannels];
                float* __restrict t2 = &tile[(2*inTileXSize+subX)*outChannels];
                float* __restrict t3 = &tile[(3*inTileXSize+subX)*outChannels];
                float* __restrict t4 = &tile[(4*inTileXSize+subX)*outChannels];
                float* __restrict t5 = &tile[(5*inTileXSize+subX)*outChannels];
                for(int oc = 0; oc < outChannels; oc++) {
                  float z0 = t0[oc];
                  float z1 = t1[oc];
                  float z2 = t2[oc];
                  float z3 = t3[oc];
                  float z4 = t4[oc];
                  float z5 = t5[oc];
                  t0[oc] = z0 + z1 + z2 + z3 + z4;
                  t1[oc] = (z1-z2) + 2.0f*(z3-z4);
                  t2[oc] = (z1+z2) + 4.0f*(z3+z4);
                  t3[oc] = (z1-z2) + 8.0f*(z3-z4) + z5;
                }
              }
            }

            if(accumulate) {
              for(int dy = 0; dy < outTileYSize; dy++) {
                for(int dx = 0; dx < outTileXSize; dx++) {
                  int x = xTile*outTileXSize+dx;
                  int y = yTile*outTileYSize+dy;
                  if(!(x < 0 || y < 0 || x >= xSize || y >= ySize)) {
                    int subTileIdx = dy * inTileXSize + dx;
                    for(int oc = 0; oc < outChannels; oc++) {
                      (*output)(oc,x,y,n) += tile[subTileIdx*outChannels+oc];
                    }
                  }
                }
              }
            }
            else {
              for(int dy = 0; dy < outTileYSize; dy++) {
                for(int dx = 0; dx < outTileXSize; dx++) {
                  int x = xTile*outTileXSize+dx;
                  int y = yTile*outTileYSize+dy;
                  if(!(x < 0 || y < 0 || x >= xSize || y >= ySize)) {
                    int subTileIdx = dy * inTileXSize + dx;
                    for(int oc = 0; oc < outChannels; oc++) {
                      (*output)(oc,x,y,n) = tile[subTileIdx*outChannels+oc];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    else {
      Eigen::array<Eigen::Index, 2> imagePatchColVectorShape = {imagePatchSize, xSize*ySize*batchSize};
      Eigen::array<Eigen::IndexPair<int>, 1> contractionDims = {Eigen::IndexPair<int>(1, 0)};
      Eigen::array<Eigen::Index, 4> outputShape = {outChannels,xSize,ySize,batchSize};
      auto imagePatches = input->extract_image_patches(convXSize,convYSize).reshape(imagePatchColVectorShape);
      auto convolution = imagePatchKernel.contract(imagePatches, contractionDims).reshape(outputShape);
      if(accumulate)
        *output += convolution;
      else
        *output = convolution;
    }
  }
};

//--------------------------------------------------------------

struct BatchNormLayer {
  string name;

  vector<float> mergedScale;
  vector<float> mergedBias;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(const BatchNormLayerDesc& desc) {
    name = desc.name;
    int numChannels = desc.numChannels;
    float epsilon = desc.epsilon;

    mergedScale.resize(numChannels);
    mergedBias.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      mergedScale[c] = desc.scale[c] / sqrt(desc.variance[c] + epsilon);
      mergedBias[c] = desc.bias[c] - mergedScale[c] * desc.mean[c];
    }
  }

  // Mask should be in 'NHW' format (no "C" channel).
  void apply(
    bool applyRelu,
    CONSTTENSORMAP4* input,
    TENSORMAP4* output,
    CONSTTENSORMAP3* mask
  ) const {
    for(int c = 0; c < input->dimension(0); c++) {
      auto inC = input->chip(c, 0);
      auto x = inC * mergedScale[c] + mergedBias[c];
      auto z = TENSOR3(mask->dimension(0), mask->dimension(1), mask->dimension(2)).setZero();
      if(applyRelu)
        output->chip(c, 0) = (*mask == 1.f).select(x.cwiseMax(0.f), z);
      else
        output->chip(c, 0) = (*mask == 1.f).select(x, z);
    }
  }
};

//--------------------------------------------------------------

struct ActivationLayer {
  string name;

  ActivationLayer() = delete;
  ActivationLayer(const ActivationLayer&) = delete;
  ActivationLayer& operator=(const ActivationLayer&) = delete;

  ActivationLayer(const ActivationLayerDesc& desc) { name = desc.name; }

  template <int N>
  void apply(const Tensor<SCALAR, N>* input, Tensor<SCALAR, N>* output) const { *output = input->cwiseMax(0.f); }
  template <int N>
  void apply(const TensorMap<Tensor<SCALAR, N>>* input, TensorMap<Tensor<SCALAR, N>>* output) const { *output = input->cwiseMax(0.f); }
};

//--------------------------------------------------------------

struct MatMulLayer {
  string name;
  TENSOR2 weights;

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(const MatMulLayerDesc& desc)
    : name(desc.name)
  {
    weights = TENSOR2(desc.outChannels, desc.inChannels);
    memcpy(weights.data(), desc.weights.data(), sizeof(SCALAR) * weights.size());
  }

  void apply(CONSTTENSORMAP2* in, TENSORMAP2* out) const {
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    *out = weights.contract(*in, product_dims);
  }
};

struct MatBiasLayer {
  string name;
  std::vector<float> weights;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(const MatBiasLayerDesc& desc)
    : name(desc.name),
      weights(desc.weights) {}

  void apply(TENSORMAP2* mat) const {
    for(int n = 0; n < mat->dimension(1); n++) {
      for(int c = 0; c < mat->dimension(0); c++) {
        (*mat)(c, n) += weights[c];
      }
    }
  }
};

// Blocks
// --------------------------------------------------------------------------------------------------------------

struct ResidualBlockIntf {
  virtual ~ResidualBlockIntf(){}

  virtual void apply(
    ComputeHandleInternal* handle,
    TENSORMAP4* trunk,
    TENSORMAP4* trunkScratch,
    TENSORMAP4* regularOut,
    TENSORMAP4* regularScratch,
    TENSORMAP4* midIn,
    TENSORMAP4* midScratch,
    TENSORMAP4* gpoolOut,
    TENSORMAP4* gpoolOut2,
    TENSORMAP2* gpoolConcat,
    TENSORMAP2* gpoolBias,
    CONSTTENSORMAP3* mask,
    const float* maskSum,
    float* convWorkspace
  ) const = 0;

  virtual size_t requiredConvWorkspaceElts(size_t maxBatchSize) const = 0;
};

struct ResidualBlock final : public ResidualBlockIntf {
  string name;
  BatchNormLayer preBN;
  ConvLayer regularConv;
  BatchNormLayer midBN;
  ConvLayer finalConv;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ~ResidualBlock(){}

  ResidualBlock(const ResidualBlockDesc& desc, int nnX, int nnY)
    : name(desc.name),
      preBN(desc.preBN),
      regularConv(desc.regularConv,nnX,nnY),
      midBN(desc.midBN),
      finalConv(desc.finalConv,nnX,nnY) {}

  size_t requiredConvWorkspaceElts(size_t maxBatchSize) const {
    return std::max(
      regularConv.requiredConvWorkspaceElts(maxBatchSize),
      finalConv.requiredConvWorkspaceElts(maxBatchSize)
    );
  }

  void apply(
    ComputeHandleInternal* handle,
    TENSORMAP4* trunk,
    TENSORMAP4* trunkScratch,
    TENSORMAP4* regularOut,
    TENSORMAP4* regularScratch,
    TENSORMAP4* midIn,
    TENSORMAP4* midScratch,
    TENSORMAP4* gpoolOut,
    TENSORMAP4* gpoolOut2,
    TENSORMAP2* gpoolConcat,
    TENSORMAP2* gpoolBias,
    CONSTTENSORMAP3* mask,
    const float* maskSum,
    float* convWorkspace
  ) const override {
    (void)regularOut;
    (void)regularScratch;
    (void)gpoolOut;
    (void)gpoolOut2;
    (void)gpoolConcat;
    (void)gpoolBias;
    (void)maskSum;
    const bool applyBNRelu = true;
    preBN.apply(applyBNRelu, trunk, trunkScratch, mask);
    regularConv.apply(handle, trunkScratch, midIn, convWorkspace, false);
    midBN.apply(applyBNRelu, midIn, midScratch, mask);
    finalConv.apply(handle, midScratch, trunk, convWorkspace, true);
  }
};


struct GlobalPoolingResidualBlock final : public ResidualBlockIntf {
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

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  ~GlobalPoolingResidualBlock(){}

  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlockDesc& desc, int nnX, int nnY)
    : name(desc.name),
      preBN(desc.preBN),
      preActivation(desc.preActivation),
      regularConv(desc.regularConv,nnX,nnY),
      gpoolConv(desc.gpoolConv,nnX,nnY),
      gpoolBN(desc.gpoolBN),
      gpoolActivation(desc.gpoolActivation),
      gpoolToBiasMul(desc.gpoolToBiasMul),
      midBN(desc.midBN),
      midActivation(desc.midActivation),
      finalConv(desc.finalConv,nnX,nnY) {}

  size_t requiredConvWorkspaceElts(size_t maxBatchSize) const {
    size_t maxElts = 0;
    maxElts = std::max(maxElts,regularConv.requiredConvWorkspaceElts(maxBatchSize));
    maxElts = std::max(maxElts,gpoolConv.requiredConvWorkspaceElts(maxBatchSize));
    maxElts = std::max(maxElts,finalConv.requiredConvWorkspaceElts(maxBatchSize));
    return maxElts;
  }

  void apply(
    ComputeHandleInternal* handle,
    TENSORMAP4* trunk,
    TENSORMAP4* trunkScratch,
    TENSORMAP4* regularOut,
    TENSORMAP4* regularScratch,
    TENSORMAP4* midIn,
    TENSORMAP4* midScratch,
    TENSORMAP4* gpoolOut,
    TENSORMAP4* gpoolOut2,
    TENSORMAP2* gpoolConcat,
    TENSORMAP2* gpoolBias,
    CONSTTENSORMAP3* mask,
    const float* maskSum,
    float* convWorkspace
  ) const override {
    (void)midIn;
    (void)midScratch;
    const bool applyBNRelu = true;
    DTENSOR("trunk", trunk);
    DTENSOR("mask", mask);
    preBN.apply(applyBNRelu, trunk, trunkScratch, mask);
    DTENSOR("trunkScratch", trunkScratch);
    regularConv.apply(handle, trunkScratch, regularOut, convWorkspace, false);
    DTENSOR("regularOut", regularOut);
    gpoolConv.apply(handle, trunkScratch, gpoolOut, convWorkspace, false);
    DTENSOR("gpoolOut", gpoolOut);
    gpoolBN.apply(applyBNRelu, gpoolOut, gpoolOut2, mask);
    DTENSOR("gpoolOut2", gpoolOut2);
    poolRowsGPool(gpoolOut2, gpoolConcat, maskSum);
    gpoolToBiasMul.apply(gpoolConcat, gpoolBias);
    addNCBiasInplace(regularOut, gpoolBias);
    midBN.apply(applyBNRelu, regularOut, regularScratch, mask);
    finalConv.apply(handle, regularScratch, trunk, convWorkspace, true);
    DSHAPE("trunk", trunk);
    DSHAPE("trunkScratch", trunkScratch);
    DSHAPE("regularOut", regularOut);
    DSHAPE("gpoolOut", gpoolOut);
    DSHAPE("gpoolOut2", gpoolOut2);
    DSHAPE("gpoolConcat", gpoolConcat);
    DSHAPE("gpoolBias", gpoolBias);
    DSHAPE("mask", mask);
  }
};

struct Trunk {
  string name;
  int version;
  int numBlocks;

  ConvLayer initialConv;
  MatMulLayer initialMatMul;
  vector<pair<int, std::unique_ptr<ResidualBlockIntf>>> blocks;
  BatchNormLayer trunkTipBN;
  ActivationLayer trunkTipActivation;

  Trunk() = delete;
  Trunk(const Trunk&) = delete;
  Trunk& operator=(const Trunk&) = delete;

  Trunk(const TrunkDesc& desc, int nnX, int nnY)
    : name(desc.name),
      version(desc.version),
      numBlocks(desc.numBlocks),
      initialConv(desc.initialConv,nnX,nnY),
      initialMatMul(desc.initialMatMul),
      trunkTipBN(desc.trunkTipBN),
      trunkTipActivation(desc.trunkTipActivation)
  {
    for (int i = 0; i < numBlocks; ++i) {
      if (desc.blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlockDesc* blockDesc = (ResidualBlockDesc*)desc.blocks[i].second.get();
        std::unique_ptr<ResidualBlockIntf> block = std::make_unique<ResidualBlock>(*blockDesc,nnX,nnY);
        blocks.push_back(make_pair(ORDINARY_BLOCK_KIND, std::move(block)));
      }
      else if (desc.blocks[i].first == DILATED_BLOCK_KIND) {
        throw StringError("Eigen backend: Dilated residual blocks are not supported right now");
      }
      else if (desc.blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlockDesc* blockDesc = (GlobalPoolingResidualBlockDesc*)desc.blocks[i].second.get();
        std::unique_ptr<GlobalPoolingResidualBlock> block = std::make_unique<GlobalPoolingResidualBlock>(*blockDesc,nnX,nnY);
        blocks.push_back(make_pair(GLOBAL_POOLING_BLOCK_KIND, std::move(block)));
      }
      else {
        ASSERT_UNREACHABLE;
      }
    }
  }

  ~Trunk() {
  }

  size_t requiredConvWorkspaceElts(size_t maxBatchSize) const {
    size_t maxElts = initialConv.requiredConvWorkspaceElts(maxBatchSize);
    for(int i = 0; i<blocks.size(); i++) {
      maxElts = std::max(maxElts,blocks[i].second->requiredConvWorkspaceElts(maxBatchSize));
    }
    return maxElts;
  }

  void apply(
    ComputeHandleInternal* handle,
    CONSTTENSORMAP4* input,
    CONSTTENSORMAP2* inputGlobal,
    TENSORMAP2* inputMatMulOut,
    TENSORMAP4* trunk,
    TENSORMAP4* trunkScratch,
    TENSORMAP4* regularOut,
    TENSORMAP4* regularScratch,
    TENSORMAP4* midIn,
    TENSORMAP4* midScratch,
    TENSORMAP4* gpoolOut,
    TENSORMAP4* gpoolOut2,
    TENSORMAP2* gpoolConcat,
    TENSORMAP2* gpoolBias,
    CONSTTENSORMAP3* mask,
    const float* maskSum,
    float* convWorkspace
  ) const {

    initialConv.apply(handle, input, trunkScratch, convWorkspace, false);
    initialMatMul.apply(inputGlobal, inputMatMulOut);
    addNCBiasInplace(trunkScratch, inputMatMulOut);

    // apply blocks
    // Flip trunkBuf and trunkScratchBuf so that the result gets accumulated in trunkScratchBuf
    for(auto& block : blocks) {
      block.second->apply(
        handle,
        trunkScratch,
        trunk,
        regularOut,
        regularScratch,
        midIn,
        midScratch,
        gpoolOut,
        gpoolOut2,
        gpoolConcat,
        gpoolBias,
        mask,
        maskSum,
        convWorkspace
      );
    }

    // And now with the final BN port it from trunkScratchBuf to trunkBuf.
    const bool applyBNRelu = true;
    trunkTipBN.apply(applyBNRelu, trunkScratch, trunk, mask);
  }
};

struct PolicyHead {
  string name;
  int version;

  ConvLayer p1Conv;
  ConvLayer g1Conv;
  BatchNormLayer g1BN;
  ActivationLayer g1Activation;
  MatMulLayer gpoolToBiasMul;
  BatchNormLayer p1BN;
  ActivationLayer p1Activation;
  ConvLayer p2Conv;
  MatMulLayer gpoolToPassMul;

  PolicyHead() = delete;
  PolicyHead(const PolicyHead&) = delete;
  PolicyHead& operator=(const PolicyHead&) = delete;

  PolicyHead(const PolicyHeadDesc& desc, int nnX, int nnY)
    : name(desc.name),
      version(desc.version),
      p1Conv(desc.p1Conv,nnX,nnY),
      g1Conv(desc.g1Conv,nnX,nnY),
      g1BN(desc.g1BN),
      g1Activation(desc.g1Activation),
      gpoolToBiasMul(desc.gpoolToBiasMul),
      p1BN(desc.p1BN),
      p1Activation(desc.p1Activation),
      p2Conv(desc.p2Conv,nnX,nnY),
      gpoolToPassMul(desc.gpoolToPassMul) {}

  size_t requiredConvWorkspaceElts(size_t maxBatchSize) const {
    size_t maxElts = 0;
    maxElts = std::max(maxElts,p1Conv.requiredConvWorkspaceElts(maxBatchSize));
    maxElts = std::max(maxElts,g1Conv.requiredConvWorkspaceElts(maxBatchSize));
    maxElts = std::max(maxElts,p2Conv.requiredConvWorkspaceElts(maxBatchSize));
    return maxElts;
  }

  void apply(
    ComputeHandleInternal* handle,
    CONSTTENSORMAP4* trunk,
    TENSORMAP4* p1Out,
    TENSORMAP4* p1Out2,
    TENSORMAP4* g1Out,
    TENSORMAP4* g1Out2,
    TENSORMAP2* g1Concat,
    TENSORMAP2* g1Bias,
    TENSORMAP2* policyPass,
    TENSORMAP4* policy,
    CONSTTENSORMAP3* mask,
    const float* maskSum,
    float* convWorkspace
  ) const {
    const bool applyBNRelu = true;
    p1Conv.apply(handle, trunk, p1Out, convWorkspace, false);
    g1Conv.apply(handle, trunk, g1Out, convWorkspace, false);
    g1BN.apply(applyBNRelu, g1Out, g1Out2, mask);
    poolRowsGPool(g1Out2, g1Concat, maskSum);
    gpoolToBiasMul.apply(g1Concat, g1Bias);
    addNCBiasInplace(p1Out, g1Bias);
    p1BN.apply(true, p1Out, p1Out2, mask);
    p2Conv.apply(handle, p1Out2, policy, convWorkspace, false);
    gpoolToPassMul.apply(g1Concat, policyPass);
  }
};

struct ValueHead {
  string name;
  int version;

  ConvLayer v1Conv;
  BatchNormLayer v1BN;
  ActivationLayer v1Activation;
  MatMulLayer v2Mul;
  MatBiasLayer v2Bias;
  ActivationLayer v2Activation;
  MatMulLayer v3Mul;
  MatBiasLayer v3Bias;
  MatMulLayer sv3Mul;
  MatBiasLayer sv3Bias;
  ConvLayer vOwnershipConv;

  ValueHead() = delete;
  ValueHead(const ValueHead&) = delete;
  ValueHead& operator=(const ValueHead&) = delete;

  ValueHead(const ValueHeadDesc& desc, int nnX, int nnY)
    : name(desc.name),
      version(desc.version),
      v1Conv(desc.v1Conv,nnX,nnY),
      v1BN(desc.v1BN),
      v1Activation(desc.v1Activation),
      v2Mul(desc.v2Mul),
      v2Bias(desc.v2Bias),
      v2Activation(desc.v2Activation),
      v3Mul(desc.v3Mul),
      v3Bias(desc.v3Bias),
      sv3Mul(desc.sv3Mul),
      sv3Bias(desc.sv3Bias),
      vOwnershipConv(desc.vOwnershipConv,nnX,nnY) {}

  size_t requiredConvWorkspaceElts(size_t maxBatchSize) const {
    size_t maxElts = 0;
    maxElts = std::max(maxElts,v1Conv.requiredConvWorkspaceElts(maxBatchSize));
    maxElts = std::max(maxElts,vOwnershipConv.requiredConvWorkspaceElts(maxBatchSize));
    return maxElts;
  }

  void apply(
    ComputeHandleInternal* handle,
    CONSTTENSORMAP4* trunk,
    TENSORMAP4* v1Out,
    TENSORMAP4* v1Out2,
    TENSORMAP2* v1Mean,
    TENSORMAP2* v2Out,
    TENSORMAP2* value,
    TENSORMAP2* scoreValue,
    TENSORMAP4* ownership,
    CONSTTENSORMAP3* mask,
    const float* maskSum,
    float* convWorkspace
  ) const {
    bool applyBNRelu = true;
    v1Conv.apply(handle, trunk, v1Out, convWorkspace, false);
    v1BN.apply(applyBNRelu, v1Out, v1Out2, mask);
    poolRowsValueHead(v1Out2, v1Mean, maskSum);
    v2Mul.apply(v1Mean, v2Out);
    v2Bias.apply(v2Out);
    v2Activation.apply(v2Out, v2Out);
    v3Mul.apply(v2Out, value);
    v3Bias.apply(value);

    sv3Mul.apply(v2Out, scoreValue);
    sv3Bias.apply(scoreValue);

    vOwnershipConv.apply(handle, v1Out2, ownership, convWorkspace, false);
  }
};


// Model and Buffer I/O ------------------------------------------------------------------------------------------------

struct Model {
  string name;
  int version;
  int numInputChannels;
  int numInputGlobalChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;

  Trunk trunk;
  PolicyHead policyHead;
  ValueHead valueHead;

  Model() = delete;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  Model(const ModelDesc& desc, int nnX, int nnY)
    : name(desc.name), version(desc.version), numInputChannels(desc.numInputChannels),
      numInputGlobalChannels(desc.numInputGlobalChannels),
      numValueChannels(desc.numValueChannels),
      numScoreValueChannels(desc.numScoreValueChannels),
      numOwnershipChannels(desc.numOwnershipChannels),
      trunk(desc.trunk,nnX,nnY),
      policyHead(desc.policyHead,nnX,nnY),
      valueHead(desc.valueHead,nnX,nnY) {}

  size_t requiredConvWorkspaceElts(size_t maxBatchSize) const {
    size_t maxElts = 0;
    maxElts = std::max(maxElts,trunk.requiredConvWorkspaceElts(maxBatchSize));
    maxElts = std::max(maxElts,policyHead.requiredConvWorkspaceElts(maxBatchSize));
    maxElts = std::max(maxElts,valueHead.requiredConvWorkspaceElts(maxBatchSize));
    return maxElts;
  }

  void apply(
    ComputeHandleInternal* handle,
    CONSTTENSORMAP4* input,
    CONSTTENSORMAP2* inputGlobal,
    TENSORMAP2* inputMatMulOut,
    TENSORMAP4* trunkBuf,
    TENSORMAP4* trunkScratch,
    TENSORMAP4* regularOut,
    TENSORMAP4* regularScratch,
    TENSORMAP4* midIn,
    TENSORMAP4* midScratch,
    TENSORMAP4* gpoolOut,
    TENSORMAP4* gpoolOut2,
    TENSORMAP2* gpoolConcat,
    TENSORMAP2* gpoolBias,

    TENSORMAP4* p1Out,
    TENSORMAP4* p1Out2,
    TENSORMAP4* g1Out,
    TENSORMAP4* g1Out2,
    TENSORMAP2* g1Concat,
    TENSORMAP2* g1Bias,
    TENSORMAP2* policyPass,
    TENSORMAP4* policy,

    TENSORMAP4* v1Out,
    TENSORMAP4* v1Out2,
    TENSORMAP2* v1Mean,
    TENSORMAP2* v2Out,
    TENSORMAP2* value,
    TENSORMAP2* scoreValue,
    TENSORMAP4* ownership,

    TENSORMAP3* mask,
    float* maskSum,
    float* convWorkspace
  ) const {
    *mask = input->chip(0,0);
    computeMaskSum(mask,maskSum);

    trunk.apply(
      handle,
      input,
      inputGlobal,
      inputMatMulOut,
      trunkBuf,
      trunkScratch,
      regularOut,
      regularScratch,
      midIn,
      midScratch,
      gpoolOut,
      gpoolOut2,
      gpoolConcat,
      gpoolBias,
      mask,
      maskSum,
      convWorkspace
    );
    policyHead.apply(
      handle,
      trunkBuf,
      p1Out,
      p1Out2,
      g1Out,
      g1Out2,
      g1Concat,
      g1Bias,
      policyPass,
      policy,
      mask,
      maskSum,
      convWorkspace
    );
    valueHead.apply(
      handle,
      trunkBuf,
      v1Out,
      v1Out2,
      v1Mean,
      v2Out,
      value,
      scoreValue,
      ownership,
      mask,
      maskSum,
      convWorkspace
    );
  }
};

//--------------------------------------------------------------

struct Buffers {
  TENSOR2 inputMatMulOut;
  TENSOR4 trunk;
  TENSOR4 trunkScratch;
  TENSOR4 regularOut;
  TENSOR4 regularScratch;
  TENSOR4 midIn;
  TENSOR4 midScratch;
  TENSOR4 gpoolOut;
  TENSOR4 gpoolOut2;
  TENSOR2 gpoolConcat;
  TENSOR2 gpoolBias;

  TENSOR4 p1Out;
  TENSOR4 p1Out2;
  TENSOR4 g1Out;
  TENSOR4 g1Out2;
  TENSOR2 g1Concat;
  TENSOR2 g1Bias;
  TENSOR2 policyPass;
  TENSOR4 policy;

  TENSOR4 v1Out;
  TENSOR4 v1Out2;
  TENSOR2 v1Mean;
  TENSOR2 v2Out;
  TENSOR2 value;
  TENSOR2 scoreValue;
  TENSOR4 ownership;

  TENSOR3 mask;
  vector<float> maskSum;
  vector<float> convWorkspace;

  Buffers(
    const ModelDesc& desc,
    const Model& m,
    int maxBatchSize,
    int nnXLen,
    int nnYLen
  ) :
    inputMatMulOut(desc.trunk.trunkNumChannels, maxBatchSize),
    trunk(desc.trunk.trunkNumChannels, nnXLen, nnYLen, maxBatchSize),
    trunkScratch(desc.trunk.trunkNumChannels, nnXLen, nnYLen, maxBatchSize),
    regularOut(desc.trunk.regularNumChannels, nnXLen, nnYLen, maxBatchSize),
    regularScratch(desc.trunk.regularNumChannels, nnXLen, nnYLen, maxBatchSize),
    midIn(desc.trunk.midNumChannels, nnXLen, nnYLen, maxBatchSize),
    midScratch(desc.trunk.midNumChannels, nnXLen, nnYLen, maxBatchSize),
    gpoolOut(desc.trunk.gpoolNumChannels, nnXLen, nnYLen, maxBatchSize),
    gpoolOut2(desc.trunk.gpoolNumChannels, nnXLen, nnYLen, maxBatchSize),
    gpoolConcat(desc.trunk.gpoolNumChannels*3, maxBatchSize),
    gpoolBias(desc.trunk.regularNumChannels, maxBatchSize),

    p1Out(desc.policyHead.p1Conv.outChannels, nnXLen, nnYLen, maxBatchSize),
    p1Out2(desc.policyHead.p1Conv.outChannels, nnXLen, nnYLen, maxBatchSize),
    g1Out(desc.policyHead.g1Conv.outChannels, nnXLen, nnYLen, maxBatchSize),
    g1Out2(desc.policyHead.g1Conv.outChannels, nnXLen, nnYLen, maxBatchSize),
    g1Concat(desc.policyHead.g1Conv.outChannels*3, maxBatchSize),
    g1Bias(desc.policyHead.gpoolToBiasMul.outChannels, maxBatchSize),
    policyPass(desc.policyHead.gpoolToPassMul.outChannels, maxBatchSize),
    policy(desc.policyHead.p2Conv.outChannels, nnXLen, nnYLen, maxBatchSize),

    v1Out(desc.valueHead.v1Conv.outChannels, nnXLen, nnYLen, maxBatchSize),
    v1Out2(desc.valueHead.v1Conv.outChannels, nnXLen, nnYLen, maxBatchSize),
    v1Mean(desc.valueHead.v1Conv.outChannels*3, maxBatchSize),
    v2Out(desc.valueHead.v2Mul.outChannels, maxBatchSize),
    value(desc.valueHead.v3Mul.outChannels, maxBatchSize),
    scoreValue(desc.valueHead.sv3Mul.outChannels, maxBatchSize),
    ownership(desc.valueHead.vOwnershipConv.outChannels, nnXLen, nnYLen, maxBatchSize),

    mask(nnXLen, nnYLen, maxBatchSize),
    maskSum(maxBatchSize),
    convWorkspace(m.requiredConvWorkspaceElts(maxBatchSize))
  {}
};

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  std::vector<float> spatialInput;
  std::vector<float> globalInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    int xSize = nnXLen;
    int ySize = nnYLen;

    maxBatchSize = maxBatchSz;
    singleInputElts = m.numInputChannels * xSize * ySize;
    singleInputGlobalElts = m.numInputGlobalChannels;

    singlePolicyPassResultElts = (size_t)(1);
    singlePolicyResultElts = (size_t)(xSize * ySize);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * xSize * ySize;

    assert(NNModelVersion::getNumSpatialFeatures(m.version) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.version) == m.numInputGlobalChannels);

    spatialInput = vector<float>(m.numInputChannels * xSize * ySize * maxBatchSize);
    globalInput = vector<float>(m.numInputGlobalChannels * maxBatchSize);
  }

  ~InputBuffers() { }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}


// NeuralNet -----------------------------------------------------------------------------------------------------------

void NeuralNet::globalInitialize() {
  // no-op for cpu
}

void NeuralNet::globalCleanup() {
  // no-op for cpu
}


//------------------------------------------------------------------------------

struct ComputeContext {
  const int nnXLen;
  const int nnYLen;
  const Model model;

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

  ComputeContext(const LoadedModel& loadedModel, int nnX, int nnY)
    : nnXLen(nnX),
      nnYLen(nnY),
      model(loadedModel.modelDesc,nnX,nnY)
  {}
  ~ComputeContext()
  {}
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

  bool useFP16 = useFP16Mode == enabled_t::True ? true : false;
  bool useNHWC = useNHWCMode == enabled_t::False ? false : true;

  if(useFP16)
    throw StringError("Eigen backend: useFP16 = true not supported");
  if(!useNHWC)
    throw StringError("Eigen backend: useNHWC = false not supported");

  ComputeContext* context = new ComputeContext(*loadedModel,nnXLen,nnYLen);
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//------------------------------------------------------------------------------

struct ComputeHandle {
  const ComputeContext* context;
  int maxBatchSize;
  bool inputsUseNHWC;
  Buffers* buffers;
  ComputeHandleInternal handleInternal;

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  ComputeHandle(const ComputeContext* ctx, const LoadedModel& loadedModel, int maxBSize, bool iNHWC)
    : context(ctx),
      maxBatchSize(maxBSize),
      inputsUseNHWC(iNHWC),
      handleInternal()
  {
    buffers = new Buffers(loadedModel.modelDesc,context->model,maxBSize,ctx->nnXLen,ctx->nnYLen);
  }

  ~ComputeHandle() {
    delete buffers;
  }
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
  if(logger != NULL) {
    logger->write("Eigen (CPU) backend thread " + Global::intToString(serverThreadIdx) + ": Model version " + Global::intToString(loadedModel->modelDesc.version));
    logger->write("Eigen (CPU) backend thread " + Global::intToString(serverThreadIdx) + ": Model name: " + loadedModel->modelDesc.name);
  }

  (void)requireExactNNLen; //We don't bother with mask optimizations if we know exact sizes right now.
  (void)gpuIdxForThisThread; //Doesn't matter

  if(!inputsUseNHWC)
    throw StringError("Eigen backend: inputsUseNHWC = false unsupported");
  return new ComputeHandle(context, *loadedModel, maxBatchSize, inputsUseNHWC);
}

void NeuralNet::freeComputeHandle(ComputeHandle* gpuHandle) {
  delete gpuHandle;
}

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  int batchSize = numBatchEltsFilled;
  int nnXLen = computeHandle->context->nnXLen;
  int nnYLen = computeHandle->context->nnYLen;
  int version = computeHandle->context->model.version;

  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(version);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(version);
  assert(numSpatialFeatures == computeHandle->context->model.numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  for(int nIdx = 0; nIdx<batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobal;
    const float* rowSpatial = inputBufs[nIdx]->rowSpatial;
    std::copy(rowGlobal,rowGlobal+numGlobalFeatures,rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, computeHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry);
  }

  Buffers& buffers = *(computeHandle->buffers);

  CONSTTENSORMAP4 input(inputBuffers->spatialInput.data(), numSpatialFeatures, nnXLen, nnYLen, batchSize);
  CONSTTENSORMAP2 inputGlobal(inputBuffers->globalInput.data(), numGlobalFeatures, batchSize);

#define MAP4(NAME) TENSORMAP4 NAME(buffers.NAME.data(), buffers.NAME.dimension(0), buffers.NAME.dimension(1), buffers.NAME.dimension(2), batchSize)
#define MAP3(NAME) TENSORMAP3 NAME(buffers.NAME.data(), buffers.NAME.dimension(0), buffers.NAME.dimension(1), batchSize)
#define MAP2(NAME) TENSORMAP2 NAME(buffers.NAME.data(), buffers.NAME.dimension(0), batchSize)

  MAP2(inputMatMulOut);
  MAP4(trunk);
  MAP4(trunkScratch);
  MAP4(regularOut);
  MAP4(regularScratch);
  MAP4(midIn);
  MAP4(midScratch);
  MAP4(gpoolOut);
  MAP4(gpoolOut2);
  MAP2(gpoolConcat);
  MAP2(gpoolBias);
  MAP4(p1Out);
  MAP4(p1Out2);
  MAP4(g1Out);
  MAP4(g1Out2);
  MAP2(g1Concat);
  MAP2(g1Bias);
  MAP2(policyPass);
  MAP4(policy);
  MAP4(v1Out);
  MAP4(v1Out2);
  MAP2(v1Mean);
  MAP2(v2Out);
  MAP2(value);
  MAP2(scoreValue);
  MAP4(ownership);
  MAP3(mask);
  vector<float>& maskSum = buffers.maskSum;
  computeMaskSum(&mask,maskSum.data());
  vector<float>& convWorkspace = buffers.convWorkspace;

  computeHandle->context->model.apply(
    &computeHandle->handleInternal,
    &input,
    &inputGlobal,
    &inputMatMulOut,
    &trunk,
    &trunkScratch,
    &regularOut,
    &regularScratch,
    &midIn,
    &midScratch,
    &gpoolOut,
    &gpoolOut2,
    &gpoolConcat,
    &gpoolBias,
    &p1Out,
    &p1Out2,
    &g1Out,
    &g1Out2,
    &g1Concat,
    &g1Bias,
    &policyPass,
    &policy,
    &v1Out,
    &v1Out2,
    &v1Mean,
    &v2Out,
    &value,
    &scoreValue,
    &ownership,
    &mask,
    maskSum.data(),
    convWorkspace.data()
  );

  assert(outputs.size() == batchSize);

  float* policyData = policy.data();
  float* policyPassData = policyPass.data();
  float* valueData = value.data();
  float* scoreValueData = scoreValue.data();
  float* ownershipData = ownership.data();

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    const float* policySrcBuf = policyData + row * inputBuffers->singlePolicyResultElts;
    float* policyProbs = output->policyProbs;

    //These are not actually correct, the client does the postprocessing to turn them into
    //policy probabilities and white game outcome probabilities
    //Also we don't fill in the nnHash here either
    SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    policyProbs[inputBuffers->singlePolicyResultElts] = policyPassData[row];

    int numValueChannels = computeHandle->context->model.numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = valueData[row * numValueChannels];
    output->whiteLossProb = valueData[row * numValueChannels + 1];
    output->whiteNoResultProb = valueData[row * numValueChannels + 2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if(output->whiteOwnerMap != NULL) {
      const float* ownershipSrcBuf = ownershipData + row * nnXLen * nnYLen;
      assert(computeHandle->context->model.numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    if(version >= 9) {
      int numScoreValueChannels = computeHandle->context->model.numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
      output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = scoreValueData[row * numScoreValueChannels + 4];
      output->shorttermScoreError = scoreValueData[row * numScoreValueChannels + 5];
    }
    else if(version >= 8) {
      int numScoreValueChannels = computeHandle->context->model.numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
      output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(version >= 4) {
      int numScoreValueChannels = computeHandle->context->model.numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
      output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    }
    else if(version >= 3) {
      int numScoreValueChannels = computeHandle->context->model.numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
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


void NeuralNet::printDevices() {
}

// FOR TESTING ---------------------------------------------------------------------------------------------------------
bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer
) {
  if(!useNHWC || useFP16)
    return false;
  ConvLayer layer(*desc,nnXLen,nnYLen);
  TENSORMAP4 inTensor(
    (float*)inputBuffer.data(), desc->inChannels, nnXLen, nnYLen, batchSize);
  TENSOR4 outTensorBuf(desc->outChannels, nnXLen, nnYLen, batchSize);
  TENSORMAP4 outTensor(outTensorBuf);
  size_t convWorkspaceElts = layer.requiredConvWorkspaceElts(batchSize);
  vector<float> convWorkspace(convWorkspaceElts);

  ComputeHandleInternal handle;
  layer.apply(&handle, &inTensor, &outTensor, convWorkspace.data(), false);

  outputBuffer.resize(outTensorBuf.size());
  memcpy(outputBuffer.data(), outTensorBuf.data(), sizeof(SCALAR) * outTensorBuf.size());
  return true;
}

// Mask should be in 'NHW' format (no "C" channel).
bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  if(!useNHWC || useFP16)
    return false;
  BatchNormLayer layer(*desc);
  TENSORMAP4 inTensor((float*)inputBuffer.data(), desc->numChannels, nnXLen, nnYLen, batchSize);
  TENSORMAP3 mask((float*)maskBuffer.data(), nnXLen, nnYLen, batchSize);
  TENSOR4 outTensorBuf(desc->numChannels, nnXLen, nnYLen, batchSize);
  TENSORMAP4 outTensor(outTensorBuf);

  layer.apply(false, &inTensor, &outTensor, &mask);

  outputBuffer.resize(outTensorBuf.size());
  memcpy(outputBuffer.data(), outTensorBuf.data(), sizeof(SCALAR) * outTensorBuf.size());
  return true;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  if(!useNHWC || useFP16)
    return false;
  ResidualBlock block(*desc,nnXLen,nnYLen);
  TENSORMAP4 inTensor((float*)inputBuffer.data(), desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  TENSORMAP3 mask((float*)maskBuffer.data(), nnXLen, nnYLen, batchSize);
  size_t convWorkspaceElts = block.requiredConvWorkspaceElts(batchSize);
  vector<float> convWorkspace(convWorkspaceElts);

  TENSOR4 trunkBuf(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  TENSOR4 trunkScratchBuf(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  TENSOR4 midBuf(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);
  TENSOR4 midScratchBuf(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);

  TENSORMAP4 trunk(trunkBuf);
  TENSORMAP4 trunkScratch(trunkScratchBuf);
  TENSORMAP4 mid(midBuf);
  TENSORMAP4 midScratch(midScratchBuf);

  trunk = inTensor;

  ComputeHandleInternal handle;
  block.apply(
    &handle,
    &trunk,
    &trunkScratch,
    NULL,
    NULL,
    &mid,
    &midScratch,
    NULL,
    NULL,
    NULL,
    NULL,
    &mask,
    NULL,
    convWorkspace.data()
  );

  outputBuffer.resize(trunk.size());
  memcpy(outputBuffer.data(), trunk.data(), sizeof(SCALAR) * trunk.size());
  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer) {
  if(!useNHWC || useFP16)
    return false;

  GlobalPoolingResidualBlock block(*desc,nnXLen,nnYLen);

  TENSORMAP4 inTensor((float*)inputBuffer.data(), desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  TENSORMAP3 mask((float*)maskBuffer.data(), nnXLen, nnYLen, batchSize);
  size_t convWorkspaceElts = block.requiredConvWorkspaceElts(batchSize);
  vector<float> convWorkspace(convWorkspaceElts);

  TENSOR4 trunkBuf(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  TENSOR4 trunkScratchBuf(desc->preBN.numChannels, nnXLen, nnYLen, batchSize);
  TENSOR4 regularOutBuf(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);
  TENSOR4 regularScratchBuf(desc->finalConv.inChannels, nnXLen, nnYLen, batchSize);
  TENSOR4 gpoolOutBuf(desc->gpoolConv.outChannels, nnXLen, nnYLen, batchSize);
  TENSOR4 gpoolOut2Buf(desc->gpoolConv.outChannels, nnXLen, nnYLen, batchSize);
  TENSOR2 gpoolConcatBuf(desc->gpoolConv.outChannels*3, batchSize);
  TENSOR2 gpoolBiasBuf(desc->gpoolToBiasMul.outChannels, batchSize);

  TENSORMAP4 trunk(trunkBuf);
  TENSORMAP4 trunkScratch(trunkScratchBuf);
  TENSORMAP4 regularOut(regularOutBuf);
  TENSORMAP4 regularScratch(regularScratchBuf);
  TENSORMAP4 gpoolOut(gpoolOutBuf);
  TENSORMAP4 gpoolOut2(gpoolOut2Buf);
  TENSORMAP2 gpoolConcat(gpoolConcatBuf);
  TENSORMAP2 gpoolBias(gpoolBiasBuf);

  std::vector<float> maskSum(batchSize);
  computeMaskSum(&mask,maskSum.data());

  trunk = inTensor;

  ComputeHandleInternal handle;
  block.apply(
    &handle,
    &trunk,
    &trunkScratch,
    &regularOut,
    &regularScratch,
    NULL,
    NULL,
    &gpoolOut,
    &gpoolOut2,
    &gpoolConcat,
    &gpoolBias,
    &mask,
    maskSum.data(),
    convWorkspace.data()
  );

  outputBuffer.resize(trunk.size());
  memcpy(outputBuffer.data(), trunk.data(), sizeof(SCALAR) * trunk.size());

  return true;
}

#endif  // USE_EIGEN_BACKEND
