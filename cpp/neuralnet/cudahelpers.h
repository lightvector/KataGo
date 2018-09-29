#ifndef CUDAHELPERS_H
#define CUDAHELPERS_H

//Given two tensors with shapes inA: [n,cA,h,w] and inB: [n,cB,h,w], that are on the GPU
//Copy them into a single tensor out: [n,cA+cB,h,w] that is also allocated on the gpu
void customCudaChannelConcat(const float* inA, const float* inB, float* out, int chwA, int chwB, int n);
void customCudaChannelConcat(const half* inA, const half* inB, half* out, int chwA, int chwB, int n);

//Given an input with shape [n,c] and an output buffer of shape [n]
//fill output buffer with sum or max or mean over c.
void customCudaPoolRowsSumNCHW(float* in, float* out, int nc, int xy);
void customCudaPoolRowsMaxNCHW(float* in, float* out, int nc, int xy);
void customCudaPoolRowsSumNHWC(const float* in, float* out, int n, int xy, int c);
void customCudaPoolRowsMaxNHWC(const float* in, float* out, int n, int xy, int c);

void customCudaNCHWTranspose(const float *in, float* out, int xSize, int ySize, int ncSize);
void customCudaNHWCTranspose(const float *in, float* out, int xSize, int ySize, int cSize, int nSize);
void customCudaNCHWTranspose(const half *in, half* out, int xSize, int ySize, int ncSize);
void customCudaNHWCTranspose(const half *in, half* out, int xSize, int ySize, int cSize, int nSize);

void customCudaMirror(const float *in, float* out, int batchSize, int mSize, int subSize);
void customCudaMirrorNCHW(const float *in, float* out, int batchSize, int cSize, int ySize, int xSize, bool mirrorY, bool mirrorX);
void customCudaMirrorNHWC(const float *in, float* out, int batchSize, int ySize, int xSize, int cSize, bool mirrorY, bool mirrorX);
void customCudaMirror(const half *in, half* out, int batchSize, int mSize, int subSize);
void customCudaMirrorNCHW(const half *in, half* out, int batchSize, int cSize, int ySize, int xSize, bool mirrorY, bool mirrorX);
void customCudaMirrorNHWC(const half *in, half* out, int batchSize, int ySize, int xSize, int cSize, bool mirrorY, bool mirrorX);

void customCudaCopyToHalf(const float* in, half* out, int n);
void customCudaCopyFromHalf(const half* in, float* out, int n);

//Given a tensor, add another tensor to it.
void customCudaAddTensorInplace(half* buf, const half* biases, int n);
//Given an input with shape [n,c] and biases of shape [c], add the biases in-place.
void customCudaAddCBiasInplaceNC(float* buf, const float* biases, int n, int c);
void customCudaAddCBiasInplaceNC(half* buf, const half* biases, int n, int c);
//Given an input with shape [n,c,xy] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNCHW(float *buf, const float* biases, int nSize, int cSize, int xySize);
void customCudaAddNCBiasInplaceNCHW(half *buf, const half* biases, int nSize, int cSize, int xySize);
//Given an input with shape [n,xy,c] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNHWC(float *buf, const float* biases, int nSize, int xySize, int cSize);
void customCudaAddNCBiasInplaceNHWC(half *buf, const half* biases, int nSize, int xySize, int cSize);

//Given an input with shape [n,c,xy] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply relu.
void customCudaApplyCScaleBiasNCHW(const float* in, float* out, const float* scale, const float* biases, int n, int c, int xy, bool applyRelu);
void customCudaApplyCScaleBiasNCHW(const half* in, half* out, const half* scale, const half* biases, int n, int c, int xy, bool applyRelu);
//Given an input with shape [n,xy,c] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply relu.
void customCudaApplyCScaleBiasNHWC(const float* in, float* out, const float* scale, const float* biases, int n, int xy, int c, bool applyRelu);
void customCudaApplyCScaleBiasNHWC(const half* in, half* out, const half* scale, const half* biases, int n, int xy, int c, bool applyRelu);


#endif
