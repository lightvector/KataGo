#ifndef CUDAHELPERS_H
#define CUDAHELPERS_H

//Given two tensors with shapes inA: [n,cA,h,w] and inB: [n,cB,h,w], that are on the GPU
//Copy them into a single tensor out: [n,cA+cB,h,w] that is also allocated on the gpu
void customCudaChannelConcat(const float* inA, const float* inB, float* out, int chwA, int chwB, int n);
void customCudaChannelConcat(const half* inA, const half* inB, half* out, int chwA, int chwB, int n);

//Given an input with shape [n,c] and an output buffer of shape [n]
//fill output buffer with sum or max or mean over c.
void customCudaPoolRowsSum(float* in, float* out, int n, int c);
void customCudaPoolRowsMax(float* in, float* out, int n, int c);

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

//Given an input in half-precision with shape [n,c] and biases of shape [c], add the biases in-place.
void customCudaAddBiasInplace(half* buf, const half* biases, int n, int c);

//Given an input in half-precision with shape [n,c,s] and scale and biases of shape [c], multiply by scale and add the biases
void customCudaApplyScaleBias(const half* in, half* out, const half* scale, const half* biases, int n, int c, int s);


#endif
