#ifndef NEURALNET_CUDAHELPERS_H_
#define NEURALNET_CUDAHELPERS_H_

#include "../neuralnet/cudaincludes.h"
#include "../neuralnet/activations.h"

//Given two tensors with shapes inA: [n,cA,h,w] and inB: [n,cB,h,w], that are on the GPU
//Copy them into a single tensor out: [n,cA+cB,h,w] that is also allocated on the gpu
void customCudaChannelConcat(const float* inA, const float* inB, float* out, int chwA, int chwB, int n);
void customCudaChannelConcat(const half* inA, const half* inB, half* out, int chwA, int chwB, int n);

//Given a tensor [n,c,hw], extract out channel 0 to [n,hw]
void customCudaChannel0ExtractNCHW(const float* in, float* out, int n, int c, int hw);
void customCudaChannel0ExtractNCHW(const half* in, half* out, int n, int c, int hw);
//Given a tensor [n,hw,c], extract out channel 0 to [n,hw]
void customCudaChannel0ExtractNHWC(const float* in, float* out, int n, int hw, int c);
void customCudaChannel0ExtractNHWC(const half* in, half* out, int n, int hw, int c);

//Given an input tensor and an output buffer of shape [n,c], fill output buffer with sum or max over c.
void customCudaPoolRowsSumNCHW(const float* in, float* out, int nSize, int cSize, int xySize, float scaleSum);
void customCudaPoolRowsSumNHWC(const float* in, float* out, int nSize, int xySize, int cSize, float scaleSum);

//Specialized operations for value head and general global pooling. Same as the other pooling, but fusedly fills
//an output buffer of shape [n,c*3].
void customCudaValueHeadPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* maskSum);
void customCudaValueHeadPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* maskSum);
void customCudaPoolRowsGPoolNCHW(const float* in, float* out, int nSize, int cSize, int xySize, const float* mask, const float* maskSum);
void customCudaPoolRowsGPoolNHWC(const float* in, float* out, int nSize, int xySize, int cSize, const float* mask, const float* maskSum);
void customCudaPoolRowsGPoolNCHW(const half* in, half* out, int nSize, int cSize, int xySize, const half* mask, const float* maskSum);
void customCudaPoolRowsGPoolNHWC(const half* in, half* out, int nSize, int xySize, int cSize, const half* mask, const float* maskSum);

void customCudaCopyToHalf(const float* in, half* out, int n);
void customCudaCopyFromHalf(const half* in, float* out, int n);

//Given a tensor, add another tensor to it.
void customCudaAddTensorInplace(half* buf, const half* biases, int n);
//Given an input with shape [n,c] and biases of shape [c], add the biases in-place.
void customCudaAddCBiasInplaceNC(float* buf, const float* biases, int n, int c, int activation);
void customCudaAddCBiasInplaceNC(half* buf, const half* biases, int n, int c, int activation);
//Given an input with shape [n,c,xy] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNCHW(float *buf, const float* biases, int nSize, int cSize, int xySize);
void customCudaAddNCBiasInplaceNCHW(half *buf, const half* biases, int nSize, int cSize, int xySize);
//Given an input with shape [n,xy,c] and biases of shape [n,c], add the biases in-place.
void customCudaAddNCBiasInplaceNHWC(float *buf, const float* biases, int nSize, int xySize, int cSize);
void customCudaAddNCBiasInplaceNHWC(half *buf, const half* biases, int nSize, int xySize, int cSize);

//Given an input with shape [n,c,xy] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply an activation.
//Optionally also multiply by mask (can be null), with shape [n,xy]
void customCudaApplyCScaleBiasNCHW(const float* in, float* out, const float* scale, const float* biases, const float* mask, int n, int c, int xy, int activation);
void customCudaApplyCScaleBiasNCHW(const half* in, half* out, const half* scale, const half* biases, const half* mask, int n, int c, int xy, int activation);
//Given an input with shape [n,xy,c] and scale and biases of shape [c], multiply by scale and add the biases
//Optionally also apply relu.
//Optionally also multiply by mask (can be null), with shape [n,xy]
void customCudaApplyCScaleBiasNHWC(const float* in, float* out, const float* scale, const float* biases, const float* mask, int n, int xy, int c, int activation);
void customCudaApplyCScaleBiasNHWC(const half* in, half* out, const half* scale, const half* biases, const half* mask, int n, int xy, int c, int activation);


/*
Input tensor of dimensions (N, H, W, C)
Output tensor of dimensions (N, H_inner, W_inner, H_outer, W_outer, C)
where H_outer = ceil(H/3), W_outer = ceil(H/3), H_inner = 3, W_inner = 3.

Sets
out[n, hi, wi, ho, wo, c] = in[n, ho*3+hi, wo*3+wi, c]
except where ho*3+hi >= H, or wo*3+wi >= W, in which case the output is set to 0.
*/
void customCudaDilationTransposeNHWC(const float* in, float* out, int nSize, int hSize, int wSize, int cSize);
void customCudaDilationTransposeNHWC(const half* in, half* out, int nSize, int hSize, int wSize, int cSize);
// Inverse transforms
void customCudaDilationUntransposeNHWC(const float* in, float* out, int nSize, int hSize, int wSize, int cSize);
void customCudaDilationUntransposeNHWC(const half* in, half* out, int nSize, int hSize, int wSize, int cSize);

/*
Input tensor of dimensions (N, C, H, W)
Output tensor of dimensions (N, H_inner, W_inner, C, H_outer, W_outer)
where H_outer = ceil(H/3), W_outer = ceil(H/3), H_inner = 3, W_inner = 3.

Sets
out[n, hi, wi, c, ho, wo] = in[n, c, ho*3+hi, wo*3+wi]
except where ho*3+hi >= H, or wo*3+wi >= W, in which case the output is set to 0.
*/
void customCudaDilationTransposeNCHW(const float* in, float* out, int nSize, int cSize, int hSize, int wSize);
void customCudaDilationTransposeNCHW(const half* in, half* out, int nSize, int cSize, int hSize, int wSize);
// Inverse transforms
void customCudaDilationUntransposeNCHW(const float* in, float* out, int nSize, int cSize, int hSize, int wSize);
void customCudaDilationUntransposeNCHW(const half* in, half* out, int nSize, int cSize, int hSize, int wSize);

/*
Sets
out[n, hi, wi, ho, wo] = 1
except where ho*3+hi >= H, or wo*3+wi >= W, in which case the output is set to 0.
*/
void customCudaDilationFillMask(float* out, int nSize, int hSize, int wSize);
void customCudaDilationFillMask(half* out, int nSize, int hSize, int wSize);


#endif  // NEURALNET_CUDAHELPERS_H_
