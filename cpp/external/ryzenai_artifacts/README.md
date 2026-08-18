# RyzenAI NPU kernel artifacts

Compiled AI Engine kernel binaries for the RyzenAI backend. Everything here is a
build product of the kernel sources in `../kernels/`, committed so that neither
end users nor people compiling KataGo need the AIE toolchain.

CMake copies this whole directory next to `katago` as `ryzenai/` at build time,
and the backend resolves it relative to the running executable, so a deployed
install is just the executable, the XRT runtime DLLs, and this directory.

## Layout

```
manifest.json          ABI contract shared with cpp/neuralnet/ryzenai/manifest.h
bf16/*.xclbin          bfloat16 kernels
```

## Status

M2 first kernel landed: `bf16/gemm_bf16_M{M}K{K}N{N}.xclbin` + `.insts.bin`
for (M,K,N) in {(384,384,384), (384,512,512), (384,768,768)} — single-core
bf16×bf16→fp32 GEMM (64×64×64 tiles via mem tile), built by
`../kernels/build_kernels.py` from `../kernels/gemm_bf16.py`, verified on
hardware against numpy. These are per-shape AOT artifacts: M/K/N are baked
into each `.insts.bin`. The shape-independent form (runtime-generated
instruction stream per the "What an xclbin is parameterized by" section
below) is the M3 goal; `manifest.json` pins the ABI either way.

## What an xclbin is parameterized by

`(target device, core program, dataflow topology, compile-time tile sizes)` —
**not** by tensor shape. M/K/N, layer count and batch size all live in the
runtime instruction stream, so changing model, board size or batch size does
not require a different xclbin.
