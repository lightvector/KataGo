# build_attention.py - AOT-compile the fused single-head-per-core attention op.
#
# One attention block instance in one dispatch: each AIE core owns one head,
# keeps K/V resident (host pre-packs each head's K and V back to back, S -> 384
# rows zero-padded), streams Q in 8-row blocks, and computes
# QK^T -> softmax -> P*V on-chip (f32 out). This replaces 3 dispatches
# (QK^T GEMM, softmax op, P*V GEMM) plus all the host marshaling between them
# with a single dispatch.
#
# A core tile has only two input DMA channels, so K and V travel as one
# packed buffer per head.
#
# Host ABI (opcode-3, like the GEMM/ops):
#   arg3: Q  [heads][384 x d] bf16, per-head contiguous, host-pre-tiled
#         (mmul A-tile order per 8x32 chunk, pad rows zero), PRE-SCALED by
#         1/sqrt(d) -- the kernel applies no attention scale
#   arg4: KV [heads][2 x 384 x d] bf16, per head: K then V, host-pre-tiled
#         (mmul B-tile order), pad rows zero
#   arg5: C  [heads][384 x dv] f32, per-head contiguous, C-tile order
#         (host un-tiles on readback)
#
# (heads == kv_heads for the models this was built for; GQA would need each
# core's KV tap aimed at its group head -- not implemented here.)
#
# No masking support: masked (smaller-than-geometry) boards must fall back to
# the staged path. RoPE is applied by the caller before packing.
#
# Usage (inside the iron env):
#   python build_attention.py --heads 6 --s 361 --xclbin out.xclbin --insts out.insts.bin

import argparse
import aiecc_cleanup
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16

KERNEL_CC = str(Path(__file__).parent / "kernels" / "attention_head.cc")

Q_BLOCK = 8  # query rows per tile (L1 budget: KV resident + 8x384 scores)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def attention(
    q_in: In,
    kv_in: In,
    c_out: Out,
    *,
    s_pad: CompileTime[int],
    heads: CompileTime[int],
    d: CompileTime[int],
    dv: CompileTime[int],
    s_real: CompileTime[int],
    heads_per_core: CompileTime[int],
):
    n_cores = heads // heads_per_core  # one core's heads are processed serially
    q_blocks = s_pad // Q_BLOCK

    q_chunk_ty = np.ndarray[(Q_BLOCK, d), np.dtype[bfloat16]]
    kv_head_ty = np.ndarray[(2 * s_pad, d), np.dtype[bfloat16]]  # K then V
    c_chunk_ty = np.ndarray[(Q_BLOCK, dv), np.dtype[np.float32]]

    q_ty = np.ndarray[(heads * s_pad, d), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(heads * 2 * s_pad, d), np.dtype[bfloat16]]
    c_ty = np.ndarray[(heads * s_pad, dv), np.dtype[np.float32]]

    # Q's fifo elements are 8-row chunks; the runtime splits each head's
    # (s_pad, d) strip tap into the 48 chunk transfers automatically.
    of_q = [ObjectFifo(q_chunk_ty, name=f"q_{i}", depth=1) for i in range(n_cores)]
    of_kv = [ObjectFifo(kv_head_ty, name=f"kv_{i}", depth=1) for i in range(n_cores)]
    of_c = [ObjectFifo(c_chunk_ty, name=f"c_{i}", depth=1) for i in range(n_cores)]

    kern = ExternalFunction(
        "attn_block_bf16",
        source_file=KERNEL_CC,
        arg_types=[
            np.ndarray[(Q_BLOCK, d), np.dtype[bfloat16]],
            kv_head_ty,
            c_chunk_ty,
        ],
        compile_flags=[f"-DATTN_S_REAL={s_real}"],
        include_dirs=[config.cxx_header_path()],
    )

    # One head's K+V is resident at a time (two heads' would not fit in L1);
    # with several heads per core they run one after another, the KV buffer
    # released and refilled between them.
    def core_fn(ofq, ofkv, ofc, k):
        for _ in range_(heads_per_core):
            ekv = ofkv.acquire(1)
            for _ in range_(q_blocks):
                eq = ofq.acquire(1)
                eo = ofc.acquire(1)
                k(eq, ekv, eo)
                ofq.release(1)
                ofc.release(1)
            ofkv.release(1)

    workers = [
        Worker(core_fn, [of_q[i].cons(), of_kv[i].cons(), of_c[i].prod(), kern])
        for i in range(n_cores)
    ]

    # Q/KV/C are all per-head contiguous flat taps. Q/K/V are pre-tiled by the
    # host into the mmul tile order the stock template reads (see
    # attention_head.cc's layout note); C comes back in C-tile order and the
    # host un-tiles it. Core i handles heads [i*heads_per_core, +heads_per_core).
    q_taps = TensorTiler2D.simple_tiler((heads * s_pad, d), (s_pad, d))
    kv_taps = TensorTiler2D.simple_tiler((heads * 2 * s_pad, d), (2 * s_pad, d))
    c_taps = TensorTiler2D.simple_tiler((heads * s_pad, dv), (s_pad, dv))

    def sequence(q, kv, c, q_prods, kv_prods, c_conses):
        for i in range(n_cores):
            for j in range(heads_per_core):
                kv_prods[i].fill(kv, kv_taps[i * heads_per_core + j])
                q_prods[i].fill(q, q_taps[i * heads_per_core + j])
        for i in range(n_cores):
            for j in range(heads_per_core):
                c_conses[i].drain(c, c_taps[i * heads_per_core + j], wait=True)

    rt = Runtime(
        sequence,
        [
            q_ty,
            kv_ty,
            c_ty,
            [f.prod() for f in of_q],
            [f.prod() for f in of_kv],
            [f.cons() for f in of_c],
        ],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="build_attention")
    p.add_argument("--heads", type=int, required=True)
    p.add_argument("--d", type=int, default=32, help="qHeadDim = vHeadDim")
    p.add_argument("--s", type=int, default=361, help="real sequence length (board points)")
    p.add_argument("--heads-per-core", type=int, default=1,
                   help="heads processed serially per core; >1 needed when heads "
                        "exceed the shim's DMA channel count (e.g. 12 heads -> 2 per core)")
    p.add_argument("--xclbin", type=str, required=True)
    p.add_argument("--insts", type=str, required=True)
    p.add_argument("--keep-work", action="store_true",
                   help="keep the aiecc .prj work tree (only useful when debugging a compile)")
    args = p.parse_args()

    s_pad = ((args.s + 31) // 32) * 32  # softmax vector width multiple
    if s_pad % Q_BLOCK != 0:
        p.error(f"padded S {s_pad} must be a multiple of Q_BLOCK {Q_BLOCK}")
    if args.d != 32:
        p.error("prototype hardcodes head dim 32 (score buffer sizing)")
    if args.heads % args.heads_per_core != 0:
        p.error("heads must be a multiple of heads-per-core")

    iron.set_current_device(from_name("npu2", n_cols=None))
    start = time.perf_counter()
    spec = attention.specialize(
        s_pad=s_pad, heads=args.heads, d=args.d, dv=args.d, s_real=args.s,
        heads_per_core=args.heads_per_core,
    )
    spec.compile(xclbin_path=args.xclbin, inst_path=args.insts)
    # Only the .xclbin and .insts.bin are inputs to anything after this; the
    # rest of what aiecc wrote is scratch. See aiecc_cleanup.
    aiecc_cleanup.clean(args.xclbin, keep=args.keep_work)
    secs = time.perf_counter() - start
    print(f"attention h{args.heads} d{args.d} S{args.s}->{s_pad} "
          f"({args.heads_per_core}/core) compiled in {secs:.1f}s")
    print(f"  -> {args.xclbin}")
    print(f"  -> {args.insts}")


if __name__ == "__main__":
    main()
