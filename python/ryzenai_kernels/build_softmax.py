# build_softmax.py - AOT-compile the row-wise bf16 softmax op for one shape.
#
# The op computes softmax over each row of a [rows x width] bf16 matrix on
# npu2, 8 columns x 1 core. Width must be a multiple of 32; rows must be a
# multiple of n_cores*chunk_rows (64). The host pads:
#   - pad columns with -1e30 (exp underflows to 0: no effect on the row sum)
#   - pad rows with 0 (uniform softmax output, discarded)
# Masked-out columns (KataGo's board mask) are likewise written as -1e30 by
# the host, so the kernel needs no mask input.
#
# Origin: adapted from the softmax_bench.py left by the kernel prototyping
# session; that script's JIT path is dead on this mlir-aie version
# (iron.tensor(device="npu") is unsupported), only its AOT path is kept here.
#
# Usage (inside the iron env, see ../README or environment.md):
#   python build_softmax.py --rows 2176 --width 384 \
#       --xclbin out.xclbin --insts out.insts.bin

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

KERNEL_CC = str(Path(__file__).parent / "kernels" / "softmax_rows.cc")

N_CORES = 8  # 8 columns, one core each
CHUNK_ROWS = 8  # rows per fifo element (8*384*2 B = 6 KiB)


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def softmax_rows(
    a_in: In,
    b_out: Out,
    *,
    rows: CompileTime[int],
    width: CompileTime[int],
    chunk_rows: CompileTime[int],
    n_cores: CompileTime[int],
):
    rows_per_core = rows // n_cores
    chunk_ty = np.ndarray[(chunk_rows, width), np.dtype[bfloat16]]
    tensor_ty = np.ndarray[(rows, width), np.dtype[bfloat16]]

    of_ins = [ObjectFifo(chunk_ty, name=f"in_{i}", depth=2) for i in range(n_cores)]
    of_outs = [ObjectFifo(chunk_ty, name=f"out_{i}", depth=2) for i in range(n_cores)]

    kern = ExternalFunction(
        "softmax_rows_bf16",
        source_file=KERNEL_CC,
        arg_types=[chunk_ty, chunk_ty, np.int32, np.int32],
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(of_in, of_out, k):
        for _ in range_(rows_per_core // chunk_rows):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            k(ei, eo, chunk_rows, width)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_fn, [of_ins[i].cons(), of_outs[i].prod(), kern])
        for i in range(n_cores)
    ]

    taps = TensorTiler2D.simple_tiler((rows, width), (rows_per_core, width))

    def sequence(a, b, in_prods, out_conses):
        for i in range(n_cores):
            in_prods[i].fill(a, taps[i])
        for i in range(n_cores):
            out_conses[i].drain(b, taps[i], wait=True)

    rt = Runtime(
        sequence,
        [
            tensor_ty,
            tensor_ty,
            [of_ins[i].prod() for i in range(n_cores)],
            [of_outs[i].cons() for i in range(n_cores)],
        ],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="build_softmax")
    p.add_argument("--rows", type=int, required=True, help="padded row count (multiple of 64)")
    p.add_argument("--width", type=int, required=True, help="padded width (multiple of 32)")
    p.add_argument("--xclbin", type=str, required=True)
    p.add_argument("--insts", type=str, required=True)
    p.add_argument("--keep-work", action="store_true",
                   help="keep the aiecc .prj work tree (only useful when debugging a compile)")
    args = p.parse_args()

    if args.rows % (N_CORES * CHUNK_ROWS) != 0:
        p.error(f"rows {args.rows} must be a multiple of {N_CORES * CHUNK_ROWS}")
    if args.width % 32 != 0:
        p.error(f"width {args.width} must be a multiple of 32")

    iron.set_current_device(from_name("npu2", n_cols=None))
    start = time.perf_counter()
    spec = softmax_rows.specialize(
        rows=args.rows, width=args.width, chunk_rows=CHUNK_ROWS, n_cores=N_CORES
    )
    spec.compile(xclbin_path=args.xclbin, inst_path=args.insts)
    # Only the .xclbin and .insts.bin are inputs to anything after this; the
    # rest of what aiecc wrote is scratch. See aiecc_cleanup.
    aiecc_cleanup.clean(args.xclbin, keep=args.keep_work)
    secs = time.perf_counter() - start
    print(f"softmax [{args.rows}x{args.width}] npu2 {N_CORES}col compiled in {secs:.1f}s")
    print(f"  -> {args.xclbin}")
    print(f"  -> {args.insts}")


if __name__ == "__main__":
    main()
