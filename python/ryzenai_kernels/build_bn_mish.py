# build_bn_mish.py - AOT-compile the fused BatchNorm+Mish row op.
#
# out[r][c] = mish(scale[c] * x[r][c] + bias[c]), bf16 in/out, f32 internal.
# 8 columns x 1 core; rows are split across cores, the scale/bias vectors are
# replicated per core on the host (each core DMAs its own copy once per
# dispatch and holds it for all its rows).
#
# Host ABI (opcode-3):
#   arg3: X [rows_pad x width] bf16 (pad rows zero)
#   arg4: scale [8 x width] bf16 (replicated per core)
#   arg5: bias [8 x width] bf16 (replicated per core)
#   arg6: Y [rows_pad x width] bf16
#
# Usage:
#   python build_bn_mish.py --rows 384 --width 768 --xclbin out.xclbin --insts out.insts.bin

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

KERNEL_CC = str(Path(__file__).parent / "kernels" / "bn_mish.cc")

N_CORES = 8
CHUNK_ROWS = 4  # rows per fifo element


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def bn_mish(
    x_in: In,
    s_in: In,
    b_in: In,
    y_out: Out,
    *,
    rows: CompileTime[int],
    width: CompileTime[int],
    dumpmode: CompileTime[int] = 0,
    n_cores: CompileTime[int],
    chunk_rows: CompileTime[int],
):
    rows_per_core = rows // n_cores
    chunk_ty = np.ndarray[(chunk_rows, width), np.dtype[bfloat16]]
    row_ty = np.ndarray[(width,), np.dtype[bfloat16]]  # one param row
    tensor_ty = np.ndarray[(rows, width), np.dtype[bfloat16]]
    param_ty = np.ndarray[(n_cores, 2 * width), np.dtype[bfloat16]]

    sb_row_ty = np.ndarray[(2 * width,), np.dtype[bfloat16]]  # [scale | bias]

    of_x = [ObjectFifo(chunk_ty, name=f"x_{i}", depth=2) for i in range(n_cores)]
    of_sb = [ObjectFifo(sb_row_ty, name=f"sb_{i}", depth=1) for i in range(n_cores)]
    of_y = [ObjectFifo(chunk_ty, name=f"y_{i}", depth=2) for i in range(n_cores)]

    kern = ExternalFunction(
        "bn_mish_bf16",
        source_file=KERNEL_CC,
        arg_types=[chunk_ty, sb_row_ty, chunk_ty, np.int32, np.int32],
        compile_flags=[f"-DBNM_DUMPMODE={dumpmode}"] + (["-DBNM_NOCLAMP"] if dumpmode == 3 else []),
        include_dirs=[config.cxx_header_path()],
    )

    def core_fn(ofx, ofsb, ofy, k):
        esb = ofsb.acquire(1)
        for _ in range_(rows_per_core // chunk_rows):
            ex = ofx.acquire(1)
            ey = ofy.acquire(1)
            k(ex, esb, ey, chunk_rows, width)
            ofx.release(1)
            ofy.release(1)
        ofsb.release(1)

    workers = [
        Worker(core_fn, [of_x[i].cons(), of_sb[i].cons(), of_y[i].prod(), kern])
        for i in range(n_cores)
    ]

    x_taps = TensorTiler2D.simple_tiler((rows, width), (rows_per_core, width))
    y_taps = TensorTiler2D.simple_tiler((rows, width), (rows_per_core, width))
    p_taps = TensorTiler2D.simple_tiler((n_cores, 2 * width), (1, 2 * width))

    def sequence(x, sb, y, x_p, sb_p, y_c):
        for i in range(n_cores):
            sb_p[i].fill(sb, p_taps[i])
        for i in range(n_cores):
            x_p[i].fill(x, x_taps[i])
        for i in range(n_cores):
            y_c[i].drain(y, y_taps[i], wait=True)

    rt = Runtime(
        sequence,
        [
            tensor_ty,
            param_ty,
            tensor_ty,
            [f.prod() for f in of_x],
            [f.prod() for f in of_sb],
            [f.cons() for f in of_y],
        ],
    )
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="build_bn_mish")
    p.add_argument("--rows", type=int, required=True, help="padded rows (multiple of 32)")
    p.add_argument("--width", type=int, required=True, help="channels (multiple of 16)")
    p.add_argument("--xclbin", type=str, required=True)
    p.add_argument("--insts", type=str, required=True)
    p.add_argument("--keep-work", action="store_true",
                   help="keep the aiecc .prj work tree (only useful when debugging a compile)")
    p.add_argument("--dumpmode", type=int, default=0)
    args = p.parse_args()

    if args.rows % (N_CORES * CHUNK_ROWS) != 0:
        p.error(f"rows {args.rows} must be a multiple of {N_CORES * CHUNK_ROWS}")
    if args.width % 16 != 0:
        p.error("width must be a multiple of 16")

    iron.set_current_device(from_name("npu2", n_cols=None))
    start = time.perf_counter()
    spec = bn_mish.specialize(
        rows=args.rows, width=args.width, n_cores=N_CORES, chunk_rows=CHUNK_ROWS,
        dumpmode=args.dumpmode,
    )
    spec.compile(xclbin_path=args.xclbin, inst_path=args.insts)
    # Only the .xclbin and .insts.bin are inputs to anything after this; the
    # rest of what aiecc wrote is scratch. See aiecc_cleanup.
    aiecc_cleanup.clean(args.xclbin, keep=args.keep_work)
    print(f"bn_mish [{args.rows}x{args.width}] compiled in {time.perf_counter()-start:.1f}s")
    print(f"  -> {args.xclbin}")
    print(f"  -> {args.insts}")


if __name__ == "__main__":
    main()
