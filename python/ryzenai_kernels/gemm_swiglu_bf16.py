# gemm_swiglu_bf16.py
#
# Variant of gemm_bf16.py whose cores apply a SwiGLU epilogue to the
# accumulated C tile before it leaves the core. Everything else -- the
# whole_array design, the DMA layout, the host ABI (row-major bf16 A/B in,
# row-major f32 C out, same opcode-3 dispatch) -- is identical to the plain
# GEMM, so the same runtime-generated instruction streams drive it.
#
# The epilogue expects B's columns pre-interleaved by the host in groups of 8:
# [linear1 ch 0-7, linearGate ch 0-7, linear1 ch 8-15, ...] and writes
# silu(l1) * gate back over the linear1 positions. The host reads C column
# (c>>3)*16 + (c&7) for out channel c and ignores the gate columns. See
# kernels/mm_swiglu_epilogue.cc.
#
# Compile-only: the on-hardware check goes through the C++ probe in the
# scratchpad (this mlir-aie build's Python side cannot allocate NPU tensors).
#
# Usage (inside the activated mlir-aie iron environment):
#   python gemm_swiglu_bf16.py -K 512 --n-aie-cols 4 \
#        --xclbin-path out.xclbin --insts-path out.insts.bin

import argparse
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    StreamDims,
    TaskGroup,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, from_name
from aie.iron.kernel import ExternalFunction
from aie.utils import config
from ml_dtypes import bfloat16

KERNEL_CC = str(Path(__file__).parent / "kernels" / "mm_swiglu_epilogue.cc")

# ---------------------------------------------------------------------------
# Target configuration (identical to gemm_bf16.py)
# ---------------------------------------------------------------------------

ARCHES = {
    "npu1": ("npu", "aie2", "XDNA1 (Ryzen AI Phoenix/Hawk Point)", 4),
    "npu2": ("npu2", "aie2p", "XDNA2 (Ryzen AI Strix/Krackan)", 8),
}

N_AIE_ROWS = 4

INPUT_DTYPE = bfloat16
OUTPUT_DTYPE = np.float32
DTYPE_IN_STR = "bf16"
DTYPE_OUT_STR = "f32"

DEFAULT_TILE = {"m": 32, "k": 64, "n": 32}


def _device_for(arch: str, n_aie_cols: int):
    dev_str = ARCHES[arch][0]
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def set_target_device(arch: str, n_aie_cols: int):
    iron.set_current_device(_device_for(arch, n_aie_cols))


def validate_shape(
    arch: str, n_aie_cols: int, M: int, K: int, N: int, m: int, k: int, n: int
) -> None:
    max_cols = ARCHES[arch][3]
    if not 1 <= n_aie_cols <= max_cols:
        raise ValueError(f"{arch} supports 1..{max_cols} AIE columns, got {n_aie_cols}")
    if M % (m * N_AIE_ROWS) != 0:
        raise ValueError(f"M={M} must be a multiple of m*n_aie_rows ({m}*{N_AIE_ROWS})")
    n_row_blocks = M // m // N_AIE_ROWS
    if n_row_blocks % 2 != 0:
        raise ValueError(
            f"M/m/n_aie_rows = {n_row_blocks} must be even (transfer-block "
            f"ping-pong pairing); try a different m or M"
        )
    if K % k != 0:
        raise ValueError(f"K={K} must be a multiple of k={k}")
    if N % (n * n_aie_cols) != 0:
        raise ValueError(f"N={N} must be a multiple of n*n_aie_cols ({n}*{n_aie_cols})")
    # The epilogue pairs adjacent 8-column sub-tiles, so each core's n-column
    # slice must hold whole (l, g) pairs.
    if n % 16 != 0:
        raise ValueError(f"n={n} must be a multiple of 16 for the swiglu pairing")


# ---------------------------------------------------------------------------
# Design: gemm_bf16.py's whole-array build plus the epilogue call in core_fn
# ---------------------------------------------------------------------------


def _build_design(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
):
    dev_str = "npu2" if isinstance(dev, NPU2) else "npu"

    n_aie_rows = N_AIE_ROWS
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=False,
        c_col_maj=False,
        use_chess=False,
        emulate_bf16_mmul_with_bfp16=False,
        vectorized=True,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    assert M % (m * n_aie_rows) == 0
    assert K % k == 0
    assert N % (n * n_aie_cols) == 0
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_taps = []
    B_taps = []
    C_taps = []

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    swiglu_kernel = ExternalFunction(
        "mm_swiglu_epilogue_f32",
        source_file=KERNEL_CC,
        arg_types=[C_l1_ty],
        compile_flags=[f"-DDIM_M={m}", f"-DDIM_N={n}"],
        include_dirs=[config.cxx_header_path()],
    )

    A_l3l2_fifos: list[ObjectFifo] = []
    A_l2l1_fifos: list[ObjectFifo] = []
    B_l3l2_fifos: list[ObjectFifo] = []
    B_l2l1_fifos: list[ObjectFifo] = []
    C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(n_aie_rows)]
    C_l2l3_fifos: list[ObjectFifo] = []

    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        a_dims: list[StreamDims] = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = a_l3l2.cons().split(
            of_offsets,
            obj_types=[A_l1_ty] * (stop_row - start_row),
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=a_dims,
        )
        A_l2l1_fifos.extend(a_tmp_fifos)

    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        b_dims: StreamDims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=b_dims,
            )
        )

        c_dims: StreamDims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
        c_l2l3 = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=c_dims,
        )
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_aie_rows)]

        c_tmp_fifos = c_l2l3.prod().join(
            of_offsets,
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j].append(c_tmp_fifos[j])

    def core_fn(in_a, in_b, out_c, zero, matmul, swiglu):
        loop = range(1)
        if n_tiles_per_core > 1:
            loop = range_(n_tiles_per_core)
        for _ in loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)

            for _ in range_(K // k):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                in_a.release(1)
                in_b.release(1)
            # K reduction done: C tile holds the raw GEMM. Apply silu(l)*g over
            # the (l, g) sub-tile pairs in place, then hand the tile off.
            swiglu(elem_out)
            out_c.release(1)

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
                swiglu_kernel,
            ],
            stack_size=0xD00,
        ),
    )

    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    B_tiles = TensorTiler2D.step_tiler(
        (K, N),
        (k, n),
        tile_group_repeats=(K // k, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        tile_group_col_major=True,
        prune_step=False,
    )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N),
        (m * n_aie_rows, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        prune_step=False,
    )
    flat_workers = [w for row in workers for w in row]

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        c_index = 0
        tg = TaskGroup()
        for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
            for pingpong in [0, 1]:
                if c_index >= len(C_tiles):
                    break

                row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                current_tb_n_rows = min(
                    [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                )

                for col in range(n_aie_cols):
                    C_taps.append(C_tiles[c_index])
                    C_hs[col].drain(
                        C,
                        tap=C_tiles[c_index],
                        wait=True,
                        group=tg,
                    )
                    c_index += 1

                    for tile_row in range(current_tb_n_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)
                        if col < n_aie_rows:
                            A_hs[col].fill(
                                A,
                                tap=A_tiles[tile_offset],
                                group=tg,
                            )
                        B_hs[col].fill(
                            B,
                            tap=B_tiles[col],
                            group=tg,
                        )
                        A_taps.append(A_tiles[tile_offset])
                        B_taps.append(B_tiles[col])

                if tb > 0 or (tb == 0 and pingpong > 0):
                    tg.finish()
                    tg = TaskGroup()
        tg.finish()

    rt = Runtime(
        sequence,
        [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses],
    )

    program = Program(dev, rt, workers=flat_workers)
    module = program.resolve_program()

    return module


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gemm_swiglu_bf16(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
):
    return _build_design(
        iron.get_current_device(),
        M,
        K,
        N,
        m,
        k,
        n,
        n_aie_cols,
        DTYPE_IN_STR,
        DTYPE_OUT_STR,
    )


# ---------------------------------------------------------------------------


def compile_aot(
    arch: str,
    n_aie_cols: int,
    M: int,
    K: int,
    N: int,
    m: int,
    k: int,
    n: int,
    xclbin_path: str,
    insts_path: str,
) -> float:
    """AOT-compile one variant to explicit paths; returns wall-clock seconds."""
    validate_shape(arch, n_aie_cols, M, K, N, m, k, n)
    set_target_device(arch, n_aie_cols)
    start = time.perf_counter()
    spec = gemm_swiglu_bf16.specialize(
        M=M,
        K=K,
        N=N,
        m=m,
        k=k,
        n=n,
        n_aie_cols=n_aie_cols,
    )
    spec.compile(xclbin_path=xclbin_path, inst_path=insts_path)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(prog="gemm_swiglu_bf16")
    parser.add_argument("--arch", choices=list(ARCHES), default="npu2")
    parser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("-M", type=int, default=256)
    parser.add_argument("-K", type=int, required=True)
    parser.add_argument("-N", type=int, default=0, help="0 picks n*cols*2")
    parser.add_argument("-m", type=int, default=DEFAULT_TILE["m"])
    parser.add_argument("-k", type=int, default=DEFAULT_TILE["k"])
    parser.add_argument("-n", type=int, default=DEFAULT_TILE["n"])
    parser.add_argument("--xclbin-path", type=str, required=True)
    parser.add_argument("--insts-path", type=str, required=True)
    args = parser.parse_args()

    N = args.N if args.N > 0 else args.n * args.n_aie_cols * 2
    try:
        validate_shape(args.arch, args.n_aie_cols, args.M, args.K, N,
                       args.m, args.k, args.n)
    except ValueError as e:
        parser.error(str(e))

    secs = compile_aot(
        args.arch, args.n_aie_cols,
        args.M, args.K, N, args.m, args.k, args.n,
        args.xclbin_path, args.insts_path,
    )
    print(f"AOT compiled swiglu-epilogue GEMM {args.M}x{args.K}x{N} "
          f"{args.arch} cols={args.n_aie_cols} tile={args.m}x{args.k}x{args.n} "
          f"in {secs:.1f}s -> {args.xclbin_path}, {args.insts_path}")


if __name__ == "__main__":
    main()
