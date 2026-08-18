# gemm_bf16.py
#
# Multi-core (whole-array) bf16 x bf16 -> fp32 GEMM, AOT-compiled for AMD
# Ryzen AI NPUs: XDNA1 / AIE2 / mlir-aie `npu` (Phoenix/Hawk Point, 4 cols)
# and XDNA2 / AIE2P / mlir-aie `npu2` (Strix/Krackan, 8 cols).
#
# Derived from mlir-aie's
# programming_examples/basic/matrix_multiplication/whole_array/whole_array.py
# (see that file for the design's own documentation).  KataGo-specific
# changes:
#   * dtypes are fixed to bfloat16 inputs / float32 output (bf16 multiply,
#     fp32 accumulate) — the host ABI is unchanged from the previous
#     single-core design (plain row-major A/B/C host buffers, same XRT
#     dispatch), only the on-chip tiling is parameterized;
#   * `--dtype bfp16` selects AIE2P's BFP16-emulated bf16 MMUL
#     (`emulate_bf16_mmul_with_bfp16`, mac_dims (8,8,8) instead of (4,8,8));
#     ignored on AIE2, where the flag is not supported;
#   * the shape/tile geometry is fully parameterized (arch, n_aie_cols,
#     M/K/N, m/k/n) so build_kernels.py can emit the whole artifact matrix.
#
# Tile/DMA constraints (enforced by validate_shape, mirroring the template):
#   * M % (m * n_aie_rows) == 0          (n_aie_rows = 4)
#   * M / (m * n_aie_rows) must be EVEN  (transfer-block ping-pong pairing;
#                                         odd counts fail in TensorTiler2D)
#   * K % k == 0
#   * N % (n * n_aie_cols) == 0
#   * (m, k, n) % (r, s, t) == 0         (mac_dims of the MMUL kernel:
#                                         (4,8,8) aie2p bf16, (8,8,8) aie2p
#                                         bfp16-emulated, (4,8,4) aie2)
#   * core-local memory 64 KiB: depth-2 A/B tiles + depth-2 C tile + stack
#     (0xD00) must fit — rules out e.g. m=k=n=64 with fp32 C.
#
# Host ABI for the pure C++ + XRT loader (see artifacts/manifest.json):
#   xrt::kernel(hw_context, "MLIR_AIE")
#   arg 0 : uint32 opcode = 3
#   arg 1 : xrt::bo instruction buffer, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1)
#   arg 2 : uint32 n_instr (instruction words, insts_bytes / 4)
#   arg 3 : A — M*K bfloat16, row-major, contiguous (XRT_BO_FLAGS_HOST_ONLY, group_id(3))
#   arg 4 : B — K*N bfloat16, row-major, contiguous (XRT_BO_FLAGS_HOST_ONLY, group_id(4))
#   arg 5 : C — M*N float32,  row-major, contiguous (XRT_BO_FLAGS_HOST_ONLY, group_id(5))
#
# Usage (inside the activated mlir-aie iron environment):
#   python gemm_bf16.py -M 512 -K 512 -N 512 -m 32 -k 64 -n 32 \
#        --arch npu2 --n-aie-cols 8 --dtype bf16          # compile + run + verify
#   python gemm_bf16.py ... --xclbin-path out.xclbin --insts-path out.insts.bin
#                                                            # AOT compile only

import argparse
import time

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
from aie.utils.benchmark import run_iters
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# Target configuration
# ---------------------------------------------------------------------------

# arch name -> (iron device name, aiecc arch, human description, max columns)
ARCHES = {
    "npu1": ("npu", "aie2", "XDNA1 (Ryzen AI Phoenix/Hawk Point)", 4),
    "npu2": ("npu2", "aie2p", "XDNA2 (Ryzen AI Strix/Krackan)", 8),
}

N_AIE_ROWS = 4  # compute rows on both generations (mem/shim rows not counted)

# Host ABI dtypes (fixed): bf16 in, fp32 out/accumulate.
INPUT_DTYPE = bfloat16
OUTPUT_DTYPE = np.float32
DTYPE_IN_STR = "bf16"
DTYPE_OUT_STR = "f32"

# Default tile geometry measured on Strix (see artifacts/manifest.json
# "tile_selection" notes written by build_kernels.py).
DEFAULT_TILE = {"m": 32, "k": 64, "n": 32}


def _device_for(arch: str, n_aie_cols: int):
    """Iron device for (arch, cols).

    On npu1 pick the matching ColN variant (or the full NPU1 when cols == 4).
    On npu2 use the unrestricted device regardless of cols so the placer has
    the full 8-column array.
    """
    dev_str = ARCHES[arch][0]
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def set_target_device(arch: str, n_aie_cols: int):
    """Explicitly bind the target device for codegen and kernel selection."""
    iron.set_current_device(_device_for(arch, n_aie_cols))


def validate_shape(
    arch: str, n_aie_cols: int, M: int, K: int, N: int, m: int, k: int, n: int
) -> None:
    """Raise ValueError on any geometry the design cannot express."""
    max_cols = ARCHES[arch][3]
    if not 1 <= n_aie_cols <= max_cols:
        raise ValueError(f"{arch} supports 1..{max_cols} AIE columns, got {n_aie_cols}")
    if M % (m * N_AIE_ROWS) != 0:
        raise ValueError(
            f"M={M} must be a multiple of m*n_aie_rows ({m}*{N_AIE_ROWS})"
        )
    n_row_blocks = M // m // N_AIE_ROWS
    if n_row_blocks % 2 != 0:
        raise ValueError(
            f"M/m/n_aie_rows = {n_row_blocks} must be even (transfer-block "
            f"ping-pong pairing); try a different m or M"
        )
    if K % k != 0:
        raise ValueError(f"K={K} must be a multiple of k={k}")
    if N % (n * n_aie_cols) != 0:
        raise ValueError(
            f"N={N} must be a multiple of n*n_aie_cols ({n}*{n_aie_cols})"
        )


# ---------------------------------------------------------------------------
# Design (verbatim copy of whole_array.py's _build_design minus the
# generate_taps visualization mode; b_col_maj/c_col_maj are always 0 for the
# KataGo ABI but the branches are kept faithful to the template)
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
    b_col_maj,
    c_col_maj,
    emulate_bf16_mmul_with_bfp16,
    use_chess,
    scalar,
):
    """Build the whole-array matmul IRON design and resolve to MLIR."""
    dev_str = "npu2" if isinstance(dev, NPU2) else "npu"

    n_aie_rows = N_AIE_ROWS
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        c_col_maj=bool(c_col_maj),
        use_chess=use_chess,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
        vectorized=not scalar,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    if dev_str == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev_str == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    assert (
        M % (m * n_aie_rows) == 0
    ), "A must be tileable into (m * n_aie_rows, k)-sized blocks"
    assert K % k == 0
    assert (
        N % (n * n_aie_cols) == 0
    ), "B must be tileable into (k, n * n_aie_cols)-sized blocks"
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
        b_dims: StreamDims = (
            [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
            if b_col_maj
            else [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        )
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=b_dims,
            )
        )

        c_dims: StreamDims = (
            [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
            if not c_col_maj
            else [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
        )
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

    def core_fn(in_a, in_b, out_c, zero, matmul):
        loop = range(1)  # Workaround for issue #1547
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
            ],
            stack_size=0xD00,
        ),
    )

    tb_max_n_rows = 4 if not c_col_maj else 2
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    if b_col_maj:
        B_tiles = TensorTiler2D.step_tiler(
            (N, K),
            (n, k),
            tile_group_repeats=(N // n // n_aie_cols, K // k),
            tile_group_steps=(n_aie_cols, 1),
            prune_step=False,
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),
            (k, n),
            tile_group_repeats=(K // k, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            tile_group_col_major=True,
            prune_step=False,
        )
    if c_col_maj:
        # Splitting n_aie_rows out of the tile dim is what lets TensorTiler emit
        # the (col-fast, row_block-slow) DMA pattern; iter_col_major matches it.
        C_tiles = TensorTiler2D.step_tiler(
            (N, M),
            (n, m),
            tile_group_repeats=(N // n // n_aie_cols, n_aie_rows),
            tile_group_steps=(n_aie_cols, 1),
            iter_col_major=True,
            prune_step=False,
        )
    else:
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
def gemm_bf16(
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
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
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
        0,  # b_col_maj
        0,  # c_col_maj
        emulate_bf16_mmul_with_bfp16,
        False,  # use_chess
        False,  # scalar
    )


# ---------------------------------------------------------------------------
# Host-side helpers (compile / metadata / verify)
# ---------------------------------------------------------------------------


def compile_aot(
    arch: str,
    n_aie_cols: int,
    dtype: str,
    M: int,
    K: int,
    N: int,
    m: int,
    k: int,
    n: int,
    xclbin_path: str,
    insts_path: str,
) -> float:
    """AOT-compile one variant to explicit paths; returns wall-clock seconds.

    Explicit output paths bypass the JIT cache, so this is always a true
    cold compile.
    """
    validate_shape(arch, n_aie_cols, M, K, N, m, k, n)
    set_target_device(arch, n_aie_cols)
    start = time.perf_counter()
    spec = gemm_bf16.specialize(
        M=M,
        K=K,
        N=N,
        m=m,
        k=k,
        n=n,
        n_aie_cols=n_aie_cols,
        emulate_bf16_mmul_with_bfp16=(dtype == "bfp16"),
    )
    spec.compile(xclbin_path=xclbin_path, inst_path=insts_path)
    return time.perf_counter() - start


def kernel_info(arch: str, n_aie_cols: int, dtype: str, m: int, k: int, n: int) -> dict:
    """Static kernel metadata for the manifest (no design compilation)."""
    set_target_device(arch, n_aie_cols)
    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=INPUT_DTYPE,
        output_dtype=OUTPUT_DTYPE,
        emulate_bf16_mmul_with_bfp16=(dtype == "bfp16"),
        vectorized=True,
    )
    return {
        "tile": {"m": m, "k": k, "n": n},
        "mac_dims": list(matmul_kernel.mac_dims),
        "core_kernel": getattr(matmul_kernel, "_name", "matmul_bf16_f32"),
        "zero_kernel": getattr(matmul_kernel.zero, "_name", "zero_f32"),
        "input_dtype": "bfloat16",
        "output_dtype": "float32",
        "iron_device": ARCHES[arch][0],
        "aie_arch": ARCHES[arch][1],
    }


def run_and_verify(
    arch: str,
    n_aie_cols: int,
    dtype: str,
    M: int,
    K: int,
    N: int,
    m: int,
    k: int,
    n: int,
    warmup: int = 3,
    iters: int = 10,
    seed: int = 0,
) -> bool:
    """Run one variant on the NPU and compare against a float64 reference."""
    validate_shape(arch, n_aie_cols, M, K, N, m, k, n)
    set_target_device(arch, n_aie_cols)
    emulate = dtype == "bfp16"
    print(f"\n=== Running gemm_bf16 {M}x{K}x{N} {dtype} {arch} cols={n_aie_cols} "
          f"tile={m}x{k}x{n} ===")
    rng = np.random.default_rng(seed)
    a_np = rng.uniform(-1.0, 1.0, size=(M, K)).astype(bfloat16)
    b_np = rng.uniform(-1.0, 1.0, size=(K, N)).astype(bfloat16)

    input0 = iron.tensor(a_np, dtype=bfloat16, device="npu")
    input1 = iron.tensor(b_np, dtype=bfloat16, device="npu")
    output = iron.zeros(M * N, dtype=OUTPUT_DTYPE, device="npu")

    # bf16 x bf16 products are exact in fp32, so a float64 reference from the
    # already-rounded bf16 inputs agrees with the NPU up to fp32 accumulation
    # ordering (~1e-3 absolute at K=768).
    ref = a_np.astype(np.float64) @ b_np.astype(np.float64)

    bench = run_iters(
        gemm_bf16,
        input0,
        input1,
        output,
        M=M,
        K=K,
        N=N,
        m=m,
        k=k,
        n=n,
        n_aie_cols=n_aie_cols,
        emulate_bf16_mmul_with_bfp16=emulate,
        warmup=warmup,
        iters=iters,
    )

    got = output.numpy().reshape(M, N).astype(np.float64)
    abs_err = np.abs(got - ref)
    rel_err = abs_err / np.maximum(np.abs(ref), 1e-6)
    print(
        f"  max abs err: {abs_err.max():.6g}   "
        f"max rel err: {rel_err.max():.6g}   "
        f"|C| max: {np.abs(ref).max():.6g}"
    )
    rtol, atol = 0.02, 0.1
    ok = bool(np.all(abs_err <= atol + rtol * np.abs(ref)))
    if bench.npu is not None:
        avg_us = bench.npu.avg_us
        gflops = 2.0 * M * K * N / (1000.0 * avg_us)
        print(f"  NPU time avg: {avg_us:.1f} us   {gflops:.2f} GFLOPS")
    if not ok:
        n_bad = int(np.sum(abs_err > atol + rtol * np.abs(ref)))
        print(f"  FAIL: {n_bad}/{ref.size} elements outside rtol={rtol} atol={atol}")
        return False
    print(f"  PASS (rtol={rtol}, atol={atol})")
    return True


def main():
    parser = argparse.ArgumentParser(prog="gemm_bf16")
    parser.add_argument("--arch", choices=list(ARCHES), default="npu2")
    parser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "bfp16"],
        default="bf16",
        help="bf16 = native bf16 MMUL; bfp16 = BFP16-emulated bf16 MMUL "
        "(AIE2P only, ~1.4x on Strix)",
    )
    parser.add_argument("-M", type=int, required=True)
    parser.add_argument("-K", type=int, required=True)
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("-m", type=int, default=DEFAULT_TILE["m"])
    parser.add_argument("-k", type=int, default=DEFAULT_TILE["k"])
    parser.add_argument("-n", type=int, default=DEFAULT_TILE["n"])
    parser.add_argument("--xclbin-path", type=str, default=None,
                        help="compile-only mode: write the xclbin here")
    parser.add_argument("--insts-path", type=str, default=None,
                        help="compile-only mode: write the instruction binary here")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    if (args.xclbin_path is None) != (args.insts_path is None):
        parser.error("--xclbin-path and --insts-path must be given together")
    if args.dtype == "bfp16" and args.arch != "npu2":
        parser.error("--dtype bfp16 (BFP16 emulation) is only supported on npu2")

    try:
        validate_shape(args.arch, args.n_aie_cols, args.M, args.K, args.N,
                       args.m, args.k, args.n)
    except ValueError as e:
        parser.error(str(e))

    if args.xclbin_path is not None:
        secs = compile_aot(
            args.arch, args.n_aie_cols, args.dtype,
            args.M, args.K, args.N, args.m, args.k, args.n,
            args.xclbin_path, args.insts_path,
        )
        print(f"AOT compiled {args.M}x{args.K}x{args.N} {args.dtype} "
              f"{args.arch} cols={args.n_aie_cols} tile={args.m}x{args.k}x{args.n} "
              f"in {secs:.1f}s -> {args.xclbin_path}, {args.insts_path}")
        return

    if args.arch != "npu2":
        parser.error("on-hardware verification requires an attached npu2 device; "
                     "use --xclbin-path/--insts-path for compile-only npu1 builds")
    if not run_and_verify(
        args.arch, args.n_aie_cols, args.dtype,
        args.M, args.K, args.N, args.m, args.k, args.n,
        warmup=args.warmup, iters=args.iters,
    ):
        raise SystemExit(1)
    print("PASS!")


if __name__ == "__main__":
    main()
