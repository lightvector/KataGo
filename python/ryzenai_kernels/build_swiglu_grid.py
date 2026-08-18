#!/usr/bin/env python3
"""Compile the SwiGLU-epilogue GEMM artifacts.

Companion to build_grid.py: one gemm_swiglu_bf16_K<K> binary per (arch, cols,
K), into bf16/<arch>_<cols>col_swiglu/. The binary bakes in only K, exactly
like the plain GEMM; M/N ride in the runtime-generated instruction stream.

Only the Ks the shipped transformer models' FFN blocks actually use (after the
forceK collapse, see RyzenAIShapes::chooseSingleK) get artifacts; any other
model falls back to the plain fused GEMM + CPU SwiGLU, which is always
correct. SwiGLU is a no-op win, never a requirement.

Must be run inside the activated mlir-aie iron environment with XRT on PATH.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gemm_swiglu_bf16

ARTIFACTS = Path(__file__).parent.parent.parent / "cpp" / "external" / "ryzenai_artifacts"

# forceK-collapsed FFN reduction dims: b10c384h6 -> 512 (inC 192),
# b10c512h8 -> 768 (inC 256), b11c768h12 -> 1152 (inC 384). b40c768 has no FFN.
MODEL_K = [512, 768, 1152]

TILE_M, TILE_K, TILE_N = 32, 64, 32


def main():
    ap = argparse.ArgumentParser(prog="build_swiglu_grid")
    ap.add_argument("--arch", choices=["npu2"], default="npu2")
    ap.add_argument("--cols", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--k", type=int, nargs="*", default=MODEL_K)
    args = ap.parse_args()

    jobs = [(c, k) for c in sorted(args.cols) for k in sorted(args.k)]
    print("%d swiglu artifact(s) to build" % len(jobs))
    total = 0.0
    for i, (cols, k) in enumerate(jobs):
        subdir = ARTIFACTS / "bf16" / ("%s_%dcol_swiglu" % (args.arch, cols))
        subdir.mkdir(parents=True, exist_ok=True)
        xclbin = subdir / ("gemm_swiglu_bf16_K%d.xclbin" % k)
        insts = subdir / ("gemm_swiglu_bf16_K%d.insts.bin" % k)
        if xclbin.exists():
            print("[%d/%d] %dcol K=%d already present" % (i + 1, len(jobs), cols, k))
            continue
        m = TILE_M * 8
        n = TILE_N * cols * 2
        try:
            with tempfile.TemporaryDirectory(prefix="ryzenai_swg_") as work:
                tmp_x = Path(work) / xclbin.name
                tmp_i = Path(work) / insts.name
                secs = gemm_swiglu_bf16.compile_aot(
                    args.arch, cols, m, k, n, TILE_M, TILE_K, TILE_N,
                    str(tmp_x), str(tmp_i))
                shutil.move(str(tmp_x), str(xclbin))
                shutil.move(str(tmp_i), str(insts))
        except Exception as e:
            print("[%d/%d] %dcol K=%d FAILED: %s" % (i + 1, len(jobs), cols, k, e))
            continue
        total += secs
        print("[%d/%d] %dcol K=%d  %.1fs  %d B" %
              (i + 1, len(jobs), cols, k, secs, xclbin.stat().st_size))
    print("done (%.1f s of compiling)" % total)


if __name__ == "__main__":
    main()
