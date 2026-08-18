#!/usr/bin/env python3
"""Compile the artifact grid.

An .xclbin bakes in the reduction dim K and nothing else -- M and N ride in the
instruction stream, which sequence.cpp generates at run time (see
INSTS_FORMAT.md for the measurements). So the grid is one dimensional: one
binary per (dtype, arch, columns, K), named for the K alone.

Each entry is compiled at a canonical (M, N); which one is irrelevant to the
resulting binary, it only has to be legal for the design.

Must be run inside the activated mlir-aie iron environment, with XRT on PATH so
that aiecc can find xclbinutil:

    set PATH=%LOCALAPPDATA%\\..\\mlir_aie\\bin;C:\\Xilinx\\XRT;%PATH%
    python build_grid.py --dtype bf16 --arch npu2

Run with --list to see what would be built without building it.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

ARTIFACTS = Path(__file__).parent.parent.parent / "cpp" / "external" / "ryzenai_artifacts"

# K values the four shipped models need, from RyzenAIShapes::report. K is a
# multiple of 64 for every one of them except the initial 3x3 convolution
# (9*22 = 198), which pads up into 256.
#
#   b10c384h6      192 384 512
#   b10c512h8      256 512 768
#   b11c768h12     384 768 1152
#   b40c768        384 768 2304 3456   (2304 = 9*256, 3456 = 9*384)
#
# Anything not listed still runs: the loader picks the smallest K above it and
# the caller zero-pads. That is what makes an arbitrary model work without a
# toolchain, at the price of some wasted multiply-accumulates.
MODEL_K = [192, 256, 384, 512, 768, 1152, 2304, 3456]

# A denser sweep for coverage of models we have never seen. 64 is the tile size,
# so it is the finest grid the design can express.
def sweep_k(lo, hi, step=64):
    return list(range(lo, hi + 1, step))

ARCH_COLS = {"npu1": [1, 2, 4], "npu2": [1, 2, 4, 8]}

TILE_M, TILE_K, TILE_N = 32, 64, 32


def canonical_shape(cols):
    """A legal (M, N) to compile at. The binary does not depend on either."""
    m = TILE_M * 8            # one chunk
    n = TILE_N * cols * 2     # two output strips per column
    return m, n


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_artifacts():
    """Every gemm grid xclbin on disk, as index entries.

    Recovers (dtype, arch, cols, K) from the path, which is the only thing the
    loader keys on. compiled_at cannot be recovered and is reported as null --
    it is provenance, and the note in the index already says it constrains
    nothing.
    """
    entries = []
    for xclbin in sorted(ARTIFACTS.glob("*/*col/gemm_bf16_K*.xclbin")):
        variant = xclbin.parent.name  # "npu2_4col"
        arch, _, cols = variant.rpartition("_")
        insts = xclbin.with_suffix("").with_suffix(".insts.bin")
        entries.append({
            "dtype": xclbin.parent.parent.name,
            "arch": arch,
            "n_aie_cols": int(cols[:-3]),
            "K": int(xclbin.stem.split("_K")[1]),
            "compiled_at": None,
            "tile": {"m": TILE_M, "k": TILE_K, "n": TILE_N},
            "xclbin": str(xclbin.relative_to(ARTIFACTS)).replace("\\", "/"),
            "xclbin_bytes": xclbin.stat().st_size,
            "xclbin_sha256": sha256(xclbin),
            "insts_golden": str(insts.relative_to(ARTIFACTS)).replace("\\", "/"),
        })
        if not insts.exists():
            print("warning: %s has no .insts.bin beside it" % xclbin.name)
    return entries


def build(dtypes, arches, ks, dry_run):
    # Imported here, not at module scope, so that --reindex (which only reads
    # the files already on disk) works outside the mlir-aie iron environment.
    import gemm_bf16

    jobs = []
    for dtype in dtypes:
        for arch in arches:
            if dtype == "bfp16" and arch != "npu2":
                continue  # BFP16 is an XDNA2-only micro-kernel
            for cols in ARCH_COLS[arch]:
                for k in ks:
                    jobs.append((dtype, arch, cols, k))

    print("%d artifact(s) to build" % len(jobs))
    if dry_run:
        for dtype, arch, cols, k in jobs:
            print("  %-5s %-4s %dcol K=%d" % (dtype, arch, cols, k))
        return

    entries = []
    total = 0.0
    for i, (dtype, arch, cols, k) in enumerate(jobs):
        subdir = ARTIFACTS / dtype / ("%s_%dcol" % (arch, cols))
        subdir.mkdir(parents=True, exist_ok=True)
        xclbin = subdir / ("gemm_bf16_K%d.xclbin" % k)
        insts = subdir / ("gemm_bf16_K%d.insts.bin" % k)
        m, n = canonical_shape(cols)

        if xclbin.exists():
            print("[%3d/%3d] %-5s %-4s %dcol K=%-5d  already present" %
                  (i + 1, len(jobs), dtype, arch, cols, k))
        else:
            # aiecc drops a .prj work directory beside its output, so compile
            # into a scratch dir and move only the two files worth keeping.
            try:
                with tempfile.TemporaryDirectory(prefix="ryzenai_grid_") as work:
                    tmp_x = Path(work) / xclbin.name
                    tmp_i = Path(work) / insts.name
                    secs = gemm_bf16.compile_aot(
                        arch, cols, dtype, m, k, n, TILE_M, TILE_K, TILE_N, str(tmp_x), str(tmp_i))
                    shutil.move(str(tmp_x), str(xclbin))
                    shutil.move(str(tmp_i), str(insts))
            except Exception as e:
                print("[%3d/%3d] %-5s %-4s %dcol K=%-5d  FAILED: %s" %
                      (i + 1, len(jobs), dtype, arch, cols, k, e))
                continue
            total += secs
            print("[%3d/%3d] %-5s %-4s %dcol K=%-5d  %5.1fs  %7d B" %
                  (i + 1, len(jobs), dtype, arch, cols, k, secs, xclbin.stat().st_size))

        entries.append({
            "dtype": dtype,
            "arch": arch,
            "n_aie_cols": cols,
            "K": k,
            "compiled_at": {"M": m, "N": n},
            "tile": {"m": TILE_M, "k": TILE_K, "n": TILE_N},
            "xclbin": str(xclbin.relative_to(ARTIFACTS)).replace("\\", "/"),
            "xclbin_bytes": xclbin.stat().st_size,
            "xclbin_sha256": sha256(xclbin),
            "insts_golden": str(insts.relative_to(ARTIFACTS)).replace("\\", "/"),
        })

    write_index(entries)
    print("(%.1f s of compiling)" % total)


def write_index(entries, replace=False):
    """Update grid.json.

    Merges into whatever is already on disk rather than replacing it: the grid
    is normally filled in over several invocations (one per dtype or K range),
    and a plain overwrite left grid.json describing only the last batch while
    the earlier xclbins sat there unlisted. Pass replace=True (--reindex) to
    make the index exactly the set of entries handed in.
    """
    grid_path = ARTIFACTS / "grid.json"
    key = lambda e: (e["dtype"], e["arch"], e["n_aie_cols"], e["K"])
    merged = {}
    if grid_path.exists() and not replace:
        try:
            for e in json.loads(grid_path.read_text()).get("artifacts", []):
                merged[key(e)] = e
        except (ValueError, KeyError) as e:
            print("warning: ignoring unreadable %s (%s)" % (grid_path, e))
    from_disk = len(merged)
    for e in entries:
        merged[key(e)] = e  # the caller wins on collision
    out = [merged[k] for k in sorted(merged)]

    grid_path.write_text(json.dumps({
        "grid_version": 1,
        "generated_by": "python/ryzenai_kernels/build_grid.py",
        "note": ("An xclbin depends only on (dtype, arch, n_aie_cols, K, tile). "
                 "M and N are carried by the instruction stream, which "
                 "sequence.cpp generates at run time, so compiled_at is "
                 "provenance only and imposes no constraint on dispatch."),
        "artifacts": out,
    }, indent=1))
    print("wrote %s: %d entries (%d supplied, %d carried over)"
          % (grid_path, len(out), len(entries), from_disk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", action="append", choices=["bf16", "bfp16"])
    ap.add_argument("--arch", action="append", choices=["npu1", "npu2"])
    ap.add_argument("--k", action="append", type=int, help="explicit K (repeatable)")
    ap.add_argument("--sweep", nargs=2, type=int, metavar=("LO", "HI"),
                    help="dense 64-step K sweep instead of the model list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--reindex", action="store_true",
                    help="rebuild grid.json from the xclbins on disk, compiling nothing")
    args = ap.parse_args()

    if args.reindex:
        write_index(scan_artifacts(), replace=True)
        return

    ks = args.k if args.k else (sweep_k(*args.sweep) if args.sweep else MODEL_K)
    build(args.dtype or ["bf16"], args.arch or ["npu2"], sorted(set(ks)), args.list)


if __name__ == "__main__":
    main()
