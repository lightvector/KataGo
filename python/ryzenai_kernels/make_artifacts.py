#!/usr/bin/env python3
"""Generate the NPU artifacts a model needs, into cpp/external/ryzenai_artifacts.

Everything the runtime loads is named by its shape, and the loader finds it by
that name alone, so generating for a new model is a matter of naming the shapes
it uses. This wraps the individual builders so that is one command.

    python make_artifacts.py --for-model 512 8          # trunk 512, 8 heads
    python make_artifacts.py --for-model 768 0          # a convnet: no heads
    python make_artifacts.py --gemm-grid                # the K grid (slow)
    python make_artifacts.py --list --for-model 512 8   # show, build nothing

What a model needs:

  gemm      one per K it reduces over, shared by every model - the shipped grid
            already covers K up to 6912, so a new model normally needs none
  bnmish    convnets with Mish: one per (rows, channel count)
  attn      transformers: one per (heads, board points)
  softmax   transformers: one per (heads * points, points), both padded
  swiglu    transformers: one per FFN hidden width (--ffn-hidden)

Run inside the mlir-aie iron environment with XRT on PATH:

    set PATH=C:\\Envs\\mlir-aie\\ironenv\\Lib\\site-packages\\mlir_aie\\bin;C:\\Xilinx\\XRT;%PATH%
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
ARTIFACTS = HERE.parent.parent / "cpp" / "external" / "ryzenai_artifacts"
OPS = ARTIFACTS / "ops"

# Rows per dispatch that the C++ side knows about. bnmish picks per call among
# these (matmul.cpp Accel::kBnMishHeights), so all three want to exist for any
# channel count that carries real work.
BNMISH_ROWS = [384, 1536, 3072]


def pad(n, to):
    return ((n + to - 1) // to) * to


def plan(trunk_c, heads, points):
    """Artifacts for one model geometry, as (builder, args, output stem)."""
    jobs = []
    # nbt convnets normalise on several widths, not just the trunk: the
    # bottleneck is trunk/2 and the heads run narrower still. Covering the
    # usual set costs a few minutes and saves a silent fallback later.
    widths = sorted({trunk_c, trunk_c // 2, trunk_c // 4, 256, 384})
    for w in widths:
        for rows in BNMISH_ROWS:
            # Only the trunk-sized widths ever see batched rows worth the tall
            # variants; the narrow head layers stay small.
            if rows > 384 and w < trunk_c // 2:
                continue
            jobs.append(("build_bn_mish.py",
                         ["--rows", str(rows), "--width", str(w)],
                         f"bnmish_{rows}x{w}"))
    if heads > 0:
        s_pad = pad(points, 32)
        # 12 heads exceed the shim's DMA channel budget one head per core, so
        # those run two heads per core (see SKILL.md pit #18).
        per_core = 2 if heads > 8 else 1
        jobs.append(("build_attention.py",
                     ["--heads", str(heads), "--s", str(points),
                      "--heads-per-core", str(per_core)],
                     f"attn_h{heads}_s{points}"))
        jobs.append(("build_softmax.py",
                     ["--rows", str(pad(heads * points, 64)), "--width", str(s_pad)],
                     f"softmax_{pad(heads * points, 64)}x{s_pad}"))
    return jobs


def run(jobs, list_only, force):
    OPS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if force or not (OPS / (j[2] + ".xclbin")).exists()]
    have = len(jobs) - len(todo)
    print("%d artifact(s): %d already present, %d to build" % (len(jobs), have, len(todo)))
    for _, _, stem in todo:
        print("   " + stem)
    if list_only or not todo:
        return 0

    failed = []
    for i, (script, args, stem) in enumerate(todo):
        # Builders drop an aiecc .prj work directory beside their output, so
        # build into scratch and move only the two files worth keeping.
        with tempfile.TemporaryDirectory(prefix="ryzenai_op_") as work:
            x, b = Path(work) / (stem + ".xclbin"), Path(work) / (stem + ".insts.bin")
            cmd = [sys.executable, str(HERE / script)] + args + \
                  ["--xclbin", str(x), "--insts", str(b)]
            print("[%d/%d] %s" % (i + 1, len(todo), stem), flush=True)
            if subprocess.call(cmd, cwd=str(HERE)) != 0 or not x.exists():
                print("        FAILED")
                failed.append(stem)
                continue
            shutil.move(str(x), str(OPS / x.name))
            shutil.move(str(b), str(OPS / b.name))

    print("\ndone: %d built, %d failed" % (len(todo) - len(failed), len(failed)))
    for f in failed:
        print("  failed: " + f)
    return 1 if failed else 0


def snapshot():
    return {p.relative_to(ARTIFACTS).as_posix()
            for p in ARTIFACTS.rglob("*")
            if p.suffix in (".xclbin", ".bin")}


def audit(before):
    """Report where this run's artifacts landed, and that nothing else did.

    The builders write to three different places - ops/ for the fused ops, and
    bf16|bfp16/<arch>_<cols>col[_swiglu]/ for the GEMM grids - so a per-directory
    count is the only way to see that a run produced what was asked for. It also
    catches an aiecc .prj work directory left inside the artifact tree, which
    happened once and quietly added a few hundred files to the deployed build.
    """
    after = snapshot()
    added = sorted(after - before)
    print("\nartifact tree: %d files (%d new this run)" % (len(after), len(added)))
    by_dir = {}
    for a in added:
        by_dir.setdefault(a.rsplit("/", 1)[0] if "/" in a else ".", []).append(a)
    for d in sorted(by_dir):
        print("   %-44s %d" % (d + "/", len(by_dir[d])))
    strays = [p for p in ARTIFACTS.rglob("*.prj") if p.is_dir()]
    if strays:
        print("WARNING: aiecc work directories left in the artifact tree - delete these:")
        for p in strays:
            print("   " + p.relative_to(ARTIFACTS).as_posix())
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Build NPU artifacts into cpp/external/ryzenai_artifacts/ops")
    ap.add_argument("--for-model", nargs="+", metavar="N", type=int,
                    help="trunk channels, heads (0 for a convnet), "
                         "and optionally board points (default 361)")
    ap.add_argument("--ffn-hidden", type=int, metavar="K",
                    help="FFN hidden width, for the SwiGLU epilogue GEMM. There is no reliable "
                         "formula for it (384->512 but 512->768 and 768->1152), so read it off "
                         "the model: run katago with ryzenaiShapeReport=true and take the N of "
                         "the ffn.linear1 row. Omitted means skip the SwiGLU artifacts.")
    ap.add_argument("--gemm-grid", action="store_true",
                    help="rebuild the shared K grid instead (slow; see build_grid.py)")
    ap.add_argument("--list", action="store_true", help="show the plan, build nothing")
    ap.add_argument("--force", action="store_true", help="rebuild artifacts that exist")
    args = ap.parse_args()

    before = snapshot()

    if args.gemm_grid:
        rc = subprocess.call(
            [sys.executable, str(HERE / "build_grid.py"), "--arch", "npu2",
             "--dtype", "bf16", "--dtype", "bfp16"], cwd=str(HERE))
        return audit(before) or rc

    if not args.for_model:
        ap.error("give --for-model TRUNK HEADS [POINTS], or --gemm-grid")
    trunk = args.for_model[0]
    heads = args.for_model[1] if len(args.for_model) > 1 else 0
    points = args.for_model[2] if len(args.for_model) > 2 else 361
    print("model: trunk %d, %d head(s), %d board points" % (trunk, heads, points))
    rc = run(plan(trunk, heads, points), args.list, args.force)

    # The SwiGLU epilogue variants live under bf16/<arch>_<cols>col_swiglu/ rather than ops/, and
    # build_swiglu_grid.py already puts them there and skips what exists, so hand off to it.
    if heads > 0 and args.ffn_hidden:
        print("\nSwiGLU epilogue GEMM for FFN hidden %d" % args.ffn_hidden)
        cmd = [sys.executable, str(HERE / "build_swiglu_grid.py"), "--k", str(args.ffn_hidden)]
        if args.list:
            print("   would run: " + " ".join(cmd[1:]))
        else:
            rc = subprocess.call(cmd, cwd=str(HERE)) or rc
    elif heads > 0:
        print("\nNote: --ffn-hidden not given, so no SwiGLU epilogue artifact was considered.\n"
              "      Without one the FFN still runs, just without the fused activation.")

    if not args.list:
        rc = audit(before) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
