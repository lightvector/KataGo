# RyzenAI NPU kernel generation

Offline tooling that compiles the `.xclbin` kernels the RyzenAI backend runs on
the NPU. Nothing here is needed to build or run KataGo: the kernels are
committed under `cpp/external/ryzenai_artifacts`, and `cmake --build` copies
them next to `katago.exe`. You only come here to support a network shape that
has no kernel yet.

## Do you actually need this?

Usually not. An `.xclbin` bakes in exactly one thing about the network - the
reduction dimension `K` of a matrix multiply - and the shipped grid covers every
`K` up to 6912 in steps of 64. A network whose `K` is not in the grid still
runs: the loader picks the next size up and the extra columns are zero-filled,
costing some wasted arithmetic and nothing else. Everything else about a shape,
including batch size and output width, is generated at run time by
`cpp/neuralnet/ryzenaisequence.cpp`, which is why an arbitrary `.bin.gz` works
with no Python at all.

What *does* need new kernels is a network using a fused operator geometry that
has not been compiled yet - a head count, board size, channel width or FFN
width outside the shipped set. Those operators are optional: without a matching
one the layer falls back to a slower path that produces identical numbers.

To find out, run KataGo with the shape report on:

```
katago.exe gtp -model <net>.bin.gz -config <cfg> -override-config ryzenaiShapeReport=true
```

It lists every shape the model asks for. With `ryzenaiVerboseDispatch=true` the
log also names any operator that was wanted but missing, e.g.

```
bn+mish NOT on NPU for 192 channels x 384 rows: no 384x192 artifact
```

## Setting up

The toolchain is a git checkout of [mlir-aie](https://github.com/Xilinx/mlir-aie),
not just a set of wheels - the GEMM generators compile against kernel sources in
its `aie_kernels` tree. `setup_env.ps1` clones it and runs the installer that
ships inside it, so the environment always matches that checkout:

```
.\setup_env.ps1 -Prefix C:\Envs\mlir-aie              # report the plan, download nothing
.\setup_env.ps1 -Prefix C:\Envs\mlir-aie -Execute     # clone and install
```

`-Prefix` is required and has no default - the tree is several gigabytes, so
where it goes is your decision. Any path works. Without `-Execute` the script
only checks prerequisites (git, Python 3.10+, the MSVC C++ toolchain, XRT) and
prints what it would do; it never installs any of those for you.

It writes `activate_iron.bat` here when it finishes.

## Generating

```
activate_iron.bat
python make_artifacts.py --list --for-model 512 8            # plan only, build nothing
python make_artifacts.py --for-model 512 8 --ffn-hidden 768  # a transformer
python make_artifacts.py --for-model 768 0                   # a convnet
python make_artifacts.py --gemm-grid                         # the shared K grid (hours)
```

The positional arguments are trunk channels, attention heads (`0` for a
convnet), and optionally board points (default 361, i.e. 19x19).

`--ffn-hidden` is the FFN hidden width, read off the shape report. There is no
formula for it - 384 channels goes with 512, but 512 goes with 768 and 768 with
1152 - so it has to be given. Without it the SwiGLU activation runs unfused,
which costs a few percent.

Artifacts land in `cpp/external/ryzenai_artifacts` in the layout the loader
expects, already-present ones are skipped, and the aiecc work trees are deleted
on the way out. The run ends with a per-directory count of what it added.

## What each piece is

| File | Role |
| --- | --- |
| `make_artifacts.py` | The entry point. Works out which artifacts a model geometry needs and calls the builders below. |
| `setup_env.ps1` | Clones mlir-aie and runs its installer. |
| `build_grid.py` | The shared GEMM grid, one binary per (dtype, arch, columns, K). `--reindex` rebuilds `grid.json` from what is on disk without compiling. |
| `build_swiglu_grid.py` | GEMM variants with the SwiGLU activation folded into the epilogue. |
| `build_attention.py` | Fused attention: QK^T, softmax and P*V on-chip, one dispatch per instance. |
| `build_softmax.py` | Standalone softmax rows. |
| `build_bn_mish.py` | Fused BatchNorm + Mish, for convnets. |
| `gemm_bf16.py`, `gemm_swiglu_bf16.py` | The GEMM designs the two grid drivers compile. |
| `kernels/*.cc` | AIE core sources for the fused operators. The plain GEMM uses mlir-aie's own templates instead. |
| `aiecc_cleanup.py` | Deletes the `.prj` work tree aiecc leaves beside each artifact. |
| `extract_layout.py` | Regenerates `cpp/neuralnet/ryzenaisequence_layout.h` from golden instruction streams. Only needed when changing which column counts are supported. |
| `parse_insts.py`, `INSTS_FORMAT.md` | Decoder and format notes for the DMA instruction streams that `ryzenaisequence.cpp` generates. |

## Notes

* Each builder can be run directly if you want one specific artifact;
  `--xclbin` and `--insts` say where to put it. `make_artifacts.py` exists so
  that you do not have to work out the names and geometry by hand.
* `--keep-work` on the operator builders keeps the aiecc `.prj` tree, which is
  worth doing when a compile fails - the logs in there are the only record of
  why.
* Compiling one operator takes tens of seconds. The full GEMM grid is hundreds
  of binaries and takes hours; it is committed precisely so that nobody has to.
