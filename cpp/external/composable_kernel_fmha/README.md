# Composable Kernel FMHA (fused multi-head attention)

Vendored glue headers + pre-generated kernel instantiations from AMD's Composable Kernel (CK)
`ck_tile` FMHA implementation, used by the ROCm backend as an optional fused attention path
(mirrors the CUDA backend's optional cudnn-frontend SDPA graph path). Requires the ck_tile base
headers from a `composablekernel-dev`/`amdrocm-ck*` system package (not vendored here). If not
found at configure time (see `KATAGO_CK_TILE_INCLUDE_DIR` in `cpp/CMakeLists.txt`), the ROCm
backend just always uses its own plain (non-fused) attention kernel instead - this is a pure
performance optimization, not required for correctness. Measured ~2x speedup in nnEvals/s on a gfx1100
(RX 7900 XTX) with a small transformer test model, FP16.

Runtime opt-out: set `rocmDisableFusedAttention = true` in the KataGo config to force the plain
kernel even when the fused path is compiled in and available.

Known unavailable on Windows as of TheRock 7.14: ck_tile's `cast_to_amdgpu_buffer_rsrc_t` calls
`std::memcpy` from device code, which requires a standard library whose `memcpy` is host+device.
libstdc++ gets that from HIP's own headers, but MSVC's `<cstring>` is already included by the time
those are reached, leaving `memcpy` host-only and the kernels uncompilable. The configure-time
probe detects this and falls back to the built-in attention kernels, so it costs speed rather than
breaking the build. TheRock 7.13's ck_tile does compile these kernels on Windows, so this is a
regression in the newer ck_tile rather than a general Windows or MSVC limitation.

Source: https://github.com/ROCm/rocm-libraries, tag `therock-7.13`,
`projects/composablekernel/example/ck_tile/01_fmha/`. Must be API-compatible with the ck_tile core
headers from the installed system package (`fmha_fwd.hpp` etc. reference internal ck_tile core APIs
that can change between releases) - if they aren't compatible, the CMake configure-time compile
probe disables the fused path and the backend falls back to its plain attention kernel. When
regenerating from a different tag, verify against the installed
`.../include/ck_tile/ops/fmha_fwd.hpp`.

## What's here

- `fmha_fwd.hpp`, `mask.hpp`, `bias.hpp`, `rotary.hpp`, `quant.hpp`: the example's glue headers
  declaring `fmha_fwd()`/`fmha_fwd_traits`/`fmha_fwd_args` and friends. Copied unmodified.
- `generated/`: kernel instantiations produced by CK's `generate.py` codegen script, narrowed to
  exactly what KataGo needs (see regeneration command below). `fmha_fwd_api.cpp` is the dispatcher
  (`fmha_fwd()`); the rest are individual `fmha_fwd_<Traits_>` template instantiations it calls into.
  The filenames are shortened relative to what `generate.py` emits, to
  `fmha_fwd_<d32|d64>_<arch>_<10-hex>.cpp` where the hex is a sha1 prefix of the original
  `generate.py` filename (see the rename step under Regenerating). `generate.py`'s own names are up
  to 173 characters of tile/pipeline configuration - long enough that object-file paths in a
  Windows build tree exceed the default 260-character `MAX_PATH`, and `git clone` breaks in all but
  very shallow directories unless long paths are enabled system-wide. The names are not
  load-bearing: nothing references them (the build globs this directory), each file's contents
  still contain its full original trait string as the template instantiation it defines (grep for
  it to map a file back to its configuration), and the contents are byte-for-byte `generate.py`
  output. Hashing the original name (rather than e.g. numbering the files) keeps each name stable
  if the generated set changes shape in a future regeneration.

## Scope (matches what the CUDA backend's cudnn-frontend SDPA path actually uses)

- fp16 only (CUDA's fused SDPA path is FP16-only too; FP32 always uses the plain kernel fallback)
- batch mode only (no group/variable-length mode)
- bias: no-bias or elementwise (matches the [B,1,S,S] additive mask-derived bias KataGo builds);
  no alibi
- mask: none (KataGo has no causal masking; padding is handled via the elementwise bias instead)
- no LSE output, no dropout, no quantization scaling, no attention sink
- hdim buckets: 32, 64 (covers KataGo's (qHeadDim, vHeadDim) combos of 32/32, 32/16, 64/64, 64/32,
  32/64 - smaller actual head dims like 16 are handled via CK's own padding within the 32 bucket)
- targets: gfx9, gfx950, gfx11, gfx115, gfx12 (as of `therock-7.13`; CK has no FMHA codegen support
  for gfx10/RDNA2 - gfx1030/1031/1032 always use the plain kernel fallback. gfx125 doesn't exist as
  a target in this codegen version either.)

## Regenerating

From a checkout of `projects/composablekernel/example/ck_tile/01_fmha/` at tag `therock-7.13` in the
CK source repo:

```
python3 generate.py --output_dir <out> --targets gfx9,gfx950,gfx11,gfx115,gfx12 -a fwd \
  -f "*_fp16_batch_*_nlogits_*bias_nmask_nlse_ndropout_nskip_nqscale_*nsink" \
  --optdim 32,64 --receipt 0 -m simplified
```

Then shorten the filenames (see "What's here" above for the scheme and why):

```
python3 - <<'EOF'
import os, hashlib
d = "<out>"
for f in os.listdir(d):
    if not f.endswith(".cpp") or f == "fmha_fwd_api.cpp":
        continue
    parts = f[:-4].split("_")
    assert parts[0] == "fmha" and parts[1] == "fwd" and parts[2] in ("d32", "d64"), f
    g = f"fmha_fwd_{parts[2]}_{parts[-1]}_{hashlib.sha1(f.encode()).hexdigest()[:10]}.cpp"
    os.rename(os.path.join(d, f), os.path.join(d, g))
EOF
```
