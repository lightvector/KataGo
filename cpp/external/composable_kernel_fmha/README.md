# Composable Kernel FMHA (fused multi-head attention)

Vendored glue headers + pre-generated kernel instantiations from AMD's Composable Kernel (CK)
`ck_tile` FMHA implementation, used by the ROCm backend as an optional fused attention path
(mirrors the CUDA backend's optional cudnn-frontend SDPA graph path). Requires the ck_tile base
headers from a `composablekernel-dev`/`amdrocm-ck*` system package (not vendored here). If not
found at configure time (see `KATAGO_CK_TILE_INCLUDE_DIR` in `cpp/CMakeLists.txt`), the ROCm
backend just always uses its own plain (non-fused) attention kernel instead - this is a pure
performance optimization, not required for correctness. Measured ~2.2x nnEvals/s on a gfx1100
(RX 7900 XTX) with a small transformer test model, FP16.

Runtime opt-out: set `rocmDisableFusedAttention = true` in the KataGo config to force the plain
kernel even when the fused path is compiled in and available.

Source: https://github.com/ROCm/rocm-libraries, tag `therock-7.13`,
`projects/composablekernel/example/ck_tile/01_fmha/`. Must match the ck_tile core headers version
from the installed `amdrocm-ck7.13` system package byte-for-byte (`fmha_fwd.hpp` etc. reference
internal ck_tile core APIs that change between releases) — verify with `diff` against the installed
`.../include/ck_tile/ops/fmha_fwd.hpp` before regenerating from a different tag.

## What's here

- `fmha_fwd.hpp`, `mask.hpp`, `bias.hpp`, `rotary.hpp`, `quant.hpp`: the example's glue headers
  declaring `fmha_fwd()`/`fmha_fwd_traits`/`fmha_fwd_args` and friends. Copied unmodified.
- `generated/`: kernel instantiations produced by CK's `generate.py` codegen script, narrowed to
  exactly what KataGo needs (see regeneration command below). `fmha_fwd_api.cpp` is the dispatcher
  (`fmha_fwd()`); the rest are individual `fmha_fwd_<Traits_>` template instantiations it calls into.

## Scope (matches what the CUDA backend's cudnn-frontend SDPA path actually uses)

- fp16 only (CUDA's fused SDPA path is FP16-only too; FP32 always uses the plain kernel fallback)
- batch mode only (no group/variable-length mode)
- bias: no-bias or elementwise (matches the [B,1,S,S] additive mask-derived bias KataGo builds);
  no alibi
- mask: none (KataGo has no causal masking; padding is handled via the elementwise bias instead)
- no LSE output, no dropout, no quantization scaling, no attention sink
- hdim buckets: 32, 64 (covers KataGo's (qHeadDim, vHeadDim) combos of 32/32, 32/16, 64/64, 64/32,
  32/64 — smaller actual head dims like 16 are handled via CK's own padding within the 32 bucket)
- targets: gfx9, gfx950, gfx11, gfx115, gfx12 (as of `therock-7.13`; CK has no FMHA codegen support
  for gfx10/RDNA2 — gfx1030/1031/1032 always use the plain kernel fallback. gfx125 doesn't exist as
  a target in this codegen version either.)

## Regenerating

From a checkout of `projects/composablekernel/example/ck_tile/01_fmha/` at tag `therock-7.13` in the
CK source repo:

```
python3 generate.py --output_dir <out> --targets gfx9,gfx950,gfx11,gfx115,gfx12 -a fwd \
  -f "*_fp16_batch_*_nlogits_*bias_nmask_nlse_ndropout_nskip_nqscale_*nsink" \
  --optdim 32,64 --receipt 0 -m simplified
```
