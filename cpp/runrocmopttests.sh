#!/bin/bash -eu
# Exercises the ROCm backend's optimization control knobs and code-path combinations.
# ROCm analog of runcudaopttests.sh - see that script for the shared rationale.
#
# Each positive case runs testgpuerror on a model/board-dataset combination with a specific set
# of config overrides, checked against the machine-local Eigen-generated reference file (see
# rungpuerrortest.sh for how those are generated). A case passes only if testgpuerror passes its
# numerical error thresholds AND the backend's one-time path-selection log lines confirm that the
# expected code paths were engaged (and that disabled ones were not) - guarding against a knob
# silently failing to switch paths. Negative cases assert that invalid option combinations fail
# at startup with a clear error instead of running with a wrong configuration.
#
# Each testgpuerror run internally covers both the configured (usually FP16) evaluator and an
# FP32 evaluator with the same knobs, and both unbatched and randomly-batched evaluation, so
# every case also exercises the FP32 variants of the selected paths and batch-size variation.
# Because of that second evaluator, layout assertions match the whole "useFP16 = X useNHWC = Y"
# pair rather than the layout alone, which on its own would be ambiguous.
#
# Model choice also covers kernel-level branches that have no log marker of their own. The
# half-precision RMSNorm launchers take their vectorized path only when the channel count is a
# multiple of 128, so the c256/c384/c512/c768 models exercise the vectorized kernels and the
# c96 models exercise the scalar fallback.
#
# Environment/arguments:
#   KATAGO_BIN      binary to test (default ./katago), must be a ROCm-backend build
#   EXPECT_CK_FMHA  set to 0 for a build configured without ck_tile (or for a GPU architecture
#                   the CK path does not cover): the CK markers become forbidden instead of
#                   expected
#   $1              optional grep pattern, only cases whose name matches are run
#
# Full per-case output goes to tests/results/rocm_opt_tests/<case>.txt.

KATAGO_BIN="${KATAGO_BIN:-./katago}"
ONLY_PATTERN="${1:-}"
EXPECT_CK_FMHA="${EXPECT_CK_FMHA:-1}"

REFERENCEDIR="tests/results/gpu_error_reference_files"
RESULTSDIR="${ROCM_OPT_RESULTS_DIR:-tests/results/rocm_opt_tests}"
CONFIG=configs/gtp_example.cfg
mkdir -p "$RESULTSDIR"

# Models. The larger ones are fetched by rungpuerrortest.sh's wget lines. Run that at least
# once first (or fetch manually) if models/ is missing them.
TMODEL=models/b10c384h6nbttflrs.bin.gz                     # transformer, 6 heads, head dim 32
TBIG=models/b11c768h12nbt3tflrs-fson-silu.bin.gz           # largest transformer, 12 heads
TGQA=tests/models/b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz   # GQA transformer, qk head dim 32, v head dim 16
TRSNH=tests/models/b4c256h4nbttflrs-fson-silu-rsnh.bin.gz  # small transformer, spatial rmsnorm trunk tip
TCNORM=tests/models/b7c96h3tfrs-test5-cnorm.bin.gz         # small transformer, channel-norm trunk
CMODEL=models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz  # nbt convnet

TMODELBASE=$(basename "$TMODEL")
TBIGBASE=$(basename "$TBIG")
TGQABASE=$(basename "$TGQA")
TRSNHBASE=$(basename "$TRSNH")
TCNORMBASE=$(basename "$TCNORM")
CMODELBASE=$(basename "$CMODEL")

# One-time path-selection log lines emitted by the ROCm backend. Multiple markers in a single
# expectation string are separated by '@'; within one marker, '|' separates alternatives of
# which any one suffices.
CK_USED="ROCm backend: using CK fused attention"
CK_UNAVAIL="ROCm backend: CK fused attention unavailable"
CK_KNOB_OFF="rocmDisableFusedAttention is set"
# Masked-path-only: with requireMaxBoardSize (no mask) the residual add folds into the
# projection GEMM instead, so there is no deferral to log on the exact-size path.
FUSED_RESNORM="using fused residual add and pre-norm"
FP16_NHWC="useFP16 = true useNHWC = true"
FP16_NCHW="useFP16 = true useNHWC = false"
FP32_NHWC="useFP16 = false useNHWC = true"
FP32_NCHW="useFP16 = false useNHWC = false"
# Fires once per handle, from the first 1x1 conv that takes the GEMM instead of a MIOpen conv.
MATMUL_1X1="running 1x1 NHWC convolutions as a GEMM"

NUM_PASS=0
NUM_FAIL=0
SUMMARY=""

matches_filter() {
  [ -z "$ONLY_PATTERN" ] || echo "$1" | grep -q -- "$ONLY_PATTERN"
}

record_result() {
  local name="$1"
  local ok="$2"
  local note="$3"
  if [ "$ok" -eq 1 ]; then
    NUM_PASS=$((NUM_PASS+1))
    SUMMARY="${SUMMARY}PASS  $name  $note"$'\n'
    echo "PASS: $name"
  else
    NUM_FAIL=$((NUM_FAIL+1))
    SUMMARY="${SUMMARY}FAIL  $name  $note"$'\n'
    echo "FAIL: $name"
  fi
}

# Echoes $1 (an '@'-separated list) with any entries containing one of $2.. removed.
filter_out() {
  local list="$1"
  shift
  local out="" item pat drop
  local oldifs="$IFS"
  IFS='@'
  for item in $list; do
    [ -z "$item" ] && continue
    drop=0
    for pat in "$@"; do
      case "$item" in
        *"$pat"*) drop=1 ;;
      esac
    done
    [ "$drop" -eq 0 ] && out="$out@$item"
  done
  IFS="$oldifs"
  echo "$out"
}

# run_case <name> <model> <boardsize> <quick|full> <reference-file-name> <overrides> <expects> <forbids>
# expects/forbids: '@'-separated lists of literal substrings that must / must not appear in the output.
run_case() {
  local name="$1" model="$2" boardsize="$3" quick="$4" reffile="$5" overrides="$6" expects="$7" forbids="$8"
  matches_filter "$name" || return 0
  # Builds or architectures without the CK fused attention path can never emit its markers,
  # so expect their absence instead and let the same case list cover those build variants.
  if [ "$EXPECT_CK_FMHA" = "0" ]; then
    expects=$(filter_out "$expects" "$CK_USED" "$CK_UNAVAIL" "$CK_KNOB_OFF")
    forbids="$forbids@$CK_USED"
  fi
  local outfile="$RESULTSDIR/$name.txt"
  local qflag=()
  if [ "$quick" = "quick" ]; then
    qflag=(-quick)
  fi
  echo "=== $name ==="
  local ok=1
  if ! "$KATAGO_BIN" testgpuerror -model "$model" -config "$CONFIG" -boardsize "$boardsize" ${qflag[@]+"${qflag[@]}"} \
      -override-config "$overrides" \
      -reference-file "$REFERENCEDIR/$reffile" > "$outfile" 2>&1; then
    echo "  error thresholds or run FAILED (see $outfile)"
    tail -5 "$outfile" | sed 's/^/    /'
    ok=0
  fi
  local pat alt found
  local oldifs="$IFS"
  IFS='@'
  for pat in $expects; do
    [ -z "$pat" ] && continue
    found=0
    IFS='|'
    for alt in $pat; do
      [ -z "$alt" ] && continue
      if grep -qF "$alt" "$outfile"; then
        found=1
      fi
    done
    IFS='@'
    if [ "$found" -eq 0 ]; then
      echo "  expected log line missing: '$pat'"
      ok=0
    fi
  done
  for pat in $forbids; do
    [ -z "$pat" ] && continue
    if grep -qF "$pat" "$outfile"; then
      echo "  forbidden log line present: '$pat'"
      ok=0
    fi
  done
  IFS="$oldifs"
  local margin
  margin=$(grep "closest margin" "$outfile" | tail -1 | sed 's/^.*closest margin: */margin /' || true)
  record_result "$name" "$ok" "${margin:-"(no margin line)"}"
}

# run_fail_case <name> <model> <boardsize> <overrides> <expected-error-substring>
# The command must FAIL at startup and the output must contain the expected error message.
run_fail_case() {
  local name="$1" model="$2" boardsize="$3" overrides="$4" errpat="$5"
  matches_filter "$name" || return 0
  local outfile="$RESULTSDIR/$name.txt"
  echo "=== $name (expected startup failure) ==="
  local ok=1
  if "$KATAGO_BIN" testgpuerror -model "$model" -config "$CONFIG" -boardsize "$boardsize" -quick \
      -override-config "$overrides" > "$outfile" 2>&1; then
    echo "  expected an error, but the command succeeded"
    ok=0
  elif ! grep -qF "$errpat" "$outfile"; then
    echo "  failed, but without the expected message: '$errpat'"
    tail -5 "$outfile" | sed 's/^/    /'
    ok=0
  fi
  record_result "$name" "$ok" "(expected-failure case)"
}

#--------------------------------------------------------------------------------------------
# Main transformer model (CK-supported head dim 32): default paths across board scenarios.
# rect = mixed masked non-square boards; exact19 = requireMaxBoardSize (mask-free) path;
# rectbuffer = small boards inside a non-square 16x11 buffer.

run_case t_default_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False" \
  "$CK_USED@$FUSED_RESNORM@$FP16_NHWC" \
  "$CK_UNAVAIL@$CK_KNOB_OFF"

run_case t_default_exact19 "$TMODEL" 19 quick "$TMODELBASE"_size19_quick.txt \
  "requireMaxBoardSize=True" \
  "$CK_USED@$FP16_NHWC" \
  "$CK_UNAVAIL@$CK_KNOB_OFF@$FUSED_RESNORM"

run_case t_default_rectbuffer "$TMODEL" 9 quick "$TMODELBASE"_size9_rectbuffer_quick.txt \
  "requireMaxBoardSize=False,maxBoardXSizeForNNBuffer=16,maxBoardYSizeForNNBuffer=11,maxBatchSize=9" \
  "$CK_USED@$FUSED_RESNORM" \
  "$CK_UNAVAIL@$CK_KNOB_OFF"

#--------------------------------------------------------------------------------------------
# Knob variants on the masked non-square scenario (the most indexing-sensitive path).

# Disabling the CK fused path must reroute attention to the plain online-softmax kernel.
run_case t_ck_off_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,rocmDisableFusedAttention=true" \
  "$CK_KNOB_OFF@$FUSED_RESNORM" \
  "$CK_USED"

# Same on the mask-free exact-size path (the CK path there runs without an additive bias).
run_case t_ck_off_exact19 "$TMODEL" 19 quick "$TMODELBASE"_size19_quick.txt \
  "requireMaxBoardSize=True,rocmDisableFusedAttention=true" \
  "$CK_KNOB_OFF" \
  "$CK_USED@$FUSED_RESNORM"

# Unfused residual add + pre-norm.
run_case t_fuseresnorm_off_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,rocmFuseResidualNorm=false" \
  "$CK_USED" \
  "$FUSED_RESNORM"

# Pure FP32: the CK path is FP16-only and must not engage. Transformers force NHWC even in FP32.
run_case t_fp32_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,useFP16=false" \
  "$FUSED_RESNORM@$FP32_NHWC" \
  "$CK_USED@$FP16_NHWC"

# Two NN server threads on one GPU: concurrent per-handle streams, then also with the
# weight-sharing registry active.
run_case t_two_threads_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,numNNServerThreadsPerModel=2" \
  "$CK_USED@$FUSED_RESNORM" \
  "$CK_UNAVAIL"

run_case t_share_weights_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,numNNServerThreadsPerModel=2,rocmShareModelWeights=true" \
  "$CK_USED@$FUSED_RESNORM" \
  "$CK_UNAVAIL"

# Weight sharing in FP32, which keys the registry on a different precision and uploads the
# unconverted float weights.
run_case t_share_weights_fp32_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,numNNServerThreadsPerModel=2,rocmShareModelWeights=true,useFP16=false" \
  "$FUSED_RESNORM@$FP32_NHWC" \
  "$CK_USED"

# The 1x1 GEMM on a transformer. Transformers force NHWC on every architecture, so this is the
# shape in which non-CDNA users get the GEMM by default.
run_case t_1x1matmul_on_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,rocmUse1x1Matmul=true" \
  "$CK_USED@$FUSED_RESNORM@$FP16_NHWC@$MATMUL_1X1" \
  "$CK_UNAVAIL@$CK_KNOB_OFF"

#--------------------------------------------------------------------------------------------
# Largest transformer (768 trunk channels, 12 heads), on the masked path.

run_case tbig_default_19 "$TBIG" 19 quick "$TBIGBASE"_size19_quick.txt \
  "requireMaxBoardSize=False" \
  "$CK_USED@$FUSED_RESNORM@$FP16_NHWC" \
  "$CK_UNAVAIL@$CK_KNOB_OFF"

#--------------------------------------------------------------------------------------------
# GQA transformer with differing q and v head dims (qk 32, v 16). Whether CK's generated kernel
# set covers a shape depends on the ck_tile version and the architecture list the build used.
# On ck_tile 7.14 for gfx942 it does. A build whose CK cannot take this shape will report the
# unavailable marker and fall back correctly, in which case relax these two to
# "$CK_USED|$CK_UNAVAIL" and drop the forbid.

run_case tgqa_default_rect "$TGQA" rectangle full "$TGQABASE"_sizerect.txt \
  "requireMaxBoardSize=False,maxBatchSize=12" \
  "$CK_USED@$FUSED_RESNORM" \
  "$CK_UNAVAIL@$CK_KNOB_OFF"

run_case tgqa_default_exact19 "$TGQA" 19 full "$TGQABASE"_size19.txt \
  "requireMaxBoardSize=True" \
  "$CK_USED" \
  "$CK_UNAVAIL@$CK_KNOB_OFF@$FUSED_RESNORM"

# GQA shapes through the plain attention kernel.
run_case tgqa_ck_off_rect "$TGQA" rectangle full "$TGQABASE"_sizerect.txt \
  "requireMaxBoardSize=False,maxBatchSize=12,rocmDisableFusedAttention=true" \
  "$CK_KNOB_OFF@$FUSED_RESNORM" \
  "$CK_USED"

#--------------------------------------------------------------------------------------------
# Small transformers: 256 channels with a spatial rmsnorm trunk tip, and 96 channels, which is
# not a multiple of 128 and so takes the scalar rather than vectorized RMSNorm kernels.

run_case trsnh_default_rectbuffer "$TRSNH" 9 full "$TRSNHBASE"_size9_rectbuffer.txt \
  "requireMaxBoardSize=False,maxBoardXSizeForNNBuffer=16,maxBoardYSizeForNNBuffer=11,maxBatchSize=15,policyOptimism=0.70" \
  "$CK_USED@$FUSED_RESNORM" \
  "$CK_UNAVAIL@$CK_KNOB_OFF"

run_case tcnorm_default_rect "$TCNORM" rectangle full "$TCNORMBASE"_sizerect.txt \
  "requireMaxBoardSize=False" \
  "$CK_USED@$FUSED_RESNORM" \
  "$CK_UNAVAIL@$CK_KNOB_OFF"

#--------------------------------------------------------------------------------------------
# Convnet: transformer-only paths must never engage. Exercise layout and precision knobs.

run_case c_default_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False" \
  "$FP16_NHWC" \
  "$CK_USED@$CK_UNAVAIL@$CK_KNOB_OFF@$FUSED_RESNORM"

run_case c_nchw_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,rocmUseNHWC=false" \
  "$FP16_NCHW" \
  "$CK_USED@$FUSED_RESNORM@$FP16_NHWC@$MATMUL_1X1"

run_case c_fp32_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,useFP16=false" \
  "$FP32_NCHW" \
  "$CK_USED@$FUSED_RESNORM@$FP16_NHWC@$MATMUL_1X1"

run_case c_fp32_nhwc_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,useFP16=false,rocmUseNHWC=true" \
  "$FP32_NHWC" \
  "$CK_USED@$FUSED_RESNORM@$FP16_NHWC"

# 1x1 NHWC convolutions as a hipBLAS GEMM instead of a MIOpen conv. On CDNA the resolved
# default is off, so the =true cases are the ones covering what ships as the default elsewhere.
run_case c_1x1matmul_on_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,rocmUse1x1Matmul=true" \
  "$FP16_NHWC@$MATMUL_1X1" \
  "$CK_USED@$FUSED_RESNORM"

# Mask-free path, where the spatial extent the GEMM folds into its token count is the full buffer.
run_case c_1x1matmul_on_exact19 "$CMODEL" 19 full "$CMODELBASE"_size19.txt \
  "requireMaxBoardSize=True,rocmUse1x1Matmul=true" \
  "$FP16_NHWC@$MATMUL_1X1" \
  "$CK_USED@$FUSED_RESNORM"

run_case c_1x1matmul_off_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,rocmUse1x1Matmul=false" \
  "$FP16_NHWC" \
  "$CK_USED@$FUSED_RESNORM@$MATMUL_1X1"

# FP32 forced through the GEMM, which is a different hipBLAS entry point than the FP16 one.
run_case c_1x1matmul_fp32_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,useFP16=false,rocmUseNHWC=true,rocmUse1x1Matmul=true" \
  "$FP32_NHWC@$MATMUL_1X1" \
  "$CK_USED@$FUSED_RESNORM@$FP16_NHWC"

# The GEMM weights are a second entry in the sharing registry, keyed on the same descriptor as
# the conv filter and distinguished only by kind.
run_case c_1x1matmul_share_weights_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,rocmUse1x1Matmul=true,numNNServerThreadsPerModel=2,rocmShareModelWeights=true" \
  "$FP16_NHWC@$MATMUL_1X1" \
  "$CK_USED@$FUSED_RESNORM"

run_case c_share_weights_exact19 "$CMODEL" 19 full "$CMODELBASE"_size19.txt \
  "requireMaxBoardSize=True,numNNServerThreadsPerModel=2,rocmShareModelWeights=true" \
  "$FP16_NHWC" \
  "$CK_USED@$FUSED_RESNORM"

# NCHW convolutions with weight sharing, which uploads the untransposed filter layout.
run_case c_nchw_share_weights_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,rocmUseNHWC=false,numNNServerThreadsPerModel=2,rocmShareModelWeights=true" \
  "$FP16_NCHW" \
  "$CK_USED@$FUSED_RESNORM@$FP16_NHWC"

#--------------------------------------------------------------------------------------------
# Invalid combinations must fail loudly at startup.

run_fail_case n_transformer_nchw "$TMODEL" rectangle \
  "requireMaxBoardSize=False,rocmUseNHWC=false" \
  "transformer models require NHWC, but rocmUseNHWC=false was set"

#--------------------------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "$SUMMARY"
echo "$NUM_PASS passed, $NUM_FAIL failed"
if [ "$NUM_FAIL" -ne 0 ]; then
  echo "SOME ROCM OPTION TESTS FAILED"
  exit 1
fi
echo "ALL ROCM OPTION TESTS PASSED"
