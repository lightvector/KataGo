#!/bin/bash -eu
# Exercises the CUDA backend's optimization control knobs and code-path combinations.
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
#
# Environment/arguments:
#   KATAGO_BIN       binary to test (default ./katago), must be a CUDA-backend build
#   EXPECT_FUSED_FFN set to 0 for a build without the CUTLASS fused FFN kernel
#                    (-DNO_CUTLASS_FUSED_FFN=1, or a toolchain that cannot build it):
#                    the fused FFN markers become forbidden instead of expected
#   EXPECT_SDPA      set to 0 for a build without the cudnn graph SDPA path
#                    (-DNO_CUDNN_SDPA=1, or cuDNN older than 8.9.3):
#                    the SDPA markers become forbidden instead of expected
#   $1               optional grep pattern, only cases whose name matches are run
#
# Full per-case output goes to tests/results/cuda_opt_tests/<case>.txt.

KATAGO_BIN="${KATAGO_BIN:-./katago}"
ONLY_PATTERN="${1:-}"
EXPECT_FUSED_FFN="${EXPECT_FUSED_FFN:-1}"
EXPECT_SDPA="${EXPECT_SDPA:-1}"

REFERENCEDIR="tests/results/gpu_error_reference_files"
RESULTSDIR="${CUDA_OPT_RESULTS_DIR:-tests/results/cuda_opt_tests}"
CONFIG=configs/gtp_example.cfg
mkdir -p "$RESULTSDIR"

# Models. The larger ones are fetched by rungpuerrortest.sh's wget lines. Run that at least
# once first (or fetch manually) if models/ is missing them.
TMODEL=models/b10c384h6nbttflrs.bin.gz                     # transformer, 6 heads, head dim 32
TGQA=tests/models/b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz   # GQA transformer, qk head dim 32, v head dim 16
TRSNH=tests/models/b4c256h4nbttflrs-fson-silu-rsnh.bin.gz  # small transformer, spatial rmsnorm trunk tip
CMODEL=models/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz  # nbt convnet

TMODELBASE=$(basename "$TMODEL")
TGQABASE=$(basename "$TGQA")
TRSNHBASE=$(basename "$TRSNH")
CMODELBASE=$(basename "$CMODEL")

# One-time path-selection log lines emitted by the CUDA backend. Multiple markers in a single
# expectation string are separated by '@'; within one marker, '|' separates alternatives of
# which any one suffices.
MMA_USED="using tensor-core mma flash attention kernel"
MMA_REJECTED="mma flash attention kernel rejected this launch"
SDPA_USED="using cudnn graph SDPA attention"
# Older cuDNN (e.g. 8.9.x) rejects some attention shapes that newer cuDNN accepts, in which case
# the backend logs this graceful per-handle disable and uses the custom kernel instead. Cases
# that route attention to SDPA accept either outcome.
SDPA_DISABLED="disabling cudnn SDPA and falling back to custom attention kernel"
SDPA_ANY="$SDPA_USED|$SDPA_DISABLED"
SDPA_KNOB_OFF="cudaDisableGraphSDPA is set"
FFN_USED="using CUTLASS fused FFN kernel"
FFN_REJECTED="does not support this FFN shape"
QKV_COMBINED="using combined QKV projection"
# Masked-path-only: with requireMaxBoardSize (no mask) the residual add folds into the
# projection GEMM instead, so there is no deferral to log on the exact-size path.
FUSED_RESNORM="using fused residual add and pre-norm"
# Fires once per handle, from the first 1x1 conv that takes the GEMM instead of a cuDNN conv.
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
  # Builds without CUTLASS or without cudnn SDPA can never emit those markers, so expect
  # their absence instead and let the same case list cover those build variants.
  if [ "$EXPECT_FUSED_FFN" = "0" ]; then
    expects=$(filter_out "$expects" "$FFN_USED" "$FFN_REJECTED")
    forbids="$forbids@$FFN_USED"
  fi
  if [ "$EXPECT_SDPA" = "0" ]; then
    expects=$(filter_out "$expects" "$SDPA_USED" "$SDPA_KNOB_OFF")
    forbids="$forbids@$SDPA_USED"
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
# Main transformer model (mma-eligible head dim 32): default paths across board scenarios.
# rect = mixed masked non-square boards; exact19 = requireMaxBoardSize (mask-free) path;
# rectbuffer = small boards inside a non-square 16x11 buffer.

run_case t_default_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False" \
  "$MMA_USED@$QKV_COMBINED@$FUSED_RESNORM@$FFN_USED@$MATMUL_1X1" \
  "$SDPA_USED@$MMA_REJECTED@$FFN_REJECTED"

run_case t_default_exact19 "$TMODEL" 19 quick "$TMODELBASE"_size19_quick.txt \
  "requireMaxBoardSize=True" \
  "$MMA_USED@$QKV_COMBINED@$FFN_USED" \
  "$SDPA_USED@$MMA_REJECTED@$FFN_REJECTED@$FUSED_RESNORM"

run_case t_default_rectbuffer "$TMODEL" 9 quick "$TMODELBASE"_size9_rectbuffer_quick.txt \
  "requireMaxBoardSize=False,maxBoardXSizeForNNBuffer=16,maxBoardYSizeForNNBuffer=11,maxBatchSize=9" \
  "$MMA_USED@$QKV_COMBINED@$FUSED_RESNORM@$FFN_USED" \
  "$SDPA_USED@$MMA_REJECTED@$FFN_REJECTED"

# Exact-size path on a non-square NN buffer (RoPE and attention at nnX != nnY, no mask).
run_case t_default_exact10x14 "$TMODEL" 10x14 quick "$TMODELBASE"_size10x14_quick.txt \
  "requireMaxBoardSize=True" \
  "$MMA_USED@$QKV_COMBINED@$FFN_USED" \
  "$SDPA_USED@$MMA_REJECTED@$FFN_REJECTED@$FUSED_RESNORM"

#--------------------------------------------------------------------------------------------
# Knob variants on the masked non-square scenario (the most indexing-sensitive path).

# Disabling mma attention must reroute to cudnn graph SDPA and turn off the combined QKV layout.
run_case t_mma_off_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,cudaUseMmaAttention=false" \
  "$SDPA_ANY@$FUSED_RESNORM@$FFN_USED" \
  "$MMA_USED@$QKV_COMBINED"

# Same on the mask-free exact-size path (SDPA without the [B,S,S] additive bias).
run_case t_mma_off_exact19 "$TMODEL" 19 quick "$TMODELBASE"_size19_quick.txt \
  "requireMaxBoardSize=True,cudaUseMmaAttention=false" \
  "$SDPA_ANY@$FFN_USED" \
  "$MMA_USED@$QKV_COMBINED@$FUSED_RESNORM"

# SDPA on the exact-size non-square buffer.
run_case t_mma_off_exact10x14 "$TMODEL" 10x14 quick "$TMODELBASE"_size10x14_quick.txt \
  "requireMaxBoardSize=True,cudaUseMmaAttention=false" \
  "$SDPA_ANY@$FFN_USED" \
  "$MMA_USED@$QKV_COMBINED@$FUSED_RESNORM"

# Disabling both mma and SDPA must fall back to the plain scalar attention kernel.
run_case t_plain_attention_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,cudaUseMmaAttention=false,cudaDisableGraphSDPA=true" \
  "$SDPA_KNOB_OFF@$FUSED_RESNORM" \
  "$MMA_USED@$SDPA_USED@$QKV_COMBINED"

# Same on the mask-free exact-size path (the plain kernel's mask == NULL branch).
run_case t_plain_attention_exact19 "$TMODEL" 19 quick "$TMODELBASE"_size19_quick.txt \
  "requireMaxBoardSize=True,cudaUseMmaAttention=false,cudaDisableGraphSDPA=true" \
  "$SDPA_KNOB_OFF" \
  "$MMA_USED@$SDPA_USED@$QKV_COMBINED@$FUSED_RESNORM"

# Disabling SDPA alone changes nothing when mma handles all shapes.
run_case t_sdpa_off_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,cudaDisableGraphSDPA=true" \
  "$MMA_USED@$QKV_COMBINED" \
  "$SDPA_USED"

# Unfused FFN path.
run_case t_ffn_off_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,cudaUseFusedFFN=false" \
  "$MMA_USED@$QKV_COMBINED@$FUSED_RESNORM" \
  "$FFN_USED"

# mma attention consuming the packed (non-interleaved) Q/K/V layout.
run_case t_qkv_off_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,cudaCombineQKV=false" \
  "$MMA_USED@$FUSED_RESNORM@$FFN_USED" \
  "$QKV_COMBINED"

# Unfused residual add + pre-norm.
run_case t_fuseresnorm_off_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,cudaFuseResidualNorm=false" \
  "$MMA_USED@$QKV_COMBINED@$FFN_USED" \
  "$FUSED_RESNORM"

# Pure FP32: none of the FP16-only paths may engage.
run_case t_fp32_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,useFP16=false" \
  "$FUSED_RESNORM" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED"

# Two NN server threads on one GPU: concurrent per-handle streams, then also with the
# weight-sharing registry active.
run_case t_two_threads_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,numNNServerThreadsPerModel=2" \
  "$MMA_USED@$QKV_COMBINED@$FUSED_RESNORM@$FFN_USED" \
  "$SDPA_USED"

run_case t_share_weights_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,numNNServerThreadsPerModel=2,cudaShareModelWeights=true" \
  "$MMA_USED@$QKV_COMBINED@$FUSED_RESNORM@$FFN_USED" \
  "$SDPA_USED"

# Weight sharing in FP32, which keys the registry on a different precision and uploads the
# unconverted float weights.
run_case t_share_weights_fp32_rect "$TMODEL" rectangle quick "$TMODELBASE"_sizerect_quick.txt \
  "requireMaxBoardSize=False,numNNServerThreadsPerModel=2,cudaShareModelWeights=true,useFP16=false" \
  "$FUSED_RESNORM" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED"

#--------------------------------------------------------------------------------------------
# GQA transformer with mma-unsupported head dims (qk 32, v 16): the mma kernel must reject the
# shape at runtime and the backend must fall back (cudnn SDPA), with combined QKV never engaged.

run_case tgqa_default_rect "$TGQA" rectangle full "$TGQABASE"_sizerect.txt \
  "requireMaxBoardSize=False,maxBatchSize=12" \
  "$MMA_REJECTED@$SDPA_ANY@$FUSED_RESNORM" \
  "$MMA_USED@$QKV_COMBINED"

run_case tgqa_default_exact19 "$TGQA" 19 full "$TGQABASE"_size19.txt \
  "requireMaxBoardSize=True" \
  "$MMA_REJECTED@$SDPA_ANY" \
  "$MMA_USED@$QKV_COMBINED@$FUSED_RESNORM"

# GQA shapes through the plain attention kernel.
run_case tgqa_plain_attention_rect "$TGQA" rectangle full "$TGQABASE"_sizerect.txt \
  "requireMaxBoardSize=False,maxBatchSize=12,cudaUseMmaAttention=false,cudaDisableGraphSDPA=true" \
  "$SDPA_KNOB_OFF@$FUSED_RESNORM" \
  "$MMA_USED@$SDPA_USED@$QKV_COMBINED"

#--------------------------------------------------------------------------------------------
# Small transformer with spatial rmsnorm trunk tip, on the non-square-buffer scenario.

run_case trsnh_default_rectbuffer "$TRSNH" 9 full "$TRSNHBASE"_size9_rectbuffer.txt \
  "requireMaxBoardSize=False,maxBoardXSizeForNNBuffer=16,maxBoardYSizeForNNBuffer=11,maxBatchSize=15,policyOptimism=0.70" \
  "$MMA_USED@$QKV_COMBINED@$FUSED_RESNORM" \
  "$SDPA_USED"

#--------------------------------------------------------------------------------------------
# Convnet: transformer-only paths must never engage. Exercise layout and precision knobs.

run_case c_default_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False" \
  "$MATMUL_1X1" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED@$FUSED_RESNORM"

run_case c_nchw_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,cudaUseNHWC=false" \
  "" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED@$FUSED_RESNORM@$MATMUL_1X1"

run_case c_fp32_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,useFP16=false" \
  "" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED@$FUSED_RESNORM@$MATMUL_1X1"

run_case c_fp32_nhwc_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,useFP16=false,cudaUseNHWC=true" \
  "" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED@$FUSED_RESNORM@$MATMUL_1X1"

# 1x1 convolutions via cuDNN instead of the default FP16 cuBLAS GEMM, and forced GEMM in FP32.
run_case c_1x1matmul_off_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,cudaUse1x1Matmul=false" \
  "" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED@$FUSED_RESNORM@$MATMUL_1X1"

# The GEMM path also requires NHWC, which FP32 does not default to, so force it.
run_case c_1x1matmul_fp32_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,useFP16=false,cudaUseNHWC=true,cudaUse1x1Matmul=true" \
  "$MATMUL_1X1" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED@$FUSED_RESNORM"

run_case c_share_weights_exact19 "$CMODEL" 19 full "$CMODELBASE"_size19.txt \
  "requireMaxBoardSize=True,numNNServerThreadsPerModel=2,cudaShareModelWeights=true" \
  "$MATMUL_1X1" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED@$FUSED_RESNORM"

# NCHW convolutions with weight sharing, which uploads the untransposed filter layout.
run_case c_nchw_share_weights_rect "$CMODEL" rectangle full "$CMODELBASE"_sizerect.txt \
  "requireMaxBoardSize=False,cudaUseNHWC=false,numNNServerThreadsPerModel=2,cudaShareModelWeights=true" \
  "" \
  "$MMA_USED@$SDPA_USED@$FFN_USED@$QKV_COMBINED@$FUSED_RESNORM@$MATMUL_1X1"

#--------------------------------------------------------------------------------------------
# Invalid combinations must fail loudly at startup.

run_fail_case n_transformer_nchw "$TMODEL" rectangle \
  "requireMaxBoardSize=False,cudaUseNHWC=false" \
  "transformer models require NHWC, but cudaUseNHWC=false was set"

if [ "$EXPECT_FUSED_FFN" = "0" ]; then
  run_fail_case n_fusedffn_fp32 "$TMODEL" rectangle \
    "requireMaxBoardSize=False,useFP16=false,cudaUseFusedFFN=true" \
    "cudaUseFusedFFN=true but this build was compiled without CUTLASS"
else
  run_fail_case n_fusedffn_fp32 "$TMODEL" rectangle \
    "requireMaxBoardSize=False,useFP16=false,cudaUseFusedFFN=true" \
    "cudaUseFusedFFN=true but the fused FFN kernel is not usable here"
fi

#--------------------------------------------------------------------------------------------

echo ""
echo "============================================================"
echo "$SUMMARY"
echo "$NUM_PASS passed, $NUM_FAIL failed"
if [ "$NUM_FAIL" -ne 0 ]; then
  echo "SOME CUDA OPTION TESTS FAILED"
  exit 1
fi
echo "ALL CUDA OPTION TESTS PASSED"
