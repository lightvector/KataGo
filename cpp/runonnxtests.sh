#!/bin/bash -eux
set -o pipefail
{
# ---------------------------------------------------------------
# ONNX backend integration tests
#
# Exercises three levels of the inference pipeline:
#   1. runtinynntests       — tiny model, full pipeline (no external model)
#   2. testgpuerror -quick  — FP32 unbatched vs batched comparison
#   3. runnnevalcanarytests — sanity checks on real game positions
# ---------------------------------------------------------------

mkdir -p tests/scratch

# 1. Tiny NN tests — self-contained, no external model needed
echo "=== runtinynntests ==="
./katago runtinynntests tests/scratch 1.0 \
  | grep -v ': nnRandSeed0 = ' \
  | grep -v 'finishing, processed'

# 2. GPU error test (quick) — compares unbatched vs batched inference
#    For CPU ONNX provider both paths are FP32, so errors should be near zero.
#    Any ownership indexing bug would surface as large ownership error.
echo "=== testgpuerror -quick ==="
./katago testgpuerror \
  -config configs/gtp_example.cfg \
  -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz \
  -quick \
  -override-config "nnRandSeed=forTesting,forDeterministicTesting=true"

# 3. NN eval canary tests — sanity checks on 5 real game positions
#    Uses symmetries 0, 3, 6 (same as runsearchtests.sh)
echo "=== runnnevalcanarytests ==="
./katago runnnevalcanarytests configs/gtp_example.cfg tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz 0 \
  | grep -v ': nnRandSeed0 = '
./katago runnnevalcanarytests configs/gtp_example.cfg tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz 3 \
  | grep -v ': nnRandSeed0 = '
./katago runnnevalcanarytests configs/gtp_example.cfg tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz 6 \
  | grep -v ': nnRandSeed0 = '

echo "=== All ONNX tests passed ==="
exit 0
}
