#!/bin/bash -eu
set -o pipefail
{
if [[ $# -lt 4 ]]
then
    echo "Usage: $0 BASEDIR TMPDIR NTHREADS BATCHSIZE"
    echo "Currently expects to be run from within the 'python' directory of the KataGo repo, or otherwise in the same dir as export_model.py."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TMPDIR scratch space, ideally on fast local disk, unique to this loop"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    echo "BATCHSIZE number of samples to concat together per batch for training, must match training"
    exit 0
fi
BASEDIR="$1"
shift
TMPDIR="$1"
shift
NTHREADS="$1"
shift
BATCHSIZE="$1"
shift

GITROOTDIR="$(git rev-parse --show-toplevel)"

basedir="$(realpath "$BASEDIRRAW")"
tmpdir="$(realpath "$TMPDIRRAW")"

mkdir -p "$basedir"/logs

(
    while true
    do
        "$GITROOTDIR"/python/selfplay/shuffle.sh "$basedir" "$tmpdir" "$NTHREADS" "$BATCHSIZE" "$@"
        sleep 20
    done
) >> "$basedir"/logs/outshuffle.txt 2>&1 & disown

exit 0
}
