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
BASEDIRRAW="$1"
shift
TMPDIRRAW="$1"
shift
NTHREADS="$1"
shift
BATCHSIZE="$1"
shift

GITROOTDIR="$(git rev-parse --show-toplevel)"

basedir="$(realpath "$BASEDIRRAW")"
tmpdir="$(realpath "$TMPDIRRAW")"

mkdir -p "$basedir"/scripts
mkdir -p "$basedir"/logs
cp "$GITROOTDIR"/python/*.py "$GITROOTDIR"/python/selfplay/*.sh "$GITROOTDIR"/python/selfplay/distributed/*.sh "$basedir"/scripts
cp -r "$GITROOTDIR"/python/katago "$basedir"/scripts

# For archival and logging purposes - you can look back and see exactly the python code on a particular date
DATE_FOR_FILENAME=$(date "+%Y%m%d-%H%M%S")
DATED_ARCHIVE="$basedir"/scripts/dated/"$DATE_FOR_FILENAME"
mkdir -p "$DATED_ARCHIVE"
cp "$GITROOTDIR"/python/*.py "$DATED_ARCHIVE"
cp -r "$GITROOTDIR"/python/katago "$DATED_ARCHIVE"
cp -r "$GITROOTDIR"/python/selfplay "$DATED_ARCHIVE"


(
    cd "$basedir"/scripts
    while true
    do
        rm -f "$basedir"/selfplay.summary.json.tmp
        time python3 ./summarize_old_selfplay_files.py "$basedir"/selfplay/ \
             -old-summary-file-to-assume-correct "$basedir"/selfplay.summary.json \
             -new-summary-file "$basedir"/selfplay.summary.json.tmp
        mv "$basedir"/selfplay.summary.json.tmp "$basedir"/selfplay.summary.json
        sleep 10

        for i in {1..10}
        do
            ./shuffle.sh "$basedir" "$tmpdir" "$NTHREADS" "$BATCHSIZE" -summary-file "$basedir"/selfplay.summary.json "$@"
            sleep 600
        done
    done
) >> "$basedir"/logs/outshuffle.txt 2>&1 & disown

exit 0
}
