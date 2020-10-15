#!/bin/bash -eu
set -o pipefail
{
if [[ $# -lt 7 ]]
then
    echo "Usage: $0 RUNNAME BASEDIR CONNECTION_CONFIG DOWNLOAD_SCRIPT TMPDIR NTHREADS BATCHSIZE"
    echo "Currently expects to be run from within the 'python' directory of the KataGo repo, or otherwise in the same dir as export_model.py."
    echo "RUNNAME should match what the server uses as the run name, try to pick something globally unique. Will prefix model names in uploaded files."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "CONNECTION_CONFIG config containing serverUrl, username, password"
    echo "DOWNLOAD_SCRIPT script that downloads the data when provided RUNNAME and BASEDIR as an argument"
    echo "TMPDIR scratch space, ideally on fast local disk, unique to this loop"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    echo "BATCHSIZE number of samples to concat together per batch for training, must match training"
    exit 0
fi
RUNNAME="$1"
shift
BASEDIRRAW="$1"
shift
CONNECTION_CONFIG="$1"
shift
DOWNLOAD_SCRIPT="$1"
shift
TMPDIRRAW="$1"
shift
NTHREADS="$1"
shift
BATCHSIZE="$1"
shift

#We're not really using gating, but the upload script expects them to be where gating would put them
#and using gating disables the export script from making extraneous selfplay data dirs.
USEGATING=1

GITROOTDIR="$(git rev-parse --show-toplevel)"

basedir="$(realpath "$BASEDIRRAW")"
tmpdir="$(realpath "$TMPDIRRAW")"

mkdir -p "$basedir"/scripts
mkdir -p "$basedir"/logs
cp "$GITROOTDIR"/python/*.py "$GITROOTDIR"/python/selfplay/*.sh "$GITROOTDIR"/python/selfplay/distributed/*.sh "$basedir"/scripts
cp "$DOWNLOAD_SCRIPT" "$basedir"/scripts/download.sh
cp "$CONNECTION_CONFIG" "$basedir"/scripts/connection.cfg

(
    cd "$basedir"/scripts
    while true
    do
        ./upload_model_for_selfplay.sh "$RUNNAME" "$basedir" connection.cfg
        sleep 20
    done
) >> "$basedir"/logs/outupload.txt 2>&1 & disown

(
    cd "$basedir"/scripts
    while true
    do
        time python3 ./summarize_old_selfplay_files.py "$basedir"/selfplay/ \
             -old-summary-file-to-assume-correct "$basedir"/selfplay.summary.json \
             -new-summary-file "$basedir"/selfplay.summary.json.tmp
        mv "$basedir"/selfplay.summary.json.tmp "$basedir"/selfplay.summary.json
        sleep 10

        for i in {1..10}
        do
            ./shuffle.sh "$basedir" "$tmpdir" "$NTHREADS" "$BATCHSIZE" -summary-file "$basedir"/selfplay.summary.json "$@"
            sleep 180
        done
    done
) >> "$basedir"/logs/outshuffle.txt 2>&1 & disown

(
    cd "$basedir"/scripts
    while true
    do
        ./export_model_for_selfplay.sh "$RUNNAME" "$basedir" "$USEGATING"
        sleep 10
    done
) >> "$basedir"/logs/outexport.txt 2>&1 & disown

(
    cd "$basedir"/scripts
    while true
    do
        ./download.sh "$RUNNAME" "$basedir"
        sleep 180
    done
) >> "$basedir"/logs/outdownload.txt 2>&1 & disown

exit 0
}
