#!/bin/bash -eu

if [[ $# -lt 4 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR TMPDIR NTHREADS USEGATING"
    echo "NAMEPREFIX string prefix for this training run, try to pick something globally unique. Will be displayed to users when KataGo loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TMPDIR scratch space, ideally on fast local disk, unique to this loop"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    echo "USEGATING = 1 to use gatekeeper, 0 to not use gatekeeper"
    exit 0
fi
NAMEPREFIX="$1"
shift
BASEDIRRAW="$1"
shift
TMPDIRRAW="$1"
shift
NTHREADS="$1"
shift
USEGATING="$1"
shift

basedir="$(realpath "$BASEDIRRAW")"
tmpdir="$(realpath "$TMPDIRRAW")"

mkdir -p "$basedir"/scripts
cp ./*.py ./selfplay/*.sh "$basedir"/scripts

(
    cd "$basedir"/scripts
    while true
    do
        ./shuffle.sh "$basedir" "$tmpdir" "$NTHREADS" "$@"
        sleep 20
    done
) >> outshuffle.txt 2>&1 & disown

(
    cd "$basedir"/scripts
    while true
    do
        ./export_model_for_selfplay.sh "$NAMEPREFIX" "$basedir" "$USEGATING"
        sleep 10
    done
) >> outexport.txt 2>&1 & disown
