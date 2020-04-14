#!/bin/bash -eu

if [[ $# -ne 3 ]]
then
    echo "Usage: $0 BASEDIR TMPDIR NTHREADS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TMPDIR scratch space, ideally on fast local disk, unique to this loop"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    exit 0
fi
BASEDIR="$1"
shift
TMPDIR="$1"
shift
NTHREADS="$1"
shift

GITROOTDIR="$(git rev-parse --show-toplevel)"

basedir="$(realpath "$BASEDIRRAW")"
tmpdir="$(realpath "$TMPDIRRAW")"

mkdir -p "$basedir"/logs

(
    while true
    do
        "$GITROOTDIR"/python/selfplay/shuffle.sh "$basedir" "$tmpdir" "$NTHREADS" "$@"
        sleep 20
    done
) >> "$basedir"/logs/outshuffle.txt 2>&1 & disown
