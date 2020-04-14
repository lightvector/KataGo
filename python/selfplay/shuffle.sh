#!/bin/bash -eu

#Shuffles and copies selfplay training from selfplay/ to shuffleddata/current/
#Should be run periodically.

if [[ $# -lt 3 ]]
then
    echo "Usage: $0 BASEDIR TMPDIR NTHREADS"
    echo "Currently expects to be run from within the `python` directory of the KataGo repo, or otherwise in the same dir as shuffle.py."
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

#------------------------------------------------------------------------------

OUTDIR=$(date "+%Y%m%d-%H%M%S")
OUTDIRTRAIN=$OUTDIR/train
OUTDIRVAL=$OUTDIR/val

mkdir -p "$BASEDIR"/shuffleddata/$OUTDIR
mkdir -p "$TMPDIR"/train
mkdir -p "$TMPDIR"/val

echo "Beginning shuffle at" $(date "+%Y-%m-%d %H:%M:%S")

#set -x
(
    time python3 ./shuffle.py \
         "$BASEDIR"/selfplay/*/tdata/ \
         -min-rows 250000 \
         -max-rows 1000000000 \
         -expand-window-per-row 0.4 \
         -taper-window-exponent 0.65 \
         -out-dir "$BASEDIR"/shuffleddata/$OUTDIRTRAIN \
         -out-tmp-dir "$TMPDIR"/train \
         -approx-rows-per-out-file 70000 \
         -num-processes "$NTHREADS" \
         -batch-size 256 \
         -keep-target-rows 1200000 \
         "$@" \
         2>&1 | tee "$BASEDIR"/shuffleddata/$OUTDIR/outtrain.txt &

    # time python3 ./shuffle.py \
    #      "$BASEDIR"/selfplay/*/vdata/ \
    #      -min-rows 12500 \
    #      -max-rows 10000000 \
    #      -expand-window-per-row 0.4 \
    #      -taper-window-exponent 0.675 \
    #      -out-dir "$BASEDIR"/shuffleddata/$OUTDIRVAL \
    #      -out-tmp-dir "$TMPDIR"/val \
    #      -approx-rows-per-out-file 70000 \
    #      -num-processes "$NTHREADS" \
    #      -batch-size 256 \
    #      -keep-target-rows 12000 \
    #      "$@" \
    #      2>&1 | tee "$BASEDIR"/shuffleddata/$OUTDIR/outval.txt &

    wait
)
#set +x

#Just in case, give a little time for nfs
sleep 10

#rm if it already exists
rm -f "$BASEDIR"/shuffleddata/current_tmp

ln -s $OUTDIR "$BASEDIR"/shuffleddata/current_tmp
mv -Tf "$BASEDIR"/shuffleddata/current_tmp "$BASEDIR"/shuffleddata/current

#Among shuffled dirs older than 2 hours, remove all but the most recent 5 of them.
#This should be very conservative and allow plenty of time for the training to switch
#to newer ones as they get generated
echo "Cleaning up any old dirs"
find "$BASEDIR"/shuffleddata/ -mindepth 1 -maxdepth 1 -type d -mmin +120 | sort | head -n -5 | xargs --no-run-if-empty rm -r

echo "Finished shuffle at" $(date "+%Y-%m-%d %H:%M:%S")
#Make a little space between shuffles
echo ""
echo ""
