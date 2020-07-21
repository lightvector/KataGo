#!/bin/bash -eu
set -o pipefail
{
#Shuffles and copies selfplay training from selfplay/ to shuffleddata/current/
#Should be run periodically.

if [[ $# -lt 4 ]]
then
    echo "Usage: $0 BASEDIR TMPDIR NTHREADS BATCHSIZE"
    echo "Currently expects to be run from within the 'python' directory of the KataGo repo, or otherwise in the same dir as shuffle.py."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TMPDIR scratch space, ideally on fast local disk, unique to this loop"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    echo "BATCHSIZE number of samples to concat together per batch for training"
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
         -expand-window-per-row 0.4 \
         -taper-window-exponent 0.65 \
         -out-dir "$BASEDIR"/shuffleddata/$OUTDIRTRAIN \
         -out-tmp-dir "$TMPDIR"/train \
         -approx-rows-per-out-file 70000 \
         -num-processes "$NTHREADS" \
         -batch-size "$BATCHSIZE" \
         "$@" \
         2>&1 | tee "$BASEDIR"/shuffleddata/$OUTDIR/outtrain.txt &

    wait
)
#set +x

#Just in case, give a little time for nfs
sleep 10

#rm if it already exists
rm -f "$BASEDIR"/shuffleddata/current_tmp

ln -s $OUTDIR "$BASEDIR"/shuffleddata/current_tmp
mv -Tf "$BASEDIR"/shuffleddata/current_tmp "$BASEDIR"/shuffleddata/current

# CLEANUP ---------------------------------------------------------------

#Among shuffled dirs older than 2 hours, remove all but the most recent 5 of them.
#This should be VERY conservative and allow plenty of time for the training to switch
#to newer ones as they get generated.
echo "Cleaning up any old dirs"
find "$BASEDIR"/shuffleddata/ -mindepth 1 -maxdepth 1 -type d -mmin +120 | sort | head -n -5 | xargs --no-run-if-empty rm -r

echo "Finished shuffle at" $(date "+%Y-%m-%d %H:%M:%S")
#Make a little space between shuffles
echo ""
echo ""

exit 0
}
