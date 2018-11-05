#!/bin/bash -eu

#Shuffles and copies selfplay training from selfplay/ to shuffleddata/current/
#Should be run periodically.

if [[ $# -ne 1 ]]
then
    echo "Usage: $0 BASEDIR"
    echo "BASEDIR containing selfplay data and models and related directories"
    exit 0
fi
BASEDIR=$1
shift

#------------------------------------------------------------------------------

OUTDIR=$(date "+%Y%m%d-%H%M%S")
OUTDIRTRAIN=$OUTDIR/train
OUTDIRVAL=$OUTDIR/val

mkdir -p $BASEDIR/shuffleddata/$OUTDIR

set -x
time python3 ./shuffle.py \
     $BASEDIR/selfplay/*/tdata/ \
     -min-rows 500000 \
     -max-rows 100000000 \
     -window-factor 2 \
     -out-dir $BASEDIR/shuffleddata/$OUTDIRTRAIN \
     -approx-rows-per-out-file 500000 \
     -num-processes 4 \
     -batch-size 256 \
    2>&1 | tee $BASEDIR/shuffleddata/$OUTDIR/outtrain.txt

time python3 ./shuffle.py \
     $BASEDIR/selfplay/*/vdata/ \
     -min-rows 0 \
     -max-rows 5000000 \
     -window-factor 2 \
     -out-dir $BASEDIR/shuffleddata/$OUTDIRVAL \
     -approx-rows-per-out-file 500000 \
     -num-processes 4 \
     -batch-size 256 \
     -keep-target-rows 40000 \
    2>&1 | tee $BASEDIR/shuffleddata/$OUTDIR/outval.txt
set +x

#Just in case, give a little time for nfs
sleep 30

ln -s $OUTDIR $BASEDIR/shuffleddata/current_tmp
mv -Tf $BASEDIR/shuffleddata/current_tmp $BASEDIR/shuffleddata/current

#Among shuffled dirs older than 2 days, remove all but the most recent 5 of them.
#This should be very conservative and allow plenty of time for the training to switch
#to newer ones as they get generated
echo "Cleaning up any old dirs"
find $BASEDIR/shuffleddata/* -maxdepth 0 -type d -mmin +2880 | sort | head -n -5 | xargs --no-run-if-empty rm -r

echo "Finished shuffle at" $(date "+%Y-%m-%d %H:%M:%S")
