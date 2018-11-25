#!/bin/bash -eu

#Runs tensorflow training in train/$TRAININGNAME
#Should be run once per persistent training process.
#Outputs results in tfsavedmodels_toexport/ in an ongoing basis.

if [[ $# -ne 2 ]]
then
    echo "Usage: $0 BASEDIR TRAININGNAME"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TRANINGNAME name to suffix models with, specific to this training daemon"
    exit 0
fi
BASEDIR=$1
shift
TRAININGNAME=$1
shift

#------------------------------------------------------------------------------
set -x

mkdir -p $BASEDIR/train/$TRAININGNAME
git show --no-patch --no-color > $BASEDIR/train/$TRAININGNAME/version.txt
git diff --no-color > $BASEDIR/train/$TRAININGNAME/diff.txt
git diff --staged --no-color > $BASEDIR/train/$TRAININGNAME/diffstaged.txt

time python3 ./trainv3.py \
     -traindir $BASEDIR/train/$TRAININGNAME \
     -datadir $BASEDIR/shuffleddata/current/ \
     -exportdir $BASEDIR/tfsavedmodels_toexport \
     -exportsuffix $TRAININGNAME \
     -pos-len 19 \
     -batch-size 256 \
     -gpu-memory-frac 0.7 \
     2>&1 | tee $BASEDIR/train/$TRAININGNAME/stdout.txt
