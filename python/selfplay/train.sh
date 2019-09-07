#!/bin/bash -eu

#Runs tensorflow training in train/$TRAININGNAME
#Should be run once per persistent training process.
#Outputs results in tfsavedmodels_toexport/ in an ongoing basis.

if [[ $# -lt 3 ]]
then
    echo "Usage: $0 BASEDIR TRAININGNAME MODELKIND OTHERARGS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TRANINGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train"
    exit 0
fi
BASEDIR="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift

#------------------------------------------------------------------------------
set -x

mkdir -p "$BASEDIR"/train/"$TRAININGNAME"
git show --no-patch --no-color > "$BASEDIR"/train/"$TRAININGNAME"/version.txt
git diff --no-color > "$BASEDIR"/train/"$TRAININGNAME"/diff.txt
git diff --staged --no-color > "$BASEDIR"/train/"$TRAININGNAME"/diffstaged.txt

time python3 ./train.py \
     -traindir "$BASEDIR"/train/"$TRAININGNAME" \
     -datadir "$BASEDIR"/shuffleddata/current/ \
     -exportdir "$BASEDIR"/tfsavedmodels_toexport \
     -exportprefix "$TRAININGNAME" \
     -pos-len 19 \
     -batch-size 128 \
     -samples-per-epoch 1000000 \
     -gpu-memory-frac 0.6 \
     -model-kind "$MODELKIND" \
     -sub-epochs 4 \
     -swa-sub-epoch-scale 4 \
     "$@" \
     2>&1 | tee -a "$BASEDIR"/train/"$TRAININGNAME"/stdout.txt
