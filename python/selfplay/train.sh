#!/bin/bash -eu

#Runs tensorflow training in $BASEDIR/train/$TRAININGNAME
#Should be run once per persistent training process.
#Outputs results in tfsavedmodels_toexport/ in an ongoing basis (EXPORTMODE == "main").
#Or, to tfsavedmodels_toexport_extra/ (EXPORTMODE == "extra").
#Or just trains without exporting (EXPORTMODE == "trainonly").

if [[ $# -lt 4 ]]
then
    echo "Usage: $0 BASEDIR TRAININGNAME MODELKIND EXPORTMODE OTHERARGS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TRANINGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train, like b10c128, see ../modelconfigs.py"
    echo "EXPORTMODE 'main': train and export for selfplay. 'extra': train and export extra non-selfplay model. 'trainonly': train without export"
    exit 0
fi
BASEDIR="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift
EXPORTMODE="$1"
shift

GITROOTDIR="$(git rev-parse --show-toplevel)"

#------------------------------------------------------------------------------
set -x

mkdir -p "$BASEDIR"/train/"$TRAININGNAME"
git show --no-patch --no-color > "$BASEDIR"/train/"$TRAININGNAME"/version.txt
git diff --no-color > "$BASEDIR"/train/"$TRAININGNAME"/diff.txt
git diff --staged --no-color > "$BASEDIR"/train/"$TRAININGNAME"/diffstaged.txt

if [ "$EXPORTMODE" == "main" ]
then
    EXPORT_SUBDIR=tfsavedmodels_toexport
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "extra" ]
then
    EXPORT_SUBDIR=tfsavedmodels_toexport_extra
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "trainonly" ]
then
    EXPORT_SUBDIR=tfsavedmodels_toexport_extra
    EXTRAFLAG="-no-export"
else
    echo "EXPORTMODE was not 'main' or 'extra' or 'trainonly', run with no arguments for usage"
    exit 1
fi

time python3 "$GITROOTDIR"/python/train.py \
     -traindir "$BASEDIR"/train/"$TRAININGNAME" \
     -datadir "$BASEDIR"/shuffleddata/current/ \
     -exportdir "$BASEDIR"/"$EXPORT_SUBDIR" \
     -exportprefix "$TRAININGNAME" \
     -pos-len 19 \
     -batch-size 256 \
     -samples-per-epoch 1000000 \
     -gpu-memory-frac 0.6 \
     -model-kind "$MODELKIND" \
     -sub-epochs 4 \
     -swa-sub-epoch-scale 4 \
     $EXTRAFLAG \
     "$@" \
     2>&1 | tee -a "$BASEDIR"/train/"$TRAININGNAME"/stdout.txt
