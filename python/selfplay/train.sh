#!/bin/bash -eu
set -o pipefail
{
# Runs training in $BASEDIR/train/$TRAININGNAME
# Should be run once per persistent training process.
# Outputs results in torchmodels_toexport/ in an ongoing basis (EXPORTMODE == "main").
# Or, to torchmodels_toexport_extra/ (EXPORTMODE == "extra").
# Or just trains without exporting (EXPORTMODE == "trainonly").

if [[ $# -lt 6 ]]
then
    echo "Usage: $0 BASEDIR TRAININGNAME MODELKIND BATCHSIZE EXPORTMODE OTHERARGS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TRAININGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train, like b10c128, see ../modelconfigs.py"
    echo "BATCHSIZE number of samples to concat together per batch for training, must match shuffle"
    echo "LRSCALE scale factor for learning rate, 1.0 is default, 0.5 is half, etc"
    echo "EXPORTMODE 'main': train and export for selfplay. 'extra': train and export extra non-selfplay model. 'trainonly': train without export"
    exit 0
fi
BASEDIR="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift
BATCHSIZE="$1"
shift
LRSCALE="$1"
shift
EXPORTMODE="$1"
shift

#------------------------------------------------------------------------------
set -x

mkdir -p "$BASEDIR"/train/"$TRAININGNAME"

if [[ -n $(pwd | grep "^$BASEDIR/scripts/") ]]
then
    echo "Already running out of snapshotted scripts directory, not snapshotting again"
else
    GITROOTDIR="$(git rev-parse --show-toplevel)"

    git show --no-patch --no-color > "$BASEDIR"/train/"$TRAININGNAME"/version.txt
    git diff --no-color > "$BASEDIR"/train/"$TRAININGNAME"/diff.txt
    git diff --staged --no-color > "$BASEDIR"/train/"$TRAININGNAME"/diffstaged.txt

    # For archival and logging purposes - you can look back and see exactly the python code on a particular date
    DATE_FOR_FILENAME=$(date "+%Y%m%d-%H%M%S")
    DATED_ARCHIVE="$BASEDIR"/scripts/train/dated/"$DATE_FOR_FILENAME"
    mkdir -p "$DATED_ARCHIVE"
    cp "$GITROOTDIR"/python/*.py "$GITROOTDIR"/python/selfplay/train.sh "$DATED_ARCHIVE"
    git show --no-patch --no-color > "$DATED_ARCHIVE"/version.txt
    git diff --no-color > "$DATED_ARCHIVE"/diff.txt
    git diff --staged --no-color > "$DATED_ARCHIVE"/diffstaged.txt
    cd "$DATED_ARCHIVE"
fi

if [ "$EXPORTMODE" == "main" ]
then
    EXPORT_SUBDIR=torchmodels_toexport
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "extra" ]
then
    EXPORT_SUBDIR=torchmodels_toexport_extra
    EXTRAFLAG=""
elif [ "$EXPORTMODE" == "trainonly" ]
then
    EXPORT_SUBDIR=torchmodels_toexport_extra
    EXTRAFLAG="-no-export"
else
    echo "EXPORTMODE was not 'main' or 'extra' or 'trainonly', run with no arguments for usage"
    exit 1
fi

# Find latest shuffle dir
LATEST_DATA=$(ls -td1 "$BASEDIR"/shuffleddata/* | head -n 1)

time python ./train.py \
     -traindir "$BASEDIR"/train/"$TRAININGNAME" \
     -datadir "$LATEST_DATA" \
     -exportdir "$BASEDIR"/"$EXPORT_SUBDIR" \
     -exportprefix "$TRAININGNAME" \
     -pos-len 19 \
     -batch-size "$BATCHSIZE" \
     -lr-scale "$LRSCALE" \
     -model-kind "$MODELKIND" \
     $EXTRAFLAG \
     "$@" \
     2>&1 | tee -a "$BASEDIR"/train/"$TRAININGNAME"/stdout.txt

exit 0
}
