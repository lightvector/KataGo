#!/bin/bash -eu

if [[ $# -lt 5 ]]
then
    echo "Usage: $0 BASEDIR SCRIPTDIR TRAININGNAME MODELKIND EXPORTMODE OTHERARGS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "SCRIPTDIR containing the KataGo python folder"
    echo "TRANINGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train"
    echo "EXPORTMODE 'main': train and export for selfplay. 'extra': train and export extra non-selfplay model. 'trainonly': train without export"
    exit 0
fi

BASEDIR="$1"
shift
SCRIPTDIR="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift
EXPORTMODE="$1"
shift

SELFPLAY="$PWD"/..

cd "$SCRIPTDIR" && "$SELFPLAY"/train.sh "$BASEDIR"/ "$TRAININGNAME" "$MODELKIND" "$EXPORTMODE" -max-epochs-this-instance 1 "$*"
