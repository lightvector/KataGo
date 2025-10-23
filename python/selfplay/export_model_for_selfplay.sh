#!/bin/bash -eu
set -o pipefail
{
#Takes any models in torchmodels_toexport/ and outputs a cuda-runnable model file to modelstobetested/
#Takes any models in torchmodels_toexport_extra/ and outputs a cuda-runnable model file to models_extra/
#Should be run periodically.

if [[ $# -ne 3 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR USEGATING"
    echo "Currently expects to be run from within the 'python' directory of the KataGo repo, or otherwise in the same dir as export_model.py."
    echo "NAMEPREFIX string prefix for this training run, try to pick something globally unique. Will be displayed to users when KataGo loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "USEGATING = 1 to use gatekeeper, 0 to not use gatekeeper and output directly to models/"
    exit 0
fi
NAMEPREFIX="$1"
shift
BASEDIR="$1"
shift
USEGATING="$1"
shift

#------------------------------------------------------------------------------

mkdir -p "$BASEDIR"/torchmodels_toexport
mkdir -p "$BASEDIR"/torchmodels_toexport_extra
mkdir -p "$BASEDIR"/modelstobetested
mkdir -p "$BASEDIR"/models_extra
mkdir -p "$BASEDIR"/models

function exportStuff() {
    FROMDIR="$1"
    TODIR="$2"

    #Sort by timestamp so that we process in order of oldest to newest if there are multiple
    for FILEPATH in $(gfind "$BASEDIR"/"$FROMDIR"/ -mindepth 1 -maxdepth 1 -printf "%T@ %p\n" | sort -n | cut -d ' ' -f 2)
    do
        #Make sure to skip tmp directories that are transiently there by the training,
        #they are probably in the process of being written
        if [ ${FILEPATH: -4} == ".tmp" ]
        then
            echo "Skipping tmp file:" "$FILEPATH"
        elif [ ${FILEPATH: -9} == ".exported" ]
        then
            echo "Skipping self tmp file:" "$FILEPATH"
        else
            echo "Found model to export:" "$FILEPATH"
            NAME="$(basename "$FILEPATH")"

            SRC="$BASEDIR"/"$FROMDIR"/"$NAME"
            TMPDST="$BASEDIR"/"$FROMDIR"/"$NAME".exported
            TARGET="$BASEDIR"/"$TODIR"/"$NAME"

            if [ -d "$BASEDIR"/modelstobetested/"$NAME" ] ||  \
               [ -d "$BASEDIR"/rejectedmodels/"$NAME" ] || \
               [ -d "$BASEDIR"/models/"$NAME" ] || \
               [ -d "$BASEDIR"/models_extra/"$NAME" ] || \
               [ -d "$BASEDIR"/modelsuploaded/"$NAME" ]
            then
                echo "Model with same name aleady exists, so skipping:" "$SRC"
            else
                rm -rf "$TMPDST"
                mkdir "$TMPDST"

                set -x
                python ./export_model_pytorch.py \
                        -checkpoint "$SRC"/model.ckpt \
                        -export-dir "$TMPDST" \
                        -model-name "$NAMEPREFIX""-""$NAME" \
                        -filename-prefix model \
                        -use-swa

                python ./clean_checkpoint.py \
                        -checkpoint "$SRC"/model.ckpt \
                        -output "$TMPDST"/model.ckpt
                set +x

                rm -r "$SRC"

                python ./convert_coreml_pytorch.py \
                       -checkpoint "$TMPDST"/model.ckpt \
                       -output "$TMPDST"/KataGoModel19x19fp16.mlpackage \
                       -use-swa

                gzip "$TMPDST"/model.bin

                #Make a bunch of the directories that selfplay will need so that there isn't a race on the selfplay
                #machines to concurrently make it, since sometimes concurrent making of the same directory can corrupt
                #a filesystem
                #Only when not gating. When gating, gatekeeper is responsible.
                if [ "$USEGATING" -eq 0 ]
                then
                    if [ "$TODIR" != "models_extra" ]
                    then
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"/sgfs
                        mkdir -p "$BASEDIR"/selfplay/"$NAME"/tdata
                    fi
                fi

                #Sleep a little to allow some tolerance on the filesystem
                sleep 5

                mv "$TMPDST" "$TARGET"
                echo "Done exporting:" "$NAME" "to" "$TARGET"
            fi
        fi
    done
}

if [ "$USEGATING" -eq 0 ]
then
    exportStuff "torchmodels_toexport" "models"
else
    exportStuff "torchmodels_toexport" "modelstobetested"
fi
exportStuff "torchmodels_toexport_extra" "models_extra"

exit 0
}
