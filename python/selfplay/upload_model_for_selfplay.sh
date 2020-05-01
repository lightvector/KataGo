#!/bin/bash -eu
set -o pipefail
{
#Takes any models in modelstobetested/ and uploads them, then moves them to modelsuploaded/
#Should be run periodically.

if [[ $# -ne 2 ]]
then
    echo "Usage: $0 NAMEPREFIX BASEDIR"
    echo "Currently expects to be run from within the `python` directory of the KataGo repo, or otherwise in the same dir as export_model.py."
    echo "NAMEPREFIX string prefix for this training run, try to pick something globally unique. Will be displayed to users when KataGo loads the model."
    echo "BASEDIR containing selfplay data and models and related directories"
    exit 0
fi
NAMEPREFIX="$1"
shift
BASEDIR="$1"
shift

#------------------------------------------------------------------------------

mkdir -p "$BASEDIR"/modelstobetested
mkdir -p "$BASEDIR"/modelsuploaded

function uploadStuff() {
    FROMDIR="$1"
    TODIR="$2"

    for FILEPATH in $(find "$BASEDIR"/"$FROMDIR"/ -mindepth 1 -maxdepth 1)
    do
        if [ ${FILEPATH: -10} == ".uploading" ]
        then
            echo "Skipping upload tmp file:" "$FILEPATH"
        else
            echo "Found model to export:" "$FILEPATH"
            NAME="$(basename "$FILEPATH")"

            SRC="$BASEDIR"/"$FROMDIR"/"$NAME"
            TMPDST="$BASEDIR"/"$FROMDIR"/"$NAME".uploading
            TARGETDIR="$BASEDIR"/"$TODIR"
            TARGET="$BASEDIR"/"$TODIR"/"$NAME"

            if [ -d "$BASEDIR"/modelsuploaded/"$NAME" ]
            then
                echo "Model with same name aleady exists, so skipping:" "$SRC"
            else
                rm -rf "$TMPDST"
                mkdir "$TMPDST"

                cp "$SRC"/model.bin "$TMPDST"/"NAMEPREFIX"-"$NAME".bin
                gzip "$TMPDST"/"NAMEPREFIX"-"$NAME".bin
                cp "$SRC"/model.config.json "$TMPDST"/model.config.json
                cp -r "$SRC"/saved_model "$TMPDST"/saved_model
                zip -r "$TMPDST"/"NAMEPREFIX"-"$NAME".zip "$TMPDST"/"NAMEPREFIX"-"$NAME".bin.gz "$TMPDST"/model.config.json "$TMPDST"/saved_model
                rm -r "$TMPDST"/"NAMEPREFIX"-"$NAME".bin.gz "$TMPDST"/model.config.json "$TMPDST"/saved_model
                cp -r "$SRC"/non_swa_saved_model "$TMPDST"/non_swa_saved_model
                zip -r "$TMPDST"/"NAMEPREFIX"-"$NAME"_non_swa.zip "$TMPDST"/non_swa_saved_model
                rm -r "$TMPDST"/non_swa_saved_model
                cp -r "$SRC"/trainhistory.json "$TMPDST"/trainhistory.json
                cp -r "$SRC"/log.txt "$TMPDST"/log.txt

                #Sleep a little to allow some tolerance on the filesystem
                sleep 3

                set -x
                python3 ./upload_model.py \
                        -run-name "$NAMEPREFIX" \
                        -model-name "$NAME" \
                        -model-file "$TMPDST"/"NAMEPREFIX"-"$NAME".bin.gz \
                        -model-zip "$TMPDST"/"NAMEPREFIX"-"$NAME".zip \
                        -upload-log-file "$TMPDST"/upload_log.txt \
                        -uploaded_dir "$TARGETDIR" \
                        -base-server-url localhost
                set +x

                #Sleep a little to allow some tolerance on the filesystem
                sleep 3

                mv "$TMPDST" "$TARGET"
                rm -r "$SRC"

                #Sleep a little to allow some tolerance on the filesystem
                sleep 3

                echo "Done exporting:" "$NAME" "to" "$TARGET"
            fi
        fi
    done
}

uploadStuff "modelstobetested" "modelsuploaded"

exit 0
}
