#!/bin/bash -eu

#Takes any models in tfsavedmodels_toexport/ and archives them in tfsavedmodels/
#and outputs a cuda-runnable model file to modelstobetested/
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

mkdir -p $BASEDIR/tfsavedmodels
mkdir -p $BASEDIR/modelstobetested

for FILEPATH in $(find $BASEDIR/tfsavedmodels_toexport/ -mindepth 1 -maxdepth 1)
do
    #Make sure to skip tmp directories that are transiently there by the tensorflow training,
    #they are probably in the process of being written
    if [ ${FILEPATH: -4} == ".tmp" ]
    then
        echo "Skipping tmp file:" $FILEPATH
    else
        echo "Found model to export:" $FILEPATH
        NAME=$(basename $FILEPATH)

        SRC=$BASEDIR/tfsavedmodels_toexport/$NAME
        SRCTGZ=$BASEDIR/tfsavedmodels_toexport/$NAME.tgz
        TMPDST=$BASEDIR/tfsavedmodels_toexport/$NAME.exported
        TARGET1=$BASEDIR/tfsavedmodels/$NAME.tgz
        TARGET2=$BASEDIR/modelstobetested/$NAME

        if [ -f $TARGET1 ]
        then
            echo "Model with same name aleady exists, so skipping:" $TARGET1
        else
            mkdir $TMPDST

            set -x
            python3 ./export_modelv3.py \
                    -saved-model-dir $SRC \
                    -export-dir $TMPDST \
                    -model-name $NAME \
                    -filename-prefix model \
                    -for-cuda
            set +x

            cp $SRC/model.config.json $TMPDST/
            cp $SRC/trainhistory.json $TMPDST/

            tar -czvf $SRCTGZ $SRC
            rm -r $SRC
            gzip $TMPDST/model.txt

            mv $SRCTGZ $TARGET1
            mv $TMPDST $TARGET2
            echo "Done exporting:" $NAME "to" $TARGET1 "and" $TARGET2
        fi
    fi
done


