#!/bin/bash -eu
if [[ $# -lt 3 ]]
then
    echo "Usage: $0 BASEDIR SCRIPTDIR KATAEXEC NTHREADS TRAININGNAME MODELKIND"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "SCRIPTDIR containing the KataGo python folder"
    echo "KATAEXEC path to the KataGo executable"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    echo "TRANINGNAME name to prefix models with, specific to this training daemon"
    echo "MODELKIND what size model to train"
    exit 0
fi
BASEDIR="$1"
shift
SCRIPTDIR="$1"
shift
KATAEXEC="$1"
shift
THREADS="$1"
shift
TRAININGNAME="$1"
shift
MODELKIND="$1"
shift

function shuffle() {
    cd "$SCRIPTDIR" && ./selfplay/shuffle.sh "$BASEDIR"/ "$BASEDIR"/scratch "$THREADS"; cd -
}

function train() {
    cd "$SCRIPTDIR" && ./selfplay/train.sh "$BASEDIR"/ "$TRAININGNAME" "$MODELKIND" $1 -max-epochs-this-instance 1 $2 $3; cd -
}

while true
do
    "$KATAEXEC" selfplay -output-dir "$BASEDIR"/selfplay -models-dir "$BASEDIR"/models -config-file selfplay1.cfg
    shuffle
    ##uncomment these lines to do cyclical learning rates
    #train trainonly -lr-scale 1.0
    #shuffle
    #train trainonly -lr-scale 0.1
    #finally generate a network
    #shuffle
    #train main -lr-scale 0.03
    
    ##comment this line when doing cyclical learning rates
    train main -lr-scale 1.0
    
    #move so we don't have to count by rows
    rsync -a "$BASEDIR"/selfplay/* "$BASEDIR"/selfplay_old --remove-source-files
    
    #1 means gatekeeper true
    cd "$SCRIPTDIR" && ./selfplay/export_model_for_selfplay.sh "$TRAININGNAME" "$BASEDIR" 1; cd -
    ./gatekeeper.sh "$KATAEXEC" "$BASEDIR" | tee -a gkeeper.txt

done
