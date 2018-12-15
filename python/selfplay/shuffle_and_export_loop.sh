#!/bin/bash -eu

if [[ $# -ne 3 ]]
then
    echo "Usage: $0 BASEDIR TMPDIR NTHREADS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TMPDIR scratch space, ideally on fast local disk, unique to this loop"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    exit 0
fi
BASEDIRRAW=$1
shift
TMPDIR=$1
shift
NTHREADS=$1
shift

basedir=$(realpath $BASEDIRRAW)

mkdir -p $basedir/scripts
cp ./*.py ./selfplay/*.sh $basedir/scripts

(
    cd $basedir/scripts
    while true
    do
        ./shuffle.sh $basedir $TMPDIR $NTHREADS
        sleep 30
    done
) >> outshuffle.txt 2>&1 &

(
    cd $basedir/scripts
    while true
    do
        ./export_modelv3_for_selfplay.sh $basedir
        sleep 10
    done
) >> outexport.txt 2>&1 &
