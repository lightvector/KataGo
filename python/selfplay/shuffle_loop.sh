#!/bin/bash -eu

if [[ $# -ne 3 ]]
then
    echo "Usage: $0 BASEDIR TMPDIR NTHREADS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "TMPDIR scratch space, ideally on fast local disk, unique to this loop"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    exit 0
fi
BASEDIR=$1
shift
TMPDIR=$1
shift
NTHREADS=$1
shift

while true
do
    ./selfplay/shuffle.sh $BASEDIR $TMPDIR $NTHREADS
    sleep 30
done
