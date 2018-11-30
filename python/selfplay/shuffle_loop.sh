#!/bin/bash -eu

if [[ $# -ne 2 ]]
then
    echo "Usage: $0 BASEDIR NTHREADS"
    echo "BASEDIR containing selfplay data and models and related directories"
    echo "NTHREADS number of parallel threads/processes to use in shuffle"
    exit 0
fi
BASEDIR=$1
shift
NTHREADS=$1
shift

while true
do
    ./selfplay/shuffle.sh $BASEDIR $NTHREADS
    sleep 300
done
