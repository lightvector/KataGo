#!/bin/bash -eu

mkdir -p tests/results/gtp
rm -f tests/results/gtp/*
for CMDFILE in tests/gtp/*
do
    echo $CMDFILE
    BASENAME=$(basename "$CMDFILE")
    ./katago gtp -config configs/gtp_example.cfg -model models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/$BASENAME.log, logTimeStamp=false, maxVisits=100, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, cudaUseFP16 = false, cudaUseNHWC = false" < $CMDFILE 1> tests/results/gtp/$BASENAME.stdout 2> tests/results/gtp/$BASENAME.stderr
done

sed 's/^Time taken:.*/Time taken: ###/g' -i tests/results/gtp/*
