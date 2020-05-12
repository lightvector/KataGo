#!/bin/bash -eu

mkdir -p tests/results/gtp
rm -f tests/results/gtp/*
for CMDFILE in tests/gtp/*
do
    echo $CMDFILE
    BASENAME=$(basename "$CMDFILE")
    ./katago gtp -config configs/gtp_example.cfg -model models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/$BASENAME.log, logDir=, logTimeStamp=false, maxVisits=100, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, cudaUseNHWC = false" < $CMDFILE 1> tests/results/gtp/$BASENAME.stdout 2> tests/results/gtp/$BASENAME.stderr
done

sed 's/^Time taken:.*/Time taken: ###/g' -i tests/results/gtp/*

mkdir -p tests/results/cmd
rm -f tests/results/cmd/*
./katago analysis -help > tests/results/cmd/analysis_help.stdout
./katago benchmark -help > tests/results/cmd/benchmark_help.stdout
./katago genconfig -help > tests/results/cmd/genconfig_help.stdout
./katago gtp -help > tests/results/cmd/gtp_help.stdout
./katago gatekeeper -help > tests/results/cmd/gatekeeper_help.stdout
./katago match -help > tests/results/cmd/match_help.stdout
./katago selfplay -help > tests/results/cmd/selfplay_help.stdout
