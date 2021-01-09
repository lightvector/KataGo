#!/bin/bash -eu

mkdir -p tests/results/gtp
rm -f tests/results/gtp/*
for CMDFILE in tests/gtp/*
do
    echo $CMDFILE
    BASENAME=$(basename "$CMDFILE")
    ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/$BASENAME.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, cudaUseNHWC = false" < $CMDFILE 1> tests/results/gtp/$BASENAME.stdout 2> tests/results/gtp/$BASENAME.stderr
done

echo 'genmove_debug b' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/defaultkomitt.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, cudaUseNHWC = false, rules=, scoringRule=AREA,koRule=POSITIONAL,multiStoneSuicideLegal=false,taxRule=NONE,hasButton=false" 1> tests/results/gtp/defaultkomitt.stdout 2> tests/results/gtp/defaultkomitt.stderr
echo 'genmove_debug b' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/defaultkomiterr.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, cudaUseNHWC = false, rules=, scoringRule=TERRITORY,koRule=POSITIONAL,multiStoneSuicideLegal=false,taxRule=NONE,hasButton=false" 1> tests/results/gtp/defaultkomiterr.stdout 2> tests/results/gtp/defaultkomiterr.stderr
echo 'genmove_debug b' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/defaultkomibutton.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, cudaUseNHWC = false, rules=, scoringRule=AREA,koRule=POSITIONAL,multiStoneSuicideLegal=false,taxRule=NONE,hasButton=true" 1> tests/results/gtp/defaultkomibutton.stdout 2> tests/results/gtp/defaultkomibutton.stderr
echo -e 'genmove_debug b\nkomi 3.5\ngenmove_debug w\nclear_board\nkomi 4.5\ngenmove_debug b\nkata-set-rules chinese\ngenmove_debug w' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/forcedkomi.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, cudaUseNHWC = false, rules=, scoringRule=AREA,koRule=POSITIONAL,multiStoneSuicideLegal=false,taxRule=NONE,hasButton=true,ignoreGTPAndForceKomi=5.5" 1> tests/results/gtp/forcedkomi.stdout 2> tests/results/gtp/forcedkomi.stderr

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


mkdir -p tests/results/analysis
rm -f tests/results/analysis/*
for CMDFILE in tests/analysis/*
do
    echo $CMDFILE
    BASENAME=$(basename "$CMDFILE")
    ./katago analysis -config configs/analysis_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/analysis/$BASENAME.log, logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, maxVisits=100, maxPlayouts=10000, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, cudaUseNHWC = false" < $CMDFILE 1> tests/results/analysis/$BASENAME.stdout 2> tests/results/analysis/$BASENAME.stderr
done

cat tests/analysis/basic.txt | ./katago analysis -config configs/analysis_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/analysis/$BASENAME.log, logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, maxVisits=100, maxPlayouts=10000, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, cudaUseNHWC = false, reportAnalysisWinratesAs=SIDETOMOVE" 1> tests/results/analysis/basic_sidetomove.stdout 2> tests/results/analysis/basic_sidetomove.stderr
