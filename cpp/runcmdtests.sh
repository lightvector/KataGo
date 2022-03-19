#!/bin/bash -eu

mkdir -p tests/results/gtp
rm -f tests/results/gtp/*
for CMDFILE in tests/gtp/*
do
    echo $CMDFILE
    BASENAME=$(basename "$CMDFILE")
    ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/$BASENAME.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false" < $CMDFILE 1> tests/results/gtp/$BASENAME.stdout 2> tests/results/gtp/$BASENAME.stderr
done

echo tests/results/gtp/defaultkomitt
echo 'genmove_debug b' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/defaultkomitt.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, rules=, scoringRule=AREA,koRule=POSITIONAL,multiStoneSuicideLegal=false,taxRule=NONE,hasButton=false" 1> tests/results/gtp/defaultkomitt.stdout 2> tests/results/gtp/defaultkomitt.stderr
echo tests/results/gtp/defaultkomiterr
echo 'genmove_debug b' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/defaultkomiterr.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, rules=, scoringRule=TERRITORY,koRule=POSITIONAL,multiStoneSuicideLegal=false,taxRule=NONE,hasButton=false" 1> tests/results/gtp/defaultkomiterr.stdout 2> tests/results/gtp/defaultkomiterr.stderr
echo tests/results/gtp/defaultkomibutton
echo 'genmove_debug b' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/defaultkomibutton.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, rules=, scoringRule=AREA,koRule=POSITIONAL,multiStoneSuicideLegal=false,taxRule=NONE,hasButton=true" 1> tests/results/gtp/defaultkomibutton.stdout 2> tests/results/gtp/defaultkomibutton.stderr
echo tests/results/gtp/forcedkomi
echo -e 'genmove_debug b\nkomi 3.5\ngenmove_debug w\nclear_board\nkomi 4.5\ngenmove_debug b\nkata-set-rules chinese\ngenmove_debug w' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/forcedkomi.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, rules=, scoringRule=AREA,koRule=POSITIONAL,multiStoneSuicideLegal=false,taxRule=NONE,hasButton=true,ignoreGTPAndForceKomi=5.5" 1> tests/results/gtp/forcedkomi.stdout 2> tests/results/gtp/forcedkomi.stderr
echo tests/results/gtp/avoidcorners
echo -e 'genmove_debug b\ngenmove_debug w\ngenmove_debug b\ngenmove_debug w\n' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/avoidcorners.log, logDir=, logTimeStamp=false, maxVisits=1500, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, avoidSgfPatternUtility=0.3, avoidSgfPatternLambda=0.0, avoidSgfPatternMinTurnNumber=0, avoidSgfPatternMaxFiles=100, avoidSgfPatternAllowedNames=, avoidSgfPatternDirs=tests/data/cornermoves.sgf, rootPolicyTemperature=1.5, cpuctUtilityStdevScale=0.4" 1> tests/results/gtp/avoidcorners.stdout 2> tests/results/gtp/avoidcorners.stderr
echo tests/results/gtp/rootsymprune
echo -e 'genmove_debug b\ngenmove_debug w\ngenmove_debug b' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/rootsymprune.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, rootSymmetryPruning=true, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false" 1> tests/results/gtp/rootsymprune.stdout 2> tests/results/gtp/rootsymprune.stderr

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
    ./katago analysis -config configs/analysis_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/analysis/$BASENAME.log, logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, maxVisits=100, maxPlayouts=10000, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false" < $CMDFILE 1> tests/results/analysis/$BASENAME.stdout 2> tests/results/analysis/$BASENAME.stderr
done

cat tests/analysis/basic.txt | ./katago analysis -config configs/analysis_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/analysis/basic_sidetomove.txt.log, logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, maxVisits=100, maxPlayouts=10000, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, reportAnalysisWinratesAs=SIDETOMOVE" 1> tests/results/analysis/basic_sidetomove.stdout 2> tests/results/analysis/basic_sidetomove.stderr

cat tests/analysis/symmetry.txt | ./katago analysis -config configs/analysis_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/analysis/symmetry_with_pruning.txt.log, logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, maxVisits=100, maxPlayouts=10000, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, rootSymmetryPruning=true, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false" 1> tests/results/analysis/symmetry_with_pruning.stdout 2> tests/results/analysis/symmetry_with_pruning.stderr

cat tests/analysis/pvvisits.txt | ./katago analysis -config configs/analysis_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/analysis/pvvisits_nograph.txt.log, logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, useGraphSearch=false" 1> tests/results/analysis/pvvisits_nograph.stdout 2> tests/results/analysis/pvvisits_nograph.stderr

echo "checkbook"
./katago checkbook -book-file tests/data/test.katabook > tests/results/checkbook.txt
echo "Done"


