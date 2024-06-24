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
echo -e 'genmove_debug b\ngenmove_debug w\ngenmove_debug b' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/rootsymprune.log, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, rootSymmetryPruning=true, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, logSearchInfoForChosenMove = true" 1> tests/results/gtp/rootsymprune.stdout 2> tests/results/gtp/rootsymprune.stderr
echo tests/results/gtp/nologconfig
echo -e 'genmove_debug b\ngenmove_debug w\ngenmove_debug b' | ./katago gtp -config configs/gtp_example.cfg -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz -override-config "logFile=tests/results/gtp/nologconfig.log, logConfigContents=false, logDir=, logTimeStamp=false, maxVisits=100, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false" 1> tests/results/gtp/nologconfig.stdout 2> tests/results/gtp/nologconfig.stderr
echo tests/results/gtp/genmoveanalyze
echo -e 'kata-genmove_analyze b\nkomi 5\nkata-genmove_analyze w rootInfo true\nkata-genmove_analyze b rootInfo true' | ./katago gtp -config configs/gtp_example.cfg -model models/b18c384nbt-uec.bin.gz -override-config "logFile=tests/results/gtp/genmoveanalyze.log, logConfigContents=false, logDir=, logTimeStamp=false, maxVisits=30, maxPlayouts=10000, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, defaultBoardSize = 9" 1> tests/results/gtp/genmoveanalyze.stdout 2> tests/results/gtp/genmoveanalyze.stderr
echo tests/results/gtp/humansl
echo -e 'genmove b\ngenmove w\ngenmove b\ngenmove w\ngenmove b\ngenmove w\ngenmove b\ngenmove w\ngenmove b\ngenmove w\nkata-raw-human-nn 0\nkata-get-params\nkata-get-param humanSLProfile\nkata-set-param humanSLProfile preaz_5d\ngenmove b\ngenmove w\ngenmove b\ngenmove w\ngenmove b\ngenmove w\ngenmove b\ngenmove w\nkata-get-param humanSLProfile' | ./katago gtp -config configs/gtp_human5k_example.cfg -model models/b18c384nbt-uec.bin.gz -human-model models//b18c384nbt-humanv0.bin.gz -override-config "logFile=tests/results/gtp/humansl.log, logConfigContents=false, logDir=, logTimeStamp=false, numSearchThreads=1, nnRandomize=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, defaultBoardSize = 19, humanSLProfile = preaz_12k" 1> tests/results/gtp/humansl.stdout 2> tests/results/gtp/humansl.stderr

sed 's/^Time taken:.*/Time taken: ###/g' -i tests/results/gtp/*

mkdir -p tests/results/cmd
rm -f tests/results/cmd/*
./katago analysis -help > tests/results/cmd/analysis_help.stdout
./katago benchmark -help > tests/results/cmd/benchmark_help.stdout
./katago contribute -help > tests/results/cmd/contribute_help.stdout
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

echo "match"
rm -f tests/results/matchsgfs/*
rm -f tests/results/matchsgfs2/*
rm tests/results/match.txt

./katago match -config tests/data/configs/matchtest.cfg -sgf-output-dir tests/results/matchsgfs/ >> tests/results/match.txt
mv tests/results/matchsgfs/* tests/results/matchsgfs/games.sgfs
./katago match -config tests/data/configs/matchtest2.cfg -sgf-output-dir tests/results/matchsgfs2/ >> tests/results/match.txt
mv tests/results/matchsgfs2/* tests/results/matchsgfs2/games.sgfs

function countSides() {
    echo $1 >> tests/results/match.txt
    (
        echo "Black AAA "; grep 'PB\[AAA\]' $1 | wc -l
        echo "Black BBB "; grep 'PB\[BBB\]' $1 | wc -l
        echo "Black CCC "; grep 'PB\[CCC\]' $1 | wc -l
        echo "Black DDD "; grep 'PB\[DDD\]' $1 | wc -l
        echo "Black EEE "; grep 'PB\[EEE\]' $1 | wc -l
        echo "Black FFF "; grep 'PB\[FFF\]' $1 | wc -l
        echo "Black GGG "; grep 'PB\[GGG\]' $1 | wc -l
        echo "Black HHH "; grep 'PB\[HHH\]' $1 | wc -l
    ) >> tests/results/match.txt
    (
        echo "White AAA "; grep 'PW\[AAA\]' $1 | wc -l
        echo "White BBB "; grep 'PW\[BBB\]' $1 | wc -l
        echo "White CCC "; grep 'PW\[CCC\]' $1 | wc -l
        echo "White DDD "; grep 'PW\[DDD\]' $1 | wc -l
        echo "White EEE "; grep 'PW\[EEE\]' $1 | wc -l
        echo "White FFF "; grep 'PW\[FFF\]' $1 | wc -l
        echo "White GGG "; grep 'PW\[GGG\]' $1 | wc -l
        echo "White HHH "; grep 'PW\[HHH\]' $1 | wc -l
    ) >> tests/results/match.txt
}
grep -v 'Avg move time used' tests/results/match.txt > tests/results/match.txt.tmp
mv tests/results/match.txt.tmp tests/results/match.txt
sed -i 's/nnRandSeed. = .*$/nnRandSeed = ###/g' tests/results/match.txt
sed -i 's/Git revision: .*$/Git revision: ###/g' tests/results/match.txt
countSides tests/results/matchsgfs/games.sgfs
countSides tests/results/matchsgfs2/games.sgfs

rm -f tests/results/sampletest-vf/*.log
./katago samplesgfs \
         -sgfdir tests/data/sampletest/ \
         -outdir tests/results/sampletest-vf/ \
         -sample-prob 1 \
         -min-turn-number-board-area-prop 0.15 \
         -force-sample-weight 10.0 \
         -value-fluctuation-model tests/models/g170-b6c96-s175395328-d26788732.bin.gz \
         -value-fluctuation-turn-scale 4.0 \
         -hash-comments \
         -training-weight 0.46 \
         -min-weight 0.01 \
         -turn-weight-lambda 0 \
         -for-testing

rm -f tests/results/sampletest-basic/*.log
./katago samplesgfs \
         -sgfdir tests/data/sampletest/ \
         -outdir tests/results/sampletest-basic/ \
         -sample-prob 1 \
         -min-turn-number-board-area-prop 0.15 \
         -force-sample-weight 10.0 \
         -hash-comments \
         -training-weight 0.36 \
         -min-weight 0.01 \
         -turn-weight-lambda 0.01 \
         -for-testing

rm -f tests/results/sampletest-hint/*.log
./katago dataminesgfs \
         -config configs/gtp_example.cfg \
         -override-config "logTimeStamp=false, maxVisits=50, numSearchThreads=1, nnRandomize=false, rootSymmetryPruning=false, nnRandSeed=forTesting, searchRandSeed=forTesting, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false, koRules=SIMPLE, scoringRules=AREA, taxRules=NONE, hasButtons=false, multiStoneSuicideLegals=false, bSizes=9, bSizeRelProbs=1, komiAuto=true" \
         -sgfdir tests/data/sampletest/ \
         -outdir tests/results/sampletest-hint/ \
         -threads 1 \
         -tree-mode \
         -min-hint-weight 0.25 \
         -model tests/models/g170-b6c96-s175395328-d26788732.bin.gz \
         -auto-komi \
         -for-testing

echo "Done"




