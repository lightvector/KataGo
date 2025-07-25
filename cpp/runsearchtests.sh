#!/bin/bash -eux
set -o pipefail
{
(
    mkdir -p models
    cd models/
    wget -nc https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b40c256-s12350780416-d3055274313.bin.gz
)
(
    mkdir -p models
    cd models/
    wget -nc https://github.com/lightvector/KataGo/releases/download/v1.12.1/b18c384nbt-uec.bin.gz
)
(
    mkdir -p models
    cd models/
    wget -nc https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s8326494464-d4628051565.bin.gz
)
(
    mkdir -p models
    cd models/
    wget -nc https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
)
./katago runsearchtests tests/models/g103-b6c96-s103408384-d26419149.txt.gz false false 0 false | tee tests/results/runSearchTests-iNCHW-cNCHW.txt
./katago runsearchtests tests/models/g103-b6c96-s103408384-d26419149.txt.gz true false 0 false | tee tests/results/runSearchTests-iNHWC-cNCHW.txt
./katago runsearchtests tests/models/g103-b6c96-s103408384-d26419149.txt.gz true true 0 false | tee tests/results/runSearchTests-iNHWC-cNHWC.txt
./katago runsearchtests tests/models/g103-b6c96-s103408384-d26419149.txt.gz true false 1 false | tee tests/results/runSearchTests-s1.txt

./katago runsearchtests tests/models/run4-s67105280-d24430742-b6c96.txt.gz false false 0 false | tee tests/results/runSearchTests-r4-iNCHW-cNCHW.txt
./katago runsearchtests tests/models/run4-s67105280-d24430742-b6c96.txt.gz true false 0 false | tee tests/results/runSearchTests-r4-iNHWC-cNCHW.txt
./katago runsearchtests tests/models/run4-s67105280-d24430742-b6c96.txt.gz true true 0 false | tee tests/results/runSearchTests-r4-iNHWC-cNHWC.txt
./katago runsearchtests tests/models/run4-s67105280-d24430742-b6c96.txt.gz true false 1 false | tee tests/results/runSearchTests-r4-s1.txt

./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz false false 0 false | tee tests/results/runSearchTestsV3-g2-iNCHW-cNCHW.txt
./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz true false 0 false | tee tests/results/runSearchTestsV3-g2-iNHWC-cNCHW.txt
./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz true true 0 false | tee tests/results/runSearchTestsV3-g2-iNHWC-cNHWC.txt
./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz true false 1 false | tee tests/results/runSearchTestsV3-g2-s1.txt
./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz false false 5 false | tee tests/results/runSearchTestsV3-g2-s5.txt

./katago runnnontinyboardtest tests/models/g170-b6c96-s175395328-d26788732.bin.gz false false 3 false | tee tests/results/runNNOnTinyBoardTest.txt
./katago runnnontinyboardtest models/kata1-b28c512nbt-s8326494464-d4628051565.bin.gz false false 6 false | tee tests/results/runNNOnTinyBoardTestB28.txt
./katago runselfplayinittests tests/models/grun50-b6c96-s156348160-d118286860.txt.gz | tee tests/results/runSelfplayInitTests.txt
./katago runsekitrainwritetests tests/models/g103-b6c96-s103408384-d26419149.txt.gz | tee tests/results/runSekiTrainWriteTests.txt
./katago runnnsymmetriestest tests/models/g103-b6c96-s103408384-d26419149.txt.gz false false false | tee tests/results/runNNSymmetriesTest.txt
./katago runselfplayinitstattests tests/models/g170-b6c96-s175395328-d26788732.bin.gz | tee tests/results/runSelfplayInitStatTests.txt

./katago runsearchtestsv8 tests/models/g170-b6c96-s175395328-d26788732.txt.gz false false false | tee tests/results/runSearchTestsV8.txt
./katago runsearchtestsv8 tests/models/g170-b6c96-s175395328-d26788732.bin.gz false false false | tee tests/results/runSearchTestsV8Bin.txt
./katago runsearchtestsv9 tests/models/g170-b6c96-s175395328-d26788732.txt.gz false false false | tee tests/results/runSearchTestsV9.txt
./katago runsearchtestsv9 models/kata1-b40c256-s12350780416-d3055274313.bin.gz false false false | tee tests/results/runSearchTestsV9B40.txt
./katago runsearchtestsv9 models/b18c384nbt-uec.bin.gz false false false | tee tests/results/runSearchTestsV9B18NBT.txt

./katago runnnbatchingtest tests/models/g170-b6c96-s175395328-d26788732.bin.gz false false false | tee tests/results/runNNBatchingTest.txt
./katago runnnbatchingtest tests/models/g170-b6c96-s175395328-d26788732.bin.gz true false false | tee tests/results/runNNBatchingTestiNHWC.txt
./katago runnnbatchingtest tests/models/g170-b6c96-s175395328-d26788732.bin.gz true true false | tee tests/results/runNNBatchingTestNHWC.txt
./katago runnnbatchingtest models/kata1-b28c512nbt-s8326494464-d4628051565.bin.gz true true false | tee tests/results/runNNBatchingTestNHWCB28.txt

./katago runnnevalcanarytests configs/gtp_example.cfg tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz 0 | grep -v ': nnRandSeed0 = ' | tee tests/results/runNNCanaryTests.txt
./katago runnnevalcanarytests configs/gtp_example.cfg tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz 3 | grep -v ': nnRandSeed0 = ' | tee -a tests/results/runNNCanaryTests.txt
./katago runnnevalcanarytests configs/gtp_example.cfg tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz 6 | grep -v ': nnRandSeed0 = ' | tee -a tests/results/runNNCanaryTests.txt

mkdir -p tests/scratch
./katago runtinynntests tests/scratch 1.0 | grep -v ': nnRandSeed0 = ' | grep -v 'finishing, processed' | tee tests/results/runTinyNNTests.txt

exit 0
}
