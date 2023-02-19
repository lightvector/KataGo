#!/bin/bash -eux
set -o pipefail
{
(
    mkdir -p models
    cd models/
    wget -nc https://github.com/lightvector/KataGo/releases/download/v1.12.1/b18c384nbt-uec.bin.gz
)
./katago runsearchtests tests/models/g103-b6c96-s103408384-d26419149.txt.gz true false 0 true | tee tests/results/runSearchTestsFP16-iNHWC-cNCHW.txt
./katago runsearchtests tests/models/g103-b6c96-s103408384-d26419149.txt.gz true true 0 true | tee tests/results/runSearchTestsFP16-iNHWC-cNHWC.txt
./katago runsearchtests tests/models/g103-b6c96-s103408384-d26419149.txt.gz true false 1 true | tee tests/results/runSearchTestsFP16-s1.txt

./katago runsearchtests tests/models/run4-s67105280-d24430742-b6c96.txt.gz true false 0 true | tee tests/results/runSearchTestsFP16-r4-iNHWC-cNCHW.txt
./katago runsearchtests tests/models/run4-s67105280-d24430742-b6c96.txt.gz true true 0 true | tee tests/results/runSearchTestsFP16-r4-iNHWC-cNHWC.txt
./katago runsearchtests tests/models/run4-s67105280-d24430742-b6c96.txt.gz true false 1 true | tee tests/results/runSearchTestsFP16-r4-s1.txt

./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz true false 0 true | tee tests/results/runSearchTestsV3FP16-g2-iNHWC-cNCHW.txt
./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz true true 0 true | tee tests/results/runSearchTestsV3FP16-g2-iNHWC-cNHWC.txt
./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz true false 1 true | tee tests/results/runSearchTestsV3FP16-g2-s1.txt

./katago runnnsymmetriestest tests/models/g103-b6c96-s103408384-d26419149.txt.gz true true true | tee tests/results/runNNSymmetriesTestFP16.txt
./katago runnnbatchingtest tests/models/g170-b6c96-s175395328-d26788732.bin.gz true true true | tee tests/results/runNNBatchingTestFP16.txt

./katago runsearchtestsv8 tests/models/g170-b6c96-s175395328-d26788732.txt.gz true true true | tee tests/results/runSearchTestsV8FP16.txt

./katago runsearchtestsv9 models/b18c384nbt-uec.bin.gz false false true | tee tests/results/runSearchTestsV9B18NBTFP16.txt
exit 0
}
