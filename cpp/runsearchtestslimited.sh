#!/bin/bash -eux
set -o pipefail
{
./katago runsearchtests tests/models/g103-b6c96-s103408384-d26419149.txt.gz false false 0 false | tee tests/results/runSearchTests-iNCHW-cNCHW.txt
./katago runsearchtests tests/models/run4-s67105280-d24430742-b6c96.txt.gz false false 0 false | tee tests/results/runSearchTests-r4-iNCHW-cNCHW.txt
./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz false false 0 false | tee tests/results/runSearchTestsV3-g2-iNCHW-cNCHW.txt
./katago runsearchtestsv3 tests/models/grun2-b6c96-s128700160-d49811312.txt.gz false false 5 false | tee tests/results/runSearchTestsV3-g2-s5.txt
./katago runnnsymmetriestest tests/models/g103-b6c96-s103408384-d26419149.txt.gz false false false | tee tests/results/runNNSymmetriesTest.txt
./katago runsearchtestsv8 tests/models/g170-b6c96-s175395328-d26788732.txt.gz false false false | tee tests/results/runSearchTestsV8.txt
./katago runsearchtestsv9 tests/models/g170-b6c96-s175395328-d26788732.txt.gz false false false | tee tests/results/runSearchTestsV9.txt

./katago runselfplayinittests tests/models/grun50-b6c96-s156348160-d118286860.txt.gz | tee tests/results/runSelfplayInitTests.txt
./katago runsekitrainwritetests tests/models/g103-b6c96-s103408384-d26419149.txt.gz | tee tests/results/runSekiTrainWriteTests.txt
./katago runnnsymmetriestest tests/models/g103-b6c96-s103408384-d26419149.txt.gz false false false | tee tests/results/runNNSymmetriesTest.txt
./katago runselfplayinitstattests tests/models/g170-b6c96-s175395328-d26788732.bin.gz | tee tests/results/runSelfplayInitStatTests.txt

mkdir -p tests/scratch
./katago runtinynntests tests/scratch 1.0 | grep -v ': nnRandSeed0 = ' | grep -v 'finishing, processed' | tee tests/results/runTinyNNTests.txt

exit 0
}
