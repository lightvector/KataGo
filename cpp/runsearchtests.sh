#!/bin/bash -eux
{
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

./katago runnnontinyboardtest tests/models/g170-b6c96-s175395328-d26788732.bin.gz false false 3 false | tee tests/results/runNNOnTinyBoardTest.txt
./katago runselfplayinittests tests/models/grun50-b6c96-s156348160-d118286860.txt.gz | tee tests/results/runSelfplayInitTests.txt
./katago runsekitrainwritetests tests/models/g103-b6c96-s103408384-d26419149.txt.gz | tee tests/results/runSekiTrainWriteTests.txt
./katago runnnsymmetriestest tests/models/g103-b6c96-s103408384-d26419149.txt.gz false false false | tee tests/results/runNNSymmetriesTest.txt

./katago runsearchtestsv8 tests/models/g170-b6c96-s175395328-d26788732.txt.gz false false false | tee tests/results/runSearchTestsV8.txt
./katago runsearchtestsv8 tests/models/g170-b6c96-s175395328-d26788732.bin.gz false false false | tee tests/results/runSearchTestsV8Bin.txt

exit 0
}
