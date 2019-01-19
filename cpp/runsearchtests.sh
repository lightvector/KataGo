#!/bin/bash -eux
{
./main runsearchtests models/v53-140-5x64.txt.gz false false 0 false | tee tests/results/runSearchTests-iNCHW-cNCHW.txt
./main runsearchtests models/v53-140-5x64.txt.gz true false 0 false | tee tests/results/runSearchTests-iNHWC-cNCHW.txt
./main runsearchtests models/v53-140-5x64.txt.gz true true 0 false | tee tests/results/runSearchTests-iNHWC-cNHWC.txt
./main runsearchtests models/v53-140-5x64.txt.gz true false 1 false | tee tests/results/runSearchTests-s1.txt

./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz false false 0 false | tee tests/results/runSearchTests-r4-iNCHW-cNCHW.txt
./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz true false 0 false | tee tests/results/runSearchTests-r4-iNHWC-cNCHW.txt
./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz true true 0 false | tee tests/results/runSearchTests-r4-iNHWC-cNHWC.txt
./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz true false 1 false | tee tests/results/runSearchTests-r4-s1.txt

./main runsearchtestsv3 models/grun2-b6c96-s128700160-d49811312.txt.gz false false 0 false | tee tests/results/runSearchTestsV3-g2-iNCHW-cNCHW.txt
./main runsearchtestsv3 models/grun2-b6c96-s128700160-d49811312.txt.gz true false 0 false | tee tests/results/runSearchTestsV3-g2-iNHWC-cNCHW.txt
./main runsearchtestsv3 models/grun2-b6c96-s128700160-d49811312.txt.gz true true 0 false | tee tests/results/runSearchTestsV3-g2-iNHWC-cNHWC.txt
./main runsearchtestsv3 models/grun2-b6c96-s128700160-d49811312.txt.gz true false 1 false | tee tests/results/runSearchTestsV3-g2-s1.txt

./main runselfplayinittests models/grun50-b6c96-s156348160-d118286860.txt.gz | tee tests/results/runSelfplayInitTests.txt

exit 0
}
