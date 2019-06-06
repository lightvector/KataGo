#!/bin/bash -eux
{
./main runsearchtests models/g103-b6c96-s103408384-d26419149.txt.gz true false 0 true | tee tests/results/runSearchTestsFP16-iNHWC-cNCHW.txt
./main runsearchtests models/g103-b6c96-s103408384-d26419149.txt.gz true true 0 true | tee tests/results/runSearchTestsFP16-iNHWC-cNHWC.txt
./main runsearchtests models/g103-b6c96-s103408384-d26419149.txt.gz true false 1 true | tee tests/results/runSearchTestsFP16-s1.txt

./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz true false 0 true | tee tests/results/runSearchTestsFP16-r4-iNHWC-cNCHW.txt
./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz true true 0 true | tee tests/results/runSearchTestsFP16-r4-iNHWC-cNHWC.txt
./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz true false 1 true | tee tests/results/runSearchTestsFP16-r4-s1.txt

./main runsearchtestsv3 models/grun2-b6c96-s128700160-d49811312.txt.gz true false 0 true | tee tests/results/runSearchTestsV3FP16-g2-iNHWC-cNCHW.txt
./main runsearchtestsv3 models/grun2-b6c96-s128700160-d49811312.txt.gz true true 0 true | tee tests/results/runSearchTestsV3FP16-g2-iNHWC-cNHWC.txt
./main runsearchtestsv3 models/grun2-b6c96-s128700160-d49811312.txt.gz true false 1 true | tee tests/results/runSearchTestsV3FP16-g2-s1.txt

exit 0
}
