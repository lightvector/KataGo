#!/bin/bash -eux
{
./main runsearchtests models/v53-140-5x64.txt.gz false false 0 false | tee tests/results/runSearchTests-iNCHW-cNCHW.txt
#Cuda not supported
# ./main runsearchtests models/v53-140-5x64.txt.gz false true 0 | tee tests/results/runSearchTests-iNCHW-cNHWC.txt
./main runsearchtests models/v53-140-5x64.txt.gz true false 0 false | tee tests/results/runSearchTests-iNHWC-cNCHW.txt
./main runsearchtests models/v53-140-5x64.txt.gz true true 0 false | tee tests/results/runSearchTests-iNHWC-cNHWC.txt
./main runsearchtests models/v53-140-5x64.txt.gz true false 1 false | tee tests/results/runSearchTests-s1.txt

./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz false false 0 false | tee tests/results/runSearchTests-r4-iNCHW-cNCHW.txt
#Cuda not supported
#./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz false true 0 | tee tests/results/runSearchTests-r4-iNCHW-cNHWC.txt
./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz true false 0 false | tee tests/results/runSearchTests-r4-iNHWC-cNCHW.txt
./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz true true 0 false | tee tests/results/runSearchTests-r4-iNHWC-cNHWC.txt
./main runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz true false 1 false | tee tests/results/runSearchTests-r4-s1.txt


exit 0
}
