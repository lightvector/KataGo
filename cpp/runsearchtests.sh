#!/bin/bash -eux
{
#./main runSearchTests models/v53-140-5x64.txt.gz false false 0 | tee tests/results/runSearchTests-iNCHW-cNCHW.txt

#Cuda not supported
#./main runSearchTests models/v53-140-5x64.txt.gz false true 0 | tee tests/results/runSearchTests-iNCHW-cNHWC.txt

./main runSearchTests models/v53-140-5x64.txt.gz true false 0 | tee tests/results/runSearchTests-iNHWC-cNCHW.txt
./main runSearchTests models/v53-140-5x64.txt.gz true true 0 | tee tests/results/runSearchTests-iNHWC-cNHWC.txt
./main runSearchTests models/v53-140-5x64.txt.gz true false 1 | tee tests/results/runSearchTests-s1.txt

exit 0
}
