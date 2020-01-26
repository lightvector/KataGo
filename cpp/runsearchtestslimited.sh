#!/bin/bash -eux
{
./katago runsearchtests models/g103-b6c96-s103408384-d26419149.txt.gz false false 0 false | tee tests/results/runSearchTests-iNCHW-cNCHW.txt
./katago runsearchtests models/run4-s67105280-d24430742-b6c96.txt.gz false false 0 false | tee tests/results/runSearchTests-r4-iNCHW-cNCHW.txt
./katago runsearchtestsv3 models/grun2-b6c96-s128700160-d49811312.txt.gz false false 0 false | tee tests/results/runSearchTestsV3-g2-iNCHW-cNCHW.txt
./katago runnnsymmetriestest models/g103-b6c96-s103408384-d26419149.txt.gz false false false | tee tests/results/runNNSymmetriesTest.txt
./katago runsearchtestsv8 models/g170-b6c96-s175395328-d26788732.txt.gz false false false | tee tests/results/runSearchTestsV8.txt

exit 0
}
