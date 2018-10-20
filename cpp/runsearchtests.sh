#!/bin/bash -eux

./main runSearchTests models/v53-140-5x64.txt.gz | tee tests/results/runSearchTestsOutput.txt
