#!/bin/bash -eux
{
./main runoutputtests | tee tests/results/runOutputTests.txt

exit 0
}
