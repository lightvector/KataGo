#!/bin/bash -eux
{
./katago runoutputtests | tee tests/results/runOutputTests.txt

exit 0
}
