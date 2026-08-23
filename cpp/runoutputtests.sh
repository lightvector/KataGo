#!/bin/bash -eux
# Without pipefail, the `| tee` would mask the test's exit code and -e would never trigger.
set -o pipefail
{
./katago runoutputtests | tee tests/results/runOutputTests.txt

exit 0
}
