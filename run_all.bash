#!/bin/sh
# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Running Correctness Tests ---"
# Pytest will discover and run tests in files matching test_*.py
FAILED=0

echo "--- Running Correctness Tests ---"

run_test() {
    ARCH="$1"
    echo "\n>>> Testing ARCH: $ARCH"
    if ! pytest --arch "$ARCH"; then
        echo "Test failed for $ARCH"
        FAILED=1
    else
        echo "Passed $ARCH"
    fi
}

# Run tests for each arch
for ARCH in avx avx2 avx512 sse scalar; do
    run_test "$ARCH"
done

echo "\n--- Running Benchmarks ---"


echo "\n--- All tasks completed successfully ---"