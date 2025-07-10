#!/bin/bash

rm *_bench.txt
pytest tests/test_numpy_bench.py --arch avx2 --num-threads=18 | tee numpy_bench.txt &
pytest tests/test_libvecops_bench.py --arch avx2 --num-threads=18 | tee libvecops_bench.txt &
pytest tests/test_vecops_bench.py --arch avx2 --num-threads=18 | tee vecops_bench.txt &
echo ""
