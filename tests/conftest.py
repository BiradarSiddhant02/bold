# conftest.py for pytest command-line option

def pytest_addoption(parser):
    parser.addoption(
        "--arch",
        action="store",
        default="scalar",
        help="SIMD architecture to use: sse, avx, avx2, avx-512, scalar",
    )
    parser.addoption(
        "--num-threads",
        action="store",
        default=1,
        type=int,
        help="Number of threads for batched ops",
    )
