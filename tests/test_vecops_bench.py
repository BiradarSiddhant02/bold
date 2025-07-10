import os
import sys
import ctypes
import numpy as np
import pytest

def pytest_addoption(parser):
    parser.addoption('--num-threads', action='store', default=1, type=int, help='Number of threads for batched ops')

@pytest.fixture(scope="module")
def num_threads(request):
    return request.config.getoption("--num-threads")

LIBPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ClusterIndex", "libvecops.so"))

@pytest.fixture(scope="module")
def lib():
    lib = ctypes.CDLL(LIBPATH)

    # Define function signatures
    def set_sig(suffix, c_type, vec_ptr_type):
        getattr(lib, f"euclidean_distance_{suffix}").argtypes = [vec_ptr_type, vec_ptr_type, ctypes.c_size_t]
        getattr(lib, f"euclidean_distance_{suffix}").restype = c_type

        getattr(lib, f"batched_euclidean_{suffix}").argtypes = [ctypes.POINTER(vec_ptr_type), vec_ptr_type, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        getattr(lib, f"batched_euclidean_{suffix}").restype = vec_ptr_type

        getattr(lib, f"centroid_{suffix}").argtypes = [ctypes.POINTER(vec_ptr_type), ctypes.c_size_t, ctypes.c_size_t]
        getattr(lib, f"centroid_{suffix}").restype = vec_ptr_type

        getattr(lib, f"manhattan_distance_{suffix}").argtypes = [vec_ptr_type, vec_ptr_type, ctypes.c_size_t]
        getattr(lib, f"manhattan_distance_{suffix}").restype = c_type

        getattr(lib, f"batched_manhattan_{suffix}").argtypes = [ctypes.POINTER(vec_ptr_type), vec_ptr_type, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
        getattr(lib, f"batched_manhattan_{suffix}").restype = vec_ptr_type

    # Set up for float64
    set_sig('f64', ctypes.c_double, ctypes.POINTER(ctypes.c_double))

    # Set up for float32
    set_sig('f32', ctypes.c_float, ctypes.POINTER(ctypes.c_float))

    return lib

# === Fixtures for float64 ===

@pytest.fixture(scope="module")
def test_data():
    np.random.seed(42)
    n_vecs = 1000
    length = 128
    vecs = np.random.rand(n_vecs, length).astype(np.float64)
    ref = np.random.rand(length).astype(np.float64)
    a = np.random.rand(length).astype(np.float64)
    b = np.random.rand(length).astype(np.float64)
    return vecs, ref, a, b, n_vecs, length

@pytest.fixture(scope="module")
def ptrs(test_data):
    vecs, ref, a, b, n_vecs, _ = test_data
    vec_ptr_type = ctypes.POINTER(ctypes.c_double)
    vecs_ptrs = (vec_ptr_type * n_vecs)(*[v.ctypes.data_as(vec_ptr_type) for v in vecs])
    return vecs_ptrs, ref.ctypes.data_as(vec_ptr_type), a.ctypes.data_as(vec_ptr_type), b.ctypes.data_as(vec_ptr_type)

# === Fixtures for float32 ===

@pytest.fixture(scope="module")
def test_data_f32():
    np.random.seed(42)
    n_vecs = 1000
    length = 128
    vecs = np.random.rand(n_vecs, length).astype(np.float32)
    ref = np.random.rand(length).astype(np.float32)
    a = np.random.rand(length).astype(np.float32)
    b = np.random.rand(length).astype(np.float32)
    return vecs, ref, a, b, n_vecs, length

@pytest.fixture(scope="module")
def ptrs_f32(test_data_f32):
    vecs, ref, a, b, n_vecs, _ = test_data_f32
    vec_ptr_type = ctypes.POINTER(ctypes.c_float)
    vecs_ptrs = (vec_ptr_type * n_vecs)(*[v.ctypes.data_as(vec_ptr_type) for v in vecs])
    return vecs_ptrs, ref.ctypes.data_as(vec_ptr_type), a.ctypes.data_as(vec_ptr_type), b.ctypes.data_as(vec_ptr_type)

# === Benchmarks for f64 ===

def test_bench_euclidean_distance_f64(benchmark, lib, test_data, ptrs):
    vecs_ptrs, ref_ptr, a_ptr, b_ptr = ptrs
    _, _, _, _, _, length = test_data
    benchmark(lambda: lib.euclidean_distance_f64(a_ptr, b_ptr, length))

def test_bench_batched_euclidean_f64(benchmark, lib, test_data, ptrs, num_threads):
    vecs_ptrs, ref_ptr, _, _ = ptrs
    _, _, _, _, n_vecs, length = test_data
    benchmark(lambda: lib.batched_euclidean_f64(vecs_ptrs, ref_ptr, n_vecs, length, num_threads))

def test_bench_centroid_f64(benchmark, lib, test_data, ptrs):
    vecs_ptrs, _, _, _ = ptrs
    _, _, _, _, n_vecs, length = test_data
    benchmark(lambda: lib.centroid_f64(vecs_ptrs, n_vecs, length))

def test_bench_manhattan_distance_f64(benchmark, lib, test_data, ptrs):
    _, _, a_ptr, b_ptr = ptrs
    _, _, _, _, _, length = test_data
    benchmark(lambda: lib.manhattan_distance_f64(a_ptr, b_ptr, length))

def test_bench_batched_manhattan_f64(benchmark, lib, test_data, ptrs, num_threads):
    vecs_ptrs, ref_ptr, _, _ = ptrs
    _, _, _, _, n_vecs, length = test_data
    benchmark(lambda: lib.batched_manhattan_f64(vecs_ptrs, ref_ptr, n_vecs, length, num_threads))

# === Benchmarks for f32 ===

def test_bench_euclidean_distance_f32(benchmark, lib, test_data_f32, ptrs_f32):
    _, _, a_ptr, b_ptr = ptrs_f32
    _, _, _, _, _, length = test_data_f32
    benchmark(lambda: lib.euclidean_distance_f32(a_ptr, b_ptr, length))

def test_bench_batched_euclidean_f32(benchmark, lib, test_data_f32, ptrs_f32, num_threads):
    vecs_ptrs, ref_ptr, _, _ = ptrs_f32
    _, _, _, _, n_vecs, length = test_data_f32
    benchmark(lambda: lib.batched_euclidean_f32(vecs_ptrs, ref_ptr, n_vecs, length, num_threads))

def test_bench_centroid_f32(benchmark, lib, test_data_f32, ptrs_f32):
    vecs_ptrs, _, _, _ = ptrs_f32
    _, _, _, _, n_vecs, length = test_data_f32
    benchmark(lambda: lib.centroid_f32(vecs_ptrs, n_vecs, length))

def test_bench_manhattan_distance_f32(benchmark, lib, test_data_f32, ptrs_f32):
    _, _, a_ptr, b_ptr = ptrs_f32
    _, _, _, _, _, length = test_data_f32
    benchmark(lambda: lib.manhattan_distance_f32(a_ptr, b_ptr, length))

def test_bench_batched_manhattan_f32(benchmark, lib, test_data_f32, ptrs_f32, num_threads):
    vecs_ptrs, ref_ptr, _, _ = ptrs_f32
    _, _, _, _, n_vecs, length = test_data_f32
    benchmark(lambda: lib.batched_manhattan_f32(vecs_ptrs, ref_ptr, n_vecs, length, num_threads))
