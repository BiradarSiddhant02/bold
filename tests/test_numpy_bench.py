import numpy as np
import pytest

# --- Fixtures ---

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
def test_data_f64():
    np.random.seed(42)
    n_vecs = 1000
    length = 128
    vecs = np.random.rand(n_vecs, length).astype(np.float64)
    ref = np.random.rand(length).astype(np.float64)
    a = np.random.rand(length).astype(np.float64)
    b = np.random.rand(length).astype(np.float64)
    return vecs, ref, a, b, n_vecs, length

# --- Core Ops ---

def euclidean_distance_numpy(a, b, length):
    return np.sqrt(np.sum((a[:length] - b[:length]) ** 2))

def euclidean_distance_batched_numpy(vecs, ref):
    return np.sqrt(np.sum((vecs - ref) ** 2, axis=1))

def manhattan_distance_numpy(a, b, length):
    return np.sum(np.abs(a[:length] - b[:length]))

def manhattan_distance_batched_numpy(vecs, ref):
    return np.sum(np.abs(vecs - ref), axis=1)

def centroid_numpy(vecs):
    return np.mean(vecs, axis=0)

# --- Benchmarks: f32 ---

def test_bench_euclidean_distance_numpy_f32(benchmark, test_data_f32):
    _, _, a, b, _, length = test_data_f32
    benchmark(lambda: euclidean_distance_numpy(a, b, length))

def test_bench_euclidean_distance_batched_numpy_f32(benchmark, test_data_f32):
    vecs, ref, _, _, _, _ = test_data_f32
    benchmark(lambda: euclidean_distance_batched_numpy(vecs, ref))

def test_bench_centroid_numpy_f32(benchmark, test_data_f32):
    vecs, _, _, _, _, _ = test_data_f32
    benchmark(lambda: centroid_numpy(vecs))

def test_bench_manhattan_distance_numpy_f32(benchmark, test_data_f32):
    _, _, a, b, _, length = test_data_f32
    benchmark(lambda: manhattan_distance_numpy(a, b, length))

def test_bench_manhattan_distance_batched_numpy_f32(benchmark, test_data_f32):
    vecs, ref, _, _, _, _ = test_data_f32
    benchmark(lambda: manhattan_distance_batched_numpy(vecs, ref))

# --- Benchmarks: f64 ---

def test_bench_euclidean_distance_numpy_f64(benchmark, test_data_f64):
    _, _, a, b, _, length = test_data_f64
    benchmark(lambda: euclidean_distance_numpy(a, b, length))

def test_bench_euclidean_distance_batched_numpy_f64(benchmark, test_data_f64):
    vecs, ref, _, _, _, _ = test_data_f64
    benchmark(lambda: euclidean_distance_batched_numpy(vecs, ref))

def test_bench_centroid_numpy_f64(benchmark, test_data_f64):
    vecs, _, _, _, _, _ = test_data_f64
    benchmark(lambda: centroid_numpy(vecs))

def test_bench_manhattan_distance_numpy_f64(benchmark, test_data_f64):
    _, _, a, b, _, length = test_data_f64
    benchmark(lambda: manhattan_distance_numpy(a, b, length))

def test_bench_manhattan_distance_batched_numpy_f64(benchmark, test_data_f64):
    vecs, ref, _, _, _, _ = test_data_f64
    benchmark(lambda: manhattan_distance_batched_numpy(vecs, ref))
