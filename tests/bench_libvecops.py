import os
import ctypes
import numpy as np
import time
import argparse

# Path to the shared library
LIBPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ClusterIndex", "libvecops.so"))

# Load the library
lib = ctypes.CDLL(LIBPATH)

# Use double precision for benchmarking
c_type = ctypes.c_double
vec_ptr_type = ctypes.POINTER(c_type)

# Set up function signatures
lib.euclidean_distance_f64.argtypes = [vec_ptr_type, vec_ptr_type, ctypes.c_size_t]
lib.euclidean_distance_f64.restype = c_type
lib.batched_euclidean_f64.argtypes = [ctypes.POINTER(vec_ptr_type), vec_ptr_type, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.batched_euclidean_f64.restype = vec_ptr_type
lib.centroid_f64.argtypes = [ctypes.POINTER(vec_ptr_type), ctypes.c_size_t, ctypes.c_size_t]
lib.centroid_f64.restype = vec_ptr_type
lib.manhattan_distance_f64.argtypes = [vec_ptr_type, vec_ptr_type, ctypes.c_size_t]
lib.manhattan_distance_f64.restype = c_type
lib.batched_manhattan_f64.argtypes = [ctypes.POINTER(vec_ptr_type), vec_ptr_type, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
lib.batched_manhattan_f64.restype = vec_ptr_type

# Data for benchmarking
np.random.seed(42)
n_vecs = 1000
length = 128
vecs = np.random.rand(n_vecs, length).astype(np.float64)
ref = np.random.rand(length).astype(np.float64)
a = np.random.rand(length).astype(np.float64)
b = np.random.rand(length).astype(np.float64)

# Helper to get ctypes pointers
vecs_ptrs = (vec_ptr_type * n_vecs)(*[v.ctypes.data_as(vec_ptr_type) for v in vecs])
ref_ptr = ref.ctypes.data_as(vec_ptr_type)
a_ptr = a.ctypes.data_as(vec_ptr_type)
b_ptr = b.ctypes.data_as(vec_ptr_type)

# Benchmarking utility
def bench(func, *args, repeat=1000, name=None):
    t0 = time.perf_counter()
    for _ in range(repeat):
        func(*args)
    t1 = time.perf_counter()
    print(f"{name or func.__name__}: {(t1-t0)*1e6/repeat:.2f} us per call over {repeat} runs")

parser = argparse.ArgumentParser()
parser.add_argument('--num-threads', type=int, default=1)
args, unknown = parser.parse_known_args()
num_threads = args.num_threads

if __name__ == "__main__":
    print("Benchmarking libvecops.so (direct ctypes calls, double precision)")
    bench(lib.euclidean_distance_f64, a_ptr, b_ptr, length, repeat=10000, name="euclidean_distance_f64")
    bench(lib.batched_euclidean_f64, vecs_ptrs, ref_ptr, n_vecs, length, num_threads, repeat=100, name="batched_euclidean_f64")
    bench(lib.centroid_f64, vecs_ptrs, n_vecs, length, repeat=100, name="centroid_f64")
    bench(lib.manhattan_distance_f64, a_ptr, b_ptr, length, repeat=10000, name="manhattan_distance_f64")
    bench(lib.batched_manhattan_f64, vecs_ptrs, ref_ptr, n_vecs, length, num_threads, repeat=100, name="batched_manhattan_f64")
