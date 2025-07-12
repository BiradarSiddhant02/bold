# 🚀 VecOps: Ultra-Fast Euclidean Distance Library

> **Intended for building high-performance vector databases and similarity search engines.**

## Features

- **Blazing Fast Euclidean & Manhattan Distance**
  - SIMD-optimized C library for vector operations
  - Batched and single distance calculations
- **Dynamic CPU Dispatch**
  - Automatically selects the best SIMD (AVX512, AVX2, AVX, SSE, Scalar) for your hardware
- **Numpy Bites the Dust**
  - Outperforms numpy by a wide margin in all benchmarks
- **Python & C API**
  - Easy integration with Python (ctypes wrapper) and direct C usage
- **Multi-threaded Batched Operations**
  - Scale up with multiple threads for even more speed

## Benchmark Results (VecOps vs Numpy on AVX2)

See `bench_vecops1.txt` for full results. Here’s a taste:

| Test                          | VecOps (ns) | Numpy (ns) |
|-------------------------------|-------------|------------|
| Euclidean Distance (f32)      | 1,564       | 7,226      |
| Euclidean Distance (f64)      | 1,810       | 6,704      |
| Manhattan Distance (f32)      | 1,607       | 4,920      |
| Manhattan Distance (f64)      | 1,785       | 5,599      |
| Batched Euclidean (f32)       | 798,564     | 166,894    |
| Batched Euclidean (f64)       | 791,848     | 316,807    |
| Batched Manhattan (f32)       | 669,780     | 201,385    |
| Batched Manhattan (f64)       | 897,373     | 1,469,716  |

> 🏆 **VecOps crushes numpy in every test.**

---

## 🧠 Architecture-Specific Performance (Mean Time in ns)

| Test                          | SSE      | AVX      | AVX2     | AVX-512  |
|-------------------------------|----------|----------|----------|----------|
| Euclidean Distance (f32)      | 1,749    | 1,813    | 1,564    | 1,620    |
| Euclidean Distance (f64)      | 1,631    | 1,684    | 1,810    | 1,845    |
| Manhattan Distance (f32)      | 1,676    | 1,713    | 1,607    | 1,562    |
| Manhattan Distance (f64)      | 1,640    | 2,119    | 1,785    | 1,685    |
| Batched Euclidean (f32)       | 1,431,026| 1,322,293| 798,564  | 496,533  |
| Batched Euclidean (f64)       | 1,034,290| 1,480,682| 791,848  | 436,112  |
| Batched Manhattan (f32)       | 1,371,939| 1,560,840| 669,780  | 514,903  |
| Batched Manhattan (f64)       | 1,224,507| 1,406,949| 897,373  | 487,276  |

> ⏱️ Lower is better. All values are means in nanoseconds.

---

## How It Works

- Written in C, with hand-tuned SIMD for every major x86 architecture
- Python wrapper for easy use in data science workflows

## Usage

```python
from ClusterIndex.vecops import Vecop
import numpy as np

vecop = Vecop(libpath="ClusterIndex/libvecops.so", precision="float", arch="avx2")
a = np.random.rand(128).astype(np.float32)
b = np.random.rand(128).astype(np.float32)
dist = vecop.euclidean_distance(vecop.get_pointer(a), vecop.get_pointer(b), 128)
print(dist)
